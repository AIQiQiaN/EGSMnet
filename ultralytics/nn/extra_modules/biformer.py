from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class TopkRouting(nn.Module):
    """TopkRouting with automatic topk adjustment"""

    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        self.routing_act = nn.Softmax(dim=-1)

    def forward(self, query: Tensor, key: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key)
        attn_logit = (query_hat * self.scale) @ key_hat.transpose(-2, -1)

        # 🔧 自动调整topk
        num_windows = attn_logit.size(-1)
        actual_topk = min(self.topk, num_windows)

        topk_attn_logit, topk_index = torch.topk(attn_logit, k=actual_topk, dim=-1)
        r_weight = self.routing_act(topk_attn_logit)

        return r_weight, topk_index


class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx: Tensor, r_weight: Tensor, kv: Tensor):
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)

        topk_kv = torch.gather(
            kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1),
            dim=2,
            index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv)
        )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')

        return topk_kv


class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)

    def forward(self, x):
        q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim + self.dim], dim=-1)
        return q, kv


class BiLevelRoutingAttention(nn.Module):
    """BiLevelRoutingAttention with size-safe LEPE"""

    def __init__(self, dim, n_win=7, num_heads=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None,
                 kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False,
                 diff_routing=False, soft_routing=False,
                 side_dwconv=3, auto_pad=True):
        super().__init__()

        self.dim = dim
        self.n_win = n_win
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads == 0, \
            'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5

        # LEPE - 使用 same padding 确保尺寸不变
        self.lepe = nn.Conv2d(
            dim, dim,
            kernel_size=side_dwconv,
            stride=1,
            padding=side_dwconv // 2,
            groups=dim,
            bias=False
        ) if side_dwconv > 0 else nn.Identity()

        # Router
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing

        assert not (self.param_routing and not self.diff_routing)

        self.router = TopkRouting(
            qk_dim=self.qk_dim,
            qk_scale=self.scale,
            topk=self.topk,
            diff_routing=self.diff_routing,
            param_routing=self.param_routing
        )

        if self.soft_routing:
            mul_weight = 'soft'
        elif self.diff_routing:
            mul_weight = 'hard'
        else:
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # QKV mapping
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not supported!')

        # KV downsampling
        self.kv_downsample_mode = kv_downsample_mode
        if kv_downsample_mode == 'identity':
            self.kv_down = nn.Identity()
        elif kv_downsample_mode == 'avgpool':
            assert kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(kv_downsample_ratio) if kv_downsample_ratio > 1 else nn.Identity()
        elif kv_downsample_mode == 'maxpool':
            assert kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(kv_downsample_ratio) if kv_downsample_ratio > 1 else nn.Identity()
        elif kv_downsample_mode == 'ada_avgpool':
            assert kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(kv_per_win)
        elif kv_downsample_mode == 'ada_maxpool':
            assert kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(kv_per_win)
        else:
            self.kv_down = nn.Identity()

        self.attn_act = nn.Softmax(dim=-1)
        self.auto_pad = auto_pad

    def forward(self, x, ret_attn_mask=False):
        """
        x: NCHW tensor
        Return: NCHW tensor
        """
        # 保存原始尺寸
        N, C_in, H_orig, W_orig = x.shape

        x = rearrange(x, "n c h w -> n h w c")

        # Padding
        if self.auto_pad:
            N, H_in, W_in, C = x.size()
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.size()
        else:
            N, H, W, C = x.size()
            H_in, W_in = H, W
            pad_r = pad_b = 0

        # 🔧 记录准确的窗口尺寸
        window_h = H // self.n_win
        window_w = W // self.n_win

        # Patchify
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        # QKV projection
        q, kv = self.qkv(x)  # q: (n, p^2, h, w, qk_dim), kv: (n, p^2, h, w, qk_dim+dim)

        # Pixel-wise qkv
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)

        # Window-wise qk - 🔧 修复bug：应该是 0:self.qk_dim
        q_win = q.mean([2, 3])  # (n, p^2, qk_dim)
        k_win = kv[..., 0:self.qk_dim].mean([2, 3])  # (n, p^2, qk_dim)

        ################## 🔧 完全修复的 LEPE 计算 ##################
        # 提取 V 部分
        v_for_lepe = kv[..., self.qk_dim:]  # (n, p^2, window_h, window_w, dim)

        # 重组为完整的 NCHW 特征图
        v_full = rearrange(
            v_for_lepe,
            'n (j i) h w c -> n c (j h) (i w)',
            j=self.n_win,
            i=self.n_win
        ).contiguous()  # (n, dim, H, W)

        # 应用深度卷积 - padding=kernel_size//2 确保尺寸不变
        lepe_full = self.lepe(v_full)  # (n, dim, H', W')

        # 🔧 强制调整到精确尺寸
        if lepe_full.shape[2] != H or lepe_full.shape[3] != W:
            # 使用双线性插值调整到目标尺寸
            lepe_full = F.interpolate(
                lepe_full,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

        # 转回 NHWC 格式
        lepe = rearrange(lepe_full, 'n c h w -> n h w c')  # (n, H, W, dim)
        ##########################################################

        # Routing
        r_weight, r_idx = self.router(q_win, k_win)

        # Gather KV
        kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix)
        k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)

        # Multi-head Attention
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads)
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads)

        attn_weight = (q_pix * self.scale) @ k_pix_sel
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel

        # 重组输出 - 🔧 使用计算好的窗口尺寸
        out = rearrange(
            out,
            '(n j i) m (h w) c -> n (j h) (i w) (m c)',
            j=self.n_win,
            i=self.n_win,
            h=window_h,
            w=window_w
        )  # (n, H, W, dim)

        # 🔧 最后确认尺寸匹配
        if out.shape[1:3] != lepe.shape[1:3]:
            # 如果还是不匹配，强制resize
            out_temp = rearrange(out, 'n h w c -> n c h w')
            out_temp = F.interpolate(out_temp, size=lepe.shape[1:3], mode='bilinear', align_corners=False)
            out = rearrange(out_temp, 'n c h w -> n h w c')

        # 添加 LEPE
        out = out + lepe

        # Output projection
        out = self.wo(out)

        # Crop padding
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return rearrange(out, "n h w c -> n c h w")


class Attention(nn.Module):
    """vanilla attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        _, _, H, W = x.size()
        x = rearrange(x, 'n c h w -> n (h w) c')

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, 'n (h w) c -> n c h w', h=H, w=W)
        return x


class AttentionLePE(nn.Module):
    """vanilla attention with LePE"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., side_dwconv=5):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2,
                              groups=dim) if side_dwconv > 0 else \
            lambda x: torch.zeros_like(x)

    def forward(self, x):
        _, _, H, W = x.size()
        x = rearrange(x, 'n c h w -> n (h w) c')

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        lepe = self.lepe(rearrange(x, 'n (h w) c -> n c h w', h=H, w=W))
        lepe = rearrange(lepe, 'n c h w -> n (h w) c')

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = x + lepe

        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, 'n (h w) c -> n c h w', h=H, w=W)
        return x