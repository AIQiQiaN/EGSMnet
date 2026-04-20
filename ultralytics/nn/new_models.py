"""
pgd_modules.py
──────────────────────────────────
PGD-YOLOv11 自定义模块（单流 RGB-IR 融合专用）
  - InputProxy                  : 恒等代理层
  - ChannelSplit                : 通道切片
  - ModalityPromptInjection     : 模态提示注入（MPI）
  - CrossModalGraphReasoning    : GNN 跨模态关系建模（CMGR）
  - TaskGuidedDiffusionPrior    : 任务导向扩散先验（TGDP）—— 已彻底修复 deepcopy
  - RoIJitterModule             : RoI 抖动增强（仅训练时使用）

注册方法（在训练脚本最开头调用）：
    from pgd_modules import register_pgd_modules
    register_pgd_modules()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════
#  1. InputProxy — 恒等映射
# ════════════════════════════
class InputProxy(nn.Module):
    """恒等代理层，保留 6ch 输入供 ChannelSplit 引用。"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ════════════════════════════
#  2. ChannelSplit — 通道切片
# ════════════════════════════
class ChannelSplit(nn.Module):
    """沿通道维度切片，提取 RGB 或 IR。"""
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end = end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.start:self.end, :, :]

    def extra_repr(self):
        return f"ch[{self.start}:{self.end}]"


# ════════════════════════════
#  3. ModalityPromptInjection (MPI)
# ════════════════════════════
class IlluminationAwareModule(nn.Module):
    """光照感知子模块：动态计算模态权重。"""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

    def forward(self, rgb, ir):
        v_rgb = rgb.mean(dim=[1, 2, 3])
        v_ir = ir.mean(dim=[1, 2, 3])
        v = torch.stack([v_rgb, v_ir], dim=1)
        weights = self.mlp(v)
        alpha_rgb = weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        alpha_ir = weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        return alpha_rgb, alpha_ir


class DynamicPromptGenerator(nn.Module):
    """动态提示生成器。"""
    def __init__(self, prompt_channels=16):
        super().__init__()
        self.embed_rgb = nn.Parameter(torch.randn(1, prompt_channels, 1, 1) * 0.02)
        self.embed_ir = nn.Parameter(torch.randn(1, prompt_channels, 1, 1) * 0.02)

    def forward(self, alpha_rgb, alpha_ir, H, W):
        P_rgb = (alpha_rgb * self.embed_rgb).expand(-1, -1, H, W)
        P_ir = (alpha_ir * self.embed_ir).expand(-1, -1, H, W)
        return P_rgb, P_ir


class PromptGuidedFusion(nn.Module):
    """提示引导融合。"""
    def __init__(self, in_ch=3, prompt_ch=16, out_ch=3):
        super().__init__()
        self.conv_rgb = nn.Conv2d(in_ch + prompt_ch, out_ch, 1, bias=False)
        self.conv_ir = nn.Conv2d(in_ch + prompt_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, rgb, ir, P_rgb, P_ir, alpha_rgb, alpha_ir):
        F_rgb = self.conv_rgb(torch.cat([rgb, P_rgb], dim=1))
        F_ir = self.conv_ir(torch.cat([ir, P_ir], dim=1))
        F_unified = alpha_rgb * F_rgb + alpha_ir * F_ir
        return self.bn(F_unified)


class ModalityPromptInjection(nn.Module):
    """
    MPI 完整模块
    YAML 调用: [[1, 2], 1, ModalityPromptInjection, [16]]
    """
    def __init__(self, prompt_channels=16):
        super().__init__()
        self.ias = IlluminationAwareModule()
        self.dpg = DynamicPromptGenerator(prompt_channels)
        self.pgf = PromptGuidedFusion(in_ch=3, prompt_ch=prompt_channels, out_ch=3)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            rgb, ir = x[0], x[1]
        else:
            rgb = x[:, :3, :, :]
            ir = x[:, 3:, :, :]
        B, C, H, W = rgb.shape
        alpha_rgb, alpha_ir = self.ias(rgb, ir)
        P_rgb, P_ir = self.dpg(alpha_rgb, alpha_ir, H, W)
        return self.pgf(rgb, ir, P_rgb, P_ir, alpha_rgb, alpha_ir)


# ═══════════════════════════════════════════
#  4. CrossModalGraphReasoning (CMGR)
# ═══════════════════════════════════════════
class FeatureToGraph(nn.Module):
    def __init__(self, channels, num_nodes=64):
        super().__init__()
        self.num_nodes = num_nodes
        h = w = int(num_nodes ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d((h, w))
        self.pos_embed = nn.Parameter(torch.randn(1, num_nodes, channels) * 0.02)

    def forward(self, feat):
        B, C, H, W = feat.shape
        x = self.pool(feat)
        x = x.flatten(2).permute(0, 2, 1)
        x = x + self.pos_embed
        return x, (H, W)


class DynamicEdgeConstruction(nn.Module):
    def __init__(self, channels, d_k=64, top_k=8):
        super().__init__()
        self.W_q = nn.Linear(channels, d_k, bias=False)
        self.W_k = nn.Linear(channels, d_k, bias=False)
        self.top_k = top_k
        self.scale = d_k ** -0.5

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        S = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        topk_val, topk_idx = S.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(S).scatter_(-1, topk_idx, 1.0)
        S = S.masked_fill(mask == 0, float('-inf'))
        A = F.softmax(S, dim=-1)
        return A


class GraphAttentionLayer(nn.Module):
    def __init__(self, channels, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.d_v = channels // heads
        self.W_v = nn.Linear(channels, channels, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x, A):
        B, N, C = x.shape
        V = self.W_v(x).view(B, N, self.heads, self.d_v)
        out = torch.zeros_like(V)
        for h in range(self.heads):
            v_h = V[:, :, h, :]
            agg = torch.bmm(A, v_h)
            out[:, :, h, :] = agg
        out = out.reshape(B, N, C)
        out = self.proj(self.dropout(F.leaky_relu(out, 0.2)))
        return self.norm(x + out)


class CrossModalGraphReasoning(nn.Module):
    """
    CMGR 完整模块
    YAML 调用: [from, 1, CrossModalGraphReasoning, [512, 64, 8, 4, 2]]
    """
    def __init__(self, channels, num_nodes=64, top_k=8, heads=4, num_layers=2):
        super().__init__()
        self.f2g = FeatureToGraph(channels, num_nodes)
        self.dec = DynamicEdgeConstruction(channels, d_k=64, top_k=top_k)
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(channels, heads) for _ in range(num_layers)
        ])
        self.g2f_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, feat):
        x, (H, W) = self.f2g(feat)
        A = self.dec(x)
        for gat in self.gat_layers:
            x = gat(x, A)
        h = w = int(x.shape[1] ** 0.5)
        x = x.permute(0, 2, 1).view(x.shape[0], -1, h, w)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        out = feat + self.bn(self.g2f_conv(x))
        return out


# ═══════════════════════════════════════════
#  5. TaskGuidedDiffusionPrior (TGDP) —— 已彻底修复
# ═══════════════════════════════════════════
class LightweightDenoiser(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )
        self.down = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU()
        )
        self.mid = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU()
        )
        self.up = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU()
        )
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, f_noisy, f_cond, t):
        t_embed = self.time_mlp(t.float()).unsqueeze(-1).unsqueeze(-1)
        x = torch.cat([f_noisy, f_cond], dim=1)
        x = self.down(x)
        x = self.mid(x) + t_embed
        x = self.up(x)
        return self.out(x)


class TaskGuidedDiffusionPrior(nn.Module):
    def __init__(self, channels, num_steps=3):
        super().__init__()
        betas = torch.linspace(0.0001, 0.02, num_steps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer('alpha_bar', alpha_bar)

        self.denoiser = LightweightDenoiser(channels)
        self.lambda_scale = nn.Parameter(torch.tensor(0.1))
        self.num_steps = num_steps

    def forward(self, feat):
        if self.training:
            B = feat.shape[0]
            t = torch.randint(0, self.num_steps, (B, 1), device=feat.device)
            alpha_bar_t = self.alpha_bar[t.squeeze()].view(-1, 1, 1, 1)

            eps = torch.randn_like(feat)
            f_noisy = torch.sqrt(alpha_bar_t) * feat + torch.sqrt(1 - alpha_bar_t) * eps

            eps_pred = self.denoiser(f_noisy, feat, t)
            f_refined = (f_noisy - torch.sqrt(1 - alpha_bar_t) * eps_pred) / (torch.sqrt(alpha_bar_t) + 1e-8)
        else:
            f_refined = feat

        f_out = feat + self.lambda_scale * (f_refined - feat)
        return f_out


# ═══════════════════════════════════════════
#  6. RoIJitterModule（仅训练时手动调用）
# ═══════════════════════════════════════════
class FeatureSpatialJitter(nn.Module):
    def __init__(self, sigma=0.03, prob=0.5):
        super().__init__()
        self.sigma = sigma
        self.prob = prob

    def forward(self, feat, training=True):
        if not training or torch.rand(1).item() > self.prob:
            return feat, feat
        B = feat.shape[0]
        device = feat.device
        tx = torch.randn(B, device=device) * self.sigma
        ty = torch.randn(B, device=device) * self.sigma
        theta = torch.zeros(B, 2, 3, device=device)
        theta[:, 0, 0] = 1.0
        theta[:, 1, 1] = 1.0
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, feat.size(), align_corners=False)
        feat_jittered = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return feat_jittered, feat.detach()


class ShiftConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=0.1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, feat_jittered, feat_original):
        B, C, H, W = feat_jittered.shape
        f_j = feat_jittered.view(B, C, -1)
        f_o = feat_original.view(B, C, -1)
        cos_sim = F.cosine_similarity(f_j, f_o, dim=1)
        return (1.0 - cos_sim.mean()) * self.loss_weight


class RoIJitterModule(nn.Module):
    """不在 YAML 中使用，在训练脚本 forward 中手动调用。"""
    def __init__(self, sigma_feat=0.03, prob=0.5, consistency_weight=0.1):
        super().__init__()
        self.feat_jitter = FeatureSpatialJitter(sigma_feat, prob)
        self.consistency_loss = ShiftConsistencyLoss(consistency_weight)

    def forward(self, features, training=True):
        if not training:
            return features, torch.tensor(0.0)
        loss_sc = torch.tensor(0.0)
        if isinstance(features, dict):
            features_out = {}
            for level, feat in features.items():
                feat_j, feat_orig = self.feat_jitter(feat, training=True)
                features_out[level] = feat_j
                loss_sc = loss_sc + self.consistency_loss(feat_j, feat_orig)
            loss_sc = loss_sc / max(len(features), 1)
            return features_out, loss_sc
        return features, loss_sc


# ═══════════════════════════════════════════
#  模块注册函数
# ═══════════════════════════════════════════
def register_pgd_modules():
    """在训练脚本最开头调用一次即可。"""
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    for cls in [InputProxy, ChannelSplit, ModalityPromptInjection,
                CrossModalGraphReasoning, TaskGuidedDiffusionPrior]:
        setattr(tasks, cls.__name__, cls)
        setattr(modules, cls.__name__, cls)
    print("[PGD-YOLOv11] 自定义模块注册完成：InputProxy, ChannelSplit, MPI, CMGR, TGDP")


__all__ = [
    'InputProxy', 'ChannelSplit', 'ModalityPromptInjection',
    'CrossModalGraphReasoning', 'TaskGuidedDiffusionPrior',
    'RoIJitterModule', 'register_pgd_modules'
]
