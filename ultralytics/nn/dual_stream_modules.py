"""
dual_stream_modules.py
──────────────────────────────────────────────────────────────
双流 YOLO11 自定义模块
  - InputProxy        : 恒等代理层，保存 6ch 原始输入以供双流引用
  - ChannelSplit      : 按通道切片，提取 RGB 或 IR 子张量
  - CrossModalFusion  : 模态互补门控融合（核心创新模块）

注册方法（在训练脚本最开头调用）：
    from dual_stream_modules import register_dual_stream_modules
    register_dual_stream_modules()

数据集准备：
    将 RGB 图像与对应红外图像在通道维度拼接为 6ch：
        img_6ch = torch.cat([rgb_3ch, ir_3ch], dim=0)  # (6, H, W)
──────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 依赖 Ultralytics 内部 Conv 封装 ──────────────────────────
try:
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules import C2f
    _HAS_ULTRALYTICS = True
except ImportError:
    _HAS_ULTRALYTICS = False



# ════════════════════════════════════════════════════════════
#  1. InputProxy  —  恒等映射，保留 6ch 输入供后续双流引用
# ════════════════════════════════════════════════════════════
class InputProxy(nn.Module):
    """
    恒等代理层。
    在 YOLO YAML 中作为 layer-0，使后续层可以通过
    `from: 0` 同时拿到完整的 6ch 输入。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x   # 原样透传


# ════════════════════════════════════════════════════════════
#  2. ChannelSplit  —  沿通道维度切片
# ════════════════════════════════════════════════════════════
class ChannelSplit(nn.Module):
    """
    Args:
        start (int): 起始通道索引（包含）
        end   (int): 结束通道索引（不包含）
    Example:
        ChannelSplit(0, 3)  → 取 ch0-2（可见光 RGB）
        ChannelSplit(3, 6)  → 取 ch3-5（红外 IR）
    """
    def __init__(self, start: int, end: int):
        super().__init__()
        self.start = start
        self.end   = end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, self.start:self.end, :, :]

    def extra_repr(self):
        return f"ch[{self.start}:{self.end}]"


# ════════════════════════════════════════════════════════════
#  3. CrossModalFusion  —  核心创新：模态互补门控融合
# ════════════════════════════════════════════════════════════
class ModalGate(nn.Module):
    """
    单模态重要性门控（通道注意力分支）。
    输出归一化权重，用于后续互补加权。
    """
    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        mid = max(c // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(c, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, c, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x).flatten(1))
        mx  = self.mlp(self.max_pool(x).flatten(1))
        w   = torch.sigmoid(avg + mx)          # (B, C)
        return w.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """空间注意力：强调目标显著区域。"""
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        m   = torch.cat([avg, mx], dim=1)
        return torch.sigmoid(self.conv(m))


class CrossModalFusion(nn.Module):
    """
    模态互补门控融合模块（Cross-Modal Adaptive Fusion）

    创新点：
    1. 互补门控  : RGB 权重 + IR 权重 经 Softmax 归一化，保证互补性
                   夜晚/遮挡场景自动提升 IR 权重，白天提升 RGB 权重
    2. 跨模态残差: Concat → 1×1 Conv 提取跨模态交互特征，残差叠加
    3. 空间精炼  : 融合后再施加空间注意力，突出目标区域

    Args:
        c (int): 输入/输出通道数（RGB 与 IR 通道数相同）

    Input:  list[Tensor]  [rgb_feat (B,C,H,W),  ir_feat (B,C,H,W)]
    Output: Tensor        fused_feat (B,C,H,W)
    """
    def __init__(self, c: int, reduction: int = 4):
        super().__init__()
        self.c = c

        # ── 模态门控分支 ──────────────────────────────────────
        self.rgb_gate = ModalGate(c, reduction)
        self.ir_gate  = ModalGate(c, reduction)

        # ── 跨模态交互分支（残差） ────────────────────────────
        self.cross_conv = nn.Sequential(
            Conv(c * 2, c, 1),          # 通道压缩
            Conv(c,     c, 3),          # 局部上下文
        )

        # ── 空间精炼 ──────────────────────────────────────────
        self.spatial_attn = SpatialAttention(k=7)

        # ── 输出投影 ──────────────────────────────────────────
        self.out_proj = Conv(c, c, 1)

        self._init_weights()

    # ─────────────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────
    def forward(self, x):
        """
        x: list or tuple of [rgb_feat, ir_feat]
        """
        rgb, ir = x[0], x[1]

        # 1) 互补门控权重（softmax 归一化保证互补性）
        w_rgb = self.rgb_gate(rgb)          # (B, C, 1, 1)
        w_ir  = self.ir_gate(ir)            # (B, C, 1, 1)
        # Softmax over modality dimension → 白天 rgb↑，夜晚 ir↑
        stack = torch.stack([w_rgb, w_ir], dim=0)   # (2, B, C, 1, 1)
        stack = F.softmax(stack, dim=0)
        w_rgb, w_ir = stack[0], stack[1]

        # 2) 门控加权融合
        weighted = rgb * w_rgb + ir * w_ir  # (B, C, H, W)

        # 3) 跨模态残差（Concat → Conv）
        cross = self.cross_conv(torch.cat([rgb, ir], dim=1))  # (B, C, H, W)

        # 4) 融合 + 空间精炼
        fused = weighted + cross
        fused = fused * self.spatial_attn(fused)

        # 5) 输出投影
        return self.out_proj(fused)

    def extra_repr(self):
        return f"c={self.c}"


__all__ = ['InputProxy', 'ChannelSplit', 'CrossModalFusion']