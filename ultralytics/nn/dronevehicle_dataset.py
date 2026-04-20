"""
DroneVehicle 6通道数据集（RGB + IR 单流融合）
─────────────────────────────────────────────
- 从 images/ 加载 RGB (3ch)
- 从 imagesr/ 加载 IR  (3ch, 若为单通道灰度则复制到3通道)
- 沿通道维度拼接成 6ch 张量 (RGB[0:3] + IR[3:6])
- 完全兼容 YOLOv11 的 mosaic / mixup / hsv / flip 等增强
- 与 PGD-YOLOv11 的 InputProxy + ChannelSplit + MPI 完美配合
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path

from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import LOGGER


class DroneVehicleDataset(YOLODataset):
    """
    6-channel RGB+IR 拼接数据集。
    继承 YOLODataset，仅重写图像加载逻辑，保留全部 Ultralytics 增强管线。
    """

    def __init__(self, *args, **kwargs):
        # data 字典中可通过 `channels=6` 传入；若未传，默认6
        super().__init__(*args, **kwargs)
        self._ir_cache = {}   # 可选：缓存 IR 路径映射

    # ────────────────────────────────────────
    # 核心：重写 load_image，让返回的图像就是 6ch
    # ────────────────────────────────────────
    def load_image(self, i, rect_mode=True):
        """
        返回: (img_6ch[H,W,6], (h0,w0), (h,w))
        Ultralytics 后续的 letterbox / mosaic 会直接在 6ch 上工作。
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]

        if im is None:  # 尚未缓存
            # ---------- 1. 加载 RGB ----------
            if fn.exists():
                rgb = np.load(fn)  # .npy 缓存
            else:
                rgb = cv2.imread(f)  # BGR
                if rgb is None:
                    raise FileNotFoundError(f"RGB 图像未找到: {f}")
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # ---------- 2. 计算 IR 路径 ----------
            ir_path = self._rgb_to_ir_path(f)
            ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
            if ir is None:
                raise FileNotFoundError(f"IR 图像未找到: {ir_path}")

            # IR 统一成 3ch（若单通道则复制3份，方便下游 ChannelSplit 切 [3:6]）
            if ir.ndim == 2:
                ir = np.stack([ir, ir, ir], axis=-1)
            elif ir.ndim == 3 and ir.shape[2] == 1:
                ir = np.concatenate([ir, ir, ir], axis=-1)
            elif ir.ndim == 3 and ir.shape[2] == 3:
                ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"IR shape 非法: {ir.shape}")

            # ---------- 3. 对齐尺寸（兜底，通常已对齐）----------
            if ir.shape[:2] != rgb.shape[:2]:
                ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

            # ---------- 4. 拼接成 6ch ----------
            im = np.concatenate([rgb, ir], axis=2).astype(np.uint8)  # (H,W,6)

            h0, w0 = im.shape[:2]
            # resize 到 imgsz（与父类行为一致）
            r = self.imgsz / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                # cv2.resize 对 6ch 也支持（>4 通道需用 numpy 循环兜底）
                if im.shape[2] <= 4:
                    im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
                else:
                    im_rgb = cv2.resize(im[..., :3], (int(w0 * r), int(h0 * r)), interpolation=interp)
                    im_ir = cv2.resize(im[..., 3:], (int(w0 * r), int(h0 * r)), interpolation=interp)
                    im = np.concatenate([im_rgb, im_ir], axis=2)

            self.ims[i] = im
            self.im_hw0[i] = (h0, w0)
            self.im_hw[i] = im.shape[:2]
            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    # ────────────────────────────────────────
    # 工具：根据 RGB 路径推断 IR 路径
    # ────────────────────────────────────────
    def _rgb_to_ir_path(self, rgb_path: str) -> str:
        """
        train/images/xxx.jpg  →  train/imagesr/xxx.jpg
        val/images/xxx.jpg    →  val/imagesr/xxx.jpg
        """
        if rgb_path in self._ir_cache:
            return self._ir_cache[rgb_path]

        p = Path(rgb_path)
        # 把父目录名 images → imagesr
        ir_dir = p.parent.parent / "imagesr"
        # 文件名保持一致，先试相同后缀
        ir_path = ir_dir / p.name
        if not ir_path.exists():
            # 兜底：尝试常见后缀
            for ext in [".jpg", ".png", ".tif", ".tiff", ".bmp"]:
                cand = ir_dir / (p.stem + ext)
                if cand.exists():
                    ir_path = cand
                    break

        ir_path = str(ir_path)
        self._ir_cache[rgb_path] = ir_path
        return ir_path

    # ────────────────────────────────────────
    # 关闭通道数校验（父类默认只允许3ch）
    # ────────────────────────────────────────
    def build_transforms(self, hyp=None):
        """保持父类增强管线，但把 hsv 只作用于 RGB 前3通道。"""
        transforms = super().build_transforms(hyp)
        # 包装 hsv 增强，避免污染 IR 通道
        return _wrap_hsv_rgb_only(transforms)


# ═══════════════════════════════════════════
#  辅助函数：仅对 RGB 部分做 HSV 增强
# ═══════════════════════════════════════════
def _wrap_hsv_rgb_only(transforms):
    """
    Ultralytics 的 Albumentations / RandomHSV 默认假设3ch。
    这里把 img 的前3通道拆出去做 HSV，再拼回 IR。
    """
    from ultralytics.data.augment import RandomHSV

    for t in getattr(transforms, "transforms", []):
        if isinstance(t, RandomHSV):
            orig_call = t.__call__

            def patched(labels, _orig=orig_call):
                img = labels["img"]
                if img.ndim == 3 and img.shape[2] >= 6:
                    rgb, ir = img[..., :3], img[..., 3:]
                    labels["img"] = np.ascontiguousarray(rgb)
                    labels = _orig(labels)
                    labels["img"] = np.concatenate([labels["img"], ir], axis=2)
                    return labels
                return _orig(labels)

            t.__call__ = patched
    return transforms