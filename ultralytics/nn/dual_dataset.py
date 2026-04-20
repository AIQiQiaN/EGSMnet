import cv2
import numpy as np
from pathlib import Path
from ultralytics.data.dataset import YOLODataset

# ── Patch 1：修复 RandomHSV 不支持 6 通道 ─────────────────
from ultralytics.data.augment import RandomHSV

_orig_hsv_call = RandomHSV.__call__

def _patched_hsv_call(self, labels):
    img = labels.get('img')
    if img is not None and img.ndim == 3 and img.shape[2] == 6:
        labels_rgb = dict(labels)
        labels_rgb['img'] = img[:, :, :3].copy()
        labels_rgb = _orig_hsv_call(self, labels_rgb)
        labels['img'][:, :, :3] = labels_rgb['img']
        return labels
    return _orig_hsv_call(self, labels)

RandomHSV.__call__ = _patched_hsv_call

# ── Patch 2：修复 Albumentations(CLAHE) 不支持 6 通道 ─────
from ultralytics.data.augment import Albumentations as _Alb

_orig_alb_call = _Alb.__call__

def _patched_alb_call(self, labels):
    img = labels.get('img')
    if img is not None and img.ndim == 3 and img.shape[2] == 6:
        labels_rgb = dict(labels)
        labels_rgb['img'] = img[:, :, :3].copy()
        labels_rgb = _orig_alb_call(self, labels_rgb)
        labels['img'][:, :, :3] = labels_rgb['img']
        return labels
    return _orig_alb_call(self, labels)

_Alb.__call__ = _patched_alb_call

# ── Patch 3：修复 plot_images 可视化通道不匹配（仅取前3通道）
import ultralytics.utils.plotting as _plotting

_orig_plot = _plotting.plot_images

def _patched_plot(images, *args, **kwargs):
    # 如果是 6 通道，只取 RGB 前3通道用于可视化
    if images.ndim == 4 and images.shape[1] == 6:
        images = images[:, :3, :, :]
    return _orig_plot(images, *args, **kwargs)

_plotting.plot_images = _patched_plot


# ── DualStreamDataset ──────────────────────────────────────
class DualStreamDataset(YOLODataset):

    @staticmethod
    def img2label_paths(img_paths):
        label_paths = []
        for p in img_paths:
            p = Path(p)
            label_dir = p.parent.parent / p.parent.name.replace('images', 'labels')
            label_paths.append(str(label_dir / (p.stem + '.txt')))
        return label_paths

    def load_image(self, i):
        rgb_img, orig_shape, resized_shape = super().load_image(i)
        rgb_path = Path(self.im_files[i])

        ir_dir  = rgb_path.parent.parent / rgb_path.parent.name.replace('images', 'imagesr')
        ir_path = str(ir_dir / rgb_path.name)

        ir_img = cv2.imread(ir_path)
        if ir_img is None:
            print(f"[WARN] IR not found, using zeros: {ir_path}")
            ir_img = np.zeros_like(rgb_img)
        else:
            ir_img = cv2.resize(ir_img, (rgb_img.shape[1], rgb_img.shape[0]))

        return np.concatenate([rgb_img, ir_img], axis=2), orig_shape, resized_shape