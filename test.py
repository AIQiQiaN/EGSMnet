"""
EGSMnet / YOLOv8 模型 FPS 评估脚本 - 带检测结果照片输出版
直接运行：python eval_fps.py
"""

import time
import warnings
import numpy as np
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════
# ★ 在这里修改您的配置 ★
# ═══════════════════════════════════════════════

# WEIGHTS = r"runs/train/exp_EGConv_Vis_P1+P2_In/weights/best.pt"  # ←←← 您的模型路径
WEIGHTS = r"yolo11x.pt"  # ←←← 您的模型路径
IMGSZ   = 640
IMGSZ   = 640
DEVICE  = "cuda"          # "cuda" 或 "cpu"
BATCH   = 1
ITERS   = 300

# 真实数据测试路径（支持单张图片 或 文件夹）
SOURCE  = r"D:\project\YOLOV8\923\datasets\VisDrone\VisDrone2019-DET-test-dev\images\0000278_02351_d_0000002.jpg"

# 是否保存检测结果照片（强烈建议开启）
SAVE_RESULTS = True
OUTPUT_DIR   = r"detection_results"   # 保存文件夹（会自动创建）

# ═══════════════════════════════════════════════

def make_dummy_input(imgsz, batch_size, device):
    return torch.randn(batch_size, 3, imgsz, imgsz).to(device)

def warmup(model, dummy_input, rounds=10):
    print(f"  [Warm-up] {rounds} rounds ...")
    with torch.no_grad():
        for _ in range(rounds):
            _ = model(dummy_input)
    if dummy_input.is_cuda:
        torch.cuda.synchronize()

def benchmark_torch(model, dummy_input, num_iters):
    latencies = []
    with torch.no_grad():
        for _ in range(num_iters):
            if dummy_input.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if dummy_input.is_cuda:
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
    return latencies

def benchmark_torch_fp16(model, dummy_input, num_iters):
    dummy_fp16 = dummy_input.half()
    latencies = []
    with torch.no_grad():
        for _ in range(num_iters):
            if dummy_fp16.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_fp16)
            if dummy_fp16.is_cuda:
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
    return latencies

def report(latencies, label=""):
    arr = np.array(latencies) * 1000
    arr_trim = arr[int(len(arr) * 0.05): int(len(arr) * 0.95)]
    fps = 1000 / arr_trim.mean()

    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  📊 [{label}] 推理性能报告")
    print(sep)
    print(f"  样本数 (去首尾 5%): {len(arr_trim)}")
    print(f"  平均延迟:  {arr_trim.mean():.2f} ms")
    print(f"  中位延迟:  {np.median(arr_trim):.2f} ms")
    print(f"  P95 延迟:  {np.percentile(arr_trim, 95):.2f} ms")
    print(f"  最小延迟:  {arr_trim.min():.2f} ms")
    print(f"  最大延迟:  {arr_trim.max():.2f} ms")
    print(f"  ✅ FPS:    {fps:.1f}")
    print(sep)
    return fps

def eval_pytorch():
    from ultralytics import YOLO

    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("  ⚠️  CUDA 不可用，自动回退至 CPU")
        device = "cpu"

    weights = Path(WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(f"权重文件不存在，请检查路径：{weights.resolve()}")

    print(f"\n[PyTorch FP32] 加载模型: {weights}")
    model = YOLO(str(weights))
    model.model.eval()
    model.model.to(device)

    # 参数量 & GFLOPs
    try:
        from thop import profile as thop_profile
        dummy = make_dummy_input(IMGSZ, 1, device)
        flops, params = thop_profile(model.model, inputs=(dummy,), verbose=False)
        print(f"  参数量:  {params / 1e6:.2f} M")
        print(f"  GFLOPs:  {flops / 1e9:.2f}")
    except Exception:
        print("  (thop 未安装，跳过参数量统计)")

    # FP32 测试
    dummy_input = make_dummy_input(IMGSZ, BATCH, device)
    warmup(model.model, dummy_input)
    print(f"  [FP32 Benchmark] {ITERS} iters ...")
    latencies_fp32 = benchmark_torch(model.model, dummy_input, ITERS)
    report(latencies_fp32, f"FP32 | {device.upper()} | bs={BATCH} | {IMGSZ}×{IMGSZ}")

    # FP16 半精度测试
    if device == "cuda":
        print(f"\n[PyTorch FP16] 半精度推理测试 ...")
        model_fp16 = YOLO(str(weights))
        model_fp16.model.eval()
        model_fp16.model.half().to(device)
        warmup(model_fp16.model, dummy_input.half(), rounds=10)
        latencies_fp16 = benchmark_torch_fp16(model_fp16.model, dummy_input, ITERS)
        report(latencies_fp16, f"FP16 | {device.upper()} | bs={BATCH} | {IMGSZ}×{IMGSZ}")

    # 真实图像端到端测试（带检测结果保存）
    if SOURCE:
        eval_real(model, device)

def eval_real(model, device):
    import cv2
    src = Path(SOURCE)
    print(f"\n[Real-world] 真实数据端到端测试: {src}")

    frames = []
    if src.is_file():                     # 单张图片
        frame = cv2.imread(str(src))
        if frame is not None:
            frames = [frame] * 50          # 重复50次保证测量稳定
    elif src.is_dir():                    # 文件夹
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            files.extend(sorted(src.glob(ext)))
        frames = [cv2.imread(str(f)) for f in files[:200]]
        frames = [f for f in frames if f is not None]
    else:
        print(f"  ⚠️  路径不存在或格式不支持: {src}")
        return

    if not frames:
        print("  ⚠️  未找到有效图像，跳过真实数据测试")
        return

    print(f"  共加载 {len(frames)} 张图像，开始测试 ...")

    # 创建输出文件夹
    if SAVE_RESULTS:
        output_dir = Path(OUTPUT_DIR)
        output_dir.mkdir(exist_ok=True)
        print(f"  📁 检测结果将保存至: {output_dir.resolve()}")

    # 预热
        # 预热
        for _ in range(5):
            model.predict(frames[0], imgsz=IMGSZ, device=device, verbose=False, conf=0.25)

        latencies = []
        for idx, frame in enumerate(frames):
            t0 = time.perf_counter()

            # ←←← 这里就是置信度设置的位置
            results = model.predict(
                frame,
                imgsz=IMGSZ,
                device=device,
                verbose=False,
                conf=0.3,  # ←←← 置信度阈值（可修改）
                iou=0.45  # 可选：NMS的IOU阈值
            )

            latencies.append(time.perf_counter() - t0)

            # 保存检测结果
            if SAVE_RESULTS:
                result = results[0]
                plotted_img = result.plot()  # 画框时也会使用这个conf阈值过滤
                save_path = Path(OUTPUT_DIR) / f"detected_{idx + 1:04d}.jpg"
                cv2.imwrite(str(save_path), plotted_img)

    report(latencies, f"Real-world E2E | {device.upper()} | {IMGSZ}×{IMGSZ}")

if __name__ == "__main__":
    eval_pytorch()