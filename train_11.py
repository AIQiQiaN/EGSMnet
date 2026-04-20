import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics.nn.dual_dataset import DualStreamDataset
import ultralytics.data.build as build_module
build_module.YOLODataset = DualStreamDataset

if __name__ == '__main__':

    model = YOLO('ultralytics/cfg/models/11/YOLO11_dou.yaml')  # 双流 YOLO11 配置

    model.train(
        data='datasets/DroneVehicle/DroneVehicle.yaml',  # DroneVehicle 数据集
        cache=False,          # 是否缓存数据集
        imgsz=640,            # 输入图像尺寸（DroneVehicle 原图 800×450，会自动 letterbox）
        epochs=300,           # 训练总轮数
        batch=10,             # 批次大小（双流参数量约为单流 2 倍，适当减小）
        close_mosaic=16,      # 最后 16 轮关闭 Mosaic 增强
        workers=4,            # 数据加载线程数
        device='0',           # GPU，多卡用 '0,1'
        optimizer='SGD',      # 优化器
        amp=False,            # 混合精度（双流模型建议先关闭，跑通后再开）
        lr0=0.01,             # 初始学习率
        lrf=0.01,             # 最终学习率比例
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3.0,
        # DroneVehicle 特殊增强设置（无人机俯视场景）
        fliplr=0.5,           # 左右翻转
        flipud=0.5,           # 上下翻转（俯视场景上下翻转有效）
        degrees=45.0,         # 随机旋转（俯视场景目标方向随机）
        mosaic=1.0,
        mixup=0.1,
        project='runs1/train',
        name='exp_DualStream_YOLO11_DroneVehicle_300',
    )