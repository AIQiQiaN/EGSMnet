import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ==================== 关键修复：必须在加载模型前注册 ====================
from ultralytics.nn.new_models import register_pgd_modules


register_pgd_modules()   # 必须放在 model = YOLO 之前！
# =====================================================================

from ultralytics import YOLO

if __name__ == '__main__':
    # 使用单流 PGD-YOLOv11 配置（已集成所有自定义模块）
    model = YOLO('ultralytics/cfg/models/11/YOLO11-SIg.yaml')

    model.train(data='datasets/DroneVehicle/DroneVehicle.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=64,
                close_mosaic=10,
                workers=4,
                resume=True,
                device='0',
                optimizer='SGD',
                amp=False,
                project='runs2/train',
                name='YOLO11_DroneVehicle_pgd',
                )