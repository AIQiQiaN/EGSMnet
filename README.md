# EGSMnet

# Introduce
  
The code in this repository directly corresponds to the manuscript submitted to The Visual Computer entitled “EGSMnet: Adaptive Feature Enhancement Framework for Lightweight Small Object Detection in UAV Aerial Imagery”. Readers are kindly requested to cite this manuscript.

# Environment

pytorch 2.5.0

torchvision 0.20.0

cuda 12.4


# Datasets

The link to VisDrone2019: https://github.com/VisDrone/VisDrone-Dataset.

The link to Hit-UAV: https://github.com/suojjiashun/HIT-UAV-Infrared-Thermal-Dataset.

📊 Experimental Results

VisDrone2019: 41.6% mAP@50 / 24.8% mAP@50-95 (+10.5 / +7.0 vs YOLOv8-n)
HIT-UAV: 95.1% mAP@50 / 61.5% mAP@50-95
The model size is only 1/3 of YOLOv8-n, with significantly reduced computation, suitable for real-time edge deployment

# 1. 克隆仓库
git clone https://github.com/AIQiQiaN/EGSMnet.git
cd EGSMnet

# 2. 安装依赖
pip install -r requirements.txt

# 可见光推理
python detect.py --weights runs/train/best_vis.pt --source test_graph/ --imgsz 640 --conf 0.25

# 红外推理
python detect.py --weights runs/train/best_Hit.pt --source test_graph/ --imgsz 640 --conf 0.25


