import warnings

warnings.filterwarnings('ignore')
from ultralytics import RTDETR
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8.yaml')  # 指定YOLO模型对象，并加载指定配置文件中的模型配置

    # model = YOLO('ultralytics/cfg/models/v8/yolov81015.yaml')  # 指定模型配置

    model.train(data='datasets/hit-uav3/hit_uav.yaml',
                cache=False,  # 是否缓存数据集以加快后续训练速度，False表示不缓存
                imgsz=640,  # 指定训练时使用的图像尺寸，640表示将输入图像调整为640x640像素
                epochs=300,  # 设置训练的总轮数为200轮
                batch=64,  # 设置每个训练批次的大小为16，即每次更新模型时使用16张图片
                close_mosaic=10,  # 设置在训练结束前多少轮关闭 Mosaic 数据增强，10 表示在训练的最后 10 轮中关闭 Mosaic
                workers=4,  # 设置用于数据加载的线程数为8，更多线程可以加快数据加载速度
                # patience=20,  # 在训练时，如果经过50轮性能没有提升，则停止训练（早停机制）
                resume=True, # 断点续训,YOLO初始化时选择last.pt
                device='0',  # 指定使用的设备，'0'表示使用第一块GPU进行训练
                optimizer='SGD',  # 设置优化器为SGD（随机梯度下降），用于模型参数更新
                amp=False,  # 是否使用混合精度训练，True表示使用混合精度训练，加速训练速度并减少内存占用
                project='runs/train',
                name='exp_Basic_HIT',
                )

