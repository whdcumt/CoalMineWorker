import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('替换你的模型权重文件地址') # select your model.pt path
    model.predict(source='要检测的文件的地址',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # classes=0, 是否指定检测某个类别.
                )