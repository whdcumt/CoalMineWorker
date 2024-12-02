import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('替换你的模型权重文件地址')
    model.val(data=r'数据集配置的yaml文件地址',
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # 这个保存coco精度指标的开关
              project='runs/val',
              name='exp',
              )