import os
os.environ['WANDB_DISABLED'] = 'true'
from ultralytics import YOLO,RTDETR


# 加载模型配置
#model = YOLO("res18-research2.yaml")  # 从配置文件构建新模型
#model = YOLO("hy_snn_yolov8.yaml")  # 从配置文件构建新模型
model = YOLO("hy_snn_yolov8-cifar-cls.yaml")  # 从配置文件构建新模型
#model = RTDETR("rtdetr-snn-s.yaml")
#model.load("best.pt")
# model = YOLO("./yolov8s.pt")  # 如果有预训练权重，可以使用此行

# 训练模型
#model.train(data="coco1.yaml", device=0, epochs=100, lr0=0.01, lrf=0.0001, freeze=10)  # 训练模型
model.train(data="/root/autodl-tmp/cifar100/",batch=128,imgsz=32,device=0, epochs=250,mixup=1.0,label_smoothing=0.1,workers=16,lr0=0.0005, lrf=0.00001,weight_decay=0.01,warmup_epochs=0)  # 训练模型

# 测试模型
# model = YOLO('runs/detect/train1/weights/last.pt')  # 加载已经训练好的模型（推荐进行测试）

"""
import os
from ultralytics import YOLO,RTDETR

def main():
    os.environ['WANDB_DISABLED'] = 'true'

    # 加载模型配置
    #model = YOLO("res18-research2.yaml")  # 从配置文件构建新模型
    #model = YOLO("hy_snn_yolov8.yaml")  # 从配置文件构建新模型
    model = YOLO("hy_snn_yolov8-cls.yaml")  # 从配置文件构建新模型
    #model = RTDETR("rtdetr-snn-s.yaml")
    #model.load("best.pt")
    # model = YOLO("./yolov8s.pt")  # 如果有预训练权重，可以使用此行

    # 训练模型
    #model.train(data="coco1.yaml", device=0, epochs=100, lr0=0.01, lrf=0.0001, freeze=10)  # 训练模型
    model.train(data="/root/autodl-tmp/ImageNet/", device=0, epochs=200, lr0=0.01, lrf=0.001)  # 训练模型

    # 测试模型
    # model = YOLO('runs/detect/train1/weights/last.pt')  # 加载已经训练好的模型（推荐进行测试）

if __name__ == '__main__':
    main()
"""


"""
import os

os.environ['WANDB_DISABLED'] = 'true'
from ultralytics import YOLO,RTDETR

# Load a model  COCO基准是209，voc基准是55
model =YOLO("hy_snn_yolov8s.yaml")
model.load("best.pt")

# print(model)

#train
# model.train(data="coco.yaml",device=[0],epochs=100)  # train the model
model.train(data="coco5000.yaml", device=[0,1,2,3], batch=40, epochs=170,lr0=0.005,lrf=0.004,warmup_epochs=3,warmup_bias_lr=0.003)  # train the model
"""
#TEST
# model = YOLO('runs/detect/train1/weights/last.pt')  # load a pretrained model (recommended for training)




"""
import os

os.environ['WANDB_DISABLED'] = 'true'
from ultralytics import YOLO,RTDETR

# Load a model  COCO基准是209，voc基准是55
model = RTDETR("rtdetr-snn-s.yaml")
model.load("best.pt")

# print(model)

#train
# model.train(data="coco.yaml",device=[0],epochs=100)  # train the model
model.train(data="coco5000.yaml", device=[0,1,2,3], epochs=200,lr0=0.001, lrf=0.0001, freeze=10)  # train the model

#TEST
# model = YOLO('runs/detect/train1/weights/last.pt')  # load a pretrained model (recommended for training)
"""
"""
import os
from ultralytics import YOLO,RTDETR

def main():
    os.environ['WANDB_DISABLED'] = 'true'

    # 加载模型配置
    # model = YOLO("./ultralytics/cfg/models/v8/yolov8s.yaml")  # 从配置文件构建新模型
    model = RTDETR("rtdetr-snn-s.yaml")
    model.load("best.pt")
    # model = YOLO("./yolov8s.pt")  # 如果有预训练权重，可以使用此行

    # 训练模型
    model.train(data="coco5000.yaml", device=0, epochs=100, lr0=0.01, lrf=0.0001, freeze=10)  # 训练模型

    # 测试模型
    # model = YOLO('runs/detect/train1/weights/last.pt')  # 加载已经训练好的模型（推荐进行测试）

if __name__ == '__main__':
    main()
"""