#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/9/22 17:16
# @Author  : 钟昊天2021280300
# @FileName: Demo1.py
# @Software: PyCharm
# import libraries
from PIL import Image
from torchvision import models, transforms
from torchvision.utils import draw_bounding_boxes
import torch
import time
import streamlit as st

# 定义app的标题
st.title("Simple Object Detection Application")
st.write("")

# 图像上传控件定义
file_up = st.file_uploader("Upload an image", type = "jpg")

# 定义对象检测模型的类别标签；
inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}


# 定义推理函数
def predict(image):
    """Return predictions.

    Parameters
    ----------
    :param image: uploaded image
    :type image: jpg
    :rtype: list
    :return: none
    """
    # 定义对象检测模型：可以通过取消注释的方法，以启用不同的模型架构；或采用cpu/gpu推理
    # model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True,progress=True)
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True).to('cpu')
    # model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True,progress=True)
    # model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True,progress=True).to('cuda:0')

    # 定义输入图像的预处理方法
    transform = transforms.Compose([
        transforms.ToTensor()])

    # 对输入图像进行预处理，模型推理，打印时间
    img = Image.open(image)
    # batch_t = torch.unsqueeze(transform(img), 0)
    batch_t = torch.unsqueeze(transform(img), 0).to('cpu')
    # torch.cuda.synchronize()
    time_start = time.time()
    model.eval()
    outputs = model(batch_t)
    #torch.cuda.synchronize()
    time_end = time.time()
    time_sum = time_end - time_start
    st.write('Just', time_sum, 'second!')

    # 调试代码
    # st.write(outputs)

    time_start = time.time()
    # 将预测分数大于指定阈值的对象画框标记在图像上。
    score_threshold = .8
    # st.write([inst_classes[label] for label in outputs[0]['labels'][outputs[0]['scores']>score_threshold]])
    output_labels = [inst_classes[label] for label in outputs[0]['labels'][outputs[0]['scores'] > score_threshold]]
    output_boxes = outputs[0]['boxes'][outputs[0]['scores'] > score_threshold]
    images = transform(img) * 255.0;
    images = images.byte()
    result = draw_bounding_boxes(images, boxes=output_boxes, labels=output_labels, width=5)
    st.image(result.permute(1, 2, 0).numpy(), caption='Processed Image.', use_column_width=True)
    time_end = time.time()
    time_sum = time_end - time_start
    st.write('Draw', time_sum, 'second!')
    return outputs

#主函数
if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    labels = predict(file_up)