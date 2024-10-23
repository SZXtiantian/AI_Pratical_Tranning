from ultralytics import YOLO

model  = YOLO('./exp2/yolov5n.pt')

model.train(data = './exp2/data1/data.yaml', epoch = 50)

model.val()