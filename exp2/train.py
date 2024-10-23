from ultralytics import YOLO

model  = YOLO('yolov5n.pt')

model.train(data = './exp2/data1/data.yaml', epoch = 50)

model.val()