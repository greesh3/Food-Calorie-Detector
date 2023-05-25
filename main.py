from ultralytics import YOLO
#load model
model = YOLO("yolov8n.yaml")#build a new model from scratch
#use the model
results = model.train(data="config.yaml", epochs=10)#train the model
