from ultralytics import YOLO

# Initialize model
model = YOLO("yolov8n.pt")
results = model.train(data="./data.yaml", fliplr=False, imgsz=864, batch=16)
