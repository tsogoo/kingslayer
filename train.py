from ultralytics import YOLO

# Initialize model
model = YOLO("yolov9c.pt")
results = model.train(data="./data.yaml", fliplr=False, imgsz=640, batch=4, augment=True, epochs=50, workers=8)
