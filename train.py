from ultralytics import YOLO

# Initialize model
model = YOLO("yolov8.yaml")
results = model.train(data="./data.yaml", fliplr=False)
