from ultralytics import YOLO

# Initialize model
model = YOLO("yolov8x.pt")
results = model.train(data="./data_kq.yaml", fliplr=False, batch=4)
