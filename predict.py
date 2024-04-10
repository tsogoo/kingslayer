from ultralytics import YOLO
import argparse

# Initialize model

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--image", required=False,  default="datasets/test/images/", help="Image file or directory to predict")
parser.add_argument("--weight", required=False, default="runs/detect/train/weights/best.pt", help="Weight of the training model")
args = parser.parse_args()


model = YOLO(args.weight)
model.predict(args.image, save=True, imgsz=320, conf=0.2)
