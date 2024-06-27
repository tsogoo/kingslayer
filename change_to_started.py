import cv2
import json
import time
from ultralytics import YOLO


move_model = YOLO("models/best_move.pt")

cap = cv2.VideoCapture("http://192.168.1.205:8080/video")

ret, frame = cap.read()
if not ret:
    exit()
with open("status.json", "r") as f:
    status = json.load(f)
    print(status["status"])

    cv2.imwrite("frame.jpg", frame)
    status["status"] = "started123456789"
    with open("status.json", "w") as f:
        json.dump(status, f)
    time.sleep(1)
    f.close()

cap.release()
cv2.destroyAllWindows()
