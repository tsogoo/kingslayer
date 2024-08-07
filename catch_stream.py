import cv2
import json
import time
from ultralytics import YOLO

move_model = YOLO("models/best_move.pt")

while True:
    cap = cv2.VideoCapture("http://192.168.1.98:8080/video")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        break
    with open("status.json", "r") as f:
        status = json.load(f)
        print(status["status"])
        if status["status"] == "stopped":
            height, width = frame.shape[:2]

            results = move_model.predict(frame, save=True, imgsz=640, conf=0.8)

            n_boxes = len(results[0].boxes)

            if len(results[0].boxes) > 0:
                print(f"The word 'move' was detected in the frame.")
                cv2.imwrite("frame.jpg", frame)
                status["status"] = "starteddddddddddddddddddddd"
                with open("status.json", "w") as f:
                    json.dump(status, f)
                time.sleep(1)
            else:
                status["status"] = "stopped"
                with open("status.json", "w") as f:
                    json.dump(status, f)
        f.close()
