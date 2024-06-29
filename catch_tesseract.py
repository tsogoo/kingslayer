import cv2
import json
import time
import pytesseract

while True:
    cap = cv2.VideoCapture("http://192.168.1.112:8080/video")
    ret, frame = cap.read()
    cv2.imwrite("frame2.jpg", frame)
    cap.release()
    if not ret:
        break
    with open("status.json", "r") as f:
        status = json.load(f)
        print(status["status"])
        if status["status"] == "stopped":
            height, width = frame.shape[:2]

            str = pytesseract.image_to_string(
                frame, lang="eng", config="--oem 3 --psm 11"
            )
            print(str)
            if str.lower().find("move") != -1 or str.lower().find("rnove") != -1:
                print(f"The word 'move' was detected in the frame.")
                status["status"] = "starteddddddddddddddddddddd"
                board_cap = cv2.VideoCapture("http://192.168.1.205:8080/video")
                ret, board_frame = board_cap.read()
                board_cap.release()
                if ret:
                    cv2.imwrite("frame.jpg", board_frame)
                    with open("status.json", "w") as f:
                        json.dump(status, f)
                time.sleep(1)
            else:
                status["status"] = "stopped"
                with open("status.json", "w") as f:
                    json.dump(status, f)
        f.close()

cap.release()
cv2.destroyAllWindows()
