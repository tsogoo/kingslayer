import requests
import cv2
import json
import time
import pytesseract


while True:
    image_url = "http://192.168.1.133:8080/photo.jpg"
    img_data = requests.get(image_url).content
    with open("frame2.jpg", "wb") as handler:
        handler.write(img_data)
        handler.close()
    with open("status.json", "r") as f:
        status = json.load(f)
        print(status["status"])
        if status["status"] == "stopped":
            frame = cv2.imread("frame2.jpg")
            height, width = frame.shape[:2]

            str = pytesseract.image_to_string(
                frame, lang="eng", config="--oem 3 --psm 11"
            )
            print(str)
            if str.lower().find("move") != -1 or str.lower().find("rnove") != -1:
                print(f"The word 'move' was detected in the frame.")
                status["status"] = "started123456789"
                image_url = "http://192.168.1.150:8080/photo.jpg"
                img_data = requests.get(image_url).content
                with open("frame.jpg", "wb") as handler:
                    handler.write(img_data)
                    handler.close()
                board_frame = cv2.imread("frame.jpg")
                cv2.imwrite("frame.jpg", board_frame)
                with open("status.json", "w") as f:
                    json.dump(status, f)
                time.sleep(1)
            else:
                status["status"] = "stopped"
                with open("status.json", "w") as f:
                    json.dump(status, f)
        f.close()
