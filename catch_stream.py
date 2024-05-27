import cv2
import pytesseract
import json
import time

cap = cv2.VideoCapture("http://192.168.1.56:8080/video")

while True:
    with open("status.json", "r") as f:
        status = json.load(f)
        print(status["status"])
        ret, frame = cap.read()
        if status["status"] == "stopped":
            if not ret:
                break
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            height, width = gray_frame.shape[:2]
            upper_left_section = gray_frame[0 : height // 4, 0 : width // 4]
            lower_left_section = gray_frame[height // 4 :, 0 : width // 4]
            zoomed_section = cv2.resize(
                lower_left_section, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR
            )
            custom_config = r"--oem 3 --psm 11"
            d = pytesseract.image_to_data(
                zoomed_section,
                output_type=pytesseract.Output.DICT,
                lang="eng",
                config=custom_config,
            )

            min_confidence = 45

            # Filter results based on the confidence score
            n_boxes = len(d["level"])
            text = ""
            for i in range(n_boxes):
                if int(d["conf"][i]) > min_confidence:
                    (x, y, w, h) = (
                        d["left"][i],
                        d["top"][i],
                        d["width"][i],
                        d["height"][i],
                    )
                    text = d["text"][i]
                    print(f"Text: {text}, Confidence: {d['conf'][i]}")

            if (
                text.lower().startswith("devel")
                or text.lower().startswith("visit")
                or text.lower().startswith("hotol")
            ):
                print(f"The word '{text}' was detected in the frame.")
                cv2.imwrite("frame.jpg", frame)
                print()
                status["status"] = "starteddddddddddddddddddddd"
                with open("status.json", "w") as f:
                    json.dump(status, f)

        else:
            time.sleep(1)
        f.close()

cap.release()
cv2.destroyAllWindows()
