import requests
import json

# save image from http url using requests
image_url = "http://192.168.1.150:8080/photo.jpg"
img_data = requests.get(image_url).content
with open("frame.jpg", "wb") as handler:
    handler.write(img_data)
    handler.close()

with open("status.json", "r") as f:
    status = json.load(f)
    print(status["status"])

    status["status"] = "started123456789"
    with open("status.json", "w") as f:
        json.dump(status, f)
    f.close()
