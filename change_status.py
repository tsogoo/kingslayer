import argparse
import requests
import json


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--webcam_ip",
    required=False,
    default="192.168.1.150:8080",
    help="Image file or directory to predict",
)
parser.add_argument(
    "--status",
    required=False,
    default="move",
    help="Status",
)
parser.add_argument(
    "--light_contour_number",
    required=False,
    default="80",
    help="Light Contour Number",
)
args = parser.parse_args()
webcam_ip = args.webcam_ip
print(f"webcam_ip: {webcam_ip}")
print(f"status: {args.status}")

with open("status.json", "r") as f:
    status = json.load(f)
    if args.status == "move":
        status["status"] = "started123456789"
    else:
        status["status"] = "init_camera"
        status["light_contour_number"] = int(args.light_contour_number)
    with open("status.json", "w") as f:
        json.dump(status, f)
    f.close()
image_url = f"http://{webcam_ip}/photoaf.jpg"
img_data = requests.get(image_url).content
with open("frame.jpg", "wb") as handler:
    handler.write(img_data)
    handler.close()

    print("status changed to started")
