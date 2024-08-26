import argparse
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

with open("status.json", "r") as f:
    status = json.load(f)

    status["status"] = "calibrate_board"
    with open("status.json", "w") as f:
        json.dump(status, f)
    f.close()
    print("status changed to calibrate_board")
