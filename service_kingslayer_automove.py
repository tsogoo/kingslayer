import cv2
import os
import argparse
import json
import time
import logging
from kingslayer import Kingslayer

# Configure logging to use the systemd default (stdout)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "--image",
    required=False,
    default="frame.jpg",
    help="Image file or directory to predict",
)
parser.add_argument(
    "--board_weight",
    required=False,
    default="best_board.pt",
    help="Weight of the training model",
)
parser.add_argument(
    "--chess_model_weight",
    required=False,
    default="best_yolo11s.pt",
    help="Weight of the training model",
)
args = parser.parse_args()
image = args.image
board_weight = args.board_weight
chess_model_weight = args.chess_model_weight


chess = Kingslayer(board_weight, chess_model_weight)
# best_move = chess.process_from_image(image)
# print("moving")
# chess.robot.move(
#     chess.chess_engine_helper,
#     chess,
#     'e2e4',
#     True,
# )
# chess.chess_engine_helper.destroy()
# exit()
while True:
    with open("status.json", "r") as f:
        try:
            status = json.load(f)
            if (
                status["status"] == "starting"
                or status["status"] == "started"
                or status["status"] == "init_camera"
            ):
                try:
                    if status["status"] == "init_camera":
                        chess.init_camera(image, status["light_contour_number"])
                        status["status"] = "stopped"
                        print("Stoppedddddddddddddddddddd")
                        with open("status.json", "w") as f:
                            json.dump(status, f)
                    else:
                        chess.image_url = status["image_url"]
                        chess.is_white = status["is_white"]
                        chess.automove = True
                        if status["status"] == "starting":
                            chess.init_chess_engine()
                            chess.previous_fen = None
                            status["status"] = "started"
                            print("Started")
                            with open("status.json", "w") as f:
                                json.dump(status, f)
                        best_move = chess.trigger(image)
                        # time.sleep(.1)  # delay 1 second
                except Exception as e:
                    print("Error:", e)

            elif status["status"] == "calibrate_board":
                chess.robot.calibrate_board()
                status["status"] = "stopped"
                print("Stoppedeeeeeeeeeeeeeeeeeeeeeeee")
                with open("status.json", "w") as f:
                    print("dumping status")
                    json.dump(status, f)
        except Exception as e:
            print("Error:", e)
            chess.init_chess_engine()
    f.close()
    time.sleep(0.1)
