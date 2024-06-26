import cv2
from ultralytics import YOLO
import argparse
import numpy as np
import math
from engine.helper import ChessEngineHelper
import chess
import json
import time
import os
import yaml
from robot_arm.robot import Robot
from common.config import get_config

from lib_contour import (
    get_enhanced_image,
    get_contoured_image,
    get_blurry_image,
    get_noisy_image,
)


def get_chess_model_letter(model_name):
    if model_name[:2] == "Kt":
        model_name = f"Nt{model_name[2]}"
    if model_name[2] == "0":
        return model_name[0].lower()
    else:
        return model_name[0].upper()


def distance(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def find_outer_corners(img, pts):

    rows, cols = img.shape

    bl_dst = br_dst = tl_dst = tr_dst = float("inf")

    for p in pts:

        p = p[0]

        if distance(p, (cols * 0, rows * 1)) < bl_dst:
            bl_dst = distance(p, (cols * 0, rows * 1))
            bl = p

        if distance(p, (cols * 1, rows * 1)) < br_dst:
            br_dst = distance(p, (cols * 1, rows * 1))
            br = p

        if distance(p, (cols * 0, rows * 0)) < tl_dst:
            tl_dst = distance(p, (cols * 0, rows * 0))
            tl = p

        if distance(p, (cols * 1, rows * 0)) < tr_dst:
            tr_dst = distance(p, (cols * 1, rows * 0))
            tr = p

    pts1 = np.float32([bl, br, tl, tr])

    return pts1


def find_max_contour_area(contours):

    max_area = 0 - float("inf")
    max_c = None

    for c in contours:
        area = cv2.contourArea(c)

        if area > max_area:
            max_area = area
            max_c = c

    return [max_c]


class Kingslayer:
    def __init__(self, board_weight, chess_model_weight):
        self.models = []
        self.pts_square = []
        self.pts_perspective = []
        self.margin = 86
        self.CONFIDENCE_THRESHOLD = 0.6
        self.CROP_SIZE = 640

        # conf
        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.yaml"), "r"
        ) as file:
            config = yaml.safe_load(file)
        self.config = get_config(config, "app")

        # conf
        with open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.yaml"), "r"
        ) as file:
            config = yaml.safe_load(file)
        self.config = get_config(config, "app")

        # initialize models yolo detector
        self.board_model = YOLO(f"models/{board_weight}")
        self.chess_model = YOLO(f"models/{chess_model_weight}")

        # initialize chess engine helper
        self.init_chess_engine()

        self.robot = Robot(config=config)
        # self.robot.move(self.chess_engine_helper, self, 'e7e5', self.chess_engine_helper.board.turn)
        # self.robot.go_home()

    def init_chess_engine(self):
        self.chess_engine_helper = ChessEngineHelper(
            get_config(self.config, "chess_engine")
        )
        self.chess_engine_helper.initialize_board(None)

    def augment_image(self, img):

        augmented = cv2.imread(img)
        # augmented = get_enhanced_image(augmented)
        # augmented = get_blurry_image(augmented)
        # augmented = get_contoured_image(augmented)
        augmented = get_noisy_image(augmented)
        cv2.imwrite("augmented.jpg", augmented)
        return "augmented.jpg"

    def converted_robot_point(self, xy):
        w1 = self.robot.board_square_size * 8 + 2 * self.robot.board_margin_size
        w2 = self.robot.board_square_size * 8
        scale = 1
        x = int(
            xy[0] * w1 / self.CROP_SIZE
            - self.robot.board_margin_size
        )
        y = int(
            xy[1] * scale * w1 / self.CROP_SIZE
            - self.robot.board_margin_size
           
        )
        return (x, y)

    def detect_models(self, image):

        frame_path = image
        conf = self.CONFIDENCE_THRESHOLD
        for j in range(8):
            if j < 2:
                conf = 0.8
            elif j < 3:
                conf = 0.7
            elif j < 4:
                conf = 0.6
            elif j < 5:
                conf = 0.5
            print("=============")
            print("Confidence:", conf)
            print(frame_path)
            chess_results = self.chess_model.predict(
                self.augment_image(frame_path), save=True, imgsz=864, conf=conf
            )
            img = cv2.imread(frame_path)
            frame_path = f"augmented{j}.jpg"

            # if len(chess_results) == 0 and conf <= self.CONFIDENCE_THRESHOLD:
            #     return

            for result in chess_results:
                # if len(result.boxes.xywh) == 0 and conf <= self.CONFIDENCE_THRESHOLD:
                #     return

                for i in range(len(result.boxes.xywh)):
                    x, y, w, h = result.boxes.xywh[i]
                    x, y, xend, yend = result.boxes.xyxy[i]
                    dx = 0
                    dy = h / 7

                    # if h > img.shape[0] / 16:
                    #     dy = h / 8

                    # fill detected model with white
                    img = cv2.rectangle(
                        img,
                        (int(x + dx), int(y + dy)),
                        (int(x + w), int(y + h)),
                        (100, 100, 100),
                        -1,
                    )
                    # img = cv2.rectangle(
                    #     img,
                    #     (int(x), int(y)),
                    #     (int(x + w), int(y + h)),
                    #     (0, 255, 255),
                    #     1,
                    # )

                    # gettting center of the detected model
                    y = int(y + h * 5 / 8 + (self.CROP_SIZE - y) / 80)
                    x = int(x + w / 2 + (self.CROP_SIZE / 2 - x) / 60)

                    cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

                    self.models.append(
                        (
                            result.names[int(result.boxes.cls[i])],
                            self.converted_robot_point(
                                (x, y)
                            ),  # TODO: Ene coordinate yag boardiin a1 bulangaas zaitai coordinate
                            result.boxes.conf[i],
                        )
                    )
                    # cv2 circle on x , y with radius 3
            self.print_detected_board()
            cv2.imwrite(frame_path, img)

    def print_detected_board(self):
        # set blank image
        img = np.zeros(
            (self.robot.board_square_size * 8, 8 * self.robot.board_square_size, 3),
            np.uint8,
        )

        # draw board
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    cv2.rectangle(
                        img,
                        (
                            j * self.robot.board_square_size,
                            i * self.robot.board_square_size,
                        ),
                        (
                            (j + 1) * self.robot.board_square_size,
                            (i + 1) * self.robot.board_square_size,
                        ),
                        (255, 255, 255),
                        -1,
                    )
                else:
                    cv2.rectangle(
                        img,
                        (
                            j * self.robot.board_square_size,
                            i * self.robot.board_square_size,
                        ),
                        (
                            (j + 1) * self.robot.board_square_size,
                            (i + 1) * self.robot.board_square_size,
                        ),
                        (0, 0, 0),
                        -1,
                    )

        # point models on the image
        for model in self.models:
            cv2.circle(
                img,
                (model[1][0], model[1][1]),
                3,
                (255, 0, 255),
                -1,
            )

        # save image with name board_file.jpg
        cv2.imwrite("board_file.jpg", img)

    def init_perspective_data(self, xoffset, yoffset, w, h, xend, yend, img):
        # crop image with x,y,w,h
        color = img[
            int(yoffset - self.margin) : int(yend + self.margin),
            int(xoffset - self.margin) : int(xend + self.margin),
        ]
        cv2.imwrite("im.jpg", color)

        # gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        # gray = cv2.adaptiveThreshold(
        #     color, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5
        # )

        ret, gray = cv2.threshold(color, 160, 255, 0)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = find_max_contour_area(contours)
        # gray = cv2.drawContours(gray, contours, -1, (0, 255, 0), 2).copy()

        # contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = find_max_contour_area(contours)

        cv2.imwrite("gray.jpg", gray)

        c = contours[0]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        pts = find_outer_corners(gray, approx)

        min_x = 10000
        min_y = 10000
        max_x = 0
        max_y = 0

        for p in pts:
            if p[0] < min_x:
                min_x = p[0]
            if p[0] > max_x:
                max_x = p[0]
            if p[1] < min_y:
                min_y = p[1]
            if p[1] > max_y:
                max_y = p[1]

            cv2.circle(
                img,
                (int(p[0] + xoffset - self.margin), int(p[1] + yoffset - self.margin)),
                3,
                (255, 0, 0),
                -1,
            )
        yend = int((pts[0][1] + pts[1][1]) / 2 + yoffset - self.margin)
        xend = int(pts[1][0] + xoffset - self.margin)
        ystart = int((pts[2][1] + pts[3][1]) / 2 + yoffset - self.margin)
        xstart = int(pts[0][0] + xoffset - self.margin)
        h = yend - ystart
        w = xend - xstart

        cv2.imwrite("im.jpg", img)
        # cv2.rectangle(img, (xstart, ystart), (xend, yend), (0, 255, 255), 1)
        # cv2.rectangle(
        #     color, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 255), 1
        # )

        self.pts_square = np.float32(
            [[xend - w, yend - w], [xend - w, yend], [xend, yend - w], [xend, yend]]
        )

        self.pts_perspective = np.float32(
            [
                [pts[2][0], pts[2][1]],
                [pts[0][0], pts[0][1]],
                [pts[3][0], pts[3][1]],
                [pts[1][0], pts[1][1]],
            ]
        )
        return color

    def process_warp(self, input_coordinates):
        M = cv2.getPerspectiveTransform(self.pts_perspective, self.pts_square)
        output_coordinates = cv2.perspectiveTransform(
            np.array([input_coordinates], dtype="float32"), M
        )
        return output_coordinates[0]

    def process_from_image(self, image):

        board_results = self.board_model.predict(image, save=False, imgsz=640, conf=0.7)
        # Load the image using OpenCV
        img = cv2.imread(image)
        img = get_blurry_image(img)

        cropped_image = None
        for result in board_results:
            x, y, w, h = result.boxes.xywh[0]
            self.x, self.y, self.xend, self.yend = result.boxes.xyxy[0]
            # img = cv2.rectangle(
            #     img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 1
            # )
            if w < 300 and h < 300:
                continue
            cropped_image = self.init_perspective_data(
                self.x, self.y, w, h, self.xend, self.yend, img
            )
            break

        self.models = []
        M = cv2.getPerspectiveTransform(
            self.pts_perspective,
            np.float32(
                [
                    [0, 0],
                    [0, self.CROP_SIZE],
                    [self.CROP_SIZE, 0],
                    [self.CROP_SIZE, self.CROP_SIZE],
                ]
            ),
        )

        # full image warped to square
        cropped_image = cv2.warpPerspective(
            cropped_image, M, (self.CROP_SIZE, self.CROP_SIZE)
        )

        warped_image_path = "warped_image.jpg"
        cv2.imwrite(warped_image_path, cropped_image)
        self.detect_models(warped_image_path)

        print(x, y, w, h)

        conf = self.generate_chess_board_array()
        conf = list(map(list, zip(*conf[::-1])))
        self.print_chess_board_array(conf)
        # rotate conf 180 degree
        conf = list(map(list, zip(*conf[::-1])))
        conf = list(map(list, zip(*conf[::-1])))
        self.print_chess_board_array(conf)
        # conf = [
        #     ["r", "n", "b", "q", "k", "b", "n", "r"],
        #     ["p", "p", "p", "p", " ", "p", "p", "p"],
        #     [" ", " ", " ", " ", " ", " ", " ", " "],
        #     [" ", " ", " ", " ", "p", " ", " ", " "],
        #     [" ", " ", " ", " ", "P", " ", " ", " "],
        #     [" ", " ", " ", " ", " ", " ", " ", " "],
        #     ["P", "P", "P", "P", " ", "P", "P", "P"],
        #     ["R", "N", "B", "Q", "K", "B", "N", "R"],
        # ]
        best_move = None
        best_move = self.get_movement(conf)

        # Continue with your existing code...

        # best_move = "a1a4"
        print(best_move)
        self.robot.move(
            self.chess_engine_helper,
            self,
            best_move,
            self.chess_engine_helper.board.turn,
        )
        print(self.robot.commands)
        return best_move

    def get_movement(self, conf):
        fen_rows = []
        for row in conf:
            empty = 0
            fen_row = ""
            for cell in row:
                if cell == " ":
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    fen_row += cell
            if empty > 0:
                fen_row += str(empty)
            fen_rows.append(fen_row)
        fen = "/".join(fen_rows) + " w KQkq - 0 1"
        self.chess_engine_helper.initialize_board(fen)
        best_move = self.chess_engine_helper.get_best_move()
        return best_move

    def generate_chess_board_array(self):
        chess_board_array = []
        for i in range(8):
            board_row = []
            for j in range(8):
                node_value = " "
                value = (
                    self.robot.board_square_size * j,
                    self.robot.board_square_size * i,
                    self.robot.board_square_size * (j + 1),
                    self.robot.board_square_size * (i + 1),
                )
                conf = 0
                for model in self.models:
                    if (
                        model[1][0] > value[0]
                        and model[1][1] > value[1]
                        and model[1][0] < value[2]
                        and model[1][1] < value[3]
                    ):
                        if model[2] > conf:
                            conf = model[2]
                            node_value = get_chess_model_letter(model[0])
                board_row.append(node_value)
            chess_board_array.append(board_row)
        return chess_board_array

    def print_chess_board_array(self, chess_board_array):
        print("============== Chess board ==============")
        for row in chess_board_array:
            print(row)

    def get_figure_actual_position(self, x, y, is_occupied=False):

        # new board square height, width = 370/8 = 46.25
        # x, y => figure's square coordinate on board, is_occupied => whether square is occupied
        if is_occupied:
            for model in self.models:
                if (
                    self.robot.board_square_size * 8 - model[1][1]
                    > x * self.robot.board_square_size
                    and model[1][0] > y * self.robot.board_square_size
                    and self.robot.board_square_size * 8 - model[1][1]
                    < (x + 1) * self.robot.board_square_size
                    and model[1][0] < (y + 1) * self.robot.board_square_size
                ):
                    return (self.robot.board_square_size * 8 - model[1][1], model[1][0])
            return (
                x * self.robot.board_square_size + self.robot.board_square_size / 2,
                y * self.robot.board_square_size + self.robot.board_square_size / 2,
            )
        # center of square
        return (
            x * self.robot.board_square_size + self.robot.board_square_size / 2,
            y * self.robot.board_square_size + self.robot.board_square_size / 2,
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
    default="best_cm.pt",
    help="Weight of the training model",
)
args = parser.parse_args()
image = args.image
board_weight = args.board_weight
chess_model_weight = args.chess_model_weight


chess = Kingslayer(board_weight, chess_model_weight)
# best_move = chess.process_from_image(image)

while True:
    with open("status.json", "r") as f:
        try:
            status = json.load(f)
            if status["status"] == "starteddddddddddddddddddddd":
                try:
                    best_move = chess.process_from_image(image)
                except Exception as e:
                    print("Error:", e)
                status["status"] = "stopped"
                with open("status.json", "w") as f:
                    json.dump(status, f)
            else:
                time.sleep(0.1)
            f.close()
        except Exception as e:
            chess.init_chess_engine()
    # check if the 'q' key is pressed
    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("."):
        # run 'python change_to_started.py' shell command
        os.system("python change_to_started.py")
    if cv2.waitKey(1) == ord("n"):
        best_move = chess.process_from_image(image)


# Save the image with the line
