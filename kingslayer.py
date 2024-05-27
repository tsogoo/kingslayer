import cv2
from ultralytics import YOLO
import argparse
import numpy as np
import math
from engine.helper import ChessEngineHelper
import chess
import json
import time

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
        self.margin = 40

        # initialize models yolo detector
        self.board_model = YOLO(board_weight)
        self.chess_model = YOLO(chess_model_weight)

        # initialize chess engine helper
        self.chess_engine_helper = ChessEngineHelper()
        self.chess_engine_helper.initialize_board(None)

    def init_perspective_data(self, xoffset, yoffset, w, h, xend, yend, img):
        # crop image with x,y,w,h
        color = img[
            int(yoffset - self.margin) : int(yend + self.margin),
            int(xoffset - self.margin) : int(xend + self.margin),
        ]

        # gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        gray = cv2.adaptiveThreshold(
            color, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5
        )
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = find_max_contour_area(contours)
        gray = cv2.drawContours(gray, contours, -1, (0, 255, 0), 2).copy()

        c = contours[0]
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
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

        # cv2.rectangle(img, (xstart, ystart), (xend, yend), (0, 255, 255), 1)
        cv2.rectangle(
            color, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 255), 1
        )

        cv2.imwrite("im.jpg", color)
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

        board_results = self.board_model.predict(image, save=False, imgsz=640, conf=0.2)
        chess_results = self.chess_model.predict(image, save=False, imgsz=864, conf=0.3)
        # Load the image using OpenCV
        img = cv2.imread(image)
        img = get_blurry_image(img)
        for result in board_results:
            x, y, w, h = result.boxes.xywh[0]
            self.x, self.y, self.xend, self.yend = result.boxes.xyxy[0]
            # img = cv2.rectangle(
            #     img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 1
            # )
            self.init_perspective_data(self.x, self.y, w, h, self.xend, self.yend, img)

        self.models = []
        for result in chess_results:

            # draw each box in the image and its name

            for i in range(len(result.boxes.xywh)):
                x, y, w, h = result.boxes.xywh[i]
                x, y, xend, yend = result.boxes.xyxy[i]

                img = cv2.rectangle(
                    img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 1
                )

                y = int(
                    yend - (xend - x) / 2 - self.y + self.margin
                )  # x position of model in image
                x = int(
                    (xend + x) / 2 - self.x + self.margin
                )  # y position of moedel in image

                self.models.append(
                    (
                        result.names[int(result.boxes.cls[i])],
                        self.process_warp([(x, y)])[0],
                        result.boxes.conf[i],
                    )
                )
                # print(
                #     result.names[int(result.boxes.cls[i])], (x, y), result.boxes.conf[i]
                # )

        print(x, y, w, h)

        conf = self.generate_chess_board_array()
        self.print_chess_board_array(conf)
        conf = [
            ["r", "n", "b", "q", "k", "b", "n", "r"],
            ["p", "p", "p", "p", " ", "p", "p", "p"],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", "p", " ", " ", " "],
            [" ", " ", " ", " ", "P", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            ["P", "P", "P", "P", " ", "P", "P", "P"],
            ["R", "N", "B", "Q", "K", "B", "N", "R"],
        ]
        best_move = None
        best_move = self.get_movement(conf)

        # Continue with your existing code...
        M = cv2.getPerspectiveTransform(self.pts_perspective, self.pts_square)
        img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

        cv2.imwrite("image_with_line.jpg", img)
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
        # return self.chess_engine_helper.get_position(
        #     self.chess_engine_helper.get_best_move()
        # )

    def generate_chess_board_array(self):
        chess_board_array = []

        for i in range(8):
            board_row = []
            for j in range(8):
                node_value = " "
                value = (
                    self.x + (self.xend - self.x) / 8 * j,
                    self.yend - self.xend + self.x + (self.xend - self.x) / 8 * i,
                    self.x + (self.xend - self.x) / 8 * (j + 1),
                    self.yend - self.xend + self.x + (self.xend - self.x) / 8 * (i + 1),
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
            # TODO get detected location
            return x*46+23, y*46+23
        # center of square
        return x*46+23, y*46+23


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


while True:
    with open("status.json", "r") as f:
        status = json.load(f)
        if status["status"] == "starteddddddddddddddddddddd":
            try:
                best_move = chess.process_from_image(image)
                print(best_move)
            except Exception as e:
                print("Error:", e)
            status["status"] = "stopped"
            with open("status.json", "w") as f:
                json.dump(status, f)

        else:
            time.sleep(1)
        f.close()
    if cv2.waitKey(1) == ord("q"):
        break


# Save the image with the line
