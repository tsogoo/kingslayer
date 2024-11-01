import cv2
import os
import yaml
import requests
import numpy as np
import math
from ultralytics import YOLO
from engine.helper import ChessEngineHelper
from robot_arm.cartesian_robot import Robot
from common.config import get_config

from lib_contour import (
    get_gray_image,
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
        self.is_white = True
        self.fen_kq = "KQkq"
        self.models = []
        self.pts_square = None
        self.pts_perspective = None
        self.pts = None
        self.margin = 150
        self.CONFIDENCE_THRESHOLD = 0.7
        self.CROP_SIZE = 640
        self.detected_board_data = None
        self.light_contour_number = 86
        self.image_url = None
        self.automove = False
        self.previous_fen = None
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
        return img

    def auto_contast(self, image):
        print("===auto_contrast===")
        min_val = np.min(image)
        max_val = np.max(image)

        # Apply the contrast stretching
        autocontrast = (image - min_val) * (255 / (max_val - min_val))
        # Convert to uint8
        return np.uint8(autocontrast)

    def converted_robot_point(self, xy):
        board_margin_size = self.robot.board_margin_size
        w1 = self.robot.board_square_size * 8 + 2 * board_margin_size
        w2 = self.robot.board_square_size * 8 + 2 * board_margin_size
        scale = 1
        x = int(xy[0] * w1 / self.CROP_SIZE - self.robot.board_margin_size)
        y = int(xy[1] * scale * w1 / self.CROP_SIZE - self.robot.board_margin_size)
        return (int(x), int(y))

    def detect_models(self, image):
        frame_path = image
        conf = self.CONFIDENCE_THRESHOLD
        for j in range(1):
            if j < 3:
                conf = 0.7
            # elif j < 3:
            #     conf = 0.6

            print("=============")
            print("Confidence:", conf)
            print(frame_path)
            chess_results = self.chess_model.predict(
                self.augment_image(frame_path),
                save=False,
                imgsz=self.CROP_SIZE,
                conf=conf,
            )
            img = cv2.imread(frame_path)
            frame_path = f"augmented{j}.jpg"

            # if len(chess_results) == 0 and conf <= self.CONFIDENCE_THRESHOLD:
            #     return
            print("=============detected models")
            for result in chess_results:
                # if len(result.boxes.xywh) == 0 and conf <= self.CONFIDENCE_THRESHOLD:
                #     return
                for i in range(len(result.boxes.xywh)):
                    x, y, w, h = result.boxes.xywh[i]
                    x, y, xend, yend = result.boxes.xyxy[i]
                    dx = 0
                    dy = h / 7

                    # fill detected model with white
                    # img = cv2.rectangle(
                    #     img,
                    #     (int(x + dx), int(y + dy)),
                    #     (int(x + w), int(y + h)),
                    #     (200, 200, 200),
                    #     -1,
                    # )
                    # label
                    img = cv2.putText(
                        img,
                        result.names[int(result.boxes.cls[i])],
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                    # put confidence text on image
                    img = cv2.putText(
                        img,
                        str(round(result.boxes.conf[i].item(), 2)),
                        (int(x), int(y + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                    )
                    img = cv2.rectangle(
                        img,
                        (int(x), int(y)),
                        (int(x + w), int(y + h)),
                        (0, 255, 255),
                        1,
                    )

                    # getting center of the detected model
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
        cv2.imwrite("detected_models.jpg", img)
        return img

    def print_detected_board(self):
        # set blank image
        board_square_size = self.robot.board_square_size
        # board_square_size = 80
        img = np.zeros(
            (int(board_square_size * 8), int(8 * board_square_size), 3),
            np.uint8,
        )
        # draw board
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    cv2.rectangle(
                        img,
                        (
                            j * board_square_size,
                            i * board_square_size,
                        ),
                        (
                            (j + 1) * board_square_size,
                            (i + 1) * board_square_size,
                        ),
                        (255, 255, 255),
                        -1,
                    )
                else:
                    cv2.rectangle(
                        img,
                        (
                            j * board_square_size,
                            i * board_square_size,
                        ),
                        (
                            (j + 1) * board_square_size,
                            (i + 1) * board_square_size,
                        ),
                        (0, 0, 0),
                        -1,
                    )

        # point models on the image
        for model in self.models:
            cv2.circle(
                img,
                (int(model[1][0]), int(model[1][1])),
                3,
                (255, 0, 255),
                -1,
            )

        # save image with name board_file.jpg
        cv2.imwrite("board_file.jpg", img)

    def init_perspective_data(self, xoffset, yoffset, xend, yend, w, h, img):
        # crop image with x,y,w,h
        color = img[
            int(yoffset - self.margin) : int(yend + self.margin),
            int(xoffset - self.margin) : int(xend + self.margin),
        ]
        color = self.auto_contast(color)
        cv2.imwrite("im.jpg", color)

        # gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        # gray = cv2.adaptiveThreshold(
        #     color, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 9, 5
        # )
        reverse = cv2.bitwise_not(color)
        # for dark cv2.threshold(reverse, 120, 255, 0)
        # for light cv2.threshold(reverse, 60, 255, 0)
        print("=====writing reverse image")
        print(self.light_contour_number)
        ret, gray = cv2.threshold(reverse, self.light_contour_number, 255, 0)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = find_max_contour_area(contours)
        # gray = cv2.drawContours(gray, contours, -1, (0, 255, 0), 2).copy()

        # contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = find_max_contour_area(contours)
        print("=====writing gray image")

        cv2.imwrite("gray.jpg", gray)

        if not np.any(self.pts):
            c = contours[0]
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            self.pts = find_outer_corners(gray, approx)

        min_x = 10000
        min_y = 10000
        max_x = 0
        max_y = 0

        for p in self.pts:
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

        yend = int((self.pts[0][1] + self.pts[1][1]) / 2 + yoffset - self.margin)
        xend = int(self.pts[1][0] + xoffset - self.margin)
        ystart = int((self.pts[2][1] + self.pts[3][1]) / 2 + yoffset - self.margin)
        xstart = int(self.pts[0][0] + xoffset - self.margin)
        h = yend - ystart
        w = xend - xstart
        print("======writing im2.jpg")
        cv2.imwrite("im2.jpg", img)
        # cv2.rectangle(img, (xstart, ystart), (xend, yend), (0, 255, 255), 1)
        # cv2.rectangle(
        #     color, (int(min_x), int(min_y)), (int(max_x), int(max_y)), (0, 255, 255), 1
        # )
        self.pts_square = np.float32(
            [[xend - w, yend - w], [xend - w, yend], [xend, yend - w], [xend, yend]]
        )

        self.pts_perspective = np.float32(
            [
                [self.pts[2][0], self.pts[2][1]],
                [self.pts[0][0], self.pts[0][1]],
                [self.pts[3][0], self.pts[3][1]],
                [self.pts[1][0], self.pts[1][1]],
            ]
        )
        return color

    def process_warp(self, input_coordinates):
        M = cv2.getPerspectiveTransform(self.pts_perspective, self.pts_square)
        output_coordinates = cv2.perspectiveTransform(
            np.array([input_coordinates], dtype="float32"), M
        )
        return output_coordinates[0]

    def get_board_corners(self, image):

        img = cv2.imread(image)
        gray = get_gray_image(img)

        if np.any(self.pts):
            cropped_image = self.init_perspective_data(
                self.detected_board_data[0],
                self.detected_board_data[1],
                self.detected_board_data[2],
                self.detected_board_data[3],
                self.detected_board_data[4],
                self.detected_board_data[5],
                gray,
            )
            print("======detected board data")
            return cropped_image

        print("======detecting board")
        cropped_image = None
        board_results = self.board_model.predict(
            img, save=False, imgsz=self.CROP_SIZE, conf=0.7
        )
        for result in board_results:
            x, y, w, h = result.boxes.xywh[0]
            self.x, self.y, self.xend, self.yend = result.boxes.xyxy[0]
            # img = cv2.rectangle(
            #     img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 1
            # )
            if w < 300 and h < 300:
                return None
            self.detected_board_data = (self.x, self.y, self.xend, self.yend, w, h)
            cropped_image = self.init_perspective_data(
                self.x, self.y, self.xend, self.yend, w, h, gray
            )
            return cropped_image

    def init_camera(self, image, light_contour_number):
        self.light_contour_number = light_contour_number
        self.detected_board_data = None
        self.pts = None
        return self.get_board_corners(image)

    def trigger(self, image=None):
        if self.image_url:
            img_data = requests.get(self.image_url).content
            with open("frame.jpg", "wb") as handler:
                handler.write(img_data)
                handler.close()
            return self.auto_process("frame.jpg")
        # else:
        #     return self.process_from_image(image)

    def auto_process(self, image):
        cropped_image = self.get_board_corners(image)

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
        img = self.detect_models(warped_image_path)
        conf = self.generate_chess_board_array()
        conf = list(map(list, zip(*conf[::-1])))
        self.print_chess_board_array(conf)
        best_move = None
        best_move = self.get_movement(conf)
        if best_move:
            # Continue with your existing code...
            img = cv2.putText(
                img,
                best_move,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imwrite("board_result.jpg", img)
            response = self.robot.move(
                self.chess_engine_helper,
                self,
                best_move,
                self.chess_engine_helper.board.turn if self.is_white else not self.chess_engine_helper.board.turn,
            )
            # if not response:
            #     self.previous_fen = self.buffered_previous_fen
            print(self.robot.commands)
            return best_move
        cv2.imwrite("board_result.jpg", img)

    def process_from_image(self, image):

        # Load the image using OpenCV
        cropped_image = self.get_board_corners(image)

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
        img = self.detect_models(warped_image_path)
        conf = self.generate_chess_board_array()
        conf = list(map(list, zip(*conf[::-1])))
        self.print_chess_board_array(conf)
        # rotate conf 180 degree
        # if self.is_white:
        #     conf = list(map(list, zip(*conf[::-1])))
        #     conf = list(map(list, zip(*conf[::-1])))
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
        img = cv2.putText(
            img,
            best_move,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imwrite("board_result.jpg", img)
        print(best_move)
        self.robot.move(
            self.chess_engine_helper,
            self,
            best_move,
            self.chess_engine_helper.board.turn,
        )
        print(self.robot.commands)
        return best_move

    def produce_kq(self, fen, side, fen_kq):        
        print("Producing Fen:")
        print(fen)
        if not self.chess_engine_helper.is_white_king_side_castling_possible(f"{fen} {side} {fen_kq}"):
            fen_kq = fen_kq.replace("K", "")
        if not self.chess_engine_helper.is_white_queen_side_castling_possible(f"{fen} {side} {fen_kq}"):
            fen_kq = fen_kq.replace("Q", "")
        if not self.chess_engine_helper.is_black_king_side_castling_possible(f"{fen} {side} {fen_kq}"):
            fen_kq = fen_kq.replace("k", "")
        if not self.chess_engine_helper.is_black_queen_side_castling_possible(f"{fen} {side} {fen_kq}"):
            fen_kq = fen_kq.replace("q", "")
        if fen_kq == "":
            return "-"
        return fen_kq
    
    def get_movement(self, conf):
        print("Get Movement starting ...")
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
        if self.is_white:
            # reverse array and string
            fen_rows = fen_rows[::-1]
            for i in range(len(fen_rows)):
                fen_rows[i] = fen_rows[i][::-1]

        if self.is_white:
            who = "w"
            oppose = "b"
        else:
            who = "b"
            oppose = "w"

        fen = "/".join(fen_rows)
        if not self.chess_engine_helper.is_valid_fen(f"{fen} {who} {self.fen_kq}"):
            return None
        print("==========================Fen")
        print(fen)
        last_move = None
        if fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR":
            self.fen_kq = "KQkq"
            if not self.is_white:
                print("I'm Black and waiting White move...")
                self.previous_fen = fen
                return None
            else:
                print("I'm White and I'm Moving...")
                self.previous_fen = None
                last_move = True
        elif fen == "RNBKQBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbkqbnr":
            self.is_white = True
            self.previous_fen = None
            self.fen_kq = "KQkq"
            return
        if self.previous_fen:
            print("Previous fen", self.previous_fen)
            print("Current fen", fen)
            if self.previous_fen != fen:
                camera_fen_kq = self.produce_kq(fen, who, self.fen_kq)
                last_move = self.chess_engine_helper.get_last_move(
                    f"{self.previous_fen} {oppose} {self.fen_kq}", f"{fen} {who} {camera_fen_kq}"
                )
                if not last_move:
                    print("last move not detected")
                    print(self.previous_fen)
        
        if last_move or not self.previous_fen:
            print("last move", last_move)
            print("previous fen", self.previous_fen)
            self.fen_kq = self.produce_kq(fen, who, self.fen_kq)
            self.chess_engine_helper.initialize_board(fen + f" {who} {self.fen_kq}")
            try:
                best_move = self.chess_engine_helper.get_best_move()
            except Exception:
                self.init_chess_engine()
                return None
            print("===========Fen before Best Move", f"{fen} {who} {self.fen_kq} - 0 1")
            best_fen = self.chess_engine_helper.apply_uci_move_to_fen(
                f"{fen} {who} {self.fen_kq} - 0 1", best_move
            )
            print("===========Best Fen", best_fen)
            print(best_move)
            # self.buffered_previous_fen = self.previous_fen
            self.previous_fen = best_fen.split(" ")[0]
            self.fen_kq = self.produce_kq(self.previous_fen, who, self.fen_kq)
            print("Previous fen saved", self.previous_fen)
            return best_move
        return None

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
        print(self.models)
        print(len(self.models))
        print(chess_board_array)
        return chess_board_array

    def print_chess_board_array(self, chess_board_array):
        print("============== Chess board ==============")
        for row in chess_board_array:
            print(row)

    def get_figure_actual_position(self, x, y, is_occupied=False):

        # new board square height, width = 370/8 = 46.25
        # x, y => figure's square coordinate on board, is_occupied => whether square is occupied
        if not self.is_white:
            x = 7 - x
            y = 7 - y
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
                    if (
                        self.robot.board_square_size * 8
                        - model[1][1] // self.robot.board_square_size
                        < self.robot.board_square_size / 5
                    ):  # a1 ruu iluu temuulsen bol gol
                        return (
                            self.robot.board_square_size * 8
                            - model[1][1]
                            + self.robot.board_square_size / 5
                            - self.robot.board_square_size * 8
                            - model[1][1] // self.robot.board_square_size,
                            model[1][0],
                        )
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
