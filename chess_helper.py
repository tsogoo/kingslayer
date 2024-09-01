from engine.helper import ChessEngineHelper
import chess
from engine.detect import Detector
import time
import chess.svg

# example
# helper = ChessEngineHelper()

# starting board
# helper.initialize_board(None)

# already played board, Upper letter = white
# conf = [
#     ["r", "n", "b", "q", "k", "b", "n", "r"],
#     ["p", "p", "p", "p", "p", "p", "p", "p"],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     [" ", " ", " ", " ", "P", " ", " ", " "],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     ["P", "P", "P", "P", " ", "P", "P", "P"],
#     ["R", "N", "B", "Q", "K", "B", "N", "R"]
# ]
# helper.initialize_board(conf=conf, turn=chess.BLACK)
# helper.move(helper.get_best_move())
# helper.move("e7e5")
# print(helper.get_position("e7e5"))

# detect and move
# print(helper.is_valid_move("e7e5"))
# conf = [
#     ["r", "n", "b", "q", "k", "b", "n", "r"],
#     ["p", "p", "p", "p", " ", "p", "p", "p"],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     [" ", " ", " ", " ", "p", " ", " ", " "],
#     [" ", " ", " ", " ", "P", " ", " ", " "],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     ["P", "P", "P", "P", " ", "P", "P", "P"],
#     ["R", "N", "B", "Q", "K", "B", "N", "R"]
# ]
# helper.detect_move(conf=conf)

# helper = ChessEngineHelper()
# d = Detector()
# fen = d.detect()
# helper.initialize_board(fen)
# helper.destroy()

# helper.destroy()


def main():
    with chess.engine.SimpleEngine.popen_uci("engine/stockfish/stockfish-ubuntu-x86-64") as engine:  # Replace "stockfish_path" with the actual path to Stockfish executable
        engine.configure({"Skill Level": 1})
        board = chess.Board("4k3/8/8/8/8/8/8/4K2R w KQkq")

        while not board.is_game_over():
            print(board)
            if board.turn == chess.WHITE:
                human_move = input("Enter your move (in algebraic notation): ")
                try:
                    board.push_san(human_move)
                except ValueError:
                    print("Invalid move. Try again.")
                    continue
            else:
                print("Stockfish is thinking...")
                result = engine.play(board, chess.engine.Limit(time=0.1))
                print(result.move)
                board.push(result.move)

        print("Game Over")
        print("Result: ", board.result())

# if __name__ == "__main__":
#     main()

# for testing board detection from image
def test_detect():
    from ultralytics import YOLO
    from engine.detect import detect, detect_from_camera, detect2
    board_pt = YOLO('pt/best_board.pt')
    figure_pt = YOLO('pt/best_cm.pt')
    # for i in range(1, 5, 1):
    #     detect2(board_pt, figure_pt, str(i)+'.jpg')
    #     input('continue')
    detect2(board_pt, figure_pt, 'tmp.jpg', False)
    # detect(board_pt, figure_pt, '1.jpg')
    # detect_from_camera(board_pt, figure_pt, '1.jpg')
        
# for testing board detection from image
def test_detect_from_video(start_idx:int=0, check_idx:int=0):
    # from ultralytics import YOLO
    from engine.detect import detect2
    import cv2
    # board_pt = YOLO('pt/best_board.pt')
    
    cap = cv2.VideoCapture("v.mp4")
    idx = 0
    er = []
    is_debug = True if check_idx > 0 else False
    output = True if start_idx == 0 and check_idx == 0 else False 
    while 1:
        idx = idx + 1
        ret, image = cap.read()
        if ret:
            if is_debug and idx < check_idx:
                continue
            if is_debug and idx > check_idx:
                break
            cv2.imwrite('tmp.jpg', image)
            if idx >= start_idx:
                print('idx:calc:', idx)
                detect2(None, 'tmp.jpg', is_debug= is_debug, idx= idx, output= output)
            else:
                print('idx:', idx)
        else:
            break
    print(er)
    # detect2(board_pt, figure_pt, '1.jpg')
    # detect(board_pt, figure_pt, '1.jpg')
    # detect_from_camera(board_pt, figure_pt, '1.jpg')

# for test gcode is correct, print it in robot.commands_handle
def test_gcode():
    import os
    import yaml
    from robot_arm.robot import Robot, RobotTask, RobotMove
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'app.yaml'
        ), 'r'
    ) as file:
        config = yaml.safe_load(file)
    bot = Robot(config=config)
    # r1 = RobotMove(RobotTask.Take, 90, 90)
    # r2 = RobotMove(RobotTask.Place, 120, 120)
    # bot.move_handle([r1, r2], True)
    bot.task_handle(RobotTask.Buzzer)

def chess_test():

    import paho.mqtt.client as mqtt
    client = mqtt.Client()
    client.connect('localhost')

    def on_custom_event(svg):
        client.publish(
            topic='chess', payload=svg
        )

    from common.event import EventManager
    event_manager = EventManager()
    event_manager.register("custom_event", on_custom_event)

    with chess.engine.SimpleEngine.popen_uci("engine/stockfish/stockfish-ubuntu-x86-64") as engine:  # Replace "stockfish_path" with the actual path to Stockfish executable
        engine.configure({"Skill Level": 20})
        
        # board = chess.Board("4k3/8/8/8/8/8/8/4K2R w KQkq")
        board = chess.Board()

        while not board.is_game_over():

            if board.turn == chess.WHITE:
                human_move = input("Enter your move (e2e4): ")
                try:
                    board.push_san(human_move)
                except ValueError:
                    print("Invalid move. Try again.")
                    continue
            else:
                print("Stockfish is thinking...")
                if board.is_check():
                    print('checked')
                result = engine.play(board, chess.engine.Limit(time=0.1))
                board.push(result.move)
            
            # Generate the SVG
            board_svg = chess.svg.board(board)

            event_manager.trigger("custom_event", board_svg)

def test_create_video():
    from engine.detect import create_video
    create_video()

def test_valid_board():
    from engine.helper import ChessEngineHelper
    engine = ChessEngineHelper()
    engine.initialize_board("8/7K/8/8/8/8/8/8")

def test_load_config():
    from common.config import load_config
    config = load_config()
    print(config)

def test_last_move(is_valid:bool=True):
    with chess.engine.SimpleEngine.popen_uci("engine/stockfish/stockfish-ubuntu-x86-64") as engine:  # Replace "stockfish_path" with the actual path to Stockfish executable
        engine.configure({"Skill Level": 1})
        board = chess.Board("4k3/8/8/8/8/8/8/4K2R w KQkq")
        board_new = chess.Board("4k3/8/8/8/8/8/8/5RK1 b KQkq" if is_valid else "4k3/8/8/8/8/8/8/6KR b KQkq")

        for move in board.legal_moves:
            board.push(move)
            # if (move.uci() == 'e1g1'):
            #     print(board.fen().split()[0], board_new.fen().split()[0])
            if board.fen().split()[0] == board_new.fen().split()[0]:
                return move
            board.pop()
        
        return None

# test_detect_from_video(check_idx=1)
# test_detect()
# chess_test()
# test_create_video()
# test_gcode()
# test_valid_board()
# test_load_config()
print(test_last_move(True))