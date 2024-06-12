from engine.helper import ChessEngineHelper
import chess
from engine.detect import Detector

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

if __name__ == "__main__":
    main()

# example
# class Test:

#     def __init__(self):
        
#         import os
#         import yaml

#         current_file_path = os.path.abspath(__file__)
#         parent_directory = os.path.dirname(current_file_path)
#         with open(os.path.join(parent_directory, 'app.yaml'), 'r') as file:
#             config = yaml.safe_load(file)

#         from engine.helper import ChessEngineHelper
#         from robot_arm.robot import Robot
#         self.e = ChessEngineHelper()
#         self.robot = Robot(config=config)

#     def get_figure_actual_position(self, x, y, is_occupied=False):
#         # new board square height, width = 370/8 = 46.25
#         # x, y => figure's square coordinate on board, is_occupied => whether square is occupied
#         if is_occupied:
#             # TODO get detected location
#             return x*46+23, y*46+23
#         # center of square
#         return x*46+23, y*46+23
    
#     def test(self):
#         e = self.e
#         e.initialize_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") # new game
#         # e.initialize_board("4k3/P7/8/8/8/8/8/4K3 w") # promotion
#         # e.initialize_board("4k3/8/8/8/8/8/p7/4K3 b") # promotion
#         # e.initialize_board("4k2r/p7/8/8/8/8/8/4K3 b KQkq") # castling
#         # e.initialize_board("r3k3/8/8/8/8/8/8/4K3 b KQkq") # castling
#         while not e.board.is_game_over():
#             # print(e.board)
#             m = e.get_best_move()
#             if e.is_valid_move(m):
#                 self.robot.move(e, self, m)
#                 e.move(m)
#                 p = input("type_move/e7e3/:")
#                 if e.is_valid_move(p):
#                     e.move(p)
#                 print(e.board)
# t = Test()
# t.test()