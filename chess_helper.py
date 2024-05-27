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