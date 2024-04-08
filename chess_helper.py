from engine.helper import ChessEngineHelper
import chess

# example
helper = ChessEngineHelper()

# starting board
# helper.initialize_board(None)

# already played board, Upper letter = white
conf = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", "P", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    ["P", "P", "P", "P", " ", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"]
]
helper.initialize_board(conf=conf, turn=chess.BLACK)
# helper.move(helper.get_best_move())
# helper.move("e7e5")
# print(helper.get_position("e7e5"))

# detect and move
# print(helper.is_valid_move("e7e5"))
conf = [
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", " ", "p", "p", "p"],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    [" ", " ", " ", " ", "p", " ", " ", " "],
    [" ", " ", " ", " ", "P", " ", " ", " "],
    [" ", " ", " ", " ", " ", " ", " ", " "],
    ["P", "P", "P", "P", " ", "P", "P", "P"],
    ["R", "N", "B", "Q", "K", "B", "N", "R"]
]
helper.detect_move(conf=conf)

helper.destroy()