import chess
import numpy as np

# Forsyth–Edwards Notation (FEN) is a standard notation for describing a particular board position of a chess game
# https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
# Example 8x8 matrix representing a chess position
# chess_board = [
#     ["r", "n", "b", "q", "k", "b", "n", "r"],
#     ["p", "p", "p", "p", "p", "p", "p", "p"],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     [" ", " ", " ", " ", " ", " ", " ", " "],
#     ["P", "P", "P", "P", "P", "P", "P", "P"],
#     ["R", "N", "B", "Q", "K", "B", "N", "R"]
# ]
def matrix_to_fen(board, turn=None):
    """
    Convert an 8x8 matrix representation of a chess position to FEN notation.

    Parameters:
        board (list of lists): 8x8 matrix representing the chess position.
                                Each element should be a string representing
                                a piece on the board or an empty space.

    Returns:
        str: FEN notation of the board position.
    """
    fen = ""
    empty_counter = 0

    for row in board:
        for square in row:
            if square == " ":
                empty_counter += 1
            else:
                if empty_counter > 0:
                    fen += str(empty_counter)
                    empty_counter = 0
                fen += square
        if empty_counter > 0:
            fen += str(empty_counter)
            empty_counter = 0
        fen += "/"

    # Remove the last '/'
    fen = fen[:-1]

    fen += get_turn(turn)

    return fen

figure_map = {
    "Pw0":"p",
    "Bs0":"b",
    "Kg0":"k",
    "Qn0":"q",
    "Rk0":"r",
    "Kt0":"n",
    "Pw1":"P",
    "Bs1":"B",
    "Kg1":"K",
    "Qn1":"Q",
    "Rk1":"R",
    "Kt1":"N",
    " ": " "}
def fen_figure(name):
    return figure_map[name]

def to_matrix(squares):
    return np.array(list(map(fen_figure, squares))).reshape(8, 8)

def get_turn(turn):
    return "" if turn is None else " w" if turn == chess.WHITE else " b"