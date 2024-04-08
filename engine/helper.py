import chess
import chess.engine
import os
import subprocess

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
def matrix_to_fen(board, turn=chess.WHITE):
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

    if turn is not None:
        if turn == chess.WHITE:
            fen = fen + " w"
        else:
            fen = fen + " b"
    else:
        fen = fen + " w"

    return fen


# square to x,y : 52/e7/ -> 6,4
def square_to_position(square):
    return int((square + 1) / 8), (square + 1) % 8 - 1

def get_move(player_move):
    move = None
    try:
        move = chess.Move.from_uci(player_move)
    except Exception:
        print("Invalid move, try again.")
    return move


class ChessEngineHelper:

    # prepare stockfish engine
    def __init__(self):

        # Download engine if not exists
        current_directory = os.path.dirname(os.path.realpath(__file__))
        engine_file = "stockfish/stockfish-ubuntu-x86-64"
        engine_file_absolute_path = os.path.join(current_directory, engine_file)
        if not os.path.exists(engine_file_absolute_path):
            print("Engine not found, downloading....")
            engine_url = "https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64.tar"
            command = "wget -q {} -O - | tar -C {} -xf - {}".format(engine_url, current_directory, engine_file)
            try:
                subprocess.run(command, shell=True, check=True)
                print("Engine downloaded")
            except subprocess.CalledProcessError as e:
                print("Engine download error:", e)

        # Initialize the Stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_file_absolute_path)
        print("Engine started")

    # destroy it when finished using
    def destroy(self):
        self.engine.quit()
        self.engine = None
        self.board = None
        print("Engine stopped")

    # initialize board
    def initialize_board(self, conf, turn=None):
        if conf is not None:
            self.board = chess.Board(matrix_to_fen(conf, turn))
        else:
            self.board = chess.Board()
        print(self.board)

    # get best possible move: -> e7e5
    def get_best_move(self):
        result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
        return str(result.move)

    # whether move is valid: e7e5 -> boolean
    def is_valid_move(self, player_move):
        move = get_move(player_move)
        # print("check_move:", move)
        if move is not None:
            if move in self.board.legal_moves:
                return True
            else:
                print("Illegal move, try again.")
        return False

    # move: e7e5
    def move(self, player_move):
        if self.is_valid_move(player_move):
            self.board.push(get_move(player_move))
            print(self.board)

    # check game is over or stalemate
    def is_game_over(self):
        return self.board.is_game_over() or self.board.is_stalemate()
    

    # get position: e7e5 -> [{x,y}, {x,y}] = [from, to]
    def get_position(self, player_move):
        move = get_move(player_move)
        return [square_to_position(move.from_square), square_to_position(move.to_square)]


    # check square occupied by opponent figure: e7e5
    def is_occupied(self, player_move):
        move = get_move(player_move)
        return self.board.piece_at(move.to_square) is not None
    
    # detect moved figure, then move
    def detect_move(self, conf):
        prev_board = self.board
        
        turn = "w"
        if prev_board.turn == chess.BLACK:
            turn = "b"
        
        # print("turn:", turn)
        new_board = chess.Board(matrix_to_fen(conf, turn))

        to_square = None
        from_square = None
        for square in chess.SQUARES:
            if prev_board.piece_at(square) != new_board.piece_at(square):
                if new_board.piece_at(square) is not None:
                    to_square = square
            if prev_board.piece_at(square) is not None and new_board.piece_at(square) is None:
                from_square = square
        move = chess.Move(from_square, to_square)
        # print(move)
        
        self.move(str(move))
        print(self.board)