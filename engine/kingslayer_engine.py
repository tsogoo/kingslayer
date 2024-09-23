import chess
import chess.pgn
import random
import sqlite3
import multiprocessing

from chess_evaluator import ChessEvaluator
from train_eval_position import KingslayerEvaluator

stockengine = None
evaluator = None


def handler_result(future):
    print("handler result")
    print(future.result())


def initialize():
    global stockengine
    if stockengine is None:         
        # print("initializing stockengine")
        stockengine = chess.engine.SimpleEngine.popen_uci("stockfish/stockfish-ubuntu-x86-64")
        stockengine.configure({"Skill Level": 1})
        # print("initialized stockengine")


def move_priority(board, move):
    # Capture moves have higher priority
    if board.is_capture(move):
        return 10
    # Check moves have higher priority
    if board.gives_check(move):
        return 5
    # Normal moves
    return 1


def worker(fen, move, depth, evaluator):
    global stockengine
    if stockengine is None:
        initialize()
    evaluator.set_engine(stockengine)
    board = chess.Board(fen)
    board.push(move)
    eval = minimax(board, depth - 1, float('-inf'), float('inf'), False, evaluator)
    return move, eval


def minimax(board, depth, alpha, beta, maximizing_player, evaluator):
    if depth == 0 or board.is_game_over():
        evaluation = evaluator.get_evaluation(board)
        return evaluation
    legal_moves = sorted(
        board.legal_moves, key=lambda move: move_priority(board, move))
    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            
            eval = minimax(board, depth - 1, alpha, beta, False, evaluator)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, evaluator)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


class KingslayerChessEngine:
    def __init__(self):
        # Simple opening book with common openings
        self.engine_db = "chess_engine.db"
        self.evaluator = ChessEvaluator()
        self.evaluator = KingslayerEvaluator()
        self.prune_count = 0

    def get_opening_move(self, fen):
        print("finding opening move...")
        return None
        # get move from chess_engine
        conn = sqlite3.connect(self.engine_db)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT move FROM opening_book WHERE fen LIKE ?
        ''', (f"{fen}%",))
        result = cursor.fetchmany()
        conn.close()

        if result:
            print(f"opening {len(result)} choices")
            # get random element of array
            return random.choice(result)[0].strip()
        return None

    def move_priority(self, board, move):
        # Capture moves have higher priority
        if board.is_capture(move):
            return 10
        # Check moves have higher priority
        if board.gives_check(move):
            return 5
        # Normal moves
        return 1

    def parallel_minimax(self, board, depth):
        legal_moves = sorted(
            board.legal_moves, key=lambda move: move_priority(board, move))
        move_evals = {}

        # Use ProcessPoolExecutor to parallelize the first depth's move evaluations with 4 threads
        evaluator = self.evaluator
        fen = board.fen()
        with multiprocessing.Pool(initializer=initialize) as pool:
            results = [pool.apply_async(
                worker, (fen, move, depth, evaluator)) for move in legal_moves]
            for result in results:
                move, eval = result.get()
                move_evals[move] = eval
        best_move = max(move_evals, key=move_evals.get)
        return best_move, move_evals[best_move]

    def non_parallel_minimax(self, board, depth):
        legal_moves = sorted(
            board.legal_moves, key=lambda move: move_priority(board, move))
        move_evals = {}
        for move in legal_moves:
            board.push(move)
            move_evals[move] = minimax(board, depth - 1, float('-inf'), float('inf'), False, self.evaluator)
            board.pop()
        best_move = max(move_evals, key=move_evals.get)
        return best_move, move_evals[best_move] 

    def get_random_move(self, board):
        print("getting random legal move")
        if board.legal_moves:
            return random.choice(list(board.legal_moves)).uci()
        else:
            return None

    def get_best_move(self, fen):
        # Step 1: Check opening book
        opening_move = self.get_opening_move(" ".join(fen.split()[:-2]))
        if opening_move:
            return opening_move

        # Step 2: Use early middlegame strategy
        board = chess.Board(fen)
        middle_move = self.get_early_middlegame_move(board)
        if middle_move:
            return middle_move
        # Step 3: If no opening or early middlegame move, fall back to random (for now)
        return self.get_random_move(board)

    def get_early_middlegame_move(self, board, depth=2):
        print("getting early middlegame move", chess.WHITE)
        # best_move, best_eval = self.parallel_minimax(board, depth=depth)
        best_move, best_eval = self.non_parallel_minimax(board, depth=depth)
        print("ended early middlegame move")
        return best_move.uci() if best_move else None

    def make_move(self, fen, uci_move):
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        board.push(move)
        return board.fen()

# Example usage
# engine = KingslayerChessEngine()
# fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'  # Initial position
# best_move = engine.get_best_move(fen)
# print(f"Best move for FEN '{fen}': {best_move}")