import chess
import chess.pgn
import random
import sqlite3
import concurrent.futures
import multiprocessing

stockengine = None

def handler_result(future):
    print("handler result")
    print(future.result())


def initialize_stockengine():
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
        initialize_stockengine()
    evaluator.set_engine(stockengine)
    board = chess.Board(fen)
    board.push(move)
    fen = board.fen()
    eval = minimax(fen, depth - 1, float('-inf'), float('inf'), False, evaluator)
    return move, eval


def minimax(fen, depth, alpha, beta, maximizing_player, evaluator):
    board = chess.Board(fen)
    if depth == 0 or board.is_game_over():
        evaluation = evaluator.get_evaluation(board)
        return evaluation
    legal_moves = sorted(
        board.legal_moves, key=lambda move: move_priority(board, move))
    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            
            eval = minimax(board.fen(), depth - 1, alpha, beta, False, evaluator)
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
            eval = minimax(board.fen(), depth - 1, alpha, beta, True, evaluator)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


class ChessEvaluator:
    def __init__(self):
        # Piece values
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

        # Piece-square tables
        self.pst = {
            chess.PAWN: [
                0, 0, 0, 0, 0, 0, 0, 0,
                50, 50, 50, 50, 50, 50, 50, 50,
                10, 10, 20, 30, 30, 20, 10, 10,
                5, 5, 10, 25, 25, 10, 5, 5,
                0, 0, 0, 20, 20, 0, 0, 0,
                5, -5, -10, 0, 0, -10, -5, 5,
                5, 10, 10, -20, -20, 10, 10, 5,
                0, 0, 0, 0, 0, 0, 0, 0
            ],
            chess.KNIGHT: [
                -50, -40, -30, -30, -30, -30, -40, -50,
                -40, -20, 0, 0, 0, 0, -20, -40,
                -30, 0, 10, 15, 15, 10, 0, -30,
                -30, 5, 15, 20, 20, 15, 5, -30,
                -30, 0, 15, 20, 20, 15, 0, -30,
                -30, 5, 10, 15, 15, 10, 5, -30,
                -40, -20, 0, 5, 5, 0, -20, -40,
                -50, -40, -30, -30, -30, -30, -40, -50
            ],
            chess.BISHOP: [
                -20, -10, -10, -10, -10, -10, -10, -20,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -10, 0, 5, 10, 10, 5, 0, -10,
                -10, 5, 5, 10, 10, 5, 5, -10,
                -10, 0, 10, 10, 10, 10, 0, -10,
                -10, 10, 10, 10, 10, 10, 10, -10,
                -10, 5, 0, 0, 0, 0, 5, -10,
                -20, -10, -10, -10, -10, -10, -10, -20
            ],
            chess.ROOK: [
                0, 0, 0, 0, 0, 0, 0, 0,
                5, 10, 10, 10, 10, 10, 10, 5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                -5, 0, 0, 0, 0, 0, 0, -5,
                0, 0, 0, 5, 5, 0, 0, 0
            ],
            chess.QUEEN: [
                -20, -10, -10, -5, -5, -10, -10, -20,
                -10, 0, 0, 0, 0, 0, 0, -10,
                -10, 0, 5, 5, 5, 5, 0, -10,
                -5, 0, 5, 5, 5, 5, 0, -5,
                0, 0, 5, 5, 5, 5, 0, -5,
                -10, 5, 5, 5, 5, 5, 0, -10,
                -10, 0, 5, 0, 0, 0, 0, -10,
                -20, -10, -10, -5, -5, -10, -10, -20
            ],
            chess.KING: [
                20, 30, 10, 0, 0, 10, 30, 20,
                20, 20, 0, 0, 0, 0, 20, 20,
                -10, -20, -20, -20, -20, -20, -20, -10,
                -20, -30, -30, -40, -40, -30, -30, -20,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30
            ]
        }

    def get_king_safety(self, board, color):
        king_square = board.king(color)
        if king_square is None:  # King may not be on board (game over)
            return 0

        # Get the enemy's color
        enemy_color = not color
        safety_score = 0

        # Define the squares around the king
        surrounding_squares = [king_square + offset for offset in [-9, -8, -7, -1, 1, 7, 8, 9] if chess.SQUARES[0] <= king_square + offset <= chess.SQUARES[-1]]

        # Check if enemy pieces attack those squares
        for square in surrounding_squares:
            if board.is_attacked_by(enemy_color, square):
                safety_score -= 20  # Penalize if enemy controls squares near king

        return safety_score

    def get_mobility(self, board, color):
        mobility = 0
        for move in board.legal_moves:
            if board.color_at(move.from_square) == color:
                mobility += 1
        return mobility

    def set_engine(self, engine):
        self.engine = engine

    def get_evaluation(self, board):
        if board.is_checkmate():
            if board.turn:
                return -float('inf')  # Black wins
            else:
                return float('inf')  # White wins
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0  # Draw

        eval = 0
        if self.engine:
            info = self.engine.analyse(board, chess.engine.Limit(depth=1))
            eval = info["score"].relative.score(mate_score=10000)
            # print("info score: ", info['depth'], info["score"].relative.score(mate_score=10000))
            if info["score"].turn == chess.BLACK:
                eval = -eval
            return eval
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    eval += piece_value + self.pst[piece.piece_type][square]
                else:
                    eval -= piece_value + self.pst[piece.piece_type][chess.square_mirror(square)]
        
        white_mobility = self.get_mobility(board, chess.WHITE)
        black_mobility = self.get_mobility(board, chess.BLACK)
        eval += (white_mobility - black_mobility) * 10  # Weight mobility by 10 (can be tuned)

        # Add king safety factor
        white_king_safety = self.get_king_safety(board, chess.WHITE)
        black_king_safety = self.get_king_safety(board, chess.BLACK)
        eval += (white_king_safety - black_king_safety)  # Adjust the weight if necessary
        
        return eval


class KingslayerChessEngine:
    def __init__(self):
        # Simple opening book with common openings
        self.engine_db = "chess_engine.db"
        self.evaluator = ChessEvaluator()
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
        with multiprocessing.Pool(initializer=initialize_stockengine) as pool:
            results = [pool.apply_async(
                worker, (fen, move, depth, evaluator)) for move in legal_moves]
            for result in results:
                move, eval = result.get()
                move_evals[move] = eval
        best_move = max(move_evals, key=move_evals.get)
        return best_move, move_evals[best_move]

    # Minimax with Alpha-Beta Pruning
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            evaluation = self.evaluator.get_evaluation(board, self.engine)
            return evaluation
        legal_moves = sorted(
            board.legal_moves, key=lambda move: self.move_priority(
                board, move))
        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(
                    board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    self.prune_count += 1
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(
                    board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    self.prune_count += 1
                    break
            return min_eval

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

    def get_early_middlegame_move(self, board, depth=3):
        print("getting early middlegame move", chess.WHITE)
        best_move, best_eval = self.parallel_minimax(board, depth=depth)
        print("ended early middlegame move")
        return best_move.uci() if best_move else None
        best_move = None
        best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in board.legal_moves:
            board.push(move)
            # board_value = self.minimax(board, depth, alpha, beta, board.turn)
            board_value = self.minimax(
                board, depth, alpha, beta, board.turn)
            board.pop()

            if board.turn == chess.WHITE and board_value > best_value:
                best_value = board_value
                best_move = move
            elif board.turn == chess.BLACK and board_value < best_value:
                best_value = board_value
                best_move = move

        print(f"Pruning occurred {self.prune_count} times")
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