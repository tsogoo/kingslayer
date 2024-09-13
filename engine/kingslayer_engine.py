import chess
import chess.pgn
import random
import sqlite3


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

    def get_evaluation(self, board):
        if board.is_checkmate():
            if board.turn:
                return -float('inf')  # Black wins
            else:
                return float('inf')  # White wins
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return 0  # Draw

        eval = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_value = self.piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    eval += piece_value + self.pst[piece.piece_type][square]
                else:
                    eval -= piece_value + self.pst[piece.piece_type][chess.square_mirror(square)]
        return eval


class KingslayerChessEngine:
    def __init__(self):
        # Simple opening book with common openings
        self.engine_db = "chess_engine.db"
        self.evaluator = ChessEvaluator()

    def get_opening_move(self, fen):
        print("finding opening move...")
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

    # Minimax with Alpha-Beta Pruning
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.evaluator.get_evaluation(board)

        legal_moves = list(board.legal_moves)

        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
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
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def get_early_middlegame_move(self, board, depth=4):
        print("getting early middlegame move")
        best_move = None
        best_value = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            board_value = self.minimax(board, depth - 1, float('-inf'), float('inf'), not board.turn)
            board.pop()

            if board.turn == chess.WHITE:
                if board_value > best_value:
                    best_value = board_value
                    best_move = move
            else:
                if board_value < best_value:
                    best_value = board_value
                    best_move = move

        return best_move

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
            return middle_move.uci()
        # Step 3: If no opening or early middlegame move, fall back to random (for now)
        return self.get_random_move(board)

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