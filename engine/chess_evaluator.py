import chess

DOUBLED_PAWN_PENALTY = -0.5
ISOLATED_PAWN_PENALTY = -0.75
PASSED_PAWN_BONUS = 1.0
BACKWARD_PAWN_PENALTY = -0.5


class ChessEvaluator:
    def __init__(self):
        # Piece values
        self.engine = None
        self.piece_values = {
            chess.PAWN: 10,
            chess.KNIGHT: 32,
            chess.BISHOP: 33,
            chess.ROOK: 50,
            chess.QUEEN: 90,
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

    def evaluate_pawn_structure(self, board):
        score = 0
        score += self.evaluate_doubled_pawns(board)
        score += self.evaluate_isolated_pawns(board)
        score += self.evaluate_passed_pawns(board)
        score += self.evaluate_backward_pawns(board)
        return score

    # 1. Doubled Pawns Evaluation
    def evaluate_doubled_pawns(self, board):
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            for file in chess.FILE_NAMES:
                pawns_in_file = list(board.pieces(chess.PAWN, color) & chess.BB_FILES[chess.FILE_NAMES.index(file)])
                if len(pawns_in_file) > 1:  # There are doubled pawns
                    if color == chess.WHITE:
                        score += DOUBLED_PAWN_PENALTY
                    else:
                        score -= DOUBLED_PAWN_PENALTY
        return score

    # 2. Isolated Pawns Evaluation
    def evaluate_isolated_pawns(self, board):
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            for square in board.pieces(chess.PAWN, color):
                file_index = chess.square_file(square)
                is_isolated = True
                # Check for pawns on adjacent files
                if file_index > 0 and board.pieces(chess.PAWN, color) & chess.BB_FILES[file_index - 1]:
                    is_isolated = False
                if file_index < 7 and board.pieces(chess.PAWN, color) & chess.BB_FILES[file_index + 1]:
                    is_isolated = False
                if is_isolated:
                    if color == chess.WHITE:
                        score += ISOLATED_PAWN_PENALTY
                    else:
                        score -= ISOLATED_PAWN_PENALTY
        return score

    # 3. Passed Pawns Evaluation
    def evaluate_passed_pawns(self, board):
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            for square in board.pieces(chess.PAWN, color):
                if self.is_passed_pawn(board, square, color):
                    if color == chess.WHITE:
                        score += PASSED_PAWN_BONUS
                    else:
                        score -= PASSED_PAWN_BONUS
        return score

    # Helper function to check if a pawn is passed
    def is_passed_pawn(self, board, square, color):
        file_index = chess.square_file(square)
        rank = chess.square_rank(square)

        # Check for opposing pawns in the same and adjacent files ahead of this pawn
        if color == chess.WHITE:
            for opp_rank in range(rank + 1, 8):
                for adj_file in range(max(0, file_index - 1), min(7, file_index + 1) + 1):
                    if board.piece_at(chess.square(adj_file, opp_rank)) == chess.PAWN and not board.piece_at(chess.square(adj_file, opp_rank)).color == chess.WHITE:
                        return False
        else:
            for opp_rank in range(0, rank):
                for adj_file in range(max(0, file_index - 1), min(7, file_index + 1) + 1):
                    if board.piece_at(chess.square(adj_file, opp_rank)) == chess.PAWN and not board.piece_at(chess.square(adj_file, opp_rank)).color == chess.BLACK:
                        return False
        return True

    # 4. Backward Pawns Evaluation
    def evaluate_backward_pawns(self, board):
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            for square in board.pieces(chess.PAWN, color):
                if self.is_backward_pawn(board, square, color):
                    if color == chess.WHITE:
                        score += BACKWARD_PAWN_PENALTY
                    else:
                        score -= BACKWARD_PAWN_PENALTY
        return score

    # Helper function to check if a pawn is backward
    def is_backward_pawn(self, board, square, color):
        file_index = chess.square_file(square)
        rank = chess.square_rank(square)

        if color == chess.WHITE:
            for adj_file in range(max(0, file_index - 1), min(7, file_index + 1) + 1):
                for r in range(rank, 8):
                    if board.piece_at(chess.square(adj_file, r)) == chess.PAWN and board.piece_at(chess.square(adj_file, r)).color == chess.WHITE:
                        return False
        else:
            for adj_file in range(max(0, file_index - 1), min(7, file_index + 1) + 1):
                for r in range(0, rank):
                    if board.piece_at(chess.square(adj_file, r)) == chess.PAWN and board.piece_at(chess.square(adj_file, r)).color == chess.BLACK:
                        return False
        return True

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

    # safety of queen, rook, bishop, knight, pawn
    def get_piece_safety(self, board, color):
        safety = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None and piece.color == color:
                if piece.piece_type == chess.QUEEN:
                    safety += 9
                elif piece.piece_type == chess.ROOK:
                    safety += 5
                elif piece.piece_type == chess.BISHOP:
                    safety += 3
                elif piece.piece_type == chess.KNIGHT:
                    safety += 3
                elif piece.piece_type == chess.PAWN:
                    safety += 1
        return safety

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
                    eval += piece_value # + self.pst[piece.piece_type][square]
                else:
                    eval -= piece_value # + self.pst[piece.piece_type][chess.square_mirror(square)]

        white_mobility = self.get_mobility(board, chess.WHITE)
        black_mobility = self.get_mobility(board, chess.BLACK)
        eval += (white_mobility - black_mobility) * 10  # Weight mobility by 10 (can be tuned)


        # Add piece safety factor
        white_piece_safety = self.get_piece_safety(board, chess.WHITE)
        black_piece_safety = self.get_piece_safety(board, chess.BLACK)
        eval += (white_piece_safety - black_piece_safety)  # Adjust the weight if necessary

        # Add pawn structure factor
        eval += self.evaluate_pawn_structure(board)

        # Add king safety factor
        white_king_safety = self.get_king_safety(board, chess.WHITE)
        black_king_safety = self.get_king_safety(board, chess.BLACK)
        eval += (white_king_safety - black_king_safety)  # Adjust the weight if necessary
        return eval
