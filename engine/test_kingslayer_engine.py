from kingslayer_engine import KingslayerChessEngine
import chess
results = []


def run_game():
    ts_engine = KingslayerChessEngine()
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(
        "stockfish/stockfish-ubuntu-x86-64")
    stockfish_engine.configure({"Skill Level": 1})
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)
    board_result = ""
    while True:
        # Kingslayer move
        if board.is_game_over():
            board_result = board.result()
            break
        move = ts_engine.get_best_move(fen)
        if move in [mv.uci() for mv in board.legal_moves]:
            board.push_uci(move)
        
        # Print the board state
        fen = board.fen()
        print(f"Kingslayer move: {move}, FEN: {fen}")
        if board.is_game_over():
            board_result = board.result()
            break
        # Stockfish move
        result = stockfish_engine.play(board, chess.engine.Limit(time=0.1))
        stockfish_move = result.move.uci()
        if stockfish_move in [move.uci() for move in board.legal_moves]:
            board.push_uci(stockfish_move)
        # Update FEN after Stockfish move
        fen = board.fen()
        print(f"Stockfish move: {stockfish_move}, FEN: {fen}")

    print(f"Game over: {board.result()}")
    return board_result


if __name__ == "__main__":
    # run_game()
    arr = []
    for _ in range(5):
        arr.append(run_game())
            
    print("Game over")
    print(arr)
    print("Done")
    exit()