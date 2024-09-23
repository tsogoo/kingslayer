# generate training data for stockfish like position evaluation neural network

import sqlite3
import chess.engine


# function get fen from chsess_engine.db
def get_fens(start, end):
    conn = sqlite3.connect("chess_engine.db")
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT fen FROM opening_book ORDER BY id LIMIT {start}, {end}")
    fens = cursor.fetchall()
    conn.close()
    return fens


def add_fens_with_score(fens):
    conn = sqlite3.connect("training.db")
    engine = chess.engine.SimpleEngine.popen_uci(
        "stockfish/stockfish-ubuntu-x86-64")
    engine.configure({"Skill Level": 20})
    cursor = conn.cursor()
    for fen in fens:
        board = chess.Board(fen[0])
        result = engine.analyse(board, chess.engine.Limit(time=0.1))
        score = result["score"].relative.score()
        if score or score == 0:
            cursor.execute(
                "INSERT INTO training_data (fen, score) VALUES (?, ?)", (
                    fen[0], score))
    conn.commit()
    conn.close()
    engine.quit()
    engine.close()


def initialize_table():
    conn = sqlite3.connect("training.db")
    cursor = conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fen TEXT NOT NULL,
            score INTEGER NOT NULL
        )
        '''
    )
    # empty the table
    cursor.execute("DELETE FROM training_data")
    conn.commit()
    conn.close()


def main():
    initialize_table()
    fens = get_fens(0, 10000)
    add_fens_with_score(fens)


if __name__ == "__main__":
    main()
    print("Done")
    exit()
