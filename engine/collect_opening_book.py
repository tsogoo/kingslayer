import sqlite3
import os 
import chess.pgn

import requests
from bs4 import BeautifulSoup


class CollectOpeningBook:
    def __init__(self) -> None:
        self.sqlite_path = "chess_engine.db"

    def create_sqlite(self):
        # create opening_book table
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS opening_book (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                move TEXT NOT NULL,
                fen TEXT NOT NULL,
                description TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def index_sqlite(self):
        # create index for fen of opening_book table
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_fen ON opening_book (fen)
        ''')
        conn.commit()
        conn.close()

    def update_sqlite(self, data):
        # insert or update data to opening_book table
        for row in data:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO opening_book (move, fen, description)
                VALUES (?, ?, ?)
            ''', (row[0], row[1], row[2]))
            conn.commit()
            conn.close()

    def download_pgn_from_website(self, url, base_url=""):
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            pgn_links = soup.find_all("a", href=True)
            pgn_files = [base_url+link['href'] for link in pgn_links if link['href'].endswith('.pgn')]
            
            for pgn_file in pgn_files:
                pgn_response = requests.get(pgn_file)
                if pgn_response.status_code == 200:
                    print(f"downloaded {pgn_file} and saving..")
                    with open("pgn/"+pgn_file.split("/")[-1], "wb") as f:
                        f.write(pgn_response.content)
                    # self.import_pgn("pgn/"+pgn_file.split("/")[-1])
        else:
            print("Failed to access the website.")

    def import_pgns(self, file_path):
        # check file_path is directory
        files = []
        if os.path.isdir(file_path):
            files = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
        else:
            # If it's a single file, add it to the list
            files = [file_path]
        for file in files:
            with open(file) as pgn_file:
                print(f"Importing {file}")
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    # Extract moves and FEN strings from the game
                    board = game.board()
                    for move in game.mainline_moves():
                        # Apply the move to the board
                        fen = board.fen()
                        board.push(move)
                        move_uci = move.uci()

                        self.update_sqlite([(move_uci, fen, "Description for the move")])


cob = CollectOpeningBook()
cob.create_sqlite()
cob.index_sqlite()
cob.download_pgn_from_website("https://beginchess.com/downloads", "")
#cob.import_pgns("pgn")