from torch.utils.data import Dataset
import os
import chess
import tqdm

_DATASET_NAME: str = "lichess-elite"
_DATASET_PATH: str = os.path.realpath(os.path.join("..", ".data", _DATASET_NAME))


class ChessDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_pgn(pgn_file: str) :
    pass

def load_data(file_path: str) -> None:
    assert file_path == "/home/chu/dev/chess-engine/.data/lichess-elite", (
        "Not suitable filepath with current system!"
    )

    pgn_paths = [file for file in os.listdir(file_path) if file.endswith(".pgn")]
    limit = min(len(pgn_files), 20)

    total_games = []
    for pgn_path in pgn_paths:
        games = []
        with open(os.path.join(_DATASET_PATH, pgn_path), 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
        total_games.extend(games)






    pass


if __name__ == "__main__":
    print(_DATASET_PATH)
