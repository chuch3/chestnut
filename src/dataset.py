from torch.utils.data import Dataset
import os
import chess
import chess.pgn
from tqdm import tqdm

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


def load_pgn(file_path: str) -> None:
    assert file_path == "/home/chu/dev/chess-engine/.data/lichess-elite", (
        "Not suitable filepath with current system!"
    )

    pgn_paths = [file for file in os.listdir(file_path) if file.endswith(".pgn")]
    pgn_limit = min(len(pgn_paths), 1)

    total_games = []
    for i, pgn_path in tqdm(enumerate(pgn_paths)):
        if i > pgn_limit:
            break
        games = []
        with open(os.path.join(_DATASET_PATH, pgn_path), 'r') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                print(game)
                if game is None:
                    break
                games.append(game)
        total_games.extend(games)
    total_games


if __name__ == "__main__":
    games = load_pgn(_DATASET_PATH)
    print(len(games))
