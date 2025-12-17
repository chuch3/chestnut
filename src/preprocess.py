import os
import pickle
import unittest

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import board_to_matrix

# TODO:
# - [ ] Integrate the MCTS self-play data to the dataset
# - [ ] Check if the policy / value head and the updated model architecture works
# - [ ] Check if MCTS state in the MCTSData namedtuple has y column as the next mainline move
#   - It seems like state in MCTS only has all legal moves, then we need to store the actual move next

_DATASET_NAME: str = "lichess-elite"
_DATASET_PATH: str = os.path.realpath(os.path.join("..", ".data", _DATASET_NAME))

_GAMES_NAME: str = "games.pickle"
_GAMES_PATH: str = os.path.realpath(os.path.join("..", ".data", _GAMES_NAME))

_LOADED_EXPERT_NAME: str = "load.pickle"
_LOADED_EXPERT_PATH: str = os.path.realpath(
    os.path.join("..", ".data", _LOADED_EXPERT_NAME)
)

_LOADED_SELF_PLAY_NAME: str = "SELF_PLAY_DATASET_200SIMULATIONS_200GAMES.pickle"
_LOADED_SELF_PLAY_PATH: str = os.path.realpath(
    os.path.join("..", ".data", _LOADED_SELF_PLAY_NAME)
)

_MAP_NAME: str = "movemap.pickle"
_MAP_PATH: str = os.path.realpath(os.path.join("..", ".data", _MAP_NAME))

_IDXMAP_NAME: str = "idxmovemap.pickle"
_IDXMAP_PATH: str = os.path.realpath(os.path.join("..", ".data", _IDXMAP_NAME))


class ChessExpertDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ChessSelfPlayDataset(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y_policy = y
        self.y_value = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_games(file_path: str, games_limit, file_limit):
    assert file_path == "/home/chu/dev/chess-engine/.data/lichess-elite", (
        "Not suitable filepath with current system!"
    )

    pgn_paths = [file for file in os.listdir(file_path) if file.endswith(".pgn")]
    pgn_limit = min(len(pgn_paths), file_limit)

    games_list = []
    for i, pgn_path in tqdm(enumerate(pgn_paths)):
        if i > pgn_limit:
            break
        with open(os.path.join(_DATASET_PATH, pgn_path), "r") as pgn_file:
            j = 0
            while True:
                if j > games_limit:
                    break
                j += 1
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games_list.append(game)

    with open(_GAMES_PATH, "wb") as file:
        pickle.dump(games_list, file)
        print(f"Saved pickle object {_GAMES_NAME} in {_GAMES_PATH}")

    return games_list


def load_games(games_list, load_limit):
    X, y = [], []

    for game in tqdm(games_list[:load_limit]):
        board = game.board()
        for move in game.mainline_moves():
            X.append(
                board_to_matrix(board)
            )  # Current state of the board into move matrix
            y.append(move.uci())  # Represents the next legal move
            board.push(move)
    loaded_games = (X, y)

    with open(_LOADED_EXPERT_PATH, "wb") as file:
        pickle.dump(loaded_games, file)
        print(f"Saved pickle object {_LOADED_EXPERT_NAME} in {_LOADED_EXPERT_PATH}")

    return np.array(X, dtype=np.float32), np.array(y)


def build_move_mapping():
    all_squares = [chess.square(file, rank) for rank in range(8) for file in range(8)]
    move_set = set()

    for from_square in all_squares:
        for to_square in all_squares:
            move = chess.Move(from_square, to_square)
            move_set.add(move.uci())
            for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promo_move = chess.Move(from_square, to_square, promotion=promo)
                move_set.add(promo_move.uci())

    all_moves_sorted = sorted(list(move_set))

    # Mapping of move UCI <-> index
    move_map = {move: idx for idx, move in enumerate(all_moves_sorted)}
    idx_map = {idx: move for move, idx in move_map.items()}

    return move_map, idx_map


def load_expert_dataset(move_map):
    with open(_LOADED_EXPERT_PATH, "rb") as games_file:
        X, y = pickle.load(games_file)

    # Expert target column is a list of moves, where each move belongs to each state.
    y_idx = np.array([move_map[move] for move in y], dtype=np.float32)

    X = torch.tensor(X, dtype=torch.float32)
    y_idx = torch.tensor(y_idx, dtype=torch.long)

    return X, y_idx


def load_self_play_dataset(memory, move_map):
    """
    with open(_LOADED_SELF_PLAY_PATH, "rb") as self_play:
        memory = pickle.load(self_play)
    """

    state, policy, value = [], [], []
    for mcts_data in memory:
        state.append(board_to_matrix(mcts_data.state))

        policy_vector = np.zeros(len(move_map), dtype=np.float32)
        for move, policy_value in mcts_data.policy:
            policy_vector[move_map[move]] = policy_value
        policy.append(policy_vector)

        value.append(mcts_data.value)

    X = torch.tensor(state, dtype=torch.float32)
    y_policy = torch.tensor(policy, dtype=torch.float32).to_sparse()
    y_value = torch.tensor(value, dtype=torch.float32)

    return X, y_policy, y_value


class TestPreprocess(unittest.TestCase):
    pass

    def test_build_move_mapping(self):
        move_map, idx_map = build_move_mapping()

        print(move_map)
        with open(_MAP_PATH, "wb") as file:
            pickle.dump(move_map, file)
            print(f"Saved pickle object {_MAP_NAME} in {_MAP_PATH}")

        with open(_IDXMAP_PATH, "wb") as file:
            pickle.dump(idx_map, file)
            print(f"Saved pickle object {_IDXMAP_NAME} in {_IDXMAP_PATH}")


"""
with open(_GAMES_PATH, "rb") as games_file:
    games_list = pickle.load(games_file)

X, y = load_games(games_list, 12000)
print(y)
"""


if __name__ == "__main__":
    unittest.main()
