import os
import pickle
import unittest

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Used to load the self-play dataset pickle.
from self_play import MCTSData
from utils import (
    _DATASET_PATH,
    _EXPERT_DATASET_NAME,
    _EXPERT_DATASET_PATH,
    _EXPERT_MOVE_MAP_PATH,
    _GAMES_NAME,
    _GAMES_PATH,
    _LOADED_EXPERT_GAMES_NAME,
    _LOADED_EXPERT_GAMES_PATH,
    _SELF_PLAY_DATASET_NAME,
    _SELF_PLAY_DATASET_PATH,
    _SELF_PLAY_IDXMAP_NAME,
    _SELF_PLAY_IDXMAP_PATH,
    _SELF_PLAY_MAP_NAME,
    _SELF_PLAY_MAP_PATH,
    _SELF_PLAY_MEMORY_NAME,
    _SELF_PLAY_MEMORY_PATH,
    board_to_matrix,
)


class ChessSelfPlayDataset(Dataset):
    def __init__(self, X, y_policy, y_value) -> None:
        super().__init__()
        self.X = X
        self.y_policy = y_policy
        self.y_value = y_value

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_policy[idx], self.y_value[idx]


class ChessExpertDataset(Dataset):
    def __init__(self, X, y: tuple) -> None:
        super().__init__()
        self.X = X
        self.y = y

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
        print(f"=> Saved pickle object {_GAMES_NAME} in {_GAMES_PATH}")

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

    with open(_LOADED_EXPERT_GAMES_PATH, "wb") as file:
        pickle.dump(loaded_games, file)
        print(
            f"=> Saved pickle object {_LOADED_EXPERT_GAMES_NAME} in {_LOADED_EXPERT_GAMES_PATH}"
        )

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

    # Mapping of move UCI <-> Index
    move_map = {move: idx for idx, move in enumerate(all_moves_sorted)}
    idx_map = {idx: move for move, idx in move_map.items()}

    return move_map, idx_map


def load_expert_dataset():
    with open(_EXPERT_MOVE_MAP_PATH, "rb") as file:
        move_map = pickle.load(file)

    with open(_LOADED_EXPERT_GAMES_PATH, "rb") as games_file:
        X, y = pickle.load(games_file)
        print(
            "=> Loaded {} from {}".format(
                _LOADED_EXPERT_GAMES_NAME, _LOADED_EXPERT_GAMES_PATH
            )
        )

    # Expert target column is a list of moves, where each move belongs to each state.
    y_idx = np.array([move_map[move] for move in y], dtype=np.float32)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y_idx = torch.tensor(np.array(y_idx), dtype=torch.long)

    return X, y_idx, len(move_map)


def load_self_play_dataset(memory, move_map):
    state, policy, value = [], [], []
    for mcts_data in memory:
        state.append(board_to_matrix(mcts_data.state))

        policy_vector = np.zeros(len(move_map), dtype=np.float32)
        for move, policy_value in mcts_data.policy:
            policy_vector[move_map[move]] = policy_value
        policy.append(policy_vector)

        value.append(mcts_data.value)

    X = torch.tensor(np.array(state), dtype=torch.float32)
    y_policy = torch.tensor(np.array(policy), dtype=torch.float32)
    y_value = torch.tensor(np.array(value), dtype=torch.float32)

    return X, y_policy, y_value


class TestPreprocess(unittest.TestCase):
    @unittest.skip
    def test_build_entire_move_mapping(self):
        move_map, idx_map = build_move_mapping()

        with open(_SELF_PLAY_MAP_PATH, "wb") as file:
            pickle.dump(move_map, file)
            print(
                f"=> Saved pickle object {_SELF_PLAY_MAP_NAME} in {_SELF_PLAY_MAP_PATH}"
            )

        with open(_SELF_PLAY_IDXMAP_PATH, "wb") as file:
            pickle.dump(idx_map, file)
            print(
                f"=> Saved pickle object {_SELF_PLAY_IDXMAP_NAME} in {_SELF_PLAY_IDXMAP_PATH}"
            )

    @unittest.skip
    def test_load_games():
        with open(_GAMES_PATH, "rb") as games_file:
            games_list = pickle.load(games_file)

        X, y = load_games(games_list, 12000)

    @unittest.skip
    def test_load_expert_dataset(self):
        X, y, num_classes = load_expert_dataset()
        expert_dataset = (X, y, num_classes)

        with open(_EXPERT_DATASET_PATH, "wb") as file:
            pickle.dump(expert_dataset, file)
            print(
                "=> Saved pickle object {} in {}".format(
                    _EXPERT_DATASET_NAME, _EXPERT_DATASET_PATH
                )
            )

    @unittest.skip
    def test_load_self_play_dataset(self):
        with open(_SELF_PLAY_MEMORY_PATH, "rb") as memory_file:
            memory = pickle.load(memory_file)
            print(
                "=> Saved pickle object {} in {}".format(
                    _SELF_PLAY_MEMORY_NAME, _SELF_PLAY_MEMORY_PATH
                )
            )

        with open(_SELF_PLAY_MAP_PATH, "rb") as map_file:
            move_map = pickle.load(map_file)
            print(
                "=> Saved pickle object {} in {}".format(
                    _SELF_PLAY_MAP_NAME, _SELF_PLAY_MAP_PATH
                )
            )

        X, y_policy, y_value = load_self_play_dataset(memory, move_map)
        final_self_play = (X, y_policy, y_value)

        with open(_SELF_PLAY_DATASET_NAME, "wb") as file:
            pickle.dump(final_self_play, file)
            print(
                "=> Saved pickle object {} in {}".format(
                    _SELF_PLAY_DATASET_NAME, _SELF_PLAY_DATASET_PATH
                )
            )


if __name__ == "__main__":
    unittest.main()
