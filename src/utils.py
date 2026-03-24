import argparse
import logging
import os
import pickle

import chess
import numpy as np

_NUM_UNIQUE_PIECE: int = 12
_BOARD_AXIS: int = 8  # As in 8x8 standard game grid for chess

# ------------------------------------------------------------------------------#


_EXPERT_MOVE_MAP_PATH: str = os.path.realpath(
    os.path.join("..", "data", "move_map.pickle")
)

_SELF_PLAY_MAP_NAME: str = "movemap.pickle"
_SELF_PLAY_MAP_PATH: str = os.path.realpath(
    os.path.join("..", ".data", _SELF_PLAY_MAP_NAME)
)

_OLD_MODEL_NAME: str = "CHESSMODEL_CHECKPOINT_200_EPOCHS.pth.tar"
_OLD_MODEL_PATH: str = os.path.realpath(os.path.join("..", "model", _OLD_MODEL_NAME))

_EXPERT_MODEL_NAME: str = "CHESSMODEL_EXPERT_CHECKPOINT_400_EPOCHS.pth.tar"
_EXPERT_MODEL_PATH: str = os.path.realpath(
    os.path.join("..", "model", _EXPERT_MODEL_NAME)
)

_SELF_PLAY_MODEL_NAME: str = "CHESSMODEL_SELF_PLAY_CHECKPOINT_200_EPOCHS.pth.tar"
_SELF_PLAY_MODEL_PATH: str = os.path.realpath(
    os.path.join("..", "model", _SELF_PLAY_MODEL_NAME)
)

_DATASET_NAME: str = "lichess-elite"
_DATASET_PATH: str = os.path.realpath(os.path.join("..", ".data", _DATASET_NAME))

_GAMES_NAME: str = "games.pickle"
_GAMES_PATH: str = os.path.realpath(os.path.join("..", ".data", _GAMES_NAME))

_LOADED_EXPERT_GAMES_NAME: str = "load.pickle"
_LOADED_EXPERT_GAMES_PATH: str = os.path.realpath(
    os.path.join("..", ".data", _LOADED_EXPERT_GAMES_NAME)
)

_EXPERT_DATASET_NAME: str = "EXPERT_FINAL_DATASET.pickle"
_EXPERT_DATASET_PATH: str = os.path.realpath(
    os.path.join("..", ".data", _EXPERT_DATASET_NAME)
)

_SELF_PLAY_MEMORY_NAME: str = "SELF_PLAY_150SIMULATIONS_1100GAMES.pickle"
_SELF_PLAY_MEMORY_PATH: str = os.path.realpath(
    os.path.join("..", ".data", _SELF_PLAY_MEMORY_NAME)
)

_SELF_PLAY_DATASET_NAME: str = "SELF_PLAY_FINAL_DATASET.pickle"
_SELF_PLAY_DATASET_PATH: str = os.path.realpath(
    os.path.join("..", ".data", _SELF_PLAY_DATASET_NAME)
)


_SELF_PLAY_IDXMAP_NAME: str = "idxmovemap.pickle"
_SELF_PLAY_IDXMAP_PATH: str = os.path.realpath(
    os.path.join("..", ".data", _SELF_PLAY_IDXMAP_NAME)
)


# Reference code : [link](https://github.com/Skripkon/chess-engine/tree/main)
def board_to_matrix(board: chess.Board):
    move_matrix = np.zeros(
        (_NUM_UNIQUE_PIECE + 1, _BOARD_AXIS, _BOARD_AXIS)
    )  # Creating the board moves of every unique pieces with the last one for legal moves

    # Maps every piece into its square index as key
    piece_map = board.piece_map()

    # Initializing the first 12 unique pieces move matrix, first rows are the black's side.
    for square, piece in piece_map.items():
        row, col = divmod(square, _BOARD_AXIS)
        # Retrieving the starting moving matrix's index based on the color. [W, ... , B, ... , Legal]
        piece_color = 0 if piece.color else (_NUM_UNIQUE_PIECE / 2)
        idx = piece_color + piece.piece_type - 1
        move_matrix[int(idx), row, col] = 1  # Bitmaps can be used instead

    # Initializing all legal move
    legal_moves = board.legal_moves
    for move in legal_moves:
        square = move.to_square
        row, col = divmod(square, _BOARD_AXIS)
        move_matrix[-1, row, col] = 1

        assert (move_matrix[-1] == move_matrix[_NUM_UNIQUE_PIECE]).all(), (
            "The last move matrix is not the legal move in chess!"
        )

    return move_matrix


# NOTE:
# Have it accept tuple or mutliple variables to load into it. May need to check online references.
def pickle_load(file_name, file_path):
    with open(file_path, "rb") as file:
        output = pickle.load(
            file,
        )
        print("=> Loaded {} from {}".format(file_name, file_path))
    return output


def arg_logs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-l", "--log", help="prints logging information of program", type=str
    )
    args = parser.parse_args()

    level = None
    if args.log == "debug":
        level = logging.DEBUG
    elif args.log == "info":
        level = logging.INFO

    logging.basicConfig(level=level)
