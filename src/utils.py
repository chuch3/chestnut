import chess
import numpy as np
import logging
import argparse

_NUM_UNIQUE_PIECE: int = 12
_BOARD_AXIS: int = 8  # As in 8x8 standard game grid for chess


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
