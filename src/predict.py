import os
import pickle

import chess
import numpy as np
import torch

from old_model import ChessModel
from train import load_checkpoint
from utils import board_to_matrix

# WARNING:
# The prediction requires the move and index move map to be located into the data directory.

_MOVE_MAP_PATH: str = os.path.realpath(os.path.join("..", "data", "move_map.pickle"))

_MODEL_PATH: str = os.path.realpath(
    os.path.join("..", "model", "CHESSMODEL_CHECKPOINT_100_EPOCHS.pth.tar")
)


def start_chess_model():
    with open(_MOVE_MAP_PATH, "rb") as file:
        old_move_map = pickle.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessModel(num_classes=len(old_move_map))

    model, _, _, _ = load_checkpoint(model, file_name=_MODEL_PATH)

    idx_move_map = {idx: move for move, idx in old_move_map.items()}
    model.to(device)
    model.eval()

    return model, idx_move_map


def predict_best_move(board: chess.Board, model: ChessModel, idx_move_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    move_matrix = board_to_matrix(board)
    move_tensor = torch.tensor(move_matrix, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(move_tensor)

    logits = logits.squeeze(0)
    prob = torch.softmax(logits, dim=0).cpu().numpy()

    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(prob)[::-1]

    for move_index in sorted_indices:
        move = idx_move_map[move_index]
        if move in legal_moves_uci:
            return move
    return
