import pickle

import chess
import numpy as np
import torch

from model import ChessModel
from train import load_checkpoint

# WARNING:
# The prediction requires the move and index move map to be located into the data directory.
from utils import (
    _EXPERT_MODEL_PATH,
    _EXPERT_MOVE_MAP_PATH,
    _SELF_PLAY_MAP_PATH,
    _SELF_PLAY_MODEL_PATH,
    board_to_matrix,
)


def start_chess_model():
    with open(_SELF_PLAY_MAP_PATH, "rb") as file:
        move_map = pickle.load(file)
    loss_history = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessModel(num_classes=len(move_map), self_play=True)

    model, _, _, loss_history = load_checkpoint(
        model, loss_history=loss_history, file_name=_SELF_PLAY_MODEL_PATH
    )

    idx_move_map = {idx: move for move, idx in move_map.items()}
    model.to(device)
    model.eval()

    return model, idx_move_map
    """
    with open(_EXPERT_MOVE_MAP_PATH, "rb") as file:
        move_map = pickle.load(file)
    loss_history = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessModel(num_classes=len(move_map))

    model, _, _, loss_history = load_checkpoint(
        model, loss_history=loss_history, file_name=_EXPERT_MODEL_PATH
    )

    idx_move_map = {idx: move for move, idx in move_map.items()}
    model.to(device)
    model.eval()

    return model, idx_move_map
    """


def predict_best_move(board: chess.Board, model: ChessModel, idx_move_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    move_matrix = board_to_matrix(board)
    move_tensor = torch.tensor(move_matrix, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        policy, _ = model(move_tensor)

    policy = policy.squeeze(0)
    prob = torch.softmax(policy, dim=0).cpu().numpy()

    legal_moves = list(board.legal_moves)
    legal_moves_uci = [move.uci() for move in legal_moves]
    sorted_indices = np.argsort(prob)[::-1]

    for move_index in sorted_indices:
        move = idx_move_map[move_index]
        if move in legal_moves_uci:
            return move
    return
