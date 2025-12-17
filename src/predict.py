import os
import pickle

import chess
import numpy as np
import torch

from model import ChessModel
from preprocess import _MAP_PATH
from utils import board_to_matrix

# TODO:
# - [ ] Predict has to integrate with the new model architecture.

_MODEL_PATH: str = os.path.realpath(
    os.path.join("..", "model", "CHESS_MODEL_48_EPOCHS.pth")
)


def start_chess_model():
    with open(_MAP_PATH, "rb") as file:
        move_map = pickle.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChessModel(num_classes=len(move_map))
    model.load_state_dict(torch.load(_MODEL_PATH))
    model.to(device)
    model.eval()

    idx_move_map = {idx: move for move, idx in move_map.items()}

    return model, idx_move_map


def predict_best_move(board: chess.Board, model: ChessModel, idx_move_map: dict):
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

    return None


def main():
    model, idx_move_map = start_chess_model()
    board = chess.Board()
    board.push_uci("e2e3")
    best_move = predict_best_move(
        board,
        model,
        idx_move_map,
    )
    board.push_uci(best_move)
    board.push_uci("e1e2")
    best_move = predict_best_move(
        board,
        model,
        idx_move_map,
    )
    board.push_uci(best_move)
    print(board)


if __name__ == "__main__":
    main()
