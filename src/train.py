import os
import pickle
import time
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ChessModel
from preprocess import ChessExpertDataset, ChessSelfPlayDataset, load_expert_dataset

_FINAL_SELF_PLAY_NAME: str = "SELF_PLAY_FINAL_DATASET.pickle"
_FINAL_SELF_PLAY_PATH: str = os.path.realpath(
    os.path.join("..", ".data", _FINAL_SELF_PLAY_NAME)
)

_MAP_NAME: str = "movemap.pickle"
_MAP_PATH: str = os.path.realpath(os.path.join("..", ".data", _MAP_NAME))


def load_checkpoint(
    model,
    optimizer=None,
    loss_history=None,
    file_name="CHESSMODEL_CHECKPOINT_EPOCHS.pth.tar",
):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        checkpoint = torch.load(file_name)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if loss_history:
            loss_history = checkpoint["loss_history"]
        print(
            "=> Loaded checkpoint '{}' (epoch {})".format(
                file_name, checkpoint["epoch"]
            )
        )
    else:
        print("=> No checkpoint found at '{}'".format(file_name))

    return model, optimizer, start_epoch, loss_history


def train_expert(
    X,
    y,
    num_classes,
    max_norm=1.0,
    num_epochs=50,
    learning_rate=0.0001,
    batch_size=64,
    random_state=42,
    self_play=False,
):
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChessSelfPlayDataset(X, y)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ChessModel(num_classes=num_classes, self_play=False).to(device)

    loss_policy_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    current_epoch = 0
    mask_self_play = 1.0 if self_play else 0.0  # 1.0 to take information, 0.0 to ignore

    # Intial model checkpoint if there is any
    # - load_checkpoint(model, optimizer, loss_history, "CHESSMODEL_CHECKPOINT_{current_epoch}_EPOCHS.pth.tar")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for x, y in tqdm(dataloader):
            # BUG: If the self-play is used, the y should be the value from the MCTS tuples
            x, y = x.to(device), y.to(device)

            policy, value = model(x)

            loss = loss_policy_fn(policy, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            running_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time
        minutes: int = int(epoch_time // 60)
        seconds: int = int(epoch_time) - minutes * 60
        print(
            f"Epoch {epoch + 1}/{num_epochs + 1}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s"
        )
        loss_history.append((epoch, loss))
        current_epoch += 1

        if epoch % 10 == 0:
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_history": loss_history,
            }
            torch.save(
                state, f"CHESSMODEL_EXPERT_CHECKPOINT_{current_epoch}_EPOCHS.pth.tar"
            )


def train_self_play(
    X,
    y_policy,
    y_value,
    num_classes,
    max_norm=1.0,
    num_epochs=50,
    learning_rate=0.0001,
    batch_size=64,
    random_state=42,
):
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChessExpertDataset(X, y_policy, y_value)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ChessModel(num_classes=num_classes, self_play=True).to(device)

    loss_policy_fn = nn.KLDivLoss()
    loss_value_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    current_epoch = 0
    print(len(dataset))

    # Intial model checkpoint if there is any
    # - load_checkpoint(model, optimizer, loss_history, "CHESSMODEL_CHECKPOINT_{current_epoch}_EPOCHS.pth.tar")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for x, y_policy, y_value in tqdm(dataloader):
            # BUG: If the self-play is used, the y should be the value from the MCTS tuples
            x, y_policy, y_value = x.to(device), y_policy.to(device), y_value.to(device)

            pred_policy, pred_value = model(x)

            # Converting the logits from policy head output to log probability for KLDivLoss()
            log_prob_policy = torch.log_softmax(pred_policy, dim=1)
            loss_policy = loss_policy_fn(log_prob_policy, y_policy)
            loss_value = loss_value_fn(pred_value, y_value)
            loss = loss_policy + loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            running_loss += loss.item()

        end_time = time.time()
        epoch_time = end_time - start_time
        minutes: int = int(epoch_time // 60)
        seconds: int = int(epoch_time) - minutes * 60
        print(
            f"Epoch {epoch + 1}/{num_epochs + 1}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s"
        )
        loss_history.append((epoch, loss))
        current_epoch += 1

        if epoch % 10 == 0:
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss_history": loss_history,
            }
            torch.save(
                state, f"CHESSMODEL_SELF_PLAY_CHECKPOINT_{current_epoch}_EPOCHS.pth.tar"
            )


class TestTrain(unittest.TestCase):
    @unittest.skip
    def test_expert_train(self):
        X, y, num_classes = load_expert_dataset()
        train_expert(X, y, num_classes)

    def test_self_play_train(self):
        with open(_FINAL_SELF_PLAY_PATH, "rb") as self_play_dataset_file:
            self_play_dataset = pickle.load(self_play_dataset_file)

        with open(_MAP_PATH, "rb") as move_file:
            move_map = pickle.load(move_file)

        X, y_policy, y_value = self_play_dataset
        train_self_play(X, y_policy, y_value, len(move_map))


if __name__ == "__main__":
    unittest.main()
