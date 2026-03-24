import os
import pickle
import time
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import ChessModel, ChessModelTransfer
from preprocess import ChessExpertDataset, ChessSelfPlayDataset
from utils import (
    _EXPERT_DATASET_NAME,
    _EXPERT_DATASET_PATH,
    _EXPERT_MODEL_NAME,
    _EXPERT_MODEL_PATH,
    _OLD_MODEL_PATH,
    _SELF_PLAY_DATASET_NAME,
    _SELF_PLAY_DATASET_PATH,
    _SELF_PLAY_MAP_NAME,
    _SELF_PLAY_MAP_PATH,
)

# TODO:
# - [ ] Fix the names of the chess dataset


def load_checkpoint(
    model,
    optimizer=None,
    loss_history=None,
    file_name=None,
):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if not os.path.isfile(file_name):
        print("=> No checkpoint found at '{}'".format(file_name))
    else:
        print("=> loading checkpoint '{}'".format(file_name))
        checkpoint = torch.load(file_name)
        start_epoch = checkpoint["epoch"]

        # TODO : Apply filtering to handle mismatched layers in

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])

        if loss_history is not None:
            loss_history = checkpoint["loss_history"]
        print(
            "=> Loaded checkpoint '{}' (epoch {})".format(
                file_name, checkpoint["epoch"]
            )
        )

    return model, optimizer, start_epoch, loss_history


def transfer_learning(
    model: ChessModel,
    device,
    dataloader,
    file_name,
    old_classes=1849,
    learning_rate=0.00001,
    epochs=1,
):
    if not os.path.isfile(file_name):
        print("=> No transfer learning checkpoint found at '{}'".format(file_name))
    else:
        transfer_model = ChessModelTransfer()
        print("=> Transfer learning from checkpoint '{}'".format(file_name))
        checkpoint = torch.load(file_name)  # Loading pre-trained expert model.

        transfer_model.old_policy.load_state_dict(
            checkpoint["state_dict"], strict=False
        )

        # Freezing old base model and weights to prevent learning
        for param in transfer_model.old_policy.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, transfer_model.parameters()),
            lr=learning_rate,
        )

        loss_policy_fn = nn.CrossEntropyLoss()

        # Warmup new policy weights with freezed old policy weights
        for epoch in tqdm(range(epochs)):
            for x, y_policy, _ in dataloader:
                x, y_policy = x.to(device), y_policy.to(device)

                optimizer.zero_grad()
                logits = transfer_model(x)
                new_logits = logits[
                    :, old_classes:
                ]  # Obtain the slice of the new policy weights to train upon
                loss = loss_policy_fn(new_logits, y_policy)

                loss.backward()
                optimizer.step()

        model.encoder.load_state_dict(transfer_model.encoder.state_dict(), strict=False)
        model.policy_head.load_state_dict(
            transfer_model.encoder.state_dict(), strict=False
        )

        print("=> Transfer learning succeeded '{}')".format(file_name))

    return transfer_model


def train_expert(
    X,
    y,
    num_classes,
    max_norm=1.0,
    num_epochs=500,
    learning_rate=0.0001,
    batch_size=64,
    random_state=42,
    self_play=False,
):
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChessExpertDataset(X, y)

    split_ratio = 0.5
    split_size = int(len(dataset) * split_ratio)
    dataset, _ = random_split(dataset, [split_size, len(dataset) - split_size])
    print(f"=> Number of expert moves in training : {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ChessModel(num_classes=num_classes, self_play=False).to(device)

    loss_policy_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    current_epoch = 0

    load_checkpoint(
        model,
        optimizer=optimizer,
        loss_history=loss_history,
        file_name=_EXPERT_MODEL_PATH,
    )

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for x, y in tqdm(dataloader):
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
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s"
        )
        loss_history.append((epoch + 1, loss))
        current_epoch += 1

        if epoch % (100 - 1) == 0:
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
    num_epochs=250,
    learning_rate=0.001,
    batch_size=64,
    random_state=42,
):
    torch.manual_seed(random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChessSelfPlayDataset(X, y_policy, y_value)
    print(f"=> Number of self-play moves in training : {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ChessModel(num_classes=num_classes, self_play=True).to(device)

    loss_policy_fn = nn.KLDivLoss()
    loss_value_fn = nn.MSELoss()

    loss_history = []
    current_epoch = 0

    transfer_learning(model, device, dataloader, _EXPERT_MODEL_PATH)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    """
    load_checkpoint(
        model,
        optimizer=optimizer,
        loss_history=loss_history,
        file_name=_EXPERT_MODEL_PATH,
    )
    """

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for x, y_policy, y_value in tqdm(dataloader):
            # If the self-play is used, the y should be the value from the MCTS tuples
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
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader):.4f}, Time: {minutes}m{seconds}s"
        )
        loss_history.append((epoch + 1, loss))
        current_epoch += 1

        if epoch % (100 - 1) == 0:
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
        with open(_EXPERT_DATASET_PATH, "rb") as file:
            expert_dataset = pickle.load(file)
            print(
                "=> Loaded {} from {}".format(
                    _EXPERT_DATASET_NAME, _EXPERT_DATASET_PATH
                )
            )
        X, y, num_classes = expert_dataset
        train_expert(X, y, num_classes)

    def test_self_play_train(self):
        with open(_SELF_PLAY_DATASET_PATH, "rb") as file:
            self_play_dataset = pickle.load(file)
            print(
                "=> Loaded {} from {}".format(
                    _SELF_PLAY_DATASET_NAME, _SELF_PLAY_DATASET_PATH
                )
            )
        with open(_SELF_PLAY_MAP_PATH, "rb") as file:
            move_map = pickle.load(file)
            print(
                "=> Loaded {} from {}".format(_SELF_PLAY_MAP_NAME, _SELF_PLAY_MAP_PATH)
            )

        X, y_policy, y_value = self_play_dataset
        train_self_play(X, y_policy, y_value, len(move_map))


if __name__ == "__main__":
    unittest.main()
