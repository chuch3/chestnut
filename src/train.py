import os
import time
import unittest

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ChessModel
from preprocess import ChessExpertDataset, ChessSelfPlayDataset, load_expert_dataset


def load_checkpoint(
    model,
    optimizer,
    loss_history,
    filename="CHESSMODEL_CHECKPOINT_EPOCHS.pth.tar",
):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_history = checkpoint["loss_history"]
        print(
            "=> Loaded checkpoint '{}' (epoch {})".format(filename, checkpoint["epoch"])
        )
    else:
        print("=> No checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, loss_history


def train(
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

    dataset = None if self_play else ChessExpertDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ChessModel(num_classes=num_classes, self_play=self_play).to(device)

    loss_policy_fn = nn.KLDivLoss() if self_play else nn.CrossEntropyLoss()
    loss_value_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []
    current_epoch = 0
    mask_self_play = 1.0 if self_play else 0.0  # 1.0 to take information, 0.0 to ignore

    # Intial model checkpoint
    # - load_checkpoint(model, optimizer, loss_history, "CHESSMODEL_CHECKPOINT_{current_epoch}_EPOCHS.pth.tar")

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for x, y in tqdm(dataloader):
            # BUG: If the self-play is used, the y should be the value from the MCTS tuples
            x, y = x.to(device), y.to(device)

            policy, value = model(x)

            # Converting the logits from policy head output to log probability for KLDivLoss()
            if self_play:
                policy = torch.log_softmax(policy, dim=1)
            loss_policy = loss_policy_fn(policy, y)
            loss_value = loss_value_fn(value, y)

            loss = loss_policy + mask_self_play * loss_value

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
            torch.save(state, f"CHESSMODEL_CHECKPOINT_{current_epoch}_EPOCHS.pth.tar")


class TestTrain(unittest.TestCase):
    def test_expert_train(self):
        pass


if __name__ == "__main__":
    X, y, num_classes = load_expert_dataset()
    train(X, y, num_classes)
