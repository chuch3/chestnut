import torch.nn as nn

from utils import _BOARD_AXIS


def _conv_block(in_f, out_f, pool_size, *args, **kwargs):
    conv = nn.Conv2d(in_f, out_f, *args, **kwargs)
    nn.init.xavier_uniform_(conv.weight)
    return nn.Sequential(
        conv,
        nn.ReLU(),
    )


class _Encoder(nn.Module):
    def __init__(self, enc_sizes) -> None:
        super().__init__()

        self.conv_blocks = [
            _conv_block(in_f, out_f, pool_size=2, kernel_size=3, padding=1)
            for in_f, out_f in zip(enc_sizes, enc_sizes[1:])
        ]
        self.encoder = nn.Sequential(*self.conv_blocks)

    def forward(self, X):
        return self.encoder(X)


class _PolicyHead(nn.Module):
    def __init__(self, in_size, out_size, num_classes) -> None:
        super().__init__()

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_size, out_size=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(_BOARD_AXIS**2 * out_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, X):
        return self.policy_head(X)


class _ValueHead(nn.Module):
    def __init__(self, in_size, out_size, num_classes) -> None:
        super().__init__()

        """
        Following AlphaZero's paper, the head drastically reduces the dimensions
        into feature planes. Focusing on strong encoder and smaller heads architecture 
        for parameter efficiency.
        """

        self.head = nn.Sequential(
            nn.Conv2d(in_size, out_size=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(_BOARD_AXIS**2 * out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),  # Value in range [-1, 1], hence we use tanh
        )

    def forward(self, X):
        return self.head(X)


class ChessModel(nn.Module):
    def __init__(self, num_classes, enc_sizes=[13, 64, 128], self_play=False) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.enc_sizes = enc_sizes
        self.self_play = self_play

        self.encoder = _Encoder(self.enc_sizes)
        self.value_head = _ValueHead()
        self.policy_head = _PolicyHead(enc_sizes[-1], self.num_classes)

    def forward(self, X):
        X = self.encoder(X)
        value = self.value_head(X) if self.self_play else None
        policy = self.policy_head(X)
        return policy, value
