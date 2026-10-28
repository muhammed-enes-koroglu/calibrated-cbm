"""Concept encoders: map an input to one logit per concept."""

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """Concept predictor on top of precomputed backbone features (e.g. cached ResNet-50)."""

    def __init__(self, in_features: int, n_concepts: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_concepts),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, n_concepts))

    def forward(self, x):
        return self.net(x)


class MNISTPairEncoder(nn.Module):
    """Shared CNN over the two digits of an MNIST-Addition input.

    Input (B, 2, 28, 28) -> 20 concept logits: digit-identity logits of the first image,
    then of the second.
    """

    def __init__(self, n_concepts: int = 20, dropout: float = 0.0):
        super().__init__()
        assert n_concepts == 20, "MNIST-Addition has 2 x 10 digit concepts"
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        b = x.shape[0]
        logits = self.cnn(x.reshape(b * 2, 1, 28, 28))
        return logits.reshape(b, 20)


ENCODERS = {"mlp": MLPEncoder, "mnist_pair": MNISTPairEncoder}


def build_encoder(name: str, n_concepts: int, **kwargs) -> nn.Module:
    return ENCODERS[name](n_concepts=n_concepts, **kwargs)
