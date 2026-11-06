"""Spatial CNN with an optional causal GRU for successive 100 ms event bins."""
import torch
from torch import nn


class GelNeuroTemporalCNN(nn.Module):
    def __init__(self, recurrent=True):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(256, 64), nn.LayerNorm(64),
        )
        self.memory = nn.GRU(64, 64, batch_first=True) if recurrent else None
        self.head = nn.Linear(64, 5)

    def encode(self, bins, state=None):
        """Return causal B,T,64 features and GRU state for the next call."""
        if bins.ndim != 5 or bins.shape[2:] != (2, 32, 32):
            raise ValueError("Expected B,T,2,32,32 log-count event bins")
        features = self.spatial(bins.flatten(0, 1)).unflatten(0, bins.shape[:2])
        if self.memory is None:
            return features, None
        return self.memory(features, state)

    def forward(self, bins, state=None):
        features, state = self.encode(bins, state)
        return self.head(features), state
