from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SequenceModel(nn.Module, ABC):
    """Interface for DARWIN sequence encoders."""

    output_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a fixed-size representation for each sequence in batch."""


class LSTMModel(SequenceModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.output_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out[:, -1, :]


class ProcessTransformerModel(SequenceModel):
    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        max_len: int,
        pooling_type: str = 'mean',
    ) -> None:
        super().__init__()

        self.pooling_type = pooling_type
        self.output_dim = head_dim * num_heads

        self.input_projection = nn.Linear(input_dim, self.output_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, self.output_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_dim,
            nhead=num_heads,
            dim_feedforward=self.output_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.pos_embedding.shape[1]:
            raise ValueError(
                f'Sequence length {seq_len} exceeds max_len '
                f'{self.pos_embedding.shape[1]}'
            )

        h = self.input_projection(x)
        h = h + self.pos_embedding[:, :seq_len, :]
        h = self.encoder(h)

        if self.pooling_type == 'mean':
            return torch.mean(h, dim=1)

        elif self.pooling_type == 'last':
            return h[:, -1, :]

        elif self.pooling_type == 'max':
            return torch.max(h, dim=1)[0]

        else:
            raise ValueError(f'Unknown pooling type: {self.pooling_type}')
