from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SequenceModel(nn.Module, ABC):
    """Interface for DARWIN sequence encoders."""

    output_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
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
        pooling_dropout: float,
        pooling_type: str = 'last',
    ) -> None:
        super().__init__()

        self.pooling_type = pooling_type
        self.encoder_dim = head_dim * num_heads
        self.output_dim = 128

        self.input_projection = nn.Linear(input_dim, self.encoder_dim)

        pos_embedding = torch.zeros(max_len, self.encoder_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.encoder_dim, 2).float()
            * (-math.log(10000.0) / self.encoder_dim)
        )

        pos_embedding[:, 0::2] = torch.sin(position * div_term)
        pos_embedding[:, 1::2] = torch.cos(position * div_term)

        # Reshape to (1, max_len, encoder_dim)
        pos_embedding = pos_embedding.unsqueeze(0)
        self.register_buffer('pos_embedding', pos_embedding)

        self.pooling_dropout = nn.Dropout(pooling_dropout)
        self.dense_layers = nn.Sequential(
            nn.Linear(self.encoder_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_dim,
            nhead=num_heads,
            dim_feedforward=self.encoder_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.pos_embedding.shape[1]:
            raise ValueError(
                f'Sequence length {seq_len} exceeds max_len '
                f'{self.pos_embedding.shape[1]}'
            )

        h = self.input_projection(x)
        h = h + self.pos_embedding[:, :seq_len, :]
        h = self.encoder(h, src_key_padding_mask=padding_mask)

        valid_mask = (~padding_mask).unsqueeze(-1)

        if self.pooling_type == 'mean':
            masked_h = h * valid_mask
            valid_counts = valid_mask.sum(dim=1).clamp_min(1)
            pooled = masked_h.sum(dim=1) / valid_counts

        elif self.pooling_type == 'last':
            lengths = (~padding_mask).sum(dim=1).clamp_min(1)
            last_idx = lengths - 1
            pooled = h[torch.arange(h.size(0), device=h.device), last_idx, :]

        elif self.pooling_type == 'max':
            neg_inf = torch.finfo(h.dtype).min
            masked_h = h.masked_fill(~valid_mask, neg_inf)
            pooled = torch.max(masked_h, dim=1)[0]
            all_padded = (~padding_mask).sum(dim=1) == 0
            if all_padded.any():
                pooled[all_padded] = 0.0

        else:
            raise ValueError(f'Unknown pooling type: {self.pooling_type}')

        pooled = self.pooling_dropout(pooled)

        return self.dense_layers(pooled)
