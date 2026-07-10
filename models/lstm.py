import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM classifier."""

    def __init__(
        self,
        num_classes,
        hidden_dim=128,
        dropout_rate=0.0,
        num_layers=1,
    ):
        super().__init__()

        self.initialized = False

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.categorical_vocab_sizes = None
        self.categorical_embedding_dims = None
        self.numeric_size = None
        self.input_dim = None

        self.categorical_embeddings = None
        self.lstm = None
        self.fc = None
        self.dropout = None

    def init_layers(self, x: dict):
        if self.initialized:
            return

        cat_features = x.get('cat_features')
        num_features = x.get('num_features')

        if cat_features is not None:
            self.categorical_vocab_sizes = [
                int(torch.max(cat_features[..., idx]).item()) + 1
                for idx in range(cat_features.shape[-1])
            ]
            self.categorical_embedding_dims = [16] * len(self.categorical_vocab_sizes)

            self.categorical_embeddings = nn.ModuleList(
                [
                    nn.Embedding(vocab_size, emb_dim, padding_idx=0)
                    for vocab_size, emb_dim in zip(
                        self.categorical_vocab_sizes,
                        self.categorical_embedding_dims,
                        strict=True,
                    )
                ]
            )
            embedding_dim_total = sum(self.categorical_embedding_dims)
        else:
            embedding_dim_total = 0

        self.numeric_size = num_features.shape[-1] if num_features is not None else 0

        self.input_dim = embedding_dim_total + self.numeric_size

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(self.hidden_dim, self.num_classes)
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None

        device = (
            cat_features.device if cat_features is not None else num_features.device
        )
        self.to(device)

        self.initialized = True

    def _encode_inputs(self, x: dict) -> torch.Tensor:
        cat_features = x.get('cat_features')
        num_features = x.get('num_features')

        if cat_features is None and num_features is None:
            raise ValueError(
                "No input features ('cat_features' or 'num_features') were provided."
            )

        embedded = None
        if cat_features is not None:
            embedded_features = [
                embedding(cat_features[..., idx].long())
                for idx, embedding in enumerate(self.categorical_embeddings)
            ]
            embedded = torch.cat(embedded_features, dim=-1)

        if num_features is not None:
            num_features = num_features.float()
            x_out = (
                num_features
                if embedded is None
                else torch.cat([embedded, num_features], dim=-1)
            )
        else:
            x_out = embedded

        if x_out.ndim == 2:
            x_out = x_out.unsqueeze(1)

        return x_out

    def forward(self, x: dict) -> torch.Tensor:
        if not self.initialized:
            self.init_layers(x)

        x_encoded = self._encode_inputs(x)

        out, (h_n, c_n) = self.lstm(x_encoded)

        last_hidden = h_n[-1]
        if self.dropout is not None:
            last_hidden = self.dropout(last_hidden)

        logits = self.fc(last_hidden)
        return logits
