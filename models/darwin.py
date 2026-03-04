from collections import deque
from collections.abc import Callable

import torch
import torch.nn as nn
from river import base


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        window_size: int,
        optimizer_cls: Callable,
        criterion: nn.Module,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        self.vocab = {}
        self.buffer = deque(maxlen=2 * window_size + 1)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size, bias=True)

        # Tie S and S^T
        self.output.weight.data = self.embedding.weight.data

        self.optimizer = optimizer_cls(self.parameters())
        self.criterion = criterion

    def get_id(self, activity):
        if activity not in self.vocab:
            if len(self.vocab) < self.vocab_size:
                self.vocab[activity] = len(self.vocab)
            else:
                return 0
        return self.vocab[activity]

    def update_embeddings(self, activity_id):
        self.buffer.append(activity_id)
        if len(self.buffer) < self.buffer.maxlen:
            return

        center_idx = self.window_size
        center_id = torch.tensor(self.buffer[center_idx])
        context_ids = [
            self.buffer[i] for i in range(len(self.buffer)) if i != center_idx
        ]

        context_tensor = torch.tensor(context_ids)
        hidden = self.embedding(context_tensor).mean(dim=0, keepdim=True)
        logits = self.output(hidden)

        loss = self.criterion(logits, center_id)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def encode(self, activity_id):
        with torch.no_grad():
            idx = torch.tensor([activity_id])
            return self.embedding(idx)


class DARWINClassifier(base.Classifier):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        window_size: int,
        w2v_optimizer_cls: Callable,
        w2v_criterion: nn.Module,
        clf_optimizer_cls: Callable,
        clf_criterion: nn.Module,
        lstm_layers: int,
        hidden_dim: int,
        n_classes: int,
        drift_detector: base.DriftDetector,
        end_events: set[str] | None = None,
    ):
        self.w2v = Word2Vec(
            vocab_size, embedding_dim, window_size, w2v_optimizer_cls, w2v_criterion
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, n_classes)

        self.optimizer = clf_optimizer_cls(
            list(self.lstm.parameters()) + list(self.head.parameters())
        )
        self.criterion = clf_criterion

        self.drift_detector = drift_detector
        self.n_classes = n_classes
        self.end_events = end_events or set()

        self.case_states: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.predictions: dict[
            str, int
        ] = {}  # hash table A: {case_id: last_predicted_activity}
        self.window: list[
            tuple[dict, str]
        ] = []  # adaptive window W for fine-tuning after drift

        self.label_vocab: dict[str, int] = {}
        self.label_idx_to_name: dict[int, str] = {}

    def _get_label_id(self, label: str) -> int:
        """Map an activity name to an integer class index."""
        if label not in self.label_vocab:
            idx = len(self.label_vocab)
            if idx < self.n_classes:
                self.label_vocab[label] = idx
                self.label_idx_to_name[idx] = label
            else:
                return 0
        return self.label_vocab[label]

    def _get_embedding(self, activity_name: str) -> tuple[torch.Tensor, int]:
        act_id = self.w2v.get_id(activity_name)
        self.w2v.update_embeddings(act_id)
        return self.w2v.encode(act_id).unsqueeze(0), act_id

    def learn_one(self, x: dict, y: str):
        case_id = x['case_id']
        act_name = x['activity']

        y_idx = self._get_label_id(y)

        if case_id in self.predictions:
            clf_error = 0 if self.predictions[case_id] == y_idx else 1

            self.window.append((x, y))
            self.drift_detector.update(clf_error)

            if self.drift_detector.drift_detected:
                self._fine_tune()
                self.window = []

            del self.predictions[case_id]

        emb, _ = self._get_embedding(act_name)
        h_0, c_0 = self.case_states.get(case_id, (None, None))

        self.lstm.train()
        self.head.train()

        out, (h_n, c_n) = self.lstm(emb, (h_0, c_0) if h_0 is not None else None)
        logits = self.head(out[:, -1, :])

        loss = self.criterion(logits, torch.tensor([y_idx]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.case_states[case_id] = (h_n.detach(), c_n.detach())

        if y in self.end_events:
            self.case_states.pop(case_id, None)

        return self

    def _fine_tune(self):
        """Re-train on the adaptive window after a drift is detected."""
        if not self.window:
            return

        self.lstm.train()
        self.head.train()

        for x_w, y_w in self.window:
            act_name_w = x_w['activity']
            emb, _ = self._get_embedding(act_name_w)
            out, _ = self.lstm(emb)
            logits = self.head(out[:, -1, :])
            y_idx_w = self._get_label_id(y_w)

            loss = self.criterion(logits, torch.tensor([y_idx_w]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict_one(self, x: dict):
        case_id = x['case_id']
        act_name = x['activity']

        if not act_name:
            return None

        emb, _ = self._get_embedding(act_name)
        h_0, c_0 = self.case_states.get(case_id, (None, None))

        self.lstm.eval()
        self.head.eval()
        with torch.no_grad():
            out, _ = self.lstm(emb, (h_0, c_0) if h_0 is not None else None)
            logits = self.head(out[:, -1, :])
            y_idx = torch.argmax(logits, dim=1).item()

        self.predictions[case_id] = y_idx
        return self.label_idx_to_name.get(y_idx)
