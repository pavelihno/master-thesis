from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from river import base


@dataclass
class PrefixTreeNode:
    """A node in the Prefix Tree (T) representing a process activity."""

    activity_name: str | None
    parent: 'PrefixTreeNode' | None
    children: dict[str, 'PrefixTreeNode'] = field(default_factory=dict)


@dataclass
class WindowEntry:
    """A labeled prefix sequence extracted from the stream for learning."""

    case_id: str
    prefix_node: PrefixTreeNode
    label_idx: int
    clf_error: int | None = None


class DARWINClassifier(base.Classifier):
    def __init__(
        self,
        embedding_dim: int,
        w2v_window: int,
        sequence_window: int,
        optimizer_cls: Callable,
        loss_fn: nn.Module,
        lstm_layers: int,
        hidden_dim: int,
        n_classes: int,
        drift_detector: base.DriftDetector,
        init_size: int,
        end_events: set[str] | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.w2v_window = w2v_window
        self.sequence_window = sequence_window
        self.init_size = init_size
        self.end_events = end_events or set()

        self.w2v = None

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, n_classes)

        self.optimizer = optimizer_cls(
            list(self.lstm.parameters()) + list(self.head.parameters())
        )
        self.loss_fn = loss_fn

        self.drift_detector = drift_detector
        self.n_classes = n_classes

        # Data Synopsis D = (Hash Table H, Prefix Tree T)
        self.prefix_tree = PrefixTreeNode(activity_name=None, parent=None)
        self.header_table: dict[str, PrefixTreeNode] = {}

        # Hash table A
        self.active_predictions: dict[str, int] = {}

        # Initialization buffer
        self.init_buffer: list[WindowEntry] = []

        # Window W
        self.adaptive_window: list[WindowEntry] = []

        self.vocab: dict[str, int] = {}
        self.idx_to_act: dict[int, str] = {}

        self.events_processed: int = 0
        self.initialized: bool = False

    def _update_prefix_tree(self, case_id: str, activity_name: str) -> PrefixTreeNode:
        """Update the prefix tree and header table with a new event."""
        current_node = self.header_table.get(case_id, self.prefix_tree)
        next_node = current_node.children.get(activity_name)

        if next_node is None:
            next_node = PrefixTreeNode(activity_name=activity_name, parent=current_node)
            current_node.children[activity_name] = next_node

        self.header_table[case_id] = next_node
        return next_node

    def _get_prefix(self, node: PrefixTreeNode) -> list[str]:
        """Reconstruct the activity sequence by backtracking the tree."""
        path = []
        curr = node
        while curr.parent is not None:
            if curr.activity_name is not None:
                path.append(curr.activity_name)
            curr = curr.parent
        path.reverse()
        return path[-self.sequence_window :]

    def _map_label(self, activity: str) -> int:
        """Map activity names to categorical classification indices."""
        if activity not in self.vocab:
            idx = len(self.vocab)
            if idx < self.n_classes:
                self.vocab[activity] = idx
                self.idx_to_act[idx] = activity
            else:
                return 0
        return self.vocab[activity]

    def _get_embedding(self, activity: str) -> np.ndarray:
        """Retrieve the Word2Vec embedding for an activity."""
        if self.w2v is not None and activity in self.w2v.wv:
            return self.w2v.wv[activity]
        return np.zeros(self.embedding_dim, dtype=np.float32)

    def _to_tensor(self, activities: list[str]) -> torch.Tensor:
        """Prepare a zero-padded tensor for LSTM input."""
        history = activities[-self.sequence_window :]
        vectors = [self._get_embedding(a) for a in history]

        pad = self.sequence_window - len(vectors)
        if pad > 0:
            vectors = [np.zeros(self.embedding_dim, dtype=np.float32)] * pad + vectors

        # (1, sequence_window, embedding_dim)
        return torch.tensor(np.stack(vectors), dtype=torch.float32).unsqueeze(0)

    def _adapt_model(self, samples: list[WindowEntry]) -> None:
        """Updates Word2Vec and fine-tunes the LSTM on the provided window."""
        if not samples:
            return

        # Word2Vec update
        sequences = [self._get_prefix(s.prefix_node) for s in samples]
        if self.w2v is None:
            self.w2v = Word2Vec(
                vector_size=self.embedding_dim, window=self.w2v_window, min_count=1
            )
            self.w2v.build_vocab(sequences)
        else:
            self.w2v.build_vocab(sequences, update=True)

        self.w2v.train(sequences, total_examples=len(sequences), epochs=1)

        # LSTM fine-tuning
        self.lstm.train()
        self.head.train()

        for s in samples:
            tensor = self._to_tensor(self._get_prefix(s.prefix_node))
            out, _ = self.lstm(tensor)
            logits = self.head(out[:, -1, :])

            loss = self.loss_fn(logits, torch.tensor([s.label_idx]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn_one(self, x: dict, y: str):
        """Processes a single event and manages initialization or drift adaptation."""
        cid, act = x['case_id'], x['activity']

        y_idx = self._map_label(y)
        node = self._update_prefix_tree(cid, act)

        if not self.initialized:
            self.init_buffer.append(WindowEntry(cid, node, y_idx))
            self.processed_events += 1

            if self.processed_events >= self.init_size:
                self._adapt_model(self.init_buffer)
                self.init_buffer = []
                self.initialized = True

            if act in self.end_events:
                self.header_table.pop(cid, None)

            return self

        if cid in self.active_predictions:
            error = 0 if self.active_predictions[cid] == y_idx else 1

            self.drift_detector.update(error)
            self.adaptive_window.append(WindowEntry(cid, node, y_idx, error))

            if self.drift_detector.drift_detected:
                self._adapt_model(self.adaptive_window)
                self.adaptive_window = []

            del self.active_predictions[cid]

        if act in self.end_events:
            self.header_table.pop(cid, None)
            self.active_predictions.pop(cid, None)

        return self

    def predict_one(self, x: dict) -> str | None:
        """Predicts the next activity for an ongoing trace."""
        cid, act = x['case_id'], x['activity']

        if not self.initialized or not act:
            return None

        # Predict based on current state
        node = self.header_table.get(cid, self.prefix_tree)
        full_seq = self._get_prefix(node) + [act]
        tensor = self._to_tensor(full_seq)

        self.lstm.eval()
        with torch.no_grad():
            out, _ = self.lstm(tensor)
            logits = self.head(out[:, -1, :])
            y_idx = int(torch.argmax(logits, dim=1).item())

        self.active_predictions[cid] = y_idx
        return self.idx_to_act.get(y_idx)
