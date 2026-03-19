from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from river import base


@dataclass
class PrefixTreeNode:
    """A node in the Prefix Tree (T) representing a process activity."""

    activity_name: str | None
    parent: PrefixTreeNode | None
    children: dict[str, PrefixTreeNode] = field(default_factory=dict)


@dataclass
class WindowEntry:
    """A labeled prefix sequence extracted from the stream for learning."""

    case_id: str
    prefix_node: PrefixTreeNode
    label_idx: int


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
        batch_size: int,
        dropout: float,
        end_events: set[str] | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.w2v_window = w2v_window
        self.sequence_window = sequence_window
        self.init_size = init_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.end_events = end_events or set()

        self.w2v = None

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=self.dropout,
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

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint = {
            'lstm_state_dict': self.lstm.state_dict(),
            'head_state_dict': self.head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'runtime_state': {
                'w2v': self.w2v,
                'prefix_tree': self.prefix_tree,
                'header_table': self.header_table,
                'active_predictions': self.active_predictions,
                'init_buffer': self.init_buffer,
                'adaptive_window': self.adaptive_window,
                'vocab': self.vocab,
                'idx_to_act': self.idx_to_act,
                'events_processed': self.events_processed,
                'initialized': self.initialized,
            },
        }
        torch.save(checkpoint, Path(path))

    def load_checkpoint(
        self,
        path: str | Path,
        device: str | torch.device = 'cpu',
    ) -> DARWINClassifier:
        checkpoint = torch.load(
            Path(path),
            map_location=device,
            weights_only=False,
        )

        self.lstm.load_state_dict(checkpoint['lstm_state_dict'])
        self.head.load_state_dict(checkpoint['head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        runtime_state = checkpoint['runtime_state']
        self.w2v = runtime_state['w2v']
        self.prefix_tree = runtime_state['prefix_tree']
        self.header_table = runtime_state['header_table']
        self.active_predictions = runtime_state['active_predictions']
        self.init_buffer = runtime_state['init_buffer']
        self.adaptive_window = runtime_state['adaptive_window']
        self.vocab = runtime_state['vocab']
        self.idx_to_act = runtime_state['idx_to_act']
        self.events_processed = runtime_state['events_processed']
        self.initialized = runtime_state['initialized']

        return self

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

    def _to_tensor(self, activities_list: list[list[str]] | list[str]) -> torch.Tensor:
        """Prepare zero-padded tensors for LSTM input."""
        # Handle single sequence
        if not activities_list or isinstance(activities_list[0], str):
            activities_list = [activities_list]

        vectors_batch = []
        for activities in activities_list:
            history = activities[-self.sequence_window :]
            vectors = [self._get_embedding(a) for a in history]
            pad = self.sequence_window - len(vectors)
            if pad > 0:
                vectors = [
                    np.zeros(self.embedding_dim, dtype=np.float32)
                ] * pad + vectors
            vectors_batch.append(np.stack(vectors))

        # (batch_size, sequence_window, embedding_dim)
        return torch.tensor(np.stack(vectors_batch), dtype=torch.float32)

    def _adapt_model(self, samples: list[WindowEntry]) -> None:
        """Updates Word2Vec and fine-tunes the LSTM on the provided window."""
        if not samples:
            return

        # Word2Vec update
        sequences = [self._get_prefix(s.prefix_node) for s in samples]
        if self.w2v is None:
            self.w2v = Word2Vec(
                sg=0,
                vector_size=self.embedding_dim,
                window=self.w2v_window,
                min_count=1,
                workers=1,
            )
            self.w2v.build_vocab(sequences)
        else:
            self.w2v.build_vocab(sequences, update=True)

        self.w2v.train(sequences, total_examples=len(sequences), epochs=1)

        # LSTM fine-tuning
        self.lstm.train()
        self.head.train()

        for i in range(0, len(samples), self.batch_size):
            batch = samples[i : i + self.batch_size]
            batch_sequences = [self._get_prefix(s.prefix_node) for s in batch]
            batch_labels = torch.tensor([s.label_idx for s in batch])

            tensor = self._to_tensor(batch_sequences)
            out, _ = self.lstm(tensor)
            logits = self.head(out[:, -1, :])

            loss = self.loss_fn(logits, batch_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn_one(self, x: dict, y: str):
        """Processes a single event and manages initialization or drift adaptation."""
        case_id, act = x['case_id'], x['activity']

        y_idx = self._map_label(y)
        node = self._update_prefix_tree(case_id, act)

        if not self.initialized:
            self.init_buffer.append(WindowEntry(case_id, node, y_idx))
            self.events_processed += 1

            if self.events_processed >= self.init_size:
                self._adapt_model(self.init_buffer)
                self.init_buffer = []
                self.initialized = True
                print('Initialization completed')
                print(f'{self.events_processed} events processed')
                print(f'Initial vocabulary size: {len(self.vocab)}')
                print(set(self.vocab.keys()))

            if act in self.end_events:
                self.header_table.pop(case_id, None)

            return self

        if case_id in self.active_predictions:
            if self.drift_detector is not None:
                error = 0 if self.active_predictions[case_id] == y_idx else 1
                self.drift_detector.update(error)
                self.adaptive_window.append(WindowEntry(case_id, node, y_idx))

                if self.drift_detector.drift_detected:
                    self._adapt_model(self.adaptive_window)
                    self.adaptive_window = []

            del self.active_predictions[case_id]

        if act in self.end_events:
            self.header_table.pop(case_id, None)
            self.active_predictions.pop(case_id, None)

        return self

    def predict_one(self, x: dict) -> str | None:
        """Predicts the next activity for an ongoing trace."""
        case_id, act = x['case_id'], x['activity']

        if not self.initialized or not act:
            return None

        # Predict based on current state
        node = self.header_table.get(case_id, self.prefix_tree)
        full_seq = self._get_prefix(node) + [act]
        tensor = self._to_tensor(full_seq)

        self.lstm.eval()
        with torch.no_grad():
            out, _ = self.lstm(tensor)
            logits = self.head(out[:, -1, :])
            y_idx = int(torch.argmax(logits, dim=1).item())

        self.active_predictions[case_id] = y_idx
        return self.idx_to_act.get(y_idx)
