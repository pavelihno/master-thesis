from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from river import base


@dataclass
class PrefixTreeNode:
    """A node in the Prefix Tree (T) representing a process event."""

    event_name: str | None
    parent: PrefixTreeNode | None


@dataclass
class WindowEntry:
    """A labeled prefix sequence extracted from the stream for learning."""

    case_id: str
    prefix_node: PrefixTreeNode
    target: Any


class DARWINBase(ABC):
    def __init__(
        self,
        embedding_dim: int,
        w2v_window: int,
        sequence_window: int,
        optimizer_cls: Callable,
        loss_fn: nn.Module,
        lstm_layers: int,
        hidden_dim: int,
        dropout: float,
        batch_size: int,
        drift_detector: base.DriftDetector | None,
        init_size: int,
        epochs: int = 1,
        early_stop_patience: int | None = None,
        end_events: set[str] | None = None,
    ):
        self.embedding_dim = embedding_dim
        self.w2v_window = w2v_window
        self.sequence_window = sequence_window
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim
        self.init_size = init_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.end_events = end_events or set()

        self.events_processed: int = 0
        self.initialized: bool = False

        self.w2v = None

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=self.dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = None

        self.optimizer_cls = optimizer_cls
        self.optimizer = None
        self.loss_fn = loss_fn

        self.loss_history = []

        self.drift_detector = drift_detector

        # Prefix tree and header table for the most recent event of each case
        self.prefix_tree = PrefixTreeNode(event_name=None, parent=None)
        self.header_table: dict[str, PrefixTreeNode] = {}
        self.previous_events: dict[str, set[int]] = defaultdict(set)

        # Separate table for learning
        self.learn_table: dict[str, PrefixTreeNode] = {}

        # Initialization buffer
        self.init_buffer: list[WindowEntry] = []

        # Window W
        self.adaptive_window: list[WindowEntry] = []

    @abstractmethod
    def _init_head(self) -> None:
        """Initialize the output head."""
        pass

    @abstractmethod
    def _prepare_target(self, y: Any) -> torch.Tensor:
        """Convert target to tensor format."""
        pass

    @abstractmethod
    def _encode_target(self, y: Any) -> Any | None:
        """Encode raw target into internal representation."""
        pass

    @abstractmethod
    def _get_pred(self, logits: torch.Tensor) -> tuple[Any, Any]:
        """Return user prediction and drift-comparison value."""
        pass

    @abstractmethod
    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss between model output and targets."""
        pass

    @abstractmethod
    def _get_drift_signal(self, y_true, y_pred) -> float:
        """Compute drift signal from true and predicted values."""
        pass

    @abstractmethod
    def _get_checkpoint_data(self) -> dict:
        """Return task-specific state for checkpointing."""
        pass

    @abstractmethod
    def _load_checkpoint_data(self, checkpoint: dict) -> None:
        pass

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model and runtime state."""
        checkpoint_data = self._get_checkpoint_data()
        checkpoint = {
            'lstm_state_dict': self.lstm.state_dict(),
            'head_state_dict': self.head.state_dict()
            if self.head is not None
            else None,
            'optimizer_state_dict': self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
            'runtime_state': {
                'w2v': self.w2v,
                'prefix_tree': self.prefix_tree,
                'header_table': self.header_table,
                'learn_table': self.learn_table,
                'previous_events': self.previous_events,
                'init_buffer': self.init_buffer,
                'adaptive_window': self.adaptive_window,
                'events_processed': self.events_processed,
                'initialized': self.initialized,
                'drift_detector': self.drift_detector,
            },
        }

        # Merge task specific checkpoint data
        checkpoint.update(checkpoint_data)

        torch.save(checkpoint, Path(path))

    def load_checkpoint(
        self,
        path: str | Path,
        device: str | torch.device = 'cpu',
    ) -> DARWINBase:
        """Load model and runtime state."""
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)

        self.lstm.load_state_dict(checkpoint['lstm_state_dict'])

        runtime_state = checkpoint['runtime_state']
        self.w2v = runtime_state['w2v']
        self.prefix_tree = runtime_state['prefix_tree']
        self.header_table = runtime_state['header_table']
        self.learn_table = runtime_state['learn_table']
        self.previous_events = runtime_state['previous_events']
        self.init_buffer = runtime_state['init_buffer']
        self.adaptive_window = runtime_state['adaptive_window']
        self.events_processed = runtime_state['events_processed']
        self.initialized = runtime_state['initialized']
        self.drift_detector = runtime_state.get('drift_detector', self.drift_detector)

        head_state = checkpoint.get('head_state_dict')
        if head_state is None:
            self.head = None
        else:
            out_features, in_features = head_state['weight'].shape
            self.head = nn.Linear(in_features, out_features)
            self.head.load_state_dict(head_state)

        # Reconstruct optimizer if head was loaded
        if self.head is not None:
            self.optimizer = self.optimizer_cls(
                list(self.lstm.parameters()) + list(self.head.parameters())
            )
            opt_state = checkpoint.get('optimizer_state_dict')
            if opt_state is not None:
                self.optimizer.load_state_dict(opt_state)
        else:
            self.optimizer = None

        # Load task-specific data
        self._load_checkpoint_data(checkpoint)

        return self

    def _update_prefix_tree(
        self,
        case_id: str,
        event_name: str,
        event_id: int | None = None,
        is_learn: bool = False,
    ) -> None:
        """Update the prefix tree and header table with a new event."""

        # Ignore already processed events
        if event_id not in self.previous_events.get(case_id, set()):
            current_node = self.header_table.get(case_id, self.prefix_tree)
            next_node = PrefixTreeNode(event_name=event_name, parent=current_node)
            self.header_table[case_id] = next_node

            # Mark event as processed
            if event_id is not None:
                self.previous_events[case_id].add(event_id)

        # Only update learn table during learning phase
        if is_learn:
            current_node = self.learn_table.get(case_id, self.prefix_tree)
            next_node = PrefixTreeNode(event_name=event_name, parent=current_node)
            self.learn_table[case_id] = next_node

    def _get_prefix(self, node: PrefixTreeNode) -> list[str]:
        """Reconstruct the event sequence by backtracking the tree."""
        path = []
        curr = node
        while curr.parent is not None:
            if curr.event_name is not None:
                path.append(curr.event_name)
            curr = curr.parent
        path.reverse()
        return path[-self.sequence_window :]

    def _get_embedding(self, event_name: str) -> np.ndarray:
        """Retrieve the Word2Vec embedding for an event."""
        if self.w2v is not None and event_name in self.w2v.wv:
            return self.w2v.wv[event_name]
        return np.zeros(self.embedding_dim, dtype=np.float32)

    def _to_tensor(self, events_list: list[list[str]] | list[str]) -> torch.Tensor:
        """Prepare zero-padded tensors for LSTM input."""
        # Handle single sequence
        if not events_list or isinstance(events_list[0], str):
            events_list = [events_list]

        vectors_batch = []
        for events in events_list:
            history = events[-self.sequence_window :]
            vectors = [self._get_embedding(e) for e in history]
            pad = self.sequence_window - len(vectors)
            if pad > 0:
                vectors = [
                    np.zeros(self.embedding_dim, dtype=np.float32)
                ] * pad + vectors
            vectors_batch.append(np.stack(vectors))

        # (batch_size, sequence_window, embedding_dim)
        return torch.tensor(np.stack(vectors_batch), dtype=torch.float32)

    def _clear(self, case_id: str) -> None:
        self.header_table.pop(case_id, None)
        self.previous_events.pop(case_id, None)
        self.learn_table.pop(case_id, None)

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

        self.w2v.train(
            sequences,
            total_examples=len(sequences),
            epochs=max(1, self.epochs),
        )

        if self.head is None:
            self._init_head()

        if self.optimizer is None:
            self.optimizer = self.optimizer_cls(
                list(self.lstm.parameters()) + list(self.head.parameters())
            )

        # LSTM fine-tuning
        self.lstm.train()
        self.head.train()

        best_loss = float('inf')
        early_stop_patience_counter = 0

        for _ in range(max(1, self.epochs)):
            epoch_loss_history = []

            for i in range(0, len(samples), self.batch_size):
                batch = samples[i : i + self.batch_size]
                batch_sequences = [self._get_prefix(s.prefix_node) for s in batch]
                batch_targets = self._prepare_target([s.target for s in batch])

                tensor = self._to_tensor(batch_sequences)
                out, _ = self.lstm(tensor)
                logits = self.head(out[:, -1, :])

                loss = self._compute_loss(logits, batch_targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss_history.append(loss.item())

            avg_epoch_loss = np.mean(epoch_loss_history)
            self.loss_history.append(avg_epoch_loss)

            # Early stopping
            if self.early_stop_patience is not None:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    early_stop_patience_counter = 0
                else:
                    early_stop_patience_counter += 1
                    if early_stop_patience_counter >= self.early_stop_patience:
                        break

    def learn_one(self, x: dict, y: Any):
        """Process one event and update the online learner state."""
        case_id, event_name, event_id = x['case_id'], x['event_name'], x['event_id']

        y_target = self._encode_target(y)

        self._update_prefix_tree(case_id, event_name, event_id, is_learn=True)

        node = self.learn_table.get(case_id, self.prefix_tree)

        # print(f'Learn> case_id={case_id}, current_event="{event_name}", label="{y}"')
        # print(f'Learn sequence: {self._get_prefix(node)}\n')

        if not self.initialized:
            if y_target is not None:
                self.init_buffer.append(WindowEntry(case_id, node, y_target))
            self.events_processed += 1

            if self.events_processed >= self.init_size:
                if len(self.init_buffer) > 0:
                    self._adapt_model(self.init_buffer)

                    self.init_buffer = []
                    self.initialized = True

                    # print('Initialization completed')
                    # print(f'{self.events_processed} events processed')
                    # print(f'Initial vocabulary size: {len(self.vocab)}')
                    # print(set(self.vocab.keys()), '\n')

                    # for event_name in self.vocab.keys():
                    #     if event_name in self.w2v.wv:
                    #         vector = self.w2v.wv[event_name][:5]
                    #         print(f'Event: "{event_name}", Vector: {vector}...')
                else:
                    pass
                    # print('Initialization skipped due to empty buffer')

            if event_name in self.end_events:
                self._clear(case_id)

            return self

        if self.drift_detector is not None:
            y_pred = self.predict_one(x)
            y_pred_target = self._encode_target(y_pred)

            # Only if label is known
            if y_target is not None:
                drift_signal = self._get_drift_signal(
                    y_target,
                    y_pred_target,
                )
                self.drift_detector.update(drift_signal)
                self.adaptive_window.append(WindowEntry(case_id, node, y_target))

                if self.drift_detector.drift_detected:
                    self._adapt_model(self.adaptive_window)
                    self.adaptive_window = []

        if event_name in self.end_events:
            self._clear(case_id)

        return self

    def _get_pred_logits(
        self, case_id: str, event_name: str, event_id: int | None
    ) -> torch.Tensor | None:

        self._update_prefix_tree(case_id, event_name, event_id, is_learn=False)

        if not self.initialized:
            return None

        node = self.header_table.get(case_id, self.prefix_tree)
        prediction_sequence = self._get_prefix(node)

        # print(f'Predict> case_id={case_id}, current_event="{event_name}"')
        # print(f'Prediction sequence: {prediction_sequence}\n')

        tensor = self._to_tensor(prediction_sequence)

        self.lstm.eval()
        self.head.eval()

        with torch.no_grad():
            out, _ = self.lstm(tensor)
            logits = self.head(out[:, -1, :])

        return logits

    def predict_one(self, x: dict) -> Any | None:
        """Predict target for an ongoing case."""
        case_id, event_name, event_id = x['case_id'], x['event_name'], x['event_id']

        logits = self._get_pred_logits(case_id, event_name, event_id)

        if logits is None:
            return None

        y_pred, y_pred_target = self._get_pred(logits)

        return y_pred


class DARWINClassifier(base.Classifier, DARWINBase):
    def __init__(
        self,
        embedding_dim: int,
        w2v_window: int,
        sequence_window: int,
        optimizer_cls: Callable,
        loss_fn: nn.Module,
        lstm_layers: int,
        hidden_dim: int,
        dropout: float,
        batch_size: int,
        drift_detector: base.DriftDetector | None,
        init_size: int,
        epochs: int = 1,
        early_stop_patience: int | None = None,
        end_events: set[str] | None = None,
        dynamic_n_classes: bool = False,
        n_classes: int | None = None,
        max_n_classes: int | None = None,
    ):
        DARWINBase.__init__(
            self,
            embedding_dim=embedding_dim,
            w2v_window=w2v_window,
            sequence_window=sequence_window,
            optimizer_cls=optimizer_cls,
            loss_fn=loss_fn,
            lstm_layers=lstm_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size,
            drift_detector=drift_detector,
            init_size=init_size,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            end_events=end_events,
        )

        # Class number configuration
        self.dynamic_n_classes = dynamic_n_classes
        if dynamic_n_classes:
            self.max_n_classes = max_n_classes
            self.n_classes = 0
        else:
            if n_classes is None or n_classes <= 0:
                raise ValueError('In fixed mode n_classes must be a positive integer')
            self.max_n_classes = n_classes
            self.n_classes = n_classes

        self.vocab: dict[Any, int] = {}  # label -> index
        self.idx_to_label: dict[int, Any] = {}  # index -> label

    @property
    def _multiclass(self) -> bool:
        if self.dynamic_n_classes:
            return True
        upper = self.max_n_classes if self.max_n_classes is not None else self.n_classes
        return bool(upper is not None and upper > 2)

    def _init_head(self) -> None:
        if self.head is not None:
            return

        if self.dynamic_n_classes:
            self.n_classes = len(self.vocab)
            if self.n_classes == 0:
                raise ValueError('Cannot initialize head with zero classes')
        elif self.n_classes <= 0:
            raise ValueError('In fixed mode n_classes must be a positive integer')

        self.head = nn.Linear(self.lstm.hidden_size, self.n_classes)

    def _prepare_target(self, y: Any) -> torch.Tensor:
        if isinstance(y, list):
            return torch.tensor(y, dtype=torch.long)
        return torch.tensor([y], dtype=torch.long)

    def _encode_target(self, y: Any) -> int | None:
        return self._map_label(y)

    def _get_pred(self, logits: torch.Tensor) -> tuple[Any, int]:
        y_idx = int(torch.argmax(logits, dim=1).item())
        return self.idx_to_label.get(y_idx), y_idx

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(logits, targets)

    def _get_drift_signal(self, y_true: int, y_pred: int) -> float:
        return 0.0 if y_true == y_pred else 1.0

    def _map_label(self, label: str) -> int | None:
        """Map label to categorical classification index."""
        if label not in self.vocab:
            idx = len(self.vocab)

            if self.max_n_classes is not None and idx >= self.max_n_classes:
                raise KeyError(f'Exceeded maximum number of classes: {label}')

            self.vocab[label] = idx
            self.idx_to_label[idx] = label

            if self.dynamic_n_classes:
                if self.initialized and idx >= self.n_classes:
                    # print(f'Expanding vocabulary: {label}')
                    # print(f'Current vocabulary size: {len(self.vocab)}')
                    # print(set(self.vocab.keys()))
                    # print(set(self.vocab.keys()), '\n')

                    self._expand_head()

        return self.vocab[label]

    def _expand_head(self) -> None:
        new_n_classes = len(self.vocab)
        if new_n_classes <= self.n_classes:
            return

        old_head = self.head
        old_n_classes = old_head.out_features
        hidden_dim = old_head.in_features

        new_head = nn.Linear(hidden_dim, new_n_classes)
        with torch.no_grad():
            if old_n_classes > 0:
                new_head.weight[:old_n_classes].copy_(old_head.weight)
                new_head.bias[:old_n_classes].copy_(old_head.bias)

        self.head = new_head
        self.n_classes = new_n_classes

        self.optimizer = self.optimizer_cls(
            list(self.lstm.parameters()) + list(self.head.parameters())
        )

    def _get_checkpoint_data(self) -> dict:
        return {
            'task_state': {
                'n_classes': self.n_classes,
                'vocab': self.vocab,
                'idx_to_label': self.idx_to_label,
            }
        }

    def _load_checkpoint_data(self, checkpoint: dict) -> None:
        task_state = checkpoint['task_state']

        self.vocab = task_state['vocab']
        self.n_classes = task_state.get('n_classes', len(self.vocab))
        self.idx_to_label = task_state['idx_to_label']

    def predict_proba_one(self, x: dict) -> dict[Any, float]:
        case_id, event_name, event_id = x['case_id'], x['event_name'], x['event_id']

        logits = self._get_pred_logits(case_id, event_name, event_id)

        if logits is None or len(self.idx_to_label) == 0:
            return {}

        probs = torch.softmax(logits, dim=1).reshape(-1).tolist()
        return {self.idx_to_label[i]: float(probs[i]) for i in range(len(probs))}

    def learn_one(self, x: dict, y: Any):
        return DARWINBase.learn_one(self, x, y)

    def predict_one(self, x: dict) -> Any | None:
        return DARWINBase.predict_one(self, x)


class DARWINRegressor(base.Regressor, DARWINBase):
    def __init__(
        self,
        embedding_dim: int,
        w2v_window: int,
        sequence_window: int,
        optimizer_cls: Callable,
        loss_fn: nn.Module,
        lstm_layers: int,
        hidden_dim: int,
        dropout: float,
        batch_size: int,
        drift_detector: base.DriftDetector | None,
        init_size: int,
        epochs: int = 1,
        early_stop_patience: int | None = None,
        end_events: set[str] | None = None,
    ):
        DARWINBase.__init__(
            self,
            embedding_dim=embedding_dim,
            w2v_window=w2v_window,
            sequence_window=sequence_window,
            optimizer_cls=optimizer_cls,
            loss_fn=loss_fn,
            lstm_layers=lstm_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size,
            drift_detector=drift_detector,
            init_size=init_size,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            end_events=end_events,
        )

    def _init_head(self) -> None:
        if self.head is not None:
            return
        self.head = nn.Linear(self.lstm.hidden_size, 1)

    def _prepare_target(self, y: Any) -> torch.Tensor:
        if isinstance(y, list):
            return torch.tensor(y, dtype=torch.float32)
        return torch.tensor([y], dtype=torch.float32)

    def _encode_target(self, y: Any) -> float | None:
        if y is None:
            return None
        return float(y)

    def _get_pred(self, logits: torch.Tensor) -> tuple[float, float]:
        y_pred = float(logits.reshape(-1)[0].item())
        return y_pred, y_pred

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return self.loss_fn(logits, targets)

    def _get_drift_signal(self, y_true: float, y_pred: float) -> float:
        return float(abs(y_true - y_pred))

    def _get_checkpoint_data(self) -> dict:
        return {'task_state': {}}

    def _load_checkpoint_data(self, checkpoint: dict) -> None:
        task_state = checkpoint.get('task_state', {})

    def learn_one(self, x: dict, y: Any):
        return DARWINBase.learn_one(self, x, y)

    def predict_one(self, x: dict) -> Any | None:
        return DARWINBase.predict_one(self, x)
