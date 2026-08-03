from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from river import base
from river.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon

from models.streaming.sequence import SequenceModel
from utils.streaming.base.feature_buffers import FeatureBuffer
from utils.streaming.base.sample_buffers import (
    PrefixTreeNode,
    SampleBuffer,
)


class DARWINBase(ABC):
    def __init__(
        self,
        sequence_window: int,
        sequence_model: SequenceModel,
        optimizer_cls: Callable,
        loss_fn: nn.Module,
        batch_size: int,
        init_size: int,
        drift_detector: base.DriftDetector | None,
        drift_trigger: str = 'error',
        task: str = 'classification',
        encoding: str = 'word2vec',
        w2v_window: int | None = None,
        embedding_dim: int | None = None,
        epochs: int = 1,
        feature_size: int = 0,
        sample_buffer: SampleBuffer | None = None,
        device: torch.device | None = None,
        allowed_events: set[str] | None = None,
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.allowed_events = (
            set(allowed_events) if allowed_events is not None else None
        )
        self.encoding = encoding.lower()
        self.w2v = None
        self.w2v_window = w2v_window
        self.event_to_idx = (
            {event: i for i, event in enumerate(sorted(self.allowed_events))}
            if self.allowed_events is not None
            else {}
        )

        if self.encoding == 'word2vec':
            if embedding_dim is None or embedding_dim <= 0:
                raise ValueError(
                    "embedding_dim must be specified for 'word2vec' encoding."
                )
            self.event_vector_dim = embedding_dim

        elif self.encoding == 'one_hot':
            if not self.allowed_events or len(self.allowed_events) == 0:
                raise ValueError(
                    "allowed_events must be specified for 'one_hot' encoding"
                )
            self.event_vector_dim = len(self.event_to_idx)

        else:
            raise ValueError("encoding must be either 'word2vec' or 'one_hot'.")

        self.feature_size = feature_size
        self.feature_scalers = [StandardScaler() for _ in range(self.feature_size)]

        self.events_processed: int = 0
        self.initialized: bool = False
        self.device = device or torch.device('cpu')

        self.sequence_model = sequence_model.to(self.device)
        self.head = None

        self.sequence_window = sequence_window
        self.init_size = init_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.optimizer_cls = optimizer_cls
        self.optimizer = None
        self.loss_fn = loss_fn.to(self.device)

        self.loss_history = []

        self.drift_detector = drift_detector
        self.drift_trigger = drift_trigger
        if self.drift_trigger == 'error':
            if task not in ['classification', 'regression']:
                raise ValueError('Invalid task for error-based drift detection.')
            self.task = task

        elif self.drift_trigger == 'control_flow':
            if not self.allowed_events or len(self.allowed_events) == 0:
                raise ValueError('allowed_events must be specified')

            self.baseline_dfg_matrix = None
            self.dfg_matrix = None
            effective_window = max(self.init_size, 1000)
            self.dfg_decay_factor = 1.0 - (1.0 / effective_window)

        else:
            raise ValueError("drift_trigger must be either 'error' or 'control_flow'.")

        # Prefix tree and header table for the most recent event of each case
        self.prefix_tree = PrefixTreeNode(event_name=None, parent=None)
        self.header_table: dict[str, PrefixTreeNode] = {}
        self.previous_events: dict[str, set[int]] = defaultdict(set)

        # Separate header table for learning
        self.learn_table: dict[str, PrefixTreeNode] = {}

        # Separate feature learning and prediction buffers
        self.feature_buffer = (
            FeatureBuffer(sequence_window=sequence_window) if feature_size > 0 else None
        )
        self.learn_feature_buffer = (
            FeatureBuffer(sequence_window=sequence_window) if feature_size > 0 else None
        )

        # Sample buffer for:
        # 1) initialization and 2) drift handling
        self.sample_buffer = (
            sample_buffer if sample_buffer is not None else SampleBuffer()
        )

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
            'sequence_model_state_dict': self.sequence_model.state_dict(),
            'head_state_dict': self.head.state_dict()
            if self.head is not None
            else None,
            'optimizer_state_dict': self.optimizer.state_dict()
            if self.optimizer is not None
            else None,
            'runtime_state': {
                'encoding': self.encoding,
                'allowed_events': self.allowed_events,
                'event_to_idx': self.event_to_idx,
                'event_vector_dim': self.event_vector_dim,
                'w2v': self.w2v,
                'prefix_tree': self.prefix_tree,
                'header_table': self.header_table,
                'learn_table': self.learn_table,
                'previous_events': self.previous_events,
                'sample_buffer': self.sample_buffer,
                'feature_buffer': self.feature_buffer,
                'learn_feature_buffer': self.learn_feature_buffer,
                'feature_scalers': self.feature_scalers,
                'events_processed': self.events_processed,
                'initialized': self.initialized,
                'drift_detector': self.drift_detector,
            },
        }

        # Merge task specific checkpoint data
        checkpoint.update(checkpoint_data)

        try:
            torch.save(checkpoint, Path(path))
        except Exception as e:
            print(f'Error occurred while saving checkpoint: {e}')

    def load_checkpoint(
        self,
        path: str | Path,
    ) -> DARWINBase:
        """Load model and runtime state."""
        checkpoint = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=False,
        )

        self.sequence_model.load_state_dict(checkpoint['sequence_model_state_dict'])
        self.sequence_model.to(self.device)

        runtime_state = checkpoint['runtime_state']
        self.encoding = runtime_state.get('encoding', self.encoding)
        self.allowed_events = runtime_state.get('allowed_events', self.allowed_events)
        self.event_to_idx = runtime_state.get(
            'event_to_idx', getattr(self, 'event_to_idx', {})
        )
        self.event_vector_dim = runtime_state.get(
            'event_vector_dim', getattr(self, 'event_vector_dim', 0)
        )
        if self.encoding == 'one_hot' and not self.event_to_idx and self.allowed_events:
            self.event_to_idx = {
                event: i for i, event in enumerate(sorted(self.allowed_events))
            }
            self.event_vector_dim = len(self.event_to_idx)

        self.w2v = runtime_state['w2v']
        self.prefix_tree = runtime_state['prefix_tree']
        self.header_table = runtime_state['header_table']
        self.learn_table = runtime_state['learn_table']
        self.previous_events = runtime_state['previous_events']
        self.sample_buffer = runtime_state['sample_buffer']
        self.feature_buffer = runtime_state['feature_buffer']
        self.learn_feature_buffer = runtime_state['learn_feature_buffer']
        self.feature_scalers = runtime_state['feature_scalers']
        self.events_processed = runtime_state['events_processed']
        self.initialized = runtime_state['initialized']
        self.drift_detector = runtime_state.get('drift_detector', self.drift_detector)

        head_state = checkpoint.get('head_state_dict')
        if head_state is None:
            self.head = None
        else:
            out_features, in_features = head_state['weight'].shape
            self.head = nn.Linear(in_features, out_features).to(self.device)
            self.head.load_state_dict(head_state)

        # Reconstruct optimizer if head was loaded
        if self.head is not None:
            self.optimizer = self.optimizer_cls(
                list(self.sequence_model.parameters()) + list(self.head.parameters())
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
        event_id: int,
        features: list,
        is_learn: bool = False,
    ) -> None:
        """Update the prefix tree and header table with a new event."""

        # Ignore already processed events
        if event_id not in self.previous_events.get(case_id, set()):
            current_node = self.header_table.get(case_id, self.prefix_tree)
            next_node = self._build_node(current_node, event_name)
            self.header_table[case_id] = next_node

            # Store features for the current event in the feature buffer
            if self.feature_buffer is not None:
                self.feature_buffer.add_features(
                    next_node,
                    case_id,
                    features,
                )

            # Mark event as processed
            if event_id is not None:
                self.previous_events[case_id].add(event_id)

        # Only update learn table during learning phase
        if is_learn:
            current_node = self.learn_table.get(case_id, self.prefix_tree)
            next_node = self._build_node(current_node, event_name)
            self.learn_table[case_id] = next_node

            if self.learn_feature_buffer is not None:
                self.learn_feature_buffer.add_features(
                    next_node,
                    case_id,
                    features,
                )

    def _build_node(
        self,
        parent: PrefixTreeNode,
        event_name: str,
    ) -> PrefixTreeNode:
        """Return a prefix tree node for the given event name."""
        child = parent.children.get(event_name)

        # Create new prefix tree node
        if child is None:
            child = PrefixTreeNode(
                event_name=event_name,
                parent=parent,
            )
            parent.children[event_name] = child

        return child

    def _get_prefix_nodes(self, node: PrefixTreeNode) -> list[PrefixTreeNode]:
        """Reconstruct last sequence_window prefix nodes by backtracking the tree."""
        path: list[PrefixTreeNode] = []
        curr = node
        while curr.parent is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        return path[-self.sequence_window :]

    def _get_event_sequence(self, node: PrefixTreeNode) -> list[str]:
        """Reconstruct the event sequence by backtracking the tree."""
        return [
            n.event_name
            for n in self._get_prefix_nodes(node)
            if n.event_name is not None
        ]

    def _get_event_vector(self, event_name: str) -> np.ndarray:
        """Retrieve either a continuous W2V vector or a pure static one-hot vector."""
        if self.encoding == 'word2vec':
            if self.w2v is not None and event_name in self.w2v.wv:
                return self.w2v.wv[event_name]
            return np.zeros(self.event_vector_dim, dtype=np.float32)

        elif self.encoding == 'one_hot':
            idx = self.event_to_idx.get(event_name)
            if idx is None:
                return np.zeros(self.event_vector_dim, dtype=np.float32)
            vector = np.zeros(self.event_vector_dim, dtype=np.float32)
            vector[idx] = 1.0
            return vector

        return np.zeros(self.event_vector_dim, dtype=np.float32)

    def _get_features(
        self,
        case_id: str,
        node: PrefixTreeNode,
        is_learn: bool = False,
    ) -> list[float]:
        """Retrieve features from feature buffer for the given case and node."""
        buffer = self.learn_feature_buffer if is_learn else self.feature_buffer
        if buffer is None:
            return [0.0] * self.feature_size
        features = buffer.get_features(node, case_id)
        if features is None:
            return [0.0] * self.feature_size
        return features

    def _scale_features(
        self,
        features: list[float],
        fit_scalers: bool = False,
    ) -> np.ndarray:
        """Scale feature vector using the scalers."""
        if self.feature_size == 0:
            return np.zeros(0, dtype=np.float32)

        scaled_vector = np.zeros(self.feature_size, dtype=np.float32)

        for i, (value, scaler) in enumerate(
            zip(features, self.feature_scalers, strict=True)
        ):
            float_value = float(value)

            if fit_scalers:
                scaler.learn_one({'x': float_value})

            scaled_vector[i] = scaler.transform_one({'x': float_value})['x']

        return scaled_vector

    def _to_tensor(
        self,
        prefix_nodes: list[list[PrefixTreeNode]],
        case_ids: list[str],
        is_learn: bool = False,
        fit_scalers: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare zero-padded tensors and padding mask for sequence model input."""
        # Handle single sequence
        input_dim = self.event_vector_dim + self.feature_size
        zero_vector = np.zeros(input_dim, dtype=np.float32)

        vectors_batch = []
        padding_masks_batch = []
        for nodes, case_id in zip(prefix_nodes, case_ids, strict=True):
            window_nodes = nodes[-self.sequence_window :]
            vectors = []

            for i, node in enumerate(window_nodes):
                event_name_embedding = self._get_event_vector(node.event_name)

                features = self._get_features(
                    case_id,
                    node,
                    is_learn=is_learn,
                )
                feature_vector = self._scale_features(
                    features,
                    fit_scalers=fit_scalers,
                )

                # Features only for the most recent event
                # if i == len(window_nodes) - 1:
                #     features = self._get_features(
                #         case_id,
                #         node,
                #         is_learn=is_learn,
                #     )
                #     feature_vector = self._scale_features(
                #         features,
                #         fit_scalers=fit_scalers,
                #     )
                # else:
                #     feature_vector = np.zeros(self.feature_size, dtype=np.float32)

                vector = np.concatenate([event_name_embedding, feature_vector]).astype(
                    np.float32, copy=False
                )

                vectors.append(vector)

            pad = self.sequence_window - len(vectors)
            if pad > 0:
                vectors = [zero_vector.copy() for _ in range(pad)] + vectors
            vectors_batch.append(np.stack(vectors))

            # True means padded token
            padding_masks_batch.append([True] * pad + [False] * (len(window_nodes)))

        # (batch_size, sequence_window, embedding_dim + feature_size)
        tensor = torch.tensor(
            np.stack(vectors_batch),
            dtype=torch.float32,
            device=self.device,
        )
        padding_mask = torch.tensor(
            np.asarray(padding_masks_batch, dtype=np.bool_),
            dtype=torch.bool,
            device=self.device,
        )
        return tensor, padding_mask

    def clear_case(self, case_id: str) -> None:
        self.header_table.pop(case_id, None)
        self.previous_events.pop(case_id, None)
        self.learn_table.pop(case_id, None)
        if self.feature_buffer is not None:
            self.feature_buffer.clear(case_id)
        if self.learn_feature_buffer is not None:
            self.learn_feature_buffer.clear(case_id)

    def _adapt_model(self, sample_buffer: SampleBuffer) -> None:
        """Updates Word2Vec and fine-tunes the sequence model on the provided window."""
        buffer_size = sample_buffer.size
        if buffer_size == 0:
            return

        samples = sample_buffer.get_samples()
        # print(f'\nAdapting model with {len(samples)} samples from buffer')

        # print('Sampled sequences and targets:')
        # for s in samples:
        #     sequence = self._get_event_sequence(s['prefix_node'])
        #     target = s['target']
        #     print(f'Sequence: {sequence}, Target: {target}')
        sequences = [self._get_event_sequence(s.prefix_node) for s in samples]
        # for s in samples:
        # print(f'Sample: {s.case_id} {s.prefix_node}, Target: {s.target}')

        if self.encoding == 'word2vec':
            if self.w2v is None:
                self.w2v = Word2Vec(
                    sg=0,
                    vector_size=self.event_vector_dim,
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

        elif self.encoding == 'one_hot':
            pass

        if self.head is None:
            self._init_head()

        if self.optimizer is None:
            self.optimizer = self.optimizer_cls(
                list(self.sequence_model.parameters()) + list(self.head.parameters())
            )

        # Sequence model fine-tuning
        self.sequence_model.train()
        self.head.train()

        for epoch_idx in range(max(1, self.epochs)):
            epoch_loss_history = []

            for i in range(0, len(samples), self.batch_size):
                batch = samples[i : i + self.batch_size]
                batch_prefix_nodes = [
                    self._get_prefix_nodes(s.prefix_node) for s in batch
                ]
                batch_case_ids = [s.case_id for s in batch]
                batch_targets = self._prepare_target([s.target for s in batch]).to(
                    self.device
                )

                tensor, padding_mask = self._to_tensor(
                    batch_prefix_nodes,
                    batch_case_ids,
                    is_learn=True,
                    fit_scalers=(epoch_idx == 0),
                )
                padding_mask = padding_mask.to(device=tensor.device, dtype=torch.bool)
                hidden = self.sequence_model(tensor, padding_mask=padding_mask)
                logits = self.head(hidden)

                loss = self._compute_loss(logits, batch_targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss_history.append(loss.item())

            avg_epoch_loss = np.mean(epoch_loss_history)
            self.loss_history.append(avg_epoch_loss)

    def _get_dfg_matrix(self, sample_buffer: SampleBuffer) -> np.ndarray:
        """Compute the directly-follows graph (DFG) matrix from the sample buffer."""
        dfg_matrix = np.zeros((len(self.event_to_idx), len(self.event_to_idx)))

        for sample in sample_buffer.get_samples():
            node = sample.prefix_node

            if (
                node is not None
                and node.parent is not None
                and node.parent.event_name is not None
            ):
                event_a = node.parent.event_name
                event_b = node.event_name

                idx_a = self.event_to_idx.get(event_a)
                idx_b = self.event_to_idx.get(event_b)

                if idx_a is not None and idx_b is not None:
                    dfg_matrix[idx_a, idx_b] += 1

        return dfg_matrix

    def _print_dfg_matrix(self, dfg_matrix: np.ndarray) -> None:
        """Print the DFG matrix in a readable format."""
        ordered_events = sorted(
            self.event_to_idx.keys(), key=lambda k: self.event_to_idx[k]
        )
        max_w = max(len(str(e)) for e in ordered_events)
        max_w = max(max_w, 5)

        header = (
            ' ' * max_w + ' | ' + ' | '.join(f'{e:>{max_w}}' for e in ordered_events)
        )
        print(header)
        print('-' * len(header))

        for event_name, row in zip(ordered_events, dfg_matrix, strict=True):
            row_str = ' | '.join(f'{int(val):>{max_w}}' for val in row)
            print(f'{event_name:>{max_w}} | {row_str}')

        print()

    def learn_one(self, x: dict, y: Any):
        """Process one event and update the online learner state."""
        case_id, event_name, event_id = x['case_id'], x['event_name'], x['event_id']
        features = x.get('features', [])

        y_target = self._encode_target(y)

        self._update_prefix_tree(
            case_id,
            event_name,
            event_id,
            features=features,
            is_learn=True,
        )

        node = self.learn_table.get(case_id, self.prefix_tree)

        # print(f'Learn> case_id={case_id}, current_event="{event_name}", label="{y}"')

        if not self.initialized:
            if y_target is not None:
                self.sample_buffer.add_sample(node, case_id, y_target)

            self.events_processed += 1

            if self.events_processed >= self.init_size:
                init_buffer_size = self.sample_buffer.size
                if init_buffer_size > 0:
                    self._adapt_model(self.sample_buffer)

                    if self.drift_trigger == 'error':
                        pass

                    elif self.drift_trigger == 'control_flow':
                        self.baseline_dfg_matrix = self._get_dfg_matrix(
                            self.sample_buffer
                        )
                        self.dfg_matrix = self.baseline_dfg_matrix.copy()

                        # print('Initial DFG matrix:')
                        # self._print_dfg_matrix(self.baseline_dfg_matrix)

                    self.sample_buffer.clear()
                    self.initialized = True

                else:
                    pass
                    # print('Initialization skipped due to empty buffer')

            return self

        if self.drift_detector is not None:
            logits = self._get_pred_logits(case_id, node, is_learn=True)
            y_pred, y_pred_target = self._get_pred(logits)

            drift_detected = False

            if self.drift_trigger == 'error':
                # Only if label is known
                if y_target is not None:
                    drift_signal = None
                    if self.task == 'classification':
                        drift_signal = 0.0 if y_target == y_pred_target else 1.0

                    elif self.task == 'regression':
                        drift_signal = float(abs(y_target - y_pred_target))

                    self.drift_detector.update(drift_signal)

                    drift_detected = self.drift_detector.drift_detected

            elif self.drift_trigger == 'control_flow':
                current_event = node.event_name
                previous_event = (
                    node.parent.event_name if node.parent is not None else None
                )

                # Add the new directly-follows relation
                if current_event is not None and previous_event is not None:
                    idx_prev = self.event_to_idx.get(previous_event)
                    idx_curr = self.event_to_idx.get(current_event)

                    if idx_prev is not None and idx_curr is not None:
                        self.dfg_matrix *= self.dfg_decay_factor
                        self.dfg_matrix[idx_prev, idx_curr] += 1

                # Normalize dfg matrices by rows
                epsilon = 1e-5
                p_matrix_raw = self.baseline_dfg_matrix + epsilon
                base_sums = p_matrix_raw.sum(axis=1, keepdims=True)
                p_matrix = p_matrix_raw / base_sums

                q_matrix_raw = self.dfg_matrix + epsilon
                curr_sums = q_matrix_raw.sum(axis=1, keepdims=True)
                q_matrix = q_matrix_raw / curr_sums

                js_distances = jensenshannon(p_matrix, q_matrix, axis=1)

                # Aggregation into scalar

                # 1. Mean
                # drift_signal = float(np.mean(js_distances))

                # 2. Max
                # drift_signal = float(np.max(js_distances))

                # 3. Frequency-Weighted Mean
                state_weights = base_sums.flatten() / base_sums.sum()
                drift_signal = float(np.average(js_distances, weights=state_weights))

                # 4. Top-k Mean
                # k = 3
                # top_k_indices = np.argsort(js_distances)[-k:]
                # drift_signal = float(np.mean(js_distances[top_k_indices]))

                self.drift_detector.update(drift_signal)

                drift_detected = self.drift_detector.drift_detected

                if drift_detected:
                    # print('Updated DFG matrix (with drift):')
                    # self._print_dfg_matrix(self.dfg_matrix)

                    self.baseline_dfg_matrix = self.dfg_matrix.copy()

            self.sample_buffer.add_sample(node, case_id, y_target)

            if drift_detected:
                self._adapt_model(self.sample_buffer)
                self.sample_buffer.clear()

        return self

    def _get_pred_logits(
        self,
        case_id: str,
        node: PrefixTreeNode,
        is_learn: bool = False,
    ) -> torch.Tensor | None:

        prediction_nodes = self._get_prefix_nodes(node)

        tensor, padding_mask = self._to_tensor(
            [prediction_nodes],
            [case_id],
            is_learn=is_learn,
            fit_scalers=False,
        )
        padding_mask = padding_mask.to(device=tensor.device, dtype=torch.bool)

        self.sequence_model.eval()
        self.head.eval()

        with torch.no_grad():
            hidden = self.sequence_model(tensor, padding_mask=padding_mask)
            logits = self.head(hidden)

        return logits

    def predict_one(self, x: dict) -> Any | None:
        """Predict target for an ongoing case."""
        case_id, event_name, event_id = x['case_id'], x['event_name'], x['event_id']
        features = x.get('features', [])

        self._update_prefix_tree(
            case_id,
            event_name,
            event_id,
            features=features,
            is_learn=False,
        )

        if not self.initialized:
            return None

        node = self.header_table.get(case_id)

        logits = self._get_pred_logits(case_id, node)

        if logits is None:
            return None

        y_pred, y_pred_target = self._get_pred(logits)

        return y_pred


class DARWINClassifier(base.Classifier, DARWINBase):
    def __init__(
        self,
        sequence_window: int,
        sequence_model: SequenceModel,
        optimizer_cls: Callable,
        loss_fn: nn.Module,
        batch_size: int,
        init_size: int,
        drift_detector: base.DriftDetector | None,
        drift_trigger: str = 'error',
        encoding: str = 'word2vec',
        w2v_window: int | None = None,
        embedding_dim: int | None = None,
        epochs: int = 1,
        feature_size: int = 0,
        sample_buffer: SampleBuffer | None = None,
        dynamic_n_classes: bool = False,  # NOTE: archive
        n_classes: int | None = None,
        max_n_classes: int | None = None,
        device: torch.device | None = None,
        allowed_events: set[str] | None = None,
        seed: int | None = None,
    ):
        DARWINBase.__init__(
            self,
            embedding_dim=embedding_dim,
            w2v_window=w2v_window,
            sequence_window=sequence_window,
            sequence_model=sequence_model,
            optimizer_cls=optimizer_cls,
            loss_fn=loss_fn,
            batch_size=batch_size,
            drift_detector=drift_detector,
            drift_trigger=drift_trigger,
            task='classification',
            init_size=init_size,
            epochs=epochs,
            feature_size=feature_size,
            sample_buffer=sample_buffer,
            device=device,
            allowed_events=allowed_events,
            encoding=encoding,
            seed=seed,
        )

        # Class number configuration
        if max_n_classes is not None:
            self.max_n_classes = max_n_classes
            self.n_classes = 0
            self.dynamic_n_classes = True
        else:
            if n_classes is None or n_classes <= 0:
                raise ValueError('n_classes must be positive.')
            self.max_n_classes = n_classes
            self.n_classes = n_classes
            self.dynamic_n_classes = False

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

        out_dim = self.n_classes if self.n_classes > 0 else len(self.vocab)

        if out_dim <= 0:
            raise ValueError('Cannot initialize head with zero classes')

        self.head = nn.Linear(self.sequence_model.output_dim, out_dim).to(self.device)
        self.n_classes = out_dim

    def _prepare_target(self, y: Any) -> torch.Tensor:
        if isinstance(y, list):
            return torch.tensor(y, dtype=torch.long, device=self.device)
        return torch.tensor([y], dtype=torch.long, device=self.device)

    def _encode_target(self, y: Any) -> int | None:
        return self._map_label(y)

    def _get_pred(self, logits: torch.Tensor) -> tuple[Any, int]:
        y_idx = int(torch.argmax(logits, dim=1).item())
        return self.idx_to_label.get(y_idx), y_idx

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(logits, targets)

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
                    # print(f'Expanding classification vocabulary: {label}')
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

        new_head = nn.Linear(hidden_dim, new_n_classes).to(self.device)
        with torch.no_grad():
            if old_n_classes > 0:
                new_head.weight[:old_n_classes].copy_(old_head.weight)
                new_head.bias[:old_n_classes].copy_(old_head.bias)

        self.head = new_head
        self.n_classes = new_n_classes

        self.optimizer = self.optimizer_cls(
            list(self.sequence_model.parameters()) + list(self.head.parameters())
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
        features = x.get('features', [])

        self._update_prefix_tree(
            case_id,
            event_name,
            event_id,
            features=features,
            is_learn=False,
        )

        if not self.initialized:
            return {}

        node = self.header_table.get(case_id)

        logits = self._get_pred_logits(case_id, node)

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
        sequence_window: int,
        sequence_model: SequenceModel,
        optimizer_cls: Callable,
        loss_fn: nn.Module,
        batch_size: int,
        init_size: int,
        drift_detector: base.DriftDetector | None,
        drift_trigger: str = 'error',
        encoding: str = 'word2vec',
        w2v_window: int | None = None,
        embedding_dim: int | None = None,
        epochs: int = 1,
        feature_size: int = 0,
        sample_buffer: SampleBuffer | None = None,
        device: torch.device | None = None,
        allowed_events: set[str] | None = None,
        seed: int | None = None,
    ):
        DARWINBase.__init__(
            self,
            embedding_dim=embedding_dim,
            w2v_window=w2v_window,
            sequence_window=sequence_window,
            sequence_model=sequence_model,
            optimizer_cls=optimizer_cls,
            loss_fn=loss_fn,
            batch_size=batch_size,
            drift_detector=drift_detector,
            drift_trigger=drift_trigger,
            task='regression',
            init_size=init_size,
            epochs=epochs,
            feature_size=feature_size,
            sample_buffer=sample_buffer,
            device=device,
            allowed_events=allowed_events,
            encoding=encoding,
            seed=seed,
        )

    def _init_head(self) -> None:
        if self.head is not None:
            return
        self.head = nn.Linear(self.sequence_model.output_dim, 1).to(self.device)

    def _prepare_target(self, y: Any) -> torch.Tensor:
        if isinstance(y, list):
            return torch.tensor(y, dtype=torch.float32, device=self.device)
        return torch.tensor([y], dtype=torch.float32, device=self.device)

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
        pass

    def learn_one(self, x: dict, y: Any):
        return DARWINBase.learn_one(self, x, y)

    def predict_one(self, x: dict) -> Any | None:
        return DARWINBase.predict_one(self, x)
