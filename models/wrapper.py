import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


class ModelWrapper(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        pass


class SklearnModelWrapper(ModelWrapper):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {'model': self.model}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def save(self, path):
        path = Path(path)
        with open(path / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with open(path / 'model.pkl', 'rb') as f:
            model = pickle.load(f)
        return cls(model)


class DictDataset(Dataset):
    """Dataset that preserves dictionary structure for PyTorch DataLoaders."""

    def __init__(self, X: dict, y_tensor=None):
        self.cat_features = X.get('cat_features')
        self.num_features = X.get('num_features')
        self.y_tensor = y_tensor

        if self.cat_features is not None and self.cat_features.shape[-1] > 0:
            self.length = len(self.cat_features)
        elif self.num_features is not None and self.num_features.shape[-1] > 0:
            self.length = len(self.num_features)
        else:
            raise ValueError(
                "Both 'cat_features' and 'num_features' are empty or missing."
            )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {}
        if self.cat_features is not None and self.cat_features.shape[-1] > 0:
            item['cat_features'] = torch.as_tensor(
                self.cat_features[idx], dtype=torch.long
            )
        if self.num_features is not None and self.num_features.shape[-1] > 0:
            item['num_features'] = torch.as_tensor(
                self.num_features[idx], dtype=torch.float32
            )

        if self.y_tensor is not None:
            return item, self.y_tensor[idx]
        return item


class TorchModelWrapper(ModelWrapper):
    def __init__(
        self, model, optimizer_cls, loss_fn, epochs=20, batch_size=64, device='cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer_cls = optimizer_cls
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def _encode_labels(self, y, fit=False):
        labels = np.asarray(y).ravel()
        if fit:
            encoded = self.label_encoder.fit_transform(labels)
            self.classes_ = self.label_encoder.classes_
            return encoded
        if self.classes_ is None:
            raise ValueError('Label encoder not fitted.')
        return self.label_encoder.transform(labels)

    def decode_labels(self, encoded_labels):
        if self.classes_ is None:
            raise ValueError('Label encoder not fitted.')
        return self.label_encoder.inverse_transform(np.asarray(encoded_labels).ravel())

    def fit(self, X: dict, y):
        y_encoded = self._encode_labels(y, fit=True)
        y_tensor = torch.as_tensor(y_encoded, dtype=torch.long)

        dataset = DictDataset(X, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if not self.model.initialized:
            sample_item = dataset[0][0]
            init_x = {k: v.unsqueeze(0).to(self.device) for k, v in sample_item.items()}
            self.model.init_layers(init_x)

        self.optimizer = self.optimizer_cls(self.model.parameters())

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = {k: v.to(self.device) for k, v in xb.items()}
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.loss_fn(out, yb)
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X: dict):
        self.model.eval()

        dataset = DictDataset(X)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        predictions_list = []
        with torch.no_grad():
            for xb in loader:
                xb = {k: v.to(self.device) for k, v in xb.items()}

                out = self.model(xb)
                preds = torch.argmax(out, dim=1).cpu().numpy()
                predictions_list.append(preds)

        return self.decode_labels(np.concatenate(predictions_list))

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'state_dict': self.model.state_dict(),
                'label_encoder': self.label_encoder,
                'classes_': self.classes_,
            },
            path / 'model.pt',
        )

    @classmethod
    def load(
        cls,
        path,
        model_factory,
        device,
        optimizer_cls,
        loss_fn,
        epochs=20,
        batch_size=64,
    ):
        path = Path(path)
        checkpoint = torch.load(path / 'model.pt', map_location=device)
        model = model_factory()
        model.load_state_dict(checkpoint['state_dict'])

        wrapper = cls(
            model,
            optimizer_cls,
            loss_fn,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
        wrapper.label_encoder = checkpoint.get('label_encoder', LabelEncoder())
        wrapper.classes_ = checkpoint.get('classes_')
        return wrapper
