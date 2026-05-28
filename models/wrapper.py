import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset


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


class TorchModelWrapper(ModelWrapper):
    def __init__(
        self, model, optimizer_cls, loss_fn, epochs=20, batch_size=64, device='cpu'
    ):
        self.model = model
        self.device = device
        self.optimizer_cls = optimizer_cls
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = None

    def fit(self, X, y):
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        y_tensor = torch.as_tensor(y)
        loader = DataLoader(
            TensorDataset(X_tensor, y_tensor), batch_size=self.batch_size, shuffle=True
        )

        self.model.to(self.device)
        self.optimizer = self.optimizer_cls(self.model.parameters())

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.loss_fn(out, yb)
                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(X_tensor)
            return torch.argmax(out, dim=1).cpu().numpy()

    def save(self, path):
        path = Path(path)
        torch.save({'state_dict': self.model.state_dict()}, path / 'model.pt')

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
        return cls(
            model,
            optimizer_cls,
            loss_fn,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
