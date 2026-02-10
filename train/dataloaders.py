from torch.utils.data import DataLoader, Dataset

import torch


class ExampleDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.train = train
        self.transform = transform

        train_size, test_size = 1000, 200
        feature_size = 784
        class_num = 10

        if train:
            self.features = torch.randn(train_size, feature_size)
            self.labels = torch.randint(0, class_num, (train_size,))
        else:
            self.features = torch.randn(test_size, feature_size)
            self.labels = torch.randint(0, class_num, (test_size,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def ExampleDataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, **kwargs):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs,
    )
