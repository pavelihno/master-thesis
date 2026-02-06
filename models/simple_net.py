import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    Simple feedforward neural network for classification.

    Args:
        input_size: Size of input features
        hidden_size: Size of hidden layer
        output_size: Number of output classes
        dropout: Dropout probability
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # Flatten input if needed
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)

        return x
