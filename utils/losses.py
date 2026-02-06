import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """Weighted Mean Squared Error Loss."""

    def __init__(self, weights=None, reduction='mean'):
        super().__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(self, inputs, targets):
        mse = (inputs - targets) ** 2

        if self.weights is not None:
            mse = mse * self.weights

        if self.reduction == 'mean':
            return mse.mean()
        elif self.reduction == 'sum':
            return mse.sum()
        else:
            return mse
