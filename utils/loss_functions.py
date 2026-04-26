import torch
import torch.nn as nn


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        x = pred - target
        # Numerically stable identity: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
        return torch.mean(
            torch.abs(x)
            + torch.nn.functional.softplus(-2.0 * torch.abs(x))
            - torch.log(torch.tensor(2.0))
        )
