import torch
import torch.nn as nn


class GaussianLogLikelihood(nn.Module):
    """
    Wrapper for torch.nn.GaussianNLLLoss when the network outputs
    (mean, standard_deviation) along the channel dimension.

    predicted: (N, 2, ...)
        channel 0 → mean
        channel 1 → std
    """

    def __init__(self, reduction="mean", eps=1e-6):
        super().__init__()
        self.loss = nn.GaussianNLLLoss(reduction=reduction, eps=eps)
        self.eps = eps

    def forward(self, observed, predicted):

        mean = predicted[:, 0, ...]
        std = predicted[:, 1, ...]

        # ensure strictly positive variance
        var = torch.clamp(std, min=self.eps) ** 2

        return self.loss(mean, observed, var)
