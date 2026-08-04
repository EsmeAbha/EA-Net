"""Compact EEG encoders.

Both backbones are BatchNorm-heavy by design. That is not incidental: TENT and
the norm-recalibration baseline adapt precisely the BatchNorm statistics and
affine parameters, so a backbone without BN would leave those methods nothing
to work with.

Inputs are ``(batch, n_channels, n_times)``; a singleton conv dimension is
added internally.
"""

from __future__ import annotations

import torch
from torch import nn


class _Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class _SafeLog(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.clamp(x, min=1e-6))


class EEGNet(nn.Module):
    """EEGNet v4 (Lawhern et al., 2018).

    Temporal convolution, depthwise spatial filtering per temporal filter, then
    a separable convolution. Roughly 2k parameters — small enough that
    cross-subject overfitting stays manageable.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        f1: int = 8,
        depth: int = 2,
        f2: int | None = None,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        f2 = f2 or f1 * depth

        self.block1 = nn.Sequential(
            # Temporal filtering. Padding keeps the time axis length intact.
            nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            # Depthwise spatial filter: learns one spatial pattern per temporal filter.
            nn.Conv2d(f1, f1 * depth, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            # Separable conv = depthwise temporal followed by pointwise mixing.
            nn.Conv2d(
                f1 * depth, f1 * depth, (1, 16), padding=(0, 8), groups=f1 * depth, bias=False
            ),
            nn.Conv2d(f1 * depth, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(self._feature_dim(n_channels, n_times), n_classes)

    def _feature_dim(self, n_channels: int, n_times: int) -> int:
        """Infer the flattened feature width with a dry run."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.flatten(self.block2(self.block1(dummy)))
        self.train(was_training)
        return out.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block2(self.block1(x))
        return self.classifier(self.flatten(x))


class ShallowNet(nn.Module):
    """Shallow ConvNet (Schirrmeister et al., 2017).

    Square -> average-pool -> log reproduces the log-band-power pipeline of
    FBCSP, which is why it stays a strong baseline on motor imagery despite
    its size.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int = 2,
        n_filters: int = 40,
        filter_time_length: int = 25,
        pool_time_length: int = 75,
        pool_time_stride: int = 15,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, filter_time_length), bias=False),
            nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            _Square(),
            nn.AvgPool2d((1, pool_time_length), stride=(1, pool_time_stride)),
            _SafeLog(),
            nn.Dropout(dropout),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(self._feature_dim(n_channels, n_times), n_classes)

    def _feature_dim(self, n_channels: int, n_times: int) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.flatten(self.features(dummy))
        self.train(was_training)
        return out.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.classifier(self.flatten(self.features(x)))


_REGISTRY = {"eegnet": EEGNet, "shallow": ShallowNet}


def build_model(
    name: str, n_channels: int, n_times: int, n_classes: int = 2, **kwargs
) -> nn.Module:
    """Instantiate a backbone by name."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](
        n_channels=n_channels, n_times=n_times, n_classes=n_classes, **kwargs
    )
