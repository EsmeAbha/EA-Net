"""Determinism and device helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn a config device string into a real device, falling back to CPU."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but no CUDA device is available")
    return torch.device(spec)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
