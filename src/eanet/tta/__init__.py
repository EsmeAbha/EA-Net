"""Test-time adaptation methods."""

from __future__ import annotations

from torch import nn

from eanet.tta.base import TTAMethod, softmax_entropy
from eanet.tta.norm import NormAdapt
from eanet.tta.sar import SAR, SAM
from eanet.tta.source import Source
from eanet.tta.tent import Tent

__all__ = [
    "SAM",
    "SAR",
    "NormAdapt",
    "Source",
    "TTAMethod",
    "Tent",
    "build_tta",
    "softmax_entropy",
]

_REGISTRY = {
    "source": Source,
    "norm": NormAdapt,
    "tent": Tent,
    "sar": SAR,
}


def build_tta(name: str, model: nn.Module, **kwargs) -> TTAMethod:
    """Instantiate a TTA method by name, ignoring kwargs it does not accept.

    Keeps one config block usable across every method — `source` and `norm`
    simply drop the optimiser settings the gradient-based methods need.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown TTA method {name!r}; available: {sorted(_REGISTRY)}")

    cls = _REGISTRY[key]
    accepted = set(cls.__init__.__code__.co_varnames)
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(model, **filtered)
