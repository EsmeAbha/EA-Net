"""Shared machinery for test-time adaptation.

Every method here follows the same contract: the model arrives trained on the
source domain, and adaptation happens during inference using unlabelled test
batches only. No test labels are ever touched — that is the whole point, and
it is the easiest thing to get wrong when wiring up an evaluation.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import torch
from torch import nn


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy of the softmax distribution.

    Computed from log-softmax rather than log(softmax) to avoid the numerical
    blow-up when a probability underflows to zero.
    """
    log_probs = logits.log_softmax(dim=1)
    return -(log_probs.exp() * log_probs).sum(dim=1)


def configure_bn_model(model: nn.Module) -> nn.Module:
    """Freeze everything except BatchNorm affine parameters.

    BN layers are switched to use *batch* statistics at test time by clearing
    their running estimates, which is what lets these methods track a shifted
    input distribution.
    """
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.requires_grad_(True)
            # Force use of batch stats regardless of train/eval mode.
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def collect_bn_params(model: nn.Module) -> list[nn.Parameter]:
    """Gather the BatchNorm scale/shift parameters that TENT and SAR update."""
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            for name in ("weight", "bias"):
                param = getattr(module, name, None)
                if param is not None:
                    params.append(param)
    return params


def copy_state(model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> dict:
    """Deep-copy model (and optimizer) state so it can be restored later."""
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()) if optimizer else None,
    }


def load_state(
    model: nn.Module, state: dict, optimizer: torch.optim.Optimizer | None = None
) -> None:
    """Restore a snapshot produced by :func:`copy_state`."""
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])


class TTAMethod(ABC, nn.Module):
    """Base class for test-time adaptation strategies.

    Args:
        model: source-trained network.
        episodic: if True, reset to the source state before every batch. Use
            this to measure single-batch adaptation in isolation; leave it off
            to let the model accumulate across the test stream (the realistic
            deployment setting, and the one where methods diverge).
    """

    def __init__(self, model: nn.Module, episodic: bool = False) -> None:
        super().__init__()
        self.model = model
        self.episodic = episodic

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for ``x``, adapting as a side effect."""

    def reset(self) -> None:
        """Restore the source state. Subclasses with state must override."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()
        return self.forward(x)
