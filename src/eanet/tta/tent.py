"""TENT: fully test-time adaptation by entropy minimisation (Wang et al., ICLR 2021)."""

from __future__ import annotations

import torch
from torch import nn

from eanet.tta.base import (
    TTAMethod,
    collect_bn_params,
    configure_bn_model,
    copy_state,
    load_state,
    softmax_entropy,
)


class Tent(TTAMethod):
    """Minimise prediction entropy on the test batch, updating BN affine params only.

    The premise is that confident predictions correlate with correct ones under
    shift. That premise is load-bearing: when it fails, entropy minimisation
    happily drives the model toward confident nonsense, and the classic failure
    mode is collapse onto a single class. Watch the per-class prediction
    distribution, not just accuracy.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2.5e-4,
        steps: int = 1,
        episodic: bool = False,
    ) -> None:
        super().__init__(model, episodic=episodic)
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.steps = steps

        configure_bn_model(self.model)
        params = collect_bn_params(self.model)
        if not params:
            raise ValueError("model exposes no BatchNorm parameters; TENT has nothing to adapt")
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self._source_state = copy_state(self.model, self.optimizer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            logits = self.model(x)
            loss = softmax_entropy(logits).mean()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        # Re-predict with the updated parameters so the returned logits reflect
        # the adaptation this batch paid for.
        with torch.no_grad():
            return self.model(x)

    def reset(self) -> None:
        load_state(self.model, self._source_state, self.optimizer)
