"""SAR: sharpness-aware and reliable test-time adaptation (Niu et al., ICLR 2023).

SAR patches the two ways TENT falls over in the wild:

1. **Unreliable samples.** High-entropy predictions produce large, noisy
   gradients. SAR simply discards samples above an entropy threshold.
2. **Sharp minima.** Even filtered gradients can land the model somewhere
   brittle. SAR optimises with SAM, seeking flat minima that survive further
   distribution drift.

It also carries a recovery scheme: if the moving-average loss collapses, the
model is rolled back to the source state rather than left in a degenerate one.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from eanet.tta.base import (
    TTAMethod,
    collect_bn_params,
    configure_bn_model,
    copy_state,
    inference_pass,
    load_state,
    softmax_entropy,
)


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimisation wrapper (Foret et al., 2021).

    Two-phase step: ascend to the worst-case point inside an epsilon-ball
    (``first_step``), evaluate the gradient there, then apply that gradient at
    the original weights (``second_step``).
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs) -> None:
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = {"rho": rho, **kwargs}
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Remember the perturbation so second_step can undo it exactly.
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        return torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )

    def step(self, closure=None):  # pragma: no cover - SAR drives the two phases directly
        raise RuntimeError("SAM requires explicit first_step()/second_step() calls")


class SAR(TTAMethod):
    """Reliable entropy minimisation with sharpness awareness.

    Args:
        margin: entropy threshold as a multiple of ``ln(n_classes)``. The paper
            uses 0.4; lower is stricter and discards more samples.
        reset_threshold: if the EMA of the adaptation loss falls below this,
            the model is assumed to have collapsed and is reset.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 2.5e-4,
        steps: int = 1,
        episodic: bool = False,
        margin: float = 0.4,
        rho: float = 0.05,
        reset_threshold: float = 0.2,
        ema_momentum: float = 0.9,
    ) -> None:
        super().__init__(model, episodic=episodic)
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        self.steps = steps
        self.margin_ratio = margin
        self.reset_threshold = reset_threshold
        self.ema_momentum = ema_momentum
        self.ema_loss: float | None = None

        configure_bn_model(self.model)
        params = collect_bn_params(self.model)
        if not params:
            raise ValueError("model exposes no BatchNorm parameters; SAR has nothing to adapt")
        self.optimizer = SAM(params, torch.optim.SGD, rho=rho, lr=lr, momentum=0.9)
        self._source_state = copy_state(self.model)

    def _margin(self, n_classes: int) -> float:
        return self.margin_ratio * math.log(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.steps):
            self._adapt_once(x)

        if self.ema_loss is not None and self.ema_loss < self.reset_threshold:
            # Collapse guard: the loss has bottomed out, which in practice means
            # the model is predicting one class with full confidence.
            self.reset()

        with inference_pass(self.model), torch.no_grad():
            return self.model(x)

    def _adapt_once(self, x: torch.Tensor) -> None:
        # --- first pass: filter unreliable samples, then ascend ---
        logits = self.model(x)
        margin = self._margin(logits.shape[1])
        entropy = softmax_entropy(logits)
        keep = entropy < margin
        if keep.sum() == 0:
            # Every sample looked unreliable; skip rather than adapt on noise.
            self.optimizer.zero_grad(set_to_none=True)
            return

        loss = entropy[keep].mean()
        loss.backward()
        self.optimizer.first_step(zero_grad=True)

        # --- second pass: re-filter at the perturbed weights, then descend ---
        entropy2 = softmax_entropy(self.model(x))
        keep2 = keep & (entropy2 < margin)
        if keep2.sum() == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return

        loss2 = entropy2[keep2].mean()
        loss2.backward()
        self.optimizer.second_step(zero_grad=True)

        value = loss2.item()
        if not math.isnan(value):
            self.ema_loss = (
                value
                if self.ema_loss is None
                else self.ema_momentum * self.ema_loss + (1 - self.ema_momentum) * value
            )

    def reset(self) -> None:
        load_state(self.model, self._source_state)
        self.ema_loss = None
