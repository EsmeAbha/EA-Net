"""BatchNorm statistic recalibration (Nado et al. 2020; Schneider et al. 2020)."""

from __future__ import annotations

import torch
from torch import nn

from eanet.tta.base import TTAMethod, configure_bn_model, inference_pass


class NormAdapt(TTAMethod):
    """Replace source BN statistics with test-batch statistics.

    No gradients and no learned parameters — it simply recomputes normalisation
    from the incoming batch. Often recovers a surprising fraction of the gap,
    which makes it the honest baseline for judging whether an optimisation-based
    method is earning its extra cost.

    Accuracy depends on batch size: small test batches give noisy statistics.
    """

    def __init__(self, model: nn.Module, episodic: bool = False) -> None:
        super().__init__(model, episodic=episodic)
        configure_bn_model(self.model)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with inference_pass(self.model):
            return self.model(x)
