"""No-adaptation baseline."""

from __future__ import annotations

import torch
from torch import nn

from eanet.tta.base import TTAMethod


class Source(TTAMethod):
    """Frozen source model evaluated in eval mode.

    This is the number every other method has to beat. Reporting TTA results
    without it is meaningless.
    """

    def __init__(self, model: nn.Module, episodic: bool = False) -> None:
        super().__init__(model, episodic=episodic)
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
