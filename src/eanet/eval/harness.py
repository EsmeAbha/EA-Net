"""Online test-time adaptation harness.

The test set is streamed in batches exactly once, in a fixed order. Order
matters: with ``episodic=False`` the model keeps adapting across the stream, so
shuffling changes the result. The seed is therefore part of the experiment
configuration, not an implementation detail.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from eanet.eval.metrics import compute_metrics
from eanet.tta import build_tta


def _batches(n: int, batch_size: int, seed: int | None) -> list[np.ndarray]:
    order = np.arange(n)
    if seed is not None:
        np.random.default_rng(seed).shuffle(order)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


@torch.no_grad()
def _predict_probs(logits: torch.Tensor) -> np.ndarray:
    return logits.softmax(dim=1).detach().cpu().numpy()


def evaluate_tta(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    method: str = "source",
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    seed: int | None = 0,
    groups: np.ndarray | None = None,
    **tta_kwargs,
) -> dict:
    """Stream a shifted test set through a TTA method and score it.

    Args:
        model: source-trained network. It is adapted in place, so pass a fresh
            copy per method if you are comparing several.
        x: ``(n_trials, n_channels, n_times)`` test data, already shifted.
        y: integer labels, used only for scoring — never for adaptation.
        method: key from :data:`eanet.tta._REGISTRY`.
        groups: optional per-trial subject ids for a per-subject breakdown.

    Returns:
        Metric dict, plus ``per_subject`` when ``groups`` is given.
    """
    device = torch.device(device)
    model = model.to(device)
    adapter = build_tta(method, model, **tta_kwargs)

    n_classes = int(y.max()) + 1
    probs = np.zeros((len(y), n_classes), dtype=np.float64)

    for idx in _batches(len(y), batch_size, seed):
        xb = torch.from_numpy(x[idx]).float().to(device)
        # Adaptation happens inside the call; gradients are the method's business.
        logits = adapter(xb)
        probs[idx] = _predict_probs(logits)

    results = compute_metrics(y, probs, n_classes=n_classes)
    results["method"] = method
    results["n_trials"] = int(len(y))

    if groups is not None:
        per_subject = {}
        for subject in np.unique(groups):
            mask = groups == subject
            per_subject[int(subject)] = compute_metrics(
                y[mask], probs[mask], n_classes=n_classes
            )
        results["per_subject"] = per_subject

    return results
