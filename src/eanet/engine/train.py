"""Supervised training on the source domain.

Deliberately plain: the research question is what happens at test time, so the
source recipe should be a boring, reproducible baseline rather than a tuned one.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        preds = model(xb.to(device)).argmax(dim=1).cpu()
        correct += int((preds == yb).sum())
        total += len(yb)
    return correct / max(total, 1)


def train_source(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_fraction: float = 0.2,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[nn.Module, dict]:
    """Train on the source split, keeping the best-validation weights.

    Returns ``(model, history)``. The returned model carries the best
    checkpoint, not the last epoch — TTA results are noisy enough without an
    overfit starting point.
    """
    device = torch.device(device)
    model = model.to(device)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_val = max(1, int(len(y) * val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]

    train_loader = _loader(x[train_idx], y[train_idx], batch_size, shuffle=True)
    val_loader = _loader(x[val_idx], y[val_idx], batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": []}
    best_acc, best_state = -1.0, copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        scheduler.step()

        train_loss = epoch_loss / max(len(train_idx), 1)
        val_acc = _evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(model.state_dict())

        log.info("epoch %3d  loss %.4f  val_acc %.4f", epoch + 1, train_loss, val_acc)

    model.load_state_dict(best_state)
    history["best_val_acc"] = best_acc
    return model, history
