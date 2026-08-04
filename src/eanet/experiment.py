"""Wire config -> data -> source training -> shifted evaluation."""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch

from eanet.data.shifts import apply_shift
from eanet.data.synthetic import make_synthetic
from eanet.engine.train import train_source
from eanet.eval.harness import evaluate_tta
from eanet.models import build_model
from eanet.utils import resolve_device, set_seed

log = logging.getLogger(__name__)


def build_data(cfg: dict) -> dict:
    """Materialise train/test arrays according to ``cfg['data']``.

    Synthetic mode splits by RNG seed; physionet mode splits by *subject*,
    which is the shift that actually matters for EEG.
    """
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "synthetic")

    if source == "synthetic":
        n_trials = int(data_cfg.get("n_trials", 512))
        x_train, y_train = make_synthetic(n_trials=n_trials, seed=cfg.get("seed", 0))
        x_test, y_test = make_synthetic(n_trials=n_trials // 2, seed=cfg.get("seed", 0) + 1000)
        groups = None
    elif source == "physionet":
        from eanet.data.physionet import load_split

        common = {
            "runs": data_cfg.get("runs"),
            "tmin": data_cfg.get("tmin", 0.0),
            "tmax": data_cfg.get("tmax", 3.0),
            "resample_hz": data_cfg.get("resample_hz", 128.0),
            "bandpass": tuple(data_cfg["bandpass"]) if data_cfg.get("bandpass") else None,
            "cache_dir": data_cfg.get("cache_dir"),
        }
        x_train, y_train, _ = load_split(data_cfg["subjects_train"], **common)
        x_test, y_test, groups = load_split(data_cfg["subjects_test"], **common)

        # Subjects can differ in epoch length after resampling; align to the
        # shorter of the two splits so the model input width is consistent.
        n_times = min(x_train.shape[-1], x_test.shape[-1])
        x_train, x_test = x_train[..., :n_times], x_test[..., :n_times]
    else:
        raise KeyError(f"unknown data source {source!r}; expected synthetic or physionet")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "groups": groups,
    }


def run_experiment(cfg: dict, model: torch.nn.Module | None = None) -> dict:
    """Train on source (unless a model is supplied), then evaluate under shift."""
    set_seed(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device", "auto"))

    data = build_data(cfg)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    n_channels, n_times = x_train.shape[1], x_train.shape[2]
    n_classes = int(y_train.max()) + 1

    history: dict = {}
    if model is None:
        model = build_model(
            cfg["model"]["name"],
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            dropout=cfg["model"].get("dropout", 0.25),
        )
        model, history = train_source(
            model,
            x_train,
            y_train,
            epochs=cfg["train"]["epochs"],
            batch_size=cfg["train"]["batch_size"],
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
            device=device,
            seed=cfg.get("seed", 0),
        )

    shift_cfg = cfg.get("shift", {})
    x_shifted = apply_shift(
        x_test,
        kind=shift_cfg.get("kind", "none"),
        severity=int(shift_cfg.get("severity", 2)),
        seed=cfg.get("seed", 0),
    ).astype(np.float32)

    tta_cfg = cfg.get("tta", {})
    # Adaptation mutates the model, so every method starts from its own copy.
    results = evaluate_tta(
        copy.deepcopy(model),
        x_shifted,
        y_test,
        method=tta_cfg.get("method", "source"),
        batch_size=cfg["train"]["batch_size"],
        device=device,
        seed=cfg.get("seed", 0),
        groups=data["groups"],
        lr=tta_cfg.get("lr", 2.5e-4),
        steps=int(tta_cfg.get("steps", 1)),
        episodic=bool(tta_cfg.get("episodic", False)),
        margin=tta_cfg.get("sar_margin", 0.4),
    )
    results["shift"] = shift_cfg.get("kind", "none")
    results["severity"] = int(shift_cfg.get("severity", 2))
    if history:
        results["best_val_acc"] = history.get("best_val_acc")
    return results, model


def run_sweep(cfg: dict, shifts: list[str], methods: list[str], severities: list[int]) -> list[dict]:
    """Grid over shift x severity x method, reusing one source model throughout.

    Training once is the point: differences between rows then come from
    adaptation alone, not from separate training runs.
    """
    base = copy.deepcopy(cfg)
    base["shift"] = {"kind": "none", "severity": 1}
    base["tta"] = {**cfg.get("tta", {}), "method": "source"}

    _, model = run_experiment(base)

    rows = []
    for shift in shifts:
        for severity in severities:
            for method in methods:
                trial = copy.deepcopy(cfg)
                trial["shift"] = {"kind": shift, "severity": severity}
                trial["tta"] = {**cfg.get("tta", {}), "method": method}
                result, _ = run_experiment(trial, model=copy.deepcopy(model))
                rows.append(result)
                log.info(
                    "%-16s sev=%d %-7s acc=%.3f bal=%.3f collapse=%.2f",
                    shift,
                    severity,
                    method,
                    result["accuracy"],
                    result["balanced_accuracy"],
                    result["collapse_ratio"],
                )
    return rows
