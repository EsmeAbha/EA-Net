"""End-to-end smoke tests on synthetic data — small enough to run in CI."""

import copy

import pytest

from eanet.cli import main
from eanet.config import load_config
from eanet.experiment import build_data, run_experiment

TINY = {
    "seed": 0,
    "device": "cpu",
    "data": {"source": "synthetic", "n_trials": 64},
    "model": {"name": "eegnet", "dropout": 0.25},
    "train": {"epochs": 2, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4},
    "shift": {"kind": "none", "severity": 1},
    "tta": {"method": "source", "lr": 2.5e-4, "steps": 1, "episodic": False},
}


def test_build_data_shapes():
    data = build_data(TINY)
    assert data["x_train"].ndim == 3
    assert len(data["x_train"]) == len(data["y_train"])
    assert data["x_train"].shape[1:] == data["x_test"].shape[1:]


@pytest.mark.parametrize("method", ["source", "norm", "tent", "sar"])
def test_experiment_runs_for_every_method(method):
    cfg = copy.deepcopy(TINY)
    cfg["tta"]["method"] = method
    cfg["shift"] = {"kind": "noise", "severity": 3}

    results, model = run_experiment(cfg)

    assert 0.0 <= results["accuracy"] <= 1.0
    assert 0.0 <= results["balanced_accuracy"] <= 1.0
    assert results["method"] == method
    assert results["shift"] == "noise"
    assert model is not None


def test_source_model_learns_the_synthetic_task():
    """Sanity check on the generator: the task must be learnable at all."""
    cfg = copy.deepcopy(TINY)
    cfg["train"]["epochs"] = 15
    cfg["data"]["n_trials"] = 256
    results, _ = run_experiment(cfg)
    assert results["accuracy"] > 0.7


def test_default_config_loads_and_overrides_apply():
    cfg = load_config(overrides=["tta.method=tent", "train.epochs=3"])
    assert cfg["tta"]["method"] == "tent"
    assert cfg["train"]["epochs"] == 3
    assert isinstance(cfg["train"]["epochs"], int)


def test_cli_shifts_command(capsys):
    assert main(["shifts"]) == 0
    assert "channel_dropout" in capsys.readouterr().out


def test_cli_eval_on_synthetic(tmp_path, capsys):
    out = tmp_path / "results.json"
    code = main(
        [
            "eval",
            "--set", "data.source=synthetic",
            "--set", "data.n_trials=64",
            "--set", "train.epochs=2",
            "--set", "train.batch_size=16",
            "--set", "shift.kind=noise",
            "--set", "tta.method=norm",
            "--out", str(out),
        ]
    )
    assert code == 0
    assert out.exists()
    assert "accuracy" in out.read_text(encoding="utf-8")
