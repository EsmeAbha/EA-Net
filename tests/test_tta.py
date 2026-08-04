import copy

import numpy as np
import pytest
import torch

from eanet.models import build_model
from eanet.tta import build_tta, softmax_entropy
from eanet.tta.base import collect_bn_params

METHODS = ["source", "norm", "tent", "sar"]


@pytest.fixture
def model():
    torch.manual_seed(0)
    return build_model("eegnet", n_channels=8, n_times=128, n_classes=2)


@pytest.fixture
def batch():
    torch.manual_seed(1)
    return torch.randn(16, 8, 128)


@pytest.mark.parametrize("method", METHODS)
def test_output_shape(model, batch, method):
    adapter = build_tta(method, copy.deepcopy(model))
    assert adapter(batch).shape == (16, 2)


@pytest.mark.parametrize("method", ["tent", "sar"])
def test_adaptation_changes_bn_parameters(model, batch, method):
    # margin=1.0 lets SAR accept samples up to ln(n_classes); the default 0.4
    # would reject everything from an untrained model, which is by design.
    adapter = build_tta(method, copy.deepcopy(model), lr=1e-2, margin=1.0)
    before = [p.detach().clone() for p in collect_bn_params(adapter.model)]
    adapter(batch)
    after = collect_bn_params(adapter.model)
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_sar_skips_adaptation_when_all_samples_are_unreliable(model, batch):
    """SAR's reliability filter must be able to reject an entire batch.

    An untrained model predicts near-uniformly, so with the default margin
    every sample sits above the entropy threshold and nothing should update.
    """
    adapter = build_tta("sar", copy.deepcopy(model), lr=1e-2, margin=0.01)
    before = [p.detach().clone() for p in collect_bn_params(adapter.model)]
    adapter(batch)
    after = collect_bn_params(adapter.model)
    assert all(torch.allclose(b, a) for b, a in zip(before, after))


def test_source_leaves_model_untouched(model, batch):
    adapter = build_tta("source", copy.deepcopy(model))
    before = copy.deepcopy(adapter.model.state_dict())
    adapter(batch)
    after = adapter.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


@pytest.mark.parametrize("method", ["tent", "sar"])
def test_reset_restores_source_state(model, batch, method):
    adapter = build_tta(method, copy.deepcopy(model), lr=1e-2)
    original = copy.deepcopy(adapter.model.state_dict())
    adapter(batch)
    adapter.reset()
    restored = adapter.model.state_dict()
    assert all(torch.equal(original[k], restored[k]) for k in original)


@pytest.mark.parametrize("method", ["tent", "sar"])
def test_episodic_mode_does_not_accumulate(model, batch, method):
    adapter = build_tta(method, copy.deepcopy(model), lr=1e-2, margin=1.0, episodic=True)
    first = adapter(batch)
    second = adapter(batch)
    # Reset before each batch plus a dropout-free prediction pass means the same
    # input must yield exactly the same output.
    assert torch.allclose(first, second, atol=1e-5)


@pytest.mark.parametrize("method", METHODS)
def test_predictions_are_deterministic(model, batch, method):
    """Repeated prediction on one batch must not vary with dropout sampling."""
    adapter = build_tta(method, copy.deepcopy(model), lr=1e-2, episodic=True)
    assert torch.allclose(adapter(batch), adapter(batch), atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_outputs_stay_finite(model, batch, method):
    adapter = build_tta(method, copy.deepcopy(model), lr=1e-2)
    for _ in range(3):
        assert torch.isfinite(adapter(batch)).all()


def test_tent_reduces_entropy(model, batch):
    """The whole premise of TENT — if this fails the optimiser is not working."""
    adapter = build_tta("tent", copy.deepcopy(model), lr=1e-2, steps=1)
    with torch.no_grad():
        start = softmax_entropy(adapter.model(batch)).mean().item()
    for _ in range(10):
        adapter(batch)
    with torch.no_grad():
        end = softmax_entropy(adapter.model(batch)).mean().item()
    assert end < start


def test_softmax_entropy_bounds():
    uniform = torch.zeros(4, 2)
    confident = torch.tensor([[50.0, -50.0]] * 4)
    assert torch.allclose(softmax_entropy(uniform), torch.full((4,), np.log(2)), atol=1e-5)
    assert softmax_entropy(confident).max().item() < 1e-5


def test_unknown_method_raises(model):
    with pytest.raises(KeyError):
        build_tta("magic", model)
