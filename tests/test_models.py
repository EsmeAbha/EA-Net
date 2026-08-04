import pytest
import torch
from torch import nn

from eanet.models import build_model
from eanet.utils import count_params


@pytest.mark.parametrize("name", ["eegnet", "shallow"])
def test_forward_shape(name):
    model = build_model(name, n_channels=8, n_times=256, n_classes=2)
    out = model(torch.randn(4, 8, 256))
    assert out.shape == (4, 2)


@pytest.mark.parametrize("name", ["eegnet", "shallow"])
def test_backward_reaches_every_parameter(name):
    model = build_model(name, n_channels=8, n_times=256, n_classes=2)
    model(torch.randn(4, 8, 256)).sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


@pytest.mark.parametrize("name", ["eegnet", "shallow"])
def test_exposes_batchnorm(name):
    """TENT/SAR/norm all adapt BatchNorm; a backbone without it is unusable here."""
    model = build_model(name, n_channels=8, n_times=256, n_classes=2)
    assert any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in model.modules())


@pytest.mark.parametrize("n_times", [128, 256, 384])
def test_handles_varying_input_length(n_times):
    model = build_model("eegnet", n_channels=8, n_times=n_times, n_classes=2)
    assert model(torch.randn(2, 8, n_times)).shape == (2, 2)


def test_models_stay_small():
    model = build_model("eegnet", n_channels=64, n_times=384, n_classes=2)
    assert count_params(model) < 20_000


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("transformer", n_channels=8, n_times=128)
