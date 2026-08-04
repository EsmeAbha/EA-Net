import numpy as np
import pytest

from eanet.data.shifts import SHIFTS, apply_shift
from eanet.data.synthetic import make_synthetic


@pytest.fixture
def x():
    data, _ = make_synthetic(n_trials=16, n_channels=8, n_times=128, seed=0)
    return data


@pytest.mark.parametrize("kind", sorted(SHIFTS))
def test_shape_and_finiteness_preserved(x, kind):
    out = apply_shift(x, kind=kind, severity=3, seed=0)
    assert out.shape == x.shape
    assert np.isfinite(out).all()


@pytest.mark.parametrize("kind", sorted(set(SHIFTS) - {"none"}))
def test_shift_actually_changes_data(x, kind):
    out = apply_shift(x, kind=kind, severity=3, seed=0)
    assert not np.allclose(out, x)


def test_input_is_not_mutated(x):
    before = x.copy()
    apply_shift(x, kind="noise", severity=5, seed=0)
    np.testing.assert_array_equal(x, before)


def test_same_seed_is_reproducible(x):
    a = apply_shift(x, kind="noise", severity=3, seed=7)
    b = apply_shift(x, kind="noise", severity=3, seed=7)
    np.testing.assert_array_equal(a, b)


def test_severity_is_monotonic(x):
    """Higher severity must move the data further from the original."""
    distances = [
        float(np.linalg.norm(apply_shift(x, kind="noise", severity=s, seed=0) - x))
        for s in (1, 3, 5)
    ]
    assert distances[0] < distances[1] < distances[2]


def test_channel_dropout_zeroes_channels(x):
    out = apply_shift(x, kind="channel_dropout", severity=3, seed=0)
    dead = [(out[t] == 0).all(axis=-1).sum() for t in range(len(out))]
    assert all(d >= 1 for d in dead)


def test_montage_keeps_a_fixed_channel_set(x):
    out = apply_shift(x, kind="montage", severity=3, seed=0)
    # The same channels must survive in every trial, unlike channel_dropout.
    alive = (np.abs(out).sum(axis=-1) > 0)
    assert (alive == alive[0]).all()


def test_unknown_shift_raises(x):
    with pytest.raises(KeyError):
        apply_shift(x, kind="does_not_exist")


def test_bad_severity_raises(x):
    with pytest.raises(ValueError):
        apply_shift(x, kind="noise", severity=9)


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        apply_shift(np.zeros((4, 8)), kind="noise")
