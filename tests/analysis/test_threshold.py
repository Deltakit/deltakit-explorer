import matplotlib.pyplot as plt
import numpy as np
import pytest

from deltakit_explorer.analysis import estimate_threshold
from deltakit_explorer.plotting import plot_threshold

P_TH, NU = 0.01, 1.2


def _synthetic(distances, ps, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = (ps[None, :] - P_TH) * np.asarray(distances)[:, None] ** (1 / NU)
    y = 0.3 + 4 * x + 20 * x**2  # all curves cross exactly at P_TH
    return y + rng.normal(0, noise, y.shape) if noise else y


def test_recovers_threshold_and_exponent_from_noiseless_data() -> None:
    distances, ps = [3, 5, 7, 9], np.linspace(0.008, 0.012, 9)
    res = estimate_threshold(distances, ps, _synthetic(distances, ps))
    assert res.threshold.value == pytest.approx(P_TH, rel=1e-3)
    assert res.nu.value == pytest.approx(NU, rel=1e-2)
    assert len(res.crossings) == 3
    assert all(c == pytest.approx(P_TH, rel=1e-6) for c in res.crossings)


def test_recovers_threshold_from_noisy_weighted_data() -> None:
    distances, ps = [3, 5, 7], np.linspace(0.008, 0.012, 9)
    y = _synthetic(distances, ps, noise=0.005)
    res = estimate_threshold(distances, ps, y, np.full_like(y, 0.005))
    assert abs(res.threshold.value - P_TH) < 4 * res.threshold.stddev
    assert res.threshold.stddev > 0


@pytest.mark.parametrize(
    ("distances", "ps"), [([3], np.linspace(0.008, 0.012, 9)), ([3, 5], [0.01] * 4)]
)
def test_too_little_data_raises(distances, ps) -> None:
    with pytest.raises(ValueError, match="At least"):
        estimate_threshold(distances, ps, np.zeros((len(distances), len(ps))))


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        estimate_threshold([3, 5], np.linspace(0.008, 0.012, 9), np.zeros((2, 4)))


def test_plot_threshold_draws_one_curve_per_distance() -> None:
    distances, ps = [3, 5], np.linspace(0.008, 0.012, 9)
    y = _synthetic(distances, ps)
    _, ax = plot_threshold(distances, ps, y, estimate_threshold(distances, ps, y))
    assert len(ax.lines) == 3  # two curves + threshold line
    plt.close("all")


def test_threshold_outside_data_range_warns() -> None:
    distances, ps = [3, 5, 7], np.linspace(0.02, 0.03, 9)  # all above P_TH
    with pytest.warns(UserWarning, match="outside"):
        estimate_threshold(distances, ps, _synthetic(distances, ps) + 0.01 * ps)
