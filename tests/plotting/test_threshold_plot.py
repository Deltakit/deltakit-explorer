# (c) Copyright Riverlane 2020-2026. All rights reserved.
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from deltakit_explorer.plotting._threshold import (
    _crossing_label,
    _fit_power_law,
    _sci_tick,
    plot_threshold,
)
from deltakit_explorer.plotting.plotting import ThresholdData, plot

# Use non-interactive backend for CI
mpl.use("Agg")

HISTORY = {
    3: ([0.003, 0.004, 0.005], [0.030, 0.050, 0.080], [0.001, 0.001, 0.001]),
    5: ([0.003, 0.004, 0.005], [0.025, 0.051, 0.100], [0.001, 0.001, 0.001]),
    7: ([0.003, 0.004, 0.005], [0.020, 0.052, 0.120], [0.001, 0.001, 0.001]),
}
CROSSINGS = {(3, 5): 0.0039, (5, 7): 0.0041}


def _star_points(ax: plt.Axes) -> list[tuple[float, float]]:
    return [
        (float(line.get_xdata()[0]), float(line.get_ydata()[0]))
        for line in ax.get_lines()
        if line.get_marker() == "*"
    ]


class TestPlotThreshold:
    def test_marks_each_crossing_on_main_axes(self) -> None:
        _, ax = plot_threshold(HISTORY, CROSSINGS)
        xs = sorted(x for x, _ in _star_points(ax))
        assert xs == pytest.approx(sorted(CROSSINGS.values()))
        plt.close("all")

    def test_crossing_marker_sits_between_the_two_curves(self) -> None:
        _, ax = plot_threshold(HISTORY, {(3, 5): 0.0039})
        ((_, y),) = _star_points(ax)
        assert 0.040 < y < 0.055
        plt.close("all")

    def test_inset_zooms_on_crossings_and_labels_them(self) -> None:
        _, ax = plot_threshold(HISTORY, CROSSINGS)
        (ax_ins,) = ax.child_axes
        x_min, x_max = ax_ins.get_xlim()
        assert x_min <= min(CROSSINGS.values()) <= max(CROSSINGS.values()) <= x_max
        labels = [t.get_text() for t in ax_ins.get_legend().get_texts()]
        assert labels == [
            r"d = (3, 5): $p = 3.900 \times 10^{-3}$",
            r"d = (5, 7): $p = 4.100 \times 10^{-3}$",
        ]
        plt.close("all")

    def test_points_are_unjoined_diamonds_with_fit_line(self) -> None:
        _, ax = plot_threshold(HISTORY, {})
        diamonds = [line for line in ax.get_lines() if line.get_marker() == "D"]
        assert len(diamonds) == len(HISTORY)
        assert all(line.get_linestyle() == "None" for line in diamonds)
        fits = [
            line
            for line in ax.get_lines()
            if line.get_marker() in ("None", "") and line.get_linestyle() == "-"
        ]
        assert len(fits) == len(HISTORY)
        plt.close("all")

    def test_fit_recovers_exact_power_law(self) -> None:
        p = [0.002, 0.004, 0.008]
        data = (p, [3.0 * x**2 for x in p], [1e-6, 1e-6, 1e-6])
        log_a, slope = _fit_power_law(data)
        assert slope == pytest.approx(2.0)
        assert np.exp(log_a) == pytest.approx(3.0)

    def test_fit_needs_two_distinct_points(self) -> None:
        assert _fit_power_law(([0.003, 0.003], [0.1, 0.1], [0.01, 0.01])) is None

    def test_close_crossings_get_one_inset_tick(self) -> None:
        _, ax = plot_threshold(HISTORY, {(3, 5): 0.0041, (5, 7): 0.00412})
        (ax_ins,) = ax.child_axes
        assert len(ax_ins.get_xticks()) == 1
        plt.close("all")

    def test_crossing_errors_draw_error_bars_and_label(self) -> None:
        _, ax = plot_threshold(HISTORY, {(3, 5): 0.0039}, {(3, 5): 0.0002})
        (ax_ins,) = ax.child_axes
        assert [t.get_text() for t in ax_ins.get_legend().get_texts()] == [
            r"d = (3, 5): $p = (3.9 \pm 0.2) \times 10^{-3}$"
        ]
        x_min, x_max = ax_ins.get_xlim()
        assert x_min <= 0.0037
        assert x_max >= 0.0041
        caps = [line for line in ax.get_lines() if line.get_marker() == "|"]
        assert len(caps) == 2
        plt.close("all")

    def test_no_crossings_draws_curves_without_inset(self) -> None:
        _, ax = plot_threshold(HISTORY, {})
        assert not ax.child_axes
        assert not _star_points(ax)
        plt.close("all")

    def test_dispatches_from_plot(self) -> None:
        fig, ax = plot(ThresholdData(history_data=HISTORY, crossing_points=CROSSINGS))
        assert len(_star_points(ax)) == len(CROSSINGS)
        assert np.isfinite(ax.get_xlim()).all()
        plt.close(fig)


@pytest.mark.parametrize(
    ("p", "error", "expected"),
    [
        (0.0037773, 3.5e-5, r"$p = (3.78 \pm 0.04) \times 10^{-3}$"),
        (0.0038125, 5.625e-4, r"$p = (3.8 \pm 0.6) \times 10^{-3}$"),
        (0.0038125, 1.2e-4, r"$p = (3.81 \pm 0.12) \times 10^{-3}$"),
        (0.0123, None, r"$p = 1.230 \times 10^{-2}$"),
    ],
)
def test_crossing_label_scientific_notation(
    p: float, error: float | None, expected: str
) -> None:
    assert _crossing_label((3, 5), p, error) == f"d = (3, 5): {expected}"


def test_inset_ticks_use_scientific_notation() -> None:
    _, ax = plot_threshold(HISTORY, CROSSINGS)
    (ax_ins,) = ax.child_axes
    assert [t.get_text() for t in ax_ins.get_xticklabels()] == [
        _sci_tick(0.0039),
        _sci_tick(0.0041),
    ]
    assert _sci_tick(0.0039) == r"$3.90 \times 10^{-3}$"
    plt.close("all")
