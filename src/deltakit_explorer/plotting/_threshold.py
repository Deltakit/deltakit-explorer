# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Plotting helpers for threshold estimates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis._threshold import ThresholdData
from deltakit_explorer.plotting._utils import get_figure_and_axes


def plot_threshold(
    distances: Sequence[int],
    physical_error_rates: Sequence[float] | npt.NDArray[np.floating],
    logical_error_rates: Sequence[Sequence[float]] | npt.NDArray[np.floating],
    threshold: ThresholdData | None = None,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot logical error rate against physical error rate, one curve per distance.

    Args:
        distances: code distances, one per row of ``logical_error_rates``.
        physical_error_rates: physical error rates shared by all distances.
        logical_error_rates: logical error rates of shape
            ``(len(distances), len(physical_error_rates))``.
        threshold: if provided, the estimated threshold is drawn as a vertical line
            with a band of one standard deviation.
        fig: A matplotlib Figure object to plot on. If None, a new figure
            will be created. Default is None.
        ax: A matplotlib Axes object to plot on. If None, a new axes will
            be created. Default is None.
        title: An optional custom title for the plot.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Examples:

        Estimating and plotting a threshold::

            from deltakit_explorer.analysis import estimate_threshold
            from deltakit_explorer.plotting import plot_threshold

            result = estimate_threshold(distances, ps, leps)
            fig, ax = plot_threshold(distances, ps, leps, result)

    """
    fig, ax = get_figure_and_axes(fig, ax)
    for d, y in zip(distances, np.asarray(logical_error_rates)):
        ax.plot(physical_error_rates, y, marker="o", label=f"d = {d}")
    if threshold is not None:
        value, stddev = threshold.threshold
        ax.axvline(
            value, color="black", linestyle="--", label=f"threshold = {value:.4g}"
        )
        ax.axvspan(value - stddev, value + stddev, color="black", alpha=0.15)
    ax.set_xlabel("Physical error rate")
    ax.set_ylabel("Logical error rate")
    ax.set_title(title or "Threshold")
    ax.legend()
    return fig, ax
