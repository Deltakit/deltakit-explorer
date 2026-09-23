# (c) Copyright Riverlane 2020-2026. All rights reserved.

from __future__ import annotations

import logging
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.enums._basic_enums import THRESHOLD_DISTANCE_COLORS

if sys.platform == "darwin":
    try:
        mpl.use("MacOSX")
    except Exception:
        mpl.use("Agg")
else:
    mpl.use("Agg")

logger = logging.getLogger(__name__)


def plot_threshold(
    data_dict: dict[int, tuple[list[float], list[float], list[float]]],
    estimated_threshold: float,
    threshold_error: float = 0.0005,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Generates a log-log threshold plot with an inset zoom and uncertainty bounds.

    Args:
        data_dict: Dictionary mapping distances to (p_vals, lep_vals, lep_errors).
        estimated_threshold: The calculated asymptotic threshold value.
        threshold_error: The statistical uncertainty of the threshold.
        fig: A matplotlib Figure object to plot on. If None, a new figure will be created.
        ax: A matplotlib Axes object to plot on. If None, a new axes will be created.
        title: An optional custom title for the plot. If None, a default title is used.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    if title is None:
        title = "Surface Code Threshold Crossing"

    # --- Main Plot ---
    for distance, data_tuple in sorted(data_dict.items()):
        p_vals = data_tuple[0]
        lep_vals = data_tuple[1]
        lep_errors = data_tuple[2] if len(data_tuple) > 2 else np.zeros_like(p_vals)

        color = THRESHOLD_DISTANCE_COLORS.get(distance, "black")
        ax.errorbar(
            p_vals,
            lep_vals,
            yerr=lep_errors,
            marker="o",
            linestyle="-",
            color=color,
            markersize=6,
            capsize=3,
            label=f"d = {distance}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate (p)", fontsize=12)
    ax.set_ylabel("Logical Error Probability (LEP)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", ls="--", color="grey", alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)

    # --- Inset Zoom ---
    ax_ins = ax.inset_axes([0.52, 0.08, 0.45, 0.35])

    for distance, data_tuple in sorted(data_dict.items()):
        p_vals = data_tuple[0]
        lep_vals = data_tuple[1]
        lep_errors = data_tuple[2] if len(data_tuple) > 2 else np.zeros_like(p_vals)

        color = THRESHOLD_DISTANCE_COLORS.get(distance, "black")
        ax_ins.errorbar(
            p_vals,
            lep_vals,
            yerr=lep_errors,
            marker="o",
            linestyle="-",
            color=color,
            markersize=4,
            capsize=2,
        )

    zoom_xlim = [estimated_threshold * 0.85, estimated_threshold * 1.15]
    all_lep_near_threshold = [
        lep
        for data_tuple in data_dict.values()
        for p, lep in zip(data_tuple[0], data_tuple[1])
        if zoom_xlim[0] <= p <= zoom_xlim[1]
    ]

    zoom_ylim = (
        [min(all_lep_near_threshold) * 0.8, max(all_lep_near_threshold) * 1.2]
        if all_lep_near_threshold
        else [1e-3, 1e-1]
    )

    ax_ins.set_xscale("log")
    ax_ins.set_yscale("log")
    ax_ins.set_xlim(zoom_xlim)
    ax_ins.set_ylim(zoom_ylim)
    ax_ins.grid(True, which="both", ls="--", color="grey", alpha=0.3)

    ax_ins.axvline(
        x=estimated_threshold,
        color="#b2182b",
        linestyle="--",
        label=f"Threshold: {estimated_threshold:.5f}",
    )

    if threshold_error:
        ax_ins.axvspan(
            estimated_threshold - threshold_error,
            estimated_threshold + threshold_error,
            color="#b2182b",
            alpha=0.15,
            label=r"$\pm$ Uncertainty",
        )

    ax_ins.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)
    ax.indicate_inset_zoom(ax_ins, edgecolor="gray")

    fig.tight_layout()

    return fig, ax
