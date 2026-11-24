# (c) Copyright Riverlane 2020-2026. All rights reserved.

from __future__ import annotations

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from deltakit_explorer.enums._basic_enums import THRESHOLD_DISTANCE_COLORS

logger = logging.getLogger(__name__)


def _sci_tick(p: float) -> str:
    r"""Formats a tick value in scientific notation, e.g. ``3.78 x 10^-3``.

    Args:
        p: The value to format.

    Returns:
        The tick label as matplotlib mathtext.
    """
    if p <= 0:
        return f"{p:g}"
    exponent = math.floor(math.log10(p))
    return rf"${p / 10**exponent:.2f} \times 10^{{{exponent}}}$"


def _crossing_label(pair: tuple[int, int], p: float, error: float | None) -> str:
    r"""Formats a crossing point for the legend in scientific notation.

    With an error the value and error share one exponent, e.g.
    ``d = (3, 5): p = (3.78 +/- 0.04) x 10^-3``, with the error rounded up to
    one significant figure (two when it starts with 1) and the value given to
    the same decimal place. Without an error the value has three decimals.

    Args:
        pair: The (d_low, d_high) code distances.
        p: The crossing physical error rate.
        error: The uncertainty on ``p``, or None.

    Returns:
        The legend label, using matplotlib mathtext for the exponent.
    """
    prefix = f"d = ({pair[0]}, {pair[1]}): "
    if p <= 0:
        return prefix + f"p = {p:g}"
    exponent = math.floor(math.log10(p))
    mantissa = p / 10**exponent
    if not error:
        return prefix + rf"$p = {mantissa:.3f} \times 10^{{{exponent}}}$"
    scaled_error = error / 10**exponent
    leading = math.floor(math.log10(scaled_error))
    sig_figs = 2 if f"{scaled_error:e}".startswith("1") else 1
    decimals = max(0, sig_figs - 1 - leading)
    # Round the error up so the quoted uncertainty is never understated.
    scaled_error = math.ceil(round(scaled_error * 10**decimals, 9)) / 10**decimals
    return prefix + (
        rf"$p = ({mantissa:.{decimals}f} \pm {scaled_error:.{decimals}f})"
        rf" \times 10^{{{exponent}}}$"
    )


def _fit_power_law(
    data_tuple: tuple[list[float], list[float], list[float]],
) -> tuple[float, float] | None:
    """Fits ``LEP = A * p**b`` to one distance's points by weighted least squares in log-log space.

    Each point is weighted by ``lep / lep_error``, the inverse of its standard
    error in log space, so well-sampled points dominate the fit. Points with a
    non-positive error rate or LEP are ignored; if any remaining error bar is
    non-positive, the fit is unweighted.

    Args:
        data_tuple: The (p_vals, lep_vals, lep_errors) history for one distance.

    Returns:
        The fitted (log_A, b) coefficients, or None if fewer than two distinct
        usable points are available.
    """
    p_vals = list(data_tuple[0])
    lep_vals = list(data_tuple[1])
    lep_errors = list(data_tuple[2]) if len(data_tuple) > 2 else [0.0] * len(p_vals)
    points = [
        (p, lep, err)
        for p, lep, err in zip(p_vals, lep_vals, lep_errors)
        if p > 0 and lep > 0
    ]
    if len({p for p, _, _ in points}) < 2:
        return None
    log_p = np.log([p for p, _, _ in points])
    log_lep = np.log([lep for _, lep, _ in points])
    errors = np.array([err for _, _, err in points])
    weights = (
        np.array([lep for _, lep, _ in points]) / errors if np.all(errors > 0) else None
    )
    slope, intercept = np.polyfit(log_p, log_lep, 1, w=weights)
    return float(intercept), float(slope)


def _lep_at(
    data_tuple: tuple[list[float], list[float], list[float]], p: float
) -> float | None:
    """Evaluates a distance's fitted power law at ``p``.

    Args:
        data_tuple: The (p_vals, lep_vals, lep_errors) history for one distance.
        p: The physical error rate to evaluate at.

    Returns:
        The fitted logical error probability, or None if no fit is possible.
    """
    fit = _fit_power_law(data_tuple)
    if fit is None:
        return None
    log_a, slope = fit
    return float(np.exp(log_a + slope * np.log(p)))


def _draw_curves(
    ax: Axes,
    data_dict: dict[int, tuple[list[float], list[float], list[float]]],
    *,
    markersize: float,
    capsize: float,
    label: bool,
) -> None:
    """Draws each code distance's points as diamonds with a power-law fit line.

    The points are not joined: they come from independent bisection steps with
    different shot counts, so joining them makes shot noise look like structure.

    Args:
        ax: The axes to draw on.
        data_dict: Dictionary mapping distances to (p_vals, lep_vals, lep_errors).
        markersize: Marker size for the data points.
        capsize: Error-bar cap size.
        label: Whether to label curves for the legend.
    """
    for distance, data_tuple in sorted(data_dict.items()):
        p_vals = data_tuple[0]
        lep_vals = data_tuple[1]
        lep_errors = data_tuple[2] if len(data_tuple) > 2 else np.zeros_like(p_vals)
        color = THRESHOLD_DISTANCE_COLORS.get(distance, "black")

        ax.errorbar(
            p_vals,
            lep_vals,
            yerr=lep_errors,
            marker="D",
            linestyle="none",
            color=color,
            markersize=markersize,
            capsize=capsize,
            label=f"d = {distance}" if label else None,
        )

        fit = _fit_power_law(data_tuple)
        if fit is not None:
            log_a, slope = fit
            positive_p = [p for p in p_vals if p > 0]
            p_grid = np.geomspace(min(positive_p), max(positive_p), 200)
            ax.plot(
                p_grid,
                np.exp(log_a + slope * np.log(p_grid)),
                linestyle="-",
                color=color,
                alpha=0.8,
            )


def _crossing_markers(
    data_dict: dict[int, tuple[list[float], list[float], list[float]]],
    crossing_points: dict[tuple[int, int], float],
) -> list[tuple[tuple[int, int], float, float | None]]:
    """Locates each crossing point on the plot.

    The marker height is the geometric mean of the two distances' fitted
    logical error probabilities at the crossing.

    Args:
        data_dict: Dictionary mapping distances to (p_vals, lep_vals, lep_errors).
        crossing_points: Mapping from (d_low, d_high) to the crossing error rate.

    Returns:
        A list of (pair, p_cross, lep_at_crossing) sorted by pair; the last
        element is None when either distance cannot be fitted.
    """
    markers = []
    for pair, p_cross in sorted(crossing_points.items()):
        leps = [_lep_at(data_dict[d], p_cross) for d in pair if d in data_dict]
        valid = [lep for lep in leps if lep is not None]
        lep = float(np.sqrt(np.prod(valid))) if len(valid) == 2 else None
        markers.append((pair, p_cross, lep))
    return markers


def plot_threshold(
    data_dict: dict[int, tuple[list[float], list[float], list[float]]],
    crossing_points: dict[tuple[int, int], float],
    crossing_errors: dict[tuple[int, int], float] | None = None,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Generates a log-log threshold plot marking each pairwise crossing point.

    Args:
        data_dict: Dictionary mapping distances to (p_vals, lep_vals, lep_errors).
        crossing_points: Mapping from each (d_low, d_high) pair to the physical
            error rate at which their logical error probabilities cross.
        crossing_errors: Optional mapping from each (d_low, d_high) pair to the
            uncertainty on its crossing point, drawn as a horizontal error bar
            and quoted in the inset legend.
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

    markers = _crossing_markers(data_dict, crossing_points)
    errors = crossing_errors or {}

    # --- Main Plot ---
    _draw_curves(ax, data_dict, markersize=6, capsize=3, label=True)
    for pair, p_cross, lep in markers:
        if lep is not None:
            if errors.get(pair):
                ax.errorbar(
                    p_cross,
                    lep,
                    xerr=errors[pair],
                    fmt="none",
                    ecolor=THRESHOLD_DISTANCE_COLORS.get(pair[1], "black"),
                    elinewidth=1.5,
                    capsize=5,
                    zorder=4,
                )
            ax.plot(
                p_cross,
                lep,
                marker="*",
                markersize=16,
                color=THRESHOLD_DISTANCE_COLORS.get(pair[1], "black"),
                markeredgecolor="black",
                linestyle="none",
                zorder=5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Error Rate (p)", fontsize=12)
    ax.set_ylabel("Logical Error Probability (LEP)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which="both", ls="--", color="grey", alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)

    if not markers:
        fig.tight_layout()
        return fig, ax

    # --- Inset Zoom around the crossing points ---
    ax_ins = ax.inset_axes([0.52, 0.08, 0.45, 0.35])
    _draw_curves(ax_ins, data_dict, markersize=4, capsize=2, label=False)

    p_crosses = [p_cross for _, p_cross, _ in markers]
    zoom_xlim = [
        min(p - errors.get(pair, 0.0) for pair, p, _ in markers) * 0.9,
        max(p + errors.get(pair, 0.0) for pair, p, _ in markers) * 1.1,
    ]
    leps_in_zoom = [
        lep
        for data_tuple in data_dict.values()
        for p, lep in zip(data_tuple[0], data_tuple[1])
        if zoom_xlim[0] <= p <= zoom_xlim[1] and lep > 0
    ] + [lep for _, _, lep in markers if lep is not None]
    zoom_ylim = (
        [min(leps_in_zoom) * 0.8, max(leps_in_zoom) * 1.2]
        if leps_in_zoom
        else [1e-3, 1e-1]
    )

    # Keep the zoom inside the main axes, otherwise the zoom indicator spills
    # outside the plot (e.g. when a crossing's error bar is wider than the data).
    main_xlim, main_ylim = ax.get_xlim(), ax.get_ylim()
    zoom_xlim = [max(zoom_xlim[0], main_xlim[0]), min(zoom_xlim[1], main_xlim[1])]
    zoom_ylim = [max(zoom_ylim[0], main_ylim[0]), min(zoom_ylim[1], main_ylim[1])]

    for pair, p_cross, lep in markers:
        color = THRESHOLD_DISTANCE_COLORS.get(pair[1], "black")
        ax_ins.axvline(x=p_cross, color=color, linestyle=":", alpha=0.7)
        y = lep if lep is not None else np.sqrt(zoom_ylim[0] * zoom_ylim[1])
        label = _crossing_label(pair, p_cross, errors.get(pair))
        if errors.get(pair):
            ax_ins.errorbar(
                p_cross,
                y,
                xerr=errors[pair],
                fmt="none",
                ecolor=color,
                elinewidth=1.5,
                capsize=4,
                zorder=4,
            )
        ax_ins.plot(
            p_cross,
            y,
            marker="*",
            markersize=12,
            color=color,
            markeredgecolor="black",
            linestyle="none",
            zorder=5,
            label=label,
        )

    ax_ins.set_xscale("log")
    ax_ins.set_yscale("log")
    ax_ins.set_xlim(zoom_xlim)
    ax_ins.set_ylim(zoom_ylim)
    # Tick the inset's x-axis at the crossings; default log ticks overlap here.
    # Skip crossings within 3% of the previous tick so labels don't collide.
    tick_ps: list[float] = []
    for p in sorted(p_crosses):
        if not tick_ps or p > tick_ps[-1] * 1.03:
            tick_ps.append(p)
    ax_ins.xaxis.set_minor_locator(NullLocator())
    ax_ins.set_xticks(tick_ps, labels=[_sci_tick(p) for p in tick_ps], fontsize=8)
    ax_ins.grid(True, which="both", ls="--", color="grey", alpha=0.3)
    ax_ins.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        fontsize=9,
        title="Crossing points",
        title_fontsize=9,
    )
    ax.indicate_inset_zoom(ax_ins, edgecolor="gray")

    fig.tight_layout()

    return fig, ax
