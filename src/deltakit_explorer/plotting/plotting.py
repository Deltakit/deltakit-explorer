# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Generic dispatch-based plotting interface for deltakit-explorer."""

from __future__ import annotations

from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LambdaData
from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundData as LEPPRData
from deltakit_explorer.plotting._lambda import plot_lambda
from deltakit_explorer.plotting._leppr import plot_leppr
from deltakit_explorer.plotting._threshold import plot_threshold
from deltakit_explorer.plotting.results import interpolate_lambda, interpolate_leppr


@dataclass
class ThresholdData:
    """Container for threshold estimation plotting data.

    Attributes:
        history_data: Mapping from code distance to a tuple of
            (physical error rates, logical error probabilities, their errors).
        crossing_points: Mapping from each (d_low, d_high) pair to the physical
            error rate at which their logical error probabilities cross.
        crossing_errors: Optional mapping from each (d_low, d_high) pair to the
            uncertainty on its crossing point, drawn as a horizontal error bar.
    """

    history_data: dict[int, tuple[list[float], list[float], list[float]]]
    crossing_points: dict[tuple[int, int], float]
    crossing_errors: dict[tuple[int, int], float] | None = None


def plot(
    result: LambdaData | LEPPRData | ThresholdData,
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Interpolate raw analysis data and dispatch to the specialised plotter.

    This high-level plotting function accepts raw analysis data, prepares the
    ready-to-plot interpolated result, and dispatches that result to the
    specialised renderer for its type.

    Args:
        result: Raw Lambda, logical error probability per round data, or threshold data.
        num_sigmas: Number of standard deviations for the error band. Default 3.
        num_points: Number of interpolation points. Default 200.
        fig: An existing matplotlib Figure. If None, a new figure will be
            created by the specialised plotting function. Default is None.
        ax: An existing matplotlib Axes. If None, a new axes will be created by
            the specialised plotting function. Default is None.
        title: An optional custom title for the plot. If None, a default title
            based on the result type will be used.

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Raises:
        TypeError: If the ``result`` type is not supported.

    Examples:

        Plotting raw Lambda data::

            fig, ax = plot(lambda_data)

        Plotting threshold data::

            threshold_data = ThresholdData(
                history_data=threshold_dict,
                crossing_points={(3, 5): 0.0038, (5, 7): 0.0041},
            )
            fig, ax = plot(threshold_data)

    """
    match result:
        case LambdaData():
            lambda_result = interpolate_lambda(
                result, num_sigmas=num_sigmas, num_points=num_points
            )
            return plot_lambda(lambda_result, fig=fig, ax=ax, title=title)

        case LEPPRData():
            leppr_result = interpolate_leppr(
                result, num_sigmas=num_sigmas, num_points=num_points
            )
            return plot_leppr(leppr_result, fig=fig, ax=ax, title=title)

        case ThresholdData():
            return plot_threshold(
                data_dict=result.history_data,
                crossing_points=result.crossing_points,
                crossing_errors=result.crossing_errors,
                fig=fig,
                ax=ax,
                title=title,
            )

        case _:
            msg = (
                f"Unsupported result type: {type(result).__name__}. "
                "Expected `LambdaData`, `LogicalErrorProbabilityPerRoundData`, or `ThresholdData`."
            )
            raise TypeError(msg)
