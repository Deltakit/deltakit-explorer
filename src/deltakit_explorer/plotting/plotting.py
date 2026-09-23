# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Generic dispatch-based plotting interface for deltakit-explorer."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from deltakit_explorer.analysis import LambdaData
from deltakit_explorer.analysis import LogicalErrorProbabilityPerRoundData as LEPPRData
from deltakit_explorer.plotting._lambda import plot_lambda
from deltakit_explorer.plotting._leppr import plot_leppr
from deltakit_explorer.plotting._threshold import plot_threshold
from deltakit_explorer.plotting.results import interpolate_lambda, interpolate_leppr


def plot(
    result: LambdaData | LEPPRData | dict,
    *,
    num_sigmas: int = 3,
    num_points: int = 200,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    """Interpolate raw analysis data and dispatch to the specialised plotter.

    This high-level plotting function accepts raw analysis data, prepares the
    ready-to-plot interpolated result, and dispatches that result to the
    specialised renderer for its type.

    Args:
        result: Raw Lambda, logical error probability per round data, or threshold dictionary.
        num_sigmas: Number of standard deviations for the error band. Default 3.
        num_points: Number of interpolation points. Default 200.
        fig: An existing matplotlib Figure. If None, a new figure will be
            created by the specialised plotting function. Default is None.
        ax: An existing matplotlib Axes. If None, a new axes will be created by
            the specialised plotting function. Default is None.
        title: An optional custom title for the plot. If None, a default title
            based on the result type will be used.
        **kwargs: Additional keyword arguments for specific plotters (e.g.,
            `estimated_threshold` and `threshold_error` for threshold plots).

    Returns:
        The matplotlib Figure and Axes objects containing the plot.

    Raises:
        TypeError: If the ``result`` type is not supported.
        ValueError: If required kwargs are missing for a specific plot type.

    Examples:

        Plotting raw Lambda data::

            fig, ax = plot(lambda_data)

        Plotting threshold data::

            fig, ax = plot(
                threshold_dict,
                estimated_threshold=0.01,
                threshold_error=0.0005
            )

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

        case dict():
            if "estimated_threshold" not in kwargs:
                msg = "You must provide 'estimated_threshold' in kwargs when plotting threshold data."
                raise ValueError(
                    msg
                )
            return plot_threshold(
                data_dict=result,
                estimated_threshold=kwargs["estimated_threshold"],
                threshold_error=kwargs.get("threshold_error", 0.0005),
                fig=fig,
                ax=ax,
                title=title,
            )

        case _:
            msg = (
                f"Unsupported result type: {type(result).__name__}. "
                "Expected `LambdaData`, `LogicalErrorProbabilityPerRoundData`, or `dict`."
            )
            raise TypeError(msg)
