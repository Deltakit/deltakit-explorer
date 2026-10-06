# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Estimation of the error-correction threshold from logical error rates."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from itertools import pairwise
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit

from deltakit_explorer.analysis._estimate import Estimate


class ThresholdData(NamedTuple):
    """Result of :func:`estimate_threshold`.

    Attributes:
        threshold: Estimated threshold physical error rate, from a finite-size
            scaling fit, with its standard deviation.
        nu: Fitted critical exponent ``nu``, with its standard deviation.
        crossings: Physical error rates at which the curves of consecutive code
            distances cross. Empty if no crossing was found in the provided range.
    """

    threshold: Estimate
    nu: Estimate
    crossings: tuple[float, ...]


def _consecutive_crossings(
    p: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[float, ...]:
    """Return the first crossing (if any) of each pair of consecutive-distance curves.

    ``y[i]`` is the curve of the ``i``-th distance and ``p`` is sorted increasingly.
    """
    crossings = []
    for lo, hi in pairwise(y):
        diff = hi - lo
        flips = np.flatnonzero((diff[:-1] * diff[1:] <= 0) & (diff[:-1] != diff[1:]))
        if len(flips) == 0:
            continue
        i = flips[0]
        # Linear interpolation of the zero of ``diff`` between ``p[i]`` and ``p[i+1]``.
        slope = (p[i + 1] - p[i]) / (diff[i + 1] - diff[i])
        crossings.append(float(p[i] - diff[i] * slope))
    return tuple(crossings)


def estimate_threshold(
    distances: Sequence[int],
    physical_error_rates: Sequence[float] | npt.NDArray[np.floating],
    logical_error_rates: Sequence[Sequence[float]] | npt.NDArray[np.floating],
    logical_error_rates_stddev: Sequence[Sequence[float]]
    | npt.NDArray[np.floating]
    | None = None,
) -> ThresholdData:
    """Estimate the threshold of a code family with a finite-size scaling fit.

    The logical error rate ``p_L`` of a code of distance ``d`` at physical error rate
    ``p`` is assumed to follow the ansatz of
    https://arxiv.org/abs/quant-ph/0207088 close to the threshold ``p_th``:

    ``p_L = A + B x + C x²``, with ``x = (p - p_th) d^(1/nu)``.

    ``p_th``, ``nu``, ``A``, ``B`` and ``C`` are fitted to the data. The ansatz is only
    valid near the threshold, so provide data on a window around it.

    Args:
        distances: code distances, at least two.
        physical_error_rates: physical error rates shared by all distances, at least
            five.
        logical_error_rates: logical error rates as an array of shape
            ``(len(distances), len(physical_error_rates))``.
        logical_error_rates_stddev: standard deviations of ``logical_error_rates``, same
            shape. If not provided, all points are weighted equally and the returned
            standard deviations come from the scatter of the fit residuals.

    Returns:
        the fitted threshold and critical exponent, and the crossings of consecutive
        distances, which can be used as a model-independent cross-check.

    Warns:
        UserWarning: if the fitted threshold lies outside ``physical_error_rates``.

    Raises:
        ValueError: if fewer than 2 distances or 5 physical error rates are provided,
            if the shapes do not match, or if the fit does not converge.
    """
    d = np.asarray(distances, dtype=float)
    p = np.asarray(physical_error_rates, dtype=float)
    y = np.asarray(logical_error_rates, dtype=float)
    sigma = (
        None
        if logical_error_rates_stddev is None
        else np.asarray(logical_error_rates_stddev, dtype=float)
    )
    if len(d) < 2 or len(p) < 5:
        msg = "At least 2 distances and 5 physical error rates are required."
        raise ValueError(msg)
    if y.shape != (len(d), len(p)) or (sigma is not None and sigma.shape != y.shape):
        msg = "Logical error rates must have shape (len(distances), len(physical_error_rates))."
        raise ValueError(msg)

    order = np.argsort(d)
    d, y = d[order], y[order]
    sigma = None if sigma is None else sigma[order]
    order = np.argsort(p)
    p, y = p[order], y[:, order]
    sigma = None if sigma is None else sigma[:, order]

    crossings = _consecutive_crossings(p, y)

    def model(xy: npt.NDArray[np.float64], pth, nu, a, b, c):
        x = (xy[0] - pth) * xy[1] ** (1 / nu)
        return a + b * x + c * x**2

    pp, dd = np.meshgrid(p, d)
    # Start from the crossings and ``nu = 1``; a quadratic fit gives ``A``, ``B``, ``C``
    # (a flat start has a singular Jacobian in ``p_th`` and ``nu``).
    pth0 = float(np.mean(crossings)) if crossings else float(np.median(p))
    c0, b0, a0 = np.polyfit(((pp - pth0) * dd).ravel(), y.ravel(), 2)
    guess = [pth0, 1.0, a0, b0, c0]
    try:
        popt, pcov = curve_fit(
            model,
            np.vstack([pp.ravel(), dd.ravel()]),
            y.ravel(),
            p0=guess,
            sigma=None if sigma is None else sigma.ravel(),
            absolute_sigma=sigma is not None,
            maxfev=20000,
        )
    except RuntimeError as e:
        msg = f"Finite-size scaling fit did not converge: {e}"
        raise ValueError(msg) from e
    if not p[0] <= popt[0] <= p[-1]:
        warnings.warn(
            f"Fitted threshold {popt[0]:.4g} is outside the provided physical error "
            f"rates [{p[0]:.4g}, {p[-1]:.4g}]; provide data on a window around it.",
            stacklevel=2,
        )
    perr = np.sqrt(np.diag(pcov))
    return ThresholdData(
        Estimate(float(popt[0]), float(perr[0])),
        Estimate(float(popt[1]), float(perr[1])),
        crossings,
    )
