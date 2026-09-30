# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Pilot sampling to choose intervals for error-budget gradients."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from numbers import Integral, Real

import numpy as np
import pandas as pd
from deltakit_circuit import Circuit

from . import _lambda
from ._discretisation import DiscretisationStrategy
from ._memory import MemoryGenerator, get_rotated_surface_code_memory_circuit
from ._parameters import FittingParameters, SamplingParameters
from ._post_processing import _compute_lambda_from_results


@dataclass(frozen=True)
class BoundSearchParameters:
    """Limits for a pilot search, independent of production sampling.

    Attributes:
        parameter_domains: Finite (lower, upper) limits, one per noise parameter.
        shots_per_trial: Shots per distance/round circuit at each probed value.
        min_logical_failures: Required failures in every pilot circuit.
        max_lep: Strict upper limit on each circuit's measured LEP.
        sensitivity_z_score: Required endpoint inverse-Lambda signal-to-noise ratio.
        max_iterations: Maximum endpoint pairs, in addition to the centre probe.

    Raises:
        ValueError: If domains, counts, or thresholds are invalid.
    """

    parameter_domains: Sequence[tuple[float, float]]
    shots_per_trial: int = 10_000
    min_logical_failures: int = 10
    max_lep: float = 0.45
    sensitivity_z_score: float = 3.0
    max_iterations: int = 8

    def __post_init__(self) -> None:
        domains = np.asarray(self.parameter_domains, dtype=float)
        if (
            domains.ndim != 2
            or domains.shape[1] != 2
            or not len(domains)
            or not np.isfinite(domains).all()
            or np.any(domains[:, 0] >= domains[:, 1])
        ):
            msg = "parameter_domains must contain finite, ordered pairs."
            raise ValueError(msg)
        object.__setattr__(self, "parameter_domains", tuple(map(tuple, domains)))
        for value in (
            self.shots_per_trial,
            self.min_logical_failures,
            self.max_iterations,
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                msg = "Pilot shot, failure, and iteration counts must be positive integers."
                raise ValueError(msg)
        if (
            not 0 < self.max_lep < 0.5
            or not np.isfinite(self.sensitivity_z_score)
            or self.sensitivity_z_score <= 0
        ):
            msg = "Require 0 < max_lep < 0.5 and a finite positive SNR threshold."
            raise ValueError(msg)


@dataclass
class BoundsSearchResult:
    """A scalar interval and the evidence used to select it.

    Attributes:
        bounds: Validated (lower, upper) endpoints, or None if unresolved.
        pilot_data: All raw circuit counts, including unusable probes. Columns
            inverse_lambda and inverse_lambda_stddev retain available estimates;
            NaN denotes an unusable point. noise_0 is the varied scalar parameter.
    """

    bounds: tuple[float, float] | None
    pilot_data: pd.DataFrame


class BoundsDiscoveryError(RuntimeError):
    """Automatic budgeting failed to find a usable interval.

    Args:
        result: Unresolved scalar search, including all its pilot evidence.
    """

    def __init__(self, result: BoundsSearchResult) -> None:
        super().__init__(
            "No usable bounds found; inspect result.pilot_data, increase pilot shots, or supply explicit bounds."
        )
        self.result = result


def _scalar_noise(model, circuit, vector):
    # Copy precomputed circuits before applying a potentially mutating model.
    return model(deepcopy(circuit), float(vector[0]))


def find_error_budget_bounds(
    noise_model: Callable[[Circuit, float], Circuit],
    noise_parameter: float,
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    *,
    search_parameters: BoundSearchParameters,
    gradient_evaluation_scale: float = 0.5,
    fitting_parameters: FittingParameters = FittingParameters(),
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
) -> BoundsSearchResult:
    """Find scalar bounds using fixed-shot pilots and endpoint sensitivity.

    Start at ±25% of the scaled calibration value. Double feasible distances
    from the centre until sensitivity passes; bisect toward a feasible point
    when a probe violates the count or LEP limits. Probes are cached by exact
    value. This assumes a connected feasible interval around the centre.
    Endpoint SNR is a proxy for gradient precision, not a guarantee about fit bias.

    Args:
        noise_model: Callable adding noise using one scalar parameter.
        noise_parameter: Finite nonzero calibration value.
        num_rounds_by_distances: Round counts greater than one for at least two distances.
        search_parameters: Pilot limits with exactly one parameter domain.
        gradient_evaluation_scale: Positive calibration multiplier; defaults to p/2.
        fitting_parameters: Enforces positive bounds for logarithmic discretisation.
        sampling_parameters: Batch size and workers; production stopping rules are ignored.
        memory_generator: Noiseless circuit generator or precomputed circuit mapping.

    Returns:
        A usable interval or None bounds, with raw counts and inverse-Lambda estimates.

    Raises:
        ValueError: If calibration, experiments, or execution settings are invalid.
    """
    search = search_parameters
    if (
        any(
            isinstance(x, bool) or not isinstance(x, Real) or not np.isfinite(x)
            for x in (noise_parameter, gradient_evaluation_scale)
        )
        or gradient_evaluation_scale <= 0
    ):
        msg = "Calibration must be finite and gradient_evaluation_scale positive."
        raise ValueError(msg)
    center = float(noise_parameter * gradient_evaluation_scale)
    logarithmic = (
        fitting_parameters.discretisation_strategy == DiscretisationStrategy.LOGARITHMIC
    )
    if len(search.parameter_domains) != 1:
        msg = "Scalar discovery requires exactly one parameter domain."
        raise ValueError(msg)
    domain = search.parameter_domains[0]
    valid_center = domain[0] < center < domain[1] and center != 0
    if not valid_center or (logarithmic and center <= 0):
        msg = "The evaluation point must be nonzero, inside its domain, and positive for a logarithmic fit."
        raise ValueError(msg)
    required = {(d, r) for d, rounds in num_rounds_by_distances.items() for r in rounds}
    if len(num_rounds_by_distances) < 2 or any(
        not isinstance(d, Integral)
        or d < 1
        or not len(rounds)
        or len(set(rounds)) != len(rounds)
        or any(not isinstance(r, Integral) or r <= 1 for r in rounds)
        for d, rounds in num_rounds_by_distances.items()
    ):
        msg = "Require two distances and distinct round counts greater than one."
        raise ValueError(msg)
    if any(
        isinstance(x, bool) or not isinstance(x, Integral) or x < 1
        for x in (sampling_parameters.batch_size, sampling_parameters.max_workers)
    ):
        msg = "batch_size and max_workers must be positive integers."
        raise ValueError(msg)
    sampling = replace(
        sampling_parameters, max_shots=search.shots_per_trial, lep_target_rse=0.0
    )
    cache, reports = {}, []

    def probe(value):
        if value not in cache:
            _, _, report = _lambda._run_lambda_engine(
                partial(_scalar_noise, noise_model),
                [value],
                num_rounds_by_distances,
                sampling,
                memory_generator,
            )
            report = report.assign(inverse_lambda=np.nan, inverse_lambda_stddev=np.nan)
            reports.append(report)
            cache[value] = None
            counts = report.groupby(["distance", "num_rounds"]).sum(numeric_only=True)
            if (
                set(counts.index) == required
                and (counts.shots == search.shots_per_trial).all()
                and (counts.fails >= search.min_logical_failures).all()
                and (counts.fails / counts.shots < search.max_lep).all()
            ):
                try:
                    with np.errstate(divide="raise", invalid="raise", over="raise"):
                        fit = _compute_lambda_from_results(
                            num_rounds_by_distances, report
                        )
                        estimate = (
                            1 / fit.lambda_,
                            _lambda.reciprocal_stddev(fit.lambda_, fit.lambda_std),
                        )
                    if np.isfinite(estimate).all() and min(estimate) > 0:
                        cache[value] = estimate
                        report[["inverse_lambda", "inverse_lambda_stddev"]] = estimate
                except (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError):
                    pass
        return cache[value]

    bounds = None
    if probe(center) is not None:
        feasible = [center, center]
        infeasible: list[float | None] = [None, None]
        candidates = [
            max(domain[0], center - abs(center) / 4),
            min(domain[1], center + abs(center) / 4),
        ]
        for _ in range(search.max_iterations):
            for side, value in enumerate(candidates):
                if probe(value) is None:
                    infeasible[side] = value
                else:
                    feasible[side] = value
            if feasible[0] < center < feasible[1]:
                low, high = (cache[x] for x in feasible)
                if abs(high[0] - low[0]) >= search.sensitivity_z_score * np.hypot(
                    low[1], high[1]
                ):
                    bounds = tuple(feasible)
                    break
            for side in range(2):
                value = (
                    feasible[side] / 2 + infeasible[side] / 2
                    if infeasible[side] is not None
                    else np.clip(center + 2 * (feasible[side] - center), *domain)
                )
                candidates[side] = (
                    feasible[side] / 2 if logarithmic and value <= 0 else float(value)
                )
    return BoundsSearchResult(bounds, pd.concat(reports, ignore_index=True))
