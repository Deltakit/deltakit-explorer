# (c) Copyright Riverlane 2020-2026. All rights reserved.
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import astuple, dataclass, replace
from functools import partial

import numpy as np
import numpy.typing as npt
import pandas as pd
from deltakit_circuit._circuit import Circuit

from deltakit_explorer.analysis.error_budget._bounds import (
    BoundsDiscoveryError,
    BoundSearchTimings,
    BoundsSearchResult,
    find_error_budget_bounds,
)
from deltakit_explorer.analysis.error_budget._gradient import inverse_lambda_gradient_at
from deltakit_explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.analysis.error_budget._parameters import (
    BoundSearchParameters,
    FittingParameters,
    SamplingParameters,
    _resolve_gradient_point,
)


@dataclass
class ErrorBudgetResult:
    """Result of an error budgeting computation.

    Attributes:
        contributions: contributions for each of the noise parameters to the error budget.
        contribution_stddevs: estimation of the standard deviation of each of the ``contributions``.
        bound_search_result: Discovery evidence for automatic bounds, or None when
            explicit bounds were supplied.
    """

    contributions: tuple[float, ...]
    contribution_stddevs: tuple[float, ...]
    bound_search_result: BoundsSearchResult | None = None

    @property
    def lambda_estimate(self) -> float:
        """Returns the estimation of Λ according to the computed budget."""
        return float(np.sum(self.contributions))

    @property
    def lambda_stddev_estimate(self) -> float:
        """Returns an estimation of the standard deviation on Λ according to the computed budget."""
        return float(np.sqrt(np.sum(np.asarray(self.contribution_stddevs) ** 2)))

    @staticmethod
    def from_gradient(
        gradient: npt.NDArray[np.floating],
        gradient_stddevs: npt.NDArray[np.floating],
        noise_parameters: npt.NDArray[np.floating],
    ) -> ErrorBudgetResult:
        """Create an instance from gradient and noise parameters."""
        contributions = np.abs(gradient * noise_parameters)
        stddevs = np.abs(gradient_stddevs * noise_parameters)
        return ErrorBudgetResult(
            tuple(map(float, contributions.ravel())), tuple(map(float, stddevs.ravel()))
        )


def _vary_noise_parameter(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    point: npt.NDArray[np.floating],
    index: int,
    circuit: Circuit,
    value: float,
) -> Circuit:
    """Vary one noise coordinate while preserving the other evaluation values."""
    vector = point.copy()
    vector[index] = value
    return noise_model(circuit, vector)


def get_error_budget(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    noise_parameters_exploration_bounds: list[tuple[float, float]] | None = None,
    fitting_parameters: FittingParameters = FittingParameters(),
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
    *,
    gradient_evaluation_scale: float = 0.5,
    bound_search_parameters: BoundSearchParameters | None = None,
    bound_search_shots_per_trial: int = 10_000,
    enable_correlations: bool = False,
    seed: int | None = None,
) -> ErrorBudgetResult:
    """Compute the error budget of the provided ``noise_model``.

    Args:
        noise_model (Callable[[Circuit, npt.NDArray[np.floating]], Circuit]): a callable
            adding noise to the provided circuit, according to the parameters provided.
        noise_parameters (npt.NDArray[numpy.floating] | Sequence[float]): valid
            calibrated parameters to forward to ``noise_model``. The gradient is
            evaluated at ``gradient_evaluation_scale * noise_parameters`` and the
            contributions are weighted by the original calibrated parameters.
        num_rounds_by_distances (Mapping[int, Sequence[int]]): a mapping from each code
            distance that should be tested to the number of rounds that should be
            sampled in order to estimate the logical error-probability per round, to
            ultimately get 1 / Λ.
        noise_parameters_exploration_bounds: ``(min, max)`` bounds for each noise
            parameter, or ``None`` to discover each parameter separately while holding the others
            at their evaluation values. Explicit bounds bypass
            discovery. A degree
            ``fitting_degree`` polynomial will be fitted on the interval ``[min, max]``.
            The corresponding scaled evaluation coordinate should
            be strictly contained in ``[min, max]`` (i.e., for any valid ``i``, the
            following is true:
            ``noise_parameters_exploration_bounds[i][0] <
            gradient_evaluation_scale * noise_parameters[i] <
            noise_parameters_exploration_bounds[i][1]``). Ideally, the lower (resp.
            upper) bound provided must be such that the logical error probability when
            replacing the parameter with its lower (resp. upper) bound is above
            ``100 / max_shots`` to ensure enough fails are observed with ``max_shots``
            shots (resp. below ``1 / 2`` to ensure that we can compute the logical error
            probability per round).
        fitting_parameters: additional parameters relating to how the gradient is
            estimated.
        sampling_parameters: additional parameters relating to the sampling tasks used to
            estimate 1 / Λ indirectly.
        memory_generator (MemoryGenerator): a callable that can generate a memory
            experiment. The resulting circuit will go through the provided
            ``noise_model`` for different values of the noise parameters.
        gradient_evaluation_scale: finite positive scalar multiplying the calibration
            vector to select the gradient point. Defaults to 0.5.
        bound_search_parameters: search configuration with parameter domains,
            required when exploration bounds are ``None``. When supplied, its
            domain count is validated against the calibration vector.
        bound_search_shots_per_trial: pilot shots per distance/round circuit at each
            probed vector, forwarded to discovery separately from production shots.
        enable_correlations: use correlated matching for pilot and production decoding.
        seed: optional root seed. Pilot and production use separate streams and
            distinct task seeds. Reproducibility requires fixed backend, version,
            batching, and worker configuration.

    Returns:
        the error-budgeting result, which consists of an array of contributions for each
        of the noise parameters of the provided ``noise_model`` along with their
        associated standard deviations.

    Raises:
        BoundsDiscoveryError: If discovery leaves any parameter unresolved. The
            exception's result attribute contains the partial search result.
        ValueError: If the calibration, scale, or search configuration is invalid.
    """
    parameters = np.asarray(noise_parameters)
    point = _resolve_gradient_point(parameters, gradient_evaluation_scale)
    if bound_search_parameters is not None:
        bound_search_parameters.validate_parameter_count(len(parameters))
    search_result = None
    if noise_parameters_exploration_bounds is None:
        if bound_search_parameters is None:
            msg = "Automatic discovery requires bound_search_parameters with parameter domains."
            raise ValueError(msg)
        results = []
        observations = []
        reports = []
        for index, parameter in enumerate(parameters):
            result = find_error_budget_bounds(
                partial(_vary_noise_parameter, noise_model, point, index),
                float(parameter),
                num_rounds_by_distances,
                search_parameters=replace(
                    bound_search_parameters,
                    parameter_domains=(
                        bound_search_parameters.parameter_domains[index],
                    ),
                ),
                gradient_evaluation_scale=gradient_evaluation_scale,
                shots_per_trial=bound_search_shots_per_trial,
                fitting_parameters=fitting_parameters,
                sampling_parameters=sampling_parameters,
                memory_generator=memory_generator,
                enable_correlations=enable_correlations,
                seed=seed,
            )
            results.append(result)
            offset = len(observations)
            for observation in result.observations:
                vector = point.copy()
                vector[index] = observation.noise_parameters[0]
                observations.append(
                    replace(
                        observation,
                        point_id=offset + observation.point_id,
                        noise_parameters=tuple(vector),
                    )
                )
            if not result.pilot_data.empty:
                report = result.pilot_data.copy()
                report["point_id"] += offset
                if "noise_0" in report:
                    values = report["noise_0"].copy()
                    for axis, center in enumerate(point):
                        report[f"noise_{axis}"] = values if axis == index else center
                reports.append(report)
        search_result = BoundsSearchResult(
            bounds=tuple(result.bounds[0] for result in results),
            evaluation_point=tuple(point),
            gradient_evaluation_scale=gradient_evaluation_scale,
            diagnostics=tuple(
                replace(result.diagnostics[0], parameter_index=index)
                for index, result in enumerate(results)
            ),
            pilot_data=pd.concat(reports, ignore_index=True)
            if reports
            else pd.DataFrame(),
            observations=tuple(observations),
            phase_timings=BoundSearchTimings(
                *map(sum, zip(*(astuple(r.phase_timings) for r in results)))
            ),
        )
        if not search_result.success:
            unresolved = [
                i for i, bound in enumerate(search_result.bounds) if bound is None
            ]
            msg = f"Bound discovery is incomplete for parameter indices {unresolved}."
            raise BoundsDiscoveryError(msg, result=search_result)
        noise_parameters_exploration_bounds = [
            bound for bound in search_result.bounds if bound is not None
        ]
    # Evaluate the gradient.
    gradient, gradient_stddev = inverse_lambda_gradient_at(
        noise_model,
        point,
        num_rounds_by_distances,
        noise_parameters_exploration_bounds,
        fitting_parameters,
        sampling_parameters,
        memory_generator,
        enable_correlations=enable_correlations,
        seed=seed,
    )
    result = ErrorBudgetResult.from_gradient(gradient, gradient_stddev, parameters)
    result.bound_search_result = search_result
    return result
