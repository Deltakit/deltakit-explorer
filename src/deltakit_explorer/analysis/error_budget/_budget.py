# (c) Copyright Riverlane 2020-2026. All rights reserved.
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from deltakit_circuit._circuit import Circuit

from deltakit_explorer.analysis.error_budget._bounds import (
    BoundsDiscoveryError,
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

    Note:
        The pilot simulation adapter is not connected yet, so
        automatic mode currently raises BoundsDiscoveryError with an unresolved
        result. Explicit-bound budgeting remains available.

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
            parameter, or ``None`` to invoke discovery. Explicit bounds bypass
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
        enable_correlations: correlation setting forwarded to discovery. Reserved
            for its future sampling adapter; production decoding is unchanged.
        seed: seed forwarded to discovery. Reserved for its future sampling
            adapter; this does not currently seed production sampling.

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
        search_result = find_error_budget_bounds(
            noise_model,
            parameters,
            num_rounds_by_distances,
            search_parameters=bound_search_parameters,
            gradient_evaluation_scale=gradient_evaluation_scale,
            shots_per_trial=bound_search_shots_per_trial,
            fitting_parameters=fitting_parameters,
            sampling_parameters=sampling_parameters,
            memory_generator=memory_generator,
            enable_correlations=enable_correlations,
            seed=seed,
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
    )
    result = ErrorBudgetResult.from_gradient(gradient, gradient_stddev, parameters)
    result.bound_search_result = search_result
    return result
