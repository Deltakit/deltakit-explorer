# (c) Copyright Riverlane 2020-2026. All rights reserved.
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
import numpy.typing as npt
from deltakit_circuit._circuit import Circuit

from deltakit_explorer.analysis.error_budget._bounds import (
    BoundsDiscoveryError,
    BoundSearchParameters,
    BoundsSearchResult,
    find_error_budget_bounds,
)
from deltakit_explorer.analysis.error_budget._gradient import inverse_lambda_gradient_at
from deltakit_explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    get_rotated_surface_code_memory_circuit,
)
from deltakit_explorer.analysis.error_budget._parameters import (
    FittingParameters,
    SamplingParameters,
)


@dataclass
class ErrorBudgetResult:
    """Result of an error budgeting computation.

    Attributes:
        contributions: contributions for each of the noise parameters to the error budget.
        contribution_stddevs: estimation of the standard deviation of each of the ``contributions``.
        bound_search_results: Pilot evidence per parameter; empty for explicit bounds.
    """

    contributions: tuple[float, ...]
    contribution_stddevs: tuple[float, ...]
    bound_search_results: tuple[BoundsSearchResult, ...] = ()

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


def _axis_noise(model, point, index, circuit, value):
    vector = point.copy()
    vector[index] = value
    return model(circuit, vector)


def get_error_budget(
    noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
    noise_parameters: npt.NDArray[np.floating] | Sequence[float],
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    noise_parameters_exploration_bounds: list[tuple[float, float]] | None,
    fitting_parameters: FittingParameters = FittingParameters(),
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
    *,
    gradient_evaluation_scale: float = 0.5,
    bound_search_parameters: BoundSearchParameters | None = None,
) -> ErrorBudgetResult:
    """Compute the error budget of the provided ``noise_model``.

    Args:
        noise_model (Callable[[Circuit, npt.NDArray[np.floating]], Circuit]): a callable
            adding noise to the provided circuit, according to the parameters provided.
        noise_parameters (npt.NDArray[numpy.floating] | Sequence[float]): valid
            calibration parameters; the gradient is evaluated at these values multiplied
            by ``gradient_evaluation_scale``. Contributions use the original values.
        num_rounds_by_distances (Mapping[int, Sequence[int]]): a mapping from each code
            distance that should be tested to the number of rounds that should be
            sampled in order to estimate the logical error-probability per round, to
            ultimately get 1 / Λ.
        noise_parameters_exploration_bounds (list[tuple[float, float]] | None): ``(min, max)``
            bounds for each noise parameter of the provided ``noise_model``. A degree
            ``fitting_degree`` polynomial will be fitted on the interval ``[min, max]``.
            The corresponding scaled calibration parameter should
            be strictly contained in ``[min, max]`` (i.e., for any valid ``i``, the
            following is true:
            ``noise_parameters_exploration_bounds[i][0] <
            noise_parameters[i] * gradient_evaluation_scale <
            noise_parameters_exploration_bounds[i][1]``). Ideally, the lower (resp.
            upper) bound provided must be such that the logical error probability when
            replacing the parameter with its lower (resp. upper) bound is above
            ``100 / max_shots`` to ensure enough fails are observed with ``max_shots``
            shots (resp. below ``1 / 2`` to ensure that we can compute the logical error
            probability per round). Pass None to discover bounds one axis at a time,
            holding the other parameters fixed at the gradient evaluation point.
        fitting_parameters: additional parameters relating to how the gradient is
            estimated.
        sampling_parameters: additional parameters relating to the sampling tasks used to
            estimate 1 / Λ indirectly.
        memory_generator (MemoryGenerator): a callable that can generate a memory
            experiment. The resulting circuit will go through the provided
            ``noise_model`` for different values of the noise parameters.
        gradient_evaluation_scale: Positive calibration multiplier; defaults to p/2.
        bound_search_parameters: Domains and pilot limits; required for automatic bounds.

    Returns:
        the error-budgeting result, which consists of an array of contributions for each
        of the noise parameters of the provided ``noise_model`` along with their
        associated standard deviations.

    Raises:
        BoundsDiscoveryError: If discovery fails; its result retains pilot evidence.
        ValueError: If the calibration, scale, or number of domains is invalid.
    """
    # We will compute the gradient at the half point following the methodology outlined in
    # https://doi.org/10.1038/s41586-021-03588-y (Supplementary materials, Section VIII.C.).
    parameters = np.asarray(noise_parameters, dtype=float)
    if (
        parameters.ndim != 1
        or not parameters.size
        or not np.isfinite(parameters).all()
        or not np.isscalar(gradient_evaluation_scale)
        or not np.isfinite(gradient_evaluation_scale)
        or gradient_evaluation_scale <= 0
    ):
        msg = "Provide a finite calibration vector and a positive finite scale."
        raise ValueError(msg)
    point = parameters * gradient_evaluation_scale
    searches = []
    if noise_parameters_exploration_bounds is None:
        config = bound_search_parameters
        if config is None or len(config.parameter_domains) != len(parameters):
            msg = "Automatic bounds require one parameter domain per calibration value."
            raise ValueError(msg)
        for index, parameter in enumerate(parameters):
            result = find_error_budget_bounds(
                partial(_axis_noise, noise_model, point, index),
                float(parameter),
                num_rounds_by_distances,
                search_parameters=replace(
                    config, parameter_domains=(config.parameter_domains[index],)
                ),
                gradient_evaluation_scale=gradient_evaluation_scale,
                fitting_parameters=fitting_parameters,
                sampling_parameters=sampling_parameters,
                memory_generator=memory_generator,
            )
            if result.bounds is None:
                raise BoundsDiscoveryError(result)
            searches.append(result)
        noise_parameters_exploration_bounds = [result.bounds for result in searches]
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
    result.bound_search_results = tuple(searches)
    return result
