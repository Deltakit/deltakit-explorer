# (c) Copyright Riverlane 2020-2026. All rights reserved.
from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from functools import partial
from numbers import Integral, Real
from time import perf_counter

import numpy as np
import pandas as pd
from deltakit_circuit._circuit import Circuit

from deltakit_explorer.analysis._binomial_fit import ConfidenceInterval, fit_binomial
from deltakit_explorer.analysis._estimate import Estimate
from deltakit_explorer.analysis.error_budget._discretisation import (
    DiscretisationStrategy,
)
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


class BoundSearchStatus(Enum):
    """Termination status of one parameter's bound search.

    Attributes:
        NOT_EVALUATED: No pilot evaluation has been performed.
        CONVERGED: Feasible endpoints meet the sensitivity threshold.
        INSUFFICIENT_COUNTS: A required circuit has too few failures.
        LOW_SNR: Endpoint sensitivity is insufficient.
        SATURATED: An LEP upper bound reaches the configured ceiling.
        DOMAIN_LIMIT: No further probe fits the domain or numerical bracket.
        TRIAL_LIMIT: The configured probe allowance is exhausted.
        INVALID_ESTIMATE: A required statistical estimate is unusable.
        NON_MONOTONE: Observed LEP decreases contradict increasing noise strength.
    """

    NOT_EVALUATED = auto()
    CONVERGED = auto()
    INSUFFICIENT_COUNTS = auto()
    LOW_SNR = auto()
    SATURATED = auto()
    DOMAIN_LIMIT = auto()
    TRIAL_LIMIT = auto()
    INVALID_ESTIMATE = auto()
    NON_MONOTONE = auto()


@dataclass(frozen=True)
class CircuitPilotObservation:
    """Raw counts and logical-error probability for one pilot circuit.

    Attributes:
        distance: Code distance.
        num_rounds: Number of memory-experiment rounds.
        fails: Actual observed logical failures.
        shots: Actual completed shots, including unsuccessful probes.
        lep_interval: Binomial LEP estimate and bounds, if available.
    """

    distance: int
    num_rounds: int
    fails: int
    shots: int
    lep_interval: ConfidenceInterval | None = None


@dataclass(frozen=True)
class LambdaPilotObservation:
    """Observation at one exact noise vector, including failed probes.

    Attributes:
        point_id: Stable identifier within this discovery run.
        noise_parameters: Full noise vector, preserving distinct nearby points.
        circuits: Per-circuit counts and LEP intervals.
        lambda_estimate: Reported Lambda estimate, or None when unavailable.
        lambda_stddev: Reported standard deviation of Lambda.
        inverse_lambda_estimate: Reported inverse-Lambda estimate, if available.
        inverse_lambda_stddev: Reported standard deviation of inverse Lambda.
        warnings: Estimator warnings retained without discarding raw evidence.
        invalidity_reasons: Reasons this point cannot support a valid estimate.
        logical_error_rates: Per-distance logical error rates per round and their
            standard deviations; every configured distance is required for screening.
    """

    point_id: int
    noise_parameters: tuple[float, ...]
    circuits: tuple[CircuitPilotObservation, ...]
    lambda_estimate: float | None = None
    lambda_stddev: float | None = None
    inverse_lambda_estimate: float | None = None
    inverse_lambda_stddev: float | None = None
    warnings: tuple[str, ...] = ()
    invalidity_reasons: tuple[str, ...] = ()
    logical_error_rates: Mapping[int, Estimate] = field(default_factory=dict)


@dataclass(frozen=True)
class ParameterSearchDiagnostic:
    """Outcome and evidence for one axis search.

    Attributes:
        parameter_index: Coordinate being varied.
        status: Search termination status.
        candidate_interval: Last proposed interval, even when unresolved.
        trial_count: Actual noncentral trials performed for this parameter.
        endpoint_snr: Candidate inverse-Lambda signal-to-noise ratio, if both
            candidate endpoints are usable.
        min_failures: Minimum observed failure count across endpoint circuits.
        max_lep: Maximum observed endpoint logical-error probability.
        reason: Explanation of the outcome or missing evidence.
    """

    parameter_index: int
    status: BoundSearchStatus
    candidate_interval: tuple[float, float] | None = None
    trial_count: int = 0
    endpoint_snr: float | None = None
    min_failures: int | None = None
    max_lep: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class BoundSearchTimings:
    """Measured discovery durations in seconds; never runtime deadlines.

    Attributes:
        construction_seconds: Circuit and decoder construction time.
        sampling_seconds: Combined sampling and decoding time.
        post_processing_seconds: Statistical post-processing time.
        total_seconds: Total elapsed time, including setup and orchestration.
    """

    construction_seconds: float = 0.0
    sampling_seconds: float = 0.0
    post_processing_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass(frozen=True)
class BoundsSearchResult:
    """Validated bounds and retained evidence from a possibly incomplete search.

    Attributes:
        bounds: One validated interval per coordinate; None for unresolved axes.
            Unvalidated candidate intervals belong only in diagnostics.
        evaluation_point: Full gradient evaluation vector.
        gradient_evaluation_scale: Multiplier used to resolve the evaluation point.
        diagnostics: Per-parameter search outcomes in coordinate order.
        pilot_data: Raw sampler report rows, including failed probes.
        observations: One observation per sampled point in each parameter search.
            Cached lookups within a search do not contribute to totals.
        phase_timings: Measured phase and total discovery durations.

    Raises:
        ValueError: If bounds do not match the evaluation vector or a resolved
            interval is non-finite or does not strictly contain its coordinate.
    """

    bounds: tuple[tuple[float, float] | None, ...]
    evaluation_point: tuple[float, ...]
    gradient_evaluation_scale: float
    diagnostics: tuple[ParameterSearchDiagnostic, ...]
    pilot_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    observations: tuple[LambdaPilotObservation, ...] = ()
    phase_timings: BoundSearchTimings = field(default_factory=BoundSearchTimings)

    def __post_init__(self) -> None:
        if not self.bounds or len(self.bounds) != len(self.evaluation_point):
            msg = "bounds must contain one entry per nonempty evaluation coordinate."
            raise ValueError(msg)
        for bound, center in zip(self.bounds, self.evaluation_point, strict=True):
            if bound is not None and (
                len(bound) != 2
                or not np.all(np.isfinite(bound))
                or not bound[0] < center < bound[1]
            ):
                msg = "Resolved bounds must be finite and strictly contain the evaluation point."
                raise ValueError(msg)

    @property
    def success(self) -> bool:
        """Whether every parameter has a validated interval."""
        return all(bound is not None for bound in self.bounds)

    @property
    def total_trials(self) -> int:
        """Actual sampled points, including centers and failed probes."""
        return len(self.observations)

    @property
    def total_shots(self) -> int:
        """Actual shots across all pilot circuits, including failed probes."""
        return sum(
            circuit.shots for point in self.observations for circuit in point.circuits
        )


class BoundsDiscoveryError(RuntimeError):
    """Discovery failure retaining its partial result for inspection.

    Args:
        message: Explanation of the failure.
        result: Partial result, if discovery produced one.

    Attributes:
        result: Partial discovery result, when one was produced.
    """

    result: BoundsSearchResult | None

    def __init__(self, message: str, result: BoundsSearchResult | None = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class _PilotBatch:
    """Evaluator output: one observation per requested ID and its raw report.

    Attributes:
        observations: Statistical evidence for each requested point.
        pilot_data: Raw sampler report rows.
        phase_timings: Time spent constructing, sampling, and fitting this batch.
    """

    observations: tuple[LambdaPilotObservation, ...]
    pilot_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    phase_timings: BoundSearchTimings = field(default_factory=BoundSearchTimings)


_PilotEvaluator = Callable[[Mapping[int, tuple[float, ...]]], _PilotBatch]


def _make_pilot_evaluator(**_configuration: object) -> _PilotEvaluator:
    """Create the simulation adapter independently of the search state machine.

    Evaluators receive stable point IDs mapped to exact noise vectors. They must
    sample every configured circuit exactly shots_per_trial times, retain all
    raw rows, and return per-distance rates and Lambda estimates when available.
    Return order is irrelevant; exceptions propagate to the caller.

    Args:
        **_configuration: Keyword arguments forwarded to the pilot sampler.

    Returns:
        A callable that samples and fits requested noise vectors.
    """
    # The adapter consumes the result types above; defer its import to avoid a cycle.
    from deltakit_explorer.analysis.error_budget._pilot import (  # noqa: PLC0415
        _PilotSampler,
    )

    return _PilotSampler(**_configuration)


def _validate_experiments(
    rounds_by_distance: Mapping[int, Sequence[int]],
) -> tuple[str, ...]:
    """Validate estimator support and report the existing zero-SPAM assumption.

    Args:
        rounds_by_distance: Required round counts for each code distance.

    Returns:
        Warnings for distances with only one round count.

    Raises:
        ValueError: If distances or round counts cannot support estimation.
    """
    if len(rounds_by_distance) < 2:
        msg = "Bound search requires at least two distinct code distances."
        raise ValueError(msg)
    single_round_distances = []
    for distance, rounds in rounds_by_distance.items():
        if (
            isinstance(distance, bool)
            or not isinstance(distance, Integral)
            or distance < 1
        ):
            msg = "Code distances must be positive integers."
            raise ValueError(msg)
        if (
            not len(rounds)
            or any(
                isinstance(r, bool) or not isinstance(r, Integral) or r <= 1
                for r in rounds
            )
            or len(set(rounds)) != len(rounds)
        ):
            msg = f"Distance {distance} needs distinct integer round counts greater than one."
            raise ValueError(msg)
        if len(rounds) == 1:
            single_round_distances.append(distance)
    if single_round_distances:
        message = (
            "Only one valid data-point provided for logical error probability per "
            "round. Continuing computation assuming that SPAM error is negligible. "
            f"Distances: {single_round_distances}."
        )
        warnings.warn(message, stacklevel=3)
        return (message,)
    return ()


def _screen_pilot_observation(
    observation: LambdaPilotObservation,
    rounds_by_distance: Mapping[int, Sequence[int]],
    search: BoundSearchParameters,
    estimator_warnings: tuple[str, ...],
) -> tuple[LambdaPilotObservation, BoundSearchStatus, str]:
    """Screen all required circuits and estimates, retaining failed evidence.

    Args:
        observation: Raw counts and fitted estimates at one point.
        rounds_by_distance: Required round counts for each code distance.
        search: Feasibility thresholds.
        estimator_warnings: Warnings shared by all points in this search.

    Returns:
        The annotated observation, screening status, and explanation.
    """
    required = {(d, r) for d, rounds in rounds_by_distance.items() for r in rounds}
    seen = [(c.distance, c.num_rounds) for c in observation.circuits]
    status = BoundSearchStatus.CONVERGED
    reason = "All required circuits and estimates are usable."
    retained_warnings = list(
        dict.fromkeys((*observation.warnings, *estimator_warnings))
    )
    circuits = []
    malformed = len(seen) != len(required) or set(seen) != required
    for circuit in observation.circuits:
        if (
            not isinstance(circuit.shots, Integral)
            or circuit.shots <= 0
            or not isinstance(circuit.fails, Integral)
            or not 0 <= circuit.fails <= circuit.shots
        ):
            malformed = True
            circuits.append(circuit)
            continue
        interval = fit_binomial(circuit.shots, circuit.fails)
        circuits.append(replace(circuit, lep_interval=interval))
        if interval.best < search.max_lep <= interval.high:
            retained_warnings.append(
                f"LEP interval overlaps max_lep for distance={circuit.distance}, "
                f"num_rounds={circuit.num_rounds}."
            )
    if malformed:
        status, reason = (
            BoundSearchStatus.INVALID_ESTIMATE,
            "Missing, duplicated, or invalid required circuit counts.",
        )
    elif any(c.fails < search.min_logical_failures for c in circuits):
        status, reason = (
            BoundSearchStatus.INSUFFICIENT_COUNTS,
            "Insufficient logical failures in at least one required circuit.",
        )
    elif any(c.lep_interval.high >= search.max_lep for c in circuits):
        status, reason = (
            BoundSearchStatus.SATURATED,
            "An endpoint LEP upper bound reaches max_lep (saturated or uncertain).",
        )
    else:
        estimates = (
            observation.lambda_estimate,
            observation.lambda_stddev,
            observation.inverse_lambda_estimate,
            observation.inverse_lambda_stddev,
            *(
                value
                for estimate in observation.logical_error_rates.values()
                for value in estimate
            ),
        )
        if (
            observation.invalidity_reasons
            or set(observation.logical_error_rates) != set(rounds_by_distance)
            or any(
                value is None or not np.isfinite(value) or value <= 0
                for value in estimates
            )
        ):
            status, reason = (
                BoundSearchStatus.INVALID_ESTIMATE,
                "Missing or nonpositive/non-finite per-round rate, Lambda estimate, or uncertainty.",
            )
    reasons = observation.invalidity_reasons
    if status is not BoundSearchStatus.CONVERGED:
        reasons = tuple(dict.fromkeys((*reasons, reason)))
    return (
        replace(
            observation,
            circuits=tuple(circuits),
            warnings=tuple(retained_warnings),
            invalidity_reasons=reasons,
        ),
        status,
        reason,
    )


@dataclass
class _SearchSide:
    value: float
    feasible: float
    infeasible: float | None = None


def _next_probe(
    side: _SearchSide,
    center: float,
    limit: float,
    factor: float,
    logarithmic_lower: bool,
) -> float | None:
    """Bisect a known bracket or expand from the outermost feasible point.

    Args:
        side: Current probe and known feasible and infeasible limits.
        center: Gradient evaluation coordinate.
        limit: Parameter domain boundary on this side.
        factor: Multiplicative interval expansion factor.
        logarithmic_lower: Whether the lower probe must stay positive.

    Returns:
        The next coordinate, or None when no further probe is representable.
    """
    if side.infeasible is not None:
        proposal = side.feasible / 2 + side.infeasible / 2
    else:
        with np.errstate(over="ignore"):
            proposal = center + factor * (side.feasible - center)
        if limit < center:
            proposal = max(limit, proposal)
            if logarithmic_lower and proposal <= 0:
                proposal = side.feasible / 2
        else:
            proposal = min(limit, proposal)
    if (logarithmic_lower and proposal <= 0) or proposal in (
        center,
        side.value,
        side.feasible,
        side.infeasible,
    ):
        return None
    return float(proposal)


def _monotonicity_contradiction(observations: Sequence[LambdaPilotObservation]) -> bool:
    """Detect decreases supported by disjoint binomial LEP intervals.

    Args:
        observations: Screened evidence at the sampled coordinates.

    Returns:
        Whether any circuit contradicts increasing error strength.
    """
    ordered = sorted(observations, key=lambda o: o.noise_parameters[0])
    for left_index, left in enumerate(ordered):
        intervals = {(c.distance, c.num_rounds): c.lep_interval for c in left.circuits}
        for right in ordered[left_index + 1 :]:
            for circuit in right.circuits:
                before = intervals.get((circuit.distance, circuit.num_rounds))
                after = circuit.lep_interval
                if before is not None and after is not None and before.low > after.high:
                    return True
    return False


@dataclass
class _AxisSearch:
    lower: _SearchSide
    upper: _SearchSide
    trials: int = 0
    status: BoundSearchStatus | None = None
    reason: str = ""
    snr: float | None = None
    history: list[LambdaPilotObservation] = field(default_factory=list)


def _run_bound_search(
    center_value: float,
    initial_interval: tuple[float, float],
    rounds_by_distance: Mapping[int, Sequence[int]],
    search: BoundSearchParameters,
    scale: float,
    shots_per_trial: int,
    logarithmic: bool,
    evaluate: _PilotEvaluator,
    estimator_warnings: tuple[str, ...],
    started: float,
) -> BoundsSearchResult:
    """Search one parameter against a batch evaluator, without simulation.

    Args:
        center_value: Gradient evaluation coordinate.
        initial_interval: Starting lower and upper probe coordinates.
        rounds_by_distance: Required round counts for each code distance.
        search: Domain, feasibility thresholds, and probe allowance.
        scale: Calibration multiplier used to obtain the center.
        shots_per_trial: Required shots for each circuit at each probe.
        logarithmic: Whether the interval must remain strictly positive.
        evaluate: Callable supplying pilot observations for requested points.
        estimator_warnings: Warnings shared by all points in this search.
        started: Start time from perf_counter for total elapsed time.

    Returns:
        Validated bounds or an unresolved result with retained pilot evidence.
    """
    center = (center_value,)
    registry: dict[tuple[float, ...], int] = {}
    cache: dict[
        tuple[float, ...], tuple[LambdaPilotObservation, BoundSearchStatus, str]
    ] = {}
    reports = []
    construction_seconds = sampling_seconds = post_processing_seconds = 0.0

    def point(value: float) -> tuple[float, ...]:
        return (float(value),)

    def sample(vectors: Sequence[tuple[float, ...]]) -> None:
        nonlocal construction_seconds, sampling_seconds, post_processing_seconds
        requests = {}
        for vector in vectors:
            if vector not in registry:
                registry[vector] = len(registry)
                requests[registry[vector]] = vector
        if not requests:
            return
        batch = evaluate(requests)
        returned = {o.point_id: o for o in batch.observations}
        if len(returned) != len(batch.observations) or set(returned) != set(requests):
            msg = "Pilot evaluator must return exactly one observation per requested point ID."
            raise ValueError(msg)
        construction_seconds += batch.phase_timings.construction_seconds
        sampling_seconds += batch.phase_timings.sampling_seconds
        post_processing_seconds += batch.phase_timings.post_processing_seconds
        if not batch.pilot_data.empty:
            reports.append(batch.pilot_data)
        processing_started = perf_counter()
        for point_id, vector in requests.items():
            observation = returned[point_id]
            if observation.noise_parameters != vector:
                msg = (
                    "Pilot evaluator returned a different noise vector for a point ID."
                )
                raise ValueError(msg)
            if any(c.shots != shots_per_trial for c in observation.circuits):
                msg = "Pilot evaluator must use exactly shots_per_trial shots per circuit."
                raise ValueError(msg)
            cache[vector] = _screen_pilot_observation(
                observation, rounds_by_distance, search, estimator_warnings
            )
        post_processing_seconds += perf_counter() - processing_started

    sample([center])
    center_observation, center_status, center_reason = cache[center]
    axis = _AxisSearch(
        _SearchSide(initial_interval[0], center_value),
        _SearchSide(initial_interval[1], center_value),
        history=[center_observation],
    )
    if center_status is not BoundSearchStatus.CONVERGED:
        axis.status = center_status
        axis.reason = (
            f"Center: {center_reason} Increase pilot shots or revise the experiments."
        )

    while axis.status is None:
        proposals = []
        for side in (axis.lower, axis.upper):
            vector = point(side.value)
            if vector not in registry and axis.trials < search.max_trials_per_param:
                proposals.append(vector)
                axis.trials += 1
        sample(proposals)
        processing_started = perf_counter()
        try:
            records = [
                cache.get(point(side.value)) for side in (axis.lower, axis.upper)
            ]
            for side, record in zip((axis.lower, axis.upper), records, strict=True):
                if record is None:
                    continue
                observation, status, reason = record
                if not any(o.point_id == observation.point_id for o in axis.history):
                    axis.history.append(observation)
                if status is BoundSearchStatus.CONVERGED:
                    side.feasible = side.value
                elif status is BoundSearchStatus.INVALID_ESTIMATE:
                    axis.status, axis.reason = status, reason
                else:
                    side.infeasible = side.value
            if axis.status is not None:
                continue
            if _monotonicity_contradiction(axis.history):
                axis.status = BoundSearchStatus.NON_MONOTONE
                axis.reason = "Ordered probes contradict monotone error strength: disjoint binomial LEP intervals."
                continue
            if axis.lower.feasible < center_value < axis.upper.feasible:
                lo = cache[point(axis.lower.feasible)][0]
                hi = cache[point(axis.upper.feasible)][0]
                axis.snr = float(
                    abs(hi.inverse_lambda_estimate - lo.inverse_lambda_estimate)
                    / np.hypot(hi.inverse_lambda_stddev, lo.inverse_lambda_stddev)
                )
                if axis.snr >= search.sensitivity_z_score:
                    axis.status = BoundSearchStatus.CONVERGED
                    axis.reason = (
                        "Feasible endpoints meet the inverse-Lambda SNR threshold."
                    )
                    axis.lower.value, axis.upper.value = (
                        axis.lower.feasible,
                        axis.upper.feasible,
                    )
                    continue
            problems = [
                record[2]
                for record in records
                if record is not None and record[1] is not BoundSearchStatus.CONVERGED
            ]
            if problems:
                axis.reason = " ".join(problems)
            elif axis.snr is None:
                axis.reason = (
                    "A feasible endpoint is missing on at least one side of the center."
                )
            else:
                axis.reason = "Insufficient endpoint inverse-Lambda SNR."
            if axis.trials >= search.max_trials_per_param:
                axis.status = BoundSearchStatus.TRIAL_LIMIT
                axis.reason = "Probe allowance exhausted. " + axis.reason
                continue
            domain = search.parameter_domains[0]
            next_values = [
                _next_probe(
                    axis.lower,
                    center_value,
                    domain[0],
                    search.expansion_factor,
                    logarithmic,
                ),
                _next_probe(
                    axis.upper, center_value, domain[1], search.expansion_factor, False
                ),
            ]
            if all(value is None for value in next_values):
                axis.status = BoundSearchStatus.DOMAIN_LIMIT
                axis.reason = (
                    "Domain or representable bracket exhausted. " + axis.reason
                )
                continue
            for side, value in zip((axis.lower, axis.upper), next_values, strict=True):
                if value is not None:
                    side.value = value
        finally:
            post_processing_seconds += perf_counter() - processing_started

    endpoint_observations = [
        cache[point(side.value)][0]
        for side in (axis.lower, axis.upper)
        if point(side.value) in cache
    ]
    evidence = endpoint_observations or [center_observation]
    circuits = [c for o in evidence for c in o.circuits]
    interval = (axis.lower.value, axis.upper.value)
    bound = interval if axis.status is BoundSearchStatus.CONVERGED else None
    diagnostic = ParameterSearchDiagnostic(
        parameter_index=0,
        status=axis.status,
        candidate_interval=interval,
        trial_count=axis.trials,
        endpoint_snr=(
            axis.snr
            if axis.lower.value == axis.lower.feasible
            and axis.upper.value == axis.upper.feasible
            else None
        ),
        min_failures=min((c.fails for c in circuits), default=None),
        max_lep=max(
            (c.lep_interval.best for c in circuits if c.lep_interval is not None),
            default=None,
        ),
        reason=axis.reason,
    )

    return BoundsSearchResult(
        bounds=(bound,),
        evaluation_point=center,
        gradient_evaluation_scale=scale,
        diagnostics=(diagnostic,),
        pilot_data=pd.concat(reports, ignore_index=True) if reports else pd.DataFrame(),
        observations=tuple(record[0] for record in cache.values()),
        phase_timings=BoundSearchTimings(
            construction_seconds,
            sampling_seconds,
            post_processing_seconds,
            perf_counter() - started,
        ),
    )


def _apply_scalar_noise(
    noise_model: Callable[[Circuit, float], Circuit],
    circuit: Circuit,
    vector: Sequence[float],
) -> Circuit:
    """Adapt a scalar noise model to the pilot sampler's vector interface.

    Args:
        noise_model: Callable accepting one scalar noise parameter.
        circuit: Noiseless circuit to annotate.
        vector: A vector containing exactly one noise coordinate.

    Returns:
        The circuit returned by the scalar noise model.
    """
    return noise_model(circuit, float(vector[0]))


def find_error_budget_bounds(
    noise_model: Callable[[Circuit, float], Circuit],
    noise_parameter: float,
    num_rounds_by_distances: Mapping[int, Sequence[int]],
    *,
    search_parameters: BoundSearchParameters,
    gradient_evaluation_scale: float = 0.5,
    shots_per_trial: int = 10_000,
    fitting_parameters: FittingParameters = FittingParameters(),
    sampling_parameters: SamplingParameters = SamplingParameters(),
    memory_generator: MemoryGenerator
    | Mapping[int, Mapping[int, Circuit]] = get_rotated_surface_code_memory_circuit,
    enable_correlations: bool = False,
    seed: int | None = None,
) -> BoundsSearchResult:
    """Search for feasible endpoints for one noise parameter.

    The state machine consumes a fixed-shot batch evaluator independently of
    simulation. Pilot observations and noiseless circuits are cached within this
    call; production budgeting performs fresh sampling.

    The parameter must represent increasing error strength: increasing it must
    not decrease the logical-error probability of any configured circuit.
    Statistically supported decreases terminate the search as NON_MONOTONE.
    Parameters such as coherence times must first be converted to error strengths.

    Args:
        noise_model: Callable adding noise to a circuit using the supplied scalar parameter.
        noise_parameter: Finite scalar calibration value.
        num_rounds_by_distances: Memory-experiment round counts for each distance.
        search_parameters: Required search configuration with exactly one domain.
            The scaled center must lie inside this domain.
        gradient_evaluation_scale: Finite positive scalar multiplying the
            calibration value. Defaults to 0.5, selecting half calibration.
        shots_per_trial: Positive number of shots per distance/round circuit at
            each probed value, not a total divided across the circuits. Partial
            final batches are included and early stopping is disabled.
        fitting_parameters: Production fit configuration. Used to enforce positive
            intervals for logarithmic discretisation; degree and point count are
            preserved and no fit design is generated here.
        sampling_parameters: Execution settings for discovery. Only batch_size
            and max_workers are used; max_shots and early-stopping settings are
            ignored in favour of shots_per_trial and fixed-shot sampling.
        memory_generator: Callable generating noiseless memory circuits, or a
            distance-to-rounds mapping of precomputed circuits.
        enable_correlations: Enable correlated matching during construction and decoding.
        seed: Optional root seed for the pilot stream, separate from production.
            Reproducibility requires fixed backend, version, batching, and workers.

    Returns:
        One validated interval and retained pilot evidence, with None if unresolved.
        Unresolved candidate intervals and their reasons remain in diagnostics.

    Raises:
        ValueError: If calibration, scale, experiments, shot/execution settings,
            domains, or initial intervals are invalid or incompatible with the
            fit strategy, or an evaluator violates the batch contract.
    """
    started = perf_counter()
    scale = gradient_evaluation_scale
    if (
        isinstance(noise_parameter, bool)
        or not isinstance(noise_parameter, Real)
        or not np.isfinite(noise_parameter)
    ):
        msg = "noise_parameter must be a finite scalar."
        raise ValueError(msg)
    center = float(
        _resolve_gradient_point(np.array([float(noise_parameter)]), scale)[0]
    )
    search_parameters.validate_parameter_count(1)
    for name, value in (
        ("shots_per_trial", shots_per_trial),
        ("batch_size", sampling_parameters.batch_size),
        ("max_workers", sampling_parameters.max_workers),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            msg = f"{name} must be a positive integer."
            raise ValueError(msg)

    domain = search_parameters.parameter_domains[0]
    if not domain[0] < center < domain[1]:
        msg = "The evaluation point must lie strictly inside the parameter domain."
        raise ValueError(msg)
    if center == 0:
        msg = "A zero evaluation coordinate needs explicit bounds; relative width is undefined."
        raise ValueError(msg)
    logarithmic = (
        fitting_parameters.discretisation_strategy == DiscretisationStrategy.LOGARITHMIC
    )
    if logarithmic and center <= 0:
        msg = "A logarithmic fit requires a strictly positive evaluation coordinate."
        raise ValueError(msg)
    with np.errstate(over="ignore", under="ignore"):
        half_width = search_parameters.initial_relative_half_width * abs(center)
        lower = max(domain[0], center - half_width)
        upper = min(domain[1], center + half_width)
    if logarithmic and lower <= 0:
        lower = center / 2
    if not lower < center < upper or (logarithmic and lower <= 0):
        msg = "Initial interval cannot strictly contain the evaluation point; supply explicit bounds."
        raise ValueError(msg)

    estimator_warnings = _validate_experiments(num_rounds_by_distances)
    evaluator = _make_pilot_evaluator(
        noise_model=partial(_apply_scalar_noise, noise_model),
        num_rounds_by_distances=num_rounds_by_distances,
        shots_per_trial=shots_per_trial,
        sampling_parameters=sampling_parameters,
        memory_generator=memory_generator,
        enable_correlations=enable_correlations,
        seed=seed,
        search_parameters=search_parameters,
    )
    return _run_bound_search(
        center,
        (lower, upper),
        num_rounds_by_distances,
        search_parameters,
        scale,
        shots_per_trial,
        logarithmic,
        evaluator,
        estimator_warnings,
        started,
    )
