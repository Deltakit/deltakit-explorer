# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Fixed-shot simulation and statistical fitting for bound discovery."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from time import perf_counter

import numpy as np
import numpy.typing as npt
import pandas as pd
from deltakit_circuit import Circuit
from deltakit_decode.analysis import RunAllAnalysisEngine

from deltakit_explorer.analysis import calculate_lambda_and_lambda_stddev
from deltakit_explorer.analysis._binomial_fit import fit_binomial
from deltakit_explorer.analysis._estimate import Estimate
from deltakit_explorer.analysis.error_budget._bounds import (
    BoundSearchTimings,
    CircuitPilotObservation,
    LambdaPilotObservation,
    _PilotBatch,
)
from deltakit_explorer.analysis.error_budget._generation import (
    _seed_stream,
    generate_decoder_managers_for_lambda,
)
from deltakit_explorer.analysis.error_budget._lambda import reciprocal_stddev
from deltakit_explorer.analysis.error_budget._memory import (
    MemoryGenerator,
    PreComputedMemoryGenerator,
)
from deltakit_explorer.analysis.error_budget._parameters import (
    BoundSearchParameters,
    SamplingParameters,
)
from deltakit_explorer.analysis.error_budget._post_processing import (
    _compute_logical_error_rate_per_round_from_results,
)


def _observe_point(
    point_id: int,
    vector: tuple[float, ...],
    data: pd.DataFrame,
    rounds_by_distance: Mapping[int, Sequence[int]],
    search: BoundSearchParameters,
) -> LambdaPilotObservation:
    """Fit a point without omitting any required circuit or distance.

    Args:
        point_id: Stable ID of the sampled point.
        vector: Exact noise vector used for sampling.
        data: Raw report rows for this point.
        rounds_by_distance: Required round counts for each code distance.
        search: Failure-count and LEP thresholds for fitting.

    Returns:
        Raw counts and available estimates, including any fitting warnings.

    Raises:
        RuntimeError: If fitting raises an unexpected programming error.
    """
    counts = data.groupby(["distance", "num_rounds"])[["fails", "shots"]].sum()
    circuits = tuple(
        CircuitPilotObservation(
            int(row.distance),
            int(row.num_rounds),
            int(row.fails),
            int(row.shots),
            fit_binomial(int(row.shots), int(row.fails)),
        )
        for row in counts.reset_index().itertuples(index=False)
    )
    observation = LambdaPilotObservation(point_id, vector, circuits)
    required = {(d, r) for d, rounds in rounds_by_distance.items() for r in rounds}
    if {(c.distance, c.num_rounds) for c in circuits} != required or any(
        c.fails < search.min_logical_failures or c.lep_interval.high >= search.max_lep
        for c in circuits
    ):
        # The state machine classifies these counts, retaining every experiment.
        return observation

    rates = {}
    reasons = []
    estimates = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                for distance, rounds in sorted(rounds_by_distance.items()):
                    rate = _compute_logical_error_rate_per_round_from_results(
                        rounds,
                        data[data.distance == distance],
                    )
                    rates[distance] = Estimate(rate.leppr, rate.leppr_stddev)
                if any(
                    not np.isfinite(v) or v <= 0
                    for rate in rates.values()
                    for v in rate
                ):
                    reasons.append(
                        "Nonpositive or non-finite per-round rate or uncertainty."
                    )
                else:
                    fit = calculate_lambda_and_lambda_stddev(
                        list(rates),
                        [r.value for r in rates.values()],
                        [r.stddev for r in rates.values()],
                    )
                    estimates.update(
                        lambda_estimate=fit.lambda_, lambda_stddev=fit.lambda_std
                    )
                    if np.isfinite(fit.lambda_) and fit.lambda_ > 0:
                        estimates.update(
                            inverse_lambda_estimate=1 / fit.lambda_,
                            inverse_lambda_stddev=reciprocal_stddev(
                                fit.lambda_, fit.lambda_std
                            ),
                        )
        except (
            np.linalg.LinAlgError,
            FloatingPointError,
            OverflowError,
            ZeroDivisionError,
        ) as exc:
            reasons.append(f"Statistical estimation failed: {exc}")
        except RuntimeError as exc:
            # scipy curve_fit documents this specific statistical failure. Other
            # RuntimeErrors, like model and programming failures, must propagate.
            if not str(exc).startswith("Optimal parameters not found"):
                raise
            reasons.append(f"Statistical estimation failed: {exc}")
    return replace(
        observation,
        logical_error_rates=rates,
        **estimates,
        warnings=tuple(str(w.message) for w in caught),
        invalidity_reasons=tuple(reasons),
    )


class _PilotSampler:
    """A discovery-call-local batch evaluator with cached noiseless circuits.

    Args:
        noise_model: Callable adding noise for the supplied vector.
        num_rounds_by_distances: Required round counts for each code distance.
        shots_per_trial: Fixed shot count per circuit at each sampled point.
        sampling_parameters: Batch size and worker configuration.
        memory_generator: Callable or mapping supplying noiseless circuits.
        enable_correlations: Whether to use correlated matching.
        seed: Optional root seed for this search's pilot stream.
        search_parameters: Failure-count and LEP thresholds for fitting.
    """

    def __init__(
        self,
        *,
        noise_model: Callable[[Circuit, npt.NDArray[np.floating]], Circuit],
        num_rounds_by_distances: Mapping[int, Sequence[int]],
        shots_per_trial: int,
        sampling_parameters: SamplingParameters,
        memory_generator: MemoryGenerator | Mapping[int, Mapping[int, Circuit]],
        enable_correlations: bool,
        seed: int | None,
        search_parameters: BoundSearchParameters,
    ) -> None:
        self.noise_model = noise_model
        self.rounds = num_rounds_by_distances
        self.shots = shots_per_trial
        self.sampling = sampling_parameters
        self.memory_generator = (
            PreComputedMemoryGenerator(memory_generator)
            if isinstance(memory_generator, Mapping)
            else memory_generator
        )
        self.enable_correlations = enable_correlations
        self.seed = _seed_stream(seed, 0)
        self.search = search_parameters
        self.circuits: dict[int, dict[int, Circuit]] = {}

    def __call__(self, points: Mapping[int, tuple[float, ...]]) -> _PilotBatch:
        started = perf_counter()
        for distance, rounds in self.rounds.items():
            if distance not in self.circuits:
                self.circuits[distance] = {
                    r: self.memory_generator(distance, r) for r in rounds
                }
        vectors = np.asarray(list(points.values())).T
        managers = generate_decoder_managers_for_lambda(
            vectors,
            self.noise_model,
            self.rounds,
            self.sampling.max_workers,
            memory_generator=PreComputedMemoryGenerator(self.circuits),
            point_ids=list(points),
            enable_correlations=self.enable_correlations,
            seed=self.seed,
            batch_size=self.sampling.batch_size,
        )
        constructed = perf_counter()
        engine = RunAllAnalysisEngine(
            experiment_name="Discovering error-budget bounds",
            decoder_managers=managers,
            max_shots=self.shots,
            batch_size=self.sampling.batch_size,
            loop_condition=None,
            num_parallel_processes=self.sampling.max_workers,
        )
        report = engine.run()
        sampled = perf_counter()
        if set(report.point_id) != set(points):
            msg = "Pilot report point IDs do not match the requested points."
            raise ValueError(msg)
        observations = tuple(
            _observe_point(
                point_id,
                vector,
                report[report.point_id == point_id],
                self.rounds,
                self.search,
            )
            for point_id, vector in points.items()
        )
        finished = perf_counter()
        return _PilotBatch(
            observations,
            report,
            BoundSearchTimings(
                constructed - started,
                sampled - constructed,
                finished - sampled,
                finished - started,
            ),
        )
