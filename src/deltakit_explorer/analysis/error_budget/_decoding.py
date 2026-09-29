# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""Private matching and seeded-worker adapters for error-budget sampling."""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from tempfile import TemporaryDirectory

import deltakit_stim as stim
import numpy as np
import numpy.typing as npt
import pymatching  # type: ignore[import-untyped]
from deltakit_circuit import Circuit
from deltakit_core.decoding_graphs import OrderedSyndrome
from deltakit_decode.analysis._matching_decoder_managers import StimDecoderManager


class _CorrelatedMatchingDecoder:
    """Keep the complete decomposed DEM instead of reducing it to a graph.

    Only the logical-flip interface consumed by StimDecoderManager is needed.
    The matcher is rebuilt from DEM text after worker-process serialization.
    """

    def __init__(self, dem_text: str, num_detectors: int, num_observables: int) -> None:
        self._dem_text = dem_text
        self._num_detectors = num_detectors
        self.logicals = tuple(range(num_observables))

    @classmethod
    def construct_decoder_and_stim_circuit(
        cls,
        circuit: Circuit,
    ) -> tuple[_CorrelatedMatchingDecoder, stim.Circuit]:
        stim_circuit = circuit.as_stim_circuit()
        dem = stim_circuit.detector_error_model(decompose_errors=True)
        decoder = cls(
            str(dem), stim_circuit.num_detectors, stim_circuit.num_observables
        )
        # Build during construction, so its cost is not attributed to sampling.
        _ = decoder._matcher
        return decoder, stim_circuit

    @cached_property
    def _matcher(self) -> pymatching.Matching:
        # Closing the file before loading also supports Windows worker processes.
        with TemporaryDirectory(prefix="deltakit-dem-") as directory:
            path = Path(directory) / "model.dem"
            path.write_text(self._dem_text, encoding="utf-8")
            matcher = pymatching.Matching.from_detector_error_model_file(
                path,
                enable_correlations=True,
            )
        matcher.ensure_num_fault_ids(len(self.logicals))
        return matcher

    def decode_to_logical_flip(self, syndrome: OrderedSyndrome) -> tuple[bool, ...]:
        correction = self._matcher.decode(
            syndrome.as_bitstring(self._num_detectors),
            enable_correlations=True,
        )
        return tuple(map(bool, correction))

    def decode_batch_to_logical_flip(
        self,
        syndromes: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        return self._matcher.decode_batch(syndromes, enable_correlations=True)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state.pop("_matcher", None)
        return state

    def __str__(self) -> str:
        return "MWPM (correlated)"


def _run_seeded_batch(task):
    circuit, decoder, seed, batch_size, shots = task
    manager = StimDecoderManager(circuit, decoder, seed=seed, batch_size=batch_size)
    manager.run_batch_shots(shots)
    return manager.empirical_decoding_error_distribution


class _SeededStimDecoderManager(StimDecoderManager):
    """Use fresh deterministic seeds for each parallel batch and worker.

    The upstream worker resets to its initial seed on every batch and advances
    the parent using a process-dependent hash. Use independent child streams
    here, identified by the number of shots already completed and the job index.
    """

    def run_batch_shots(self, batch_limit: int | None) -> tuple[int, int]:
        if batch_limit is not None and batch_limit > 0 and not self.reporters:
            # A one-shot remainder must not restart the seed through the separate
            # scalar generator after the batch generator has already been used.
            self._exec_shots_batch(self.batch_error_generator, batch_limit)
            return self.shots, self.fails
        return super().run_batch_shots(batch_limit)

    def run_batch_shots_parallel(
        self,
        batch_limit: int | None,
        processes: int,
        pool,
        min_tasks_per_process: int = 50,
    ) -> tuple[int, int]:
        if batch_limit is None:
            return super().run_batch_shots_parallel(
                batch_limit,
                processes,
                pool,
                min_tasks_per_process,
            )
        processes = min(processes, max(1, batch_limit // min_tasks_per_process))
        seeds = np.random.SeedSequence(
            self._start_seed,
            spawn_key=(self.shots,),
        ).spawn(processes)
        per_worker, remainder = divmod(batch_limit, processes)
        tasks = [
            (
                self._stim_noise_circuit,
                self._decoder,
                int(seed.generate_state(1, dtype=np.uint64)[0]),
                self.batch_size,
                per_worker + (i < remainder),
            )
            for i, seed in enumerate(seeds)
        ]
        for distribution in pool.map(_run_seeded_batch, tasks):
            self._empirical_decoding_error_distribution += distribution
        return self.shots, self.fails
