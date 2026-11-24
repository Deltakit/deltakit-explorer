# (c) Copyright Riverlane 2020-2026. All rights reserved.

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np
from deltakit_circuit.gates import PauliBasis
from deltakit_decode._mwpm_decoder import PyMatchingDecoder

from deltakit_explorer.codes import RotatedPlanarCode, css_code_memory_circuit
from deltakit_explorer.enums import DecoderType
from deltakit_explorer.qpu import QPU, SI1000Noise
from deltakit_explorer.types import Decoder

logger = logging.getLogger(__name__)


def get_error_bar(lep: float, num_shots: int) -> float:
    """Calculates the binomial standard error for a logical error probability.

    Args:
        lep: The logical error probability.
        num_shots: The total number of shot samples executed.

    Returns:
        The estimated standard error.
    """
    if math.isclose(lep, 0.0, abs_tol=1e-9):
        return 0.0
    return math.sqrt((lep * (1.0 - lep)) / num_shots)


class ThresholdEstimator:
    """Estimates the quantum error correction threshold using bisection search.

    Args:
        min_p: Minimum physical error probability to consider.
        max_p: Maximum physical error probability to consider.
        precision: Precision required for the threshold estimate.
        num_shots: Number of shots used for each simulation.
        code_class: Quantum error correction code class to use.
        noise_model_class: Noise model class to use.
        decoder: Decoder to use. If None, MWPM is used.
        experiment_builder: Callable building the experiment circuit from a code
            and distance. If None, a Z-basis CSS memory circuit with ``distance``
            rounds is used.
    """

    def __init__(
        self,
        min_p: float = 0.001,
        max_p: float = 0.05,
        precision: float = 0.0001,
        num_shots: int = 100_000,
        code_class: type = RotatedPlanarCode,
        noise_model_class: type = SI1000Noise,
        decoder: Decoder | None = None,
        experiment_builder: Callable | None = None,
    ):
        self.min_p = min_p
        self.max_p = max_p
        self.precision = precision
        self.num_shots = num_shots
        self.code_class = code_class
        self.noise_model_class = noise_model_class
        self.decoder = (
            decoder if decoder is not None else Decoder(decoder_type=DecoderType.MWPM)
        )
        self.history_data = {}
        self.crossing_brackets: dict[tuple[int, int], tuple[float, float]] = {}

        # Default to the CSS memory circuit if no custom builder is provided
        self.experiment_builder = experiment_builder or (
            lambda code, distance: css_code_memory_circuit(
                code,
                num_rounds=distance,
                logical_basis=PauliBasis.Z,
            )
        )

    def run_simulation(self, p_value: float, distance: int, shots: int):
        """Executes a quantum memory circuit simulation under a specific physical error rate.

        Args:
            p_value: The physical error probability.
            distance: The code distance (width and height) of the layout.
            shots: The number of Monte Carlo sampling shots to execute.

        Raises:
            NotImplementedError: If the simulation is not yet implemented.

        Returns:
            The measured logical error probability (lep).
        """
        code = self.code_class(width=distance, height=distance)
        compiled_circuit = self.experiment_builder(code, distance)

        noise_model = self.noise_model_class(p=p_value)

        qpu = QPU(qubits=compiled_circuit.qubits, noise_model=noise_model)
        noisy_circuit = qpu.compile_and_add_noise_to_circuit(compiled_circuit)

        if self.decoder.decoder_type == DecoderType.MWPM:
            matcher, stim_circuit = (
                PyMatchingDecoder.construct_decoder_and_stim_circuit(noisy_circuit)
            )
        else:
            msg = f"Local decoding for {self.decoder.decoder_type} is not yet implemented in this pipeline."
            raise NotImplementedError(msg)

        detectors, observables = stim_circuit.compile_detector_sampler().sample(
            shots=shots, separate_observables=True
        )
        predictions = matcher.decode_batch_to_logical_flip(detectors)

        logical_errors = int(np.sum(np.any(predictions != observables, axis=1)))

        return float(logical_errors) / float(shots)

    def run_single_pair_search(self, d_low: int, d_high: int) -> float:
        """Performs a bisection search to find the threshold crossing for a code distance pair.

        The final bracket is stored in ``crossing_brackets``. If the search
        stops early because the two distances cannot be told apart at the
        midpoint, the crossing is taken to be that midpoint, and the bracket is
        narrowed to the p range over which the LEP gap would stay hidden by the
        noise: ``mid_p +/- combined_error / slope``, where ``slope`` is the rate
        of change of the gap estimated from the gaps measured at the two bracket
        ends. It is never wider than the bracket itself.

        Args:
            d_low: The lower code distance value.
            d_high: The higher code distance value.

        Returns:
            The estimated physical error rate at the threshold crossover point.
        """
        current_min = self.min_p
        current_max = self.max_p
        # Signed gap (lep_low - lep_high) measured at each bracket end: positive
        # below threshold, negative above. None until that end has been sampled.
        gap_at_min: float | None = None
        gap_at_max: float | None = None
        SAFETY = 0.8
        MAX_SHOTS = 1_000_000

        while (current_max - current_min) > self.precision:
            mid_p: float = (current_min + current_max) / 2.0
            current_shots: int = self.num_shots

            # Re-simulate with more shots until the two distances' error bars
            # no longer overlap, i.e. until we can tell which distance is worse.
            while True:
                lep_low = self.run_simulation(
                    mid_p, distance=d_low, shots=current_shots
                )
                lep_high = self.run_simulation(
                    mid_p, distance=d_high, shots=current_shots
                )
                error_low = get_error_bar(lep_low, current_shots)
                error_high = get_error_bar(lep_high, current_shots)

                gap = abs(lep_low - lep_high)
                combined_error = error_low + error_high

                if gap > combined_error:
                    break

                if current_shots >= MAX_SHOTS:
                    break

                current_shots = min(
                    MAX_SHOTS,
                    math.ceil(current_shots * (combined_error / (SAFETY * gap)) ** 2)
                    if gap > 0
                    else current_shots * 10,
                )

            if d_low not in self.history_data:
                self.history_data[d_low] = {}
            if d_high not in self.history_data:
                self.history_data[d_high] = {}

            self.history_data[d_low][mid_p] = (lep_low, current_shots)
            self.history_data[d_high][mid_p] = (lep_high, current_shots)
            logger.info(
                "d=(%d, %d) p=%.6g: lep=%.4g vs %.4g (%d shots)",
                d_low,
                d_high,
                mid_p,
                lep_low,
                lep_high,
                current_shots,
            )

            if gap <= combined_error:
                logger.warning(
                    "d=(%d, %d): error bars still overlap at p=%.6g with %d shots; "
                    "stopping search at bracket [%.6g, %.6g].",
                    d_low,
                    d_high,
                    mid_p,
                    current_shots,
                    current_min,
                    current_max,
                )
                half_width = (current_max - current_min) / 2.0
                if gap_at_min is not None and gap_at_max is not None:
                    slope = (gap_at_min - gap_at_max) / (current_max - current_min)
                    if slope > 0:
                        half_width = min(half_width, combined_error / slope)
                self.crossing_brackets[(d_low, d_high)] = (
                    mid_p - half_width,
                    mid_p + half_width,
                )
                return mid_p

            if lep_low > lep_high:
                current_min = mid_p
                gap_at_min = lep_low - lep_high
            elif lep_low < lep_high:
                current_max = mid_p
                gap_at_max = lep_low - lep_high
            elif lep_low == lep_high:
                break

        self.crossing_brackets[(d_low, d_high)] = (current_min, current_max)
        return (current_min + current_max) / 2.0

    def estimate_crossing_points(
        self, distance_pairs: list[tuple[int, int]]
    ) -> dict[tuple[int, int], float]:
        """Finds the threshold crossing point for each pair of code distances.

        Each pair is searched independently by bisection; the searches run one
        after another (the simulations are CPU-bound and hold the GIL, and
        concurrent first calls into the deltakit_stim extension could deadlock).

        No extrapolation to an asymptotic threshold is attempted: with only a
        few pairs, a finite-size-scaling fit is underdetermined. The crossing
        points themselves are finite-size estimates of the threshold.

        Args:
            distance_pairs: A list of tuples containing lower and upper code distances.

        Returns:
            A dictionary mapping each distance pair to the physical error rate
            at which their logical error probabilities cross.
        """
        return {
            (d_low, d_high): self.run_single_pair_search(d_low, d_high)
            for d_low, d_high in distance_pairs
        }

    def get_crossing_uncertainties(self) -> dict[tuple[int, int], float]:
        """Returns the bisection uncertainty of each crossing point found so far.

        The uncertainty is half the width of the final bracket. When the search
        converges, that is at most ``precision / 2``. When it stops early
        because the two distances cannot be told apart, it is the p range over
        which the LEP gap stays below the combined error bars, estimated from
        the gap's slope across the bracket (see ``run_single_pair_search``).
        Either way it is an estimate of the search's resolution, not a rigorous
        confidence interval: a bisection step decided on a statistical
        fluctuation can leave the true crossing outside the bracket.

        Returns:
            A dictionary mapping each searched (d_low, d_high) pair to the
            half-width of its final bisection bracket.
        """
        return {
            pair: (p_max - p_min) / 2.0
            for pair, (p_min, p_max) in self.crossing_brackets.items()
        }

    def get_search_history(self) -> dict:
        """Formats the saved simulation data into structured arrays for matplotlib visualisation.

        Returns:
            A dictionary mapping each code distance to a 3-tuple of lists containing (p_vals, lep_vals, lep_errors).
        """
        formatted_history = {}
        for d, points_dict in self.history_data.items():
            sorted_points = sorted(points_dict.items())

            p_vals = [pt[0] for pt in sorted_points]
            lep_vals = [pt[1][0] for pt in sorted_points]
            shots_vals = [pt[1][1] for pt in sorted_points]

            lep_errors = [
                get_error_bar(lep, shots) for lep, shots in zip(lep_vals, shots_vals)
            ]

            formatted_history[d] = (p_vals, lep_vals, lep_errors)

        return formatted_history
