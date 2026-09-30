# (c) Copyright Riverlane 2020-2026. All rights reserved.

from __future__ import annotations

import concurrent.futures
import logging
import math

import deltakit_stim  # type: ignore[import-untyped]
import numpy as np
import pymatching  # type: ignore[import-untyped]
from deltakit_circuit.gates import PauliBasis
from scipy.optimize import curve_fit

from deltakit_explorer.codes import RotatedPlanarCode, css_code_memory_circuit
from deltakit_explorer.enums import DecoderType
from deltakit_explorer.qpu import QPU
from deltakit_explorer.types import (
    Decoder,
    SI1000NoiseModel,
)

logger = logging.getLogger(__name__)


def get_error_bar(lep: float, num_shots: int) -> float:
    """Calculates the binomial standard error for a logical error probability.

    Args:
        lep: The logical error probability.
        num_shots: The total number of shot samples executed.

    Returns:
        The estimated standard error.
    """
    if lep == 0:
        return 0.0
    return math.sqrt((lep * (1.0 - lep)) / num_shots)


def finite_size_scaling_drift(d: float, p_th: float, C: float, omega: float) -> float:
    """The physical drift equation for finite-size scaling.

    p_cross(d) = p_th + C * d^(-omega)

    Reference:
    Wang, C., Harrington, J., & Preskill, J. (2003).
    "Confinement-Higgs transition in a disordered gauge theory and the accuracy threshold for quantum memory."
    arXiv:quant-ph/0207088v2
    """
    return p_th + C * (d ** -omega)


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
    """

    def __init__(
        self,
        min_p: float = 0.001,
        max_p: float = 0.05,
        precision: float = 0.0001,
        num_shots: int = 100_000,
        code_class=RotatedPlanarCode,
        noise_model_class=SI1000NoiseModel,
        decoder=None,
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
        compiled_circuit = css_code_memory_circuit(
            code,
            num_rounds=distance,
            logical_basis=PauliBasis.Z,
        )

        try:
            noise_model = self.noise_model_class(p=p_value, p_l=0.0)
        except TypeError:
            noise_model = self.noise_model_class(p=p_value)

        qpu = QPU(qubits=compiled_circuit.qubits, noise_model=noise_model)
        noisy_circuit = qpu.compile_and_add_noise_to_circuit(
            compiled_circuit
        ).as_stim_circuit()

        pure_stim_circuit = deltakit_stim.Circuit(str(noisy_circuit))

        detectors, observables = pure_stim_circuit.compile_detector_sampler().sample(
            shots=shots, separate_observables=True
        )

        error_model = pure_stim_circuit.detector_error_model(decompose_errors=True)

        if self.decoder.decoder_type == DecoderType.MWPM:
            matching = pymatching.Matching.from_detector_error_model(error_model)
            predictions = matching.decode_batch(detectors)
        else:
            msg = f"Local decoding for {self.decoder.decoder_type} is not yet implemented in this pipeline."
            raise NotImplementedError(msg)

        logical_errors = int(np.sum(np.any(predictions != observables, axis=1)))

        return float(logical_errors) / float(shots)

    def run_single_pair_search(self, d_low: int, d_high: int) -> float:
        """Performs a bisection search to find the threshold crossing for a code distance pair.

        Args:
            d_low: The lower code distance value.
            d_high: The higher code distance value.

        Raises:
            NotImplementedError: If the simulation is not yet implemented.

        Returns:
            The estimated physical error rate at the threshold crossover point.
        """
        current_min = self.min_p
        current_max = self.max_p
        SAFETY = 0.8
        MAX_SHOTS = 1_000_000

        while (current_max - current_min) > self.precision:
            mid_p: float = (current_min + current_max) / 2.0
            current_shots: int = self.num_shots
            overlap: bool = True

            while overlap:
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

                if gap <= combined_error:
                    current_shots = (
                        math.ceil(
                            current_shots * (combined_error / (SAFETY * gap)) ** 2
                        )
                        if gap > 0
                        else current_shots * 10
                    )
                    # NEW SAFETY LIMIT TO PREVENT FREEZING:
                    if current_shots > MAX_SHOTS:
                        msg = (
                            f"Could not obtain non-overlapping error bars "
                            f"for p={mid_p:.6g} within {MAX_SHOTS} shots."
                        )
                        raise RuntimeError(msg)

                    overlap = False

            if d_low not in self.history_data:
                self.history_data[d_low] = {}
            if d_high not in self.history_data:
                self.history_data[d_high] = {}

            self.history_data[d_low][mid_p] = (lep_low, current_shots)
            self.history_data[d_high][mid_p] = (lep_high, current_shots)

            if lep_low > lep_high:
                current_min = mid_p
            elif lep_low < lep_high:
                current_max = mid_p
            elif lep_low == lep_high:
                break

        return (current_min + current_max) / 2.0

    def run_parallel_searches(self, distance_pairs: list[tuple[int, int]]) -> dict:
        """Executes multiple bisection searches concurrently across different distance pairs.

        Args:
            distance_pairs: A list of tuples containing lower and upper code distances.

        Returns:
            A dictionary mapping each distance pair to its respective threshold estimate.
        """
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.run_single_pair_search, d_low, d_high): (
                    d_low,
                    d_high,
                )
                for d_low, d_high in distance_pairs
            }
            for future in concurrent.futures.as_completed(futures):
                pair = futures[future]
                results[pair] = future.result()
        return results

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

    def estimate_asymptotic_threshold(self, distance_pairs: list[tuple[int, int]]) -> tuple[float, float, float, dict]:
        """Runs parallel searches and fits the data to find the asymptotic threshold.

        Args:
            distance_pairs: A list of tuples containing lower and upper code distances.

        Returns:
            A tuple containing (p_th_final, p_th_error, omega_final, raw_results_dict).
        """
        results = self.run_parallel_searches(distance_pairs)

        average_distances: list[float] = []
        crossing_points: list[float] = []

        for pair, p_cross in sorted(results.items()):
            d_low, d_high = pair
            average_distances.append((d_low + d_high) / 2.0)
            crossing_points.append(p_cross)

        # Initial guesses: [p_th, C, omega]
        initial_guess = [0.01, 0.01, 1.0]

        popt, pcov = curve_fit(
            finite_size_scaling_drift,
            np.array(average_distances),
            np.array(crossing_points),
            p0=initial_guess,
            bounds=([0.0, -10.0, 0.1], [0.15, 10.0, 5.0])
        )

        p_th_final, _, omega_final = popt
        p_th_error = float(np.sqrt(np.diag(pcov))[0])

        return p_th_final, p_th_error, omega_final, results
