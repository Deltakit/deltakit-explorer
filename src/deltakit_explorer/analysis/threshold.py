from __future__ import annotations

import math
import logging
import concurrent.futures

from deltakit_explorer import Client
from deltakit_explorer.enums import DecoderType
from deltakit_explorer.types import (
    Decoder,
    SI1000NoiseModel,
)
from deltakit_explorer import codes
from deltakit_explorer.codes import css_code_memory_circuit
from deltakit_circuit.gates import PauliBasis

logger = logging.getLogger(__name__)

def get_error_bar(lep: float, num_shots: int) -> float:
    if lep == 0:
        return 0.0
    return math.sqrt((lep * (1.0 - lep)) / num_shots)


class ThresholdEstimator:
    # Notice client is right here!
    def __init__(
        self, 
        client: Client,
        min_p: float = 0.001, 
        max_p: float = 0.05, 
        precision: float = 0.0001, 
        num_shots: int = 100_000,
        code_class = codes.RotatedPlanarCode,
        noise_model_class = SI1000NoiseModel
    ):
        self.min_p = min_p
        self.max_p = max_p
        self.precision = precision
        self.num_shots = num_shots
        self.code_class = code_class
        self.noise_model_class = noise_model_class
        self.decoder = Decoder(DecoderType.MWPM)
        self.client = client

    def run_simulation(self, p_value: float, distance: int, shots: int) -> float:
        code = self.code_class(width=distance, height=distance)
        compiled_circuit = css_code_memory_circuit(
            code,
            num_rounds=distance,
            logical_basis=PauliBasis.Z,
        ).as_stim_circuit()
        
        noise_model = self.noise_model_class(p=p_value, p_l=0.0)
        
        noisy_circuit = self.client.add_noise(
            stim_circuit=compiled_circuit,
            noise_model=noise_model,
        )
        
        measurements, _ = self.client.simulate_stim_circuit(
            stim_circuit=noisy_circuit,
            shots=shots,
        )
        
        decode_result = self.client.decode_measurements(
            measurements=measurements,
            decoder=self.decoder,
            ideal_stim_circuit=compiled_circuit,
            noise_model=noise_model,
        )
        
        return decode_result.get_logical_error_probability()

    def run_single_pair_search(self, d_low: int, d_high: int) -> float:
        current_min = self.min_p
        current_max = self.max_p
        
        logger.info(f"Starting bisection search between {current_min} and {current_max} for distances {d_low} & {d_high}...")
        
        while (current_max - current_min) > self.precision:
            mid_p: float = (current_min + current_max) / 2.0
            logger.info(f"--- Testing midpoint: p = {mid_p} for d={d_low},{d_high} ---")
            
            current_shots: int = self.num_shots
            overlap: bool = True
            SAFETY: float = 0.8
            
            while overlap:
                lep_low = self.run_simulation(mid_p, distance=d_low, shots=current_shots)
                lep_high = self.run_simulation(mid_p, distance=d_high, shots=current_shots)
                
                error_low = get_error_bar(lep_low, current_shots)
                error_high = get_error_bar(lep_high, current_shots)
                
                gap = abs(lep_low - lep_high)
                combined_error = error_low + error_high
                
                if gap <= combined_error:
                    if gap == 0:
                        required_shots = current_shots * 10
                    else:
                        required_shots = current_shots * (combined_error / (SAFETY * gap)) ** 2
                        
                    new_shots = math.ceil(required_shots)
                    current_shots = new_shots
                else:
                    overlap = False
                    
            if lep_low > lep_high:
                current_min = mid_p
            elif lep_low < lep_high:
                current_max = mid_p
            else:
                break
                
        return (current_min + current_max) / 2.0

    def run_parallel_searches(self, distance_pairs: list[tuple[int, int]]) -> dict:
        results = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.run_single_pair_search, d_low, d_high): (d_low, d_high) 
                for d_low, d_high in distance_pairs
            }
            for future in concurrent.futures.as_completed(futures):
                pair = futures[future]
                results[pair] = future.result()
        return results
