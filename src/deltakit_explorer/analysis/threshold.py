from __future__ import annotations

import math
import matplotlib.pyplot as plt

from deltakit.explorer import Client
from deltakit.explorer.enums import DecoderType
from deltakit.explorer.types import (
    Decoder,
    SI1000NoiseModel,
)
from deltakit.explorer import codes
from deltakit.explorer.codes import css_code_memory_circuit
from deltakit.circuit.gates import PauliBasis


def get_error_bar(lep: float, num_shots: int) -> float:
    """Calculates the standard error (1-sigma error bar) for a given LEP."""
    if lep == 0:
        return 0.0
    return math.sqrt((lep * (1.0 - lep)) / num_shots)


class ThresholdEstimator:
    def __init__(
        self, 
        min_p: float = 0.001, 
        max_p: float = 0.05, 
        precision: float = 0.0001, 
        num_shots: int = 100_000
    ):
        
        self.min_p = min_p
        self.max_p = max_p
        self.precision = precision
        self.num_shots = num_shots
        
        
        self.client = Client.get_instance()
        self.decoder = Decoder(DecoderType.MWPM)

    
    def run_simulation(self, p_value: float, distance: int, shots: int) -> float:
        """Run a single simulation at a specific physical error rate and distance."""
        code = codes.RotatedPlanarCode(width=distance, height=distance)
        circuit = css_code_memory_circuit(
            code,
            num_rounds=distance,
            logical_basis=PauliBasis.Z,
        )
        compiled_circuit = circuit.as_stim_circuit()
        
        noise_model = SI1000NoiseModel(p=p_value, p_l=0.0)
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

    def run_search(self):
        """Execute the bisection search to find the threshold."""
        print(f"Starting bisection search between {self.min_p} and {self.max_p}...")
        
        while (self.max_p - self.min_p) > self.precision:
            mid_p = (self.min_p + self.max_p) / 2
            print(f"\n--- Testing new midpoint: p = {mid_p} ---")
            
            
            current_shots = self.num_shots
            overlap = True
            SAFETY = 0.8  

            while overlap:
                # Run simulations
                lep_d3 = self.run_simulation(mid_p, distance=3, shots=current_shots)
                lep_d5 = self.run_simulation(mid_p, distance=5, shots=current_shots)

                # Calculate error bars
                error_d3 = get_error_bar(lep_d3, current_shots)
                error_d5 = get_error_bar(lep_d5, current_shots)

                # Gap between LEPs
                gap = abs(lep_d3 - lep_d5)

                # Combined uncertainty
                combined_error = error_d3 + error_d5

                if gap <= combined_error:
                    # Prevents a divide-by-zero error if the gap is 0
                    if gap == 0:
                        required_shots = current_shots * 10
                    else:
                        
                        required_shots = current_shots * (combined_error / (SAFETY * gap))**2

                    # Round up to the next integer
                    new_shots = math.ceil(required_shots)

                    print(
                        f"Gap={gap:.3e}, Error={combined_error:.3e}. "
                        f"Increasing shots from {current_shots:,} to {new_shots:,}"
                    )

                    current_shots = new_shots
                else:
                    overlap = False
            
            print(f"d=3, p={mid_p} -> Final LEP: {lep_d3} (Shots: {current_shots:,})")
            print(f"d=5, p={mid_p} -> Final LEP: {lep_d5} (Shots: {current_shots:,})")
            
           
            if lep_d3 > lep_d5:
                self.min_p = mid_p
            elif lep_d3 < lep_d5:
                self.max_p = mid_p
            else:
                break
                
        threshold = (self.min_p + self.max_p) / 2
        print(f"\nEstimated Threshold: {threshold}")
        return threshold


if __name__ == "__main__":
    estimator = ThresholdEstimator()
    estimator.run_search()        
