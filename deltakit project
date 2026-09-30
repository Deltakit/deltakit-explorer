from __future__ import annotations

import matplotlib.pyplot as plt
import stim

from deltakit.explorer import Client
from deltakit.explorer.enums import DecoderType
from deltakit.explorer.types import (
    Decoder,
    SI1000NoiseModel,
)

from deltakit.explorer import codes
from deltakit.explorer.codes import css_code_memory_circuit
from deltakit.circuit.gates import PauliBasis


client = Client.get_instance()
decoder = Decoder(DecoderType.MWPM)


# distance-3 rotated surface code
code = codes.RotatedPlanarCode(width=3, height=3)


circuit = css_code_memory_circuit(
    code,
    num_rounds=1,
    logical_basis=PauliBasis.Z,
)

# Converts Deltakit circuit into a Stim circuit
compiled_circuit = circuit.as_stim_circuit()

# Add noise
noise_model = SI1000NoiseModel(p=0.01, p_l=0.0)

noisy_circuit = client.add_noise(
    stim_circuit=compiled_circuit,
    noise_model=noise_model,
)

# Simulate
num_shots = 100_000

measurements, _ = client.simulate_stim_circuit(
    stim_circuit=noisy_circuit,
    shots=num_shots,
)

# Decode
decode_result = client.decode_measurements(
    measurements=measurements,
    decoder=decoder,
    ideal_stim_circuit=compiled_circuit,
    noise_model=noise_model,
)

print(
    "Logical error probability (initial run):",
    decode_result.get_logical_error_probability()
)


distances = [3, 5]

min_p = 0.001 
max_p = 0.05
precision= 0.0001

while (max_p - min_p) > precision:
    mid_p = (min_p + max_p) / 2
    current_leps = {}
    print(f"\n--- Testing new midpoint: p = {mid_p} ---")
    
    for d in distances:
        code = codes.RotatedPlanarCode(width=d, height=d)
        circuit = css_code_memory_circuit(
            code,
            num_rounds=d, 
            logical_basis=PauliBasis.Z,
        )
        compiled_circuit = circuit.as_stim_circuit()
            
        # Add noise to the fresh circuit
        current_noise_model = SI1000NoiseModel(p=mid_p, p_l=0.0)
            
        noisy_circuit = client.add_noise(
            stim_circuit=compiled_circuit,
            noise_model=current_noise_model,
        )
            
        # Simulate
        measurements, _ = client.simulate_stim_circuit(
            stim_circuit=noisy_circuit,
            shots=num_shots, 
        )
            
        # Decode
        decode_result = client.decode_measurements(
            measurements=measurements,
            decoder=decoder,
            ideal_stim_circuit=compiled_circuit,
            noise_model=current_noise_model,
        )
            
        current_leps[d] = decode_result.get_logical_error_probability()
        print(f"d={d}, p={mid_p} -> LEP: {current_leps[d]}")
            
    lep_d3 = current_leps[3]
    lep_d5 = current_leps[5]
    
    if lep_d3 > lep_d5:
        min_p = mid_p
    elif lep_d3 < lep_d5:
        max_p = mid_p
        
print(f"\nEstimated Threshold: {mid_p}")
