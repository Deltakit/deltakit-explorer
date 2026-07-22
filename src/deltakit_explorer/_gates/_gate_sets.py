# (c) Copyright Riverlane 2020-2026. All rights reserved.
"""
This module introduces named gate sets for convenient type checks
and default initialisation.
"""

from deltakit_circuit.gates import CX, MZ, RZ, H, S, X, Y, Z

DEFAULT_ONE_QUBIT_GATES = {X, Y, Z, H, S}

DEFAULT_TWO_QUBIT_GATES = {CX}

DEFAULT_MEASUREMENT_GATES = {MZ}

DEFAULT_RESET_GATES = {RZ}
