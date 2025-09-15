# (c) Copyright Riverlane 2020-2025.
"""
This module contains classes used to capture native gate sets and gate times
of a QPU.
"""

from typing import Dict, Optional, Set, Type

from deltakit_circuit.gates import (MEASUREMENT_GATES, ONE_QUBIT_GATES,
                                    ONE_QUBIT_MEASUREMENT_GATES, RESET_GATES,
                                    TWO_QUBIT_GATES, Gate,
                                    OneQubitCliffordGate, OneQubitResetGate,
                                    TwoOperandGate, _MeasurementGate)
from deltakit_circuit.gates._measurement_gates import MPP
from deltakit_explorer._gates._gate_sets import (DEFAULT_MEASUREMENT_GATES,
                                                 DEFAULT_ONE_QUBIT_GATES,
                                                 DEFAULT_RESET_GATES,
                                                 DEFAULT_TWO_QUBIT_GATES)


class NativeGateSetAndTimes:
    """
    Class for capturing native gate sets of a quantum computer and times of
    gate execution (in seconds).

    Parameters
    ----------
    one_qubit_gates : Optional[Dict[Type[OneQubitCliffordGate], float]]
        Dictionary of the one-qubit gates available on the quantum computer and the
        associated gate times. By default, the gates are those in
        DEFAULT_ONE_QUBIT_GATES with times all equal to 1.0.
    two_qubit_gates : Optional[Dict[Type[TwoOperandGate], float]]
        Dictionary of the two-qubit gates available on the quantum computer and the
        associated gate times. By default, the gates are those in
        DEFAULT_TWO_QUBIT_GATES with times all equal to 1.0.
    reset_gates : Optional[Dict[Type[OneQubitResetGate], float]]
        Dictionary of the reset gates available on the quantum computer and the
        associated gate times. By default, the gates are those in DEFAULT_RESET_GATES
        with times all equal to 1.0.
    measurement_gates : Optional[Dict[Type[_MeasurementGate], float]]
        Dictionary of the one-qubit measurement gates available on the quantum
        computer and the associated gate times. By default, the gates are those in
        DEFAULT_MEASUREMENT_GATES with times all equal to 1.0.

    Raises
    ------
    ValueError
        If any supplied one-qubit gate is not a valid one-qubit gate.
    ValueError
        If any supplied two-qubit gate is not a valid two-qubit gate.
    ValueError
        If any supplied reset gate is not a valid reset gate.
    ValueError
        If any supplied measurement gate is not a valid one-qubit measurement gate.
    ValueError
    ValueError
        If any time is not a positive float.
    """

    def __init__(
        self,
        one_qubit_gates: Optional[Dict[Type[OneQubitCliffordGate], float]] = None,
        two_qubit_gates: Optional[Dict[Type[TwoOperandGate], float]] = None,
        reset_gates: Optional[Dict[Type[OneQubitResetGate], float]] = None,
        measurement_gates: Optional[Dict[Type[_MeasurementGate], float]] = None,
    ):
        self.one_qubit_gates = (
            {gate: 1.0 for gate in DEFAULT_ONE_QUBIT_GATES}
            if one_qubit_gates is None
            else one_qubit_gates
        )
        self.two_qubit_gates = (
            {gate: 1.0 for gate in DEFAULT_TWO_QUBIT_GATES}
            if two_qubit_gates is None
            else two_qubit_gates
        )
        self.reset_gates = (
            {gate: 1.0 for gate in DEFAULT_RESET_GATES}
            if reset_gates is None
            else reset_gates
        )
        self.measurement_gates = (
            {gate: 1.0 for gate in DEFAULT_MEASUREMENT_GATES}
            if measurement_gates is None
            else measurement_gates
        )

        self.native_gates = (
            set(self.one_qubit_gates)
            | set(self.two_qubit_gates)
            | set(self.reset_gates)
            | set(self.measurement_gates)
        )

        if not set(self.one_qubit_gates) <= ONE_QUBIT_GATES:
            raise ValueError("Element in one-qubit gate list is not a valid gate.")

        if not set(self.two_qubit_gates) <= TWO_QUBIT_GATES:
            raise ValueError("Element in two-qubit gate list is not a valid gate.")

        if not set(self.reset_gates) <= RESET_GATES:
            raise ValueError("Element in reset gate list is not a valid gate.")

        if not set(self.measurement_gates) <= ONE_QUBIT_MEASUREMENT_GATES.union({MPP}):
            raise ValueError("Element in measurement gate list is not a valid gate.")

        gate_time_dicts = [
            self.one_qubit_gates,
            self.two_qubit_gates,
            self.reset_gates,
            self.measurement_gates,
        ]
        for gate_time_dict in gate_time_dicts:
            for gate, time in gate_time_dict.items():
                self._check_time(gate, time)

    @staticmethod
    def _check_time(gate: Type[Gate], time: float) -> None:
        if time < 0.0:
            raise ValueError(
                "A gate time must be a non-negative float but that for "
                f"{gate.stim_string} is {time}."
            )

    def add_gate(self, gate: Type[Gate], time: float = 1.0) -> None:
        """
        Add a gate and associated time to the native gate set.

        Parameters
        ----------
        gate : Type[Gate]
            Gate to be added.
        time : float, optional
            Time of the gate to be added. By default, 1.0.
        """
        self._check_time(gate, time)
        if gate in ONE_QUBIT_GATES:
            self.one_qubit_gates[gate] = time
        elif gate in TWO_QUBIT_GATES:
            self.two_qubit_gates[gate] = time
        elif gate in RESET_GATES:
            self.reset_gates[gate] = time
        elif gate in ONE_QUBIT_MEASUREMENT_GATES.union({MPP}):
            self.measurement_gates[gate] = time
        else:
            raise ValueError(f"Unknown gate {gate} supplied.")
        self.native_gates.add(gate)

    @staticmethod
    def from_times(time_1_qubit_gate: float, time_2_qubit_gate: float,
                   time_reset: float, time_measurement: float,
                   native_gates: Optional['NativeGateSet'] = None):
        """Assign times to gates based on class (1-qubit, 2-qubit, reset, measurement).

        Parameters
        ----------
        time_1_qubit_gate: float
            Time to execute a 1-qubit gate.
        time_2_qubit_gate: float
            Time to execute a 2-qubit gate.
        time_measurement: float
            Time to measure a qubit.
        time_reset: float
            Time to reset a qubit.
        native_gates: Optional[NativeGateSet]
            An instance of `NativeGateSet` specifying the native gates of the QPU.

        Returns
        -------
        native_gates_and_times: NativeGateSetAndTimes
            A `NativeGateSetAndTimes` object representing the native gates and times
            of a QPU.

        """

        if native_gates is not None:
            one_qubit_gates = native_gates.one_qubit_gates
            two_qubit_gates = native_gates.two_qubit_gates
            reset_gates = native_gates.reset_gates
            measurement_gates = native_gates.measurement_gates
        else:
            one_qubit_gates = ONE_QUBIT_GATES
            two_qubit_gates = TWO_QUBIT_GATES
            reset_gates = RESET_GATES
            measurement_gates = MEASUREMENT_GATES

        native_gates_and_times = NativeGateSetAndTimes(
            one_qubit_gates={g: time_1_qubit_gate
                             for g in one_qubit_gates},
            two_qubit_gates={g: time_2_qubit_gate
                             for g in two_qubit_gates},
            reset_gates={r: time_reset
                         for r in reset_gates},
            measurement_gates={m: time_measurement
                               for m in measurement_gates},
        )
        return native_gates_and_times


class NativeGateSet(NativeGateSetAndTimes):
    """
    Class for capturing native gate sets of a quantum computer.

    Parameters
    ----------
    one_qubit_gates : Optional[Set[Type[OneQubitCliffordGate]]]
        Set of the one-qubit gates available on the quantum computer. By default,
        the gates are those in DEFAULT_ONE_QUBIT_GATES.
    two_qubit_gates : Optional[Set[Type[TwoOperandGate]]]
        Set of the two-qubit gates available on the quantum computer. By default,
        the gates are those in DEFAULT_TWO_QUBIT_GATES.
    reset_gates : Optional[Set[Type[OneQubitResetGate]]]
        Set of the reset gates available on the quantum computer. By default, the
        gates are those in DEFAULT_RESET_GATES.
    measurement_gates : Optional[Set[Type[OneQubitMeasurementGate]]]
        Set of the one-qubit measurement gates available on the quantum computer.
        By default, the gates are those in DEFAULT_MEASUREMENT_GATES.
    """

    def __init__(
        self,
        one_qubit_gates: Optional[Set[Type[OneQubitCliffordGate]]] = None,
        two_qubit_gates: Optional[Set[Type[TwoOperandGate]]] = None,
        reset_gates: Optional[Set[Type[OneQubitResetGate]]] = None,
        measurement_gates: Optional[Set[Type[_MeasurementGate]]] = None,
    ):
        one_qubit_gates_and_times = (
            {gate: 1.0 for gate in one_qubit_gates}
            if one_qubit_gates is not None
            else None
        )
        two_qubit_gates_and_times = (
            {gate: 1.0 for gate in two_qubit_gates}
            if two_qubit_gates is not None
            else None
        )
        reset_gates_and_times = (
            {gate: 1.0 for gate in reset_gates} if reset_gates is not None else None
        )
        measurement_gates_and_times = (
            {gate: 1.0 for gate in measurement_gates}
            if measurement_gates is not None
            else None
        )

        super().__init__(
            one_qubit_gates=one_qubit_gates_and_times,
            two_qubit_gates=two_qubit_gates_and_times,
            reset_gates=reset_gates_and_times,
            measurement_gates=measurement_gates_and_times,
        )


class ExhaustiveGateSet(NativeGateSet):
    """
    Class for capturing gateset of a quantum computer that can perform all gates
    natively.
    """

    def __init__(self):
        super().__init__(
            one_qubit_gates=ONE_QUBIT_GATES,
            two_qubit_gates=TWO_QUBIT_GATES,
            reset_gates=RESET_GATES,
            measurement_gates=MEASUREMENT_GATES
        )
