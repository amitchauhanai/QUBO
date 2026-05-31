import copy
import numpy as np
from typing import Callable
from qubo.simulator import StatevectorSimulator


def expectation_z(state: np.ndarray, qubit: int) -> float:
    """Return expectation <Z_q> from a statevector."""
    n = int(np.log2(len(state)))
    probs = np.abs(state) ** 2
    exp = 0.0
    for idx, amp in enumerate(state):
        bit = (idx >> qubit) & 1
        val = 1.0 if bit == 0 else -1.0
        exp += val * (abs(amp) ** 2)
    return float(exp)


def parameter_shift_gradient(circuit, gate_index: int, observable: Callable[[np.ndarray], float], shots: int = 0):
    """Compute gradient d/dθ f(θ) for a parameterized gate at gate_index using the parameter-shift rule.

    circuit: QuantumCircuit containing a parameterized gate with params[0] = theta
    gate_index: index in circuit.gates of the parameterized gate
    observable: function(statevector) -> scalar (expectation)
    shots: if >0, run sampler with shots and estimate expectation from counts (not used here)
    """
    g = circuit.gates[gate_index]
    if not g.params:
        raise ValueError("Gate has no parameter to differentiate")
    theta = float(g.params[0])
    shift = np.pi / 2

    # forward
    c_plus = copy.deepcopy(circuit)
    c_plus.gates[gate_index].params[0] = theta + shift
    sim_plus = StatevectorSimulator(c_plus)
    res_plus = sim_plus.run(shots=shots) if shots > 0 else sim_plus.run()
    if isinstance(res_plus, dict):
        # convert counts to probabilities to compute expectation via samples - not implemented
        raise NotImplementedError("Shot-based gradient not implemented")
    f_plus = observable(res_plus)

    # backward
    c_minus = copy.deepcopy(circuit)
    c_minus.gates[gate_index].params[0] = theta - shift
    sim_minus = StatevectorSimulator(c_minus)
    res_minus = sim_minus.run(shots=shots) if shots > 0 else sim_minus.run()
    if isinstance(res_minus, dict):
        raise NotImplementedError("Shot-based gradient not implemented")
    f_minus = observable(res_minus)

    grad = 0.5 * (f_plus - f_minus)
    return float(grad)
