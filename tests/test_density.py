import numpy as np
from qubo.circuit import QuantumCircuit
from qubo.density import DensityMatrixSimulator, amplitude_damping_kraus


def test_density_amplitude_damping():
    qc = QuantumCircuit(1)
    sim = DensityMatrixSimulator(qc)
    # apply amplitude damping kraus to qubit 0
    K = amplitude_damping_kraus(0.5)
    sim.apply_kraus(K, targets=[0])
    probs = sim.measure_probabilities()
    assert abs(sum(probs.values()) - 1.0) < 1e-6
