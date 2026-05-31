import numpy as np
import pytest
from qubo.circuit import QuantumCircuit
from qubo.simulator import StatevectorSimulator


def test_h_on_zero():
	qc = QuantumCircuit(1)
	qc.add_gate('H', targets=[0])
	sim = StatevectorSimulator(qc)
	state = sim.run()
	probs = np.abs(state) ** 2
	assert pytest.approx(probs.sum(), rel=1e-6) == 1.0

