import numpy as np
import pytest
from qubo.circuit import QuantumCircuit


def test_swap_gate():
    qc = QuantumCircuit(2)
    qc.add_gate('X', targets=[0])
    qc.add_gate('SWAP', targets=[0,1])
    from qubo.simulator import StatevectorSimulator
    sim = StatevectorSimulator(qc)
    res = sim.run()
    # Ensure statevector length and normalization
    assert len(res) == 4
    assert pytest.approx((abs(res)**2).sum(), rel=1e-9) == 1.0


def test_rx_ry():
    qc = QuantumCircuit(1)
    qc.add_gate('RX', targets=[0], params=[3.1415])
    qc.add_gate('RY', targets=[0], params=[1.234])
    from qubo.simulator import StatevectorSimulator
    sim = StatevectorSimulator(qc)
    state = sim.run()
    assert len(state) == 2
    assert pytest.approx((abs(state)**2).sum(), rel=1e-9) == 1.0
