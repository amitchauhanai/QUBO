import numpy as np
from qubo.circuit import QuantumCircuit
from qubo.gates import RZ
from qubo.autodiff import parameter_shift_gradient, expectation_z


def test_parameter_shift_rz():
    qc = QuantumCircuit(1)
    qc.add_gate('RZ', targets=[0], params=[0.5])

    def obs(state):
        return expectation_z(state, 0)

    g = parameter_shift_gradient(qc, 0, obs)
    assert isinstance(g, float)
