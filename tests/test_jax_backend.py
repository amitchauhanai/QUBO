import pytest
from qubo.backends import jax_backend


def test_jax_availability_or_skip():
    if not jax_backend.is_available():
        pytest.skip("JAX not installed; skipping jax backend test")
    # otherwise, at least ensure interface works for trivial RZ
    from qubo.circuit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.add_gate('RZ', targets=[0], params=[0.1])
    sv = jax_backend.simulate_statevector(qc)
    assert sv.shape[0] == 2
