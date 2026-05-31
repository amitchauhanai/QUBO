import pytest
from qubo.circuit import QuantumCircuit, Gate


def test_add_and_repr():
	qc = QuantumCircuit(2)
	qc.add_gate('H', targets=[0])
	qc.add_gate(Gate('X', [1]))
	assert qc.num_qubits == 2
	assert len(qc.gates) == 2
	assert 'H' in repr(qc)

