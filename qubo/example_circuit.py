from qubo.circuit import QuantumCircuit
from qubo.gates import H, X, Measure

qc = QuantumCircuit(2)
qc.add_gate(H(0))
qc.add_gate(X(1))
qc.add_gate(Measure(0))
qc.add_gate(Measure(1))
