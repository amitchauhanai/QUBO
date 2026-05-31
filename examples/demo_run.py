# Demo script for the qubo package
from qubo.circuit import QuantumCircuit
from qubo.gates import H, CNOT, RZ
from qubo.simulator import StatevectorSimulator

# build a 2-qubit Bell state with an RZ on qubit 0
qc = QuantumCircuit(2)
qc.add_gate(H(0))
qc.add_gate(CNOT(0, 1))
qc.add_gate(RZ(0, 3.1415/2))

print('Circuit:')
qc.draw()

sim = StatevectorSimulator(qc, backend='numpy')
res = sim.run()

# If measurement returned, print counts; else print probabilities
import numpy as np
if isinstance(res, dict):
    print('Counts:', res)
else:
    probs = (np.abs(res) ** 2).real
    labels = [format(i, f'0{qc.num_qubits}b') for i in range(len(probs))]
    print('Probabilities:')
    for l, p in zip(labels, probs):
        print(l, p)

# Quick import smoke tests
print('\nImport smoke test:')
import qubo
print('Imported qubo:', qubo.__name__)
print('Done demo')
