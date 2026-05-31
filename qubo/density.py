import numpy as np
from typing import Callable, Optional


class DensityMatrixSimulator:
    """A minimal density-matrix simulator for small qubit counts.

    Note: This is intentionally simple and not optimized for many qubits.
    """

    def __init__(self, circuit, dtype=complex):
        self.circuit = circuit
        self.num_qubits = int(circuit.num_qubits)
        self.dim = 2 ** self.num_qubits
        self.rho = np.zeros((self.dim, self.dim), dtype=dtype)
        self.rho[0, 0] = 1.0

    def _kron(self, mats):
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def apply_unitary(self, U, targets):
        ops = []
        tq = set(targets)
        for i in range(self.num_qubits):
            if i in tq:
                ops.append(U)
            else:
                ops.append(np.eye(2, dtype=U.dtype))
        full = self._kron(ops)
        self.rho = full @ self.rho @ full.conj().T

    def apply_kraus(self, kraus_ops, targets):
        for t in targets:
            new_rho = np.zeros_like(self.rho)
            for K in kraus_ops:
                ops = []
                for i in range(self.num_qubits):
                    if i == t:
                        ops.append(K)
                    else:
                        ops.append(np.eye(2, dtype=K.dtype))
                full = self._kron(ops)
                new_rho += full @ self.rho @ full.conj().T
            self.rho = new_rho

    def measure_probabilities(self):
        probs = np.real(np.diag(self.rho))
        labels = [format(i, f'0{self.num_qubits}b') for i in range(len(probs))]
        return dict(zip(labels, probs.tolist()))

    def run(self, shots: int = 1024, noise_hook: Optional[Callable] = None):
        for gate in self.circuit.gates:
            name = gate.name
            if name == 'H':
                H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
                self.apply_unitary(H, gate.targets)
            elif name == 'X':
                X = np.array([[0, 1], [1, 0]], dtype=complex)
                self.apply_unitary(X, gate.targets)
            elif name == 'RX':
                theta = gate.params[0] if gate.params else 0.0
                RX = np.array([
                    [np.cos(theta / 2), -1j * np.sin(theta / 2)],
                    [-1j * np.sin(theta / 2), np.cos(theta / 2)],
                ], dtype=complex)
                self.apply_unitary(RX, gate.targets)
            elif name == 'RY':
                theta = gate.params[0] if gate.params else 0.0
                RY = np.array(
                    [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]],
                    dtype=complex,
                )
                self.apply_unitary(RY, gate.targets)
            elif name == 'RZ':
                theta = gate.params[0] if gate.params else 0.0
                RZ = np.array(
                    [[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=complex
                )
                self.apply_unitary(RZ, gate.targets)
            elif name == 'CNOT':
                control, target = gate.targets[0], gate.targets[1]
                CNOT = np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=complex,
                )
                self.apply_unitary(CNOT, [control, target])
            elif name == 'SWAP':
                a, b = gate.targets[0], gate.targets[1]
                SWAP = np.array(
                    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                    dtype=complex,
                )
                self.apply_unitary(SWAP, [a, b])
            elif name in ('M', 'Measure'):
                probs = self.measure_probabilities()
                labels = list(probs.keys())
                weights = np.array(list(probs.values()), dtype=float)
                results = np.random.choice(len(labels), size=shots, p=weights)
                counts = {}
                for r in results:
                    counts[labels[r]] = counts.get(labels[r], 0) + 1
                return counts

            if noise_hook is not None:
                try:
                    self.rho = noise_hook(self.rho, gate)
                except Exception:
                    pass

        return self.rho


# Kraus helpers

def amplitude_damping_kraus(p):
    K0 = np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)
    return [K0, K1]


def depolarizing_kraus(p):
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    K0 = np.sqrt(1 - p) * I
    K1 = np.sqrt(p / 3.0) * X
    K2 = np.sqrt(p / 3.0) * Y
    K3 = np.sqrt(p / 3.0) * Z
    return [K0, K1, K2, K3]
