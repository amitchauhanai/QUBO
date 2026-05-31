import numpy as np
from typing import Callable, Optional


class StatevectorSimulator:
	"""Statevector simulator with a small gate set and noise hook.

	Supports: H, X, RX, RY, RZ, CNOT, SWAP, M(Measure).
	"""

	def __init__(self, circuit, noise_hook: Optional[Callable] = None, seed: Optional[int] = None, dtype: type = complex, backend: str = 'numpy'):
		self.circuit = circuit
		self.num_qubits = int(circuit.num_qubits)
		self.dtype = dtype
		self.state = np.zeros(2 ** self.num_qubits, dtype=self.dtype)
		self.state[0] = 1.0
		self.backend = backend
		# lazy import of optional backends
		self._jax = None
		if backend == 'jax':
			try:
				from qubo.backends import jax_backend as _jb
				if _jb.is_available():
					self._jax = _jb
				else:
					raise ImportError('jax not available')
			except Exception:
				self._jax = None
		self.noise_hook = noise_hook
		if seed is not None:
			np.random.seed(int(seed))

	def _apply_single_qubit_unitary(self, state: np.ndarray, U: np.ndarray, target: int) -> np.ndarray:
		"""Apply single-qubit unitary using moveaxis which is faster for large tensors."""
		n = self.num_qubits
		tensor = state.reshape([2] * n)
		# move target axis to front
		t = np.moveaxis(tensor, target, 0)
		t = t.reshape(2, -1)
		t = U @ t
		t = t.reshape([2] + [2] * (n - 1))
		# move axis back to original position
		t = np.moveaxis(t, 0, target)
		return t.reshape(-1)

	def _apply_cnot(self, state: np.ndarray, control: int, target: int) -> np.ndarray:
		n = self.num_qubits
		new = state.copy()
		for i in range(len(state)):
			if ((i >> control) & 1) == 1:
				j = i ^ (1 << target)
				new[i], new[j] = state[j], state[i]
		return new

	def _apply_swap(self, state: np.ndarray, a: int, b: int) -> np.ndarray:
		# SWAP by swapping amplitude indices
		n = self.num_qubits
		new = state.copy()
		for i in range(len(state)):
			bit_a = (i >> a) & 1
			bit_b = (i >> b) & 1
			if bit_a != bit_b:
				j = i ^ ((1 << a) | (1 << b))
				new[i] = state[j]
			else:
				new[i] = state[i]
		return new

	def run(self, shots: int = 1024):
		# if jax backend requested and available, delegate
		if self.backend == 'jax' and self._jax is not None:
			# jax backend returns jax array or counts
			res = self._jax.simulate_statevector(self.circuit, dtype=self.dtype)
			# convert to numpy if statevector
			try:
				import numpy as _np
				if hasattr(res, 'astype'):
					return _np.asarray(res)
			except Exception:
				return res

		state = self.state.copy()
		for gate in self.circuit.gates:
			name = gate.name
			if name == 'H':
				H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
				state = self._apply_single_qubit_unitary(state, H, gate.targets[0])
			elif name == 'X':
				X = np.array([[0, 1], [1, 0]], dtype=complex)
				state = self._apply_single_qubit_unitary(state, X, gate.targets[0])
			elif name == 'RX':
				theta = gate.params[0] if gate.params else 0.0
				RX = np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
				state = self._apply_single_qubit_unitary(state, RX, gate.targets[0])
			elif name == 'RY':
				theta = gate.params[0] if gate.params else 0.0
				RY = np.array([[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]], dtype=complex)
				state = self._apply_single_qubit_unitary(state, RY, gate.targets[0])
			elif name == 'RZ':
				theta = gate.params[0] if gate.params else 0.0
				RZ = np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=complex)
				state = self._apply_single_qubit_unitary(state, RZ, gate.targets[0])
			elif name == 'CNOT':
				control, target = gate.targets[0], gate.targets[1]
				state = self._apply_cnot(state, control, target)
			elif name == 'SWAP':
				a, b = gate.targets[0], gate.targets[1]
				state = self._apply_swap(state, a, b)
			elif name in ('M', 'Measure'):
				probs = np.abs(state) ** 2
				results = np.random.choice(len(probs), size=shots, p=probs)
				counts = {}
				for r in results:
					b = format(r, f'0{self.num_qubits}b')
					counts[b] = counts.get(b, 0) + 1
				return counts

			if self.noise_hook is not None:
				try:
					state = self.noise_hook(state, gate)
				except Exception:
					pass

		return state

