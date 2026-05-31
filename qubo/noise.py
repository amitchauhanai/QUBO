
import numpy as np

def bit_flip(state, p, target):
	"""Bit-flip noise channel on target qubit with probability p."""
	n = int(np.log2(len(state)))
	new_state = state.copy()
	for i in range(len(state)):
		if np.random.rand() < p:
			j = i ^ (1 << target)
			new_state[i], new_state[j] = new_state[j], new_state[i]
	return new_state

def phase_flip(state, p, target):
	"""Phase-flip noise channel on target qubit with probability p."""
	n = int(np.log2(len(state)))
	new_state = state.copy()
	for i in range(len(state)):
		if ((i >> target) & 1) and np.random.rand() < p:
			new_state[i] *= -1
	return new_state

def depolarizing(state, p, target):
	"""Depolarizing noise channel on target qubit with probability p."""
	if np.random.rand() < p:
		# Randomly apply X, Y, or Z
		op = np.random.choice(['X', 'Y', 'Z'])
		if op == 'X':
			return bit_flip(state, 1.0, target)
		elif op == 'Y':
			# Y = iXZ
			s = bit_flip(state, 1.0, target)
			s = phase_flip(s, 1.0, target)
			return 1j * s
		elif op == 'Z':
			return phase_flip(state, 1.0, target)
	return state

def amplitude_damping(state, p, target):
	"""Amplitude damping channel (simplified, not full Kraus)."""
	n = int(np.log2(len(state)))
	new_state = state.copy()
	for i in range(len(state)):
		if ((i >> target) & 1) and np.random.rand() < p:
			j = i & ~(1 << target)
			new_state[j] += new_state[i]
			new_state[i] = 0
	return new_state


def make_noise_hook(channels: dict):
	"""Return a noise_hook(state, gate) which applies channels when gate.targets mention a qubit.

	channels: mapping from gate name or 'global' to a tuple (func, p) where func(state,p,target) -> state
	Example: {'bitflip': (bit_flip, 0.01)}
	"""

	def hook(state, gate):
		# apply all channel functions listed in channels to gate.targets
		for name, (func, p) in channels.items():
			for t in gate.targets:
				state = func(state, p, t)
		return state

	return hook
