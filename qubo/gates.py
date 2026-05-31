
from qubo.circuit import Gate


def H(target):
	"""Hadamard gate"""
	return Gate('H', [target])


def X(target):
	"""Pauli-X gate"""
	return Gate('X', [target])


def CNOT(control, target):
	"""Controlled-NOT gate"""
	return Gate('CNOT', [control, target])


def RZ(target, theta):
	"""Z-rotation by angle theta (radians)"""
	return Gate('RZ', [target], params=[theta])


def RX(target, theta):
	"""X-rotation by angle theta"""
	return Gate('RX', [target], params=[theta])


def RY(target, theta):
	"""Y-rotation by angle theta"""
	return Gate('RY', [target], params=[theta])


def SWAP(a, b):
	"""Swap qubits a and b"""
	return Gate('SWAP', [a, b])


def Measure(target):
	"""Measurement operation"""
	return Gate('M', [target])

