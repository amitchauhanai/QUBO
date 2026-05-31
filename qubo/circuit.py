
import numpy as np
from typing import List, Callable, Any, Union, Optional

class Gate:
	"""Base class for quantum gates."""
	def __init__(self, name: str, targets: List[int], params: List[Any] = None):
		self.name = name
		self.targets = list(targets)
		self.params = params or []

	def __repr__(self):
		return f"Gate({self.name}, targets={self.targets}, params={self.params})"



class QuantumCircuit:
	"""Simple quantum circuit class."""
	def __init__(self, num_qubits: int):
		self.num_qubits = int(num_qubits)
		self.gates: List[Gate] = []
		self.registers = {}

	def add_register(self, name: str, size: int):
		"""Add a named register (convenience)."""
		self.registers[name] = int(size)

	def add_gate(self, gate: Union[Gate, str], targets: List[int] = None, params: List[Any] = None):
		"""Add a Gate instance or provide name/targets/params.

		Examples:
			add_gate(Gate('H', [0]))
			add_gate('H', targets=[0])
		"""
		if isinstance(gate, Gate):
			self.gates.append(gate)
			return
		if isinstance(gate, str):
			if targets is None:
				raise ValueError('targets required when adding gate by name')
			self.gates.append(Gate(gate, targets, params))
			return
		# fallback
		raise TypeError('gate must be Gate or str')

	def draw(self):
		# Simple ASCII circuit diagram
		lines = [f"q[{i}]: " for i in range(self.num_qubits)]
		for gate in self.gates:
			for i in range(self.num_qubits):
				if i in gate.targets:
					lines[i] += f"--{gate.name}--"
				else:
					lines[i] += "-------"
		for l in lines:
			print(l)

	def __repr__(self):
		return f"QuantumCircuit(num_qubits={self.num_qubits}, gates={self.gates})"
