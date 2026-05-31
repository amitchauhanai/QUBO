
import click
import importlib.util
import sys
import os
import json

# Ensure the package root is available when running this script directly from inside the package folder.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(_this_dir)
if _package_root not in sys.path:
    sys.path.insert(0, _package_root)


@click.command()
@click.argument('circuit_file')
@click.option('--shots', '-s', default=1024, help='Number of measurement shots (if measuring).', type=int)
@click.option('--seed', default=None, help='RNG seed for deterministic sampling.', type=int)
@click.option('--json', 'as_json', is_flag=True, default=False, help='Output results as JSON')
@click.option('--backend', default='numpy', help='Backend to use: "numpy" or "jax" (if available)')
@click.option('--simulator', default='statevector', type=click.Choice(['statevector', 'density'], case_sensitive=False), help='Simulator type to use.')
@click.option('--noise', default='none', type=click.Choice(['none', 'bit-flip', 'phase-flip', 'depolarizing', 'amplitude-damping'], case_sensitive=False), help='Noise model to apply during simulation.')
@click.option('--noise-prob', default=0.01, help='Noise probability for noise channels.', type=float)
def main(circuit_file, shots: int, seed: int, as_json: bool, backend: str, simulator: str, noise: str, noise_prob: float):
	"""Run a quantum circuit from a Python file and print results.

	Example:
		qubo-cli my_circuit.py --shots 1000 --json
	"""
	spec = importlib.util.spec_from_file_location("user_circuit", circuit_file)
	user_mod = importlib.util.module_from_spec(spec)
	sys.modules["user_circuit"] = user_mod
	spec.loader.exec_module(user_mod)
	if hasattr(user_mod, 'qc'):
		noise_hook = None
		if noise.lower() != 'none':
			from qubo import noise as qubo_noise
			func = {
				'bit-flip': qubo_noise.bit_flip,
				'phase-flip': qubo_noise.phase_flip,
				'depolarizing': qubo_noise.depolarizing,
				'amplitude-damping': qubo_noise.amplitude_damping,
			}.get(noise.lower())
			if func is not None:
				p = noise_prob if noise_prob is not None else 0.01
				def _hook(state, gate, _f=func, _p=p):
					for t in gate.targets:
						state = _f(state, _p, t)
					return state
				noise_hook = _hook

		if simulator.lower() == 'density':
			from qubo.density import DensityMatrixSimulator
			sim = DensityMatrixSimulator(user_mod.qc)
			result = sim.run(shots=shots, noise_hook=noise_hook)
		else:
			from qubo.simulator import StatevectorSimulator
			sim = StatevectorSimulator(user_mod.qc, seed=seed, backend=backend, noise_hook=noise_hook)
			result = sim.run(shots=shots)

		if isinstance(result, dict):
			if as_json:
				print(json.dumps({"counts": result}))
			else:
				print(result)
		else:
			import numpy as _np
			if hasattr(result, 'ndim') and result.ndim == 2:
				probs = _np.real(_np.diag(result))
			else:
				probs = (_np.abs(result) ** 2).real
			labels = [format(i, f'0{user_mod.qc.num_qubits}b') for i in range(len(probs))]
			out = {labels[i]: float(probs[i]) for i in range(len(probs))}
			if as_json:
				print(json.dumps({"probabilities": out}))
			else:
				print('Probabilities:')
				for k, v in out.items():
					print(k, v)
	else:
		print("Please define a 'qc' QuantumCircuit in your file.")


if __name__ == '__main__':
	main()
