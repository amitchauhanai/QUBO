"""Optional JAX backend for faster simulation and autodiff.

This module is a best-effort adapter: if `jax` is not installed the functions
will raise ImportError or return a safe falsey indicator. Tests should skip
when jax is absent.
"""
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False


def is_available() -> bool:
    return JAX_AVAILABLE


if JAX_AVAILABLE:
    def rz_matrix(theta):
        return jnp.array([[jnp.exp(-0.5j * theta), 0.0], [0.0, jnp.exp(0.5j * theta)]], dtype=jnp.complex64)

    def apply_single(state, U, target, n_qubits):
        # reshape and apply via moveaxis/tensordot for JAX
        tensor = jnp.reshape(state, [2] * n_qubits)
        tensor = jnp.moveaxis(tensor, target, 0)
        flat = jnp.reshape(tensor, (2, -1))
        out = U @ flat
        out = jnp.reshape(out, [2] + [2] * (n_qubits - 1))
        out = jnp.moveaxis(out, 0, target)
        return jnp.reshape(out, (-1,))

    def apply_two_qubit(state, U4, a, b, n_qubits):
        tensor = jnp.reshape(state, [2] * n_qubits)
        # move axes a,b to front (0,1) preserving order
        tensor = jnp.moveaxis(tensor, (a, b), (0, 1))
        flat = jnp.reshape(tensor, (4, -1))
        out = U4 @ flat
        out = jnp.reshape(out, [2, 2] + [2] * (n_qubits - 2))
        out = jnp.moveaxis(out, (0, 1), (a, b))
        return jnp.reshape(out, (-1,))

    # two-qubit unitaries
    CNOT_4 = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=jnp.complex64)
    SWAP_4 = jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=jnp.complex64)

    def simulate_statevector(circuit, dtype=jnp.complex64):
        n = int(circuit.num_qubits)
        state = jnp.zeros((2 ** n,), dtype=dtype)
        state = state.at[0].set(1.0)
        for gate in circuit.gates:
            name = gate.name
            if name == 'H':
                H = (1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=dtype)
                state = apply_single(state, H, gate.targets[0], n)
            elif name == 'X':
                X = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=dtype)
                state = apply_single(state, X, gate.targets[0], n)
            elif name == 'RX':
                theta = gate.params[0] if gate.params else 0.0
                RX = jnp.array([[jnp.cos(theta / 2), -1j * jnp.sin(theta / 2)], [-1j * jnp.sin(theta / 2), jnp.cos(theta / 2)]], dtype=dtype)
                state = apply_single(state, RX, gate.targets[0], n)
            elif name == 'RY':
                theta = gate.params[0] if gate.params else 0.0
                RY = jnp.array([[jnp.cos(theta / 2), -jnp.sin(theta / 2)], [jnp.sin(theta / 2), jnp.cos(theta / 2)]], dtype=dtype)
                state = apply_single(state, RY, gate.targets[0], n)
            elif name == 'RZ':
                theta = gate.params[0] if gate.params else 0.0
                RZ = rz_matrix(theta)
                state = apply_single(state, RZ, gate.targets[0], n)
            elif name == 'CNOT':
                control, target = gate.targets[0], gate.targets[1]
                state = apply_two_qubit(state, CNOT_4, control, target, n)
            elif name == 'SWAP':
                a, b = gate.targets[0], gate.targets[1]
                state = apply_two_qubit(state, SWAP_4, a, b, n)
            elif name in ('M', 'Measure'):
                probs = jnp.abs(state) ** 2
                probs_np = jnp.asarray(probs).astype(float)
                # sampling via numpy for simplicity
                import numpy as _np
                results = _np.random.choice(len(probs_np), size=1024, p=probs_np)
                counts = {}
                for r in results:
                    b = format(int(r), f'0{n}b')
                    counts[b] = counts.get(b, 0) + 1
                return counts
            else:
                raise NotImplementedError(f'Gate {name} not implemented in jax backend')

        # return numpy array for compatibility
        return jnp.asarray(state)

    def grad_parameter_shift(circuit, gate_index, observable_fn):
        # Example: use JAX autodiff for parameterized gates by building a function
        def f_param(theta):
            # clone circuit and set parameter
            import copy
            c = copy.deepcopy(circuit)
            c.gates[gate_index].params[0] = float(theta)
            sv = simulate_statevector(c)
            # convert to numpy array and call observable
            return float(observable_fn(jnp.array(sv)))

        g = jax.grad(f_param)
        theta0 = circuit.gates[gate_index].params[0]
        return float(g(theta0))
