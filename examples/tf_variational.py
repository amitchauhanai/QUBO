"""TensorFlow variational training example wrapping QUBO simulator calls.

This script demonstrates using a TensorFlow optimizer to apply gradients computed
by the `parameter_shift_gradient` (numeric) from `qubo.autodiff`.

Note: TensorFlow is optional. This example falls back to a pure-Python loop
if TensorFlow is not available.
"""

import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

from qubo.circuit import QuantumCircuit
from qubo.gates import RX
from qubo.copilot import train_variational


def observable_expectation_z_from_state(state):
    import numpy as _np
    exp = 0.0
    for idx, amp in enumerate(state):
        bit = (idx >> 0) & 1
        val = 1.0 if bit == 0 else -1.0
        exp += val * (abs(amp) ** 2)
    return float(abs(exp))


def run_tf_style_training(epochs=10, lr=0.1):
    qc = QuantumCircuit(1)
    qc.add_gate(RX(0, 0.5))

    if not TF_AVAILABLE:
        print('TensorFlow not available; running copilot.train_variational fallback')
        hist = train_variational(qc, [0], observable_expectation_z_from_state, epochs=epochs, lr=lr)
        print(hist)
        return

    # Build a simple TF variable to hold the parameter and a TF optimizer that will apply
    # externally-computed gradients.
    theta = tf.Variable(0.5, dtype=tf.float32)
    opt = tf.keras.optimizers.SGD(learning_rate=lr)

    def loss_for_theta(val):
        # set the circuit parameter
        qc.gates[0].params[0] = float(val)
        from qubo.simulator import StatevectorSimulator
        sim = StatevectorSimulator(qc)
        state = sim.run()
        return observable_expectation_z_from_state(state)

    for ep in range(epochs):
        # compute gradients via parameter-shift (numerical) using our copilot helper
        from qubo.autodiff import parameter_shift_gradient
        grad = parameter_shift_gradient(qc, 0, observable_expectation_z_from_state, shots=0)
        # apply using TF optimizer with externally computed gradients
        opt.apply_gradients([(tf.constant(grad, dtype=theta.dtype), theta)])
        print(f'epoch {ep} theta={theta.numpy():.4f} loss={loss_for_theta(theta.numpy()):.6f}')


if __name__ == '__main__':
    run_tf_style_training()
