import numpy as np
from qubo.noise import bit_flip, phase_flip, depolarizing, amplitude_damping


def test_noise_channels_apply():
	# start in |0> state
	state = np.array([1.0 + 0j, 0j])
	s1 = bit_flip(state, 1.0, 0)
	assert len(s1) == 2
	s2 = phase_flip(state, 1.0, 0)
	assert len(s2) == 2
	s3 = depolarizing(state, 1.0, 0)
	assert len(s3) == 2
	s4 = amplitude_damping(state, 1.0, 0)
	assert len(s4) == 2

