
import os
import tempfile

_mpl_config_dir = os.path.join(tempfile.gettempdir(), 'qubo-matplotlib')
try:
	os.makedirs(_mpl_config_dir, exist_ok=True)
	os.environ.setdefault('MPLCONFIGDIR', _mpl_config_dir)
except Exception:
	pass

import matplotlib.pyplot as plt
import numpy as np

def plot_statevector(state):
	"""Bar chart of statevector amplitudes and probabilities."""
	n = int(np.log2(len(state)))
	probs = np.abs(state) ** 2
	labels = [format(i, f'0{n}b') for i in range(len(state))]
	fig, ax = plt.subplots(1,2, figsize=(10,4))
	ax[0].bar(labels, state.real, color='b', alpha=0.7)
	ax[0].set_title('Real part')
	ax[1].bar(labels, probs, color='g', alpha=0.7)
	ax[1].set_title('Probability')
	plt.tight_layout()
	plt.show()
