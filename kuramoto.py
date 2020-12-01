import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from cml import *

np.random.seed(9999)

class CML_kuramoto(CML):
	def __init__(self, n: int, K: float, omega: float, eps: float):
		L = set(range(n))
		N = lambda x: L
		f = lambda u: np.mod(u + omega - K*np.sin(2*np.pi*u)/(2*np.pi), 1)
		u0 = lambda x: np.random.uniform()
		super().__init__(f, L, N, eps, u0=u0)

	def coherence(self):
		return np.sin(2*np.pi(self.value - self.value.mean()) + np.pi/2).mean()

model = CML_kuramoto(100, 0.001, 0.5, 0.7)
r = np.ones_like(model.value)

fig, axs = plt.subplots(1, 3, figsize=(12, 4), subplot_kw={'projection': 'polar'})
axs[0].scatter(2*np.pi*model.value, r)
# axs[0].scatter(2*np.pi*model.value.mean(), model.coherence(), color='orange')
axs[0].set_ylim(0, 1.1)
axs[0].set_yticklabels([])

for _ in range(2):
	model.step()

axs[1].scatter(2*np.pi*model.value, r)
# axs[1].scatter(2*np.pi*model.value.mean(), model.coherence(), color='orange')
axs[1].set_ylim(0, 1.1)
axs[1].set_yticklabels([])

for _ in range(20):
	model.step()

axs[2].scatter(2*np.pi*model.value, r)
# axs[2].scatter(2*np.pi*model.value.mean(), model.coherence(), color='orange')
axs[2].set_ylim(0, 1.1)
axs[2].set_yticklabels([])

fig.tight_layout()
fig.savefig('kuramoto.png', bbox_inches='tight', pad_inches=0)
plt.show()