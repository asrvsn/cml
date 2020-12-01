import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import colorcet as cc

from cml import *

np.random.seed(9999)

class CML_kuramoto(CML_1D):
	def __init__(self, n: int, K: float, omega: float, eps: float):
		f = lambda u: np.mod(u + omega - K*np.sin(2*np.pi*u)/(2*np.pi), 1)
		u0 = lambda x: np.random.uniform()
		super().__init__(f, n, eps, u0=u0)

	def coherence(self):
		return np.sin(2*np.pi(self.value - self.value.mean()) + np.pi/2).mean()

model = CML_kuramoto(100, 0.1, 0.44, 0.3)

data = []
for _ in range(70):
	data.append(model.value)
	model.step()

data = np.array(data).T
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(data, cmap=cc.bgy, ax=ax, cbar=False)
ax.set_yticklabels([])
ax.set_xlabel('Time')

fig.tight_layout()
fig.savefig('kuramoto_1d.png', bbox_inches='tight', pad_inches=0)
plt.show()