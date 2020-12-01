import seaborn as sns
import matplotlib.pyplot as plt

from cml import *

model = CML_2D(lambda u: u, 20, 0.5, u0=lambda x: 1.0 if x == (10,10) else 0.0)

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
sns.heatmap(model.value, ax=axs[0], xticklabels=False, yticklabels=False)

for _ in range(30):
	model.step()

sns.heatmap(model.value,  ax=axs[1], xticklabels=False, yticklabels=False)

for _ in range(30):
	model.step()

sns.heatmap(model.value,  ax=axs[2], xticklabels=False, yticklabels=False)

fig.tight_layout()
fig.savefig('diffusion.png', bbox_inches='tight', pad_inches=0)
plt.show()