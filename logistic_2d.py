import seaborn as sns
import matplotlib.pyplot as plt

from cml import *

np.random.seed(9999)

class CML_logistic(CML_2D):
	def __init__(self, n: int, r: float, eps: float=0.5):
		f = lambda u: r*u*(1-u)
		u0 = lambda x: np.random.uniform()
		super().__init__(f, n, eps, u0=u0)

model = CML_logistic(40, 3.9, eps=0.5)

''' Save as plot ''' 
# fig, axs = plt.subplots(1, 4, figsize=(16, 4))
# sns.heatmap(model.value, ax=axs[0], xticklabels=False, yticklabels=False, cbar=False)

# for _ in range(10):
# 	model.step()

# sns.heatmap(model.value,  ax=axs[1], xticklabels=False, yticklabels=False, cbar=False)

# for _ in range(20):
# 	model.step()

# sns.heatmap(model.value,  ax=axs[2], xticklabels=False, yticklabels=False, cbar=False)

# for _ in range(100):
# 	model.step()

# sns.heatmap(model.value,  ax=axs[3], xticklabels=False, yticklabels=False, cbar=False)

# fig.tight_layout()
# fig.savefig('logistic_2d.png', bbox_inches='tight', pad_inches=0)
# plt.show()

''' Save temporal section ''' 
model.reset()
data = []
for _ in range(100):
	data.append(model.value[20,:])
	model.step()
data = np.array(data).T

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(data, ax=ax, cbar=False)
ax.set_yticklabels([])
ax.set_xlabel('Time')

fig.tight_layout()
fig.savefig('logistic_2d_section.png', bbox_inches='tight', pad_inches=0)
plt.show()

''' Save as video ''' 
# from matplotlib.animation import FFMpegWriter

# model.reset()
# writer = FFMpegWriter(fps=15, metadata={'title': 'logistic_2d'})
# fig, ax = plt.subplots(figsize=(4,4))

# with writer.saving(fig, 'logistic_2d.mp4', dpi=100):
# 	for _ in range(200):
# 		sns.heatmap(model.value, ax=ax, xticklabels=False, yticklabels=False, cbar=False)
# 		writer.grab_frame()
# 		model.step()