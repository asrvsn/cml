import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import logm
from numpy.linalg import eigvals
import pdb

from cml import *

np.random.seed(9999)

class CML_logistic(CML_2D):
	def __init__(self, n: int, r: float, eps: float=0.5):
		f = lambda u: r*u*(1-u)
		fprime = lambda u: r*(1-2*u)
		u0 = lambda x: np.random.uniform()
		super().__init__(f, n, eps, u0=u0, fprime=fprime)

n = 20
T = 100
t0 = 10

''' Lyapunov spectra of varying r ''' 

eps = 0.5
r_set = np.linspace(2.9, 4, 100)
spectra = []
for r in r_set:
	model = CML_logistic(n, r, eps=eps)
	for _ in range(t0):
		model.step()
	term = np.eye(n*n)
	for _ in range(T):
		jac = model.jacobian
		term = jac@term@jac.T
	term = logm(term) / (2*T)
	spectra.append(eigvals(term))
spectra = np.real(np.array(spectra))
# pdb.set_trace()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(r_set, np.zeros_like(r_set), color='black')
for i in range(spectra.shape[1]):
	ax.plot(r_set, spectra[:,i], color='grey')

fig.tight_layout()
fig.savefig('logistic_r_le.png', bbox_inches='tight', pad_inches=0)
plt.show()

''' Lyapunov spectra of varying eps ''' 

# r = 3.9
# eps_set = [0.0001]
# spectra = []
# for eps in eps_set:
# 	model = CML_logistic(n, r, eps=eps)
# 	for _ in range(t0):
# 		model.step()
# 	term = np.eye(n*n)
# 	for _ in range(T):
# 		jac = model.jacobian
# 		term = jac@term@jac.T
# 	term = logm(term) / (2*T)
# 	spectra.append(eigvals(term))

# fig, ax = plt.subplots(figsize=(12, 5))
# ax.scatter(spectra, r_set)

# fig.tight_layout()
# fig.savefig('logistic_eps_le.png', bbox_inches='tight', pad_inches=0)
# plt.show()