import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import logm
from numpy.linalg import eigvals
import numpy as np
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
T = 10
t0 = 10

''' Lyapunov spectra of varying r ''' 

eps = 0.5
r_set = np.linspace(0, 4.0, 200)
spectra = []
for r in r_set:
	print(f'r: {r}')
	model = CML_logistic(n, r, eps=eps)
	for _ in range(t0):
		model.step()
	term = np.zeros(n)
	for _ in range(T):
		jac = model.jacobian
		Q, R = np.linalg.qr(jac, mode='complete')
		term += np.log(np.abs(np.diag(R)[:n]))
	term /= T
	spectra.append(term)
spectra = np.array(spectra)

fig, ax = plt.subplots(figsize=(12, 5))
for i in range(spectra.shape[1]):
	ax.scatter(r_set, spectra[:,i], color='blue', s=1)
envelope = spectra.max(axis=1)
ax.plot(r_set, envelope, color='black', alpha=0.5)
ax.plot(r_set, np.zeros_like(r_set), color='black')
ax.set_xlabel('Nonlinearity $r$')

fig.tight_layout()
fig.savefig('logistic_r_le.png', bbox_inches='tight', pad_inches=0)
plt.show()

''' Lyapunov spectra of varying eps ''' 

r = 3.2
eps_set = np.linspace(0, 1., 200)
spectra = []
for eps in eps_set:
	print(f'eps: {eps}')
	model = CML_logistic(n, r, eps=eps)
	for _ in range(t0):
		model.step()
	term = np.zeros(n)
	for _ in range(T):
		jac = model.jacobian
		Q, R = np.linalg.qr(jac, mode='complete')
		term += np.log(np.abs(np.diag(R)[:n]))
	term /= T
	spectra.append(term)
spectra = np.array(spectra)

fig, ax = plt.subplots(figsize=(12, 5))
for i in range(spectra.shape[1]):
	ax.scatter(eps_set, spectra[:,i], color='blue', s=1)
envelope = spectra.max(axis=1)
ax.plot(eps_set, envelope, color='black', alpha=0.5)
ax.plot(eps_set, np.zeros_like(eps_set), color='black')
ax.set_xlabel('Coupling constant $\\varepsilon$')

fig.tight_layout()
fig.savefig('logistic_eps_le.png', bbox_inches='tight', pad_inches=0)
plt.show()