''' Lyapunov spectra of Rayleigh-Benard convection ''' 

import seaborn as sns
import matplotlib.pyplot as plt
import pdb
import numpy as np
from typing import Tuple
import colorcet as cc
from tqdm import tqdm

from cml import *
from rb_convection import CML_convection

np.random.seed(9999)

if __name__ == '__main__':
	m, n = 15, 30
	c = 1.0
	nu = 0.5
	eta = 0.3
	kappa = 0.8

	temp_rng = np.linspace(0, 10., 100)

	t0 = 150
	tl = 10
	n_perturb = 4
	eps_perturb = 1e-3
	k_exponents = m*n

	spectra = []
	for temp in tqdm(temp_rng):
		model = CML_convection((m, n), temp, nu, eta, kappa, c)

		for _ in range(t0):
			model.step()

		perturbed = []
		for _ in range(n_perturb):
			model_ = CML_convection((m, n), temp, nu, eta, kappa, c)
			model_.T[:,:], model.Vx[:,:], model.Vy[:,:] = model.T[:,:], model.Vx[:,:], model.Vy[:,:]
			model_.T += np.random.normal(size=(m, n), scale=eps_perturb)
			perturbed.append(model_)

		spectrum = np.zeros(k_exponents)
		ds = np.vstack([(model_.T - model.T).ravel() for model_ in perturbed])
		for _ in range(tl):
			for model_ in perturbed:
				model_.step()
			model.step()
			ds_ = np.vstack([(model_.T - model.T).ravel() for model_ in perturbed])
			jac = np.linalg.lstsq(ds, ds_)[0]
			Q, R = np.linalg.qr(jac, mode='complete')
			spectrum += np.log(np.abs(np.diag(R)[:k_exponents]))
			ds = ds_
		spectra.append(spectrum)
	spectra = np.array(spectra)

	fig, ax = plt.subplots(figsize=(12, 5))
	for i in range(spectra.shape[1]):
		ax.scatter(temp_rng, spectra[:,i], color='blue', s=1)
	envelope = spectra.max(axis=1)
	ax.plot(temp_rng, envelope, color='black', alpha=0.5)
	ax.plot(temp_rng, np.zeros_like(temp_rng), color='black')
	ax.set_xlabel('External temperature $E$')

	fig.tight_layout()
	fig.savefig('rb_conv_lyapunov.png', bbox_inches='tight', pad_inches=0)
	plt.show()
