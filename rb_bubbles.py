''' Rayleigh-Benard convection with bubble nucleation ''' 

import seaborn as sns
import matplotlib.pyplot as plt
import pdb
import numpy as np
from typing import Tuple
import colorcet as cc
from tqdm import tqdm

from cml import *

np.random.seed(9999)

class CML_boiling:
	def __init__(self, shape: Tuple, 
			temp: float, nu: float, eta1: float, kappa: float, c: float, 
			sigma: float, alpha: float, eta2: float, T_c: float,
			dt: float=1.0
		):
		'''
		shape: (m, n) grid
		temp: temperature difference between upper and lower plates
		nu: viscosity
		kappa: thermal diffusion coefficient
		eta1: pressure effect
		kappa: thermal expansion coefficient
		dt (optional): rate of particle advection 
		'''
		self.shape = shape
		self.temp = temp
		self.dt = dt
		self.nu = nu
		self.eta1 = eta1
		self.kappa = kappa
		self.c = c

		self.sigma = sigma
		self.alpha = alpha
		self.eta2 = eta2
		self.T_c = T_c

		self.reset()

	def laplace_x(self, u):
		ret = -u
		ret += np.c_[u[:,1:],u[:,0]] / 2
		ret += np.c_[u[:,-1],u[:,:-1]] / 2
		return ret

	def laplace_y(self, u):
		ret = -u
		ret[0] = 0
		ret[-1] = 0
		ret[1:-1] += (u[:-2] + u[2:]) / 2
		return ret

	def laplace(self, u):
		return (self.laplace_x(u) + self.laplace_y(u)) / 2

	def divergence(self, u):
		ret = np.zeros_like(u)
		m, n = self.shape
		for y in range(1, m-1):
			for x in range(n):
				ret[y, x] = (u[y-1,(x+1)%n] + u[y+1,(x-1)%n] - u[y-1,(x-1)%n] - u[y+1,(x+1)%n]) / 4
		return ret

	def advect(self, u):
		ret = np.zeros_like(u)
		m, n = self.shape
		for y in range(1, m-1):
			for x in range(n):
				# Particle displacements
				dx = self.dt*self.Vx[y,x]
				dy = -self.dt*self.Vy[y,x] # Numpy rows are in reverse
				# Bounding box
				x_ = (x+int(dx)) % n
				y_ = max(min(m-1, y+int(dy)), 0)
				x__ = (x_+int(np.sign(dx))) % n
				y__ = max(min(m-1, y_+int(np.sign(dy))), 0)
				# Lever rule 
				wx = np.abs(dx-int(dx))
				wy = np.abs(dy-int(dy))
				ret[y_,x_] += u[y,x]*(1-wx)*(1-wy)
				ret[y_,x__] += u[y,x]*wx*(1-wy)
				ret[y__,x_] += u[y,x]*(1-wx)*wy
				ret[y__,x__] += u[y,x]*wx*wy
		return ret

	def density(self, T):
		return np.tanh(self.alpha*(T-self.T_c))

	def step(self):
		# Euler step
		T = self.T + self.kappa*self.laplace(self.T)
		self.Vy += -self.c*self.laplace_x(T)
		self.Vy += self.nu*self.laplace(self.Vy) + self.eta1*(self.laplace_y(self.Vy) + self.divergence(self.Vx))
		self.Vx += self.nu*self.laplace(self.Vx) + self.eta1*(self.laplace_x(self.Vx) + self.divergence(self.Vy))

		# Bubble nucleation
		T[1:-1] -= (self.sigma/2)*T[1:-1]*(self.density(T[:-2]) - self.density(T[2:]))

		# Evaporative cooling / condensative heating
		m, n = self.shape
		for y in range (1, m-1):
			for x in range(n):
				mult = 0.
				if T[y, x] > self.T_c and self.T[y, x] < self.T_c:
					mult = -1.
				elif T[y, x] < self.T_c and self.T[y, x] > self.T_c:
					mult = 1.
				T[y, (x+1)%m] += mult*self.eta2
				T[y, (x-1)%m] += mult*self.eta2
				T[y-1, x] += mult*self.eta2
				T[y+1, x] += mult*self.eta2

		self.T = T

		# Lagrange step
		self.T = self.advect(self.T)
		self.Vy = self.advect(self.Vy)
		self.Vx = self.advect(self.Vx)

		# Boundary conditions
		self.T[0,:] = 0.
		self.T[-1,:] = self.temp

	def reset(self):
		# Symmetry-breaking initial conditions
		self.T = self.temp / 2 + np.random.normal(size=self.shape, scale=1e-3)
		self.T[0,:] = 0.
		self.T[-1,:] = self.temp
		self.Vx = np.random.normal(scale=1e-3, size=self.shape)
		self.Vy = np.random.normal(scale=1e-3, size=self.shape)

if __name__ == '__main__':
	m, n = 25, 50

	temp = 9.95

	nu = 0.1
	eta1 = 0.3
	kappa = 0.8
	c = 1.0

	sigma = 0.1
	alpha = 5
	eta2 = 0.1
	T_c = 10

	model = CML_boiling((m, n), temp, nu, eta1, kappa, c, sigma, alpha, eta2, T_c, dt=1.0)

	# pdb.set_trace()

	''' Save as plot ''' 

	for _ in range(900):
		model.step()

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.contourf(model.T, 20, cmap='inferno')
	ax.invert_yaxis()
	# sns.heatmap(model.T, ax=ax)
	ax.axis('off')

	fig.tight_layout()
	fig.savefig('rb_bubbles.png', bbox_inches='tight', pad_inches=0)

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.quiver(model.Vx, model.Vy)
	ax.axis('off')

	fig.tight_layout()
	fig.savefig('rb_bubbles_field.png', bbox_inches='tight', pad_inches=0)

	plt.show()

	''' Save as video ''' 
	# from matplotlib.animation import FFMpegWriter

	# writer = FFMpegWriter(fps=15, metadata={'title': 'rb_bubbles'})
	# fig, axs = plt.subplots(1, 2, figsize=(13, 4))

	# with writer.saving(fig, 'rb_bubbles.mp4', dpi=100):
	# 	model.reset()
	# 	for _ in tqdm(range(1000)):
	# 		axs[0].clear()
	# 		axs[1].clear()
	# 		axs[0].contourf(model.T, 20, cmap='inferno')
	# 		axs[0].invert_yaxis()
	# 		axs[1].quiver(model.Vx, model.Vy)
	# 		writer.grab_frame()
	# 		model.step()