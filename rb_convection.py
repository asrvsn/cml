''' Rayleigh-Benard convection ''' 

import seaborn as sns
import matplotlib.pyplot as plt
import pdb
import numpy as np
from typing import Tuple
import colorcet as cc
from tqdm import tqdm

from cml import *

np.random.seed(9999)

class CML_convection:
	def __init__(self, shape: Tuple, temp: float, nu: float, eta: float, kappa: float, c: float, dt: float=1.0):
		'''
		shape: (m, n) grid
		temp: temperature difference between upper and lower plates
		nu: viscosity
		kappa: thermal diffusion coefficient
		eta: pressure effect
		kappa: thermal expansion coefficient
		dt (optional): rate of particle advection 
		'''
		self.shape = shape
		self.temp = temp
		self.dt = dt
		self.nu = nu
		self.eta = eta
		self.kappa = kappa
		self.c = c
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

	def step(self):
		# Euler step
		self.T += self.kappa*self.laplace(self.T)
		self.Vy += -self.c*self.laplace_x(self.T)
		self.Vy += self.nu*self.laplace(self.Vy) + self.eta*(self.laplace_y(self.Vy) + self.divergence(self.Vx))
		self.Vx += self.nu*self.laplace(self.Vx) + self.eta*(self.laplace_x(self.Vx) + self.divergence(self.Vy))

		# Lagrange step
		self.T = self.advect(self.T)
		self.Vy = self.advect(self.Vy)
		self.Vx = self.advect(self.Vx)

		# Boundary conditions
		self.T[0,:] = -self.temp
		self.T[-1,:] = self.temp

	def reset(self):
		self.T = np.zeros(self.shape)
		self.T[0,:] = -self.temp
		self.T[-1,:] = self.temp
		# Symmetry-breaking initial conditions
		self.Vx = np.random.normal(scale=1e-3, size=self.shape)
		self.Vy = np.random.normal(scale=1e-3, size=self.shape)

if __name__ == '__main__':
	m, n = 25, 50
	c = 1.0
	temp = 10.0
	nu = 0.5
	eta = 0.3
	kappa = 0.8
	model = CML_convection((m, n), temp, nu, eta, kappa, c)

	# pdb.set_trace()

	''' Save as plot ''' 

	for _ in range(600):
		model.step()

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.contourf(model.T, 20, cmap='inferno_r')
	ax.axis('off')

	fig.tight_layout()
	fig.savefig('rb_convection.png', bbox_inches='tight', pad_inches=0)

	fig, ax = plt.subplots(figsize=(12, 6))
	ax.quiver(model.Vx, model.Vy)
	ax.axis('off')

	fig.tight_layout()
	fig.savefig('rb_convection_field.png', bbox_inches='tight', pad_inches=0)

	plt.show()

	''' Save as video ''' 
	# from matplotlib.animation import FFMpegWriter

	# writer = FFMpegWriter(fps=15, metadata={'title': 'rb_convectoin'})
	# fig, axs = plt.subplots(1, 2, figsize=(13, 4))

	# with writer.saving(fig, 'rb_convection.mp4', dpi=100):
	# 	model.reset()
	# 	for _ in tqdm(range(1000)):
	# 		axs[0].clear()
	# 		axs[1].clear()
	# 		axs[0].contourf(model.T, 20, cmap='inferno_r')
	# 		axs[1].quiver(model.Vx, model.Vy)
	# 		writer.grab_frame()
	# 		model.step()