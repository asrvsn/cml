import numpy as np
import networkx as nx
from typing import Callable, Set, List
import pdb

class CML:
	def __init__(self, f: Callable, L: Set, N: Callable, eps: float, u0: Callable=None, fprime: Callable=None):
		self.f = f
		self.L = L
		self.N = N
		self.eps = eps
		if u0 is None:
			u0 = lambda _: np.random.normal()
		self.u0 = u0
		self.index = {x: i for i, x in enumerate(L)}
		self.N_index = [np.array([self.index[y] for y in N(x)], dtype=np.intp) for x in L]
		self.u = np.array([u0(x) for x in L], dtype=np.float64)
		n = self.u.shape[0]
		self.fprime = fprime
		if fprime is not None:
			self.A = np.zeros((n,n))
			for i in range(n):
				for j in range(n):
					if i == j:
						self.A[i,j] = 1-eps
					elif j in self.N_index[i]:
						self.A[i,j] = eps/len(self.N_index[i])
					else:
						self.A[i,j] = 0

	def step(self):
		neighbors = np.array([self.f(self.u[ns]).sum()/ns.shape[0] for ns in self.N_index])
		self.u = (1-self.eps)*self.f(self.u) + self.eps*neighbors

	def reset(self):
		self.u = np.array([self.u0(x) for x in self.L], dtype=np.float64)

	@property 
	def value(self):
		return self.u

	@property
	def jacobian(self):
		assert self.fprime is not None
		return np.diag(self.fprime(self.u)) @ self.A

class CML_1D(CML):
	def __init__(self, f: Callable, n: int, eps: float, u0: Callable=None, periodic: bool=True, **kwargs):
		self.G = nx.grid_2d_graph(n, 1, periodic=periodic)
		L = self.G.nodes()
		N = lambda x: self.G.neighbors(x)
		super().__init__(f, L, N, eps, u0=u0, **kwargs)

class CML_2D(CML):
	def __init__(self, f: Callable, n: int, eps: float, u0: Callable=None, periodic: bool=True, **kwargs):
		self.G = nx.grid_2d_graph(n, n, periodic=periodic)
		L = self.G.nodes()
		N = lambda v: self.G.neighbors(v)
		super().__init__(f, L, N, eps, u0=u0, **kwargs)
		self.V = np.array([[self.index[(x,y)] for x in range(n)] for y in range(n)], dtype=np.intp)

	@property 
	def value(self):
		return self.u[self.V]