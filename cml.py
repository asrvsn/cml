import numpy as np
import networkx as nx
from typing import Callable, Set, List
import pdb

class CML:
	def __init__(self, f: Callable, L: Set, N: Callable, eps: float, u0: Callable=None):
		self.f = f
		self.L = L
		self.N = N
		self.eps = eps
		if u0 is None:
			u0 = lambda _: np.random.normal()
		self.index = {x: i for i, x in enumerate(L)}
		self.N_index = [np.array([self.index[y] for y in N(x)], dtype=np.intp) for x in L]
		self.u = np.array([u0(x) for x in L], dtype=np.float64)

	def step(self):
		neighbors = np.array([self.f(self.u[ns]).sum()/ns.shape[0] for ns in self.N_index])
		self.u = (1-self.eps)*self.f(self.u) + self.eps*neighbors

	@property 
	def value(self):
		return self.u

class CML_1D(CML):
	def __init__(self, f: Callable, n: int, eps: float, u0: Callable=None, periodic: bool=True):
		self.G = nx.grid_2d_graph(n, 1, periodic=periodic)
		L = self.G.nodes()
		N = lambda x: self.G.neighbors(x)
		super().__init__(f, L, N, eps, u0=u0)

class CML_2D(CML):
	def __init__(self, f: Callable, n: int, eps: float, u0: Callable=None, periodic: bool=True):
		self.G = nx.grid_2d_graph(n, n, periodic=periodic)
		L = self.G.nodes()
		N = lambda v: self.G.neighbors(v)
		super().__init__(f, L, N, eps, u0=u0)
		self.V = np.array([[self.index[(x,y)] for x in range(n)] for y in range(n)], dtype=np.intp)

	@property 
	def value(self):
		return self.u[self.V]