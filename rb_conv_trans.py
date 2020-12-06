''' Transition to R-B convection ''' 

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

m, n = 25, 50
c = 1.0
nu = 0.5
eta = 0.3
kappa = 0.8 

temp_rng = np.linspace(0, 3.0, 80)
data = []

for temp in tqdm(temp_rng):
	model = CML_convection((m, n), temp, nu, eta, kappa, c)
	for _ in range(300):
		model.step()
	mid = np.sqrt(model.Vx[12]**2 + model.Vy[12]**2)
	rms = np.sqrt((mid**2).mean())
	data.append(rms)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(temp_rng, data)

fig.tight_layout()
fig.savefig('rb_conv_trans.png', bbox_inches='tight', pad_inches=0)
plt.show()