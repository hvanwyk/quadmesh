"""
This script determines the spatial error convergence of the finite element solution of a 1D advection-diffusion equation for different parameter dimensions.
"""
import sys

import comm

if '/home/hans-werner/git/quadmesh/src' not in sys.path:
    sys.path.append('/home/hans-werner/git/quadmesh/src')

from mesh import QuadMesh, Mesh1D
from gmrf import Covariance, GaussianField
from fem import Basis, DofHandler, QuadFE   
from function import Nodal
from assembler import Assembler, Form
from gmrf import Covariance, GaussianField
from diagnostics import Verbose

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot import Plot

from scipy.sparse import linalg as spla
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np

comment = Verbose()

# Mesh
mesh = Mesh1D(resolution=(1,), box=[0, 1])
mesh.record(0)
for l in range(1, 11):
    mesh.cells.refine()
    mesh.record(l)

# Mark boundaries    
left_bnd =  lambda x: abs(x) < 1e-6
right_bnd = lambda x: abs(x-1) < 1e-6
mesh.mark_region('left', left_bnd, entity_type='vertex', on_boundary=True)
mesh.mark_region('right', right_bnd, entity_type='vertex', on_boundary=True)

# Element
Q1 = QuadFE(mesh.dim(), 'Q1')

# DofHandler
dh = DofHandler(mesh, Q1)
dh.distribute_dofs()

# Basis function on refined mesh
v_ref = Basis(dofhandler=dh, derivative='v')
vx_ref = Basis(dofhandler=dh, derivative='vx')

#
# Plot solution of the advection-diffusion equation
# 
  # Define Gaussian random field
comment.tic("Create Covariance")
cov = Covariance(dh,name='matern',parameters={'sgm': 1,'nu': 1, 'l':0.05})
comment.toc()

# Create Gaussian random field
comment.tic("Create Gaussian random field")
eta = GaussianField(dh.n_dofs(), covariance=cov)
comment.toc()

# Sample from the Gaussian random field
comment.tic("Sample from Gaussian random field")
n_samples = 100
eta_smpl = eta.sample(n_samples=n_samples)
comment.toc()

eta_fn = Nodal(basis=v_ref, data=eta_smpl)

# Plot the samples
comment.tic("Plot samples")
plot = Plot(quickview=False)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for i in range(n_samples):
    ax = plot.line(eta_fn, axis=ax,i_sample=i,
              plot_kwargs={'color':'black', 'alpha':0.1})

ax.set_title("Samples from Gaussian Random Field")
ax.set_xlabel("x")
ax.set_ylabel("Value")

plt.tight_layout()
comment.toc()  
plt.show()