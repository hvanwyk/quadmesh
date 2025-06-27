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
from function import Nodal, Constant
from assembler import Assembler, Form, Kernel
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

#
# Mesh
#
n_levels = 10 
mesh = Mesh1D(resolution=(1,), box=[0, 1])
mesh.record(0)
for l in range(1, n_levels + 1):
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
cov = Covariance(dh,name='matern',parameters={'sgm': 1,'nu': 0.1, 'l':0.01})
comment.toc()

# Create Gaussian random field
comment.tic("Create Gaussian random field")
eta = GaussianField(dh.n_dofs(), covariance=cov)
comment.toc()

# Sample from the Gaussian random field
comment.tic("Sample from Gaussian random field")
eta_smpl = eta.sample()
comment.toc()

q = Nodal(basis=v_ref, data=0.0001 + 50*np.exp(eta_smpl))

# Plot the samples
comment.tic("Plot samples")
plot = Plot(quickview=False)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax = plot.line(q, axis=ax, plot_kwargs={'color':'black', 'alpha':0.1})
ax.set_title("Sample from Gaussian Random Field")
ax.set_xlabel("x")
ax.set_ylabel("Value")

plt.tight_layout()
comment.toc()  
plt.show()

#
# Solve the advection-diffusion equation
# 
# Problem parameters
f =  Constant(100.0)  # Right-hand side function
b =  Constant(1.0)  # Advection coefficient



# Define the weak form of the advection-diffusion equation
FDiff = Form(Kernel(q), test=vx_ref, trial=vx_ref)
FAdv = Form(Kernel(b), test=vx_ref, trial=v_ref)
FSource = Form(Kernel(f), test=v_ref)

# Assemble the finite element system
comment.tic("Assemble the finite element system")   
problems = [FDiff, FAdv, FSource]
assembler = Assembler(problems, mesh)

# Add Dirichlet boundary conditions
assembler.add_dirichlet('left', Constant(0.0))
assembler.add_dirichlet('right', Constant(1.0))

assembler.assemble()
comment.toc()

# Solve the linear system
comment.tic("Solve the linear system")
u_ref = assembler.solve()
comment.toc()


# Plot the solution
comment.tic("Plot the solution")
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax = plot.line(Nodal(basis=v_ref, data=u_ref), axis=ax, plot_kwargs={'color':'black'})
ax.set_title("Solution of the Advection-Diffusion Equation")
ax.set_xlabel("x")
ax.set_ylabel("Value")

plt.tight_layout()
comment.toc()
plt.show()