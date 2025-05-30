"""
In this code, we will use solutions related to the coarse parameter field to help compute the solution related to the fine-scale parameter field.

We will use the GMRES method to solve the linear system iteratively.

Problem:

    - nabla \cdot (a \nabla u) = 0, on \Omega
    u = 0 on \partial \Omega



"""
import sys


if '/home/hans-werner/git/quadmesh/src' not in sys.path:
    sys.path.append('/home/hans-werner/git/quadmesh/src')

# Import modules from quadmesh
from mesh import Mesh1D
from fem import QuadFE, DofHandler,Basis
from function import Nodal, Constant
from assembler import Assembler, Form, Kernel
from gmrf import Covariance, GaussianField
from plot import Plot

# Built-in modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import linalg as spla
from scipy import sparse
from diagnostics import Verbose

# Generate a 1D mesh
mesh = Mesh1D(box=[0, 1], resolution=(101,))

# Mark boundary
bnd_fn = lambda x: abs(x) < 1e-6 or abs(1 - x) < 1e-6
mesh.mark_region('bnd', bnd_fn, entity_type='vertex', on_boundary=True)

# Define the elements 
Q1 = QuadFE(mesh.dim(), 'Q1')  # Linear for output
dQ1 = DofHandler(mesh, Q1)
dQ1.distribute_dofs()
v = Basis(dQ1,'v')
vx = Basis(dQ1,'vx')

# Define the covariance function
cov = Covariance(dQ1, name='matern', parameters={'sgm':1, 'nu': 1.5, 'l': 0.1})
eta = GaussianField(dQ1.n_dofs(), covariance=cov)
eta_smpl = eta.sample(10)
eta_fn = Nodal(basis=v,data=0.1+np.exp(eta_smpl))

plot = Plot(quickview=False)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for i in range(10):
    plot.line(eta_fn,axis=ax,i_sample=i) 
ax.set_title('Sample realization of the random field')
ax.set_xlabel('x')
#plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
eta_coarse_smpl = eta.KL_sample(i_min=0, i_max=8)
eta_tail_smpl = eta.KL_sample(i_min=9, n_samples=10)
eta_cond_smpl = np.zeros((dQ1.n_dofs(), 10))
for i in range(10):
    eta_cond_smpl[:, i] = eta_coarse_smpl[:,0] + eta_tail_smpl[:, i]

eta_coarse_fn = Nodal(basis=v, data=0.1 + np.exp(eta_coarse_smpl))
eta_cond_fn = Nodal(basis=v, data=0.1 + np.exp(eta_cond_smpl))

ax = plot.line(eta_coarse_fn,axis=ax,i_sample=0,plot_kwargs={'color':'blue'})
for i in range(10):
    plot.line(eta_cond_fn,axis=ax,i_sample=i,plot_kwargs={'color':'black','alpha':0.5}) 
ax.set_title('Sample realization of the coarse random field')
ax.set_xlabel('x')
plt.ylim(0, 7)
#plt.show()

# Assemble the System
K_coarse = Kernel(f=eta_coarse_fn)
problems = [Form(kernel=K_coarse, trial=vx, test=vx),
            Form(kernel=Kernel(Constant(1)), test=v)]
assembler = Assembler(problems,mesh)
assembler.add_dirichlet('bnd')
assembler.assemble()
Ac = assembler.get_matrix()
bc = assembler.get_vector()
uc_vec = assembler.solve()

K_cond = Kernel(f=eta_cond_fn)
problems = [Form(kernel=K_cond, trial=vx, test=vx),
            Form(kernel=Kernel(Constant(1)), test=v)]
assembler = Assembler(problems,mesh)
assembler.add_dirichlet('bnd')
assembler.assemble()

uf_vec = np.zeros((dQ1.n_dofs(), 10))
for i in range(10):
    uf_vec[:,i] = assembler.solve(i_matrix=i)
print('uf_vec', uf_vec)
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
uc_fn = Nodal(basis=v, data=uc_vec)
uf_fn = Nodal(basis=v, data=uf_vec)
plot.line(uc_fn,axis=ax,i_sample=0,plot_kwargs={'color':'blue'})
for i in range(10):
    plot.line(uf_fn,axis=ax,i_sample=i,plot_kwargs={'color':'black','alpha':0.5})
    ax.set_xlabel('x')
plt.ylim(-0.1, 0.3)
plt.show()
"""
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
uc_fn = Nodal(basis=v, data=uc_vec)
ax = plot.line(uc_fn,axis=ax,i_sample=0,plot_kwargs={'color':'blue'})
plt.title('Solution of the coarse problem')
ax.set_xlabel('x')
plt.show()"""