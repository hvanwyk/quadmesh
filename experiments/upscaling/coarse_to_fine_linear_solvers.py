"""
In this code, we will use solutions related to the coarse parameter field to help compute the solution related to the fine-scale parameter field.

We will use the GMRES method to solve the linear system iteratively.

Problem:

    - nabla \cdot (a \nabla u) = 1, on \Omega
    u = 0 on \partial \Omega



"""
#
# Imports 
# 
import sys

import plot
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
from scipy import sparse as sp
from diagnostics import Verbose
from scipy.sparse.linalg import LinearOperator


#
# Geometry and Mesh
# 

# Generate a 1D mesh
mesh = Mesh1D(box=[0, 1], resolution=(101,))

# Mark boundary
bnd_fn = lambda x: abs(x) < 1e-6 or abs(1 - x) < 1e-6
mesh.mark_region('bnd', bnd_fn, entity_type='vertex', on_boundary=True)


#
# Elements and Basis
# 

# Shape functions
Q1 = QuadFE(mesh.dim(), 'Q1')  # Linear for output

# Degree of freedom handler
dQ1 = DofHandler(mesh, Q1)
dQ1.distribute_dofs()

# Basis functions
v = Basis(dQ1,'v')
vx = Basis(dQ1,'vx')


#
# Gaussian Random Field
# 

# Define the covariance function
cov = Covariance(dQ1, name='matern', parameters={'sgm':1, 'nu': 1.5, 'l': 0.1})

# Define the Gaussian field
eta = GaussianField(dQ1.n_dofs(), covariance=cov)

# Sample the random field
eta_smpl = eta.sample(10)


#
#  Coarse system
# 
n_eigs = 5  # Number of eigenvalues to keep

# Generate coarse sample of coefficient's log
eta_coarse_smpl = eta.KL_sample(i_min=0, i_max=n_eigs)

# Sample of the coarse random field
eta_coarse_fn = Nodal(basis=v, data=0.1 + np.exp(eta_coarse_smpl))

# Assemble the coarse system
K_coarse = Kernel(f=eta_coarse_fn)
problems = [Form(kernel=K_coarse, trial=vx, test=vx),
            Form(kernel=Kernel(Constant(1)), test=v)]
assembler_coarse = Assembler(problems,mesh)
assembler_coarse.add_dirichlet('bnd')
assembler_coarse.assemble()

# Extract the coarse matrix and vector
Ac = assembler_coarse.get_matrix().tocsc()
bc = assembler_coarse.get_vector()

#
# Explicit solve
# 
# Get the interior degrees of freedom
int_dofs = assembler_coarse.get_dofs('interior')

x0 = assembler_coarse.assembled_bnd()

clu = spla.splu(Ac)

# Define the preconditioner as a LinearOperator
M = LinearOperator(Ac.shape, matvec=clu.solve)

err = []
def callback(xk):
    r = bc - Ac.dot(xk)
    err.append(np.linalg.norm(r))

# Use the preconditioner in CG
uc_int = spla.cg(Ac, bc - Ac.dot(x0), callback=callback)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.semilogy(np.array(err), label='Convergence of CG')
ax.set_xlabel('Iteration')
ax.set_ylabel('Residual Norm')
ax.set_title('Convergence of the Conjugate Gradient Method')
ax.legend()


# Resolve dirichlet nodes
dir_dofs, dir_vals = assembler_coarse.get_dirichlet(asdict=False)

# Create a vector for the solution
n_dofs = dQ1.n_dofs()
print(f"Number of degrees of freedom: {n_dofs}")
uc_vec = np.zeros(n_dofs)
uc_vec[dir_dofs] = dir_vals[:,0]
uc_vec[int_dofs] = uc_int[0]                           
#uc_vec = assembler_coarse.solve()

clu = spla.splu(Ac)
uc_lu = np.zeros(n_dofs)
uc_lu[int_dofs] = clu.solve(bc - Ac.dot(x0))
uc_lu[dir_dofs] = dir_vals[:,0]

assert np.allclose(uc_vec, uc_lu), "The solutions do not match!"

#
# Conditional Fine System
# 

# Sample the fine random field
eta_tail_smpl = eta.KL_sample(i_min=6, n_samples=20)

# Form the log-normal conditional random field
eta_cond_smpl = np.zeros((dQ1.n_dofs(), 10))
for i in range(10):
    eta_cond_smpl[:, i] = eta_coarse_smpl[:,0] + eta_tail_smpl[:, i]
eta_cond_fn = Nodal(basis=v, data=0.1 + np.exp(eta_cond_smpl))

# Assemble the conditional fine system
K_cond = Kernel(f=eta_cond_fn)
problems = [Form(kernel=K_cond, trial=vx, test=vx),
            Form(kernel=Kernel(Constant(1)), test=v)]
assembler_fine = Assembler(problems,mesh)
assembler_fine.add_dirichlet('bnd')
assembler_fine.assemble()


bf = assembler_fine.get_vector()

err_samples = []
for i in range(10):
    Af = assembler_fine.get_matrix(i_sample=i).tocsc()    
    uf = np.zeros(dQ1.n_dofs())
    err = []
    def callback(xk):
        r = bf - Af.dot(xk)
        err.append(np.linalg.norm(r))

    uf[int_dofs] = spla.cg(Af, bf - Af.dot(x0),M=M,callback=callback)[0]
    uf[dir_dofs] = dir_vals[:,0]
    err_samples.append(err)
    
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for i in range(10):
    ax.semilogy(np.array(err_samples[i]), color='k')
ax.set_xlabel('Iteration')
ax.set_ylabel('Residual Norm')
ax.set_title('Convergence of the Conjugate Gradient Method for Conditional Samples')
plt.show()  
"""

plot = Plot(quickview=False)
eta_fn = Nodal(basis=v,data=0.1+np.exp(eta_smpl))
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for i in range(10):
    plot.line(eta_fn,axis=ax,i_sample=i) 
ax.set_title('Sample realization of the random field')
ax.set_xlabel('x')
#plt.show()




fig, ax = plt.subplots(1, 1, figsize=(8, 4))


ax = plot.line(eta_coarse_fn,axis=ax,i_sample=0,plot_kwargs={'color':'blue'})
for i in range(10):
    plot.line(eta_cond_fn,axis=ax,i_sample=i,plot_kwargs={'color':'black','alpha':0.5}) 
ax.set_title('Sample realization of the coarse random field')
ax.set_xlabel('x')
plt.ylim(0, 7)
#plt.show()




print((Ac != Ac.T).nnz == 0)  # Check if the matrix is symmetric


spla.spchole_factor(Ac.tocsc(), use_umfpack=True)

print(dir(LU))


uf_vec = np.zeros((dQ1.n_dofs(), 10))
for i in range(10):
    uf_vec[:,i] = assembler_fine.solve(i_matrix=i)
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
"""
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
uc_fn = Nodal(basis=v, data=uc_vec)
ax = plot.line(uc_fn,axis=ax,i_sample=0,plot_kwargs={'color':'blue'})
plt.title('Solution of the coarse problem')
ax.set_xlabel('x')
plt.show()"""