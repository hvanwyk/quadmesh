"""
Run a variable selection algorithm based on local linearization 

Step 1. Simulate 

Step 2. 
"""

from mesh import QuadMesh
from function import Explicit, Constant, Nodal
from fem import Basis, DofHandler, QuadFE
from assembler import Assembler, Form
from gmrf import GaussianField, Covariance
from diagnostics import Verbose
from plot import Plot
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def assemble_projection_matrix(v_fne, v_crs):
    """
    Description: 

        Assemble the projection matrix from a fine basis to a coarse basis.
    
    Inputs:

        v_fne: Basis, Fine basis functions.

        v_crs: Basis, Coarse basis functions.

    Outputs:

        P: np.ndarray, Projection matrix from fine to coarse basis.

            (v_crs, v_crs)P = (v_crs, v_fne)
    """
    mesh = v_fne.mesh()
    subff = v_fne.subforest_flag()

    # Define the forms
    problem_cc = [Form(trial=v_crs, test=v_crs)]
    problem_cf = [Form(trial=v_fne, test=v_crs)]

    # Define the assembler
    assembler = Assembler([problem_cc, problem_cf],
                          mesh=mesh, subforest_flag=subff)
    
    # Assemble the matrices
    assembler.assemble()

    # Extract the mass matrices
    M_cc = assembler.get_matrix(0).tocsc()
    M_cf = assembler.get_matrix(1).tocsc()

    # Solve for projection matrix
    P = spla.spsolve(M_cc, M_cf)

    return P
# 
# Generate initial mesh
#
mesh = QuadMesh(box=(0, 1, 0, 1), resolution=(10, 10))

# Refine Mesh
l_max = 1 
n_l = l_max + 1
mesh.cells.record(0)
for l in range(l_max):
    mesh.cells.refine(new_label=l+1)

plot = Plot(quickview=False)
fig, ax = plt.subplots(1,n_l, figsize=(6,6/(n_l)))
for l in range(n_l):
    ax[l] = plot.mesh(mesh, subforest_flag=l, axis=ax[l])
plt.show()

#
# Define the Function Spaces
# 
DQ0 = QuadFE(mesh.dim(), 'DQ0')
Q1 = QuadFE(mesh.dim(), 'Q1')

#
# DoF Handlers
# 
# Discontinuous pw-constant space for parameters
dh_DQ0 = DofHandler(mesh, DQ0)
dh_DQ0.distribute_dofs()

# Piecewise constant basis functions for parameters
w = [Basis(dh_DQ0,'v', i) for i in range(n_l)]

# Continuous piecewise linear space for state
dh_Q1 = DofHandler(mesh, Q1)
dh_Q1.distribute_dofs()

# Continuous, piecewise linear basis functions for state
v = [Basis(dh_Q1,'v', i) for i in range(n_l)]
vx = [Basis(dh_Q1,'vx', i) for i in range(n_l)]
vy = [Basis(dh_Q1,'vy', i) for i in range(n_l)]

n_samples = 1000
n_dofs_w = w[l_max].n_dofs()

cov = Covariance(dh_DQ0, name='exponential', parameters={'l':0.2, 'sgm':1.0})
eta = GaussianField(n_dofs_w, covariance=cov)

q_sample = Nodal(basis=w[l_max], data=eta.sample(n_samples))

fig, ax = plt.subplots(3,3, figsize=(12,12))
plot = Plot(quickview=False)
for i in range(3):
    for j in range(3):
        ax[i,j] = plot.contour(q_sample, axis=ax[i,j], n_sample=i*3+j)
plt.show()

