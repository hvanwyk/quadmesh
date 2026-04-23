"""
This example shows how to generate samples of the solution of a PDE sequentially
for a sequence of Gaussian field inputs along a hierarchical mesh sequence. 
This can be thought of as a sample from a refinement tree. 

The PDE we solve is the steady-state heat equation

    -div(q grad u) = f,  in D
    u = 0, on Gamma_D
    q grad u . n = g, on Gamma_N

with a log-Gaussian diffusion coefficient q = exp(Z).

The solution will be computed on the finest level of the mesh hierarchy, but 
the Gaussian field samples will be generated sequentially along the mesh 
hierarchy.
"""
#
# Imports 
# 

# Python Imports
from re import I

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla


# QuadMesh Imports
import assembler
from mesh import Mesh1D, QuadMesh
from fem import Basis, DofHandler, QuadFE
from function import Nodal, Explicit, Constant
from plot import Plot
from gmrf import Covariance, GaussianField
from assembler import Form, Assembler, Kernel


def projection_matrix(v_fne, v_crs):
    """
    Description: 

        Compute the projection matrix from a fine basis to a coarse basis.
    
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

def flux(q, ux, uy, region=None):
    """
    Description:

        Compute the flux at the outflow boundary, i.e. 

         flux = -(xi*grad u) * n

    Inputs:

        q: double, diffusion coefficient.

        ux: double, x-derivative of solution.

        uy: double, y-derivative of solution.
    """
    n = region.unit_normal()
    return -q*(ux*n[0]+uy*n[1])

#
# Computational mesh
# 
mesh = QuadMesh(box=[0,2,0,1], resolution=(2,2))
mesh.record(0)
L = 5  # number of levels in the mesh hierarchy
for l in range(1, L):
    mesh.cells.refine()
    mesh.record(l)

# Mark Dirichlet boundaries
i_inflow = lambda x,y: np.abs(x)<1e-6 and (0<=y) and (y<=0.5)
i_outflow = lambda x,y: np.abs(x-2)<1e-6 and (0.5<=y) and (y<=1)
mesh.mark_region('inflow', i_inflow, entity_type='vertex', on_boundary=True)
mesh.mark_region('outflow', i_outflow, entity_type='vertex', on_boundary=True)
mesh.mark_region('outflow_edge', i_outflow, entity_type='half_edge', on_boundary=True)
mesh.mark_region('outflow_cell', i_outflow, entity_type='cell',\
                 strict_containment=False, on_boundary=True)
plot = Plot(quickview=False)
fig, ax = plt.subplots(L,3, figsize=(12, 3*L))
for l in range(L):
    ax[l,0] = plot.mesh(mesh, axis=ax[l,0], 
                        regions=[('inflow','vertex'), ('outflow','vertex')],
                        subforest_flag=l)
    ax[l,0].set_title(f"Level {l}")


#
# Finite element spaces
# 
# -----------------------------------------------------------------------------
# Approximation of the parameter field
# -----------------------------------------------------------------------------
# Element 
DQ0 = QuadFE(mesh.dim(), 'DQ0')

# DofHandler
dhQ0 = DofHandler(mesh, DQ0)
dhQ0.distribute_dofs()

# Basis
w = [Basis(dhQ0, 'w', subforest_flag=l) for l in range(L)]

# -----------------------------------------------------------------------------
# Approximation of the solution field
# -----------------------------------------------------------------------------

# Finite Element
Q1 = QuadFE(mesh.dim(), 'Q1')

# DofHandler
dhQ1 = DofHandler(mesh, Q1)
dhQ1.distribute_dofs()

# Basis
v = Basis(dhQ1, 'v')
v_x = Basis(dhQ1, 'vx')
v_y = Basis(dhQ1, 'vy')

#
# Gaussian random field
#  
cov = []  # list of covariance objects for each level
eta = []  # list of Gaussian fields for each level
P   = []  # list of projection matrices from fine to coarse basis for each level
for l in range(L):
    cov.append(Covariance(dhQ0, name='exponential', parameters={'l':0.2, 'sgm':1.0}, subforest_flag=l))
    eta.append(GaussianField(dhQ0.n_dofs(subforest_flag=l), covariance=cov[l]))

    # Projection matrix from fine to coarse basis
    if l>0:
        P.append(projection_matrix(w[l], w[l-1]))

# Plot samples from Gaussian field at each level
"""
for l in range(L):
    Z = eta[l].sample()
    Zs = Nodal(basis=w[l], data=Z)
    ax[l,1] = plot.contour(Zs, axis=ax[l,1])
    ax[l,1].set_title(f"Gaussian Field Sample at Level {l}")
"""


# Sample from the Gaussian field sequentially with conditioning on the previous level
Z = []
for l in range(L):
    if l==0:
        Z.append(eta[l].sample())
        
    else:
        Z.append(eta[l].condition(P[l-1], Z[l-1], n_samples=1))

# Plot the conditioned samples
for l in range(L):
    ax[l,1] = plot.contour(Nodal(basis=w[l], data=Z[l]), axis=ax[l,1])
    ax[l,1].set_title(f"Gaussian Field Sample at Level {l} with Conditioning")
    ax[l,1].set_axis_off()

# Solve the PDE sequentially for each sample of the Gaussian field
u, ux, uy = [], [], []
Q_flux = []

f = Constant(1)  # source term
g = Constant(0)  # Neumann boundary condition
u_inflow = Constant(1)  # Dirichlet boundary condition at inflow
u_outflow = Constant(0)  # Dirichlet boundary condition at outflow
for l in range(L):
    q = Nodal(basis=w[l], data=0.1+np.exp(Z[l]))
    problems = [Form(kernel=q, test=v_x, trial=v_x),
                Form(kernel=q, test=v_y, trial=v_y),
                Form(kernel=f, test=v)]
                
    assembler = Assembler(problems, mesh, subforest_flag=L-1)
    assembler.add_dirichlet(dir_marker='inflow', dir_fn=u_inflow)
    assembler.add_dirichlet(dir_marker='outflow', dir_fn=u_outflow)
    
    assembler.assemble()
    
    u_vec = assembler.solve()
    u.append(Nodal(data=u_vec, basis=v))
    ux.append(Nodal(data=u_vec, basis=v_x))
    uy.append(Nodal(data=u_vec, basis=v_y))

    # Plot the solution
    ax[l,2] = plot.contour(u[l], axis=ax[l,2])
    ax[l,2].set_title(f"Solution at Level {l}")
    ax[l,2].set_axis_off()

    #
    # Compute the ouflow flux at the outflow boundary
    #
    k_flux = Kernel(f=[q,ux[l],uy[l]], F=flux)
    flux_problem = [Form(kernel=k_flux, flag='outflow_edge', dmu='ds')]
    flux_assembler = Assembler(flux_problem, mesh, subforest_flag=L-1)
    flux_assembler.assemble(region_flag='outflow_cell')
    flux_assembled_form = flux_assembler.assembled_forms(0)

    print(f"Flux at level {l}: {flux_assembled_form}")
    #Q_flux.append(flux_assembler.get_scalar(i_problem=0))
    #print(f"Flux at level {l}: {Q_flux[-1]}")
    
plt.tight_layout()
plt.show()


