"""
Variational Multiscale Method for Advection-Diffusion Equation


"""
from mesh import QuadMesh
from fem import QuadFE, Basis, DofHandler
from function import Explicit, Nodal, Constant
from assembler import Assembler, Form, Kernel
from plot import Plot
import matplotlib.pyplot as plt
import numpy as np
from gmrf import Covariance, GaussianField
from diagnostics import Verbose
from scipy.sparse.linalg import spsolve
from solver import LinearSystem
 
# Initialize plot
plot = Plot(quickview=False)
comment = Verbose()

#
# Mesh 
# 

# Computational domain
domain = [-2,2,-1,1]

# Boundary regions
infn = lambda x,y: (x==-2) and (-1<=y) and (y<=0)  # inflow boundary
outfn = lambda x,y: (x==2) and (0<=y) and (y<=1)  # outflow boundary

# Define the mesh
mesh = QuadMesh(box=domain, resolution=(20,10))

# Various refinement levels
for i in range(3):
    if i==0:
        mesh.record(0)
    else:
        mesh.cells.refine(new_label=i)
    
    # Mark inflow
    mesh.mark_region('inflow', infn, entity_type='half_edge', 
                     on_boundary=True, subforest_flag=i)
    
    # Mark outflow
    mesh.mark_region('outflow', outfn, entity_type='half_edge', 
                     on_boundary=True, subforest_flag=i)
    
    
#
# Plot meshes 
#  
""" 
fig, ax = plt.subplots(3,1)  
for i in range(3):
    ax[i] = plot.mesh(mesh,axis=ax[i], 
                      regions=[('inflow','edge'),('outflow','edge')],
                      subforest_flag=i)
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('y')
plt.show()
"""

#
# Define DofHandlers and Basis 
#

# Piecewise Constant
Q0 = QuadFE(2,'DQ0')  # element
dh0 = DofHandler(mesh,Q0)  # degrees of freedom handler
dh0.distribute_dofs()
v00 = Basis(dh0, subforest_flag=0)  # Q0 basis on coarsest level
v01 = Basis(dh0, subforest_flag=1)  # Q0 basis on intermediate level
v02 = Basis(dh0, subforest_flag=2)  # Q0 basis on finest level

# Piecewise Linear 
Q1 = QuadFE(2,'DQ1')  # linear element
dh1 = DofHandler(mesh,Q1)  # linear DOF handler
dh1.distribute_dofs()
v10 = Basis(dh1, subforest_flag=0)  # Q1 basis on coarsest level 
v11 = Basis(dh1, subforest_flag=1)  # Q1 basis on intermediate level 
v12 = Basis(dh1, subforest_flag=2)  # Q1 basis on finest level

# 
# Parameters
# 
a = Constant(1)  # advection parameter

# Diffusion coefficient
cov = Covariance(dh0,name='matern',parameters={'sgm': 1,'nu': 1, 'l':1})
Z = GaussianField(dh0.n_dofs(), K=cov)

"""
# Plot realizations of the diffusion coefficient
fig, ax = plt.subplots(3,1)
for i in range(3):
    qs = Nodal(basis=v02, data=np.exp(Z.sample()))
    ax[i] = plot.contour(qs,axis=ax[i])
plt.show()
"""

# Sample from the diffusion coefficient
q2 = Nodal(basis=v02, data=Z.sample())

# TODO: Assembly of shape functions defined over different submeshes. 

# Compute the average 
problem = [Form(trial=v00,test=v00), Form(kernel=q2, test=v00)]
assembler = Assembler(problem, mesh=mesh, subforest_flag=2)
assembler.assemble()

M = assembler.get_matrix()
b = assembler.get_vector()

solver = LinearSystem(v00,M,b)
solver.solve_system()
q1 = solver.get_solution()


fig, ax = plt.subplots(2,1)
for i,q in enumerate([q1,q2]):
    ax[i] = plot.contour(q,axis=ax[i])
plt.show()
