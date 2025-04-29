"""
TODO: Unfinished
"""


import sys
sys.path.append('/home/hans-werner/git/quadmesh/src')

from assembler import Assembler
from assembler import Kernel
from assembler import Form
from fem import DofHandler
from fem import QuadFE
from fem import Basis
from function import Nodal
from gmrf import Covariance
from gmrf import GaussianField
from mesh import QuadMesh
from plot import Plot
import TasmanianSG
import time
from diagnostics import Verbose

# Built-in modules
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


"""
Investigate local error estimates on the resolution of a random field
"""

plot = Plot()

#
# Computational mesh
# 
mesh = QuadMesh(resolution=(4,4))


# Mark boundary
bnd_fn = lambda x,y: abs(x)<1e-6 or abs(1-x)<1e-6 or abs(y)<1e-6 or abs(1-y)<1e-6 
mesh.mark_region('bnd', bnd_fn, entity_type='half_edge', on_boundary=True)

# Mark averaging region
dmn_fn = lambda x,y: x>=0.75 and x<=1 and y>=0.75 and y<=1
mesh.mark_region('dmn', dmn_fn, entity_type='cell', strict_containment=True, on_boundary=False) 
#cells = mesh.get_region(flag='dmn', entity_type='cell', on_boundary=False, subforest_flag=None)
plot.mesh(mesh, regions=[('bnd','edge'),('dmn','cell')])

#
# Elements
#  
Q0 = QuadFE(mesh.dim(), 'DQ0')  # Constants for parameter
Q1 = QuadFE(mesh.dim(), 'Q1')  # Linear for output
Q2 = QuadFE(mesh.dim(), 'Q2')  # Quadratic for adjoint

#
# DofHandlers
# 
dQ0 = DofHandler(mesh,Q0)
dQ1 = DofHandler(mesh,Q1)
dQ2 = DofHandler(mesh,Q2)

# Distribute DOFs
dQ0.distribute_dofs()
dQ1.distribute_dofs()
dQ2.distribute_dofs()

# 
# Basis functions 
# 
phi_0 = Basis(dQ0)

phi_1 = Basis(dQ1)
phix_1 = Basis(dQ1,'vx')
phiy_1 = Basis(dQ1,'vy')

phi_2 = Basis(dQ2)
phix_2 = Basis(dQ2,'vx')
phiy_2 = Basis(dQ2,'vy')

#
# Define Random field
# 
cov = Covariance(dQ0, name='gaussian', parameters={'l':0.01})
cov.compute_eig_decomp()
q = GaussianField(dQ0.n_dofs(), K=cov)


# Sample Random field
n_samples = 100
eq = Nodal(basis=phi_0, data=np.exp(q.sample(n_samples)))

plot.contour(eq, n_sample=25)

#
# Compute state 
#

# Define weak form 
state = [[Form(eq, test=phix_1, trial=phix_1), 
          Form(eq, test=phiy_1, trial=phiy_1),
          Form(1, test=phi_1)],
         [Form(1,test=phi_1,flag='dmn')]]

# Assemble system
assembler = Assembler(state)
assembler.add_dirichlet('bnd')
assembler.assemble()

J = assembler.get_vector(1)

# Solve system
u_vec = assembler.solve()
u = Nodal(basis=phi_1, data=u_vec)

plot.contour(u)
plt.title('Sample Path')

# Solve the adjoint system
adjoint = [Form(eq, test=phix_2, trial=phix_2), 
           Form(eq, test=phiy_2, trial=phiy_2),
           Form(1, test=phi_2, flag='dmn')]

assembler = Assembler(adjoint)
assembler.add_dirichlet('bnd')
assembler.assemble()

#%%
z_data = np.zeros((dQ2.n_dofs(), n_samples))
for i in range(n_samples):
    z_data[:,i] = assembler.solve(i_matrix=i)

z = Nodal(basis=phi_2, data=z_data)

#%%
plot.contour(z, n_sample=25)

#%%
grid = TasmanianSG.TasmanianSparseGrid()
dimensions = 2
depth = 5
outputs = 1
type = 'level'
rule = 'gauss-hermite'
grid.makeGlobalGrid(dimensions, outputs, depth, type, rule)

# Get Sample Points
zzSG = grid.getPoints()
zSG = np.sqrt(2)*zzSG

  
