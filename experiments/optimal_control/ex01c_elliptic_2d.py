import sys
sys.path.append('/home/hans-werner/git/quadmesh/src')

from assembler import Assembler
from assembler import Form
from assembler import Kernel

from diagnostics import Verbose

from fem import QuadFE
from fem import DofHandler
from fem import Basis

from function import Constant
from function import Explicit
from function import Map
from function import Nodal

from gmrf import Covariance
from gmrf import GaussianField

from mesh import QuadMesh

from plot import Plot

from solver import LinearSystem as LS

import numpy as np
import matplotlib.pyplot as plt
import gc
import scipy.sparse as sp
#from tqdm import tqdm
"""
System 

    -div(exp(K)*grad(y)) = b + Fu,  x in D
                       y = g     ,  x in D_Dir
        exp(K)*grad(y)*n = 0     ,  x in D_Neu
    
    
Random field:
    
    K ~ GaussianField 

Cost Functional
    
    J(u) = E(|y(u)-y_d|**2) + alpha/2*|u|**2-
"""

vb = Verbose()

# =============================================================================
# Mesh and Elements
# =============================================================================
# Finite element mesh

x_min = 0
x_max = 2
y_min = 0
y_max = 1 

mesh = QuadMesh(box=[x_min, x_max, y_min, y_max], resolution=(20,10))

# Mark Dirichlet Edges
mesh.mark_region('left', lambda x,y: np.abs(x)<1e-9, entity_type='half_edge')
mesh.mark_region('right', lambda x,y: np.abs(x-2)<1e-9, entity_type='half_edge')

# Element
element_Q0 = QuadFE(mesh.dim(), 'DQ0')
element_Q1 = QuadFE(mesh.dim(), 'Q1')

dh_Q0 = DofHandler(mesh, element_Q0)
dh_Q0.distribute_dofs()
n_Q0 = dh_Q0.n_dofs()

dh_Q1 = DofHandler(mesh, element_Q1)
dh_Q1.distribute_dofs()
n_Q1 = dh_Q1.n_dofs()

# Basis functions
phi   = Basis(dh_Q1, 'v')
phi_x = Basis(dh_Q1, 'vx')
phi_y = Basis(dh_Q1, 'vy')

# =============================================================================
# Parameters
# =============================================================================
# 

#
#  Locations of production wells (measurements)
# 
n_production = (4,3)  # resolution
x_production = np.linspace(0.5, 1.5, n_production[0])
y_production = np.linspace(0.2, 0.8, n_production[1])
X,Y = np.meshgrid(x_production, y_production)
xy = np.array([X.ravel(),Y.ravel()]).T
cells_production = mesh.bin_points(xy)

# Mark vertices 
for cell, dummy in cells_production:
    cell.get_vertex(2).mark('production')

# Extract degrees of freedom
production_dofs = dh_Q1.get_region_dofs(entity_type='vertex',
                                        entity_flag='production')

v_production = dh_Q1.get_dof_vertices(dofs=production_dofs)

# Target pressure at production wells
z_fn = Explicit(f=lambda x: 3-4*(x[:,0]-1)**2 - 8*(x[:,1]-0.5)**2, dim=2, mesh=mesh)
y_target = z_fn.eval(v_production)

#
# Locations of injection wells
# 
n_injection = (5,4)  # resolution
x_injection = np.linspace(0.5, 1.5, n_injection[0])
y_injection = np.linspace(0.25, 0.75, n_injection[1])
X,Y = np.meshgrid(x_injection, y_injection)
xy = np.array([X.ravel(),Y.ravel()]).T
cells_injection = mesh.bin_points(xy)

# Mark vertices
for cell, dummy in cells_injection:
    cell.get_vertex(0).mark('injection')

# Dofs at production wells 
injection_dofs = dh_Q1.get_region_dofs(entity_type='vertex', 
                                       entity_flag='injection')

# Initial control
n_Q1 = dh_Q1.n_dofs()
data = np.zeros((n_Q1,1))
data[production_dofs,:] = 1
u = Nodal(dofhandler=dh_Q1, data=data, dim=2)

#
# Random diffusion coefficient
# 
cov = Covariance(dh_Q1, name='gaussian', parameters={'l':0.1})
n_samples = 1000
k = GaussianField(n_Q1, K=cov)
k.update_support()
kfn = Nodal(dofhandler=dh_Q1, data=k.sample(n_samples=n_samples))


# =============================================================================
# Assembly
# =============================================================================
vb.comment('assembling system')
vb.tic()
K = Kernel(kfn, F=lambda f:np.exp(f))  # diffusivity

problems = [[Form(K, test=phi_x, trial=phi_x), 
             Form(K, test=phi_y, trial=phi_y),
             Form(0, test=phi)],
            [Form(test=phi, trial=phi)]]

assembler = Assembler(problems, mesh)
assembler.assemble()

# Mass matrix (for control)
M = assembler.af[1]['bilinear'].get_matrix()
alpha = 0.1 
vb.toc()

#
# Define State and Adjoint Systems
# 
b = assembler.af[0]['linear'].get_matrix()

state = LS(phi)
adjoint = LS(phi)

# Apply Dirichlet Constraints (state)
state.add_dirichlet_constraint('left',1)
state.add_dirichlet_constraint('right',0)
state.set_constraint_relation()

# Apply Dirichlet Constraints (adjoint)
adjoint.add_dirichlet_constraint('left',0)
adjoint.add_dirichlet_constraint('right',0)
adjoint.set_constraint_relation()

J = []
u_iter = []
alpha = 0.1
for i in range(n_samples):
    u_iter.append(u)
    
    # Get current sample of system matrix
    A = assembler.af[0]['bilinear'].get_matrix()[i]
    
    
    # =============================================================================
    # Solve state equations
    # =============================================================================
    #
    # State Equation
    # 
    state.set_matrix(A)
    state.set_rhs(b)
    
    # Update constraints  
    state.constrain_matrix()
    state.constrain_rhs()
    
    # Solve 
    state.solve_system()
    if i==0:
        y = state.get_solution(as_function=True)
    else:
        y.add_data(state.get_solution(as_function=False))
    
    #
    # Compute cost functional
    # 
    residual = np.zeros((n_Q1,1))
    y_data = y.data()[production_dofs,i][:,None]
    residual[production_dofs,:] = y_data - y_target
    Ji = 0.5*residual.T.dot(residual)+0.5*gamma*u.data().T.dot(M.dot(u.data()))
    J.append(Ji[0,0])

    #
    # Adjoint Equation
    # 
    adjoint.set_matrix(A)
    adjoint.set_rhs(residual)
    
    # update constraints
    adjoint.constrain_matrix()
    adjoint.constrain_rhs()
    
    # Solve adjoint equation 
    adjoint.solve_system()
    if i==0:
        p = adjoint.get_solution(as_function=True)
    else:
        p.add_data(adjoint.get_solution(as_function=False))
    
    #
    # Compute gradient
    # 
    g = p.data() + gamma*u.data()
    
    #
    # Update ak
    #
    alpha_k = alpha/n 
    
    #
    # Update u
    #
    u_data = u.data() - alpha_k*g   
    u.set_data(u_data)
        
    
vb.toc()
plt.hist(np.array(J))
plt.show()
