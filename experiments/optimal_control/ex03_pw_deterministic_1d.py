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
from mesh import Mesh1D

from plot import Plot

from solver import LS

import numpy as np
import matplotlib.pyplot as plt
import gc
import scipy.sparse as sp
from tqdm import tqdm
from scipy.optimize import minimize
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

def cost_gradient(x,n,z_data,state,adjoint,A,M,gamma,dofs_inj,dofs_prod):
    """
    Return the cost function and Jacobian
    """
    # -------------------------------------------------------------------------
    # State equation
    # -------------------------------------------------------------------------
    
    u_data = np.zeros((n,1))
    u_data[dofs_inj,:] = x[:,None]  
    b = M.dot(u_data)
    state.set_matrix(sp.csr_matrix(A,copy=True))
    state.set_rhs(b)
    state.solve_system()
    y_data = state.get_solution(as_function=False)
    
    
    # -------------------------------------------------------------------------
    # Compute cost functional    
    # -------------------------------------------------------------------------
    residual = np.zeros((ny,1))
    y_data = y_data[dofs_prod,0][:,None]
    residual[dofs_prod,:] = y_data - z_data
    f = 0.5*residual.T.dot(residual) + \
         0.5*gamma*u_data.T.dot(M.dot(u_data))
    f = f[0,0]
    
    # -------------------------------------------------------------------------
    # Adjoint Equation
    # -------------------------------------------------------------------------
    adjoint.set_matrix(sp.csr_matrix(A, copy=True))
    adjoint.set_rhs(residual)
    
    # Solve adjoint equation 
    adjoint.solve_system()
    p = adjoint.get_solution(as_function=False)
    
    # -------------------------------------------------------------------------
    # Compute gradient
    # -------------------------------------------------------------------------
    g = M.dot(p + gamma*u_data)[dofs_inj]
    
    print(np.linalg.norm(g))
    
    return f,g.ravel()


# =============================================================================
# Mesh
# =============================================================================
# Computational domain
x_min = 0
x_max = 2

mesh = Mesh1D(box=[x_min, x_max], resolution=(512,))

# Mark Dirichlet Vertices
mesh.mark_region('left', lambda x: np.abs(x)<1e-9)
mesh.mark_region('right', lambda x: np.abs(x-2)<1e-9)

#
# Finite element spaces 
# 
Q1 = QuadFE(mesh.dim(), 'Q1')

# Dofhandler for state
dh_y = DofHandler(mesh, Q1)
dh_y.distribute_dofs()
ny = dh_y.n_dofs()

# Basis functions
phi   = Basis(dh_y, 'v')
phi_x = Basis(dh_y, 'vx')

# -----------------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------------- 
# Determine vertices corresponding to production wells
n_prod = 4
h = (x_max-x_min)/(n_prod+2)
x_prod = np.array([(i+1)*h for i in range(n_prod)])
v = dh_y.get_dof_vertices()
dofs_prod = []
for x in x_prod:
    dofs_prod.append(np.argmin(abs(v-x)))
#   
# Target pressure at production wells
#
z_fn = Explicit(f=lambda x: 3-4*(x[:,0]-1)**2, dim=1, mesh=mesh)
z_data = z_fn.eval(v[dofs_prod])

# -----------------------------------------------------------------------------
# Control
# -----------------------------------------------------------------------------
# Determine the vertices corresponding to the injection wells
n_inj = 6
h = (x_max-x_min)/(n_inj+2)
x_inj = np.array([(i+1)*h for i in range(n_inj)])
dofs_inj = []
for x in x_inj:
    dofs_inj.append(np.argmin(abs(v-x)))
 
u_data = np.zeros((ny,1))
u_data[dofs_inj] = 1
u = Nodal(dofhandler=dh_y, data=u_data, dim=1)


#
# Regularization parameter
# 
gamma = 0.00001
#gamma = 0.1
#
# Random diffusion coefficient
# 
cov = Covariance(dh_y, name='gaussian', parameters={'l':0.1})
k = GaussianField(ny, K=cov)
k.update_support()
kfn = Nodal(dofhandler=dh_y, data=k.sample(n_samples=1))

    
# =============================================================================
# Assembly
# =============================================================================
K = Kernel(kfn, F=lambda f:np.exp(f))  # diffusivity

problems = [[Form(K, test=phi_x, trial=phi_x)], 
            [Form(test=phi, trial=phi)]]

assembler = Assembler(problems, mesh)
assembler.assemble()

# Mass matrix (for control)
A = assembler.af[0]['bilinear'].get_matrix()
M = assembler.af[1]['bilinear'].get_matrix()


# =============================================================================
# Define State and Adjoint Systems
# =============================================================================
state = LS(phi) #, A=sp.csr_matrix(A, copy=True))
adjoint = LS(phi) #, A=sp.csr_matrix(A, copy=True))

# Apply Dirichlet Constraints (state)
state.add_dirichlet_constraint('left',1)
state.add_dirichlet_constraint('right',0)
state.set_constraint_relation()

# Apply Dirichlet Constraints (adjoint)
adjoint.add_dirichlet_constraint('left',0)
adjoint.add_dirichlet_constraint('right',0)
adjoint.set_constraint_relation()

# =============================================================================
# Optimization
# =============================================================================
res = minimize(cost_gradient, u_data[dofs_inj], 
               args=(ny,z_data,state, adjoint,A,M,gamma,dofs_inj,dofs_prod),
               jac=True)

print(res.x)
# =============================================================================
# Plot results
# =============================================================================
u_data = np.zeros((ny,1))
u_data[dofs_inj,:] = res.x[:,None]  
b = M.dot(u_data)
state.set_matrix(sp.csr_matrix(A,copy=True))
state.set_rhs(b)
state.solve_system()

y_data = state.get_solution(as_function=True)

fig, ax = plt.subplots(1,1)
plot = Plot(quickview=False)
ax.plot(v[dofs_prod],z_data,'ro')
ax = plot.line(y_data, axis=ax)
ax.plot(v[dofs_inj],np.zeros((len(dofs_inj),1)), 'C0o')
plt.show()    

