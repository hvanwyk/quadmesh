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

import TasmanianSG

def cost_gradient(state,adjoint,A,M,g,u,y_data,alpha):
    """
    Compute the cost function and gradients
    
    Inputs:
    
        state: LinearSystem, representing the state equations
        
        adjoint: LinearSystem representing the adjoint equations
        
        A: double, sparse stiffness matrix
        
        M: double, sparse mass matrix
        
        g: double, vector, forcing term
        
        u: double, vector representing the control
        
        y_data: double, target state
        
        alpha: double >0, regularization parameter
        
        
    Outputs:
    
        y: double, state vector
        
        p: double, adjoint vector
        
        J: double, cost function
        
        dJ: double, gradient vector
    """
    #
    # Solve state equation
    # 
    state.set_matrix(sp.csr_matrix(A, copy=True))
    b = M.dot(g+u)
    state.set_rhs(b)
    state.solve_system()
    y = state.get_solution(as_function=False)
    dy = y-y_data
    
    # Cost 
    J = 0.5*dy.T.dot(M.dot(dy)) + 0.5*alpha*u.T.dot(M.dot(u))
    
    #
    # Solve adjoint equation
    # 
    adjoint.set_matrix(sp.csr_matrix(A, copy=True))
    adjoint.set_rhs(M.dot(dy))
    adjoint.solve_system()
    p = adjoint.get_solution(as_function=False)
    
    # Gradient
    dJ = p+alpha*u
    
    return y,p,J,dJ
    

def reference_solution():
    """
    Use sparse grid method to compute a benchmark solution
    """
    
    mesh = QuadMesh(resolution=(10,10))
    mesh.mark_region('boundary', lambda x,y:True, 
                     entity_type='half_edge', on_boundary=True)
    element = QuadFE(mesh.dim(), 'Q1')
    dofhandler = DofHandler(mesh, element)
    dofhandler.distribute_dofs()
    n = dofhandler.n_dofs()
    phi = Basis(dofhandler, 'u')
    phi_x = Basis(dofhandler, 'ux')
    phi_y = Basis(dofhandler, 'uy')
    yd_fn = Explicit(lambda x: np.sin(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1]),dim=2)
    g = np.ones((n,1))
    
    #
    # Random diffusion coefficient
    # 
    
    # Sparse grid
    tasmanian_library="/home/hans-werner/bin/TASMANIAN-6.0/libtasmaniansparsegrid.so"
    grid = TasmanianSG.TasmanianSparseGrid(tasmanian_library=tasmanian_library)
    dimensions = 4
    outputs = 1
    depth = 4
    type = 'tensor'
    rule = 'gauss-legendre'
    grid.makeGlobalGrid(dimensions, outputs, depth, type, rule)
    Y = grid.getPoints()
    w = grid.getQuadratureWeights()
    n_samples = grid.getNumPoints()
    
    
    x = dofhandler.get_dof_vertices()
    a_nodal = 1 + 0.1*(np.outer(np.cos(np.pi*x[:,1]),Y[:,0])+\
                       np.outer(np.cos(np.pi*x[:,0]),Y[:,1])+\
                       np.outer(np.sin(2*np.pi*x[:,1]),Y[:,2])+\
                       np.outer(np.sin(2*np.pi*x[:,0]),Y[:,3]))
    a = Nodal(data=a_nodal, dofhandler=dofhandler)
    yd_vec = yd_fn.eval(x)
    
    problems = [[Form(a, test=phi_x, trial=phi_x), 
                Form(a, test=phi_y, trial=phi_y)],
                [Form(1, test=phi, trial=phi)]]
    
    assembler = Assembler(problems, mesh)
    assembler.assemble()
    
    A = assembler.af[0]['bilinear'].get_matrix()
    M = assembler.af[1]['bilinear'].get_matrix()
    
    state = LS(phi)
    state.add_dirichlet_constraint('boundary')
    
    adjoint = LS(phi)
    adjoint.add_dirichlet_constraint('boundary')
    
    tau = 10
    k_max = 20
    alpha = 0.1
    u = np.zeros((n,1))
    norm_dJ_iter = []
    J_iter = []
    u_iter = []
    for k in range(k_max):
        print('iteration', k)
        # 
        # Compute average cost and gradient
        # 
        dJ = np.zeros((n,1))
        J = 0
        print('sampling')
        for n in range(n_samples):
            print(n)
            yn, pn, Jn, dJn = cost_gradient(state,adjoint,A[n],M,
                                            g,u,yd_vec,alpha)
            
            J += w[n]*Jn
            dJ += w[n]*dJn
        print('')
        norm_dJ = np.sqrt(dJ.T.dot(M.dot(dJ)))
        
        #
        # Store current iterates
        # 
        norm_dJ_iter.append(norm_dJ)
        J_iter.append(J)
        u_iter.append(u)
        
        #
        # Check for convergence
        # 
        if norm_dJ<1e-8:
            break
        #
        # Update iterate
        # 
        u -= tau*dJ
        
        
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


reference_solution()
"""
#
# Define multiresolution mesh
# 
n_resolutions = 9
mesh = Mesh1D(box=[x_min, x_max])
mesh.record(0)
for i in range(n_resolutions):
    mesh.cells.refine()
    mesh.cells.record(i+1)
"""
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
#gamma = 0.00001
gamma = 0.1

#
# Random diffusion coefficient
# 
n_samples = 200
cov = Covariance(dh_y, name='gaussian', parameters={'l':0.1})
k = GaussianField(ny, K=cov)
k.update_support()
kfn = Nodal(dofhandler=dh_y, data=k.sample(n_samples=n_samples))

    
# =============================================================================
# Assembly
# =============================================================================
K = Kernel(kfn, F=lambda f:np.exp(f))  # diffusivity

problems = [[Form(K, test=phi_x, trial=phi_x)], 
            [Form(test=phi, trial=phi)]]

assembler = Assembler(problems, mesh)
assembler.assemble()

# Mass matrix (for control)
M = assembler.af[1]['bilinear'].get_matrix()


# =============================================================================
# Define State and Adjoint Systems
# =============================================================================
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

# Initial guess
x = np.array([-520.02634251, -378.1222316,  1182.85592199,  273.54809863,  198.63821595,
  171.72602221])[:,None]
x = np.ones(x.shape)
alpha_0 = 1
f_iter = []
print('|'+'-'*80+'|')
for k in range(n_samples):
    """
    Iterate over number of samples
    """
    # Get current sample of system matrix
    A = assembler.af[0]['bilinear'].get_matrix()[k]
    
    # -------------------------------------------------------------------------
    # State equation
    # -------------------------------------------------------------------------
    u_data = np.zeros((ny,1))
    u_data[dofs_inj,:] = x  
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
    f = 0.5*residual.T.dot(M.dot(residual)) + \
         0.5*gamma*u_data.T.dot(M.dot(u_data))
    f = f[0,0]
    f_iter.append(f)
    
    # -------------------------------------------------------------------------
    # Adjoint Equation
    # -------------------------------------------------------------------------
    adjoint.set_matrix(sp.csr_matrix(A, copy=True))
    adjoint.set_rhs(M.dot(residual))
    
    # Solve adjoint equation 
    adjoint.solve_system()
    p = adjoint.get_solution(as_function=False)
    
    # -------------------------------------------------------------------------
    # Compute gradient
    # -------------------------------------------------------------------------
    g = (p + gamma*u_data)[dofs_inj]
    
    # -------------------------------------------------------------------------
    # Update x
    # -------------------------------------------------------------------------
    alpha_k = alpha_0/(1+k)
    x = x - alpha_k*g

    print('.',end='')
    
# =============================================================================
# Plot results
# =============================================================================
u_data = np.zeros((ny,1))
u_data[dofs_inj,:] = x  
b = M.dot(u_data)
state.set_matrix(sp.csr_matrix(A,copy=True))
state.set_rhs(b)
state.solve_system()

y_data = state.get_solution(as_function=True)

fig, ax = plt.subplots(1,2)
plot = Plot(quickview=False)
ax[0].plot(v[dofs_prod],z_data,'ro')
ax[0] = plot.line(y_data, axis=ax[0])
ax[0].plot(v[dofs_inj],np.zeros((len(dofs_inj),1)), 'C0o')

ax[1].plot(np.array(f_iter))
plt.show()  


vb = Verbose()

# =============================================================================
# Mesh
# =============================================================================
# Computational domain
x_min = 0
x_max = 2


# 
# Plot
#
""" 
plot = Plot(quickview=False)
fig, ax = plt.subplots(n_resolutions,1)
for i in range(n_resolutions):
    ax[i] = plot.mesh(mesh, axis=ax[i], subforest_flag=i)
"""


# Mark Dirichlet Vertices
mesh.mark_region('left', lambda x: np.abs(x)<1e-9)
mesh.mark_region('right', lambda x: np.abs(x-2)<1e-9)

#
# Finite element spaces 
# 
Q0 = QuadFE(mesh.dim(), 'DQ0')
Q1 = QuadFE(mesh.dim(), 'Q1')

# Dofhandler for state
dh_y = DofHandler(mesh, Q1)
dh_y.distribute_dofs()
ny = dh_y.n_dofs()

# Basis functions
phi   = Basis(dh_y, 'v')
phi_x = Basis(dh_y, 'vx')


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
z_d = z_fn.eval(v[dofs_prod])

# Determine the vertices corresponding to the injection wells
n_inj = 6
h = (x_max-x_min)/(n_inj+2)
x_inj = np.array([(i+1)*h for i in range(n_inj)])
dofs_inj = []
for x in x_inj:
    dofs_inj.append(np.argmin(abs(v-x)))

#
# Control function
# 
u_data = np.zeros((ny,1))
u_data[dofs_inj] = 1
u = Nodal(dofhandler=dh_y, data=u_data, dim=1)

"""
#
# Plot injection and production wells
# 
fig, ax = plt.subplots(1,1, figsize=(4,3))
ax = plot.line(u, axis=ax)

z_data = np.zeros((ny,1))
z_data[dofs_prod] = z_d
z_nodal = Nodal(dofhandler=dh_y, data=z_data, dim=1)
ax = plot.line(z_nodal, axis=ax)
"""
#
# Regularization parameter
# 
gamma = 1e-5

#
# Random diffusion coefficient
# 
cost = []
"""
fig_q, ax_q = plt.subplots(3,3)
fig_y, ax_y = plt.subplots(3,3)
"""
    

for i in range(n_resolutions-1, n_resolutions):
    
    fig, ax = plt.subplots(1,1)    

    
    dh_q = DofHandler(mesh, Q1)
    dh_q.distribute_dofs(subforest_flag=i)
    cov = Covariance(dh_q, name='gaussian', parameters={'l':0.1}, subforest_flag=i)
    n_samples = 50
    nq = dh_q.n_dofs(subforest_flag=i)
    k = GaussianField(nq, K=cov)
    k.update_support()
    
    kfn = Nodal(dofhandler=dh_q, data=k.sample(n_samples=n_samples), subforest_flag=i)
    

    """
    #
    # Plot log diffusion
    #
    ll,mm = np.unravel_index(i,(3,3))
    ax_q[ll,mm] = plot.line(kfn, axis=ax_q[ll,mm])
    """
    
    # =============================================================================
    # Assembly
    # =============================================================================
    vb.comment('assembling system')
    vb.tic()
    K = Kernel(kfn, F=lambda f:np.exp(f))  # diffusivity
    
    problems = [[Form(K, test=phi_x, trial=phi_x), 
                 Form(u, test=phi)],
                [Form(test=phi, trial=phi)]]
    
    assembler = Assembler(problems, mesh)
    assembler.assemble()
    
    # Mass matrix (for control)
    M = assembler.af[1]['bilinear'].get_matrix()
    alpha = 0.01
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
    vb.comment('Sampling')
    vb.tic()
    for n in tqdm(range(n_samples)):
        
        # Get current sample of system matrix
        A = assembler.af[0]['bilinear'].get_matrix()[n]
        
        # =============================================================================
        # Solve state equations
        # =============================================================================
        #
        # State Equation
        # 
        state.set_matrix(A)
        
        b = M.dot(u.data())
        state.set_rhs(b)
                
        # Solve 
        state.solve_system()
        if n==0:
            y = state.get_solution(as_function=True)
        else:
            y.add_data(state.get_solution(as_function=False))
        
        """
        #
        # Plot state
        # 
        if n<3:
            ll,mm = np.unravel_index(i,(3,3))
            ax_y[ll,mm] = plot.line(y, axis=ax_y[ll,mm], i_sample=n)
        """
        
        #
        # Compute cost functional    
        # 
        residual = np.zeros((ny,1))
        y_data = y.data()[dofs_prod,n][:,None]
        residual[dofs_prod,:] = y_data - z_d
        Jn = 0.5*residual.T.dot(residual) + \
             0.5*gamma*u.data().T.dot(M.dot(u.data()))
        J.append(Jn[0,0])
        
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
        if n==0:
            p = adjoint.get_solution(as_function=True)
        else:
            p.add_data(adjoint.get_solution(as_function=False))
            
         
        #
        # Compute gradient
        # 
        g = p.data()[:,n] + gamma*u.data()
        
        #
        # Update ak
        #
        alpha_k = alpha/(n+1) 
        
        #
        # Update u
        #
        u_data = u.data() - alpha_k*g   
        u.set_data(u_data)
            
        
    #
    # Store the cost functional in a list
    # 
    cost.append(np.array(J))
    
    #Ji = cost[i]
    one_to_n = np.arange(1,n_samples+1)
    Ei = np.cumsum(J)/one_to_n
    ax.plot(Ei,alpha=0.7, label='n=%d'%(2**i))
    plt.legend()
    plt.show()
#ax.hist(np.array(J), cumulative=True, density=True)