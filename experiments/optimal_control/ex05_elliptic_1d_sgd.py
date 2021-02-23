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
from scipy import linalg as la
from scipy.stats import norm
import scipy.sparse as sp

import matplotlib.pyplot as plt
import TasmanianSG
from tqdm import tqdm
"""
System 

    -div(exp(K)*grad(y)) = b + u,  x in D
                       y = g     ,  x in D_Dir
        exp(K)*grad(y)*n = 0     ,  x in D_Neu
    
    
Random field:
    
    K ~ GaussianField 

Cost Functional
    
    J(u) = E(|y(u)-y_d|**2) + alpha/2*|u|**2-

Minimize using a sparse grids to estimate E
"""
def sample_cost_gradient(state,adjoint,A,M,u,y_data,gamma):
    """
    Evaluate the cost functional at 
    """
    #
    # Solve state equation
    # 
    state.set_matrix(sp.csr_matrix(A, copy=True))
    b = M.dot(u)
    state.set_rhs(b)
    state.solve_system()
    y = state.get_solution(as_function=False)
    dy = y-y_data
    
    # Cost 
    f = 0.5*dy.T.dot(M.dot(dy)) + 0.5*gamma*u.T.dot(M.dot(u))
    
    #
    # Solve adjoint equation
    # 
    adjoint.set_matrix(sp.csr_matrix(A, copy=True))
    adjoint.set_rhs(M.dot(dy))
    adjoint.solve_system()
    p = adjoint.get_solution(as_function=False)
    
    # Gradient
    g = M.dot(p+u)
    
    return f, g, y, p



    
# =============================================================================
# Variational Form
# =============================================================================
comment = Verbose()
# 
# Mesh
# 
# Computational domain
x_min = 0
x_max = 2

mesh = Mesh1D(box=[x_min, x_max], resolution=(100,))

# Mark Dirichlet Vertices    
mesh.mark_region('left', lambda x: np.abs(x)<1e-9)
mesh.mark_region('right', lambda x: np.abs(x-2)<1e-9)

#
# Finite element spaces 
# 
Q1 = QuadFE(mesh.dim(), 'Q1')

# Dofhandler for state
dh = DofHandler(mesh, Q1)
dh.distribute_dofs()
m = dh.n_dofs()
dh.set_dof_vertices()
x = dh.get_dof_vertices()

# Basis functions
phi   = Basis(dh, 'v')
phi_x = Basis(dh, 'vx')

state = LS(phi)
state.add_dirichlet_constraint('left',1)
state.add_dirichlet_constraint('right',0)
state.set_constraint_relation()


adjoint = LS(phi)
adjoint.add_dirichlet_constraint('left',0)
adjoint.add_dirichlet_constraint('right',0)
adjoint.set_constraint_relation()


# =============================================================================
# System Parameters
# =============================================================================
# Target 
y_target = Nodal(f=lambda x: 3-4*(x[:,0]-1)**2, dim=1, dofhandler=dh)
y_data = y_target.data()
 
# Regularization parameter 
#gamma = 0.00001
gamma = 1e-5

# Inital guess 
u = np.zeros((m,1))

# =============================================================================
# Random Diffusion Parameter
# =============================================================================

# Initialize diffusion 
q = Nodal(data=np.empty((m,1)), dofhandler=dh)

# Log-normal field
sgm = 1


# Diffusion covariance function
cov = Covariance(dh, name='gaussian', parameters={'sgm':1,'l':0.001})

# Compute KL expansion
lmd, V = la.eigh(cov.get_matrix())
i_sorted = np.argsort(lmd)[::-1]
lmd = lmd[i_sorted]
V = V[:,i_sorted]

# Determine number of KL terms
tol_KL = 1e-5
r_max = 10
total_energy = np.sum(lmd**2)
for r in range(10):
    lr = lmd[:r]
    relative_error = 1-np.sum(lr**2)/total_energy
    if relative_error < tol_KL:
        break
print('Number of terms in the KL expansion:', r)
print('Relative error:', relative_error)   
Vr = V[:,:r]


# =============================================================================
# Monte Carlo Sample
# =============================================================================

n_batches = 100
n_samples_per_batch = 100
n_samples = n_batches*n_samples_per_batch
f_mc = []
g_mc = []
for n_batch in tqdm(range(n_batches)):
    #
    # Generate random sample
    # 
    z = np.random.normal(size=(r,n_samples_per_batch))
    q_smpl = Vr.dot(np.diag(np.sqrt(lr)).dot(z)) 
    q.set_data(q_smpl)
    expq = Kernel(q, F=lambda f:np.exp(f))
    
    plot = Plot()
    plot.line(q, i_sample=np.arange(100))
    #
    # Assemble system
    # 
    problems = [[Form(expq, test=phi_x, trial=phi_x)], 
                [Form(test=phi, trial=phi)]]
    assembler = Assembler(problems, mesh)
    assembler.assemble()
    
    M = assembler.af[0]['bilinear'].get_matrix()[0]
    for n in range(n_samples_per_batch):
        A = assembler.af[0]['bilinear'].get_matrix()[n]
        fn, gn, yn, pn = sample_cost_gradient(state,adjoint,A,M,u,y_data,gamma)   
        f_mc.append(fn)
        g_mc.append(gn)
        
f_mc = np.concatenate(f_mc, axis=1)
g_mc = np.concatenate(g_mc, axis=1)
np.save('f_mc',f_mc)
np.save('g_mc',g_mc)


# =============================================================================
# Sparse grid sample
# =============================================================================

tasmanian_library="/home/hans-werner/bin/TASMANIAN-6.0/libtasmaniansparsegrid.so"
f_grid = TasmanianSG.TasmanianSparseGrid(tasmanian_library=tasmanian_library)
g_grid = TasmanianSG.TasmanianSparseGrid(tasmanian_library=tasmanian_library)
n_levels = 5


for level in range(n_levels):
    f_grid.makeGlobalGrid(r,1,level,'level','gauss-hermite')
    g_grid.makeGlobalGrid(r,m,level,'level','gauss-hermite')
    
    z = np.sqrt(2)*f_grid.getPoints()
    q_smpl = Vr.dot(np.diag(np.sqrt(lr)).dot(z.T))
    
    
    # Determine number of batches
    n_samples = f_grid.getNumPoints()
    print('level', level)
    print('n_samples',n_samples)
    batch_size = 100
    n_batches = n_samples//batch_size
    batch_size_last = np.mod(n_samples,batch_size)
    
    fl_sg = []
    gl_sg = []
    pl_sg = []
    yl_sg = []
    i = 0
    for n_batch in range(n_batches+1):
        # get batch_size
        if n_batch == n_batches:
            n_batch_sample = batch_size_last
        else:
            n_batch_sample = batch_size
        
        q.set_data(q_smpl[:,i:i+n_batch_sample])
        expq = Kernel(q, F=lambda f:np.exp(f))
        
        #
        # Assemble system
        # 
        problems = [[Form(expq, test=phi_x, trial=phi_x)], 
                    [Form(test=phi, trial=phi)]]
        assembler = Assembler(problems, mesh)
        assembler.assemble()
        M = assembler.af[0]['bilinear'].get_matrix()
        if type(M) is list:
            M = M[0]
            
        for n in range(n_batch_sample):           
            A = assembler.af[0]['bilinear'].get_matrix()
            if type(A) is list:
                A = A[n]
            fn, gn, yn, pn = sample_cost_gradient(state,adjoint,A,M,u,y_data,gamma)   
            fl_sg.append(fn)
            gl_sg.append(gn)
            pl_sg.append(pn)
            yl_sg.append(yn)      
        i += n_batch_sample
    
    fl_sg = np.concatenate(fl_sg, axis=1)
    gl_sg = np.concatenate(gl_sg, axis=1)
    yl_sg = np.concatenate(yl_sg, axis=1)
    pl_sg = np.concatenate(pl_sg, axis=1)
        
    np.save('f%d_sg'%(level),fl_sg)
    np.save('y%d_sg'%(level),yl_sg)
    np.save('g%d_sg'%(level),gl_sg)        
plt.plot(x,yl_sg,'k',linewidth=0.1, alpha=0.1)
plt.show()
#print(M.toarray())

f_mc = np.load('f_mc.npy').ravel()
g_mc = np.load('g_mc.npy')

fig_f, ax_f = plt.subplots(2,3)
fig_g, ax_g = plt.subplots(2,3)
F = []
G = []
n_pts = []
for level in range(n_levels):
    fl_sg = np.load('f%d_sg.npy'%(level))
    gl_sg = np.load('g%d_sg.npy'%(level))
    
    f_grid.makeGlobalGrid(r,1,level,'level','gauss-hermite')
    g_grid.makeGlobalGrid(r,m,level,'level','gauss-hermite')
    
    f_grid.loadNeededPoints(fl_sg.T)
    g_grid.loadNeededPoints(gl_sg.T)
    
    w_f = f_grid.getQuadratureWeights()
    w_g = g_grid.getQuadratureWeights()
    
    normalizer = np.sqrt(np.pi)**r
    Fn = 0
    Gn = np.zeros(m,)
    for i in range(f_grid.getNumPoints()):
        Fn += fl_sg[:,i]*w_f[i]/normalizer
        Gn += gl_sg[:,i]*w_g[i]/normalizer
    
    FFn = f_grid.integrate()/normalizer
    GGn = g_grid.integrate()/normalizer

    F.append(Fn)
    G.append(Gn)
    
    # Monte Carlo estimate using the sparse grid
    zz = np.random.normal(size=(10000,r))/np.sqrt(2)
    fl_sg_sample = f_grid.evaluateBatch(zz)
    gl_sg_sample = g_grid.evaluateBatch(zz).T
    
    n_sample = f_grid.getNumPoints()
    n_pts.append(n_sample)
    i,j = np.unravel_index(level,(2,3))
    ax_f[i,j].hist(f_mc, bins=30, alpha=0.1, density=True, label='MC')
    ax_f[i,j].plot(fl_sg.T,np.zeros((n_sample,1)),'.',label=r'$z_i$')
    if level!=0:
        ax_f[i,j].hist(fl_sg_sample, bins=30, alpha=0.1, density=True, label='SG')
    ax_f[i,j].legend()
    
    ax_g[i,j].plot(x,np.mean(g_mc,axis=1),'k', alpha=0.1,label='MC')
    #print(np.any(np.isnan(gl_sg_sample)))
    #print(np.any(np.isinf(gl_sg_sample)))
    ax_g[i,j].plot(x.ravel(),Gn, 'r', alpha=0.1)
    

plt.show()
F = np.array(F).ravel()
fig, ax = plt.subplots(1,1)
ax.plot(np.cumsum(f_mc)/np.arange(1,len(f_mc)+1))
ax.plot(n_pts, F)
plt.show()

