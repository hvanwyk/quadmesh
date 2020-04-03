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

import Tasmanian

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gc
import scipy.sparse as sp


def qfn(x,z):
    """
    Evaluate the random field 
    """    
    q = 1\
        +   0.1*z[0]*np.cos(np.pi*x[:,0])\
        +  0.05*z[1]*np.cos(2*np.pi*x[:,0])\
        +  0.01*z[2]*np.cos(3*np.pi*x[:,0])\
        + 0.005*z[3]*np.cos(4*np.pi*x[:,0])
        
    return q


def sample_qoi(q, dofhandler):
    """
    Sample total energy of output for a given sample of q's 
    """
    # 
    # Set up weak form 
    # 
    
    # Basis 
    phi   = Basis(dofhandler, 'v')
    phi_x = Basis(dofhandler, 'vx')
       
    # Elliptic problem
    problems = [[Form(q, test=phi_x, trial=phi_x), Form(1, test=phi)],
                [Form(1, test=phi, trial=phi)]]
    
    # Assemble
    assembler = Assembler(problems, mesh)
    assembler.assemble()
    
    # System matrices
    A = assembler.af[0]['bilinear'].get_matrix()
    b = assembler.af[0]['linear'].get_matrix()
    M = assembler.af[1]['bilinear'].get_matrix()
    
    # Define linear system
    system = LS(phi)
    system.add_dirichlet_constraint('left',1)
    system.add_dirichlet_constraint('right',0)
    
    n_samples = q.n_samples()
    y_smpl = []
    QoI_smpl = []
    for i in range(n_samples):
        # Sample system 
        if n_samples > 1:
            Ai = A[i]
        else:
            Ai = A
        system.set_matrix(Ai)
        system.set_rhs(b.copy())
        
        # Solve system
        system.solve_system()
        
        # Record solution and qoi
        y = system.get_solution(as_function=False)
        y_smpl.append(y)
        QoI_smpl.append(y.T.dot(M.dot(y)))
    
    # Convert to numpy array    
    y_smpl = np.concatenate(y_smpl,axis=1)
    QoI = np.concatenate(QoI_smpl, axis=1).ravel()
    
    return y_smpl, QoI
    
"""
Goal: We split the uncertain diffusion coefficient into a low dimensional and 
    a high dimensional component. We then use sparse grids to approximate the 
    low dimensional component and Monte Carlo for the high dimensional part. 
    If we use KL expansions, the random vectors are independent 
    
    
System 

    -div(exp(K)*grad(y)) = 0,  x in D
                    y(0) = 1,  
                    y(1) = 0
        exp(K)*grad(y)*n = 0,  x in D_Neu
    
    
Random field:
    
    K ~ GaussianField 

Cost Functional
    
    QoI = E(||y||**2)
"""
plot = Plot(quickview=False)
# -----------------------------------------------------------------------------
# Spatial Discretization
# -----------------------------------------------------------------------------
#
# Mesh
#
mesh = Mesh1D(resolution=(20,))

# Mark Dirichlet Vertices
mesh.mark_region('left', lambda x: np.abs(x)<1e-9)
mesh.mark_region('right', lambda x: np.abs(x-1)<1e-9)

#
# Element
# 
Q1 = QuadFE(mesh.dim(), 'Q1')

# Dofhandler
dofhandler = DofHandler(mesh, Q1)
dofhandler.distribute_dofs()
x = dofhandler.get_dof_vertices()
n = dofhandler.n_dofs()

# -----------------------------------------------------------------------------
# Sample full dimensional parameter space
# -----------------------------------------------------------------------------
#
# Set up sparse grid on coarse parameter space 
#
tasmanian_library="/home/hans-werner/bin/TASMANIAN-6.0/libtasmaniansparsegrid.so"
grid = TasmanianSG.TasmanianSparseGrid(tasmanian_library=tasmanian_library)
dimensions = 4
outputs = 1
max_depth = 8
type = 'level'
rule = 'gauss-legendre'

"""
fig = plt.figure(figsize=(10,8), constrained_layout=True)
gs = fig.add_gridspec(3,4)

EQoI = []
ns = []
plot_cols = [0,2,4,6]
plot_count = 0
for depth in range(max_depth):
    #
    # Get nodes and weights
    # 
    grid.makeGlobalGrid(dimensions, outputs, depth, type, rule)
    z = grid.getPoints()
    w = 0.5**dimensions*grid.getQuadratureWeights()
    n_samples = grid.getNumPoints()
    
    # Sample random parameter
    qSG = np.empty((n,n_samples))
    for i in range(n_samples):
        qSG[:,i] = qfn(x,z[i,:])  
    q = Nodal(data=qSG, dofhandler=dofhandler)
    
    # Sample state and qoi
    y, QoI = sample_qoi(q,dofhandler)
    
    ns.append(n_samples)
    EQoI.append(np.sum(w*QoI))

    # Plot 
    if depth in plot_cols:
        q_ax = fig.add_subplot(gs[0,plot_count])
        q_ax = plot.line(q,axis=q_ax,i_sample=np.arange(n_samples),
                         plot_kwargs={'color':'k','linewidth':0.5})
        q_ax.set_title(r'$q(x,z^{(%d)})$'%(depth))
        
        y_ax = fig.add_subplot(gs[1,plot_count])
        y_ax.plot(x,y,'k',linewidth=0.5)
        y_ax.set_title(r'$y(x,z^{(%d)})$'%(depth))
        plot_count += 1
    
EQoI = np.array(EQoI)
ns   = np.array(ns)
err_ax = fig.add_subplot(gs[2,:])
err_ax.loglog(ns[:-1], np.abs(EQoI[:-1]-EQoI[-1]))
err_ax.loglog(ns[plot_cols], np.abs(EQoI[plot_cols]-EQoI[-1]),'k.',markersize=12)
plt.savefig('ex01_full_parspace.pdf')
plt.show()
"""
"""
# -----------------------------------------------------------------------------
# Reference Solution
# -----------------------------------------------------------------------------

grid = TasmanianSG.TasmanianSparseGrid(tasmanian_library=tasmanian_library)
dimensions = 4
outputs = 1
depth = 8
type = 'level'
rule = 'gauss-legendre'
grid.makeGlobalGrid(dimensions, outputs, depth, type, rule)
z = grid.getPoints()
w = 0.5**dimensions*grid.getQuadratureWeights()
n_samples = grid.getNumPoints()
print(n_samples)

# Sample random parameter
qSG = np.empty((n,n_samples))
for i in range(n_samples):
    qSG[:,i] = qfn(x,z[i,:])  
q = Nodal(data=qSG, dofhandler=dofhandler)

# Sample state and qoi
y, QoI = sample_qoi(q,dofhandler)

EQoI_ref = np.sum(w*QoI)
np.save('EQoI_ref',EQoI_ref)

# -----------------------------------------------------------------------------
# Sample both low dimensional and remaining space by sparse grid. 
# -----------------------------------------------------------------------------
#
# Initialize sparse grid of low dimensional and complementary parameter space  
#  
dimensions = 2
outputs = 1
max_depth = 6
type = 'level'
rule = 'gauss-legendre'
lgrid = TasmanianSG.TasmanianSparseGrid(tasmanian_library=tasmanian_library)
hgrid = TasmanianSG.TasmanianSparseGrid(tasmanian_library=tasmanian_library)

EQoI = np.zeros((max_depth,max_depth))
CEQoI = []
nls = []
n_tot = 0
for ldepth in range(max_depth):
    print('ldepth:%d'%(ldepth))
    #
    # Grid in the low-dimensional parameter space 
    #
    lgrid.makeGlobalGrid(dimensions, outputs, ldepth, type, rule)
    zl = lgrid.getPoints()
    wl = 0.5**dimensions*lgrid.getQuadratureWeights()
    nl = lgrid.getNumPoints()
    nls.append(nl)
    for hdepth in range(max_depth):
        
        #
        # Grid on high dimensional parameter space s
        #
        hgrid.makeGlobalGrid(dimensions, outputs, hdepth, type, rule)
        zh = hgrid.getPoints()
        wh = 0.5**dimensions*hgrid.getQuadratureWeights() 
        nh = hgrid.getNumPoints()
        
        #
        # Combine samples
        #
        n_samples = nh*nl
        n_tot += n_samples
        print('  hdepth:%d (%d)'%(hdepth,n_samples))
        z = np.empty((n_samples,4))
        I,J = np.mgrid[0:nl,0:nh]
        z[:,:2] = zl[I.ravel()]
        z[:,2:] = zh[J.ravel()]
        
        #
        # Sample random parameter
        #
        qSG = np.empty((n,n_samples))
        for i in range(n_samples):
            qSG[:,i] = qfn(x,z[i,:])  
        q = Nodal(data=qSG, dofhandler=dofhandler)
        
        #
        # Sample output and QoI
        # 
        y, QoI = sample_qoi(q, dofhandler)
        
        # Each row corresponds to a conditional sample
        CQoI = QoI.reshape(-1,nh)
        
        cEQoI = np.array([np.sum(wh*CQoI[i,:]) for i in range(nl)])
        
        CEQoI.append(cEQoI)
        EQoI[ldepth,hdepth] = np.sum(wl*cEQoI)
        
        np.save('EQoI', EQoI)
np.save('nls',nls)
"""

nls = np.load('nls.npy')
EQoI = np.load('EQoI.npy')
EQoI_ref = np.load('EQoI_ref.npy')

ll = tuple([r'$l_0=%d$'%(level+1) for level in range(6)])

nls = np.array(nls)    
p = plt.loglog(nls,np.abs(EQoI-EQoI_ref),'.-')
plt.xlabel('Sample size (fine)')
plt.ylabel('Quadrature error')
plt.legend(iter(p),ll)
plt.savefig('ex01_conditional_quadrature_sg.pdf')

plt.show()
        