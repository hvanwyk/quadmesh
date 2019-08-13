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

import TasmanianSG

import numpy as np
import matplotlib.pyplot as plt
import gc
import scipy.sparse as sp
import multiprocessing
from tqdm import tqdm

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
    
    QoI = E(|y(|**2)
"""

# Mesh
mesh = Mesh1D(resolution=(20,))

# Mark Dirichlet Vertices
mesh.mark_region('left', lambda x: np.abs(x)<1e-9)
mesh.mark_region('right', lambda x: np.abs(x-1)<1e-9)

# Element 
Q1 = QuadFE(mesh.dim(), 'Q1')

# Dofhandler
dofhandler = DofHandler(mesh, Q1)
dofhandler.distribute_dofs()
n = dofhandler.n_dofs()

# Basis functions
phi   = Basis(dofhandler, 'v')
phi_x = Basis(dofhandler, 'vx')

#
# Simple random parameter
# 
x = dofhandler.get_dof_vertices()
n_batches = 1
batch_size = 1000
QoI_smpl = []
for n_batch in range(n_batches):
    print('batch: ', n_batch)
    # Generate sample of Qs
    Z = np.random.rand(batch_size,4) 
    qMC = np.empty((n,batch_size))
    for i in range(batch_size):
        qMC[:,i] = 1 + 0.1*Z[i,0]*np.cos(np.pi*x[:,0]) + 0.05*Z[i,1]*np.cos(2*np.pi*x[:,0]) + \
                       0.01*Z[i,2]*np.cos(3*np.pi*x[:,0]) + 0.005*Z[i,3]*np.cos(4*np.pi*x[:,0])
     
    q = Nodal(data=qMC, dofhandler=dofhandler)
              
    plot = Plot()
    plot.line(q,i_sample=np.arange(batch_size))
    
    # Define elliptic problem
    problems = [[Form(q, test=phi_x, trial=phi_x), Form(1, test=phi)],
                [Form(1, test=phi, trial=phi)]]
    assembler = Assembler(problems, mesh)
    assembler.assemble()
    
    A = assembler.af[0]['bilinear'].get_matrix()
    b = assembler.af[0]['linear'].get_matrix()
    M = assembler.af[1]['bilinear'].get_matrix()
    
    system = LS(phi)
    system.add_dirichlet_constraint('left',1)
    system.add_dirichlet_constraint('right',0)
    
    for i in range(batch_size):
        Ai = A[i]
        system.set_matrix(Ai)
        system.set_rhs(b.copy())
        system.solve_system()
        y = system.get_solution(as_function=False)
        QoIi = y.T.dot(M.dot(y))[0]
        QoI_smpl.append(QoIi)


QoI_smpl = np.array(QoI_smpl)
EQoI_ref = np.mean(QoI_smpl)
n_samples = batch_size*n_batches
EQoI_mc = np.cumsum(QoI_smpl)/np.arange(1,n_samples+1) 
plt.loglog(np.abs(EQoI_mc-EQoI_ref),label='MC')


# Sparse grid approximation 
tasmanian_library="/home/hans-werner/bin/TASMANIAN-6.0/libtasmaniansparsegrid.so"
grid = TasmanianSG.TasmanianSparseGrid(tasmanian_library=tasmanian_library)
dimensions = 4
outputs = 1
depth = 4
type = 'tensor'
rule = 'gauss-legendre'

EQoI = []
ns = []
print('Sparse Grids')
for depth in range(7):
    grid.makeGlobalGrid(dimensions, outputs, depth, type, rule)
    Z = 0.5*grid.getPoints()+0.5
    w = 0.5**dimensions*grid.getQuadratureWeights()
    n_samples = grid.getNumPoints()
    print('sparse grid depth: ', depth)
    qSG = np.empty((n,n_samples))
    for i in range(n_samples):
        qSG[:,i] = 1 + \
                   0.1*Z[i,0]*np.cos(np.pi*x[:,0]) + \
                   0.05*Z[i,1]*np.cos(2*np.pi*x[:,0]) + \
                   0.01*Z[i,2]*np.cos(3*np.pi*x[:,0]) + \
                   0.005*Z[i,3]*np.cos(4*np.pi*x[:,0])
     
    q = Nodal(data=qSG, dofhandler=dofhandler)
              
    
    
    # Define elliptic problem
    problems = [[Form(q, test=phi_x, trial=phi_x), Form(1, test=phi)],
                [Form(1, test=phi, trial=phi)]]
    assembler = Assembler(problems, mesh)
    assembler.assemble()
    
    A = assembler.af[0]['bilinear'].get_matrix()
    b = assembler.af[0]['linear'].get_matrix()
    M = assembler.af[1]['bilinear'].get_matrix()
    
    system = LS(phi)
    system.add_dirichlet_constraint('left',1)
    system.add_dirichlet_constraint('right',0)
    y_smpl = []
    QoI_smpl = []
    for i in range(n_samples):
        if n_samples > 1:
            Ai = A[i]
        else:
            Ai = A
        system.set_matrix(Ai)
        system.set_rhs(b.copy())
        system.solve_system()
        y = system.get_solution(as_function=False)
        y_smpl.append(y)
        QoI_smpl.append(y.T.dot(M.dot(y)))
    y_smpl = np.concatenate(y_smpl,axis=1)
    QoI = np.concatenate(QoI_smpl, axis=1).ravel()
        
    EQoI.append(np.sum(w*QoI))
    ns.append(n_samples)
EQoI_sg = np.array(EQoI)
ns = np.array(ns)
plt.loglog(ns,np.abs(EQoI_sg-EQoI_ref),'.-', label='SG')
plt.legend()
plt.savefig('sg_error.eps')