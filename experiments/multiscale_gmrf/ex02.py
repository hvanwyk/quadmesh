from assembler import Assembler
from assembler import Form
from fem import DofHandler
from fem import QuadFE
from fem import Basis
from function import Nodal
from gmrf import Covariance
from gmrf import GaussianField
from mesh import Mesh1D
from plot import Plot
from solver import LS
import TasmanianSG

# Built-in modules
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

 

"""
Consider the elliptic equation

-d/dx(e^q dy/dx) = f
y(0) = 1
y(1) = 0

on (0,1), where q is a normal gaussian field.

Split the diffusion coefficient into a low- and a high dimensional component

Use sparse grids to integrate the low dimensional approximation and Monte Carlo
for the high dimensional region. 
"""
# =============================================================================
# Finite Element Discretization
# ============================================================================= 

# Computational Mesh
mesh = Mesh1D(resolution=(100,))
mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)

# Element
element = QuadFE(mesh.dim(), 'Q1')
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()
m = dofhandler.n_dofs()
x = dofhandler.get_dof_vertices()

# =============================================================================
# Random field
# =============================================================================
n_samples = 5

cov = Covariance(dofhandler, name='gaussian', parameters={'l':0.07})
#cov = Covariance(dofhandler, name='exponential', parameters={'l':0.1})
cov.compute_eig_decomp()
lmd,V = cov.get_eig_decomp()

plt.semilogy(lmd,'.')
plt.show()

# Plot low dimensional field
d0 = 5
d  = len(lmd)
Lmd0 = np.diag(np.sqrt(lmd[:d0]))
V0 = V[:,:d0]
Z0  = np.random.randn(d0,n_samples)
log_q0 = V0.dot(Lmd0.dot(Z0))
plt.plot(x,log_q0)
plt.show()

# Plot high dimensional field conditional on low
Dc = np.diag(np.sqrt(lmd[d0:]))
Vc = V[:,d0:]
for n in range(n_samples):
    Zc = np.random.randn(d-d0,100)
    log_qc = Vc.dot(Dc.dot(Zc))
    plt.plot(x,(log_q0[:,n].T+log_qc.T).T, 'k', linewidth=0.1, alpha=0.5)
plt.show()

# =============================================================================
# Sparse Grid Loop
# =============================================================================
tasmanian_library="/home/hans-werner/bin/TASMANIAN-6.0/libtasmaniansparsegrid.so"
grid = TasmanianSG.TasmanianSparseGrid(tasmanian_library=tasmanian_library)
dimensions = d0
outputs = m
depth = 8
type = 'level'
rule = 'gauss-hermite'
grid.makeGlobalGrid(dimensions, outputs, depth, type, rule)

# Get Sample Points
zzSG = grid.getPoints()
zSG = np.sqrt(2)*zzSG                # transform to N(0,1)

wSG = grid.getQuadratureWeights()
n0 = grid.getNumPoints()

# Sample low resolution parameter
log_qSG = V0.dot(Lmd0.dot(zSG.T))
log_q0 = Nodal(data=log_qSG, dofhandler=dofhandler)

# Sample state
qfn = Nodal(dofhandler=dofhandler, data=np.exp(log_qSG))


# =============================================================================
# Compute Sparse Grid Expectation
# ============================================================================= 
phi = Basis(dofhandler, 'u')
phi_x = Basis(dofhandler, 'ux')

problems = [[Form(kernel=qfn, trial=phi_x, test=phi_x), Form(1, test=phi)],
            [Form(1, test=phi, trial=phi)]]

assembler = Assembler(problems, mesh)
assembler.assemble()

A = assembler.af[0]['bilinear'].get_matrix()
b = assembler.af[0]['linear'].get_matrix()

linsys = LS(phi)
linsys.add_dirichlet_constraint('left',1)
linsys.add_dirichlet_constraint('right',0)

"""
y_data = np.empty((m,n0))
k = 0
for n in range(n0):
    if (n/n0 >= k/10):
        print('%d0 percent complete'%(k))
        k+=1
        
    linsys.set_matrix(A[n].copy())
    linsys.set_rhs(b.copy())
    linsys.solve_system()
    y_data[:,[n]] = linsys.get_solution(as_function=False)

np.save('y_SG',y_data)
"""

y_SG = np.load('y_SG.npy')
c_norm = np.sqrt(np.pi)**d0     # normalization constant

y_ave_SG = np.zeros(m)
for n in range(n0):
    y_ave_SG += wSG[n]*y_SG[:,n]/c_norm
plt.plot(x,y_ave_SG)
plt.show()


n1 = 20
# Plot high dimensional field conditional on low
Dc = np.diag(np.sqrt(lmd[d0:]))
Vc = V[:,d0:]

n = np.random.randint(n0)
yc_ave_MC = np.empty((m,n0))
k = 0
for i in range(n0):
    if (n/n0 >= k/10):
        print('%d0 percent complete'%(k))
        k+=1
        
    Zc = np.random.randn(d-d0,m)
    log_qc = Vc.dot(Dc.dot(Zc))
    qfn = Nodal(dofhandler=dofhandler, data=np.exp(log_qc))
    assembler.assemble()
    
    # Compute conditional expectation
    yc_data = np.empty((m,n1))
    for j in range(n1):
        linsys.set_matrix(A[j])
        linsys.set_rhs(b.copy())
        linsys.solve_system()
        yc_data[:,[j]] = linsys.get_solution(as_function=False)
    
    #if i==5:
    #    plt.plot(x,yc_data,'k', linewidth=0.1, alpha=0.5)
    #    plt.title('Solution conditional on q0')
    #    plt.show()
    
    # Compute conditional average using Monte Carlo
    yc_ave_MC[:,i] = 1/n1*np.sum(yc_data,axis=1)
np.save('yc_ave_MC',yc_ave_MC)

y_ave_HYB =  np.zeros(m)
for n in range(n0):
    y_ave_HYB += wSG[n]*y_SG[:,n]/c_norm
    #plt.plot(x,(log_q0.data()[:,n].T+log_qc.T).T, 'k', linewidth=0.1, alpha=0.5)
plt.plot(x,y_ave_SG, 'k', label='coarse')
plt.plot(x,y_ave_HYB, 'k--',label='hybrid')
plt.legend()
plt.show()

"""
  
# =============================================================================
# Compute Reduced Order Model
# ============================================================================= 
M = assembler.af[1]['bilinear'].get_matrix()
y_train = y_data[:,i_train]
y_test = y_data[:,i_test]
U,S,Vt = la.svd(y_train)

x = dofhandler.get_dof_vertices()

m = 8
d = 7

Um = U[:,:m]
plt.plot(x,Um,'k')

# Test functions
i_left = dofhandler.get_region_dofs(entity_flag='left', entity_type='vertex')
B = Um[i_left,:].T

plt.plot(np.tile(x[i_left],B.shape),B,'r.')
plt.show()

Q,R = la.qr(B, mode='full')
psi = Um.dot(Q[:,1:])
plt.plot(x,psi)
plt.show()


rom_tol = 1e-10
rom_error = 1-np.cumsum(S)/np.sum(S)
n_rom = np.sum(rom_error>=rom_tol)
print(n_rom)
Ur = U[:,:n_rom]

Am = np.empty((m,m))
Am[:d,:] = Q[:,1:].T.dot(Um.T.dot(A[0].dot(Um)))
Am[-1,:] = B.ravel()

bm = np.zeros((m,1))
bm[:d,:] = Q[:,1:].T.dot(Um.T.dot(b.toarray()))
bm[-1,:] = 1


c = la.solve(Am,bm)
plt.plot(x,y_data[:,[0]],'k',x,Um.dot(c),'r') 
plt.show()

print(Am.shape)
#plt.plot(x,Ur)
#plt.show()

# =============================================================================
# Predict output using ROM
# ============================================================================= 
u_rom = np.empty((n,n_train))
br = b.T.dot(Ur).T 
for i in np.arange(n_train):
    Ar = Ur.T.dot(A[i_train[i]].dot(Ur)) 
    cr = la.solve(Ar, br)
    u_rom[:,[i]] = Ur.dot(cr)


# =============================================================================
# Compare ROM output with direct numerical simulation
# ============================================================================= 

#plt.plot(x,u_rom,'k',x,y_data[:,i_train])
#plt.show()

du = np.empty((n,n_train))
for i in range(n_train):
    du[:,i] = u_rom[:,i]-y_train[:,i]
    #du[:,i] = Ur.dot(Ur.T.dot(u_test[:,i])) - u_test[:,i]


u_error = Nodal(dofhandler=dofhandler, data=du)
#u_error = np.dot(du.T, M.dot(du))
#plot.line(u_error, i_sample=np.arange(0,n_train))
"""
