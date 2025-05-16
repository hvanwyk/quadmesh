import sys
if '/home/hans-werner/git/quadmesh/src' not in sys.path:
    sys.path.append('/home/hans-werner/git/quadmesh/src')

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
from solver import LinearSystem

# Built-in modules
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

"""
Implement Reduced order Model 

-d/dx(q(x,w)d/dx u(x)) = f(x)
u(0) = 1
u(1) = 0

"""
mesh = Mesh1D(resolution=(100,))
mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)

element = QuadFE(mesh.dim(), 'Q1')
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()
n = dofhandler.n_dofs()

# =============================================================================
# Random field
# =============================================================================
n_samples = 10
n_train, n_test = 8, 2
i_train = np.arange(n_train)
i_test = np.arange(n_train,n_samples)

cov = Covariance(dofhandler, name='gaussian', parameters={'l':0.1})
cov.compute_eig_decomp()
d,V = cov.get_eig_decomp()
plt.semilogy(d,'.')
plt.show()
log_q = GaussianField(n, K=cov)
log_q.update_support()
qfn = Nodal(dofhandler=dofhandler, 
            data=np.exp(log_q.sample(n_samples=n_samples)))

plot = Plot()
plot.line(qfn,i_sample=np.arange(n_samples))

# =============================================================================
# Generate Snapshot Set
# ============================================================================= 
phi = Basis(dofhandler, 'u')
phi_x = Basis(dofhandler, 'ux')

problems = [[Form(kernel=qfn, trial=phi_x, test=phi_x), Form(1, test=phi)],
            [Form(1, test=phi, trial=phi)]]

assembler = Assembler(problems, mesh)
assembler.assemble()

A = assembler.af[0]['bilinear'].get_matrix()
b = assembler.af[0]['linear'].get_matrix()

linsys = LinearSystem(phi)
linsys.add_dirichlet_constraint('left',1)
linsys.add_dirichlet_constraint('right',0)

y_snap = Nodal(dofhandler=dofhandler, 
               data=np.empty((n,n_samples)))
y_data = np.empty((n,n_samples))
for i in range(n_samples):
    linsys.set_matrix(A[i].copy())
    linsys.set_rhs(b.copy())
    linsys.solve_system()
    y_data[:,[i]] = linsys.get_solution(as_function=False)
y_snap.set_data(y_data)
plot = Plot()
plot.line(y_snap, i_sample=np.arange(n_samples))

  
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

