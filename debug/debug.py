from mesh import Mesh1D
from mesh import QuadMesh
from fem import DofHandler
from function import Function
from function import Nodal
from fem import QuadFE
from assembler import Kernel
from assembler import Form
from fem import Basis
from assembler import Assembler
from solver import LinearSystem
from solver import LS
from plot import Plot
import numpy as np
from mesh import HalfEdge
import matplotlib.pyplot as plt
from scipy import linalg
from sksparse.cholmod import cholesky, cholesky_AAt, Factor
from sklearn.datasets import make_sparse_spd_matrix
import scipy.sparse as sp
from gmrf import modchol_ldlt
from gmrf import KLField
from gmrf import CovKernel
import TasmanianSG
'''
# Eigenvectors 
oort = 1/np.sqrt(2)
V = np.array([[0.5, oort, 0, 0.5], 
              [0.5, 0, -oort, -0.5],
              [0.5, -oort, 0, 0.5],
              [0.5, 0, oort, -0.5]])

# Eigenvalues
d = np.array([4,3,2,0], dtype=float)
Lmd = np.diag(d)

# Covariance matrix
K = V.dot(Lmd.dot(V.T))

# Transformation
A = np.array([[1,2,3,4],
              [2,4,6,8]], dtype=float)

# Nullspace of covariance
Vn = V[:,3][:,None]

for v in A:
    u = v - Vn.dot(Vn.T.dot(v))
    if not np.allclose(u,0):
        u = u/linalg.norm(u)
        Vn = np.append(Vn, u[:,None], axis=1)


Q,R = linalg.qr(A.T, mode='economic')
print(Q)
print(R)
Q,R = linalg.qr_insert(Vn,Lmdn,A[0,:], -1, which='col')
#Q,R = linalg.qr_insert(Q,R,A[1,:], -1, which='col')
Q.T.dot(A[1,:].T)
pp = A[1,:] - Q.dot(Q.T.dot(A[1,:]))

print(pp)
print('R\n',R)

print(A.dot(V))
Q,R = linalg.qr(A, mode='economic')
r = np.diag(R)
print(len(r[np.abs(r)>1e-13]))
print(Q,'\n',R)

'''
print("TasmanianSG version: {0:s}".format(TasmanianSG.__version__))
print("TasmanianSG license: {0:s}".format(TasmanianSG.__license__))


mesh = Mesh1D(resolution=(2,))
element = QuadFE(mesh.dim(),'Q1')
dofhandler = DofHandler(mesh, element)

phi_x = Basis(dofhandler, 'ux')

problems = [Form(1, test=phi_x, trial=phi_x)]
assembler = Assembler(problems, mesh)
assembler.assemble()

A = assembler.af[0]['bilinear'].get_matrix()
n = dofhandler.n_dofs()
b = np.ones((n,1))


mesh.mark_region('left',lambda x: np.abs(x)<1e-9)
mesh.mark_region('right',lambda x: np.abs(1-x)<1e-9)

print('A before constraint', A.toarray())

system = LS(phi_x)
system.add_dirichlet_constraint('left',1)
system.add_dirichlet_constraint('right',0)

system.set_matrix(sp.csr_matrix(A, copy=True))
system.set_rhs(b) 




system.solve_system()

print('A after constraint\n', system.get_matrix().toarray())
print('column records\n', system.column_records)
print('rhs after constraint\n', system.get_rhs().toarray())
y = system.get_solution()


plot = Plot()
plot.line(y)


b = np.zeros((n,1))
system.set_matrix(sp.csr_matrix(A, copy=True))
system.set_rhs(b)
system.solve_system()
print('column records\n')
print([c.toarray() for c in system.column_records])
print('rhs after constraint\n', system.get_rhs().toarray())
y = system.get_solution()
plot = Plot()
plot.line(y)


