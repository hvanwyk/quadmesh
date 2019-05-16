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
from plot import Plot
import numpy as np
from mesh import HalfEdge
import matplotlib.pyplot as plt
from scipy import linalg
from sksparse.cholmod import cholesky, cholesky_AAt, Factor
from sklearn.datasets import make_sparse_spd_matrix
import scipy.sparse as sp
from gmrf import modchol_ldlt
from gmrf import KLExpansion
from gmrf import CovKernel
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

mesh = QuadMesh(resolution=(2,2))
element = QuadFE(2,'Q1')
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()
dofhandler.set_dof_vertices()
x = dofhandler.get_dof_vertices()
dim = 2
k = CovKernel('gaussian', {'sgm': 2, 'l': 0.2, 'M': None}, dim)
print(x,'\n')
I,J = np.mgrid[0:9,0:9]
X = x[I,:]
Y = x[J,:]
print(X.reshape((81,2)),'\n')
print(Y.reshape((81,2)))
X = KLExpansion(k, dofhandler)
