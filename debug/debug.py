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

def test_matrix(n, sparse=False):
    """
    Returns symmetric matrices on which to test algorithms
    
    Inputs:
    
        n: int, matrix size
        
        sparse: bool (False), sparsity
        
        rank: str/int, if 'full', then rank=n, otherwise rank=r in {1,2,...,n}.
        
    Output:
    
        A: double, symmetric positive definite matrix with specified rank
            (hopefully) and sparsity.
    """    
    if sparse:
        #
        # Sparse matrix 
        # 
        A = make_sparse_spd_matrix(dim=n, alpha=0.95, norm_diag=False,
                           smallest_coef=.1, largest_coef=.9);
        A = sp.csc_matrix(A)
    else:
        #
        # Full matrix
        #
        X = np.random.rand(n, n)
        X = X + X.T
        U, s, V = linalg.svd(np.dot(X.T, X))
        A = np.dot(np.dot(U, 0.5 + np.diag(np.random.rand(n))), V)
         
    return A

n = 100
A = test_matrix(n,True)
s = linalg.eigvalsh(A.toarray())
F = cholesky(A)

L = F.L()
P = F.P()

AA = L.dot(L.T)
A = A.toarray()
AAA = A[P[:, np.newaxis], P[np.newaxis, :]] 
#print(AA.toarray())

print(np.allclose(AA.toarray(),AAA))

