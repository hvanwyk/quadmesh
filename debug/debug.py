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
import scipy.sparse as sp

A = np.array([[1, 1, 0, 1],
              [1, 1, 1, 0],
              [0, 1, 1, 1],
              [1, 0, 1, 1]])

s, dummy = linalg.eigh(A)
print('eigenvalues', s)

delta = None

if delta is None:
    eps = np.finfo(float).eps
    delta = np.sqrt(eps)*linalg.norm(A, 'fro')
else:
    assert delta>0, 'Input "delta" should be positive.'

n = max(A.shape)

print('n=', n)
print('delta=',delta)


L,D,p = linalg.ldl(A)  
DMC = np.eye(n)

print('L=', L)
print('D=',D)
print('p=',p)

print('L(p)', L[p,:])
print('A-LDL.t', A-L.dot(D.dot(L.T)))

# Modified Cholesky perturbations.
k = 0
while k < n:
    print('k=',k)
    one_by_one = False
    if k == n-1:
        one_by_one = True
    elif D[k,k+1] == 0:
        one_by_one = True
        
    if one_by_one:
        #            
        # 1-by-1 block
        #
        print('1x1 block')
        if D[k,k] <= delta:
            DMC[k,k] = delta
        else:
            DMC[k,k] = D[k,k]
     
        k += 1
  
    else:  
        print('2x2 block')
        #            
        # 2-by-2 block
        #
        E = D[k:k+2,k:k+2]
        T,U = linalg.eigh(E)
        T = np.diag(T)
        for ii in range(2):
            if T[ii,ii] <= delta:
                T[ii,ii] = delta
        
        temp = np.dot(U,np.dot(T,U.T))
        print(temp)
        DMC[k:k+2,k:k+2] = (temp + temp.T)/2  # Ensure symmetric.
        k += 2
        
P = np.eye(n) 
P = P[p,:]
    
print('DMC',DMC)
print('PAP.t', P.dot(A.dot(P.T)))
print('LDL.t-A', L.dot(D.dot(L.T))-A)
print('LD0L.t', L.dot(DMC.dot(L.T)))


# Test Cholmod
A = sp.diags(np.random.rand(5)-0.5)
f = cholesky(A.tocsc(),mode='supernodal')
print(A.toarray())
L, D = f.L_D()
print(L.toarray())
print(D)

U,S,VT = np.linalg.svd(A.toarray())
print(S)
print(U)
print(VT)
"""
Debug conditioning
"""
"""

scipy_version()
V = np.array([[0.5, 1/np.sqrt(2), 0, 0.5], 
              [0.5, 0, -1/np.sqrt(2), -0.5],
              [0.5, -1/np.sqrt(2), 0, 0.5], 
              [0.5, 0, 1/np.sqrt(2), -0.5]])

s = np.array([4,3,2,0])
S = V.dot(np.dot(np.diag(s),V.T))

# Generate a sample of x
Z = np.random.normal(size=(4,1))
X = V.dot(np.dot(np.diag(np.sqrt(s)),Z))



Vk = V[:,:3]
Pi = np.dot(Vk, Vk.T)

A = np.eye(4,4)[0,None]
"""