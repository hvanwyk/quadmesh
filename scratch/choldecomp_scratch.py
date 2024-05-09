
from spd import CholeskyDecomposition, SPDMatrix
import numpy as np


#import sys
"""
# Get the folder path
folder_path = '/home/hans-werner/git/quadmesh/src'

# Add the folder path to the system path if it's not already there
if folder_path not in sys.path:
    sys.path.append(folder_path)
"""

A = np.array([[10, 0, 3, 0],
              [0, 5, 0, -2],
              [3, 0, 5, 0],
              [0, -2, 0, 2]])
b = np.array([1, 2, 3, 4])
x = np.linalg.solve(A, b)

# Compute the Cholesky decomposition of A
cholesky = CholeskyDecomposition(A)
print('Degenerate?',cholesky.isdegenerate())
print('Sparse?',cholesky.issparse())
print('Size:', cholesky.size())

L = cholesky.get_factors()
print(L)

AA = cholesky.reconstruct()
print('Reconstructed matrix correct?', np.allclose(A, AA))

# Solve using Cholesky decomposition
x = cholesky.solve(b)
print('Solution:', x)
print('Solution correct?', np.allclose(np.dot(A, x), b))

# Multiply via Cholesky decomposition
Ax = cholesky.dot(x)
print('Matrix-vector product close?', np.allclose(Ax, b))
print(np.linalg.matrix_rank(A))
L = np.linalg.cholesky(A)
chol = CholeskyDecomposition(A)
print(issubclass(CholeskyDecomposition, SPDMatrix))
print(CholeskyDecomposition.mro())
#chol.set_degeneracy(False)
print('Degenerate?', chol.isdegenerate())
print('Sparse?', chol.issparse())
