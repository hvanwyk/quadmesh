
from spd import CholeskyDecomposition, SPDMatrix
import numpy as np
import scipy.sparse as sp

#import sys
"""
# Get the folder path
folder_path = '/home/hans-werner/git/quadmesh/src'

# Add the folder path to the system path if it's not already there
if folder_path not in sys.path:
    sys.path.append(folder_path)
"""

"""
Things to check:
isdegenerate
issparse
reconstruct
dot
solve
sqrt_dot
sqrt_solve
"""
#
# Positive definite, full matrix
#
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

#
# Positive definite, sparse matrix
# 
A_sp = sp.csc_matrix(A)
cholesky = CholeskyDecomposition(A_sp)
assert cholesky.issparse() is True,\
    'Cholesky decomposition of sparse matrix is not sparse'
assert cholesky.isdegenerate() is False,\
    'Cholesky decomposition of matrix should be non-degenerate'

# Reconstruct the matrix
AA = cholesky.reconstruct()
assert np.allclose(A_sp.toarray(), AA.toarray()),\
    'Reconstructed matrix is incorrect'

# Solve using Cholesky decomposition
assert np.allclose(cholesky.solve(b),x),\
    'Solution using Cholesky decomposition is incorrect'

# Multiply via Cholesky decomposition
assert np.allclose(cholesky.dot(x),b),\
    'Matrix-vector product using Cholesky decomposition is incorrect'

# Check size
assert cholesky.size() == 4,\
    'Size of Cholesky decomposition is incorrect'

# Test sqrt_dot
y = cholesky.sqrt_dot(x,transpose=True)
bb = cholesky.sqrt_dot(y,transpose=False)
print(bb-b)
assert np.allclose(bb,b), 'sqrt_dot is incorrect'

# Test sqrt_solve

L = cholesky.get_factors()
print('permutation matrix', L.P())
L = np.linalg.cholesky(A)
chol = CholeskyDecomposition(A)

#chol.set_degeneracy(False)
print('Degenerate?', chol.isdegenerate())
print('Sparse?', chol.issparse())