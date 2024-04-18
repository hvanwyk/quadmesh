import gmrf
import importlib
importlib.reload(gmrf)

from gmrf import CholeskyDecomposition, SPDMatrix
import numpy as np
print(gmrf.__file__)

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

chol = CholeskyDecomposition(A)
print(issubclass(CholeskyDecomposition, SPDMatrix))
print(CholeskyDecomposition.mro())
#chol.set_degeneracy(False)
print('Degenerate?', chol.isdegenerate())
print('Sparse?', chol.issparse())
