import sys

# Get the folder path
folder_path = '/home/hans-werner/git/quadmesh/quadmesh/src'

# Add the folder path to the system path if it's not already there
if folder_path not in sys.path:
    sys.path.append(folder_path)

from spd import CholeskyDecomposition
from scipy import sparse
import numpy as np

import unittest


def test_matrix(dense=False, degenerate=False):
    """
    Returns a test matrix for the Cholesky decomposition.
    """
    A_pd = np.array([[10, 0, 3, 0],
                   [0, 5, 0, -2],
                   [3, 0, 5, 0],
                   [0, -2, 0, 2]])
    
    A_spd = np.array([[1, 2, 3], 
                     [2, 5, 7], 
                     [3, 7, 10]])
    if dense:
        #
        # Returns a dense symmetric test matrix
        #
        if degenerate:
            #
            # Not positive definite
            # 
            return A_spd
        else:
            #
            # Positive definite
            #
            return A_pd
    else:
        #
        # Returns a sparse symmetric test matrix
        #
        if degenerate:
            #
            # Not positive definite
            # 
            sparse.csc_matrix(A_spd)
        else:
            #
            # Positive definite
            #
            return sparse.csc_matrix(A_pd)


class TestCholeskyDecomposition(unittest.TestCase):
    
    def test_indicators(self):
        # Test Cholesky decomposition for a positive definite matrix
        for dense in [True, False]:
            #
            # Dense and sparse matrices
            # 
            for degenerate in [True, False]:
                #
                # Degenerate and non-degenerate matrices
                #
                A = test_matrix(dense=dense, degenerate=degenerate)
                cholesky = CholeskyDecomposition(A)
                
                # Assert that the matrix is/isn't positive definite
                self.assertEqual(cholesky.isdegenerate(),degenerate)

                # Assert that the matrix is/isn't full
                self.assertEqual(cholesky.issparse(),sparse)        

                # Check the size
                print(cholesky.size())
                self.assertEqual(cholesky.size(), A.shape[0])

    def test_reconstruct_positive_definite(self):
        """
        Decompose a positive definite matrix and reconstruct it.
        """
        for dense in [True, False]:
            #
            # Dense and sparse matrices
            # 
            A = test_matrix(dense=dense)
            cholesky = CholeskyDecomposition(A)
            A_reconstructed = cholesky.reconstruct()
            self.assertTrue(np.allclose(A, A_reconstructed))
        #L = cholesky.get_factors()
        #expected_L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        #self.assertEqual(L, expected_L)

    def test_modified_cholesky(self):
        # Test Cholesky decomposition for a non-positive definite matrix
        A = test_matrix(degenerate=True)
        cholesky = CholeskyDecomposition(A)
   

    def test_reconstruct(self):
        # Test reconstruction of the original matrix
        A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
        cholesky = CholeskyDecomposition(A)
        L = cholesky.get_factors()
        A_reconstructed = cholesky.reconstruct(L)
        self.assertTrue(np.allclose(A, A_reconstructed))

    def test_solve(self):
        # Test solving a linear system of equations
        A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
        cholesky = CholeskyDecomposition(A)
        L = cholesky.get_factors()
        b = np.array([1, 2, 3])
        x = cholesky.solve(L, b)
        expected_x = np.array([1, 2, 3])
        self.assertTrue(np.allclose(x, expected_x)) 
    
    def test_solve_sparse(self):
        # Test solving a linear system of equations
        A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
        cholesky = CholeskyDecomposition(A)
        L = cholesky.get_factors(sparse=True)
        b = np.array([1, 2, 3])
        x = cholesky.solve(L, b)
        expected_x = np.array([1, 2, 3])
        self.assertTrue(np.allclose(x, expected_x))

if __name__ == '__main__':
    unittest.main()
