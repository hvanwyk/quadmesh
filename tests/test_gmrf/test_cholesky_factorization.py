from operator import is_
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
    
    A_spd = np.array([[1, 2, 3, 0], 
                      [2, 5, 7, 0], 
                      [3, 7, 10, 0],
                      [0, 0, 0, 1]])
    
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
            return sparse.csc_matrix(A_spd)
        else:
            #
            # Positive definite
            #
            return sparse.csc_matrix(A_pd)


class TestCholeskyDecomposition(unittest.TestCase):
    
    def test_indicators(self):
        # Test Cholesky decomposition for a (semi) positive definite matrix
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
                self.assertEqual(cholesky.is_degenerate(),degenerate)

                # Assert that the matrix is/isn't full
                self.assertEqual(cholesky.is_sparse(),not(dense) and not(degenerate))        

                # Check the size
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

            if cholesky.is_sparse():
                #
                # Sparse matrix
                #
                A_rec = cholesky.reconstruct()
                self.assertTrue(np.allclose(A.toarray(), A_rec.toarray()))
            else:
                #
                # Dense matrix
                #
                A_rec = cholesky.reconstruct()
                self.assertTrue(np.allclose(A, A_rec))
 

    def test_modified_cholesky(self):
        # Test Cholesky decomposition for a non-positive definite matrix
        A = test_matrix(degenerate=True)
        cholesky = CholeskyDecomposition(A)
   

    def test_reconstruct(self):
        # Test reconstruction of the original matrix
        A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
        cholesky = CholeskyDecomposition(A)
        L = cholesky.get_factors()
        A_reconstructed = cholesky.reconstruct()
        self.assertTrue(np.allclose(A, A_reconstructed))

    def test_solve_non_degenerate(self):
        """
        Test solving a linear system of equations. 
        Only non-degenerate matrices are considered.
        """
        # Test solving a linear system of equations
        for dense in [True, False]:
            #
            # Dense and sparse matrices
            # 
            # Generate a test matrix
            A = test_matrix(dense=dense, degenerate=False)

            # Compute the Cholesky decomposition of A
            cholesky = CholeskyDecomposition(A)

            # Construct the right hand side
            x_ref = np.array([1, 2, 3, 4])
            b = A.dot(x_ref)

            # Solve the linear system via Cholesky factors
            x = cholesky.solve(b)

            self.assertTrue(np.allclose(x_ref, x))

    def test_solve_degenerate(self):
        """
        Test solving a linear system of equations with a degenerate matrix.
        """
        # Test solving a linear system of equations
        for dense in [True, False]:
            #
            # Dense and sparse matrices
            # 
            # Generate a test matrix
            A = test_matrix(dense=dense, degenerate=True)

            # Compute the Cholesky decomposition of A
            cholesky = CholeskyDecomposition(A)

            # Construct the right hand side
            x_ref = np.array([1, 2, 3, 4])
            b = A.dot(x_ref)

            # Solve the linear system via Cholesky factors
            x = cholesky.solve(b)

            # The solutions aren't necessarily equal, but they should both be in the nullspace of A
            self.assertTrue(np.allclose(A.dot(x),b))

    def test_dot(self):
        for dense in [True, False]:
            #
            # Dense and sparse matrices
            # 
            for degenerate in [True, False]:
                #
                # Degenerate and non-degenerate matrices
                #
                # Generate a test matrix
                A = test_matrix(dense=dense, degenerate=degenerate)

                # Compute the Cholesky decomposition of A
                cholesky = CholeskyDecomposition(A)

                # Construct the right hand side
                x = np.array([1, 2, 3, 4])
                b = A.dot(x)

                # Multiply via Cholesky decomposition
                if not(dense):
                    # Sparse matrix multiplied by a sparse vector
                    x_sp = sparse.csc_matrix(x[:,np.newaxis])
                    Ax = cholesky.dot(x_sp)
                    self.assertTrue(np.allclose(Ax.toarray().ravel(), b))

                    # Sparse matrix multiplied by a dense vector
                    Ax = cholesky.dot(x)
                    self.assertTrue(np.allclose(Ax, b))
                else:
                    # Dense matrix multiplied by a dense vector
                    Ax = cholesky.dot(x)
                    self.assertTrue(np.allclose(Ax, b)) 
        
    def test_sqrt_dot(self):
        x = np.random.rand(4)
        for dense in [True, False]:
            #
            # Dense and sparse matrices
            # 
            for degenerate in [True, False]:
                #
                # Degenerate and non-degenerate matrices
                #
                # Generate a test matrix
                A = test_matrix(dense=dense, degenerate=degenerate)

                # Compute the Cholesky decomposition of A
                cholesky = CholeskyDecomposition(A)

                #
                # Test: Multiply via Cholesky decomposition
                #
                y = cholesky.sqrt_dot(x, transpose=True)
                b = cholesky.sqrt_dot(y, transpose=False)
                self.assertTrue(np.allclose(b, A.dot(x)))

    def test_sqrt_solve(self):
        x_ref = np.random.rand(4)
        for dense in [True, False]:
            #
            # Dense and sparse matrices
            # 
            for degenerate in [True, False]:
                #
                # Degenerate and non-degenerate matrices
                #
                # Generate a test matrix
                A = test_matrix(dense=dense, degenerate=degenerate)
                b = A.dot(x_ref)

                # Compute the Cholesky decomposition of A
                cholesky = CholeskyDecomposition(A)

                #
                # Test: Solve using Cholesky decomposition
                #
                y = cholesky.sqrt_solve(b, transpose=False)
                x = cholesky.sqrt_solve(y, transpose=True)
                print('degenerate:',degenerate,'sparse:',not(dense))

                if not(degenerate) and not(dense):
                    """
                    CHOLMOD
                    """
                    print('A:',A.toarray())
                    print('b:',b)
                    L = cholesky.get_factors()
                    LL = L.L().toarray()
                    bb = L.apply_Pt(LL.dot(LL.T.dot(L.apply_P(x_ref))))
                    print('Factor',LL)
                    print('Permutation\n',L.P())

                    yy = np.linalg.solve(LL,L.apply_Pt(b))
                    yyy = L.solve_L(L.apply_P(b))
                    print('yyy:',yyy)
                    print('yy:',yy)
                    print('y',y )
                    #print('bb:',bb)
                print('Computed solution', x)
                print('Reference solution', x_ref)
                #self.assertTrue(np.allclose(b, A.dot(x)))
        

if __name__ == '__main__':
    unittest.main()
