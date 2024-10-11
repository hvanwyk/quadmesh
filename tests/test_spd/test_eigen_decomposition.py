import unittest
import xxlimited
from spd import EigenDecomposition
from numpy import array, zeros, allclose, random, diag, eye, abs, max
from numpy.linalg import eig, qr, det, norm

def generate_spd_matrix(n,rank):
    # Generate a random symmetric positive semi-definite matrix with a given rank

    # Generate random orthogonal matrix Q (eigenvectors)
    A = random.rand(n,n)
    Q, _ = qr(A)
    
    # Generate diagonal matrix D with rank nonzero diagonal entries (eigenvalues)
    d = zeros(n)
    d[:rank] = array([1.0/(i+1) for i in range(rank)])
    D = diag(d)
    
    # Form the symmetric positive semi-definite matrix A = Q D Q^T
    A = Q.dot(D).dot(Q.T)

    return A, d, Q

class TestEigenDecomposition(unittest.TestCase):

    def test_eigenvalues(self):
        # Existing test for eigenvalues
        
        # Generate a random full rank matrix
        n = 10 # Size of the matrix
        r = n  # Rank of the matrix (full rank)
        A, d, _ = generate_spd_matrix(n,r)
        
        # Compute the eigendecomposition of the matrix
        factor = EigenDecomposition(A)

        # Check the eigenvalues
        dd = factor.get_eigenvalues()
        self.assertTrue(allclose(d, dd))

        # Generate a random rank deficient matrix
        n = 10 
        r = 5
        A, d, _ = generate_spd_matrix(n,r)

        # Compute the eigendecomposition of the matrix (default threshold)
        factor = EigenDecomposition(A)

        # Check the non-zero eigenvalues 
        dd = factor.get_eigenvalues()
        self.assertTrue(allclose(d[:5], dd[:5]))

        # Check whether the zero eigenvalues are correctly modified
        delta = factor.get_eigenvalue_lower_bound()
        self.assertTrue(allclose(dd[5:], delta))

        # Compute the eigendecomposition of the matrix (0 threshold)
        factor = EigenDecomposition(A, delta=0)

        # Check whether the zero eigenvalues are maintained
        dd = factor.get_eigenvalues()
        self.assertTrue(allclose(dd[r:],0))


    def test_eigenvectors(self):
        # 
        # Test whether the eigenvectors are correctly computed
        # 
        # Generate a random full rank matrix
        n = 10
        r = n
        A, _, Q = generate_spd_matrix(n,r)

        # Compute the eigendecomposition of the matrix
        factor = EigenDecomposition(A)

        # Check the eigenvectors
        QQ = factor.get_eigenvectors()

        # Check whether the eigenvectors are orthogonal
        self.assertTrue(allclose(QQ.T.dot(QQ), eye(n)))

        # Check whether the eigenvectors have length 1
        for i in range(n):
            self.assertTrue(allclose(norm(QQ[:,i]), 1))

        # Check whether eigenvectors are orthonormal
        self.assertTrue(allclose(QQ.T.dot(QQ), eye(n)))


    def test_rank(self):
        # Check that the rank is correctly computed
        for r in [5, 10]:
            A, _, _ = generate_spd_matrix(10,r)
            factor = EigenDecomposition(A,delta=0)
            self.assertEqual(factor.get_rank(), r)


    def test_reconstruct(self):
        # Test the reconstruction of the original matrix
        A,_,_ = generate_spd_matrix(10,10)
        factor = EigenDecomposition(A)
        A_rec = factor.reconstruct()
        self.assertTrue(allclose(A,A_rec))
        

    def test_dot(self):
        # Test matrix-vector multiplication
        A,_,_ = generate_spd_matrix(10,10)
        factor = EigenDecomposition(A)
        x = random.rand(10)
        b = factor.dot(x)
        self.assertTrue(allclose(b,A.dot(x)))
    

    def test_solve(self):
        # 
        # Test the solution of a linear system
        # 
        # Generate a random full rank matrix
        n = 10
        r = n
        A, _, _ = generate_spd_matrix(n,r)
        factor = EigenDecomposition(A)

        # Compute the solution of a linear system of equations
        x = random.rand(n)
        b = A.dot(x)

        # Solve the linear system of equations
        xx = factor.solve(b)

        # Check the solution
        self.assertTrue(allclose(x,xx))


    def test_sqrt_dot(self):
        # Test matrix-square root multiplication
        A,_,_ = generate_spd_matrix(10,10)
        factor = EigenDecomposition(A)
        x = random.rand(10)
        b = factor.sqrt_dot(x,transpose=True)
        bb = factor.sqrt_dot(b, transpose=False)
        self.assertTrue(allclose(bb,A.dot(x)))


    def test_sqrt_solve(self):
        # Test the solution of a linear system using the square root
        # Generate a random full rank matrix
        n = 10
        r = n
        A, _, _ = generate_spd_matrix(n,r)
        factor = EigenDecomposition(A)

        # Compute the solution of a linear system of equations
        x = random.rand(n)
        b = A.dot(x)

        # Solve the linear system of equations
        y  = factor.sqrt_dot(x, transpose=True)
        yy = factor.sqrt_solve(b, transpose=False)
        self.assertTrue(allclose(y,yy))

        xx = factor.sqrt_solve(yy, transpose=True)

        # Check the solution
        self.assertTrue(allclose(x,xx))

if __name__ == '__main__':
    unittest.main()