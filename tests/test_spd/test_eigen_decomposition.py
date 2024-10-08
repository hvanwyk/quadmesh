import unittest
from spd import EigenDecomposition
from numpy import array, allclose
from numpy.linalg import eig, qr, det, norm

def generate_spd_matrix(n,rank):
    # Generate a random symmetric positive semi-definite matrix with a given rank
    
    A = array([[0.0]*n]*n)
    for i in range(n):
        for j in range(i+1):
            A[i,j] = A[j,i] = 0.1*(i+1)*(j+1)
    Q, R = qr(A)
    D = array([[0.0]*n]*n)
    for i in range(n):
        D[i,i] = 1.0/(i+1)
    A = Q.dot(D).dot(Q.T)
    return A

class TestEigenDecomposition(unittest.TestCase):
    def test_eigenvalues(self):
        # Existing test for eigenvalues
        # ...
        # Generate a random symmetric posi
        n = 10
        r = 5
        A = generate_spd_matrix(n,r)
        eig_fac = EigenDecomposition(A)

        # Check the eigenvalues
        

    def test_eigenvectors(self):
        # Existing test for eigenvectors
        # ...
        pass

    def test_trace(self):
        # Existing test for trace
        # ...
        pass

    def test_determinant(self):
        # Existing test for determinant
        # ...
        pass

    def test_rank(self):
        # New test to check an additional property of the eigen decomposition
        # ...
        pass

    def test_dot(self):
        # New test to check another property of the eigen decomposition
        # ...
        pass
    

    def test_solve(self):
        # New test to check the solve method
        # ...
        pass

if __name__ == '__main__':
    unittest.main()