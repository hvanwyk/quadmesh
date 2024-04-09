import unittest
from gmrf import CholeskyDecomposition

class TestCholeskyDecomposition(unittest.TestCase):
    def test_decomposition(self):
        # Test Cholesky decomposition for a positive definite matrix
        A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        cholesky = CholeskyDecomposition(A)
        L = cholesky.decompose()
        expected_L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
        self.assertEqual(L, expected_L)

    def test_invalid_matrix(self):
        # Test Cholesky decomposition for a non-positive definite matrix
        A = [[1, 2, 3], [2, 5, 7], [3, 7, 10]]
        cholesky = CholeskyDecomposition(A)
        with self.assertRaises(ValueError):
            cholesky.decompose()

if __name__ == '__main__':
    unittest.main()
