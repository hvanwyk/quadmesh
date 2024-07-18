import unittest
from spd import CholeskyDecomposition

class CholeskyDecompositionTests(unittest.TestCase):
    def test_decomposition(self):
        # Existing test case
        matrix = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        cholesky = CholeskyDecomposition(matrix)
        self.assertEqual(cholesky.decomposition(), [[2, 0, 0], [6, 1, 0], [-8, 5, 3]])

    def test_positive_definite(self):
        # New test case to check positive definiteness
        matrix = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        cholesky = CholeskyDecomposition(matrix)
        self.assertTrue(cholesky.is_positive_definite())

    def test_symmetric(self):
        # New test case to check symmetry
        matrix = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        cholesky = CholeskyDecomposition(matrix)
        self.assertTrue(cholesky.is_symmetric())

    def test_lower_triangular(self):
        # New test case to check lower triangularity
        matrix = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
        cholesky = CholeskyDecomposition(matrix)
        self.assertTrue(cholesky.is_lower_triangular())

if __name__ == '__main__':
    unittest.main()