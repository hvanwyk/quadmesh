import unittest
from fem import QuadFE


class TestFiniteElement(unittest.TestCase):
    """
    Test FiniteElement class
    """
    def test_cell_type(self):
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            t = element.cell_type()
            self.assertEqual(t,'quadrilateral','Type should be quadrilateral.')
