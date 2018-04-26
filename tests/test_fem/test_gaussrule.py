import unittest
from fem import GaussRule
from mesh import Cell, QuadCell, HalfEdge, Vertex
import numpy as np 


class TestGaussRule(unittest.TestCase):
    """
    Test GaussRule class
    """
    def test_constructor(self):
        pass
    
    def test_nodes(self):
        pass
    
    def test_weights(self):
        pass
    
    def test_n_nodes(self):
        pass
    
    def test_map(self):
        pass
    
    def test_inverse_map(self):
        pass
    
    def test_jabobian(self):
        pass
    

    def test_line_integral(self):
        # Define quadrature rule
        rule = GaussRule(2, shape='edge')
        w = rule.weights()
        
        # function f to be integrated over edge e
        f = lambda x,y: x**2*y
        e = HalfEdge(Vertex((0,0)),Vertex((1,1)))
        
        # Map rule to physical entity
        x_ref = rule.map(e)
        jac = rule.jacobian(e)
        fvec = f(x_ref[:,0],x_ref[:,1])
        
        self.assertAlmostEqual(np.sum(np.dot(fvec,w))*jac,1/np.sqrt(2)/2,places=10,\
                               msg='Failed to integrate x^2y.')
        self.assertAlmostEqual(np.sum(w)*jac, np.sqrt(2), places=10,\
                               msg='Failed to integrate 1.')
        
   