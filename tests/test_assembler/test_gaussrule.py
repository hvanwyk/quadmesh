import unittest
from assembler import GaussRule
from fem import  QuadFE
from mesh import Cell, QuadCell, HalfEdge, Vertex, convert_to_array
import numpy as np 
class TestGaussRule(unittest.TestCase):
    """
    Test GaussRule class
    """
    def test_accuracy(self):
        """
        Test the Accuracy of the Gauss rule on the reference cell
        
        1D Rule: (Interval) 
            Number of Nodes: 1,2,3,4,5,6
            Accuracy: 2N-1 
             
        2D Rule (Quadrilateral): 
            Number of Nodes: 1,4,9,16,25,36 
            Accuracy: (2n-1, 2n-1), where n = sqrt(N)
            
        TODO: 2D Rule (Triangle) 
        """
        #
        # 1D Polynomials and exact integrals
        # 
        order_1d = [1,2,3,4,5,6]
        polynomials_1d = {1: lambda x: 2*np.ones(x.shape), 
                          2: lambda x: x**3-2*x**2 + x + 1, 
                          3: lambda x: x**5 - 2*x**2 + 2,
                          4: lambda x: x**7 + 2*x**6 + 3*x,
                          5: lambda x: x**9 + 3*x**4 + 1, 
                          6: lambda x: x**11 - 3*x**8 + 1}
        
        integrals_1d = {1: 2,
                        2: 13/12,
                        3: 3/2, 
                        4: 107/56, 
                        5: 17/10,
                        6: 3/4}
        
        #
        # 2D Polynomials and exact integrals
        # 
        order_2d = [1,4,9,16,25,36]
        
        polynomials_2d = {1: lambda x,y: np.ones(x.shape),
                          4: lambda x,y: x*y + x + y,
                          9: lambda x,y: (x**5 - 2*x**2 + 2)*(y**4- 2*y),
                          16: lambda x,y: (x**7 + 2*x**6 + 3*x)*(y**5 + 2*y**2) ,
                          25: lambda x,y: (x**9 + 3*x**4 + 1)*(y**9 + 3*y**4 + 1), 
                          36: lambda x,y: (x**11 - 3*x**8 + 1)*(y**11 + 2*y)
                          }
        
        integrals_2d = {1: 1,
                        4: 5/4, 
                        9: -6/5,
                        16: 535/336, 
                        25: 289/100,
                        36: 13/16}
        
        # Combine information 
        order = {1: order_1d, 2: order_2d}
        polynomials = {1: polynomials_1d, 2: polynomials_2d}
        integrals = {1: integrals_1d, 2: integrals_2d}
        
        # 
        # Iterate over dimensions
        #
        for dim in [1,2]:
            element = QuadFE(dim, 'Q1')
            #
            # Iterate over orders
            # 
            for n in order[dim]:

                # Define Gauss Rule
                rule = GaussRule(n, element)
                x = rule.nodes()
                w = rule.weights()
                
                # Compute approximate integral 
                f = polynomials[dim][n]
                if dim==1:                
                    Ia = np.dot(w, f(x))
                elif dim==2:
                    Ia = np.dot(w, f(x[:,0],x[:,1]))
                    
                # Compute exact integral    
                Ie = integrals[dim][n]
                
                # Compare    
                self.assertAlmostEqual(Ia, Ie, 10)
            
    
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
        rule = GaussRule(2, shape='interval')
        w = rule.weights()
        x_ref = rule.nodes()
        
        # function f to be integrated over edge e
        f = lambda x,y: x**2*y
        e = HalfEdge(Vertex((0,0)),Vertex((1,1)))
        
        # Map rule to physical entity
        x_phys, mg = e.reference_map(x_ref, jac_r2p=True)
        jacobian =  mg['jac_r2p']
        
        x_phys = convert_to_array(x_phys, dim=2)
        fvec = f(x_phys[:,0],x_phys[:,1])
        
        jac = np.linalg.norm(jacobian[0])
        self.assertAlmostEqual(np.dot(fvec,w)*jac,np.sqrt(2)/4,places=10,\
                               msg='Failed to integrate x^2y.')
        self.assertAlmostEqual(np.sum(w)*jac, np.sqrt(2), places=10,\
                               msg='Failed to integrate 1.')
        
if __name__ == '__main__':
    unittest.main()