from mesh import Vertex, HalfEdge, Interval
from mesh import convert_to_array
import numpy as np
import unittest

class TestInterval(unittest.TestCase):
    """
    Test Interval Class
    """
    def test_constructor(self):
        #
        # Proper Interval
        # 
        v1 = Vertex(0)
        v2 = Vertex(1)
        I = Interval(v1,v2)
        self.assertTrue(isinstance(I,HalfEdge))
        #
        # Interval using 2D Vertices
        # 
        w1 = Vertex((0,0))
        w2 = Vertex((1,1))
        
        self.assertRaises(Exception, Interval, *(w1,w2))
        
    def test_get_vertices(self):
        #
        # Make interval
        #
        v1 = Vertex(0)
        v2 = Vertex(1)
        I = Interval(v1,v2)

        self.assertEqual(I.get_vertices(),[v1,v2])
            
    def test_get_vertex(self):
        #
        # Make interval
        #
        v1 = Vertex(0)
        v2 = Vertex(1)
        I = Interval(v1,v2)
        
        self.assertEqual(I.get_vertex(0),v1)
        self.assertEqual(I.get_vertex(1),v2)
        
    
    def test_assign_next(self):
        #
        # Make interval
        #
        v1 = Vertex(0)
        v2 = Vertex(1)
        I = Interval(v1,v2)
        
        #
        # Left interval
        # 
        v0 = Vertex(-1)
        I0 = Interval(v0,v1)
        
        #
        # Right interval 
        # 
        v3 = Vertex(2)
        I1 = Interval(v2,v3)
        
        I0.assign_next(I)
        self.assertEqual(I0.next(),I)
        self.assertEqual(I.previous(),I0)
        
        I.assign_next(I1)
        self.assertEqual(I.next(),I1)
        self.assertEqual(I1.previous(),I)
         
         
    def test_get_neighbor(self):
        """
        TODO: Test flagged neighbor
        """
        #
        # Make interval
        #
        v1 = Vertex(0)
        v2 = Vertex(1)
        I = Interval(v1,v2)
        
        #
        # Left interval
        # 
        v0 = Vertex(-1)
        I0 = Interval(v0,v1)
        
        #
        # Right interval 
        # 
        v3 = Vertex(2)
        I1 = Interval(v2,v3)
        
        I0.assign_next(I)
        self.assertEqual(I0.next(),I)
        I.assign_next(I1)
        
        self.assertEqual(I.get_neighbor(1),I1)
        self.assertEqual(I.get_neighbor(0),I0)
    
    
    def test_split(self):
        # New (regular) Interval
        I = Interval(Vertex(0), Vertex(1), n_children=3)
        I.split()
        for i in range(2):
            self.assertEqual(I.get_child(i).next(), I.get_child(i+1))
        
        for i in np.arange(1,3):
            self.assertEqual(I.get_child(i).previous(), I.get_child(i-1))
        #
        # Split the middle child
        # 
        middle_child = I.get_child(1)
        # Check that you can't split the middle child into 2 because the 
        # tree is not regular.
        self.assertRaises(Exception, middle_child.split, **{'n_children':2})
         
        # Split using the default number of children
        middle_child.split()
        
        # Check that middle child has 3 children
        self.assertEqual(middle_child.n_children(), 3)
        
        # Check that middle child's right child has no next interval
        self.assertEqual(middle_child.get_child(2).get_neighbor(1), I.get_child(2))
        
        # 
        # Split the right child
        #
        right_child = I.get_child(2)
        right_child.split()
        
        # Check that the right child has the correct number of children
        self.assertEqual(right_child.n_children(), 3)
        
        # Now the middle child's right child has a right neighbor
        self.assertEqual(middle_child.get_child(2).get_neighbor(1),\
                         right_child.get_child(0))
        
        #
        # Irregular Interval
        # 
        I = Interval(Vertex(0),Vertex(1),regular=False, n_children=3)
        I.split(n_children=2)
        for child in I.get_children():
            self.assertEqual(child.n_children(),2)
     
    
    def test_locate_point(self):
        pass 
        
    
    def test_subcell_position(self):
        """
        Test method for determining relative position and width of sub-interval
        """ 
        #
        # Define Interval
        # 
        I = Interval(Vertex(3), Vertex(5))
        
        # Check for exception for Interval not contained in reference
        I_bad = Interval(Vertex(1), Vertex(4))
        self.assertRaises(Exception, I.subcell_position, I_bad)
        
        # Check whether the relative position and width are correct for a known
        # sub-cell.     
        I_good = Interval(Vertex(4), Vertex(4.5))
        ref_pos, ref_width = 0.5, 0.25
        pos, width = I.subcell_position(I_good)
        self.assertEqual(ref_pos, pos)
        self.assertEqual(ref_width, width)
        
            
    def test_reference_map(self):
        # New interval
        I = Interval(Vertex(2),Vertex(5))
        
        # New point
        x = np.array([0,1, 0.5])
        
        # Map point to physical interval
        y, mg = I.reference_map(x, jac_r2p=True, hess_r2p=True)
        
        # Verify
        jac = mg['jac_r2p']
        hess = mg['hess_r2p']
        
        self.assertTrue(type(y) is np.ndarray)
        
        self.assertTrue(np.allclose(y, convert_to_array([(2,),(5,),(3.5,)],dim=1)))
        self.assertTrue(type(jac) is list)
        self.assertEqual(jac, [3,3,3])
        self.assertTrue(type(hess) is list)
        self.assertEqual(hess, [0,0,0])
        
        # Map back to reference domain
        xx, mg = I.reference_map(y, jac_p2r=True, hess_p2r=True, mapsto='reference')
        ijac = mg['jac_p2r']
        ihess = mg['hess_p2r']
        
        # Verify
        for i in range(3):
            self.assertEqual(xx[i],x[i])
            self.assertEqual(ijac[i], 1/jac[i])
            self.assertEqual(ihess[i], hess[i])
        
            