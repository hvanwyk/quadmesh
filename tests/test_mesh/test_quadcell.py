from mesh import Vertex, HalfEdge, QuadCell
from mesh import convert_to_array
from assembler import GaussRule
import numpy as np
import unittest



class TestQuadCell(unittest.TestCase):
    """
    Test QuadCell Class
    """   
    def test_constructor(self):
        # Check the right number of halfedges
        self.assertRaises(Exception, QuadCell, *([1,2,2,2,2]))
    
        # Rectangle
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((1,1))
        v4 = Vertex((0,1))
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h34 = HalfEdge(v3, v4)
        h41 = HalfEdge(v4, v1)
    
        cell = QuadCell([h12, h23, h34, h41])
        self.assertTrue(cell.is_rectangle())

        
    def test_is_rectangle(self):
        #
        # Rectangle
        #
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((1,1))
        v4 = Vertex((0,1))
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h34 = HalfEdge(v3, v4)
        h41 = HalfEdge(v4, v1)
        
        cell = QuadCell([h12, h23, h34, h41])
        
        # Check cell
        self.assertTrue(cell.is_rectangle())
        
        cell.split()
        
        # Check child
        self.assertTrue(cell.get_child(0).is_rectangle())
    
        #
        # Not a rectangle
        # 
        v5 = Vertex((2,2))
        h25 = HalfEdge(v2, v5)
        h54 = HalfEdge(v5, v4)
        
        cell = QuadCell([h12, h25, h54, h41])
        
        # Check cell
        self.assertFalse(cell.is_rectangle())
        
        cell.split()
        
        # Check child
        self.assertFalse(cell.get_child(0).is_rectangle())
        
        
    def test_split(self):
        # Rectangle
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((1,1))
        v4 = Vertex((0,1))
        v5 = Vertex((2,0))
        v6 = Vertex((2,1))
        
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h34 = HalfEdge(v3, v4)
        h41 = HalfEdge(v4, v1)
    
        cell = QuadCell([h12, h23, h34, h41])
        
        cell.split()
        
        self.assertTrue(cell.has_children())
        
        # Check that interior half_edges are twinned
        child_0 = cell.get_child(0)
        child_1 = cell.get_child(1)
        self.assertEqual(child_0.get_half_edge(1).twin(), \
                         child_1.get_half_edge(3))
        
        # Make another cell, check that it is a neighbor, and then split it 
        h25 = HalfEdge(v2, v5)
        h56 = HalfEdge(v5, v6)
        h63 = HalfEdge(v6, v3)
        h32 = h23.make_twin()
        
        cell_1 = QuadCell([h25, h56, h63, h32])
        
        # Check that they are neighbors
        self.assertEqual(cell_1.get_neighbors(h32),cell)
        
        # Child_s doesn't have a neighbor
        self.assertIsNone(child_1.get_neighbors(child_1.get_half_edge(1)))
        
        cell_1.split()
        
        # Now the child has a neighbor
        self.assertEqual(child_1.get_neighbors(child_1.get_half_edge(1)),
                         cell_1.get_child(0))
        
    
    def test_locate_point(self):
        pass
    
        
    def test_reference_map(self):
        v_sw = Vertex((0,0))
        v_se = Vertex((3,1))
        v_ne = Vertex((2,3))
        v_nw = Vertex((-1,1))
        
        h12 = HalfEdge(v_sw, v_se)
        h23 = HalfEdge(v_se, v_ne)
        h34 = HalfEdge(v_ne, v_nw)
        h41 = HalfEdge(v_nw, v_sw)
        cell = QuadCell([h12,h23,h34,h41])
        
        #
        # Map corner vertices of reference cell to physical vertices
        #
        y_refs = np.array([[0,0],[1,0],[1,1],[0,1]])
        x = list(convert_to_array(cell.get_vertices()))
        x_phys = cell.reference_map(list(y_refs))
        self.assertTrue(np.allclose(np.array(x),x_phys),\
                        'Mapped vertices should coincide '+\
                        'with physical cell vertices.')
        
        #
        # Jacobian: Area of cell by integration
        #  
        rule_2d = GaussRule(order=4, shape='quadrilateral')
        r = rule_2d.nodes()
        wg = rule_2d.weights()
        dummy, jac = cell.reference_map(list(r), jacobian=True)
        area = 0
        for i in range(4):
            j = jac[i]
            w = wg[i]
            area += np.abs(np.linalg.det(j))*w
        self.assertAlmostEqual(cell.area(), area, 7,\
                               'Area computed via numerical quadrature '+\
                               'not close to actual area')
        #
        # Try different formats
        # 
        # Array
        x = np.array(x)
        x_ref = cell.reference_map(x, mapsto='reference')
        self.assertTrue(np.allclose(y_refs, np.array(x_ref)),\
                        'Map array to reference: incorrect output.')
        # Single point
        x = x[0,:]
        x_ref = cell.reference_map(x, mapsto='reference')
        self.assertTrue(np.allclose(x, x_ref))
        
        #
        # Map corner vertices to reference points
        #
        x = convert_to_array(cell.get_vertices())
        y = cell.reference_map(x, mapsto='reference')
        self.assertTrue(np.allclose(y, y_refs), \
                        'Corner vertices should map '+\
                        'onto (0,0),(1,0),(1,1),(0,1).')
         
        #
        # Map random points in [0,1]^2 onto cell and back again
        # 
        # Generate random points
        t = np.random.rand(5)
        s = np.random.rand(5)
        x = np.array([s,t]).T
        
        # Map to physical cell
        x_phy = cell.reference_map(x)
        
        # Check whether points are contained in cell
        in_cell = cell.contains_points(x_phy)
        self.assertTrue(all(in_cell), \
                        'All points mapped from [0,1]^2 '+\
                        'should be contained in the cell.')
        
        # Map back to reference cell
        x_ref = cell.reference_map(x_phy, mapsto='reference')
        self.assertTrue(np.allclose(np.array(x_ref), np.array(x)),\
                        'Points mapped to physical cell and back should '+\
                        'be unchanged.')
        

        #
        # Compute the hessian and compare with finite difference approximation 
        #
        h = 1e-8
        x = np.array([[0.5, 0.5],[0.5+h,0.5],[0.5-h,0.5],
                      [0.5,0.5+h],[0.5,0.5-h]])
        
        x_ref, J, H  = cell.reference_map(x, mapsto='reference', 
                                          hessian=True, jacobian=True)
        
        # sxx
        sxx_fd = (J[1][0,0]-J[2][0,0])/(2*h)
        sxx    = H[0][0,0,0]  
        self.assertAlmostEqual(sxx_fd, sxx, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')
        
        
        # syx
        syx_fd = (J[1][0,1]-J[2][0,1])/(2*h)
        sxy    = H[0][0,1,0]
        syx    = H[0][1,0,0]
        self.assertAlmostEqual(sxy, syx, 7, 'Mixed derivatives not equal.')
        self.assertAlmostEqual(syx_fd, sxy, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')
        
        # syy
        syy_fd = (J[3][0,1]-J[4][0,1])/(2*h)
        syy    = H[0][1,1,0]
        self.assertAlmostEqual(syy_fd, syy, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')

        # txx
        txx_fd = (J[1][1,0]-J[2][1,0])/(2*h)
        txx = H[0][0,0,1]
        self.assertAlmostEqual(txx_fd, txx, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')
        
        # txy
        txy_fd = (J[3][1,0]-J[4][1,0])/(2*h)
        txy    = H[0][0,1,1]
        tyx    = H[0][1,0,1]
        self.assertAlmostEqual(txy, tyx, 7, 'Mixed derivatives not equal.')
        self.assertAlmostEqual(txy_fd, txy, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')
        
        # tyy
        tyy_fd = (J[3][1,1]-J[4][1,1])/(2*h)
        tyy    = H[0][1,1,1] 
        self.assertAlmostEqual(tyy_fd, tyy, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')    
       
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testNode']
    unittest.main() 