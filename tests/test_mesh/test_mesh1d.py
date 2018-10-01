from mesh import Mesh1D, Interval, Vertex
import unittest
import numpy as np
from plot import Plot
class TestMesh1D(unittest.TestCase):
    """
    Test Mesh1D
    """
    def test_constructor(self):
        """
        Constructor 
        """
        #
        # Standard Mesh
        #
        x = np.linspace(-1,1,5)
        mesh = Mesh1D(x=x)
        #
        # Check that the interval endpoints are correct
        # 
        i = 0
        for interval in mesh.cells.get_leaves():
            a, = interval.base().coordinates()
            b, = interval.head().coordinates()
            self.assertAlmostEqual(a, x[i])
            self.assertAlmostEqual(b, x[i+1])
            i += 1
        #
        # Traverse intervals in 2 different ways
        # 
        leaves = mesh.cells.get_leaves()
        interval = mesh.cells.get_child(0)
        for leaf in leaves:
            self.assertEqual(leaf, interval)
            interval = interval.next()
        #
        # Refine first subinterval
        # 
        mesh.cells.get_child(0).mark(1)
        mesh.cells.refine(refinement_flag=1)
        
        #
        # Make sure that the intervals are still connected properly
        # 
        leaves = mesh.cells.get_leaves(mode='depth-first')
        interval = mesh.cells.get_child(0).get_child(0)
        for leaf in leaves:
            self.assertEqual(leaf, interval)
            interval = interval.get_neighbor(1)
            
        # 
        # Periodic mesh
        #
        mesh = Mesh1D(x=np.linspace(-1,1,5), periodic=True)
        
        # Last interval's next is the first interval
        self.assertEqual(mesh.cells.get_child(-1).get_neighbor(1), mesh.cells.get_child(0))
        
        # Get next interval until you return to the beginning of the loop
        interval = mesh.cells.get_child(0)
        for i in range(5):
            interval = interval.get_neighbor(1)
        self.assertEqual(interval, mesh.cells.get_child(1))
        
        
    def test_bounding_box(self):
        # New mesh
        x = np.linspace(2,4,100)
        mesh = Mesh1D(x=x)
        
        # Get endpoints 
        x0, x1 = mesh.bounding_box()
        
        # Check if they are correct
        self.assertEqual(x0,2)
        self.assertEqual(x1,4)
        
        
    def test_locate_point(self):
        # =====================================================================
        # Straightforward
        # =====================================================================
        # New mesh
        x = np.linspace(0,4,5)
        mesh = Mesh1D(x=x)
        
        # Locate the interval containing pi
        interval = mesh.locate_point(np.pi)
        
        # Check if its the right interval
        self.assertEqual(interval.get_node_position(),3)
        
        # Check if interval endpoints encompass point
        a, = interval.base().coordinates()
        b, = interval.head().coordinates()
        self.assertLess(a, np.pi)
        self.assertLess(np.pi, b)
    
        # =====================================================================
        # Using a submesh flag
        # =====================================================================
        # New mesh
        x = np.linspace(0, 4, 3)
        mesh = Mesh1D(x=x)

        mesh.cells.refine()
        
        mesh.cells.get_child(0).mark(1, recursive=True)
        mesh.cells.get_child(1).mark(1)
        
        # Consider only flagged submesh
        self.assertEqual(mesh.locate_point(3.8, flag=1), 
                         mesh.cells.get_child(-1))
        
        # Consider the entire mesh.
        self.assertEqual(mesh.locate_point(3.9), 
                         mesh.cells.get_child(1).get_child(1))
        
    def test_mark_boundary(self):
        # New mesh
        x = np.linspace(0,1,11)
        mesh = Mesh1D(x=x)
        
        #
        # Mark both sides
        # 
        f = lambda dummy: True
        flag = '1'
        mesh.mark_region(flag, f, entity='vertex', on_boundary=True)
        v0, v1 = mesh.get_boundary_vertices()
        self.assertTrue(v0.is_marked(flag))
        self.assertTrue(v1.is_marked(flag))
        
        # Unmark vertices
        v0.unmark(flag)
        v1.unmark(flag)
        
        #
        # Mark only one side
        # 
        f = lambda x: np.abs(x-1)<1e-9
        mesh.mark_region(flag, f, entity='vertex', on_boundary=True)
        self.assertTrue(v1.is_marked(flag))
        self.assertFalse(v0.is_marked(flag))
        
        