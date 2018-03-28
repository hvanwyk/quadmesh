from mesh import Cell, QuadCell, HalfEdge, Vertex
from mesh import convert_to_array
from fem import GaussRule
import numpy as np
import unittest

class TestCell(unittest.TestCase):
    """
    Test Cell object(s).
    """
    def test_constructor(self):
        """
        Constructor
        """
        #
        # Triangle
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((0,1))
        
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h31 = HalfEdge(v3, v1)
        
        # Vertices not in order
        bad_list_1 = [h12, h31, h23]
        self.assertRaises(Exception, Cell, *[bad_list_1])
        
        # Not a closed loop
        bad_list_2 = [h12, h23]
        self.assertRaises(Exception, Cell, *[bad_list_2])
    
        triangle_half_edges = [h12, h23, h31]
        cell = Cell(triangle_half_edges)
        self.assertAlmostEqual(cell.area(),0.5)
        self.assertEqual(cell.n_vertices(),3)
        self.assertEqual(cell.n_half_edges(),3)
        half_edge = cell.get_half_edge(0)
        for i in range(3):
            self.assertEqual(half_edge.next(), triangle_half_edges[(i+1)%3])
            half_edge = half_edge.next()
        
        #
        # Square 
        # 
        v4 = Vertex((1,1))
        h24 = HalfEdge(v2,v4)
        h43 = HalfEdge(v4,v3)
        square_half_edges = [h12, h24, h43, h31]
        cell = Cell(square_half_edges)
        self.assertAlmostEqual(cell.area(),1)
        self.assertEqual(cell.n_vertices(),4)
        self.assertEqual(cell.n_half_edges(),4)
            
    
    def test_get_half_edge(self):
        #
        # Construct Cell
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((0,1))
        
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h31 = HalfEdge(v3, v1)
        
        # Check whether you get the right he's back
        hes = [h12, h23, h31]
        cell = Cell(hes)
        for i in range(3):
            self.assertEqual(cell.get_half_edge(i), hes[i])
    
    
    def test_get_half_edges(self):
        #
        # Construct Cell
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((0,1))
        
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h31 = HalfEdge(v3, v1)
        
        # Check whether you get the right he's back
        hes = [h12, h23, h31]
        cell = Cell(hes)
        self.assertEqual(cell.get_half_edges(), hes)
        
    
        
    def test_get_vertex(self):
        #
        # Construct Cell
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((0,1))
        
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h31 = HalfEdge(v3, v1)
        
        # Check whether you get the right he's back
        vs = [v1, v2, v3]
        cell = Cell([h12,h23,h31])
        for i in range(3):
            self.assertEqual(cell.get_vertex(i), vs[i])
    
    
    def test_get_vertices(self):
        #
        # Construct Cell
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((0,1))
        
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h31 = HalfEdge(v3, v1)
        
        # Check whether you get the right he's back
        vs = [v1, v2, v3]
        cell = Cell([h12,h23,h31])
        self.assertEqual(cell.get_vertices(), vs)
    
    
    def test_contains_points(self):
        #
        # Triangle
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((0,1))
        
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h31 = HalfEdge(v3, v1)
        
        cell = Cell([h12, h23, h31])
        
        # Vertices
        in_cell = cell.contains_points([v1,v2,v3])
        in_cell_ref = np.ones(3, dtype=np.bool)
        for i in range(3):
            self.assertEqual(in_cell_ref[i], in_cell[i])
        
        # Random points
        points = np.random.rand(100,2)
        in_cell_ref = (points[:,1]<1-points[:,0])
        in_cell = cell.contains_points(points)
        for i in range(100):
            self.assertEqual(in_cell_ref[i], in_cell[i])
        
        #
        # Square 
        # 
        v4 = Vertex((1,1))
        h24 = HalfEdge(v2,v4)
        h43 = HalfEdge(v4,v3)
        square_half_edges = [h12, h24, h43, h31]
        cell = Cell(square_half_edges)
        
        points = [(2,0), (-1,0), (0.5,0.5)]
        in_cell_ref = np.array([0,0,1], dtype=np.bool)
        in_cell = cell.contains_points(points)
        for i in range(3):
            self.assertEqual(in_cell_ref[i], in_cell[i])
        
        #
        # Single points
        # 
        # Vertex 
        point = Vertex((3,3))
        self.assertFalse(cell.contains_points(point))
        # Tuple
        point = (1,0)
        self.assertTrue(cell.contains_points(point)) 
        # Array
        point = np.array([1,0])
        self.assertTrue(cell.contains_points(point))
    
    
    def test_intersects_line_segment(self):
        vertices = [Vertex((0,0)), Vertex((3,1)), 
                    Vertex((2,3)), Vertex((-1,1))]
        
        h_edges = []
        for i in range(4):
            h_edges.append(HalfEdge(vertices[i], vertices[(i+1)%4]))
        cell = Cell(h_edges)
        
        #
        # Line beginning in cell and ending outside
        # 
        line_1 = [(1,1),(3,0)]
        self.assertTrue(cell.intersects_line_segment(line_1),\
                        'Cell should intersect line segment.')
        #
        # Line inside cell
        #
        line_2 = [(1,1),(1.1,1.1)]
        self.assertTrue(cell.intersects_line_segment(line_2),\
                        'Cell contains line segment.')
        #
        # Line outside cell
        # 
        line_3 = [(3,0),(5,6)]
        self.assertFalse(cell.intersects_line_segment(line_3),\
                         'Cell does not intersect line segment.')
    
    def test_incident_half_edge(self):
        #
        # Triangle
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((0,1))
        
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h31 = HalfEdge(v3, v1)
        
        cell = Cell([h12, h23, h31])
        
        hes_forward = [h31, h12, h23]
        hes_reverse = [h12, h23, h31]
        vs = [v1,v2,v3]
        for i in range(3):
            # forward
            self.assertEqual(cell.incident_half_edge(vs[i]),hes_forward[i])
            
            # backward
            self.assertEqual(cell.incident_half_edge(vs[i], reverse=True),\
                             hes_reverse[i])
    
    
    def test_get_neighbors(self):
        #
        # HalfEdge pivot
        # 
        
        #
        # Cell with no neighbors
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,0))
        v3 = Vertex((0,1))
        
        h12 = HalfEdge(v1, v2)
        h23 = HalfEdge(v2, v3)
        h31 = HalfEdge(v3, v1)
        #
        # Make triangle
        # 
        cell = Cell([h12, h23, h31])
        # No neighbors
        self.assertIsNone(cell.get_neighbors(h12))
        self.assertEqual(cell.get_neighbors(v1),[])
        
        # Add a new neighboring triangle
        v4 = Vertex((1,1))
        h24 = HalfEdge(v2, v4)
        h43 = HalfEdge(v4 ,v3)
        h32 = h23.make_twin()
        
        ncell_1 = Cell([h24, h43, h32])
        
        # Make sure they are neighbors wrt halfedge
        self.assertEqual(cell.get_neighbors(h23),ncell_1)
        
        # Neighbors wrt vertices
        self.assertEqual(cell.get_neighbors(v2),[ncell_1])
        self.assertEqual(cell.get_neighbors(v3),[ncell_1])
        
        #
        # Add a third neighboring triangle
        #
        v5 = Vertex((1,2))
        h34 = h43.make_twin()
        h45 = HalfEdge(v4, v5)
        h53 = HalfEdge(v5, v3)
        
        ncell_2 = Cell([h34, h45, h53])
        
        # Check if it's a neighbor wrt halfedge
        self.assertEqual(ncell_1.get_neighbors(h43), ncell_2)
        
        # 2 Neighbors wrt v3 
        self.assertEqual(cell.get_neighbors(v3),[ncell_1, ncell_2])
        self.assertEqual(ncell_1.get_neighbors(v3), [ncell_2, cell])
        self.assertEqual(ncell_2.get_neighbors(v3), [cell, ncell_1])
        
        #
        # Split h31 and make an illegal neighbor
        #
        v6 = Vertex((-1,0.5))
         
        h31.split()
        h331 = h31.get_child(0)
        
        h133 = h331.make_twin()
        h36 = HalfEdge(v3, v6)
        h613 = HalfEdge(v6, h133.base())
        
        ncell_3 = Cell([h133, h36, h613])
        
        # No neighbors wrt shared edges
        self.assertIsNone(cell.get_neighbors(h31))
        self.assertIsNone(ncell_3.get_neighbors(h133))
        
        # Neighbors wrt vertices remain as they are.
        self.assertEqual(cell.get_neighbors(v3),[ncell_1, ncell_2])
        self.assertEqual(ncell_1.get_neighbors(v3), [ncell_2, cell])
        self.assertEqual(ncell_2.get_neighbors(v3), [cell, ncell_1])
        self.assertEqual(ncell_3.get_neighbors(v3), [])
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testNode']
    unittest.main()