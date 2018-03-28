from mesh import HalfEdge, Vertex
import numpy as np
import unittest

class TestHalfEdge(unittest.TestCase):
    """
    Test HalfEdge objects
    """
    def test_constructor(self):
        v1 = Vertex((0,0))
        v2 = Vertex((1,1))
        v3 = Vertex((2,2))
        half_edge_1 = HalfEdge(v1, v2)
        half_edge_2 = HalfEdge(v2, v3)
        
        # Should complain about tuple input
        self.assertRaises(Exception, HalfEdge, *[v1,(1,1)]) 
        
        # Should complain about incompatible twin 
        self.assertRaises(Exception, HalfEdge, \
                          *(v2, v3), **{'twin':half_edge_1})
        
        # Should complain about incompatible previous
        self.assertRaises(Exception, HalfEdge, \
                          *(v3, v2), **{'previous':half_edge_1})
        
        # Should complain about incompatible nxt
        self.assertRaises(Exception, HalfEdge, *(v3, v2), \
                          **{'next':half_edge_1})
        
        # Should complain about nxt of incompatible type
        self.assertRaises(Exception, HalfEdge, *(v3, v2), \
                          **{'next':2})
        
        # Should complain about incompatible parent
        self.assertRaises(Exception, HalfEdge, *(v3, v2), \
                          **{'parent':half_edge_2})
        
        
    def test_base(self):
        #
        # Retrieve base
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,1))
        half_edge = HalfEdge(v1,v2)
        self.assertEqual(half_edge.base(), v1, 'Incorrect base.')
        
    
    def test_head(self):
        #
        # Retrieve head
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,1))
        half_edge = HalfEdge(v1,v2)
        self.assertEqual(half_edge.head(), v2, 'Incorrect head.')
    
    
    def test_cell(self):
        pass
    
    
    def test_assign_cell(self):
        pass
    
    
    def test_twin(self):
        v1 = Vertex((0,0))
        v2 = Vertex((1,1))
        half_edge_1 = HalfEdge(v1,v2)
        half_edge_2 = HalfEdge(v2,v1, twin=half_edge_1)
        self.assertEqual(half_edge_2.twin(), half_edge_1,\
                         'Twin incorrectly specified.')
        self.assertIsNone(half_edge_1.twin(), \
                          'half_edge_2 has no twin yet.')
    
    
    def test_assign_twin(self):
        #
        # New half-edge
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((1,1))
        half_edge = HalfEdge(v1, v2)
        #
        # Try to assign incorrect twin
        # 
        false_twin = HalfEdge(v1, v2)
        self.assertRaises(Exception, half_edge.assign_twin, false_twin,\
                          'This twin should not be assignable.')
        #
        # Assign a good twin and check 
        # 
        good_twin = HalfEdge(v2, v1)
        half_edge.assign_twin(good_twin)
        self.assertEqual(good_twin.base(),half_edge.head(), \
                         'Heads and bases do not match.')
        self.assertEqual(good_twin.head(),half_edge.base(), \
                         'Heads and bases do not match.')
        
     
    def test_make_twin(self):
        #
        # New HalfEdge
        #
        v1 = Vertex((0,0))
        v2 = Vertex((2,2))
        h12 = HalfEdge(v1,v2)
        h21 = h12.make_twin()
        self.assertIsNotNone(h12.twin())
        self.assertIsNotNone(h21.twin())
        self.assertEqual(h12.base(),h21.head())
        self.assertEqual(h21.base(),h12.head())
        
        
    def test_next(self):
        #
        # New HalfEdge
        #
        v1 = Vertex((0,0))
        v2 = Vertex((2,2))
        v3 = Vertex((3,4))
        next_half_edge = HalfEdge(v1, v2)
        half_edge = HalfEdge(v3, v1, nxt=next_half_edge)
        
        # Check
        self.assertEqual(half_edge.next(),next_half_edge,\
                         'Next HalfEdge incorrect.')
        
    
    def test_assign_next(self):
        #
        # New HalfEdge
        #
        v1 = Vertex((0,0))
        v2 = Vertex((2,2))
        v3 = Vertex((3,4))
        h = HalfEdge(v1, v2)
        #
        # Incompatible next half_edge
        # 
        bad_h_next = HalfEdge(v1, v3)
        self.assertRaises(Exception, h.assign_next, (bad_h_next))
        #
        # Assign good next half-edge
        # 
        good_h_next = HalfEdge(v2, v3)
        h.assign_next(good_h_next)
        
        # Check
        self.assertEqual(h.next(),good_h_next,\
                         'Next HalfEdge incorrect.')
        self.assertEqual(h.head(), h.next().base(), \
                         'Bases and heads should align.')
    
    
    def test_previous(self):
        #
        # New HalfEdge
        #
        v1 = Vertex((0,0))
        v2 = Vertex((2,2))
        v3 = Vertex((3,4))
        previous_half_edge = HalfEdge(v1, v2)
        half_edge = HalfEdge(v2, v3, previous=previous_half_edge)
        
        # Check
        self.assertEqual(half_edge.previous(),previous_half_edge,\
                         'Previous HalfEdge incorrect.')
    
    
    def test_assign_previous(self):
        #
        # New HalfEdge
        #
        v1 = Vertex((0,0))
        v2 = Vertex((2,2))
        v3 = Vertex((3,4))
        h = HalfEdge(v1, v2)
        #
        # Incompatible previous half_edge
        # 
        bad_h_prev = HalfEdge(v1, v3)
        self.assertRaises(Exception, h.assign_previous, (bad_h_prev))
        #
        # Assign good next half-edge
        # 
        good_h_previous = HalfEdge(v2, v1)
        h.assign_previous(good_h_previous)
        
        # Check
        self.assertEqual(h.previous(),good_h_previous,\
                         'Next HalfEdge incorrect.')
        self.assertEqual(h.base(), h.previous().head(), \
                         'Bases and heads should align.')
    
    
    def test_has_parent(self):
        #
        # New HalfEdge
        #
        v1 = Vertex((0,0))
        v2 = Vertex((2,2))
        half_edge = HalfEdge(v1, v2)
        
        self.assertFalse(half_edge.has_parent(), \
                         'Half Edge should not have a parent.')
        
        # Split edge
        half_edge.split()
        
        # Children should have parent
        for child in half_edge.get_children():
            self.assertTrue(child.has_parent(), \
                            'HalfEdge children should have a parent.')
    
    
    def test_get_parent(self):
        #
        # New HalfEdge
        #
        v1 = Vertex((0,0))
        v2 = Vertex((2,2))
        half_edge = HalfEdge(v1, v2)
        
        # Split
        half_edge.split()
        
        # Check if half_edge is children's parent
        for child in half_edge.get_children():
            self.assertEqual(half_edge, child.get_parent(),\
                             'HalfEdge should be child"s parent')
    
    
    def test_has_children(self):
        #
        # New HalfEdge
        #
        v1 = Vertex((0,0))
        v2 = Vertex((2,2))
        half_edge = HalfEdge(v1, v2)
        
        # Shouldn't have children
        self.assertFalse(half_edge.has_children(),\
                         'Half Edge should not have children')
        
        # Split edge
        half_edge.split()
        
        # Should have children
        self.assertTrue(half_edge.has_children(),\
                        'HalfEdge should have children.')

    
    def test_split(self):
        #
        # New HalfEdge
        #
        v1 = Vertex((0,0))
        v2 = Vertex((2,2))
        half_edge = HalfEdge(v1, v2)
        #
        # Split it
        # 
        half_edge.split()
        #
        # Check if the children behave
        # 
        child_0, child_1 = half_edge.get_children()
        self.assertEqual(child_0.head(), child_1.base(),\
                         'First child head should equal second child base.')    
        #
        # Define twin half edge
        # 
        twin = HalfEdge(v2, v1)
        half_edge.assign_twin(twin)
        twin.assign_twin(half_edge)
        
        #
        # Split twin
        # 
        twin.split()
        
        #
        # Check wether children are twinned
        # 
        c0, c1 = half_edge.get_children()
        t0, t1 = twin.get_children()
        self.assertEqual(c0.base(),t1.head(),\
                         'Children have incorrect twins.')
        self.assertEqual(c0.head(),t1.base(),\
                         'Children have incorrect twins.')
        self.assertEqual(c1.base(),t0.head(),\
                         'Children have incorrect twins.')
        self.assertEqual(c1.head(),t0.base(),\
                         'Children have incorrect twins.')
    
    
    def test_mark(self):
        #
        # Define a HalfEdge
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((0,1))
        h_edge = HalfEdge(v1,v2)
        #
        # Mark it
        # 
        h_edge.mark(1)
        self.assertTrue(h_edge.is_marked(1),'HalfEdge should be marked.') 
        #
        # Mark when initializing
        # 
        h_edge = HalfEdge(v1, v2, flag=1)
        self.assertTrue(h_edge.is_marked(1),'HalfEdge should be marked.')
        #
        # Split and mark recursively
        # 
        h_edge.split()
        h_edge.mark(flag=1, recursive=True)
        for child in h_edge.get_children():
            self.assertTrue(child.is_marked(1),'HalfEdge should be marked.')
        
    
    def test_unmark(self):
        #
        # Define a HalfEdge
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((0,1))
        h_edge = HalfEdge(v1,v2)
        #
        # Mark it with a specific flag
        # 
        h_edge.mark(1)
        self.assertTrue(h_edge.is_marked(1),'HalfEdge should be marked.')
        #
        # Unmark it 
        # 
        h_edge.unmark(1)
        self.assertFalse(h_edge.is_marked(),'HalfEdge should be marked.')
        self.assertFalse(h_edge.is_marked(1),'HalfEdge should be marked.')
    
    
    def test_is_marked(self):
        #
        # Define a HalfEdge
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((0,1))
        h_edge = HalfEdge(v1,v2)
        #
        # Mark it with a specific flag
        # 
        h_edge.mark(1)
        self.assertTrue(h_edge.is_marked(1),'HalfEdge should be marked.') 
        self.assertTrue(h_edge.is_marked(),'HalfEdge should be marked.')
        self.assertFalse(h_edge.is_marked(2),'HalfEdge should not be marked.')
        #
        # Unmark and mark with bool
        # 
        h_edge.unmark()
        h_edge.mark()
        self.assertTrue(h_edge.is_marked(),'HalfEdge should be marked.')
        self.assertFalse(h_edge.is_marked(2),'HalfEdge should be marked.')
         
        
    def test_unit_normal(self):
        # 
        # Define a HalfEdge
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((0,1))
        h1 = HalfEdge(v1,v2)
        #
        # Make sure computed unit normal is correct
        #
        u  = np.array([1,0])
        self.assertTrue(np.allclose(h1.unit_normal(), u),\
                               'Incorrect unit normal')
        #
        # Reverse direction of unit HalfEdge
        # 
        h2 = HalfEdge(v2,v1)
        #
        # Check unit normal
        # 
        u = np.array([-1,0])
        self.assertTrue(np.allclose(h2.unit_normal(), u),\
                               'Incorrect unit normal')
     
    
    def test_contains_points(self):
        #
        # Define 2D HalfEdge
        #
        v1 = Vertex((0,0))
        v2 = Vertex((1,1))
        h = HalfEdge(v1,v2)
        
        points = np.array([[0,0],[0.5,0.5],[3,2]])
        on_half_edge = h.contains_points(points)
        self.assertTrue(all(on_half_edge==[True, True, False]))
        
        v1 = Vertex(0)
        v2 = Vertex(1)
        h = HalfEdge(v1,v2)
        points = [Vertex(0), Vertex(0.5), Vertex(3)]
        on_half_edge = h.contains_points(points)
        self.assertTrue(all(on_half_edge==[True, True, False]))
        
        
    def test_intersects_line_segment(self):
        # 
        # Define a HalfEdge
        # 
        v1 = Vertex((0,0))
        v2 = Vertex((0,1))
        h_edge = HalfEdge(v1,v2)
        #
        # Line 1 intersects h_edge in the middle
        # 
        line_1 = [(-0.5,0.5),(0.5,0.5)]
        self.assertTrue(h_edge.intersects_line_segment(line_1),\
                        'HalfEdge should intersect line_1.')
        #
        # Line 2 intersects h_edge at the vertex
        # 
        line_2 = [(-1,1),(0,1)]
        self.assertTrue(h_edge.intersects_line_segment(line_2),\
                        'HalfEdge should intersect line_2.')
        #
        # Line 3 is on top of h_edge
        # 
        line_3 = [(0,0),(0,1)]
        self.assertTrue(h_edge.intersects_line_segment(line_3),\
                        'HalfEdge should intersect line_3.')
        #
        # Line 4 does not intersect h_edge
        # 
        line_4 = [(1,2), (3,3)]
        self.assertFalse(h_edge.intersects_line_segment(line_4),\
                        'HalfEdge should intersect line_4.')