from mesh import Vertex
import unittest

class TestVertex(unittest.TestCase):
    """
    Test Vertex Class
    """
    def test_constructor(self):
        self.assertRaises(Exception, Vertex, *(1,1))
        self.assertRaises(Exception, Vertex, *((1,1,1)))
        
        
    
    def test_coordinates(self):
        self.assertEqual(Vertex(0).coordinates(), (0,))
        self.assertEqual(Vertex((1,1)).coordinates(), (1,1))
    
    
    def test_mark(self):
        v = Vertex(0)
        v.mark()
        self.assertTrue(v.is_marked())
        
        v = Vertex(0)
        v.mark('sflkj')
        self.assertTrue(v.is_marked())
        
    
    def test_unmark(self):
        v = Vertex(0)
        v.mark(1)
        v.mark('N')
        self.assertTrue(v.is_marked(1))
        self.assertTrue(v.is_marked('N'))
        
        v.unmark(1)
        self.assertFalse(v.is_marked(1))
        self.assertTrue(v.is_marked('N'))
        
        v.unmark()
        self.assertFalse(v.is_marked())
                