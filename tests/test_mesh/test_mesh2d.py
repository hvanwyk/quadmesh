from mesh import Mesh2D
from plot import Plot
import unittest


class TestMesh2D(unittest.TestCase):
    """
    Test Class Mesh2D
    """
    def test_constructor(self):
        #
        # Rectangular Mesh
        #
        mesh = Mesh2D(resolution=(3,3))
        self.assertTrue(mesh.is_rectangular())
        self.assertFalse(mesh.is_periodic())
        self.assertTrue(mesh.is_quadmesh())
        
        #
        # Periodic in x-direction
        # 
        mesh = Mesh2D(resolution=(2,2), periodic={0})
        self.assertTrue(mesh.is_periodic())
        self.assertTrue(mesh.is_periodic({0}))
        self.assertFalse(mesh.is_periodic({0,1}))
        
        #
        # Periodic in both directions
        # 
        mesh = Mesh2D(resolution=(2,2), periodic={0,1})
        self.assertTrue(mesh.is_periodic())
        self.assertTrue(mesh.is_periodic({0}))
        self.assertTrue(mesh.is_periodic({0,1}))
        
        #
        # From Gmsh 
        # 
        mesh = Mesh2D(file_path='quarter_circle_triangle.msh')
        self.assertFalse(mesh.is_periodic())
        self.assertFalse(mesh.is_quadmesh())
        
        #
        # QuadMesh
        # 
        mesh = Mesh2D(file_path='quarter_circle_quad.msh')
        self.assertTrue(mesh.is_quadmesh())
        self.assertFalse(mesh.is_rectangular())
        
    
    def test_locate_point(self):
        pass
    
    
    def test_get_boundary_segments(self):
        mesh = Mesh2D(resolution=(2,2))
        for segment in mesh.get_boundary_segments():
            for he in segment:
                print(he.base().coordinates(), he.head().coordinates())
                
    
    def test_get_boundary_vertices(self):
        pass
