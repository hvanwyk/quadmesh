from mesh import Mesh2D
import unittest
import numpy as np

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
    
    
    def test(self):
        #
        # Check that every half-edge has a cell
        # 
        mesh = Mesh2D(resolution=(2,2), periodic={0})
        for cell in mesh.cells.get_children():
            for half_edge in cell.get_half_edges():
                self.assertIsNotNone(half_edge.cell())
        
        for half_edge in mesh.half_edges.get_children():
            self.assertIsNotNone(half_edge.cell())
            
                    
    def test_periodic_pairing(self):
        #
        # Periodic in x-direction
        # 
        mesh = Mesh2D(resolution=(2,2), periodic={0})
        for he in mesh.half_edges.get_children():
            if he.is_periodic():
                nbr = he.twin().cell()
                for v in [he.base(), he.head()]:
                    self.assertTrue(v.is_periodic())
                    v_nbr = v.get_periodic_pair(nbr)
                    v1 = v_nbr.get_periodic_pair(he.cell())
                    self.assertEqual(v,v1)
                
    
    def test_locate_point(self):
        mesh = Mesh2D(file_path='quarter_circle_triangle.msh')
        point = (0.25,0.25)
        cell = mesh.locate_point(point)
        self.assertTrue(cell.contains_points(point))
     
        #mesh.cells[0].mark(1)
        #self.assertIsNone(mesh.locate_point(point, flag=1))
        
    
    def test_get_boundary_segments(self):
        """
        In each case, get the boundary segments and check that
        (i)  The twins of all half_edges are None
        (ii) The halfedges are in order
        """
        #
        # Define Meshes
        # 
        mesh_1 = Mesh2D(resolution=(2,2))
        mesh_2 = Mesh2D(resolution=(2,2), periodic={0})
        mesh_3 = Mesh2D(resolution=(2,2), periodic={1})
        mesh_4 = Mesh2D(resolution=(2,2), periodic={0,1})
        mesh_5 = Mesh2D(file_path='quarter_circle_triangle.msh')
        mesh_6 = Mesh2D(file_path='quarter_circle_quad.msh')

        meshes = [mesh_1, mesh_2, mesh_3, mesh_4, mesh_5, mesh_6]
        for mesh in meshes:
            # Check boundary
            bnd_segments = mesh.get_boundary_segments()
            for segment in bnd_segments:
                he_current = segment[0]
                for i in np.arange(1,len(segment)):
                    he_next = segment[i]
                    self.assertEqual(he_current.head(), he_next.base())
                    self.assertIsNone(he_current.twin())
                    self.assertIsNone(he_next.twin())
                    he_current = he_next