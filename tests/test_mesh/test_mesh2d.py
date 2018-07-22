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
        mesh = Mesh2D(resolution=(1,1), periodic={0})
        self.assertTrue(mesh.is_periodic())
        self.assertTrue(mesh.is_periodic({0}))
        self.assertFalse(mesh.is_periodic({0,1}))
        
        #
        # Periodic in both directions
        # 
        mesh = Mesh2D(resolution=(1,1), periodic={0,1})
        self.assertTrue(mesh.is_periodic())
        self.assertTrue(mesh.is_periodic({0}))
        self.assertTrue(mesh.is_periodic({0,1}))
        
        #
        # From Gmsh 
        # 
        mesh_folder = '/home/hans-werner/git/quadmesh/tests/test_mesh/'
        mesh = Mesh2D(file_path=mesh_folder+'quarter_circle_triangle.msh')
        self.assertFalse(mesh.is_periodic())
        self.assertFalse(mesh.is_quadmesh())
        
        #
        # QuadMesh
        # 
        mesh = Mesh2D(file_path=mesh_folder+'quarter_circle_quad.msh')
        self.assertTrue(mesh.is_quadmesh())
        self.assertFalse(mesh.is_rectangular())
    
    
    def test_half_edge_has_cell(self):
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
                    for v_nbr in v.get_periodic_pair(nbr):
                        v1 = v_nbr.get_periodic_pair(he.cell())
                        self.assertEqual(v,v1[0])
        
        #
        # Periodic in x and y directions
        # 
        mesh = Mesh2D(resolution=(2,2), periodic={0,1})
        c00 = mesh.cells.get_child(0)
        v00 = c00.get_vertex(0)
        c10 = mesh.cells.get_child(1)
        v10 = c10.get_vertex(1)
        c01 = mesh.cells.get_child(2)
        v01 = c01.get_vertex(3)
        c11 = mesh.cells.get_child(3)
        v11 = c11.get_vertex(2)
        
        # Check v00 has 4 periodic pairs
        self.assertEqual(len(v00.get_periodic_pair()),4)
        
        # Check periodic paired vertices within each subcell
        self.assertEqual(v00.get_periodic_pair(c00)[0], v00)    
        self.assertEqual(v00.get_periodic_pair(c10)[0], v10)
        self.assertEqual(v00.get_periodic_pair(c01)[0], v01)
        self.assertEqual(v00.get_periodic_pair(c11)[0], v11)
        
        
        
    def test_locate_point(self):
        mesh_folder = '/home/hans-werner/git/quadmesh/tests/test_mesh/'
        mesh = Mesh2D(file_path=mesh_folder+'quarter_circle_triangle.msh')
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
        mesh_folder = '/home/hans-werner/git/quadmesh/tests/test_mesh/'
        #
        # Define Meshes
        # 
        mesh_1 = Mesh2D(resolution=(2,2))
        mesh_2 = Mesh2D(resolution=(2,2), periodic={0})
        mesh_3 = Mesh2D(resolution=(2,2), periodic={1})
        mesh_4 = Mesh2D(resolution=(2,2), periodic={0,1})
        mesh_5 = Mesh2D(file_path=mesh_folder+'quarter_circle_triangle.msh')
        mesh_6 = Mesh2D(file_path=mesh_folder+'quarter_circle_quad.msh')

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
                    
    
    def test_mark_boundary(self):
        """
        Test boundary vertex and -half-edge marker
        """
        mesh = Mesh2D(resolution=(2,2))
        flag = '1'
        tol = 1e-9
        # Bottom half-edges
        f = lambda he: np.alltrue([np.abs(he.base().coordinates()[0])< tol, \
                                   np.abs(he.head().coordinates()[0])<tol])
        
        mesh.mark_boundary_edges(flag, f)
        
        for segment in mesh.get_boundary_segments():
            for he in segment:
                if f(he):
                    self.assertTrue(he.is_marked(flag))
                else:
                    self.assertFalse(he.is_marked(flag))
                    
        # Top vertices
        f = lambda x,y: np.abs(y-1)<tol
        flag = '2'
        mesh.mark_boundary_vertices(flag, f)
        for v in mesh.get_boundary_vertices():
            if f(*v.coordinates()):
                self.assertTrue(v.is_marked(flag))
            else:
                self.assertFalse(v.is_marked(flag))
                
        
        