from mesh import QuadMesh
from plot import Plot
import numpy as np
import unittest

class TestQuadMesh(unittest.TestCase):
    """
    Test QuadMesh class
    """
    def test_constructor(self):
        #
        # Simple mesh 
        # 
        mesh = QuadMesh(resolution=(2,2))
        self.assertTrue(mesh.is_quadmesh())
        
        
    def test_bin_points(self):
        # Construct mesh
        mesh = QuadMesh(resolution=(2,2))
        mesh.record(1)
        mesh.cells.get_child(0).mark('refine')
        mesh.cells.refine(refinement_flag='refine')
        
        #
        # Bin random points using the coarsest mesh
        #
        x = np.random.rand(5,2)
        bins = mesh.bin_points(x,subforest_flag=1)
        for cell, dummy in bins:
            self.assertEqual(cell.get_depth(),0)
        
        #
        # x
        # 
        x = np.array([[0.125,0.125]])
        bins = mesh.bin_points(x, subforest_flag=1)
        self.assertEqual(len(bins),1)
        for cell, dummy in bins:
            # Cell should be on level 0
            self.assertEqual(cell.get_depth(),0)
            
        # Bin over all cells
        bins = mesh.bin_points(x)
        self.assertEqual(len(bins),1)
        for cell, dummy in bins:
            # Cell should now be on level  1
            self.assertEqual(cell.get_depth(),1)
        
        # 
        # Out of bounds x
        # 
        x = np.array([[-1,-1]])
        self.assertRaises(Exception, mesh.bin_points, *(x,))
        
            
    
    def test_is_balanced(self):
        
        mesh_1 = QuadMesh(resolution=(2,2))
        mesh_2 = QuadMesh(resolution=(2,2), periodic={1})
        mesh_3 = QuadMesh(resolution=(2,2), periodic={0,1})
        
        count = 0
        for mesh in [mesh_1, mesh_2, mesh_3]:
            # Initial Meshes should be balanced
            self.assertTrue(mesh.is_balanced())
            
            # Refine mesh and label it
            mesh.cells.get_child(0).mark(1)
            mesh.cells.refine(refinement_flag=1, new_label='b1')
            
            # Check if refined mesh is balanced
            mesh.is_balanced(subforest_flag='b1')
            
            # Now refine again - unbalanced
            mesh.cells.get_child(0).get_child(0).mark(1)
            mesh.cells.refine(refinement_flag=1, subforest_flag='b1', new_label='ub1')
            
            # New mesh should not be balanced
            if count in [0]:
                self.assertTrue(mesh.is_balanced())
                self.assertTrue(mesh.is_balanced(subforest_flag='ub1'))
            else:
                self.assertFalse(mesh.is_balanced(subforest_flag='ub1'))
                self.assertFalse(mesh.is_balanced())
            
            # Old mesh should still be balanced
            self.assertTrue(mesh.is_balanced(subforest_flag='b1'))
            
            count += 1
    
    
    def test_get_boundary_segments(self):
        """
        Test 
        """
        #
        # Define Mesh
        # 
        mesh = QuadMesh(resolution=(2,2))
        mesh.record('mesh_1')
        mesh.cells.get_child(2).mark('1')
        mesh.cells.refine(refinement_flag='1') 
        
        #
        # 
        # 
        for segment in mesh.get_boundary_segments(subforest_flag=None):
            for he in segment:
                pass
            
                #print(he.base().coordinates(), he.head().coordinates())
                
    
    def test_mark_region(self):
        """
        This is a method in Mesh2D, but 
        """
        #
        # Define Mesh
        # 
        mesh = QuadMesh(resolution=(2,2))
        mesh.cells.get_child(2).mark('1')
        mesh.cells.refine(refinement_flag='1')
        
        #
        # Mark left boundary vertices 
        #  
        f_left = lambda x,dummy: np.abs(x)<1e-9
        mesh.mark_region('left', f_left, on_boundary=True)
        
        #
        # Check that left boundary vertices are the only  
        # 
        count = 0
        for segment in mesh.get_boundary_segments():
            for he in segment:
                # Half-edge should not be marked 
                self.assertFalse(he.is_marked('left'))
                
                # Cell should not be marked
                self.assertFalse(he.cell().is_marked('left'))
                
                for v in he.get_vertices():
                    if f_left(*v.coordinates()):
                        #
                        # Left boundary vertices should be marked
                        # 
                        self.assertTrue(v.is_marked('left'))
                        count += 1
                    else:
                        #
                        # No other boundary vertices should be marked
                        # 
                        self.assertFalse(v.is_marked('left'))
        self.assertEqual(count, 8)
        
        
    def test_get_region(self):
        #
        # Define Mesh
        # 
        mesh = QuadMesh(resolution=(2,2))
        mesh.cells.get_child(2).mark('1')
        mesh.cells.refine(refinement_flag='1')
        
        #
        # Mark left boundary vertices 
        #  
        f_left = lambda x,dummy: np.abs(x)<1e-9
        mesh.mark_region('left', f_left, on_boundary=True)
        
        
        
        
        for v,cell in mesh.get_region('left', entity_type='vertex', 
                                      return_cells=True, on_boundary=True):
            self.assertTrue(v.is_marked('left'))
            #print(v.coordinates())  
            
            
                                  
    def test_balance(self):
        mesh_1 = QuadMesh(resolution=(2,2))
        mesh_2 = QuadMesh(resolution=(2,2), periodic={1})
        mesh_3 = QuadMesh(resolution=(2,2), periodic={0,1})
        
        plot = Plot()
        for mesh in [mesh_1, mesh_2, mesh_3]:
            
            # Refine mesh and label it
            mesh.cells.get_child(0).mark(1)
            mesh.cells.refine(refinement_flag=1, new_label='b1')
            
            # Check if refined mesh is balanced
            mesh.is_balanced(subforest_flag='b1')
            
            # Now refine again - unbalanced
            mesh.cells.get_child(0).get_child(0).mark(1)
            mesh.cells.refine(refinement_flag=1, subforest_flag='b1', new_label='ub1')
            
            #plot.mesh(mesh, mesh_flag='ub1')        
            mesh.balance(subforest_flag='ub1')
            #plot.mesh(mesh, mesh_flag='ub1')
        
        #mesh_2.cells.get_child(2).split(flag='ub1')  
            
    def test_remove_supports(self):
        pass