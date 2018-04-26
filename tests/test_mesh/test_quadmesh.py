from mesh import QuadMesh
from plot import Plot
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
        
        
    def test_locate_point(self):
        pass
    
    
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
            
            plot.mesh(mesh, mesh_flag='ub1')        
            mesh.balance(subforest_flag='ub1')
            plot.mesh(mesh, mesh_flag='ub1')
        
        #mesh_2.cells.get_child(2).split(flag='ub1')  
            
    def test_remove_supports(self):
        pass