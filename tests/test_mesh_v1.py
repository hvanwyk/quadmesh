from mesh import Mesh, DCEL
import numpy as np
import unittest

class TestMesh(unittest.TestCase):
    """
    Tests Mesh class
    """
    def test_constructor(self):
        # =====================================================================
        # 1D Mesh
        # ===================================================================== 
        n_points = 11
        x = np.linspace(0,1,n_points)
        connectivity = [[i, i+1] for i in range(n_points-1)]
        mesh = Mesh(x=x, dim=1) 
    
            
        # =====================================================================
        # Triangles
        # =====================================================================
        #
        # Construct Mesh
        #
        x = [(0,1), (0,0), (0.5,1), (1,0), (2,1), (1.5,2)]
        e_conn = [[0,1,2], [1,3,2], [3,4,2], [2,4,5]]
        mesh = Mesh(x=x, connectivity=e_conn)
        
        # =====================================================================
        # Quadrilaterals
        # =====================================================================
        x = [(0,0),(1,0),(0.5,1),(0,1),(2,1.5),(0.5,2),(0,3)]
        e_conn = [[0,1,2,3], [1,4,5,2],[3,2,5,6]]
        mesh = Mesh(x=x, connectivity=e_conn)
        
        # =====================================================================
        # Mixed
        # =====================================================================
        x = [(0,0),(1,0),(0.5,1),(0,1),(2,1.5),(0.5,2)]
        e_conn = [[0,1,2,3], [1,4,5,2],[3,2,5]]
        mesh = Mesh(x=x, connectivity=e_conn)

    
    def test_is_quadmesh(self):
        pass
    
    
    def test_refine(self):
        pass
    
    
    def test_coarsen(self):
        pass
    
    
    def test_record(self):
        pass
    
    
    def test_is_balanced(self):
        pass
    
    
    def test_balance(self):
        pass
    
    
    def test_get_boundary_edges(self):
        pass
    
    
    def test_get_boundary_nodes(self):
        pass
        