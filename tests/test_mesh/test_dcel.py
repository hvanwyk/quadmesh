
#
# Imports 
# 
import unittest
from mesh import DCEL
import numpy as np

class TestDCEL(unittest.TestCase):
    """
    Test DCEL class
    """
    def test_regular_grid(self):
        #
        # 1D regular grid
        # 
        grid = DCEL(box=(0,1), resolution=(2,))
        
        self.assertEqual(grid.get_neighbor(0,1), 1, \
                         'Right Neighbor of interval 0 is interval 1')
        self.assertIsNone(grid.get_neighbor(0,0),\
                        'Left neighbor should be None')
        self.assertTrue(all([tpe=='interval' for tpe in grid.faces['type']]), \
                         'DCEL face type should be interval.')
        self.assertEqual(grid.half_edges['n'],2, \
                         'There should be 2 intervals')
        self.assertEqual(grid.points['n'], 3, \
                         'There should be 3 points.')
        
        grid = DCEL(box=(0,1), resolution=(2,), periodic={0})
        self.assertEqual(grid.half_edges['next'][-1],0)

        #
        # 2D cartesian grid
        #
        grid = DCEL(box=(0,1,0,1), resolution=(2,2))
        #
        # Check vertices of first face
        #
        f0 = grid.faces['connectivity'][0]
        xy_loc = [(0,0),(0.5,0),(0.5,0.5),(0,0.5)]
        for i in range(4):
            xy = grid.points['coordinates'][f0[i]]
            self.assertAlmostEqual(xy, xy_loc[i], 8, \
                                   'First cell vertices incorrect')
        
        #
        # Periodic Grid
        # 
        grid = DCEL(box=(0,1,0,1), resolution=(2,2), periodic={0})
       
        
        # Check that grid is periodic in the x-direction 
        self.assertEqual(grid.half_edges['twin'][5], 3)
        
        # Check that grid is not periodic in y-direction
        self.assertEqual(grid.half_edges['twin'][0], -1)
       
        #
        # Grid periodic in both directions
        # 
        grid = DCEL(box=(0,1,0,1), resolution=(2,2), periodic={0,1})
        
        # Check periodicity in x-direction
        self.assertEqual(grid.half_edges['twin'][5], 3)
        
        # Check periodicity in y-direction
        self.assertEqual(grid.half_edges['twin'][14], 4)
        '''
        #
        # Check boundary edges
        # 
        boundary_edges = grid.get_boundary_half_edges()
        for i_he in boundary_edges:
            i_fc = grid.half_edges['face'][i_he]
            self.assertIsNone(grid.get_neighbor(i_fc, i_he),\
                              'Neighbor of boundary edge should be None')
        ''' 
    def test_grid_from_connectivity(self):
        # =====================================================================
        # 1D Mesh
        # ===================================================================== 
        n_points = 11
        x = np.linspace(0,1,n_points)
        connectivity = [[i, i+1] for i in range(n_points-1)]
        dcel = DCEL(x=x, dim=1)
        n_hes = dcel.half_edges['n']
        #
        # Check connectivity
        # 
        for i in range(n_hes):
            self.assertEqual(list(dcel.half_edges['connectivity'][i]), 
                             connectivity[i])
        
        '''
        #
        # Check twins 
        #
        for i in range(n_hes):
            i_twin = dcel.half_edges['twin'][i]
            self.assertEqual(list(dcel.half_edges['connectivity'][i_twin]),
                             list(dcel.half_edges['connectivity'][i])[::-1])
        '''
            
        # =====================================================================
        # Triangles
        # =====================================================================
        #
        # Construct DCEL
        #
        x = [(0,1), (0,0), (0.5,1), (1,0), (2,1), (1.5,2)]
        e_conn = [[0,1,2], [1,3,2], [3,4,2], [2,4,5]]
        dcel = DCEL(x=x, connectivity=e_conn)
        #
        # Check connectivity
        #
        for i in range(dcel.faces['n']):
            self.assertEqual(dcel.faces['connectivity'][i], e_conn[i])
        #    
        # Check a few half-edge connectivities
        #
        self.assertEqual(list(dcel.half_edges['connectivity'][0,:]), [0,1])    
        self.assertEqual(list(dcel.half_edges['connectivity'][7,:]), [4,2])    
        self.assertEqual(list(dcel.half_edges['connectivity'][11,:]), [5,2])    
        self.assertEqual(list(dcel.half_edges['connectivity'][5,:]), [2,1])    
        #
        # Check incident faces
        # 
        self.assertEqual(dcel.half_edges['face'][0], 0)    
        self.assertEqual(dcel.half_edges['face'][7], 2)    
        self.assertEqual(dcel.half_edges['face'][11], 3)    
        self.assertEqual(dcel.half_edges['face'][5], 1)
        #
        # Check twins 
        # 
        self.assertEqual(dcel.half_edges['twin'][0], -1)    
        self.assertEqual(dcel.half_edges['twin'][7], 9)    
        self.assertEqual(dcel.half_edges['twin'][11], -1)    
        self.assertEqual(dcel.half_edges['twin'][5], 1)
        #
        # Check boundary half_edges have no twins
        # 
        bnd_hes = dcel.get_boundary_half_edges()[0]
        for i_he in bnd_hes:
            self.assertEqual(dcel.half_edges['twin'][i_he], -1)
        
        # =====================================================================
        # Quadrilaterals
        # =====================================================================
        x = [(0,0),(1,0),(0.5,1),(0,1),(2,1.5),(0.5,2),(0,3)]
        e_conn = [[0,1,2,3], [1,4,5,2],[3,2,5,6]]
        dcel = DCEL(x=x, connectivity=e_conn)
        #
        # Check connectivity
        #
        for i in range(dcel.faces['n']):
            self.assertEqual(dcel.faces['connectivity'][i], e_conn[i])
        #    
        # Check a few half-edge connectivities
        #
        self.assertEqual(list(dcel.half_edges['connectivity'][0,:]), [0,1])    
        self.assertEqual(list(dcel.half_edges['connectivity'][7,:]), [2,1])    
        self.assertEqual(list(dcel.half_edges['connectivity'][11,:]), [6,3])    
        self.assertEqual(list(dcel.half_edges['connectivity'][5,:]), [4,5])    
        #
        # Check incident faces
        # 
        self.assertEqual(dcel.half_edges['face'][0], 0)    
        self.assertEqual(dcel.half_edges['face'][7], 1)    
        self.assertEqual(dcel.half_edges['face'][11], 2)    
        self.assertEqual(dcel.half_edges['face'][5], 1)
        #
        # Check twins 
        # 
        self.assertEqual(dcel.half_edges['twin'][0], -1)    
        self.assertEqual(dcel.half_edges['twin'][7], 1)    
        self.assertEqual(dcel.half_edges['twin'][1], 7)
        self.assertEqual(dcel.half_edges['twin'][11], -1)    
        self.assertEqual(dcel.half_edges['twin'][5], -1)
        self.assertEqual(dcel.half_edges['twin'][9], 6)
        #
        # Check boundary half_edges have no twins
        # 
        bnd_hes = dcel.get_boundary_half_edges()[0]
        self.assertEqual(len(bnd_hes),6)
        for i_he in bnd_hes:
            self.assertEqual(dcel.half_edges['twin'][i_he], -1)
        
        # =====================================================================
        # Mixed
        # =====================================================================
        x = [(0,0),(1,0),(0.5,1),(0,1),(2,1.5),(0.5,2)]
        e_conn = [[0,1,2,3], [1,4,5,2],[3,2,5]]
        dcel = DCEL(x=x, connectivity=e_conn)
        #
        # Check connectivity
        #
        for i in range(dcel.faces['n']):
            self.assertEqual(dcel.faces['connectivity'][i], e_conn[i])
        #    
        # Check a few half-edge connectivities
        #
        self.assertEqual(list(dcel.half_edges['connectivity'][0,:]), [0,1])    
        self.assertEqual(list(dcel.half_edges['connectivity'][7,:]), [2,1])    
        self.assertEqual(list(dcel.half_edges['connectivity'][10,:]), [5,3])    
        self.assertEqual(list(dcel.half_edges['connectivity'][5,:]), [4,5])    
        #
        # Check incident faces
        # 
        self.assertEqual(dcel.half_edges['face'][0], 0)    
        self.assertEqual(dcel.half_edges['face'][7], 1)    
        self.assertEqual(dcel.half_edges['face'][10], 2)    
        self.assertEqual(dcel.half_edges['face'][5], 1)
        #
        # Check next/previous for one cell
        # 
        self.assertEqual(dcel.faces['half_edge'][1], 4)
        hes = [4,5,6,7]
        he = 4
        for i in range(4):
            self.assertEqual(dcel.half_edges['next'][he], hes[(i+1)%4])
            he = hes[(i+1)%4]
        #
        # Check twins 
        # 
        self.assertEqual(dcel.half_edges['twin'][0], -1)    
        self.assertEqual(dcel.half_edges['twin'][7], 1)    
        self.assertEqual(dcel.half_edges['twin'][1], 7)
        self.assertEqual(dcel.half_edges['twin'][10], -1)    
        self.assertEqual(dcel.half_edges['twin'][5], -1)
        self.assertEqual(dcel.half_edges['twin'][9], 6)
        #
        # Check boundary half_edges have no twins
        # 
        bnd_hes = dcel.get_boundary_half_edges()[0]
        self.assertEqual(len(bnd_hes),5)
        for i_he in bnd_hes:
            self.assertEqual(dcel.half_edges['twin'][i_he], -1)
        
        
    def test_grid_from_gmsh(self):
        #
        # 2D Triangular grid
        file_path = '/home/hans-werner/Dropbox/work/code/matlab/'+\
                    'finite_elements/mesh/gmsh_matlab/examples/'+\
                    'quarter_disk.msh'
        grid = DCEL(file_path=file_path)
        boundary_edges = grid.get_boundary_half_edges()[0]
        for i_e in boundary_edges:
            self.assertEqual(grid.half_edges['twin'][i_e],-1,\
                             'Neighbor of boundary half-edge should be -1.')

        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testNode']
    unittest.main() 