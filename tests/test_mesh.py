'''
Created on Oct 23, 2016

@author: hans-werner
'''
import unittest
from fem import GaussRule
from mesh import Mesh, Grid
from mesh import BiNode, QuadNode
from mesh import BiCell, QuadCell
from mesh import Edge, Vertex
from mesh import convert_to_array
#from plot import Plot
#import matplotlib.pyplot as plt
import numpy as np
#from collections import deque


class TestMesh(unittest.TestCase):
    """
    Test Mesh Class
    """
    
    def test_convert_to_array(self):
        x = [Vertex((i,)) for i in range(5)]
        y = convert_to_array(x)
        self.assertEqual(y.shape, (5,1), 'Incorrect shape.')
    def test_constructor(self):
        
        mesh = Mesh(node=QuadNode())
        self.assertEqual(mesh.dim(),2,'Mesh dimension should be 2.')
        self.assertIsNone(mesh.grid, 'Mesh has no grid.')
        
        mesh = Mesh(cell=QuadCell())
        self.assertEqual(mesh.dim(),2,'Mesh dimension should be 2.')
        self.assertIsNone(mesh.grid, 'Mesh has no grid.')
        
        mesh = Mesh(grid=Grid(dim=2))
        self.assertEqual(mesh.dim(),2,'Mesh dimension should be 2.')
        self.assertIsNotNone(mesh.grid, 'Mesh has a grid.')
        
        mesh = Mesh(node=BiNode())
        self.assertEqual(mesh.dim(),1,'Mesh dimension should be 1.')
        self.assertIsNone(mesh.grid, 'Mesh has no grid.')
        
        mesh = Mesh(cell=BiCell())
        self.assertEqual(mesh.dim(),1,'Mesh dimension should be 1.')
        self.assertIsNone(mesh.grid, 'Mesh has no grid.')
        
        mesh = Mesh(grid=Grid(dim=1))
        self.assertEqual(mesh.dim(),1,'Mesh dimension should be 1.')
        self.assertIsNotNone(mesh.grid, 'Mesh has a grid.')
    
    def test_box(self):
        #
        # 1D
        #  
        
        #
        # 2D
        #
        
        # Rectangular grid on [0,1,0,1]
        grid = Grid(dim=2)
        mesh = Mesh(grid=grid)
        x_min, x_max, y_min, y_max = mesh.bounding_box()
        self.assertAlmostEqual(x_min, 0, 9, 'x_min should be 0')
        self.assertAlmostEqual(x_max, 1, 9, 'x_min should be 0')
        self.assertAlmostEqual(y_min, 0, 9, 'x_min should be 0')
        self.assertAlmostEqual(y_max, 1, 9, 'x_min should be 0')
        
        # Non-Standard Grid on [0,1,0,1]
        print('Importing grid from file')
        file_path = '/home/hans-werner/git/quadmesh/tests/quarter_circle.msh'
        grid = Grid(file_path=file_path)
        
        # TODO: Finish
        #mesh = Mesh(grid=grid)
        x_min, x_max, y_min, y_max = mesh.bounding_box()
        self.assertAlmostEqual(x_min, 0, 9, 'x_min should be 0')
        self.assertAlmostEqual(x_max, 1, 9, 'x_min should be 0')
        self.assertAlmostEqual(y_min, 0, 9, 'x_min should be 0')
        self.assertAlmostEqual(y_max, 1, 9, 'x_min should be 0')
        
    def test_depth(self):
        mesh = Mesh()
        for _ in range(3):
            mesh.refine()
        self.assertEqual(mesh.depth(),3,\
                         'Refined mesh thrice, depth should be 3.')
        
    
    def test_root_node(self):
        node = QuadNode()
        mesh = Mesh(node=node)
        for _ in range(4):
            mesh.refine()
        self.assertEqual(mesh.root_node(), node, \
                         'Did not return root node.')
        
    
    def test_n_cells(self):
        mesh = Mesh()
        self.assertEqual(mesh.n_nodes(),1,\
                         'Mesh consists only of one node.')
        
        mesh.refine()
        self.assertEqual(mesh.n_nodes(),4,\
                         'Mesh should now have 4 nodes')
        
        mesh.root_node().children['SW'].mark(1)
        self.assertEqual(mesh.n_nodes(flag=1),1,\
                         'Mesh should have 1 marked node.')
        
        
    def test_coarsen(self):
        mesh = Mesh()
        self.assertFalse(mesh.root_node().has_children(),\
                         'ROOT node has no children.')
        mesh.refine()
        self.assertTrue(mesh.root_node().has_children(),\
                         'ROOT node now has children.')
        mesh.coarsen()
        self.assertFalse(mesh.root_node().has_children(),\
                         'ROOT node has no children.')
        
        # 
        # Mark 1 child -> not enough to coarsen
        #  
        mesh = Mesh()
        root = mesh.root_node()
        mesh.refine()
        root.children['SW'].mark(1)
        mesh.coarsen(flag=1)
        child_count = 0
        for _ in root.get_children():
            child_count += 1
        self.assertEqual(child_count,4, 'Root should still have 4 children.')
        
        #
        # Mark all children -> enough to coarsen
        # 
        mesh = Mesh()
        mesh.refine()
        root = mesh.root_node()
        for child in root.get_children():
            child.mark(flag=1)
        mesh.coarsen(flag=1)
        self.assertFalse(root.has_children(),'Coarsening should have occured.')
        
        #
        # Unflagged mesh with multiple layers
        # 
        mesh = Mesh()
        mesh.refine()
        node = mesh.root_node()
        for _ in range(3):
            node.children['SW'].mark(flag=1)
            mesh.refine(flag=1)
            node = node.children['SW']
        tree_depth = mesh.root_node().tree_depth()
        mesh.coarsen()
        self.assertEqual(mesh.root_node().tree_depth(),tree_depth-1,\
                         'Tree depth should have reduced by 1.')
        
        
        
    '''   
    
    def test_mesh_boundary(self):
        # TODO: Finish
        """
        mesh = Mesh.newmesh()
        mesh.refine()
        mesh.root_node().children['SW'].remove()
        mesh.root_node().info()
        for leaf in mesh.root_node().get_leaves():
            leaf.info()
        
        fig, ax = plt.subplots()
        plot = Plot()
        plot.mesh(ax,mesh)
        plt.show()
        print(len(mesh.boundary('vertices')))
        print(len(mesh.boundary('edges')))
        print(len(mesh.boundary('quadcells')))
        """    
    
    def test_mesh_iter_quadcells(self):
        #
        # Define new mesh and refine
        # 
        mesh = Mesh.newmesh(box=[0.,3.,0.,2], grid_size=(3,2))
        mesh.root_node().mark('split')
        mesh.refine('split')
        child_10 = mesh.root_node().children[(1,0)]
        child_10.mark('split')
        mesh.refine('split')
        child_10_ne = child_10.children['NE']
        child_10_ne.mark('split')
        mesh.refine('split')
        
        # Iterate quadcells
        mesh_quadcells = mesh.iter_quadcells()
           
        self.assertEqual(mesh_quadcells[0], \
                         mesh.root_node().children[(0,0)].quadcell(),\
                         'QuadCell (0,0) incorrectly numbered.')
        self.assertEqual(mesh_quadcells[2],\
                         child_10.children['SE'].quadcell(), \
                         'QuadCell (1,0)-SE incorrectly numbered.')
        self.assertEqual(mesh_quadcells[6],\
                         child_10_ne.children['NW'].quadcell(), \
                         'QuadCell (1,0)-NE-NW incorrectly numbered.')
            
    
    def test_mesh_iter_quadedges(self):     
        #
        # Define new mesh and refine
        # 
        mesh = Mesh.newmesh(box=[0.,3.,0.,2], grid_size=(3,2))
        mesh.root_node().mark('split')
        mesh.refine('split')
        
        child_10 = mesh.root_node().children[(1,0)] 
        child_10.mark('split')
        mesh.refine('split')
        
        child_10_ne = child_10.children['NE']
        child_10_ne.mark('split')        
        mesh.refine('split')
          
        # Iterate quadcells
        mesh_quadedges = mesh.iter_quadedges()
        
        test_edge_0 = mesh.root_node().children[(0,0)].quadcell().edges['SE','NE']
        self.assertEqual(mesh_quadedges[1],test_edge_0, \
                         'Edge (0,0)-E numbered incorrectly.')
        
        test_edge_1 = child_10.children['SW'].quadcell().edges['NE','NW']
        self.assertEqual(mesh_quadedges[7],test_edge_1, \
                         'Edge [(1,0),SW]-N numbered incorrectly.')
        
        test_edge_2 = child_10.children['NW'].quadcell().edges['SW','SE']
        self.assertEqual(mesh_quadedges[7],test_edge_2, \
                         'Edge [(1,0),NW]-S numbered incorrectly.')
        
        
    
    def test_mesh_iter_quadvertices(self):
        
        #
        # Define new mesh and refine
        # 
        mesh = Mesh.newmesh(box=[0.,3.,0.,2], grid_size=(3,2))
        mesh.root_node().mark('split')
        mesh.refine('split')
        child_10 = mesh.root_node().children[(1,0)]
        child_10.mark('split')
        mesh.refine('split')
        child_10_ne = child_10.children['NE']
        child_10_ne.mark('split')
        mesh.refine('split')
        
        quadvertices = mesh.quadvertices()    
    
        self.assertEqual(len(quadvertices),22,'There should be 22 vertices in total.')
        
    
    def test_mesh_balance(self):
        # TODO: After balancing, randomly says its balanced and not.
        mesh = Mesh.newmesh()
        mesh.refine()
        # Refine mesh arbitrarily (most likely not balanced)
        for _ in range(3):
            for leaf in mesh.root_node().get_leaves():
                if np.random.rand() < 0.5:
                    leaf.mark(1)
            mesh.refine(1)
        #print('Before balancing', mesh.is_balanced())
        mesh.balance()
        #print('After balancing', mesh.is_balanced())
    
    
    def test_record(self):
        #
        # Define and record simple 2,2 mesh
        # 
        mesh = Mesh.newmesh()
        mesh.refine()
        mesh.record()
        
        #
        # Refine and record
        #    
        mesh.root_node().children['SW'].mark('r1')
        mesh.refine('r1')
        mesh.record()
        """
        print("FINE MESH")
        for node in mesh.root_node().get_leaves(1):
            node.info()
        print("COARSE MESH")
        for node in mesh.root_node().get_leaves(0):
            node.info()
        """
        

    
    def test_mesh_plot_trimesh(self):
        pass
    '''
     
     
class TestGrid(unittest.TestCase):
    """
    Test Grid class
    """
    def test_constructor(self):
        #
        # 1D regular grid
        # 
        grid = Grid(box=(0,1), resolution=(2,))
        
        self.assertEqual(grid.get_neighbor(0,'R'), 1, \
                         'Right Neighbor of interval 0 is interval 1')
        self.assertIsNone(grid.get_neighbor(0,'L'),\
                        'Left neighbor should be None')
        self.assertTrue(all([tpe=='interval' for tpe in grid.faces['type']]), \
                         'Grid face type should be interval.')
        self.assertEqual(grid.faces['n'],2, \
                         'There should be two faces.')
        self.assertEqual(grid.points['n'], 3, \
                         'There should be 3 points.')
        
        #
        # 2D cartesian grid
        #
        grid = Grid(box=(0,1,0,1), resolution=(2,2))
        #
        # Check vertices of first face
        #
        f0 = grid.faces['connectivity'][0]
        xy_loc = [(0,0),(0.5,0),(0.5,0.5),(0,0.5)]
        for i in range(4):
            xy = grid.points['coordinates'][f0[i]]
            self.assertAlmostEqual(xy.coordinate(), xy_loc[i], 8, \
                                   'First cell vertices incorrect')
        #
        # Check neighbors
        # 
        directions = ['E','W','N','NE']
        nbrs = [1, None, 2, 3]
        for i in range(4):
            direction = directions[i]
            self.assertEqual(grid.get_neighbor(0,direction),nbrs[i],\
                             'Incorrect Neighbor.')
        
        #
        # Check boundary edges
        # 
        boundary_edges = grid.get_boundary_edges()
        for i_e in boundary_edges:
            i_he = grid.edges['half_edge'][i_e]
            i_fc = grid.half_edges['face'][i_he]
            direction = grid.half_edges['position'][i_he]
            self.assertIsNone(grid.get_neighbor(i_fc, direction),\
                              'Neighbor of boundary edge should be None')
         
        
        #
        # 2D Quadrilateral grid
        #
        '''
        # FIXME: Assigning directions to sides of quadrilaterals in a mesh is
        not consistent!!! 
        
         
        file_path = '/home/hans-werner/git/quadmesh/tests/quarter_circle.msh' 
        # file_path = '/home/hans-werner/git/quadmesh/debug/circle_mesh.msh'
        #file_path = '/home/hans-werner/Dropbox/work/code/matlab/'+\
        #            'finite_elements/examples/reaction_diffusion/'+\
        #            'circle_mesh.msh'
        grid = Grid(file_path=file_path)
        
        #
        # Check boundary edges
        # 
        boundary_edges = grid.get_boundary_edges()
        for i_e in boundary_edges:
            i_he = grid.edges['half_edge'][i_e]
            i_fc = grid.half_edges['face'][i_he]
            direction = grid.half_edges['position'][i_he]
            self.assertIsNone(grid.get_neighbor(i_fc, direction),\
                              'Neighbor of boundary edge should be None')
        '''
            
        #
        # 2D Triangular grid
        file_path = '/home/hans-werner/Dropbox/work/code/matlab/'+\
                    'finite_elements/mesh/gmsh_matlab/examples/'+\
                    'quarter_disk.msh'
        grid = Grid(file_path=file_path)
        boundary_edges = grid.get_boundary_edges()
        for i_e in boundary_edges:
            i_he = grid.edges['half_edge'][i_e]
            self.assertEqual(grid.half_edges['twin'][i_he],-1,\
                             'Neighbor of boundary half-edge should be -1.')
        #print(len(grid['faces']['tags']['phys']))
        
    '''
    def test_half_edge_positions(self):
        """
        Test whether the grid faces have the correct directions associated
        
        THIS CANNOT WORK FOR GENERAL MESHES.
        """
        file_path = '/home/hans-werner/git/quadmesh/tests/quarter_circle.msh' 
        grid = Grid(file_path=file_path)
        for i in range(grid.faces['n']):
            assert len(grid.faces['connectivity'][i]) == 4, 'Grid not made of quads.'
        
        #
        # Ensure each face contains 4 unique half-edges
        #         
        for i_fc in range(grid.faces['n']):
            i_he = grid.faces['half_edge'][i_fc]
            pos  = grid.half_edges['position'][i_he]
            i_hes = [i_he]
            positions = [pos]
            for _ in range(3):
                i_he = grid.half_edges['next'][i_he]
                i_hes.append(i_he)
                positions.append(grid.half_edges['position'][i_he]) 
            
            self.assertEqual(len(list(set(i_hes))), 4, \
                             'There should be 4 unique half-edges.')
            
            # TODO: This doesnt hold
            self.assertEqual(len(list(set(positions))), 4, \
                             'There should be 4 unique half-edges \n'+
                             '{0}.'.format(positions))
            
        """    
        # Reproduce position assignment
        for i_fc in range(grid.faces['n']):
            i_he = grid.faces['half_edge'][i_fc]
        """ 
            
        for pos in range(grid.faces['n']):
            if pos==20:
                vertices = dict.fromkeys(['SW','SE','NW','NE'])
                i_conn = grid.faces['connectivity'][pos]
                vs = [grid.points['coordinates'][i] for i in i_conn]
                print(vs)
                print('\n\n')

                i_he = grid.faces['half_edge'][pos]
                sub_dirs = {'S': ['SW','SE'], 'E': ['SE', 'NE'], 
                            'N': ['NE','NW'], 'W': ['NW','SW'] }
                for _ in range(3):
                    print(i_he)
                    direction = grid.half_edges['position'][i_he]
                    
                    for j in range(2):
                        sub_dir = sub_dirs[direction][j]
                        i_vtx = grid.half_edges['connectivity'][i_he][j] 
                        vertices[sub_dir] = grid.points['coordinates'][i_vtx]
                    # Proceed to next half-edge
                    i_he = grid.half_edges['next'][i_he]
                for p in ['SW', 'SE','NE','NW']:
                    print(vertices[p])
            cell = QuadCell(position=pos, grid=grid)
    '''
                
class TestNode(unittest.TestCase):
    """
    Test Node Class
    """

    def test_node_constructor(self):
        #
        # Children standard
        #
        for node in [BiNode(), QuadNode()]:
            self.assertEqual(node.depth, 0, 'Node depth should be zero.')
            self.assertTrue(node.type=='ROOT', 'Node should be of type ROOT.')
            if isinstance(node, BiNode):
                generic_children = {'L': None, 'R': None}
            elif isinstance(node, QuadNode):
                generic_children = {'SW':None, 'SE':None, 'NE':None, 'NW':None}
            self.assertEqual(node.children, generic_children, \
                             'Incorrect form for children.')
        self.assertEqual(node.grid, None, \
                         'Child grid should be None.')
        #
        # Children in grid
        #
        binode = BiNode(grid=Grid(resolution=(2,)))
        self.assertEqual(binode.grid_size(),2, 
                         'Child grid size should be 2.')
        grid = Grid(resolution=(2,2))
        quadnode = QuadNode(grid=grid)
        self.assertEqual(quadnode.grid.resolution,(2,2), 
                         'Child grid resolution should be (2,2).')
    
    
    def test_copy(self):
        pass
        '''
        node = QuadNode(grid_size=(2,1))
        node.split()
        e_child = node.children[(1,0)]
        e_child.split()
        e_ne_child = e_child.children['NE']
        e_ne_child.split()
        cnode = node.copy()    
        self.assertNotEqual(cnode, node, \
                            'Copied node should be different from original.')
        '''
    
    def test_grid_size(self):
        pass
       
     
    def test_tree_depth(self):       
        count = 0
        for node in [QuadNode(), BiNode()]:
            positions = ['SW', 'L']
            inode = node
            for _ in range(5):
                inode.split()
                inode = inode.children[positions[count]]
            count += 1
            self.assertEqual(node.tree_depth(), 5, 
                             'Tree depth should be 5.')
         
         
    def test_traverse(self):        
        #
        # 1D
        #  
        # Standard
        node = BiNode()
        node.split()
        node.children['L'].split()
        node.children['L'].children['R'].remove()
        addresses = {'breadth-first': [[],[0],[1],[0,0]], 
                     'depth-first': [[],[0],[0,0],[1]]}
 
        for mode in ['depth-first','breadth-first']:
            count = 0
            for leaf in node.traverse(mode=mode):
                self.assertEqual(leaf.address, addresses[mode][count],
                                 'BiNode traversal incorrect.')
                count += 1
        
        #
        # Standard Node
        # 
        node = QuadNode()
        node.split()
        node.children['SE'].split()
        node.children['SE'].children['NW'].remove()
        addresses = [[],[0],[1],[2],[3],[1,0],[1,1],[1,3]]
        count = 0
        for n in node.traverse(mode='breadth-first'):
            self.assertEqual(n.address, addresses[count],\
                             'Incorrect address.')
            count += 1
         
        #
        # Gridded Node
        #  
        grid = Grid(resolution=(3,3))   
        node = QuadNode(grid=grid)
        
        node.split()
        addresses = [[]]
        count = 0
        for _ in range(3):
            for _ in range(3):
                addresses.append([count])
                count += 1
        count = 0
        for n in node.traverse(mode='breadth-first'):
            self.assertEqual(n.address, addresses[count],\
                             'Incorrect address.')
            count += 1
       
            
    def test_get_leaves(self):
        #
        # 1D
        # 
        node = BiNode()
        leaves = node.get_leaves()
        self.assertEqual(leaves, [node], 'Cell should be its own leaf.')
        
        #
        # Split cell and L child - find leaves
        # 
        node.split()
        l_child = node.children['L']
        l_child.split()
        leaves = node.get_leaves()
        self.assertEqual(len(leaves),3, 'Cell should have 3 leaves.')
        
        #
        # Depth first order
        # 
        addresses_depth_first = [[0,0],[0,1],[1]]
        leaves = node.get_leaves(nested=False)
        for i in range(len(leaves)):
            leaf = leaves[i]
            self.assertEqual(leaf.address, addresses_depth_first[i],
                             'Incorrect order, depth first search.')
        #
        # Breadth first order
        # 
        addresses_breadth_first = [[1],[0,0],[0,1]]
        leaves = node.get_leaves(nested=True)
        for i in range(len(leaves)):
            leaf = leaves[i]
            self.assertEqual(leaf.address, addresses_breadth_first[i],
                             'Incorrect order, breadth first search.')
        
        
        node.children['L'].children['L'].mark('1')
        node.children['R'].mark('1')
        leaves = node.get_leaves(flag='1', nested='True')
        self.assertEqual(len(leaves),2, \
                         'There should only be 2 flagged leaves')
        
        #
        # 2D
        # 
        node = QuadNode()
        
        #
        # Split cell and SW child - find leaves
        # 
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        leaves = node.get_leaves()
        self.assertEqual(len(leaves), 7, 'Node should have 7 leaves.')
        
        #
        # Nested traversal
        #
        leaves = node.get_leaves(nested=True)
        self.assertEqual(leaves[0].address,[1], \
            'The first leaf in the nested enumeration should have address [1]')
        
        leaves = node.get_leaves()
        self.assertEqual(leaves[0].address, [0,0], \
                         'First leaf in un-nested enumeration should be [0,0].')
        
        #
        # Merge SW child - find leaves
        # 
        for child in sw_child.get_children():
            child.remove()
        leaves = node.get_leaves()
        self.assertEqual(len(leaves), 4, 'Node should have 4 leaves.')
        
        
        #
        # Marked Leaves
        # 
        node = QuadNode()
        node.mark(1)
        self.assertTrue(node in node.get_leaves(flag=1), \
                        'Node should be a marked leaf node.')
        self.assertTrue(node in node.get_leaves(), \
                        'Node should be a marked leaf node.')
    
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        sw_child.mark(1)
        self.assertEqual(node.get_leaves(flag=1), \
                         [sw_child], 'SW child should be only marked leaf')
        
        sw_child.remove()
        self.assertEqual(node.get_leaves(flag=1), \
                         [node], 'node should be only marked leaf')
        
        #
        # Nested traversal
        # 
        node = QuadNode()
        node.split()
        for child in node.get_children():
            child.split()
            
        node.children['SE'].mark(1, recursive=True)
        node.children['NE'].mark(1)
        
        leaves = node.get_leaves(nested=True, flag=1)
        self.assertEqual(len(leaves), 5, 
                         'This tree has 5 flagged LEAF nodes.')
        self.assertEqual(leaves[0], node.children['NE'], 
                         'The first leaf should be the NE child.')
        self.assertEqual(leaves[3], node.children['SE'].children['NW'],
                         '4th flagged leaf should be SE-NW grandchild.')
                
        
    def test_get_root(self):
        count = 0
        pos = ['L','SE']
        for node in [BiNode(), QuadNode()]:
            self.assertEqual(node.get_root(), node, 
                             'Node is its own root.')
            node.split()
            child = node.children[pos[count]]
            self.assertEqual(child.get_root(), node, 
                             'Node is its childs root.')
            count +=1
    
    
    def test_find_node(self):
        # TODO: 1D
        
        # 2D
        node = QuadNode()
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        #
        # Find root node
        # 
        node_address = []
        self.assertEqual(node.find_node(node_address), node, 'ROOT node has address [].')
        #
        # SW -> NE grandchild
        # 
        node_address = [0,3]
        sw_ne_grandchild = sw_child.children['NE']
        self.assertEqual(node.find_node(node_address), sw_ne_grandchild, \
                         'SW, NE grandchild has address [0,3].')
        grid = Grid(resolution=(2,2))
        node = QuadNode(grid=grid)
        node.split()
        lb_child = node.children[0]
        lb_child.split()
        address = [0,3]
        lb_ne_grandchild = lb_child.children['NE']
        self.assertEqual(node.find_node(address), lb_ne_grandchild, \
                         'Left bottom, NE grandchild has address [(0,0),2].')
        
        
    def test_has_children(self):
        # TODO: Test 1D
        
        # 2D
        node = QuadNode()
        node.split()
        node.children['NW'].remove()
        node.children['SE'].mark(1)
        node.children['NE'].mark('hihihi')
        
        self.assertFalse(node.has_children(position='NW'), \
                         'Node should not have a NW child.')
        self.assertTrue(node.has_children(flag=1), \
                        'Node should have a child marked "1".')
        self.assertTrue(node.has_children(position='SE',flag=1),
                        'SE child is marked 1.')
        self.assertFalse(node.has_children(position='NE', flag=1),
                         'NE child is not marked 1.')
        self.assertTrue(node.has_children(),'Node has children')
    
    
    def test_get_children(self):
        
        # TODO: 1D

        # 2D
        node = QuadNode()
        node.split()
        count = 0
        pos = ['SW','SE','NW','NE']
        for child in node.get_children():
            self.assertEqual(child.position,pos[count],\
                             'Incorrect child.')
            count += 1
            
        #
        # Now remove child
        # 
        node.children['SW'].remove()
        count = 0 
        for child in node.get_children():
            count += 1
        self.assertEqual(count, 3, 'There should only be 3 children left.')   
        #
        # Node with no children   
        # 
        node = QuadNode()
        for child in node.get_children():
            print('Hallo')
            
        #
        # Try a logical statement 
        # 
        self.assertFalse(any([child.is_marked(1) for \
                              child in node.get_children()]), \
                         'No marked children because there are none.')
        
        
    def test_has_parent(self):
        for node in [BiNode(), QuadNode()]:
            node.split()
            for child in node.get_children():
                self.assertTrue(child.has_parent(),\
                                'Nodes children should have a parent.')  
            node.mark(1)
            for child in node.get_children():
                self.assertTrue(child.has_parent(1), \
                                'Children should have parent labeled 1.')
                self.assertFalse(child.has_parent(2),\
                                 'Children do not have parent labeled 2.')
            
            self.assertFalse(node.has_parent(1), \
                             'Root node should not have parents of type 1.')
            self.assertFalse(node.has_parent(), \
                             'Root node should not have parents.')
            
        
        
        
    def test_get_parent(self):
        count = 0
        pos1 = ['L','SW']
        pos2 = ['R','NE']
        for node in [BiNode(), QuadNode()]:            
            node.mark(1)
            node.split()
            child = node.children[pos1[count]]
            child.split()
            self.assertEqual(node,child.children[pos2[count]].get_parent(1),\
                             'First marked ancestor should be node.')     
            child.mark(1)
            self.assertEqual(child,child.children[pos2[count]].get_parent(1),\
                             'First marked ancestor should be child.')
            count += 1
        
        
    def test_node_in_grid(self):  
        #
        # 2D
        # 
        node = QuadNode()
        node.split()
        sw_child = node.children['SW']
        self.assertFalse(sw_child.in_grid(), 'Child is not in grid.')
        
        grid = Grid(resolution=(2,2))
        node = QuadNode(grid=grid)
        node.split()
        lb_child = node.children[0]
        self.assertTrue(lb_child.in_grid(), 'Child lives in grid.')
    
        #
        # 1D
        #
        node = BiNode()
        node.split()
        r_child = node.children['R']
        self.assertFalse(r_child.in_grid(), 'Child is not in grid.')
        
        node = BiNode(grid=Grid(resolution=(2,)))
        node.split()
        l_child = node.children[0]
        self.assertTrue(l_child.in_grid(), 'Child lives in grid.')
    
        
    def test_node_is_marked(self):
        for node in [BiNode(), QuadNode()]:
            node.mark()
            self.assertTrue(node.is_marked(),'Node should be marked.')
            node.unmark()
            self.assertFalse(node.is_marked(),'Node should not be marked.')
        
        
    def test_mark(self):
        for node in [BiNode(), QuadNode()]:
            node.mark()
            self.assertTrue(node.is_marked(),'Node should be marked.')
    
    
    def test_unmark(self):
        #
        # 3 Generations of marked nodes
        # 
        pos1 = ['L','SW']
        pos2 = ['L','SW']
        count = 0
        for node in [BiNode(), QuadNode()]:
            
            node.mark()
            node.split()
            child = node.children[pos1[count]]
            child.mark()
            child.split()
            grandchild = child.children[pos2[count]]
            grandchild.mark()
            
            #
            # Unmark sw_child node
            #
            child.unmark()
            self.assertTrue(node.is_marked(), \
                            'Node should still be marked.')
            self.assertFalse(child.is_marked(),\
                             'Child should be unmarked.')
            self.assertTrue(grandchild.is_marked(),\
                            'Grandchild should be marked.')
              
            # Reset
            child.mark()
            
            #
            # Unmark recursively
            # 
            child.unmark(recursive=True)
            self.assertTrue(node.is_marked(), \
                            'Node should still be marked.')
            self.assertFalse(child.is_marked(),\
                             'Child should be unmarked.')
            self.assertFalse(grandchild.is_marked(),\
                             'Grandchild should be unmarked.')
            
            # Reset
            grandchild.mark()
            child.mark()
            
            #
            # Unmark all
            # 
            node.unmark(recursive=True)
            self.assertFalse(node.is_marked(), 'Node should still be marked.')
            self.assertFalse(child.is_marked(),'Child should be unmarked.')
            self.assertFalse(grandchild.is_marked(),'Grandchild should be marked.')
        
            count += 1
    
    
    
    def test_is_linked(self):
        pass
    
    
    def test_link(self):
        pass
    
    
    def test_unlink(self):
        pass
    

    def test_cell(self):
        pass
    
        
    def test_node_merge(self):
        for node in [BiNode(), QuadNode()]:
            node.split()
            node.merge()
            self.assertFalse(node.has_children(),
                             'Node should not have children.')
        grid = Grid(resolution=(2,2))
        for node in [BiNode(grid=Grid(resolution=(2,))), QuadNode(grid=grid)]:
            node.split()
            node.merge()
            self.assertFalse(node.has_children(),\
                             'Node should not have children.')
        
        
    def test_remove(self):
        pos = ['L','SW']
        count = 0
        for node in [BiNode(), QuadNode()]:
            node.split()
            node.children[pos[count]].remove()
            self.assertEqual(node.children[pos[count]],None,\
                             'Node should have been removed.')
            count += 1
        
        # Gridded
        
        grid = Grid(resolution=(2,))
        node = BiNode(grid=grid)
        node.split()
        node.children[0].remove()
        self.assertEqual(node.children[0],None,\
                         'Child 0 should have been removed.')
        grid = Grid(resolution=(2,2))    
        node = QuadNode(grid=grid)
        node.split()
        node.children[0].remove()
        self.assertEqual(node.children[0],None, \
                         'Node should have been removed.')
    
    
    def test_node_split(self):
        for node in [BiNode(), QuadNode()]:
            node.split()
            self.assertTrue(node.has_children(),\
                            'Split node should have children.')


class TestBiNode(unittest.TestCase):
    """
    Test the BiNode subclass of Node
    """
    def test_find_neighbor(self):
        binode = BiNode()
        self.assertIsNone(binode.get_neighbor('L'), \
                          'neighbor should be None.')
        
        binode.split()
        l_child = binode.children['L']
        self.assertEqual(l_child.get_neighbor('R'), binode.children['R'],\
                         'neighbor interior to parent cell not identified.')
        
        l_child.split()
        lr_grandchild = l_child.children['R']
        self.assertEqual(lr_grandchild.get_neighbor('R'), 
                         binode.children['R'], 
                         'neighbor exterior to parent cell not identified.')
        
        binode.children['R'].split()
        self.assertEqual(lr_grandchild.get_neighbor('R'),\
                         binode.children['R'].children['L'],\
                         'neighbor exterior to parent cell not identified.')
        
        binode = BiNode(grid=Grid(resolution=(3,)))
        
        binode.split()
        lchild = binode.children[0]
        self.assertEqual(lchild.get_neighbor('L'),None,
                         'neighbor of gridded cell not identified as None.')
        
        self.assertEqual(lchild.get_neighbor('R'),binode.children[1],
                         'neighbor of gridded cell not identified.')

  
    def test_pos2id(self):
        binode = BiNode(grid=Grid(resolution=(3,)))
        binode.split()
        self.assertEqual(binode.pos2id(0), 0, 
                         'Position in grid incorrectly converted.')
        
        
class TestQuadNode(unittest.TestCase):
    """
    Test the QuadNode class/a subclass of Node
    """

    def test_node_find_neighbor(self):
        # 2D
        node = QuadNode()
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        nw_grandchild = sw_child.children['NW']
        ne_grandchild = sw_child.children['NE']
        #
        # Neighbor exterior to parent cell
        #  
        self.assertEqual(nw_grandchild.get_neighbor('N'), node.children['NW'], 
                         'Neighbor should be NW child of ROOT cell.')
        self.assertEqual(ne_grandchild.get_neighbor('NE'), node.children['NE'], 
                         'Neighbor should be NE child of ROOT cell.')
        #
        # Neighbor is sibling cell
        #  
        self.assertEqual(nw_grandchild.get_neighbor('S'), sw_child.children['SW'], 
                         'Neighbor should be SW sibling.')
        self.assertEqual(nw_grandchild.get_neighbor('SE'),sw_child.children['SE'],
                         'Neighbor should be SE sibling.')
        #
        # Neighbor is None
        # 
        self.assertEqual(nw_grandchild.get_neighbor('W'), None, 
                         'Neighbor should be None.')
        self.assertEqual(nw_grandchild.get_neighbor('NE'),None,
                         'Neighbor should be None.')
        
        node.children['NW'].split()
        self.assertEqual(nw_grandchild.get_neighbor('NE'),
                         node.children['NW'].children['SE'],
                         'Neighbor should be the NW-SE grandchild.')
        
            
    def test_node_pos2id(self):
        node = QuadNode()
        node.split()
        node = node.children['SW']
        self.assertEqual(node.pos2id('SW'),0,'sw -> 0.')
        self.assertEqual(node.pos2id('SE'),1,'se -> 1.')
        self.assertEqual(node.pos2id('NW'),2,'nw -> 2.')
        self.assertEqual(node.pos2id('NE'),3,'ne -> 3.')
        
        self.assertRaises(Exception, node.pos2id, (0,0))       
        self.assertRaises(Exception, node.pos2id, [0,0])
        
        
    def test_node_id2pos(self):
        node = QuadNode()
        node.split()
        node = node.children['SW']
        self.assertEqual(node.id2pos(0),'SW','sw <- 0.')
        self.assertEqual(node.id2pos(1),'SE','se <- 1.')
        self.assertEqual(node.id2pos(2),'NW','nw <- 2.')
        self.assertEqual(node.id2pos(3),'NE','ne <- 3.')
        
        self.assertRaises(Exception, node.id2pos, (0,0))       
        self.assertRaises(Exception, node.id2pos, [0,0])
        

    def test_balance(self):
        node = QuadNode()
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        sw_ne_grandchild = sw_child.children['NE']
        sw_ne_grandchild.split()
        
        self.assertFalse(node.children['NW'].has_children(),\
                         'NW child should not have children before balance.')
        self.assertFalse(node.children['SE'].has_children(),\
                         'SE child should not have children before balance.')
        node.balance()
        
        self.assertTrue(node.children['NW'].has_children(),\
                         'NW child should have children after balance.')
        self.assertTrue(node.children['SE'].has_children(),\
                         'SE child should have children after balance.')


    def test_is_balanced(self):
        node = QuadNode()
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        sw_ne_grandchild = sw_child.children['NE']
        sw_ne_grandchild.split()
        self.assertFalse(node.is_balanced(),'Tree is not balanced.')
        
        node.balance()
        self.assertTrue(node.is_balanced(),'Tree is balanced.')
        
        #
        # Test 2
        # 
        node = QuadNode()
        node.split()
        # Split node arbitrarily (most likely not balanced)
        for _ in range(3):
            for leaf in node.get_leaves():
                if np.random.rand() < 0.5:
                    leaf.split()
        node.balance()
        self.assertTrue(node.is_balanced(),'Node should be balanced.')
        """
        Debugging: 
        
        if not node.is_balanced():
            for leaf in node.get_leaves():
                for direction in ['N','S','E','W']:
                    nb = leaf.get_neighbor(direction)
                    if nb is not None and nb.has_children():
                        for child in nb.children.values():
                            if child.type != 'LEAF':
                                print('child is not leaf')
                                print('Node:')
                                leaf.info()
                                print('\n\nNeighbor:')
                                nb.info()
                                print('\n\nChild:')
                                child.info()
                                for gchild in child.get_children():
                                    print('\n Grandchild:')
                                    gchild.info()
        """
    
    
    def test_node_remove_supports(self):
        #
        # Split and balance
        # 
        node = QuadNode()
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        sw_ne_grandchild = sw_child.children['NE']
        sw_ne_grandchild.split()
        node.balance()
        
        #
        # Coarsen and remove supports
        # 
        sw_ne_grandchild.merge()
        node.remove_supports()
        
        self.assertFalse(node.children['NW'].has_children(),\
                         'NW child should not have children after coarsening.')
        self.assertFalse(node.children['SE'].has_children(),\
                         'SE child should not have children after coarsening.') 
        
           
class TestCell(unittest.TestCase):
    """
    Test Cell Class
    """   
    def test_get_vertices(self):
        # 1D 
        cell1d = BiCell()
        cell1d_vertices = np.array([[0],[1],[0.5]])
        self.assertTrue(np.allclose(cell1d.get_vertices(),cell1d_vertices),\
                        'BiCell vertices not correct.')
        self.assertTrue(np.allclose(cell1d.get_vertices('L'), np.array([[0]])),\
                        'BiCell get specific vertex not correct.')
        
        # 2D 
        cell2d = QuadCell()
        cell2d_vertices = np.array([[0,0],[0.5,0],[1,0],[1,0.5],[1,1],\
                                  [0.5,1],[0,1],[0,0.5],[0.5,0.5]])
        self.assertTrue(np.allclose(cell2d.get_vertices(),cell2d_vertices),\
                        'QuadCell vertices not correct.')
        self.assertTrue(np.allclose(cell2d.get_vertices('M'), np.array([[0.5,0.5]])),\
                        'QuadCell get specific vertex not correct.')
    
    def test_traverse(self):
        #
        # 1D
        #  
        # Standard
        cell = BiCell()
        cell.split()
        cell.children['L'].split()
        cell.children['L'].children['R'].remove()
        addresses = {'breadth-first': [[],[0],[1],[0,0]], 
                     'depth-first': [[],[0],[0,0],[1]]}
 
        for mode in ['depth-first','breadth-first']:
            count = 0
            for leaf in cell.traverse(mode=mode):
                self.assertEqual(leaf.address, addresses[mode][count],
                                 'Bicell traversal incorrect.')
                count += 1
        
        
        #
        # Standard QuadCell
        # 
        cell = QuadCell()
        cell.split()
        cell.children['SE'].split()
        cell.children['SE'].children['NW'].remove()
        addresses = [[],[0],[1],[2],[3],[1,0],[1,1],[1,3]]
        count = 0
        for n in cell.traverse(mode='breadth-first'):
            self.assertEqual(n.address, addresses[count],\
                             'Incorrect address.')
            count += 1
         
    
        
    def test_find_leaves(self):
        #
        # 1D
        # 
        cell = BiCell()
        leaves = cell.get_leaves()
        self.assertEqual(leaves, [cell], 'Cell should be its own leaf.')
        
        #
        # Split cell and L child - find leaves
        # 
        cell.split()
        l_child = cell.children['L']
        l_child.split()
        leaves = cell.get_leaves()
        self.assertEqual(len(leaves),3, 'Cell should have 3 leaves.')
        
        #
        # Depth first order
        # 
        addresses_depth_first = [[0,0],[0,1],[1]]
        leaves = cell.get_leaves(nested=False)
        for i in range(len(leaves)):
            leaf = leaves[i]
            self.assertEqual(leaf.address, addresses_depth_first[i],
                             'Incorrect order, depth first search.')
        #
        # Breadth first order
        # 
        addresses_breadth_first = [[1],[0,0],[0,1]]
        leaves = cell.get_leaves(nested=True)
        for i in range(len(leaves)):
            leaf = leaves[i]
            self.assertEqual(leaf.address, addresses_breadth_first[i],
                             'Incorrect order, breadth first search.')
        
        
        cell.children['L'].children['L'].mark('1')
        cell.children['R'].mark('1')
        leaves = cell.get_leaves(flag='1', nested='True')
        self.assertEqual(len(leaves),2, \
                         'There should only be 2 flagged leaves')
    
        #
        # 2D
        # 
        cell = QuadCell()
        
        #
        # Split cell and SW child - find leaves
        # 
        cell.split()
        sw_child = cell.children['SW']
        sw_child.split()
        leaves = cell.get_leaves()
        self.assertEqual(len(leaves), 7, 'Node should have 7 leaves.')
        
        #
        # Nested traversal
        #
        leaves = cell.get_leaves(nested=True)
        self.assertEqual(leaves[0].address,[1], \
            'The first leaf in the nested enumeration should have address [1]')
        
        leaves = cell.get_leaves()
        self.assertEqual(leaves[0].address, [0,0], \
                         'First leaf in un-nested enumeration should be [0,0].')
        
        #
        # Merge SW child - find leaves
        # 
        for child in sw_child.get_children():
            child.remove()
        leaves = cell.get_leaves()
        self.assertEqual(len(leaves), 4, 'Node should have 4 leaves.')
        
    
    def test_cells_at_depth(self):
        pass
    
    
    def test_get_root(self):
        for rootcell in [BiCell(), QuadCell()]:
            cell = rootcell
            for _ in range(10):
                cell.split()
                i = np.random.randint(0,2)
                pos = cell._child_positions[i]
                cell = cell.children[pos]
            
            self.assertEqual(cell.get_root(), rootcell, \
                             'Root cell not found')
    
    
    def test_has_children(self):
        marked_children = ['L','SW']
        count = 0
        for cell in [BiCell(), QuadCell()]:
            self.assertFalse(cell.has_children(), 
                             'Cell should not have children')
            cell.split()
            self.assertTrue(cell.has_children(),
                            'Cell should have children')
            self.assertFalse(cell.has_children(flag='1'),
                             'Cell should not have marked children')
            cell.children[marked_children[count]].mark('1')
            self.assertTrue(cell.has_children(flag='1'),
                            'Cell should have a child marked "1".')
            count += 1
   
    
    def test_has_parent(self):
        for cell in [BiCell(), QuadCell()]:
            self.assertFalse(cell.has_parent(),
                             'Root cell should not have a parent')
            cell.split()
            for child in cell.get_children():
                self.assertTrue(child.has_parent(),
                                'Child cell should have a parent.')
            
    
    
    def test_marking(self):
        for cell in [BiCell(), QuadCell()]:
            cell.mark()
            self.assertTrue(cell.is_marked(),'Cell should be marked.')
            
            cell.unmark()
            self.assertFalse(cell.is_marked(),'Cell should not be marked.')

            cell.mark('66')
            self.assertTrue(cell.is_marked(), 
                            'Cell should be marked.')
            self.assertFalse(cell.is_marked('o'), 
                             'Cell should not be marked "o".')
            self.assertTrue(cell.is_marked('66'), 
                            'Cell should be marked 66.')
    
    
class TestBiCell(unittest.TestCase):
    """
    Test the BiCell Clas
    """    
    def test_constructor(self):
        pass
    
    def test_area(self):
        bicell = BiCell()
        self.assertEqual(bicell.area(), 1, \
                         'area incorrect: default bicell')
        
    def test_find_neighbor(self):
        bicell = BiCell()
        self.assertIsNone(bicell.get_neighbor('L'), \
                          'neighbor should be None.')
        
        bicell.split()
        l_child = bicell.children['L']
        self.assertEqual(l_child.get_neighbor('R'), bicell.children['R'],\
                         'neighbor interior to parent cell not identified.')
        
        l_child.split()
        lr_grandchild = l_child.children['R']
        self.assertEqual(lr_grandchild.get_neighbor('R'), 
                         bicell.children['R'], 
                         'neighbor exterior to parent cell not identified.')
        
        bicell.children['R'].split()
        self.assertEqual(lr_grandchild.get_neighbor('R'),\
                         bicell.children['R'].children['L'],\
                         'neighbor exterior to parent cell not identified.')
        """
        bicell = BiCell(grid_size=3)
        bicell.split()
        lchild = bicell.children[0]
        self.assertEqual(lchild.get_neighbor('L'),None,
                         'neighbor of gridded cell not identified as None.')
        
        self.assertEqual(lchild.get_neighbor('R'),bicell.children[1],
                         'neighbor of gridded cell not identified.')
        """
        
    def test_get_root(self):
        bicell = BiCell()
        cell = bicell
        for _ in range(10):
            cell.split()
            cell = cell.children['L']
        self.assertEqual(bicell, cell.get_root(),\
                         'Root cell incorrectly identified.') 
        
    
    def test_contains_point(self):
        bicell = BiCell()
        self.assertFalse(bicell.contains_point(3))
        self.assertTrue(all(bicell.contains_point(np.array([1,2]))==\
                         np.array([True,False],dtype=np.bool)), 
                         'Inclusion of vector in cell incorrectly determined')
    
            
    def test_locate_point(self):
        bicell = BiCell()
        cell = bicell
        for _ in range(5):
            cell.split()
            cell = cell.children['L']
        self.assertEqual(bicell.locate_point(1/64),cell,
                         'Smallest cell containing point incorrectly detrmnd')
    
    
    def test_reference_map(self):
        #
        # Backward Map
        #
        bicell = BiCell(corner_vertices=[-2,10])
        y, jac = bicell.reference_map([-2,10], jacobian=True, mapsto='reference') 
        self.assertTrue(np.allclose(y, np.array([0,1])),\
            'Points incorrectly mapped to reference.')
        self.assertAlmostEqual(jac[0], 1/12,'Derivative incorrectly computed.')
        
        #
        # Forward Map
        # 
        bicell = BiCell(corner_vertices=[-2,10])
        y, jac = bicell.reference_map([0,1], jacobian=True, mapsto='physical')
        self.assertTrue(np.allclose(y, np.array([-2,10])),\
                        'Points incorrectly mapped from reference.')
        self.assertAlmostEqual(jac[0], 12,'Derivative incorrectly computed.')
    
    
    def test_split(self):
        
        # Ungridded
        bicell = BiCell()
        self.assertFalse(bicell.has_children(), 
                         'Bicell should not have children')
        bicell.split()
        self.assertTrue(bicell.has_children(),
                        'Bicell should have children.')
        self.assertEqual(bicell.children['L'].box(), (0,0.5),
                         'Bicell left child incorrect bounds.')

        
        # gridded
        grid = Grid(dim=1, resolution=(3,))
        bicell = BiCell(grid=grid, position=0)
        self.assertFalse(bicell.has_children(), 
                         'Bicell should not have children')
        bicell.split()
        count = 0
        for _ in bicell.get_children():
            count += 1
        self.assertEqual(count, 2, 'Bicell should have 3 children.')
    
    
    def test_pos2id(self):
        grid = Grid(dim=1, resolution=(3,))
        bicell = BiCell(grid=grid, position=0)
        bicell.split()
        self.assertEqual(bicell.pos2id(0), 0, 
                         'Position in grid incorrectly converted.')
    
        
class TestQuadCell(unittest.TestCase):
    """
    Test QuadCell Class
    """
    def test_constructor(self):
        # TODO: unfinished
        
        # Define basic QuadCell
        box = [0.,1.,0.,1.]
        Q1 = QuadCell()
        #_,ax0 = plt.subplots()
        #Q1.plot(ax0)
        """        
        _,ax1 = plt.subplots()
        Q2 = QuadCell(box=box, grid_size=(2,2))
        Q2.plot(ax1)
        plt.title('No Refinement')
        _,ax2 = plt.subplots()
        Q2.split()
        Q2.plot(ax2)
        plt.title('First Refinement')
        
        Q2_00 = Q2.children[0,0]
        Q2_00.split()
        
        q2002 = Q2_00.children['NE']
        print('-'*10)
        print(q2002.address)
        print('-'*10)
        print('Neighbors')
        for direction in ['N','S','E','W']:
            nb = q2002.get_neighbor(direction)
            print('{0}: {1}'.format(direction, repr(nb.box())))
            
        q2002.split()
        q2002.children['NW'].split()
            #plt.plot(Q2.vertices[v].coordinate(),'.')
        
        
        _,ax = plt.subplots()
        Q2.plot(ax)
        plt.title('Second Refinement')
        
        
        #x = numpy.linspace(0,1,100)
        #plt.plot(x,numpy.sin(x))
        plt.show()
        """        
    def test_area(self):
        #
        # Standard Quadcell
        # 
        cell = QuadCell()
        self.assertEqual(cell.area(), 1, 'Area should be 1: default cell.')
        
        # TODO: test nonstandard case
        
    def test_is_rectangle(self):
        #
        # Standard QuadCell
        # 
        cell = QuadCell()
        self.assertTrue(cell.is_rectangle(),'Cell should be a rectangle')
        
        #
        # Non-rectangular cell
        # 
        cnr_vs = [Vertex((0,0)), Vertex((1,1)), Vertex((0,2)), Vertex((0,1))]
        cell = QuadCell(corner_vertices=cnr_vs)
        self.assertFalse(cell.is_rectangle(), 'Cell should not be a rectangle')
        
        #
        # Rectangular cell
        # 
        cnr_vs = [Vertex((0,0)), Vertex((2,0)), Vertex((2,5)), Vertex((0,5))]
        cell = QuadCell(corner_vertices=cnr_vs)
        self.assertTrue(cell.is_rectangle(), 'Cell should be a rectangle')
        
        #
        # Cell in regular grid
        #
        grid = Grid(box=[0,1,0,1], resolution=(2,4))
        cell = QuadCell(position=0, grid=grid)
        self.assertTrue(cell.is_rectangle(), 'Cell should be a rectangle')
        
        
    def test_mark(self):
        box = [0.,1.,0.,1.]
        qcell = QuadCell(corner_vertices=box)
        qcell.mark()
        self.assertTrue(qcell.is_marked(),'Quadcell should be marked.')


    def test_unmark(self):
        #
        # 3 Generations of marked cells
        # 
        box = [0.,1.,0.,1.]
        qcell = QuadCell(corner_vertices=box)
        qcell.mark()
        qcell.split()
        sw_child = qcell.children['SW']
        sw_child.mark()
        sw_child.split()
        sw_sw_child = sw_child.children['SW']
        sw_sw_child.mark()
        
        #
        # Unmark only SW child
        # 
        sw_child.unmark()
        self.assertTrue(qcell.is_marked(),'Quadcell should be marked.')
        self.assertFalse(sw_child.is_marked(), 'SW child should not be marked.')
        self.assertTrue(sw_sw_child.is_marked(), 'SW-SW child should be marked')
        
        # Restore
        sw_child.mark()
        
        #
        # Unmark recursively
        # 
        sw_child.unmark(recursive=True)
        self.assertTrue(qcell.is_marked(),'Quadcell should be marked.')
        self.assertFalse(sw_child.is_marked(), 'SW child should not be marked.')
        self.assertFalse(sw_sw_child.is_marked(), 'SW-SW child should not be marked.')
        
        # Restore 
        sw_child.mark()
        sw_sw_child.mark()
        
        #
        # Unmark all 
        # 
        sw_child.get_root().unmark(recursive=True)
        self.assertFalse(qcell.is_marked(),'Quadcell should not be marked.')
        self.assertFalse(sw_child.is_marked(), 'SW child should not be marked.')
        self.assertFalse(sw_sw_child.is_marked(), 'SW-SW child should not be marked')
       
    
    def test_unit_normal(self):
        box = [0.,1.,0.,1.]
        qc = QuadCell(corner_vertices=box)
        ew = qc.get_edges('W')
        ee = qc.get_edges('E')
        es = qc.get_edges('S')
        en = qc.get_edges('N')
        self.assertEqual(np.sum(np.array([-1.,0])-qc.unit_normal(ew)),0.0, 
                         'Unit normal should be [-1,0].')
        self.assertEqual(np.sum(np.array([1.,0])-qc.unit_normal(ee)),0.0, 
                         'Unit normal should be [1,0].')
        self.assertEqual(np.sum(np.array([0.,-1.])-qc.unit_normal(es)),0.0, 
                         'Unit normal should be [0,-1].')
        self.assertEqual(np.sum(np.array([0.,1.])-qc.unit_normal(en)),0.0, 
                         'Unit normal should be [0,1].')

    
        cnr = [Vertex((0,0)), Vertex((3,1)), Vertex((2,3)), Vertex((-1,1))]
        cell = QuadCell(corner_vertices=cnr)
        edge = cell.edges[('SW','SE')]
        self.assertTrue(np.allclose(cell.unit_normal(edge), 
                                    1/np.sqrt(10)*np.array([1,-3])), 
                        'Unit normal should be [1,-3]/sqrt(10)')
        
    
    def test_contains_point(self):
        v_sw = Vertex((0,0))
        v_se = Vertex((3,1))
        v_ne = Vertex((2,3))
        v_nw = Vertex((-1,1))
        cell = QuadCell(corner_vertices=[v_sw, v_se, v_ne, v_nw])
        
        points = [(1,1),(3,0),(-0.5,0.5)]
        in_cell = cell.contains_point(points)
        
        self.assertTrue(in_cell[0],'Point (1,1) lies in cell.')
        self.assertFalse(in_cell[1], 'Point (3,0) lies outside cell.')
        self.assertTrue(in_cell[2], 'Point (-0.5,0.5) lies on cell boundary.')
    
    
    def test_intersects_line_segment(self):
        v_sw = Vertex((0,0))
        v_se = Vertex((3,1))
        v_ne = Vertex((2,3))
        v_nw = Vertex((-1,1))
        cell = QuadCell(corner_vertices=[v_sw, v_se, v_ne, v_nw])
        
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
   
   
    def test_reference_map(self):
        v_sw = Vertex((0,0))
        v_se = Vertex((3,1))
        v_ne = Vertex((2,3))
        v_nw = Vertex((-1,1))
        cell = QuadCell(corner_vertices=[v_sw, v_se, v_ne, v_nw])
        
        #
        # Map corner vertices of reference cell to physical vertices
        #
        y_refs = np.array([[0,0],[1,0],[1,1],[0,1]])
        x = list(cell.get_vertices(pos='corners', as_array=True))
        x_phys = cell.reference_map(list(y_refs))
        self.assertTrue(np.allclose(np.array(x),x_phys),\
                        'Mapped vertices should coincide '+\
                        'with physical cell vertices.')
        
        #
        # Jacobian: Area of cell by integration
        #  
        rule_2d = GaussRule(order=4, shape='quadrilateral')
        r = rule_2d.nodes()
        wg = rule_2d.weights()
        _, jac = cell.reference_map(list(r), jacobian=True)
        area = 0
        for i in range(4):
            j = jac[i]
            w = wg[i]
            area += np.abs(np.linalg.det(j))*w
        self.assertAlmostEqual(cell.area(), area, 7,\
                               'Area computed via numerical quadrature '+\
                               'not close to actual area')
        #
        # Try different formats
        # 
        # Array
        x = np.array(x)
        x_ref = cell.reference_map(x, mapsto='reference')
        self.assertTrue(np.allclose(y_refs, np.array(x_ref)),\
                        'Map array to reference: incorrect output.')
        # Single point
        x = x[0,:]
        x_ref = cell.reference_map(x, mapsto='reference')
        self.assertTrue(np.allclose(x, x_ref))
        
        #
        # Map corner vertices to reference points
        #
        x = cell.get_vertices(pos='corners', as_array=True)
        y = cell.reference_map(x, mapsto='reference')
        self.assertTrue(np.allclose(y, y_refs), \
                        'Corner vertices should map '+\
                        'onto (0,0),(1,0),(1,1),(0,1).')
         
        #
        # Map random points in [0,1]^2 onto cell and back again
        # 
        # Generate random points
        t = np.random.rand(5)
        s = np.random.rand(5)
        x = np.array([s,t]).T
        
        # Map to physical cell
        x_phy = cell.reference_map(x)
        
        # Check whether points are contained in cell
        in_cell = cell.contains_point(x_phy)
        self.assertTrue(all(in_cell), \
                        'All points mapped from [0,1]^2 '+\
                        'should be contained in the cell.')
        
        # Map back to reference cell
        x_ref = cell.reference_map(x_phy, mapsto='reference')
        self.assertTrue(np.allclose(np.array(x_ref), np.array(x)),\
                        'Points mapped to physical cell and back should '+\
                        'be unchanged.')
        

        #
        # Compute the hessian and compare with finite difference approximation 
        #
        h = 1e-8
        x = np.array([[0.5, 0.5],[0.5+h,0.5],[0.5-h,0.5],
                      [0.5,0.5+h],[0.5,0.5-h]])
        
        x_ref, J, H  = cell.reference_map(x, mapsto='reference', 
                                          hessian=True, jacobian=True)
        
        # sxx
        sxx_fd = (J[1][0,0]-J[2][0,0])/(2*h)
        sxx    = H[0][0,0,0]  
        self.assertAlmostEqual(sxx_fd, sxx, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')
        
        
        # syx
        syx_fd = (J[1][0,1]-J[2][0,1])/(2*h)
        sxy    = H[0][0,1,0]
        syx    = H[0][1,0,0]
        self.assertAlmostEqual(sxy, syx, 7, 'Mixed derivatives not equal.')
        self.assertAlmostEqual(syx_fd, sxy, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')
        
        # syy
        syy_fd = (J[3][0,1]-J[4][0,1])/(2*h)
        syy    = H[0][1,1,0]
        self.assertAlmostEqual(syy_fd, syy, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')

        # txx
        txx_fd = (J[1][1,0]-J[2][1,0])/(2*h)
        txx = H[0][0,0,1]
        self.assertAlmostEqual(txx_fd, txx, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')
        
        # txy
        txy_fd = (J[3][1,0]-J[4][1,0])/(2*h)
        txy    = H[0][0,1,1]
        tyx    = H[0][1,0,1]
        self.assertAlmostEqual(txy, tyx, 7, 'Mixed derivatives not equal.')
        self.assertAlmostEqual(txy_fd, txy, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')
        
        # tyy
        tyy_fd = (J[3][1,1]-J[4][1,1])/(2*h)
        tyy    = H[0][1,1,1] 
        self.assertAlmostEqual(tyy_fd, tyy, 7, \
                               'Hessian calculation not close to '+\
                               'finite difference approximation')
        
        
class TestEdge(unittest.TestCase):
    """
    Test Edge Class
    """
    pass


class TestVertex(unittest.TestCase):
    """
    Test Vertex Class
    """
    def test_dim(self):
        pass
        
        
 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testNode']
    unittest.main()