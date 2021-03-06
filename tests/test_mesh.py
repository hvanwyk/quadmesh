'''
Created on Oct 23, 2016

@author: hans-werner
'''
import unittest
from mesh import Mesh, Node, BiCell, QuadCell, TriCell, Edge, Vertex
from plot import Plot
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class TestMesh(unittest.TestCase):
    """
    Test Mesh Class
    """
    def test_mesh_constructor(self):
        # =============================
        # Using given QuadCell and Node
        # =============================
        #
        # Simple
        # 
        node = Node()
        box = [0.,1.,0.,1.] 
        quadcell = QuadCell(box=box)
        mesh = Mesh(quadcell=quadcell, root_node=node)
        self.assertTrue(mesh.root_node()==node,\
                        'Node not associated with mesh.')
        self.assertTrue(mesh.root_node().quadcell()==quadcell, \
                        'QuadCell not associated with mesh.')
        
        #
        # Node and Quadcell incompatible
        # 
        node = Node(grid_size=(1,2))
        quadcell = QuadCell(box=box, grid_size=(2,2))
        self.assertRaises(AssertionError, Mesh, \
                          quadcell=quadcell, root_node=node)
        
        # ==============================
        # Using Mesh.newmesh()
        # ==============================
        mesh = Mesh.newmesh(box=[0.,2.,0.,1.], grid_size=(2,1))
        right_box = mesh.root_node().quadcell().box()
        self.assertEqual(right_box,(0.,2.,0.,1.),'Box should be (0,2,0,1).')

        
        # ==============================
        # Using Mesh.submesh()
        # ==============================
        # TODO: Unfinished - perhaps it's better to work with labels
        mesh.root_node().mark()
        mesh.refine()
        #print(mesh.root_node().children.keys())
        mesh.root_node().children[(1,0)].mark()
        mesh.refine()
        #print(mesh.root_node().children.keys())
        #print(mesh.root_node().children[(1,0)].children.keys())
        #submesh = Mesh.submesh(mesh)
        
        #_,ax = plt.subplots()
        #submesh.plot_quadmesh(ax)
        #mesh.plot_quadmesh(ax)
        #
        # Make sure submesh is independent
        # 
        mesh.unmark(nodes=True)
        mesh.root_node().children[(1,0)].mark()
        mesh.coarsen()
    
        
    def test_mesh_box(self):
        pass
    
    
    def test_mesh_grid_size(self):
        pass 
    
    
    def test_mesh_depth(self):
        pass
    
    
    def test_mesh_get_number_of_cells(self):
        pass
    
        
    def test_mesh_root_node(self):
        pass
    
    
    def test_mesh_boundary(self):
        # TODO: Finish
        """
        mesh = Mesh.newmesh()
        mesh.refine()
        mesh.root_node().children['SW'].remove()
        mesh.root_node().info()
        for leaf in mesh.root_node().find_leaves():
            leaf.info()
        
        fig, ax = plt.subplots()
        plot = Plot()
        plot.mesh(ax,mesh)
        plt.show()
        print(len(mesh.boundary('vertices')))
        print(len(mesh.boundary('edges')))
        print(len(mesh.boundary('quadcells')))
        """

    def test_mesh_tree_structure(self):
        pass
    
    
    def test_mesh_quadcell(self):
        pass
    
    
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
            for leaf in mesh.root_node().find_leaves():
                if np.random.rand() < 0.5:
                    leaf.mark(1)
            mesh.refine(1)
        #print('Before balancing', mesh.is_balanced())
        mesh.balance()
        #print('After balancing', mesh.is_balanced())
    
    
    def test_mesh_is_balanced(self):
        pass
    
        
    def test_mesh_refine(self):
        pass
    
    
    def test_coarsen(self):
        pass
    
    
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
        for node in mesh.root_node().find_leaves(1):
            node.info()
        print("COARSE MESH")
        for node in mesh.root_node().find_leaves(0):
            node.info()
        """
        

    
    def test_mesh_plot_trimesh(self):
        pass
    
     
class TestNode(unittest.TestCase):
    """
    Test Node Class
    """

    def test_node_constructor(self):
        #
        # Children standard
        #
        node = Node()
        self.assertEqual(node.depth, 0, 'Node depth should be zero.')
        self.assertTrue(node.type=='ROOT', 'Node should be of type ROOT.')
        generic_children = {'SW':None, 'SE':None, 'NE':None, 'NW':None}
        self.assertEqual(node.children, generic_children, 'Incorrect form for children.')
        self.assertEqual(node.grid_size(), None, 'Child grid size should be None.')
        #
        # Children in grid
        # 
        node = Node(grid_size=(2,2))
        self.assertEqual(node.grid_size(),(2,2), 'Child grid size should be (2,2).')
    
    
    def test_node_copy(self):
        node = Node(grid_size=(2,1))
        node.split()
        e_child = node.children[(1,0)]
        e_child.split()
        e_ne_child = e_child.children['NE']
        e_ne_child.split()
        cnode = node.copy()    
        
        
    def test_node_is_gridded(self):
        pass
    
    
    def test_node_grid_size(self):
        pass
    
    
    def test_node_find_neighbor(self):
        node = Node()
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        nw_grandchild = sw_child.children['NW']
        ne_grandchild = sw_child.children['NE']
        #
        # Neighbor exterior to parent cell
        #  
        self.assertEqual(nw_grandchild.find_neighbor('N'), node.children['NW'], 
                         'Neighbor should be NW child of ROOT cell.')
        self.assertEqual(ne_grandchild.find_neighbor('NE'), node.children['NE'], 
                         'Neighbor should be NE child of ROOT cell.')
        #
        # Neighbor is sibling cell
        #  
        self.assertEqual(nw_grandchild.find_neighbor('S'), sw_child.children['SW'], 
                         'Neighbor should be SW sibling.')
        self.assertEqual(nw_grandchild.find_neighbor('SE'),sw_child.children['SE'],
                         'Neighbor should be SE sibling.')
        #
        # Neighbor is None
        # 
        self.assertEqual(nw_grandchild.find_neighbor('W'), None, 
                         'Neighbor should be None.')
        self.assertEqual(nw_grandchild.find_neighbor('NE'),None,
                         'Neighbor should be None.')
        
        node.children['NW'].split()
        self.assertEqual(nw_grandchild.find_neighbor('NE'),
                         node.children['NW'].children['SE'],
                         'Neighbor should be the NW-SE grandchild.')
    
        
    def test_node_traverse_tree(self):
        pass
    
    
    def test_traverse_depthwise(self):
        #
        # Standard Node
        # 
        node = Node()
        node.split()
        node.children['SE'].split()
        node.children['SE'].children['NW'].remove()
        addresses = [[],[0],[1],[2],[3],[1,0],[1,1],[1,3]]
        count = 0
        for n in node.traverse_depthwise():
            self.assertEqual(n.address, addresses[count],\
                             'Incorrect address.')
            count += 1
         
        #
        # Gridded Node
        #     
        node = Node(grid_size=(3,3))
        node.split()
        addresses = [[]]
        for j in range(3):
            for i in range(3):
                addresses.append([(i,j)])
        count = 0
        for n in node.traverse_depthwise():
            self.assertEqual(n.address, addresses[count],\
                             'Incorrect address.')
            count += 1
            
        
       
            
    def test_node_find_leaves(self):
        #
        # Single node
        # 
        node = Node()
        leaves = node.find_leaves()
        self.assertEqual(leaves, [node], 'Node should be its own leaf.')
        
        #
        # Split cell and SW child - find leaves
        # 
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        leaves = node.find_leaves()
        self.assertEqual(len(leaves), 7, 'Node should have 7 leaves.')
        
        #
        # Nested traversal
        #
        leaves = node.find_leaves(nested=True)
        self.assertEqual(leaves[0].address,[1], \
            'The first leaf in the nested enumeration should have address [1]')
        
        leaves = node.find_leaves()
        self.assertEqual(leaves[0].address, [0,0], \
                         'First leaf in un-nested enumeration should be [0,0].')
        
        #
        # Merge SW child - find leaves
        # 
        sw_child.merge()
        leaves = node.find_leaves()
        self.assertEqual(len(leaves), 4, 'Node should have 4 leaves.')
    
        
        
        
    def test_node_find_marked_leaves(self):
        node = Node()
        node.mark(1)
        self.assertTrue(node in node.find_leaves(flag=1), \
                        'Node should be a marked leaf node.')
        self.assertTrue(node in node.find_leaves(), \
                        'Node should be a marked leaf node.')
    
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        sw_child.mark(1)
        self.assertEqual(node.find_leaves(flag=1), \
                         [sw_child], 'SW child should be only marked leaf')
        
        sw_child.remove()
        self.assertEqual(node.find_leaves(flag=1), \
                         [node], 'node should be only marked leaf')
        
        #
        # Nested traversal
        # 
        node = Node()
        node.split()
        for child in node.get_children():
            child.split()
            
        node.children['SE'].mark(1, recursive=True)
        node.children['NE'].mark(1)
        
        leaves = node.find_leaves(nested=True, flag=1)
        self.assertEqual(len(leaves), 5, 
                         'This tree has 5 flagged LEAF nodes.')
        self.assertEqual(leaves[0], node.children['NE'], 
                         'The first leaf should be the NE child.')
        self.assertEqual(leaves[3], node.children['SE'].children['NW'],
                         '4th flagged leaf should be SE-NW grandchild.')
        
        
    def test_node_find_root(self):
        node = Node()
        self.assertEqual(node.get_root(), node, 'Node is its own root.')
        
        node.split()
        sw_child = node.children['SW']
        self.assertEqual(sw_child.get_root(), node, 'Node is its childs root.')
    
    
    def test_node_find_node(self):
        node = Node()
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
        node = Node(grid_size=(2,2))
        node.split()
        lb_child = node.children[(0,0)]
        lb_child.split()
        address = [(0,0),3]
        lb_ne_grandchild = lb_child.children['NE']
        self.assertEqual(node.find_node(address), lb_ne_grandchild, \
                         'Left bottom, NE grandchild has address [(0,0),2].')
        
        
    def test_node_has_children(self):
        node = Node()
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
        mesh = Mesh.newmesh()
        mesh.refine()
        node = mesh.root_node()
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
        node = Node()
        for child in node.get_children():
            print('Hallo')
            
        #
        # Try a logical statement 
        # 
        self.assertFalse(any([child.is_marked(1) for \
                              child in node.get_children()]), \
                         'No marked children because there are none.')
        
        
    def test_node_has_parent(self):
        node = Node()
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
        
        
    def test_node_get_parent(self):
        node = Node()
        node.mark(1)
        node.split()
        sw_child = node.children['SW']
        sw_child.split()
        self.assertEqual(node,sw_child.children['NE'].get_parent(1),\
                         'First marked ancestor should be node.')     
        sw_child.mark(1)
        self.assertEqual(sw_child,sw_child.children['NE'].get_parent(1),\
                         'First marked ancestor should be sw_child.')
        
        
        
    def test_node_in_grid(self):
        #
        # Standard positioning
        # 
        node = Node()
        node.split()
        sw_child = node.children['SW']
        self.assertFalse(sw_child.in_grid(), 'Child is not in grid.')
        
        #
        # Grid positioning
        # 
        node = Node(grid_size=(2,2))
        node.split()
        lb_child = node.children[(0,0)]
        self.assertTrue(lb_child.in_grid(), 'Child lives in grid.')
    
    def test_node_is_marked(self):
        node = Node()
        node.mark()
        self.assertTrue(node.is_marked(),'Node should be marked.')
        node.unmark()
        self.assertFalse(node.is_marked(),'Node should not be marked.')
        
        
    def test_node_mark(self):
        node = Node()
        node.mark()
        self.assertTrue(node.is_marked(),'Node should be marked.')
    
    
    def test_node_mark_support(self):
        pass
    
    
    def test_node_unmark(self):
        #
        # 3 Generations of marked nodes
        # 
        node = Node()
        node.mark()
        node.split()
        sw_child = node.children['SW']
        sw_child.mark()
        sw_child.split()
        sw_sw_child = sw_child.children['SW']
        sw_sw_child.mark()
        
        #
        # Unmark sw_child node
        #
        sw_child.unmark()
        self.assertTrue(node.is_marked(), 'Node should still be marked.')
        self.assertFalse(sw_child.is_marked(),'SW child should be unmarked.')
        self.assertTrue(sw_sw_child.is_marked(),'SWSW grandchild should be marked.')
        
        # Reset
        sw_child.mark()
        
        #
        # Unmark recursively
        # 
        sw_child.unmark(recursive=True)
        self.assertTrue(node.is_marked(), 'Node should still be marked.')
        self.assertFalse(sw_child.is_marked(),'SW child should be unmarked.')
        self.assertFalse(sw_sw_child.is_marked(),'SWSW grandchild should be unmarked.')
        
        # Reset
        sw_sw_child.mark()
        sw_child.mark()
        
        #
        # Unmark all
        # 
        node.unmark(recursive=True)
        self.assertFalse(node.is_marked(), 'Node should still be marked.')
        self.assertFalse(sw_child.is_marked(),'SW child should be unmarked.')
        self.assertFalse(sw_sw_child.is_marked(),'SWSW grandchild should be marked.')
    
    
    def test_node_is_linked(self):
        pass
    
    
    def test_node_link(self):
        pass
    
    
    def test_node_unlink(self):
        pass
    

    def test_quadcell(self):
        pass
    
    
    def add_tricells(self):
        pass
            
            
    def has_tricells(self):
        pass
    
        
    def test_node_merge(self):
        node = Node()
        node.split()
        node.merge()
        self.assertFalse(node.has_children(),'Node should not have children.')
    
        node = Node(grid_size=(2,2))
        node.split()
        node.merge()
        self.assertFalse(node.has_children(),'Node should not have children.')
        
        
    def test_node_remove(self):
        node = Node()
        node.split()
        node.children['SW'].remove()
        self.assertEqual(node.children['SW'],None,'Node should have been removed.')
        
        node = Node(grid_size=(2,2))
        node.split()
        node.children[(0,0)].remove()
        self.assertEqual(node.children[(0,0)],None, 'Node should have been removed.')
    
    
    def test_node_split(self):
        node = Node()
        node.split()
        self.assertTrue(node.has_children(),'Split node should have children.')
        
            
    def test_node_is_balanced(self):
        node = Node()
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
        node = Node()
        node.split()
        # Split node arbitrarily (most likely not balanced)
        for _ in range(3):
            for leaf in node.find_leaves():
                if np.random.rand() < 0.5:
                    leaf.split()
        node.balance()
        self.assertTrue(node.is_balanced(),'Node should be balanced.')
        """
        Debugging: 
        
        if not node.is_balanced():
            for leaf in node.find_leaves():
                for direction in ['N','S','E','W']:
                    nb = leaf.find_neighbor(direction)
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
        
        
    def test_node_balance(self):
        node = Node()
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
        
    
    def test_node_remove_supports(self):
        #
        # Split and balance
        # 
        node = Node()
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
        
        
    def test_node_pos2id(self):
        node = Node()
        self.assertEqual(node.pos2id('SW'),0,'sw -> 0.')
        self.assertEqual(node.pos2id('SE'),1,'se -> 1.')
        self.assertEqual(node.pos2id('NW'),2,'nw -> 2.')
        self.assertEqual(node.pos2id('NE'),3,'ne -> 3.')
        
        self.assertEqual(node.pos2id((0,0)),(0,0),'(0,0) -> (0,0).')       
        self.assertRaises(Exception, node.pos2id, [0,0])
        
    def test_node_id2pos(self):
        node = Node()
        self.assertEqual(node.id2pos(0),'SW','sw <- 0.')
        self.assertEqual(node.id2pos(1),'SE','se <- 1.')
        self.assertEqual(node.id2pos(2),'NW','nw <- 2.')
        self.assertEqual(node.id2pos(3),'NE','ne <- 3.')
        
        self.assertEqual(node.id2pos((0,0)),(0,0),'(0,0) -> (0,0).')       
        self.assertRaises(Exception, node.id2pos, [0,0])


class TestBiNode(unittest.TestCase):
    """
    Test the BiNode subclass of Node
    """
    pass

    
class TestQuadNode(unittest.TestCase):
    """
    Test the QuadNode class/a subclass of Node
    """
    def test_is_balanced(self):
        pass
    
    def test_balance(self):
        pass



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
        # Standard Node
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
         
        #
        # Gridded Node
        #     
        cell = QuadCell(grid_size=(3,3))
        cell.split()
        addresses = [[]]
        for j in range(3):
            for i in range(3):
                addresses.append([(i,j)])
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
        leaves = cell.find_leaves()
        self.assertEqual(leaves, [cell], 'Cell should be its own leaf.')
        
        #
        # Split cell and L child - find leaves
        # 
        cell.split()
        l_child = cell.children['L']
        l_child.split()
        leaves = cell.find_leaves()
        self.assertEqual(len(leaves),3, 'Cell should have 3 leaves.')
        
        #
        # Depth first order
        # 
        addresses_depth_first = [[0,0],[0,1],[1]]
        leaves = cell.find_leaves(nested=False)
        for i in range(len(leaves)):
            leaf = leaves[i]
            self.assertEqual(leaf.address, addresses_depth_first[i],
                             'Incorrect order, depth first search.')
        #
        # Breadth first order
        # 
        addresses_breadth_first = [[1],[0,0],[0,1]]
        leaves = cell.find_leaves(nested=True)
        for i in range(len(leaves)):
            leaf = leaves[i]
            self.assertEqual(leaf.address, addresses_breadth_first[i],
                             'Incorrect order, breadth first search.')
        
        
        cell.children['L'].children['L'].mark('1')
        cell.children['R'].mark('1')
        leaves = cell.find_leaves(flag='1', nested='True')
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
        leaves = cell.find_leaves()
        self.assertEqual(len(leaves), 7, 'Node should have 7 leaves.')
        
        #
        # Nested traversal
        #
        leaves = cell.find_leaves(nested=True)
        self.assertEqual(leaves[0].address,[1], \
            'The first leaf in the nested enumeration should have address [1]')
        
        leaves = cell.find_leaves()
        self.assertEqual(leaves[0].address, [0,0], \
                         'First leaf in un-nested enumeration should be [0,0].')
        
        #
        # Merge SW child - find leaves
        # 
        for child in sw_child.get_children():
            child.remove()
        leaves = cell.find_leaves()
        self.assertEqual(len(leaves), 4, 'Node should have 4 leaves.')
        
    
    def test_cells_at_depth(self):
        pass
    
    
    def test_find_root(self):
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
    
    def test_box(self):
        bicell = BiCell()
        self.assertEqual((0,1),bicell.box(), \
                        'box incorrect: default bicell')
    
        bicell = BiCell(box=[-10,2])
        self.assertEqual((-10,2), bicell.box(), \
                         'box incorrect: box specified')
        
        self.assertRaises(AssertionError, BiCell, box =[2,-1])
        
        
    def test_find_neighbor(self):
        bicell = BiCell()
        self.assertIsNone(bicell.find_neighbor('L'), \
                          'neighbor should be None.')
        
        bicell.split()
        l_child = bicell.children['L']
        self.assertEqual(l_child.find_neighbor('R'), bicell.children['R'],\
                         'neighbor interior to parent cell not identified.')
        
        l_child.split()
        lr_grandchild = l_child.children['R']
        self.assertEqual(lr_grandchild.find_neighbor('R'), 
                         bicell.children['R'], 
                         'neighbor exterior to parent cell not identified.')
        
        bicell.children['R'].split()
        self.assertEqual(lr_grandchild.find_neighbor('R'),\
                         bicell.children['R'].children['L'],\
                         'neighbor exterior to parent cell not identified.')
        
        bicell = BiCell(grid_size=3)
        bicell.split()
        lchild = bicell.children[0]
        self.assertEqual(lchild.find_neighbor('L'),None,
                         'neighbor of gridded cell not identified as None.')
        
        self.assertEqual(lchild.find_neighbor('R'),bicell.children[1],
                         'neighbor of gridded cell not identified.')
        
        
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
    
    
    def test_map_to_reference(self):
        bicell = BiCell(box=[-2,10])
        self.assertTrue(np.allclose(bicell.map_to_reference([-2,10]),\
                                    np.array([0,1])), \
                        'Points incorrectly mapped to reference.')
    
    def test_map_from_reference(self):
        bicell = BiCell(box=[-2,10])
        self.assertTrue(np.allclose(bicell.map_from_reference([0,1]),\
                                    np.array([-2,10])),
                        'Points incorrectly mapped from reference.')
                        
    
    def test_derivative_multiplier(self):
        bicell = BiCell(box=[-2,10])
        self.assertEqual(bicell.derivative_multiplier(derivative=(2,)),
                         (1/12)**2,
                         'Derivative multiplier incorrect.')
    
    
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
        bicell = BiCell(grid_size=3)
        self.assertFalse(bicell.has_children(), 
                         'Bicell should not have children')
        bicell.split()
        count = 0
        for _ in bicell.get_children():
            count += 1
        self.assertEqual(count, 3, 'Bicell should have 3 children.')
    
    
    def test_pos2id(self):
        bicell = BiCell(grid_size=3)
        bicell.split()
        self.assertEqual(bicell.pos2id(0), 0, 
                         'Position in grid incorrectly converted.')
    
        
class TestQuadCell(unittest.TestCase):
    """
    Test QuadCell Class
    """
    def test_quadcell_constructor(self):
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
            nb = q2002.find_neighbor(direction)
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
        
    def test_quadcell_mark(self):
        box = [0.,1.,0.,1.]
        qcell = QuadCell(box=box)
        qcell.mark()
        self.assertTrue(qcell.is_marked(),'Quadcell should be marked.')

    def test_quadcell_unmark(self):
        #
        # 3 Generations of marked cells
        # 
        box = [0.,1.,0.,1.]
        qcell = QuadCell(box=box)
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
       
    
    def test_quadcell_normal(self):
        box = [0.,1.,0.,1.]
        qc = QuadCell(box=box)
        ew = qc.get_edges('W')
        ee = qc.get_edges('E')
        es = qc.get_edges('S')
        en = qc.get_edges('N')
        self.assertEqual(np.sum(np.array([-1.,0])-qc.normal(ew)),0.0, 
                         'Unit normal should be [-1,0].')
        self.assertEqual(np.sum(np.array([1.,0])-qc.normal(ee)),0.0, 
                         'Unit normal should be [1,0].')
        self.assertEqual(np.sum(np.array([0.,-1.])-qc.normal(es)),0.0, 
                         'Unit normal should be [0,-1].')
        self.assertEqual(np.sum(np.array([0.,1.])-qc.normal(en)),0.0, 
                         'Unit normal should be [0,1].')


          
class TestTriCell(unittest.TestCase):
    """
    Test TriCell Class
    """
    pass


class TestEdge(unittest.TestCase):
    """
    Test Edge Class
    """
    pass


class TestVertex(unittest.TestCase):
    """
    Test Vertex Class
    """
    pass
 
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testNode']
    unittest.main()