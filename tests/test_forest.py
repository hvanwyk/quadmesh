import unittest
from mesh import Tree, Forest
import numpy as np

class TestForest(unittest.TestCase):
    """
    Test Forest Class
    """
    def test_constructor(self):
         
        t0 = Tree(2)
        
        # Check whether all entries are Trees
        self.assertRaises(Exception, Forest, **{'trees':[t0,0]})
        
        # Check whether Trees are roots
        t0.split()
        t00 = t0.get_child(0)
        self.assertRaises(Exception, Forest, **{'trees':[t0,t00]})
        
        
        t1 = Tree(4)
        forest = Forest([t0,t1])
        self.assertEqual(len(forest._trees),2) 
    
    
    def test_depth(self):
        #
        # Generate forest
        # 
        forest = Forest([Tree(2), Tree(2)])
        #
        # Split tree 0 three times
        # 
        tree = forest.get_child(0)
        for _ in range(3):
            tree.split()
            tree = tree.get_child(0)
            
        # Check that depth is 3
        self.assertEqual(forest.depth(), 3)
        
        # Remove split tree and verify that depth is 0 
        forest.remove_tree(0)
        self.assertEqual(forest.depth(), 0)
    
    
    def test_traverse(self):
        #
        # Binary Tree
        #  
        # Standard
        node = Tree(2)
        forest = Forest([node])
        
        node.split()
        node.get_child(0).split()
        node.get_child(0).get_child(1).remove()
        addresses = {'breadth-first': [[0],[0,0],[0,1],[0,0,0]], 
                     'depth-first': [[0],[0,0],[0,0,0],[0,1]]}
 
        for mode in ['depth-first','breadth-first']:
            count = 0
            for leaf in forest.traverse(mode=mode):
                self.assertEqual(leaf.get_node_address(), addresses[mode][count]),
                                 
                count += 1 
        #
        # QuadTree
        # 
        node = Tree(4)
        forest = Forest([node])
        node.split()
        node.get_child(1).split()
        node.get_child(1).get_child(2).remove()
        addresses = [[0],[0,0],[0,1],[0,2],[0,3],[0,1,0],[0,1,1],[0,1,3]]
        count = 0
        for n in node.traverse(mode='breadth-first'):
            self.assertEqual(n.get_node_address(), addresses[count],\
                             'Incorrect address.')
            count += 1
    
        #
        # Forest with one quadtree and one bitree
        # 
        bi = Tree(2)
        quad = Tree(4)
        forest = Forest([bi, quad])    
        bi.split()
        bi.get_child(0).split()
        quad.split()
        addresses = {'breadth-first': [[0],[1],[0,0],[0,1],[1,0],[1,1],[1,2],[1,3],[0,0,0],[0,0,1]], 
                     'depth-first': [[0],[0,0],[0,0,0],[0,0,1],[0,1],[1],[1,0],[1,1],[1,2],[1,3]]}
 
        for mode in ['depth-first','breadth-first']:
            count = 0
            for leaf in forest.traverse(mode=mode):
                self.assertEqual(leaf.get_node_address(), addresses[mode][count])
                
                count += 1 
 
        
    def test_get_leaves(self):
        #
        # 1D
        # 
        node = Tree(2) 
        forest = Forest([node])
        leaves = forest.get_leaves()
  
        # Only a ROOT node, it should be the only LEAF        
        self.assertEqual(leaves, [node], 'Cell should be its own leaf.')
        
        #
        # Split cell and L child - find leaves
        # 
        node.split()
        l_child = node.get_child(0)
        l_child.split()
        leaves = forest.get_leaves()
        self.assertEqual(len(leaves),3, 'Cell should have 3 leaves.')
        
        
        #
        # Depth first order
        # 
        addresses_depth_first = [[0,0,0],[0,0,1],[0,1]]
        leaves = forest.get_leaves(nested=False)
        for i in range(len(leaves)):
            leaf = leaves[i]
            self.assertEqual(leaf.get_node_address(), addresses_depth_first[i],
                             'Incorrect order, depth first search.')
        #
        # Breadth first order
        # 
        addresses_breadth_first = [[0,1],[0,0,0],[0,0,1]]
        leaves = node.get_leaves(nested=True)
        for i in range(len(leaves)):
            leaf = leaves[i]
            self.assertEqual(leaf.get_node_address(), addresses_breadth_first[i],
                             'Incorrect order, breadth first search.')
        
        
        node.get_child(0).get_child(0).mark('1')
        node.get_child(1).mark('1')
        leaves = node.get_leaves(flag='1', nested='True')
        self.assertEqual(len(leaves),2, \
                         'There should only be 2 flagged leaves')
        
        #
        # 2D
        # 
        node = Tree(4)
        forest = Forest([node])
        
        #
        # Split cell and SW child - find leaves
        # 
        node.split()
        sw_child = node.get_child(0)
        sw_child.split()
        leaves = node.get_leaves()
        self.assertEqual(len(leaves), 7, 'Node should have 7 leaves.')
        
        #
        # Nested traversal
        #
        leaves = node.get_leaves(nested=True)
        self.assertEqual(leaves[0].get_node_address(),[0,1], \
            'The first leaf in the nested enumeration should have address [1]')
        
        leaves = node.get_leaves()
        self.assertEqual(leaves[0].get_node_address(), [0,0,0], \
                         'First leaf in un-nested enumeration should be [0,0].')
        
        #
        # Merge SW child - find leaves
        # 
        sw_child.delete_children()
        
        leaves = node.get_leaves()
        self.assertEqual(len(leaves), 4, 'Node should have 4 leaves.')
        
        
        #
        # Marked Leaves
        # 
        node = Tree(4)
        node.mark(1)
        forest = Forest([node])
        self.assertTrue(node in forest.get_leaves(flag=1), \
                        'Node should be a marked leaf node.')
        self.assertTrue(node in forest.get_leaves(), \
                        'Node should be a marked leaf node.')
    
        node.split()
        sw_child = node.get_child(0)
        sw_child.split()
        sw_child.mark(1)
        self.assertEqual(node.get_leaves(flag=1), \
                         [sw_child], 'SW child should be only marked leaf')
        
        sw_child.remove()
        self.assertEqual(forest.get_leaves(flag=1), \
                         [node], 'node should be only marked leaf')
        
        #
        # Nested traversal
        # 
        node = Tree(4)
        node.split()
        forest = Forest([node])
        for child in node.get_children():
            child.split()
            
        node.get_child(1).mark(1, recursive=True)
        node.get_child(3).mark(1)
        
        leaves = forest.get_leaves(nested=True, flag=1)
        self.assertEqual(len(leaves), 5, 
                         'This tree has 5 flagged LEAF nodes.')
        self.assertEqual(leaves[0], node.get_child(3), 
                         'The first leaf should be the NE child.')
        self.assertEqual(leaves[3], node.get_child(1).get_child(2),
                         '4th flagged leaf should be SE-NW grandchild.')
        
    
    def test_find_node(self):
        t0 = Tree(2)
        t1 = Tree(3)
        forest = Forest([t0,t1])
        
        t0.split()
        t00 = t0.get_child(0)
        t00.split()
        t001 = t00.get_child(1)
        
        self.assertEqual(forest.find_node([0,0,1]),t001)
        self.assertIsNone(forest.find_node([4,5,6]))
        
    
    def test_has_trees(self):
        # Empty forest has no trees
        forest = Forest()
        self.assertFalse(forest.has_children())
        
        # Forest with two trees 
        t0 = Tree(2)
        t1 = Tree(3)
        forest = Forest([t0, t1])
        self.assertTrue(forest.has_children())
        self.assertFalse(forest.has_children(flag=0))
        
        # Mark one tree  
        t0.mark(0)
        self.assertTrue(forest.has_children(flag=0))
        self.assertFalse(forest.has_children(flag=1))
        
        
    def test_child(self):
        forest = Forest()
        self.assertRaises(Exception, forest.get_child, *(0,))
        forest = Forest([Tree(1),Tree(4)])
        self.assertEqual(forest.get_child(1).n_children(),4)
    
    def test_get_children(self):
        forest = Forest()
        self.assertEqual(forest.get_children(),[])
        
        forest = Forest([Tree(1), Tree(3), Tree(10)])
        n_children = [1,3,10]
        i = 0
        for child in forest.get_children():
            self.assertEqual(child.get_node_position(),i)
            self.assertEqual(child.n_children(), n_children[i])
            i += 1
            
        
    def test_add_remove_tree(self):
        forest = Forest()
        self.assertEqual(forest.n_children(),0)
        node = Tree(3)
        forest.add_tree(node)
        self.assertEqual(forest.n_children(),1)
        self.assertRaises(Exception, forest.add_tree, *(1,))
        forest.remove_tree(0)
        self.assertEqual(forest.n_children(),0)
    
    
    def test_record(self):
        forest = Forest([Tree(2),Tree(2),Tree(2)])
        for _ in range(5):
            for child in forest.get_children():
                if np.random.rand() > 0.5:
                    child.split()
        forest.record(1)
        for tree in forest.traverse():
            self.assertTrue(tree.is_marked(1))
    
    
    def test_coarsen(self):
        # Define a new forest with two quadtrees
        forest = Forest([Tree(4), Tree(4)])
        
        # Refine and coarsen again
        forest.refine()
        forest.coarsen()
        
        # Check that forest is as it was
        count = 0
        for _ in forest.traverse():
            count += 1
        self.assertEqual(count, 2)
 
        # Coarsen forest - there should be no change        
        forest.coarsen()
        
        # Check that forest is as it was
        count = 0
        for _ in forest.traverse():
            count += 1
        self.assertEqual(count, 2)
        
        # Refine and mark one child 
        forest.refine()
        
        # Check that forest is as it was
        count = 0
        for _ in forest.traverse():
            count += 1
        self.assertEqual(count, 10)
        
        forest.get_child(0).get_child(0).mark(1)
        forest.coarsen(label=1)
        
        # Check that forest is as it was
        count = 0
        for _ in forest.traverse():
            count += 1
        self.assertEqual(count, 10)
        
        # Nothing is marked 1
        self.assertFalse(any(child.is_marked(1) for child in forest.traverse()))
    
        # Now actually delete nodes marked 1 
        forest.get_child(0).mark(1)
        forest.coarsen(coarsening_flag=1)
        
        # Check that one node deleted its children
        count = 0
        for _ in forest.traverse():
            count += 1
        self.assertEqual(count, 6)
        
        
    def test_refine(self):
        # Define a new forest with two binary trees
        forest = Forest([Tree(2), Tree(2)])
        
        # Refine the forest indiscriminantly
        forest.refine()
        
        # Check wether the trees have been split
        count = 0
        for _ in forest.traverse():
            count += 1
        self.assertEqual(count, 6)
        
        # Mark second tree and refine only by its label
        tree = forest.get_child(1)
        tree.mark(1)
        forest.refine(label=1)
        
        # Check that all the tree's children are labeled 1 
        self.assertTrue(all([child.is_marked(1) for child in tree.get_children()]))
        
        # Other node still not labeled
        self.assertFalse(forest.get_child(0).is_marked(1))
        should_be_false = any([gchild.is_marked(1) for gchild in \
                               forest.get_child(0).get_children()])
        self.assertFalse(should_be_false)
        
        # Check whether forest has expanded - it shouldn't have
        count = 0
        for _ in forest.traverse():
            count += 1
        self.assertEqual(count, 6)
        
        # Now add another marker to identify nodes to be refined
        tree.get_child(1).mark(2)
        forest.refine(label=1, refinement_flag=2)
        
        # Check the depth
        self.assertEqual(tree.tree_depth(),2)
        
        # Check that all tree's progeny are marked 1.
        for node in tree.traverse():
            self.assertTrue(node.is_marked(1))
        
        # Check whether forest has expanded - it should have
        count = 0
        for _ in forest.traverse():
            count += 1
        self.assertEqual(count, 8)
        
        # Try to refine a tree that is not labeled 1
        forest.get_child(0).mark(3)
        forest.refine(label=1, refinement_flag=3)
        
        # Check whether forest has expanded - it shouldn't have
        count = 0
        for _ in forest.traverse():
            count += 1
        self.assertEqual(count, 8)
        
        # Check that child(0) is not marked
        for child in forest.get_child(0).traverse():
            self.assertFalse(child.is_marked(1))