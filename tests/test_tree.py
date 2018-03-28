import unittest
from mesh import Tree, Forest

class TestTree(unittest.TestCase):
    """
    Test Tree class
    """
    def test_constructor(self):
        # Must at least specify number of children
        self.assertRaises(Exception, Tree)
        
        # Quadnode
        node = Tree(n_children=4)
        self.assertEqual(len(node._children),4,'There should be 4 children.')
        
        # Tree in a forest
        forest = Forest()
        node = Tree(forest=forest, n_children=2)
        
    
    def test_tree_info(self):
        '''
        # Define a new ROOT node
        node = Tree(n_children=4)
        node.info()
        node.split()
        child = node.get_child(0)
        child.info()
        child.split()
        child.info()
        child.get_child(0).info()
        '''
        pass
        
        
    def test_get_node_type(self):
        # Define new ROOT node
        node = Tree(n_children=2)
        self.assertEqual(node.get_node_type(),'ROOT',\
                         'Output "node_type" should be "ROOT".')
        
        # Split node and assert that its child is a LEAF
        node.split()
        child = node.get_child(0)
        self.assertEqual(node.get_node_type(),'ROOT',\
                         'Output "node_type" should be "ROOT".')
        self.assertEqual(child.get_node_type(),'LEAF',\
                         'Output "node_type" should be "LEAF".')
        
        # Split the child and assert that it is now a BRANCH
        child.split()
        self.assertEqual(child.get_node_type(),'BRANCH',\
                         'Output "node_type" should be "BRANCH".')
    
    
    def test_set_node_type(self):
        # Define a new node
        node = Tree(n_children=3)
        
        # Should complain about changing root to branch
        self.assertRaises(Exception, node.set_node_type, *('BRANCH'))
        
        # Split 
        node.split()
        child = node.get_child(0)
        self.assertRaises(Exception, child.set_node_type, *('ROOT'))
        self.assertRaises(Exception, child.set_node_type, *('BRANCH'))
        
    
    def test_get_node_address(self):
        pass
        
    
    def test_get_depth(self):
        # New node should have depth 0
        node = Tree(n_children=2)
        self.assertEqual(node.get_depth(),0,'ROOT node should have depth 0')
        
        # Split node 10 times
        for _ in range(10):
            node.split()
            node = node.get_child(1)
        # Last generation should have depth 10  
        self.assertEqual(node.get_depth(),10,'Node should have depth 10.')
        self.assertEqual(node.get_root().get_depth(), 0, \
                         'ROOT node should have depth 0')
    
    
    def test_tree_depth(self):
        # New node should have depth 0
        node = Tree(n_children=2)
        self.assertEqual(node.tree_depth(),0,\
                         'Tree should have depth 0')
        
        # Split node 10 times 
        for _ in range(10):
            node.split()
            node = node.get_child(1)
        
        # All nodes should have the same tree_depth    
        self.assertEqual(node.tree_depth(),10,\
                         'Tree should have depth 10.')
        self.assertEqual(node.get_root().tree_depth(),10,\
                         'Tree should have depth 10.')
    
    
    def test_in_forest(self):
        #
        # Initialize forest with node
        # 
        node = Tree(n_children=2)
        Forest([node])
        self.assertTrue(node.in_forest(),'Node should be in forest')
        #
        # Initialize empty forest
        # 
        node = Tree(n_children=2)
        forest = Forest()
        
        # Node should not be in there
        self.assertFalse(node.in_forest(), 'Node should not be in a forest.')
        
        # Add node: NOW it's in the forest.
        forest.add_tree(node)
        self.assertTrue(node.in_forest(),'Node should be in a forest.')
        
        # Remove node: it should no longer be there.
        forest.remove_tree(node.get_node_position())
        self.assertFalse(node.in_forest(),'Node should no longer be in the forest.')
        
        
    def test_is_regular(self):
        # Make non-regular tree
        node = Tree(regular=False, n_children=2)
        self.assertFalse(node.is_regular(), 'Node is not regular.')
        
        # Make regular tree
        node = Tree(n_children=3)
        self.assertTrue(node.is_regular(), 'Node should be regular.')
    
    
    def test_mark(self):
        node = Tree(n_children = 3)
        node.mark()
        self.assertTrue(node.is_marked(),'Node should be marked.')
    
        
    
    def test_unmark(self):
        #
        # 3 Generations of marked nodes
        # 
        pos1 = [0,0]
        pos2 = [0,0]
        count = 0
        for node in [Tree(n_children=2), Tree(n_children=4)]:
            #
            # Mark self and split
            #
            node.mark()
            node.split()
            #
            # Mark specific child and split
            #
            child = node.get_child(pos1[count])
            child.mark()
            child.split()
            #
            # Mark specific grandchild
            #
            grandchild = child.get_child(pos2[count])
            grandchild.mark()
            #
            # Unmark child node
            #
            child.unmark()
            self.assertTrue(node.is_marked(), \
                            'Node should still be marked.')
            self.assertFalse(child.is_marked(),\
                             'Child should be unmarked.')
            self.assertTrue(grandchild.is_marked(),\
                            'Grandchild should be marked.')
            #  
            # Reset
            #
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
            #
            # Reset
            #
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
    
    
    def test_is_marked(self):
        #
        # Mark new node (generic)
        # 
        node = Tree(n_children=4) 
        node.mark()
        self.assertTrue(node.is_marked(),'Node should be marked.')
        #
        # Unmark node
        # 
        node.unmark()
        self.assertFalse(node.is_marked(),'Node should not be marked.')
        #
        # Mark node (specific)
        # 
        node.mark(1)
        self.assertTrue(node.is_marked(),'Node should be marked.')
        self.assertTrue(node.is_marked(1),'Node should be marked 1.')
        self.assertFalse(node.is_marked(2),'Node should not be marked 2.')
        
    
    def test_has_parent(self):
        for node in [Tree(n_children=2), Tree(n_children=4)]:
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
        pos1 = [0,0]
        pos2 = [1,3]
        for node in [Tree(n_children=2), Tree(n_children=4)]:            
            node.mark(1)
            node.split()
            child = node.get_child(pos1[count])
            child.split()
            self.assertEqual(node,node.get_child(pos2[count]).get_parent(1),\
                             'First marked ancestor should be node.')     
            child.mark(1)
            self.assertEqual(child, child.get_child(pos2[count]).get_parent(1),\
                             'First marked ancestor should be child.')
            count += 1
    
    
    def test_get_root(self):
        root_node = Tree(n_children=3)
        node = root_node
        for _ in range(10):
            node.split()
            node = node.get_child(0)
            self.assertEqual(node.get_root(),root_node,\
                             'All children should have the same root node')
    
    
    def test_n_children(self):
        for n in range(10):
            #
            # Define regular node with n children
            # 
            node = Tree(n_children=n)
            
            # check if number of children correct
            self.assertEqual(node.n_children(),n,\
                             'Number of children incorrect')
            # split node and check if children inherit the same number of children
            node.split()
            for child in node.get_children():
                self.assertEqual(child.n_children(),n,\
                                 'Children have incorrect number of children.')
                
    
    def test_get_child(self):
        # New node
        node = Tree(n_children=5)
        node.split()
        
        # Access child directly and mark it 1
        child_4 = node._children[4]
        child_4.mark(1)
        
        # Access child via function
        child_4_v1 = node.get_child(4)
        
        # Check whether it's the same child. 
        self.assertTrue(child_4_v1.is_marked(1), 'Child 4 should be marked 1.')
        self.assertEqual(child_4_v1, child_4, 'Children should be the same.')
    
    
    def test_get_children(self):
        #
        # Binomial Tree
        # 
        node = Tree(n_children=2)
        node.split()
        count = 0
        pos = [0,1]
        for child in node.get_children():
            self.assertEqual(child.get_node_position(), pos[count],\
                             'Incorrect child.')
            count += 1
        
        # Reversed 
        count = 0
        for child in node.get_children(reverse=True):
            self.assertEqual(child.get_node_position(), pos[-1-count])
            count += 1
        
        # Flagged 
        child_0 = node.get_child(0)
        child_0.mark(1)
        for child in node.get_children(flag=1):
            self.assertEqual(child, child_0, \
                             'The only marked child is child_0')    
        #
        # Quadtree
        # 
        node = Tree(n_children=4)
        node.split()
        count = 0
        pos = [0,1,2,3]
        for child in node.get_children():
            self.assertEqual(child.get_node_position(), pos[count],\
                             'Incorrect child.')
            count += 1
            
        #
        # Now remove child
        # 
        node.delete_children(position=0)
        count = 0 
        for child in node.get_children():
            count += 1
        self.assertEqual(count, 3, 'There should only be 3 children left.')   
        #
        # Node with no children   
        # 
        node = Tree(n_children=4)
        for child in node.get_children():
            print('Hallo')
            
        #
        # Try a logical statement 
        # 
        self.assertFalse(any([child.is_marked(1) for \
                              child in node.get_children()]), \
                         'No marked children because there are none.')
          
        
    def test_remove(self):
        #
        # Remove child 2 
        #
        node = Tree(n_children=4)
        node.split() 
        child = node.get_child(2)
        child.remove()
        self.assertIsNone(node.get_child(2))
    
    
    def test_delete_children(self):
        #
        # Delete child 2 
        # 
        node = Tree(n_children=4)
        node.split()
        node.delete_children(2)
        self.assertIsNone(node.get_child(2))
        #
        # Delete all children
        # 
        node.delete_children()
        self.assertFalse(node.has_children())
    
    
    def test_split(self):
        node = Tree(n_children=5)
        self.assertFalse(node.has_children(),'Node should not have children.')
        node.split()
        self.assertTrue(node.has_children(),'Node should now have children.')
    
    
    def test_traverse(self):
        #
        # Binary Tree
        #  
        # Standard
        node = Tree(n_children=2)
        node.split()
        node.get_child(0).split()
        node.get_child(0).get_child(1).remove()
        addresses = {'breadth-first': [[],[0],[1],[0,0]], 
                     'depth-first': [[],[0],[0,0],[1]]}
 
        for mode in ['depth-first','breadth-first']:
            count = 0
            for leaf in node.traverse(mode=mode):
                self.assertEqual(leaf.get_node_address(),\
                                 addresses[mode][count],\
                                 'Binary tree traversal incorrect.')
                count += 1
        
        #
        # Quadtree
        # 
        node = Tree(n_children=4)
        node.split()
        node.get_child(1).split()
        node.get_child(1).get_child(2).remove()
        addresses = [[],[0],[1],[2],[3],[1,0],[1,1],[1,3]]
        count = 0
        for n in node.traverse(mode='breadth-first'):
            self.assertEqual(n.get_node_address(), addresses[count],\
                             'Incorrect address.')
            count += 1
    
    
    def test_get_leaves(self):
        #
        # 1D
        # 
        node = Tree(n_children=2)
        leaves = node.get_leaves()
        self.assertEqual(leaves, [node], 'Cell should be its own leaf.')
        
        #
        # Split cell and L child - find leaves
        # 
        node.split()
        l_child = node.get_child(0)
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
            self.assertEqual(leaf.get_node_address(), \
                             addresses_depth_first[i],\
                             'Incorrect order, depth first search.')
        #
        # Breadth first order
        # 
        addresses_breadth_first = [[1],[0,0],[0,1]]
        leaves = node.get_leaves(nested=True)
        for i in range(len(leaves)):
            leaf = leaves[i]
            self.assertEqual(leaf.get_node_address(),\
                             addresses_breadth_first[i],\
                             'Incorrect order, breadth first search.')
        
        
        node.get_child(0).get_child(0).mark('1')
        node.get_child(1).mark('1')
        leaves = node.get_leaves(flag='1', nested='True')
        self.assertEqual(len(leaves),2, \
                         'There should only be 2 flagged leaves')
        
        #
        # 2D
        # 
        node = Tree(n_children=4)
        
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
        self.assertEqual(leaves[0].get_node_address(),[1], \
            'The first leaf in the nested enumeration should have address [1]')
        
        leaves = node.get_leaves()
        self.assertEqual(leaves[0].get_node_address(), [0,0], \
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
        node = Tree(n_children=4)
        node.mark(1)
        self.assertTrue(node in node.get_leaves(flag=1), \
                        'Node should be a marked leaf node.')
        self.assertTrue(node in node.get_leaves(), \
                        'Node should be a marked leaf node.')
    
        node.split()
        sw_child = node.get_child(0)
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
        node = Tree(n_children=4)
        node.split()
        for child in node.get_children():
            child.split()
            
        node.get_child(1).mark(1, recursive=True)
        node.get_child(3).mark(1)
        
        leaves = node.get_leaves(nested=True, flag=1)
        self.assertEqual(len(leaves), 5, 
                         'This tree has 5 flagged LEAF nodes.')
        self.assertEqual(leaves[0], node.get_child(3), 
                         'The first leaf should be the NE child.')
        self.assertEqual(leaves[3], node.get_child(1).get_child(2),
                         '4th flagged leaf should be SE-NW grandchild.')
    
    
    def test_find_node(self):
        node = Tree(n_children=4)
        address = [0,2,3,0]
        # No-one lives at this address: return None
        self.assertIsNone(node.find_node(address))
        
        # Generate node with given address and try to recover it. 
        for a in address:
            node.split()
            node = node.get_child(a)
        self.assertEqual(node, node.get_root().find_node(address))
            