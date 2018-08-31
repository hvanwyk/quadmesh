#import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import numbers
import time
#from math import isclose
"""
Created on Jun 29, 2016
@author: hans-werner

"""

def convert_to_array(x, dim=None):
    """
    Convert point or list of points to a numpy array.
    
    Inputs: 
    
        x: (list of) point(s) to be converted to an array. Allowable inputs are
            
            1. a list of Vertices,
            
            2. a list of tuples, 
            
            3. a list of numbers or (2,) arrays
            
            4. a numpy array of the approriate size
            
        dim: int, (1 or 2) optional number used to adjudicate ambiguous cases.  
        
        
    Outputs:
    
        x: double, numpy array containing the points in x. 
        
            If x is one-dimensional (i.e. a list of 1d Vertices, 1-tuples, or
            a 1d vector), convert to an (n,1) array.
            
            If x is two-dimensional (i.e. a list of 2d Vertices, 2-tupples, or
            a 2d array), return an (n,2) array.   
    """
    if type(x) is list:
        #
        # Points in list
        #
        if all(isinstance(xi, Vertex) for xi in x):
            #
            # All points are of type vertex
            #
            x = [xi.coordinates() for xi in x]
            x = np.array(x)

        elif all(type(xi) is tuple for xi in x):
            #
            # All points are tuples
            #
            x = np.array(x)
        elif all(type(xi) is numbers.Real for xi in x):
            #
            # List of real numbers -> turn into (n,1) array
            # 
            x = np.array(x)
            x = x[:,np.newaxis]
        elif all(type(xi) is np.ndarray for xi in x):
            #
            # list of (2,) arrays 
            #
            x = np.array(x)
        else:
            raise Exception(['For x, use arrays or lists'+\
                             'of tuples or vertices.'])
    elif isinstance(x, Vertex):
        #
        # A single vertex
        #    
        x = np.array([x.coordinates()])
    elif isinstance(x, numbers.Real):
        if dim is not None:
            assert dim==1, 'Dimension should be 1.'
        x = np.array([[x]]) 
    elif type(x) is tuple:
        #
        # A tuple
        #
        if len(x)==1:
            #
            # A oneple
            #
            x, = x 
            x = np.array([[x]])
        elif len(x)==2:
            #
            # A tuple
            # 
            x,y = x
            x = np.array([[x,y]])
    elif type(x) is np.ndarray:
        #
        # Points in numpy array
        #
        if len(x.shape)==1:
            #
            # x is a one-dimensional vector
            if len(x)==1:
                #
                # x is a vector with one entry
                # 
                if dim is not None:
                    assert dim==1, 'Incompatible dimensions'
                x = x[:,np.newaxis]
            if len(x) == 2:
                #
                # x is a vector 2 entries: ambiguous
                # 
                if dim == 2:
                    #
                    # Turn 2-vector into a (1,2) array
                    #
                    x = x[np.newaxis,:]
                else:          
                    #
                    # Turn vector into (2,1) array
                    # 
                    x = x[:,np.newaxis]
            else:
                #
                # Turn vector into (n,1) array
                # 
                x = x[:,np.newaxis]
                
        elif len(x.shape)==2:
            assert x.shape[1]<=2,\
            'Dimension of array should be at most 2'
        else:
            raise Exception('Only 1- or 2 dimensional arrays allowed.') 
    return x


class Tree(object):
    """
    Description: Tree object for storing and manipulating adaptively
        refined quadtree meshes.
    
    Attributes:
    
        node_type: str, specifying node's relation to parents and/or children  
            'ROOT' (no parent node), 
            'BRANCH' (parent & children), or 
            'LEAF' (parent but no children)
        
        address: int, list allowing access to node's location within the tree
            General form [k0, k1, ..., kd], d=depth, ki in [0,...,n_children_i] 
            address = [] if ROOT node. 
        
        depth: int, depth within the tree (ROOT nodes are at depth 0).
        
        parent: Tree/Mesh whose child this is
        
        children: list of child nodes. 
            
        flag: set, of str/int/bool allowing tree nodes to be marked.
        
    """
    def __init__(self, n_children=None, regular=True, flag=None,
                 parent=None, position=None, forest=None):
        """
        Constructor
        """
        #
        # Set some attributes
        # 
        self._is_regular = regular
        self._parent = parent
        self._forest = None
        self._in_forest = False
        self._node_position = position
        
        #
        # Set flags
        # 
        self._flags = set()
        if flag is not None:
            if type(flag) is set:
                # Add all flags in set 
                for f in flag:
                    self.mark(f)
            else:
                # Add single flag
                self.mark(flag)
                
        if parent is None:
            #
            # ROOT Tree
            # 
            self._node_type = 'ROOT'
            self._node_depth = 0
            self._node_address = []
            
            if self.is_regular():
                # Ensure that the number of ROOT children is specified
                assert n_children is not None, \
                    'ROOT node: Specify number of children.'
            else:
                # Not a regular tree: number of children 0 initially
                n_children = 0
            
            if forest is not None:
                #
                # Tree contained in a Forest
                # 
                assert isinstance(forest, Forest), \
                'Input grid must be an instance of Grid class.' 
                
                #
                # Add tree to forest
                # 
                forest.add_tree(self)
                
                self._in_forest = True
                self._forest = forest 
                self._node_address = [self.get_node_position()]   
            else:
                #
                # Free standing ROOT cell
                # 
                assert self.get_node_position() is None, \
                    'Unattached ROOT cell has no position.'
                
            #
            # Assign space for children
            # 
            self._children = [None]*n_children        
            self._n_children = n_children                
        else:
            #
            # LEAF Node
            #  
            position_missing = 'Position within parent cell must be specified.'
            assert self.get_node_position() is not None, position_missing
            
            self._node_type = 'LEAF'
                        
            # Determine cell's depth and address
            self._node_depth = parent.get_depth() + 1
            self._node_address = parent.get_node_address() + [position]            
            
            if regular:
                # 
                # Regular tree -> same number of children in every generation
                #
                if n_children is not None:
                    assert n_children == self.get_parent().n_children(),\
                        'Regular tree: parents should have the same ' + \
                        'number of children than oneself.'
                else:
                    n_children = self.get_parent().n_children()
            else:
                n_children = 0
            
            #
            # Assign space for children
            # 
            self._children = [None]*n_children        
            self._n_children = n_children
            
            # Change parent type (from LEAF)
            if parent.get_node_type() == 'LEAF':
                parent.set_node_type('BRANCH')
   
   
    def info(self):
        """
        Display essential information about Tree
        """
        print('')
        print('-'*50)
        print('Tree Info')
        print('-'*50)
        print('{0:10}: {1}'.format('Address', self._node_address))
        print('{0:10}: {1}'.format('Type', self._node_type))
        if self._node_type != 'ROOT':
            print('{0:10}: {1}'.format('Parent', \
                                       self.get_parent().get_node_address()))
            print('{0:10}: {1}'.format('Position', self._node_position))
        print('{0:10}: {1}'.format('Flags', self._flags))
        if self.has_children():
            child_string = ''
            for i in range(len(self._children)):
                child = self.get_child(i)
                if child is not None:
                    child_string += str(i) + ': 1,  '
                else:
                    child_string += str(i) + ': 0,  '
            print('{0:10}: {1}'.format('Children',child_string))
        else:
            child_string = 'None'
            print('{0:10}: {1}'.format('Children',child_string))
        print('')
     
    def get_node_type(self):
        """
        Returns whether node is a ROOT, a BRANCH, or a LEAF
        """
        return self._node_type
    
    
    def get_node_position(self):
        """
        Returns position of current node within parent/forest
        """
        return self._node_position
    
        
    def set_node_type(self, node_type):
        """
        Sets a node's type 
        """
        assert node_type in ['ROOT', 'BRANCH', 'LEAF'], \
            'Input "node_type" should be "ROOT", "BRANCH", or "LEAF".'
         
        if node_type == 'ROOT':
            assert not self.has_parent(), \
                'ROOT nodes should not have a parent.'
        elif node_type == 'LEAF':
            assert not self.has_children(), \
                'LEAF nodes should not have children.'
        elif node_type == 'BRANCH':
            assert self.has_parent(),\
                'BRANCH nodes should have a parent.'
        self._node_type = node_type
                
    
    def get_node_address(self):
        """
        Return address of the node
        """
        return self._node_address
    
        
    def get_depth(self):
        """
        Return depth of current node
        """
        return self._node_depth
    
    
    def tree_depth(self, flag=None):
        """
        Return the maximum depth of the tree
        """
        depth = self.get_depth()
        if self.has_children():
            for child in self.get_children(flag=flag):
                d = child.tree_depth()
                if d > depth:
                    depth = d 
        return depth
    
    
    def in_forest(self):
        """
        Determine whether a (ROOT)cell lies within a forest        
        """
        return self._in_forest 
    

    def get_forest(self):
        """
        Returns the forest containing the node
        """
        return self._forest


    def plant_in_forest(self, forest, position):
        """
        Modify own attributes to reflect node's containment within a forest 
        """
        assert self.get_node_type() == 'ROOT', \
            'Only ROOT nodes are in the forest.'
        self._node_position = position
        self._node_address = [position]
        self._in_forest = True
        self._forest = forest
         
    
    def remove_from_forest(self):
        """
        Remove node from forest
        """
        self._in_forest = False
        self._node_position = None
        self._node_address = []
        self._forest = None
    
    
    def is_regular(self):
        """
        Determine whether node is a regular tree, that is all subnodes 
        have the same number of children.
        """
        return self._is_regular
    

    def mark(self, flag=None, recursive=False, reverse=False):
        """
        Mark Cell
        
        Inputs:
        
            flag: int, optional label used to mark node
            
            recursive: bool, also mark all sub-/super nodes
        """  
        if flag is None:
            #
            # No flag specified: add "True" flag
            # 
            self._flags.add(True)
        else:
            #
            # Add given flag
            # 
            self._flags.add(flag)
        #
        # Add flag to progeny/parents
        #     
        if recursive:
            if reverse:
                #
                # Mark ancestors
                #
                if self.has_parent():
                    parent = self.get_parent()
                    parent.mark(flag=flag, recursive=recursive, \
                                reverse=reverse)
            else:
                #
                # Mark progeny
                #
                if self.has_children():
                    for child in self.get_children():
                        child.mark(flag=flag, recursive=recursive)
      
    
    def unmark(self, flag=None, recursive=False, reverse=False):
        """
        Unmark Cell
        
        Inputs: 
        
            flag: label to be removed
        
            recursive: bool, also unmark all subcells
        """
        #
        # Remove label from own list
        #
        if flag is None:
            # No flag specified -> delete all
            self._flags.clear()
        else:
            # Remove specified flag (if present)
            if flag in self._flags: self._flags.remove(flag) 
        #
        # Remove label from children if applicable   
        # 
        if recursive:
            if reverse:
                #
                # Unmark ancestors
                # 
                if self.has_parent():
                    parent = self.get_parent()
                    parent.unmark(flag=flag, recursive=recursive, \
                                  reverse=reverse)
            else:
                #
                # Unmark progeny
                #
                if self.has_children():
                    for child in self.get_children():
                        child.unmark(flag=flag, recursive=recursive)
 
         
    def is_marked(self,flag=None):
        """
        Check whether cell is marked
        
        Input: flag, label for QuadCell: usually one of the following:
            True (catchall), 'split' (split cell), 'count' (counting)
            
        """ 
        if flag is None:
            # No flag -> check whether set is empty
            if self._flags:
                return True
            else:
                return False
        else:
            # Check wether given label is contained in cell's set
            return flag in self._flags
        
    
    def has_parent(self, flag=None):
        """
        Returns True if node has (flagged) parent node, False otherwise
        """
        if flag is not None:
            return self._parent is not None and self._parent.is_marked(flag)
        else:
            return self._parent is not None
    
    
    def get_parent(self, flag=None):
        """
        Return cell's parent, or first ancestor with given flag (None if there
        are none).
        """
        if flag is None:
            if self.has_parent():
                return self._parent
        else:
            if self.has_parent(flag):
                parent = self._parent
                if parent.is_marked(flag):
                    return parent
                else:
                    return parent.get_parent(flag=flag)
     
     
    def get_root(self):
        """
        Find the ROOT cell for a given cell
        """
        if self._node_type == 'ROOT':
            return self
        else:
            return self._parent.get_root()   
    
    
    def has_children(self, position=None, flag=None):
        """
        Determine whether node has children
        
        
        Inputs:
        
            position: int, position of the child node within self
            
            flag: str/int/bool, required marker for positive answer 
        
        
        Output:
        
            has_children: bool, true if self has (marked) children, false
                otherwise.
        """
        if position is None:
            #
            # Check for any children
            #
            if flag is None:
                return any(child is not None for child in self._children)
            else:
                #
                # Check for flagged children
                #
                for child in self._children:
                    if child is not None and child.is_marked(flag):
                        return True
                return False
        else:
            #
            # Check for child in specific position
            # 
            # Ensure position is valid
            assert position < self._n_children, \
                'Position exceeds the number of children.' 
            if flag is None:
                #
                # No flag specified
                #  
                return self._children[position] is not None
            else:
                #
                # With flag
                # 
                return (self._children[position] is not None) and \
                        self._children[position].is_marked(flag) 
    
    
    def get_child(self, position):
        """
        Return the child in a given position
        """
        assert position<self.n_children() and position>-self.n_children(), \
            'Input "position" exceeds number of children.'
        return self._children[position]
            
    
    def get_children(self, flag=None, reverse=False):
        """
        Iterator: Returns (flagged) children, in (reverse) order 
        
        Inputs: 
        
            flag: [None], optional marker
            
            reverse: [False], option to list children in reverse order 
                (useful for the 'traverse' function).
        
        Note: Only returns children that are not None
              Use this to obtain a consistent iteration of children
        """
        
        if self.has_children(flag=flag):
            if not reverse:
                #
                # Go in usual order
                # 
                for child in self._children:
                    if child is not None:
                        if flag is None:
                            yield child
                        elif child.is_marked(flag):
                            yield child
            else: 
                #
                # Go in reverse order
                #
                for child in reversed(self._children): 
                    if child is not None:
                        if flag is None:
                            yield child
                        elif child.is_marked(flag):
                            yield child


    def n_children(self):
        """
        Returns the number of children
        """
        return self._n_children
    
    
    def remove(self):
        """
        Remove node (self) from parent's list of children
        """
        assert self.get_node_type() != 'ROOT', 'Cannot delete ROOT node.'
        self.get_parent()._children[self._node_position] = None    
    

    def add_child(self):
        """
        Add a child to current node (only works if node is not regular).
        """
        assert not self.is_regular(),\
            'Regular tree: add children by method "split".'
         
        child = Tree(parent=self, regular=False, position=self.n_children())
        self._children.append(child)
        self._n_children += 1
            
            
            

    def delete_children(self, position=None):
        """
        Delete all sub-nodes of given node
        """
        #
        # Change children to None
        # 
        if position is None:
            for child in self.get_children():
                child.remove()
        else:
            assert position < self.n_children(), \
            'Position exceeds number of children '
            child = self._children[position]
            child.remove()
        #
        # Change node type from LEAF to BRANCH
        # 
        if self._node_type == 'BRANCH' and not self.has_children():
            self._node_type = 'LEAF'           

        
    def split(self, n_children=None):
        """
        Split node into subnodes
        """
        if self.is_regular():
            #
            # Regular tree: Number of grandchildren inherited
            # 
            for i in range(self.n_children()):
                #
                # Instantiate Children
                # 
                self._children[i] = Tree(parent=self, position=i)
        else:
            #
            # Not a regular tree: Must specify number of children
            #
            assert self.n_children() == 0, \
                'Cannot split irregular tree with children. ' + \
                'Use "add_child" method.' 
            
            for i in range(n_children):
                #
                # Instantiate Children
                # 
                self.add_child()
                
            
    def traverse(self, queue=None, flag=None, mode='depth-first'):
        """
        Iterator: Return current cell and all its (flagged) sub-cells         
        
        Inputs: 
        
            flag [None]: cell flag
            
            mode: str, type of traversal 
                'depth-first' [default]: Each cell's progeny is visited before 
                    proceeding to next cell.
                 
                'breadth-first': All cells at a given depth are returned before
                    proceeding to the next level.
        
        Output:
        
            all_nodes: list, of all nodes in tree (marked with flag).
        """
        if queue is None:
            queue = deque([self])
            
        while len(queue) != 0:
            if mode == 'depth-first':
                node = queue.pop()
            elif mode == 'breadth-first':
                node = queue.popleft()
            else:
                raise Exception('Input "mode" must be "depth-first"'+\
                                ' or "breadth-first".')
            if node.has_children():
                reverse = True if mode=='depth-first' else False    
                for child in node.get_children(reverse=reverse):
                    queue.append(child)
            
            if flag is not None: 
                if node.is_marked(flag):
                    yield node
            else:
                yield node         
                 
                
    def get_leaves(self, flag=None, subtree_flag=None, mode='breadth-first'):
        """
        Return all marked LEAF nodes (nodes with no children) of current subtree
        
        Inputs:
        
            *flag: If flag is specified, return all leaf nodes within rooted 
                subtree marked with flag (or an empty list if there are none).
                
            *subtree_flag: Label specifying the rooted subtree (rs) within which
                to search for (flagged) leaves. 
                
            *mode: Method by which to traverse the tree ('breadth-first' or
                'depth-first').
                  

        Outputs:
        
            leaves: list, of LEAF nodes.
            
            
        Note: 
        
            The rooted subtree must contain all ancestors of a marked node 
        """
        #
        # Get all leaves of the subtree
        # 
        leaves = []
        for node in self.traverse(flag=subtree_flag, mode=mode):
            #
            # Iterate over all sub-nodes within subtree
            # 
            if not node.has_children(flag=subtree_flag):
                #
                # Nodes without marked children are the subtree leaves
                # 
                leaves.append(node)
        #
        # Return marked leaves
        # 
        if flag is None:
            return leaves
        else: 
            return [leaf for leaf in leaves if leaf.is_marked(flag)]
    
    
    def make_rooted_subtree(self, flag):
        """
        Mark all ancestors of flagged node with same flag, to turn flag into
        a subtree marker. 
        """
        #
        # Search through all nodes
        # 
        for node in self.get_root().traverse(mode='breadth-first'):
            if node.is_marked(flag):
                #
                # If node is flagged, mark all its ancestors
                # 
                ancestor = node
                while ancestor.has_parent():
                    ancestor = ancestor.get_parent()
                    ancestor.mark(flag)
      
    
    def is_rooted_subtree(self, flag):
        """
        Determine whether a given flag defines a rooted subtree
        
        Note: This takes roughly the same amount of work as make_rooted_subtree  
        """
        #
        # Search through all nodes
        # 
        for node in self.get_root().traverse(mode='breadth-first'):
            if node.is_marked(flag):
                #
                # Check that ancestors of flagged node are also marked 
                # 
                ancestor = node
                while ancestor.has_parent():
                    ancestor = ancestor.get_parent()
                    if not ancestor.is_marked(flag):
                        #
                        # Ancestor not marked: not a rooted subtree
                        # 
                        return False
        #
        # No problems: it's a rooted subtree
        #
        return True               
            
        
    def find_node(self, address):
        """
        Locate node by its address
        """
        node = self.get_root()
        if address != []:
            #
            # Not the ROOT node
            # 
            for a in address:
                if node.has_children() and a in range(node.n_children()):
                    node = node.get_child(a)
                else:
                    return None
        return node 


    def contains(self, tree):
        """
        Determine whether self contains a given node 
        """
        if tree.get_depth() < self.get_depth():
            return False
        elif tree == self:
            return True
        else:
            while tree.get_depth() > self.get_depth():
                tree = tree.get_parent()
                if self == tree:
                    return True
            #
            # Reached the end 
            # 
            return False
        
        
class Forest(object):
    """
    Collection of Trees
    """
    def __init__(self, trees=None, n_trees=None):
        """
        Constructor
        """
        if trees is not None:
            #
            # List of trees specified
            # 
            assert type(trees) is list, 'Trees should be passed as a list.'
            self._trees = []
            for tree in trees:
                self.add_tree(tree)
                
        elif n_trees is not None:
            #
            # No trees specified, only the number of slots
            #
            assert type(n_trees) is np.int and n_trees > 0,\
                'Input "n_children" should be a positive integer.'
            self._trees = [None]*n_trees
        else:
            #
            # No trees specified: create an empty list.
            # 
            self._trees = []
    
    
    def n_children(self):
        """
        Return the number of trees
        """
        return len(self._trees)
    
    
    def is_regular(self):
        """
        Determine whether the forest contains only regular trees
        """
        for tree in self._trees:
            if not tree.is_regular():
                return False
        return True 
    
    
    def depth(self):
        """
        Determine the depth of the largest tree in the forest
        """
        current_depth = 0
        for tree in self.get_children():
            new_depth = tree.tree_depth()
            if new_depth > current_depth:
                current_depth = new_depth
        return current_depth
    
        
    def traverse(self, flag=None, mode='depth-first'):
        """
        Iterator: Visit every (flagged) node in the forest         
        
        Inputs: 
        
            flag [None]: node flag
            
            mode: str, type of traversal 
                'depth-first' [default]: Each node's progeny is visited before 
                    proceeding to next cell.
                 
                'breadth-first': All nodes at a given depth are returned before
                    proceeding to the next level.
        
        Output:
        
            all_nodes: list, of all nodes in tree (marked with flag).
            
        """
        if mode=='depth-first':
            queue = deque(reversed(self._trees))
        elif mode=='breadth-first':
            queue = deque(self._trees)
            
        while len(queue) != 0:
            if mode == 'depth-first':
                node = queue.pop()
            elif mode == 'breadth-first':
                node = queue.popleft()
            else:
                raise Exception('Input "mode" must be "depth-first"'+\
                                ' or "breadth-first".')
            if node.has_children():
                reverse = True if mode=='depth-first' else False    
                for child in node.get_children(reverse=reverse):
                    queue.append(child)
            
            if flag is not None: 
                if node.is_marked(flag):
                    yield node
            else:
                yield node
        
        
    def get_leaves(self, flag=None, subforest_flag=None, mode='breadth-first'):
        """
        Return all marked LEAF nodes (nodes with no children) of current subtree
        
        Inputs:
        
            *flag: If flag is specified, return all leaf nodes within rooted 
                subtree marked with flag (or an empty list if there are none).
                
            *subforest_flag: Label specifying the rooted subtrees (rs) within which
                to search for (flagged) leaves. 
                  

        Outputs:
        
            leaves: list, of LEAF nodes.
            
            
        Note: 
        
            The rooted subtree must contain all ancestors of a marked node 
        """
        #
        # Get all leaves of the subtree
        # 
        leaves = []
        for node in self.traverse(flag=subforest_flag, mode=mode):
            if not node.has_children(flag=subforest_flag):
                leaves.append(node)
        #
        # Return marked leaves
        # 
        if flag is None:
            return leaves
        else: 
            return [leaf for leaf in leaves if leaf.is_marked(flag)]
           
        
    def make_forest_of_rooted_subtrees(self, flag):
        """
        Mark all ancestors of flagged node with same flag, to turn flag into
        a subtree marker. 
        
        Note: If no node is flagged, then only flag the root nodes.
        """
        #
        # Search through all nodes
        # 
        for root_node in self.get_children():
            #
            # Mark all root nodes with flag
            # 
            root_node.mark(flag)
            for node in root_node.traverse():
                #
                # Look for marked subnodes
                # 
                if node.is_marked(flag):
                    #
                    # If node is flagged, mark all its ancestors
                    # 
                    ancestor = node
                    while ancestor.has_parent():
                        ancestor = ancestor.get_parent()
                        ancestor.mark(flag)
          
    
    def is_forest_of_rooted_subtree(self, flag):
        """
        Determine whether a given flag defines a rooted subtree
        
        Note: This takes roughly the same amount of work as make_rooted_subtree  
        """
        if flag is None:
            #
            # Forest itself is always one of rooted subtrees
            # 
            return True
        #
        # Search through all nodes
        # 
        for root_node in self.get_children():
            #
            # Check if root nodes are marked
            # 
            if not root_node.is_marked(flag):
                return False
            else:
                for node in root_node.traverse(mode='breadth-first'):
                    if node.is_marked(flag):
                        #
                        # Check that ancestors of flagged node are also marked 
                        # 
                        ancestor = node
                        while ancestor.has_parent():
                            ancestor = ancestor.get_parent()
                            if not ancestor.is_marked(flag):
                                #
                                # Ancestor not marked: not a rooted subtree
                                # 
                                return False
        #
        # No problems: it's a forest of rooted subtrees
        #
        return True               
 
    
    
    def find_node(self, address):
        """
        Locate a tree node by its address
        """
        # Reverse address
        address = address[::-1]
        node = self
        while len(address)>0:
            a = address.pop()
            if node.has_children():
                if a not in range(node.n_children()):
                    return None
                else:
                    node = node.get_child(a)
        return node
        
    
    def has_children(self, flag=None):
        """
        Determine whether the forest contains any trees 
        """
        
        if len(self._trees) > 0:
            if flag is None:
                return True
            else:
                return any(tree for tree in self.get_children(flag=flag))
        else:
            return False
    
    
    def get_child(self, position):
        """
        Returns the tree at a given position
        """
        assert position < len(self._trees),\
            'Input "position" exceeds number of trees.'
        assert type(position) is np.int, \
            'Input "position" should be a nonnegative integer. '
        return self._trees[position]    
        
        
    def get_children(self, flag=None, reverse=False):
        """
        Iterate over (all) (flagged) trees in the forest
        """        
        if not reverse:
            if flag is None:
                return self._trees
            else: 
                children = []
                for tree in self._trees:
                    if tree.is_marked(flag):
                        children.append(tree)
                        
                return children
        else:
            if flag is None:
                return self._trees[::-1]
            else:
                children = []
                for tree in reversed(self._trees):
                    if tree.is_marked():
                        children.append(tree)
                        
    
    def add_tree(self, tree):
        """
        Add a new tree to the current forest
        """
        assert isinstance(tree, Tree), \
            'Can only add trees to the forest.'
        self._trees.append(tree)
        tree.plant_in_forest(self, self.n_children()-1)    
        
    
    def remove_tree(self, position):
        """
        Remove a tree from the forest.
        """
        assert type(position) is np.int, \
            'Input "position" should be an integer.'
        assert position < len(self._trees), \
            'Input "position" exceeds number of trees.' 
        tree = self.get_child(position)
        tree.remove_from_forest()
        del self._trees[position]  

    
    def record(self, flag):
        """
        Mark all trees in current forest with flag
        """
        for tree in self.get_children():
            tree.mark(flag, recursive=True)
    
        
    def coarsen(self, subforest_flag=None, coarsening_flag=None, 
                new_label=None, clean_up=True):
        """
        Coarsen (sub)forest (delimited by 'subforest_flag', by (possibly)
        merging (=deleting or unlabeling the siblings of) nodes marked with 
        'coarsening_flag' and labeling said nodes with new_label. 
        
        Inputs:
        
            *subforest_flag: flag, specifying the subforest being coarsened.
            
            *coarsening_flag: flag, specyfying nodes in subforest whose children
                are to be deleted/unmarked.
                
            *new_label: flag, specifying the new subforest.
            
            *clean_up: bool, remove coarsening_flag after use.
        """
        #
        # Ensure the subforest is rooted
        #
        if subforest_flag is not None:
            self.make_forest_of_rooted_subtrees(subforest_flag)
        
        #
        # Look for marked leaves within the submesh
        #
        for leaf in self.get_leaves(subforest_flag=subforest_flag):
            #
            # Find leaves that must be coarsened
            #
            if coarsening_flag is not None:
                #
                # Only coarsen flagged nodes
                #
                if leaf.has_parent(coarsening_flag):
                    parent = leaf.get_parent()
                     
                    if clean_up:
                        #
                        # Remove coarsening flag
                        # 
                        parent.unmark(coarsening_flag)
                else:
                    if new_label is not None:
                        #
                        # Nodes not flagged for coarsening should be part of new mesh
                        #
                        for child in parent.get_children(): 
                            child.mark(new_label)
                    continue
            else:
                #
                # Indiscriminate coarsening
                # 
                if leaf.has_parent():
                    parent = leaf.get_parent()  
                else:
                    if new_label is not None:
                        leaf.mark(new_label)
                    continue
            #
            # Coarsen
            # 
            if subforest_flag is None and new_label is None:
                #
                # Delete self and siblings
                # 
                parent.delete_children()
            elif new_label is None:
                #
                # Remove 'subforest_label' from leaf and siblings
                # 
                for child in parent.get_children():
                    child.unmark(subforest_flag)
            else:
                #
                # Mark parents with new_label
                # 
                parent.mark(new_label)
        #
        # Apply new label to coarsened submesh if necessary
        # 
        if new_label is not None:
            self.make_forest_of_rooted_subtrees(new_label)
            
    
    def refine(self, subforest_flag=None, refinement_flag=None, new_label=None, 
               clean_up=True):
        """
        Refine (sub)forest (delimited by 'subforest_flag'), by (possibly) 
        splitting (subforest)nodes with refinement_flag and marking their 
        children (with new_label).
        
        Inputs:
        
            subforest_flag: flag, used to specify the subforest being refined
            
            refinemenet_flag: flag, specifying the nodes within the submesh that
                are being refined. 
                
            new_label: flag, new label to be applied to refined submesh
        """        
        #
        # Ensure that the subforest is rooted
        # 
        if subforest_flag is not None:
            self.make_forest_of_rooted_subtrees(subforest_flag)
        #
        # Look for marked leaves within the submesh
        # 
        for leaf in self.get_leaves(subforest_flag=subforest_flag):
            #
            # Mark tree with new label to ensure new forest contains old subforest
            # 
            if new_label is not None:
                leaf.mark(new_label)
            #
            # If the refinement flag is used, ensure that the node is marked
            # before continuing.
            # 
            if refinement_flag is not None:
                if not leaf.is_marked(refinement_flag):
                    continue           
            #
            # Add new children if necessary
            # 
            if not leaf.has_children():
                leaf.split()
            #
            # Label each (new) child
            #
            for child in leaf.get_children(): 
                if new_label is None and subforest_flag is None:
                    #
                    # No labels specified: do nothing
                    #
                    continue
                elif new_label is None:
                    #
                    # No new label given, use the subforest label 
                    # 
                    child.mark(subforest_flag)
                else:
                    #
                    # New label given, mark child with new label
                    #
                    child.mark(new_label)
            #
            # Remove refinement flag
            # 
            if refinement_flag is not None and clean_up:
                leaf.unmark(refinement_flag)
        #
        # Label ancestors of newly labeled children
        # 
        if new_label is not None:
            self.make_forest_of_rooted_subtrees(new_label)
 
        
class Vertex(object):
    """
    Description:
    
    Attributes:
    
        coordinates: double, tuple (x,y)
        
        flag: boolean
    
    Methods: 
    """


    def __init__(self, coordinates):
        """
        Description: Constructor
        
        Inputs: 
        
            coordinates: double tuple, x- and y- coordinates of vertex
            
            on_boundary: boolean, true if on boundary
              
        """
        if isinstance(coordinates, numbers.Real):
            #
            # Coordinate passed as a real number 1D
            # 
            dim = 1
            coordinates = (coordinates,)  # recast coordinates as tuple
        elif type(coordinates) is tuple:
            #
            # Coordinate passed as a tuple
            # 
            dim = len(coordinates)
            assert dim <= 2, 'Only 1D and 2D meshes supported.'
        else:
            raise Exception('Enter coordinates as a number or a tuple.')
        self.__coordinate = coordinates
        self._flags = set()
        self.__dim = dim
        self.__periodic_pair = set()
        self.__is_periodic = False
        
    
    def coordinates(self):
        """
        Return coordinates tuple
        """
        if self.__dim == 1:
            return self.__coordinate
        else:
            return self.__coordinate
    
    
    def dim(self):
        """
        Return the dimension of the vertex
        """
        return self.__dim
        
    
    def mark(self, flag=None):
        """
        Mark Vertex
        
        Inputs:
        
            flag: int, optional label
        """  
        if flag is None:
            self._flags.add(True)
        else:
            self._flags.add(flag)
            
        
    def unmark(self, flag=None):
        """
        Unmark Vertex
        
        Inputs: 
        
            flag: label to be removed

        """
        #
        # Remove label from own list
        #
        if flag is None:
            # No flag specified -> delete all
            self._flags.clear()
        else:
            # Remove specified flag (if present)
            if flag in self._flags: self._flags.remove(flag)
        
         
    def is_marked(self,flag=None):
        """
        Check whether Vertex is marked
        
        Input: flag, label for QuadCell: usually one of the following:
            True (catchall), 'split' (split cell), 'count' (counting)
        """ 
        if flag is None:
            # No flag -> check whether set is empty
            if self._flags:
                return True
            else:
                return False
        else:
            # Check wether given label is contained in cell's set
            return flag in self._flags

    
    def is_periodic(self):
        """
        Determine whether a Vertex lies on a periodic boundary
        """
        return self.__is_periodic
    
      
    def set_periodic(self, periodic=True):
        """
        Label vertex periodic
        """  
        self.__is_periodic = periodic
      
      
    def set_periodic_pair(self, cell_vertex_pair):
        """
        Pair a periodic vertex with its periodic counterpart. The periodic 
        pair can be accessed by specifying the neighboring interval (in 1D) 
        or cell (in 2D).
        
        Inputs:
        
            half_edge: HalfEdge/Interval 
             
             In 1D: half_edge represents the Interval on which the vertex pair resides
             
             In 2D: half_edge represents the HalfEdge on which the vertex itself resides 
        
            vertex: Vertex associated with 
        
        See also: get_periodic_pair
        """
        assert self.is_periodic(), 'Vertex should be periodic.'
        if self.dim()==1:
            #
            # 1D: There is only one pairing for the entire mesh
            # 
            interval, vertex = cell_vertex_pair            
            assert isinstance(vertex, Vertex), \
                'Input "vertex" should be of class "Vertex".'
            assert isinstance(interval, Interval), \
                'Input "interval" should be of class "Interval".'
            assert vertex.is_periodic(), \
                'Input "vertex" should be periodic.'
            #
            # 1D: Store periodic pair 
            #             
            self.__periodic_pair.add((interval, vertex))
             
            
        elif self.dim()==2:
            #
            # 2D
            # 
            c_nb, v_nb = cell_vertex_pair
            assert isinstance(v_nb, Vertex), \
                'Input "cell_vertex_pair[1]" should be of class "Vertex".'
            assert isinstance(c_nb, Cell), \
                'Input "cell_vertex_pair[0]" should be of class "HalfEdge".'
            assert v_nb.is_periodic(), \
                'Input "cell_vertex_pair[1]" should be periodic.'
            
            #
            # Collect all possible c/v pairs in a set
            # 
            cell_vertex_pairs = v_nb.get_periodic_pair().union(set([cell_vertex_pair]))
            assert len(cell_vertex_pairs)!=0, 'Set of pairs should be nonempty'
            for c_nb, v_nb in cell_vertex_pairs:
                #
                # Check whether v_nb already in list
                #
                in_list = False 
                for c, v in self.get_periodic_pair():
                    if v==v_nb and c.contains(c_nb):
                        #
                        # Vertex already appears in list
                        # 
                        in_list = True
                        break
                if not in_list:
                    #
                    # Not in list, add it
                    # 
                    self.__periodic_pair.add((c_nb, v_nb))
        
                
        
    def get_periodic_pair(self, cell=None):
        """
        Returns the other vertex that is mapped onto self through periodicity
        
        Input:
        
            cell: Cell/HalfEdge in which paired vertex resides
        """
        if cell is None:
            #
            # Return all cell, vertex pairs
            # 
            return self.__periodic_pair
        else:
            #
            # Return all paired vertices within a given cell
            # 
            vertices = [v for c, v in self.__periodic_pair if c==cell]
            return vertices
        
    
class HalfEdge(Tree):
    """
    Description: Half-Edge in Quadtree mesh
    
    Attributes:
    
        base: Vertex, at base 
        
        head: Vertex, at head
        
        twin: HalfEdge, in adjoining cell pointing from head to base 
        
        cell: QuadCell, lying to half edge's left 
        
    Methods:
    
    
    """ 
    def __init__(self, base, head, cell=None, previous=None, nxt=None, 
                 twin=None, parent=None, position=None, n_children=2, 
                 regular=True, forest=None, flag=None, periodic=False):
        """
        Constructor
        
        Inputs:
        
            base: Vertex, at beginning
            
            head: Vertex, at end
            
            parent: HalfEdge, parental 
            
            cell: QuadCell, lying to the left of half edge
            
            previous: HalfEdge, whose head is self's base
            
            nxt: HalfEdge, whose base is self's head
            
            twin: Half-Edge, in adjoining cell pointing from head to base
            
            position: int, position within parental HalfEdge
            
            n_children: int, number of sub-HalfEdges 
            
            regular: bool, do all tree subnodes have the same no. of children?
            
            forest: Forest, clever list of trees containing self 
            
            flag: (set of) int/string/bool, used to mark half-edge
            
            periodic: bool, True if HalfEdge lies on a periodic boundary
        """
        #
        # Initialize Tree structure
        # 
        Tree.__init__(self, n_children=n_children, regular=regular, 
                      parent=parent, position=position, forest=forest, flag=flag)
        #
        # Assign head and base
        # 
        assert isinstance(base, Vertex) and isinstance(head, Vertex),\
            'Inputs "base" and "head" should be Vertex objects.'
        self.__base = base
        self.__head = head
        
        #
        # Check parent
        # 
        if parent is not None:
            assert isinstance(parent, HalfEdge), \
                'Parent should be a HalfEdge.'
        #
        # Assign incident cell
        # 
        if cell is not None:
            assert isinstance(cell, Cell), \
                'Input "cell" should be a Cell object.'
        self.__cell = cell
        #
        # Assign previous half-edge
        # 
        if previous is not None: 
            assert isinstance(previous, HalfEdge), \
                'Input "previous" should be a HalfEdge object.'
            assert self.base()==previous.head(),\
                'Own base should equal previous head.'
        self.__previous = previous
        
        #
        # Assign next half-edge
        #
        if nxt is not None: 
            assert isinstance(nxt, HalfEdge), \
                'Input "nxt" should be a HalfEdge object.'
            assert self.head()==nxt.base(), \
                'Own head should equal base of next.'
        self.__next = nxt
        
        #
        # Mark periodic
        # 
        self.__is_periodic = periodic

        #
        # Assign twin half-edge
        #
        if twin is not None: 
            assert isinstance(twin, HalfEdge), \
                'Input "twin" should be a HalfEdge object.'
            self.assign_twin(twin)
        else:
            self.__twin = None
        
       
    def is_periodic(self):
        """
        Returns True is the HalfEdge lies on a periodic boundary
        """
        return self.__is_periodic
        
        
    def set_periodic(self, periodic=True):
        """
        Flag HalfEdge as periodic
        """
        self.__is_periodic = periodic
    
    
    def pair_periodic_vertices(self):
        """
        Pair up HalfEdge vertices that are periodic
        """
        if self.is_periodic(): 
            #
            # Pair up periodic vertices along half_edge
            #
            cell = self.cell()
            cell_nb = self.twin().cell()
            assert cell_nb is not None,\
                'Periodic HalfEdge: Neighboring cell should not be None.'
            #
            # Pair up adjacent vertices
            # 
            for v, v_nb in [(self.base(), self.twin().head()),
                            (self.head(), self.twin().base())]:
                # Label vertices 'periodic'
                v.set_periodic()
                v_nb.set_periodic()
                
                # Add own vertex-cell pair to own set of periodic pairs
                v.set_periodic_pair((cell, v))
                v_nb.set_periodic_pair((cell_nb, v_nb))
                
                # Add adjoining vertex-cell pair to set of periodic pairs
                v.set_periodic_pair((cell_nb, v_nb))
                v_nb.set_periodic_pair((cell, v))
            
    
    def base(self):
        """
        Returns half-edge's base vertex
        """
        return self.__base
    
    
    def head(self):
        """
        Returns half-edge's head vertex
        """
        return self.__head
    
    
    def cell(self):
        """
        Returns the cell containing half-edge
        """
        return self.__cell
    
    
    def assign_cell(self, cell):
        """
        Assign cell to half-edge
        """
        self.__cell = cell
        
    
    def twin(self):
        """
        Returns the half-edge's twin
        """
        return self.__twin
    
    
    def assign_twin(self, twin):
        """
        Assigns twin to half-edge
        """
        if not self.is_periodic():
            assert self.base()==twin.head() and self.head()==twin.base(),\
                'Own head vertex should be equal to twin base vertex & vice versa.'
        self.__twin = twin
        
    
    def make_twin(self):
        """
        Construct a twin HalfEdge
        """
        assert not self.is_periodic(), \
        'Twin HalfEdge of a periodic HalfEdge may have different vertices.'
        if self.has_parent() and self.get_parent().twin() is not None:
            twin_parent = self.get_parent().twin()
            twin_position = 1-self.get_node_position()
        else:
            twin_parent = None
            twin_position = None
        twin = HalfEdge(self.head(), self.base(), parent=twin_parent, 
                        position=twin_position)
        
        self.assign_twin(twin)
        twin.assign_twin(self)
        return twin
    
    def next(self):
        """
        Returns the next half-edge, whose base is current head
        """
        return self.__next
    
    
    def assign_next(self, nxt):
        """
        Assigns half edge to next
        """
        if nxt is None:
            return
        else:
            if not self.is_periodic():
                assert self.head() == nxt.base(), \
                    'Own head vertex is not equal to next base vertex.'
            self.__next = nxt
            if nxt.previous() != self:
                nxt.assign_previous(self)
            
    
    def previous(self):
        """
        Returns previous half-edge, whose head is current base
        """
        return self.__previous
        
    
    def assign_previous(self, previous):
        """
        Assigns half-edge to previous
        """
        if previous is None:
            return 
        else:
            if not self.is_periodic():
                assert self.base() == previous.head(), \
                    'Own base vertex is not equal to previous head vertex.'
            self.__previous = previous
            if previous.next()!=self:
                previous.assign_next(self)
        
        
    def split(self):
        """
        Refine current half-edge (overwrite Tree.split)
        
        Note:
        
            This function could potentially be generalized to HalfEdges with
            multiple children (already implemented for Intervals).
        """
        #
        # Check if twin has been split 
        #  
        twin_split = False
        twin = self.twin()
        if twin is not None and twin.has_children():
            t0, t1 = twin.get_children()
            twin_split = True
        else:
            t0, t1 = None, None

        #
        # Determine whether to inherit midpoint vertex
        # 
        if twin_split and not self.is_periodic():
            #
            # Share twin's midpoint Vertex
            # 
            vm = t0.head()
        else:
            #
            # Compute new midpoint vertex
            #
            x = convert_to_array([self.base().coordinates(),\
                                  self.head().coordinates()]) 
            xm = 0.5*(x[0,:]+x[1,:]) 
            vm = Vertex(tuple(xm))                
        #
        # Define own children and combine with twin children 
        # 
        c0 = HalfEdge(self.base(), vm, parent=self, twin=t1, position=0, periodic=self.is_periodic())
        c1 = HalfEdge(vm, self.head(), parent=self, twin=t0, position=1, periodic=self.is_periodic())
        
        #
        # Assign new HalfEdges to twins if necessary
        # 
        if twin_split:
            t0.assign_twin(c1)
            t1.assign_twin(c0)
        #
        # Save the babies
        # 
        self._children[0] = c0
        self._children[1] = c1
    
    
    def to_vector(self):
        """
        Returns the vector associated with the HalfEdge 
        """
        x = convert_to_array([self.base().coordinates(),\
                              self.head().coordinates()])
        return x[1,:] - x[0,:] 
        
        
    def length(self):
        """
        Returns the HalfEdge's length
        """
        return np.linalg.norm(self.to_vector())
    
    
    def unit_normal(self):
        """
        Returns the unit normal vector of HalfEdge, pointing to the right
        
        Note: This only works in 2D
        """
        x0, y0 = self.base().coordinates()
        x1, y1 = self.head().coordinates()
        u = np.array([y1-y0, x0-x1])
        return u/np.linalg.norm(u, 2)
    
    
    def contains_points(self, points):
        """
        Determine whether points lie on a HalfEdge
        """
        x0 = convert_to_array(self.base().coordinates())
        v = self.to_vector()
        dim = x0.shape[1]
        p = convert_to_array(points, dim)
        n_points = p.shape[0]
        in_half_edge = np.ones(n_points, dtype=np.bool)
        for i in range(dim):
            t = (p[:,i]-x0[:,i])/v[i]
            if i==0:
                s = t
            else:
                #
                # Discard points where the t's differ for different components
                # 
                in_half_edge[np.abs(t-s)>1e-9] = False
            #
            # Discard points where t not in [0,1]
            # 
            in_half_edge[np.abs(t-0.5)>0.5] = False
        
        return in_half_edge
        """
        if n_points==1:
            return in_half_edge[0]
        else:
            return in_half_edge
        """
        
    
    def intersects_line_segment(self, line):
        """
        Determine whether the HalfEdge intersects with a given line segment
        
        Input: 
        
            line: double, list of two tuples
            
        Output:
        
            boolean, true if intersection, false otherwise.
            
        Note: This only works in 2D
        """        
        # Express edge as p + t*r, t in [0,1]
        
        p = np.array(self.base().coordinates())
        r = np.array(self.head().coordinates()) - p
        
        # Express line as q + u*s, u in [0,1] 
        q = np.array(line[0]) 
        s = np.array(line[1]) - q
        
        if abs(np.cross(r,s)) < 1e-14:
            #
            # Lines are parallel
            # 
            if abs(np.cross(q-p,r)) < 1e-14:
                #
                # Lines are collinear
                # 
                t0 = np.dot(q-p,r)/np.dot(r,r)
                t1 = t0 + np.dot(s,r)/np.dot(r,r)
                
                if (max(t0,t1) >= 0) and (min(t0,t1) <= 1):
                    # 
                    # Line segments overlap
                    # 
                    return True
                else:
                    return False
            else:
                #
                # Lines not collinear
                # 
                return False 
        else:
            #
            # Lines not parallel
            #   
            t = np.cross(q-p,s)/np.cross(r,s)
            u = np.cross(p-q,r)/np.cross(s,r)
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                #
                # Line segments meet
                # 
                return True
            else:
                return False 


     
    def reference_map(self, x, jacobian=False, hessian=False, mapsto='physical'):
        """
        Map points x from the reference interval to the physical HalfEdge or vice versa
        
        Inputs:
        
            x: double, (n,) array or a list of points.
            
            jacobian: bool, return jacobian for mapping.
            
            mapsto: str, 'physical' (map from reference to physical), or 
                'reference' (map from physical to reference).
                
                
        Outputs:
        
            y: double, (n,) array of mapped points
            
            jac: array (2,), jacobian [x'(t), y'(t)]
            
            hess: array (2,2), Hessian (always 0).
        """
        if mapsto=='physical':
            #
            # Check that input is an array
            # 
            assert type(x) is np.ndarray, \
            'If "mapsto" is "physical", then input should '+\
            'be a one dimensional array.'
            #
            # Check that points contained in [0,1]
            # 
            assert x.max()>=0 and x.min()<=1, \
            'Reference point should be between 0 and 1.'
            
        elif mapsto=='reference':
            x = convert_to_array(x, dim=self.head().dim())
            #
            # Check that points lie on the HalfEdge
            # 
            assert all(self.contains_points(x)), \
            'Some points are not contained in the HalfEdge.'
              
        #
        # Compute mapped points
        #     
        n_points = x.shape[0]
        x0, y0 = self.base().coordinates()
        x1, y1 = self.head().coordinates()
        if mapsto == 'physical':
            y = [(x0 + (x1-x0)*xi, y0 + (y1-y0)*xi) for xi in x]
        elif mapsto == 'reference':
            if not np.isclose(x0, x1):
                #
                # Not a vertical line
                # 
                y = list((x[:,0]-x0)/(x1-x0))
            elif not np.isclose(y0, y1):
                #
                # Not a horizontal line
                # 
                y = list((x[:,1]-y0)/(y1-y0))
        
        #
        # Compute the Jacobian
        # 
        if jacobian:
            if mapsto == 'physical':
                #
                # Derivative of mapping from refence to physical cell
                # 
                jac = [np.array([[x1-x0],[y1-y0]])]*n_points
            elif mapsto == 'reference':
                #
                # Derivative of inverse map
                # 
                jac = np.array([[1/(x1-x0), 1/(y1-y0)]]) 

        # 
        # Compute the Hessian (linear mapping, so Hessian = 0)
        #
        hess = np.zeros((2,2))
         
        #
        # Return output
        # 
        if jacobian and hessian:
            return y, jac, hess
        elif jacobian and not hessian:
            return y, jac
        elif hessian and not jacobian:
            return y, hess
        else: 
            return y


   

class Interval(HalfEdge):
    """
    Interval Class (1D equivalent of a Cell)
    """
    def __init__(self, vertex_left, vertex_right, n_children=2, \
                 regular=True, parent=None, position=None, forest=None, \
                 periodic=False):
        """
        Constructor
        """
        assert vertex_left.dim()==1 and vertex_right.dim()==1, \
            'Input "half_edge" should be one dimensional.'
        
        HalfEdge.__init__(self, vertex_left, vertex_right, \
                          n_children=n_children, regular=regular,\
                          parent=parent, position=position, forest=forest,\
                          periodic=periodic)
        
        
    def get_vertices(self):
        """
        Return interval endpoints
        """
        return [self.base(), self.head()]

    
    def get_vertex(self, position):
        """
        Return a given vertex
        """
        assert position in [0,1], 'Position should be 0 or 1.'
        return self.base() if position==0 else self.head()
    

    def assign_previous(self, prev):
        """
        Assign a previous interval
        """
        if prev is not None:
            assert isinstance(prev, Interval), \
                'Input "prev" should be an Interval.'
        HalfEdge.assign_previous(self, prev)
    
    
    def assign_next(self, nxt):
        """
        Assign the next interval
        """
        if nxt is not None:
            assert isinstance(nxt, Interval), \
                'Input "nxt" should be an Interval.'
        HalfEdge.assign_next(self,nxt)
     
     
    def get_neighbor(self, pivot, subforest_flag=None):
        """
        Returns the neighboring interval
        
        Input:
        
            pivot: int, 0 (=left) or 1 (=right)
            
            subforest_flag (optional): marker to specify submesh

        Note that neighbors in 1D can live on different levels. 
        """
        #
        # Pivot is a vertex
        # 
        if isinstance(pivot, Vertex):
            if pivot==self.base():
                pivot = 0
            elif pivot==self.head():
                pivot = 1
            else:
                raise Exception('Vertex not an interval endpoint')
        
        #
        # Move left or right
        # 
        if pivot == 0:
            #
            # Left neighbor
            # 
            itv = self
            prev = itv.previous()
            #
            # Go up the tree until thre is a "previous"
            # 
            while prev is None:
                if itv.has_parent():
                    #
                    # Go up one level and check
                    # 
                    itv = itv.get_parent()
                    prev = itv.previous()
                else:
                    #
                    # No parent: check whether vertex is periodic 
                    # 
                    if itv.base().is_periodic():
                        for pair in itv.base().get_periodic_pair():
                            prev, dummy = pair 
                    else:
                        return None
            #
            # Go down tree (to the right) as far as you can 
            #
            nxt = prev 
            while nxt.has_children(flag=subforest_flag):
                nxt = nxt.get_child(nxt.n_children()-1)
            return nxt
             
        elif pivot==1:
            #
            # Right neighbor
            # 
            itv = self
            nxt = itv.next()
            #
            # Go up the tree until there is a "next"
            # 
            while nxt is None:
                if itv.has_parent():
                    #
                    # Go up one level and check
                    #
                    itv = itv.get_parent()
                    nxt = itv.next()
                else:
                    #
                    # No parent: check whether vertex is periodic
                    # 
                    if itv.head().is_periodic():
                        for nxt, dummy in itv.head().get_periodic_pair():
                            pass
                    else:
                        return None
            #
            # Go down tree (to the left) as far as you can
            # 
            prev = nxt
            while prev.has_children(flag=subforest_flag):
                prev = prev.get_child(0)
            return prev
                 
    
    def split(self, n_children=None):
        """
        Split a given interval into subintervals
        """                 
        #
        # Determine interval endpoints
        # 
        x0, = self.base().coordinates()
        x1, = self.head().coordinates()
        n = self.n_children()
        #
        # Loop over children
        # 
        for i in range(n):
            #
            # Determine children base and head Vertices
            # 
            if i==0:
                base = self.base()   
            if i==n-1:
                head = self.head()
            else:
                head = Vertex(x0+(i+1)*(x1-x0)/n)
            #     
            # Define new child interval
            # 
            subinterval = Interval(base, head, parent=self, \
                                   regular=self.is_regular(),\
                                   position=i, n_children=n_children)
            #
            # Store in children
            # 
            self._children[i] = subinterval
            #
            # The head of the current subinterval 
            # becomes the base of the next one 
            base = subinterval.head()
        
        #
        # Assign previous/next
        #
        for child in self.get_children(): 
            i = child.get_node_position()
            #
            # Assign previous
            #
            if i != 0:
                # Middle children
                child.assign_previous(self.get_child(i-1))
                    
    
    def locate_point(self, point, flag=None):
        """
        Returns the smallest subinterval that contains a given point
        
        Note: The flag is a subtree type flag. If current interval is 
            not flagged, then its children will also not be flagged.
        """
        #
        # Check that self contains the point
        #
        if flag is not None:
            if not self.is_marked(flag):
                return None  
        if self.contains_points(point):
            #
            # Look for child that contains the point
            #  
            if self.has_children(flag=flag):
                for child in self.get_children(flag=flag):
                    if child.contains_points(point):
                        #
                        # Recursion
                        # 
                        return child.locate_point(point, flag=flag)  
            else:
                #
                # Current cell is the smallest that contains point
                # 
                return self 
        else:
            #
            # Point is not contained in self 
            # 
            return None        
    
        
    def reference_map(self, x, jacobian=False, hessian=False, mapsto='physical'):
        """
        Map points x from the reference to the physical domain or vice versa
        
        Inputs:
        
            x: double, (n,) array or a list of points.
            
            jacobian: bool, return jacobian for mapping.
            
            mapsto: str, 'physical' (map from reference to physical), or 
                'reference' (map from physical to reference).
                
                
        Outputs:
        
            y: double, (n,) array of mapped points
            
            jac: list of associated gradients
            
            hess: list of associated Hessians (always 0).
        """
        # 
        # Convert input to array
        # 
        x = convert_to_array(x,dim=1)
            
        #
        # Compute mapped points
        # 
        n = len(x)    
        x0, = self.get_vertex(0).coordinates()
        x1, = self.get_vertex(1).coordinates()
        if mapsto == 'physical':
            y = x0 + (x1-x0)*x
        elif mapsto == 'reference':
            y = (x-x0)/(x1-x0)
        
        #
        # Compute the Jacobian
        # 
        if jacobian:
            if mapsto == 'physical':
                #
                # Derivative of mapping from refence to physical cell
                # 
                jac = [(x1-x0)]*n
            elif mapsto == 'reference':
                #
                # Derivative of inverse map
                # 
                jac = [1/(x1-x0)]*n


        # 
        # Compute the Hessian (linear mapping, so Hessian = 0)
        #
        hess = list(np.zeros(n))
         
        #
        # Return output
        # 
        if jacobian and hessian:
            return y, jac, hess
        elif jacobian and not hessian:
            return y, jac
        elif hessian and not jacobian:
            return y, hess
        else: 
            return y
     
                        
class Cell(Tree):
    """
    Cell object: A two dimensional polygon 
    
    """
    def __init__(self, half_edges, n_children=0, parent=None, position=None, grid=None):            
        """
        Constructor
        
        Inputs:
        
            half_edges: HalfEdge, list of half-edges that determine the cell
 
        """    
        Tree.__init__(self, n_children=n_children, parent=parent, \
                      position=position, forest=grid)
        
        # =====================================================================
        # Half-Edges
        # =====================================================================
        assert type(half_edges) is list, 'Input "half_edges" should be a list.'
        
        #
        # 2D Cells are constructed from lists of HalfEdges 
        #
        for he in half_edges:
            assert isinstance(he, HalfEdge), 'Not a HalfEdge.'
              
        self._half_edges = half_edges
        for he in self._half_edges:
            # Assign self as incident cell
            he.assign_cell(self)
        
        #
        # String half-edges together
        #     
        n_hes = self.n_half_edges()
        for i in range(n_hes):
            he_nxt = self._half_edges[(i+1)%n_hes]
            he_cur = self._half_edges[i]    
            he_cur.assign_next(he_nxt)
            he_nxt.assign_previous(he_cur)
        #
        # Check that base of first halfedge coincides with head of last
        #
        assert half_edges[0].base()==half_edges[-1].head(),\
            'HalfEdges should form a closed loop.'   
        #
        # Check winding order
        #         
        self.check_winding_order()


    def n_half_edges(self):
        """
        Return the number of half_edges
        """
        return len(self._half_edges)

    
    def get_half_edge(self, position):
        """
        Return specific half_edge
        """
        assert position>=0 and position<self.n_half_edges(),\
            'Input "position" incompatible with number of HalfEdges'
        return self._half_edges[position]
    
    
    def get_half_edges(self):
        """
        Iterate over half-edges 
        """
        return self._half_edges
    
    
    def incident_half_edge(self, vertex, reverse=False):
        """
        Returns the edge whose head (base) is the given vertex
        """
        assert isinstance(vertex, Vertex), \
            'Input "vertex" should be of type Vertex.'
        
        for half_edge in self.get_half_edges():
            if reverse:
                #
                # HalfEdge's base coincides with vertex
                # 
                if half_edge.base()==vertex:
                    return half_edge
            else:
                #
                # HalfEdge's head coincides with vertex
                # 
                if half_edge.head()==vertex: 
                    return half_edge
        #
        # No such HalfEdge
        # 
        return None
            
    
    def area(self):
        """
        Determine the area of the polygon
        """   
        area = 0
        for half_edge in self.get_half_edges():
            x0, y0 = half_edge.base().coordinates()
            x1, y1 = half_edge.head().coordinates()
            area += (x0+x1)*(y1-y0)
        return 0.5*area
    
    
    def bounding_box(self):
        """
        Returns the cell's bounding box in the form of a tuple (x0,x1,y0,y1), 
        so that the cell is contained in the rectangle [x0,x1]x[y0,y1]
        """  
        xy = convert_to_array(self.get_vertices(), 2)
        x0 = np.min(xy[:,0], axis=0)
        x1 = np.max(xy[:,0], axis=0)
        y0 = np.min(xy[:,1], axis=0)
        y1 = np.max(xy[:,1], axis=0)
        return x0, x1, y0, y1
        
        
    def check_winding_order(self):
        """
        Check whether the winding order is correct
        """
        winding_error = 'Cell vertices not ordered correctly.'
        area = self.area()
        assert area > 0, winding_error
        
    
    def n_vertices(self):
        """
        Return the number of vertices
        """
        return self.n_half_edges()
    
    
    def get_vertex(self, position):
        """
        Return a specific vertex
        """    
        assert position < self.n_vertices(), 'Input "position" incorrect.'
        half_edge = self.get_half_edge(position)
        return half_edge.base()
    
    
    def get_vertices(self):
        """
        Returns the vertices of the current cell. 
                
        Outputs: 
        
            vertices: list of vertices    
        """            
        return [half_edge.base() for half_edge in self.get_half_edges()]
          

    def get_neighbors(self, pivot, flag=None):
        """
        Returns all neighboring cells about a given pivot
        
        Input:
        
            pivot: Vertex/HalfEdge, 
          
                - If the pivot is a HalfEdge, then neighbors are cells 
                  containing the twin HalfEdge
          
                - If it's a Vertex, then the neighbors are all cells (of
                  the "same" size) that contain the vertex
          
            flag: marker - only return neighbors with given marker

          
        Output:
        
            neighbor(s): 
            
                - If the pivot is a HalfEdge, then return a Cell/None
                
                - If the pivot is a Vertex, then return a list of Cells 
            
            
        Note: Neighbors are chosen via shared edges, which means
            
            Not OK,         Ok           + is a neighbor of o, but x is not
            -----          -----         -------------
            | x |          | x |         | + |       |
             ---*----      ----*----     -----   x
                | x |      | x | x |     | o |       |
                -----      ---------     -------------
        """
        if isinstance(pivot, HalfEdge):
            # =================================================================
            # Direction is given by a HalfEdge
            # =================================================================
            twin = pivot.twin()
            if twin is not None:
                #
                # Halfedge has a twin
                #
                neighbor = twin.cell()
                if flag is not None:
                    if neighbor.is_marked(flag):
                        return neighbor
                    else:
                        return None
                else:
                    return neighbor
                
        elif isinstance(pivot, Vertex):
            # =================================================================
            # Direction is determined by a Vertex
            # =================================================================
            #
            # Anti-clockwise
            #
            neighbors = []
            cell = self
            while True:
                #
                # Get neighbor
                # 
                half_edge = cell.incident_half_edge(pivot)
                neighbor = cell.get_neighbors(half_edge)
                #
                # Move on
                # 
                if neighbor is None:
                    break
                elif neighbor==self:
                    #
                    # Full rotation or no neighbors
                    # 
                    return neighbors
                else:
                    #
                    # Got at neighbor!
                    # 
                    neighbors.append(neighbor)
                    cell = neighbor
                    if pivot.is_periodic() and len(pivot.get_periodic_pair(cell))!=0:
                        pivot = pivot.get_periodic_pair(cell)[0]
            #    
            # Clockwise
            #
            neighbors_clockwise = []
            cell = self
            while True:
                #
                # Get neighbor
                #
                half_edge = cell.incident_half_edge(pivot, reverse=True)
                neighbor = cell.get_neighbors(half_edge)
                #
                # Move on
                # 
                if neighbor is None:
                    break
                elif neighbor==self:
                    #
                    # Full rotation or no neighbors
                    # 
                    return neighbors
                else:
                    #
                    # Got a neighbor
                    # 
                    neighbors_clockwise.append(neighbor)
                    cell = neighbor
                    if pivot.is_periodic() and len(pivot.get_periodic_pair(cell))!=0:
                        pivot = pivot.get_periodic_pair(cell)[0]
            #
            # Combine clockwise and anticlockwise neighbors
            #
            neighbors.extend(reversed(neighbors_clockwise))
            if flag is not None:
                return [nb for nb in neighbors if nb.is_marked(flag)]
            else:
                return neighbors
            
            
    def contains_points(self, points, tol=1e-10):
        """
        Determine whether the given cell contains a point
        
        Input: 
        
            point: tuple (x,y), list of tuples, or (n,2) array
            
        Output: 
        
            in_cell: boolean array (n,1), True if cell contains points, 
            False otherwise
        """                            
        xy = convert_to_array(points, 2)
        x,y = xy[:,0], xy[:,1]
            
        n_points = len(x)
        in_cell = np.ones(n_points, dtype=np.bool)
          
        for half_edge in self.get_half_edges():
            #
            # Traverse vertices in counter-clockwise order
            # 
            x0, y0 = half_edge.base().coordinates()
            x1, y1 = half_edge.head().coordinates()
        
            # Determine which points lie outside cell
            pos_means_left = (y-y0)*(x1-x0)-( x-x0)*(y1-y0) 
            in_cell[pos_means_left<-tol] = False
        
        """
        if len(in_cell)==1:
            return in_cell[0]
        else:
            return in_cell
        """
        return in_cell

    
    def intersects_line_segment(self, line):
        """
        Determine whether cell intersects with a given line segment
        
        Input: 
        
            line: double, list of two tuples (x0,y0) and (x1,y1)
            
        Output:
        
            intersects: bool, true if line segment and cell intersect
            
        Modified: 06/04/2016
        
        """               
        #
        # Check whether line is contained in rectangle
        # 
        if all(self.contains_points([line[0], line[1]])):
            return True
        #
        # Check whether line intersects with any cell half_edge
        # 
        for half_edge in self.get_half_edges():
            if half_edge.intersects_line_segment(line):
                return True
        #
        # If function has not terminated yet, there is no intersection
        #     
        return False

      
class QuadCell(Cell, Tree):
    """
    Quadrilateral cell
    """
    def __init__(self, half_edges, parent=None, position=None, grid=None):
        """
        Constructor
        """
        assert len(half_edges)==4, 'QuadCells contain only 4 HalfEdges.'
        
        Cell.__init__(self, half_edges, n_children=4, parent=parent,
                      position=position, grid=grid)
        
        #
        # Check whether cell's parent is a rectangle
        # 
        if self.has_parent():
            is_rectangle = self.get_parent().is_rectangle()
        elif self.in_forest() and self.get_forest().is_rectangular:
            is_rectangle = True
        else:
            is_rectangle = True
            for i in range(4):
                he = half_edges[i]
                he_nxt = half_edges[(i+1)%4]
                if abs(np.dot(he.to_vector(), he_nxt.to_vector())) > 1e-9:
                    is_rectangle = False
                    break
        self._is_rectangle = is_rectangle
    

    def is_rectangle(self):
        """
        Is the cell a rectangle?
        """
        return self._is_rectangle
    
        
    def split(self, flag=None):
        """
        Split QuadCell into 4 subcells (and mark children with flag)
        """
        assert not self.has_children(), 'Cell already split.'
        
        #
        # Middle Vertex
        #
        xx = convert_to_array(self.get_vertices())
        v_m = Vertex((np.mean(xx[:,0]),np.mean(xx[:,1]))) 
        
        interior_half_edges = []
        for half_edge in self.get_half_edges():
            #
            # Split each half_edge
            #
            if not half_edge.has_children():
                half_edge.split()     
            #
            # Form new HalfEdges to and from the center
            # 
            h_edge_up = HalfEdge(half_edge.get_child(0).head(),v_m)
            h_edge_down = h_edge_up.make_twin()
            
            # Add to list
            interior_half_edges.append([h_edge_up, h_edge_down])   
        #
        # Form new cells using new half_edges
        # 
        i = 0
        for half_edge in self.get_half_edges():
            #
            # Define Child's HalfEdges
            # 
            h1 = half_edge.get_child(0)
            h2 = interior_half_edges[i][0]
            h3 = interior_half_edges[(i-1)%self.n_half_edges()][1]
            h4 = half_edge.previous().get_child(1)
            hes = deque([h1, h2, h3, h4])
            hes.rotate(i)
            hes = list(hes)
            #
            # Define new QuadCell
            # 
            self._children[i] = QuadCell(hes, parent=self, position=i)

            # Increment counter
            i += 1
            
        if flag is not None:
            for child in self.get_children():
                child.mark(flag)
        #
        # Pair up periodic vertices
        #
        for half_edge in self.get_half_edges():
            for he_child in half_edge.get_children():
                if he_child.is_periodic() and he_child.twin() is not None:
                    he_child.pair_periodic_vertices()
                
                
    def locate_point(self, point, flag=None):
        """
        Returns the smallest subcell that contains a given point
        """
        #
        # Check that self contains the point
        # 
        if self.contains_points(point):
            #
            # Look for child that contains the point
            #  
            if self.has_children(flag=flag):
                for child in self.get_children(flag=flag):
                    if child.contains_points(point):
                        #
                        # Recursion
                        # 
                        return child.locate(point)  
            else:
                #
                # Current cell is the smallest that contains point
                # 
                return self 
        else:
            #
            # Point is not contained in self 
            # 
            return None
    
    
    def reference_map(self, x_in, jacobian=False, 
                      hessian=False, mapsto='physical'):
        """
        Bilinear map between reference cell [0,1]^2 and physical cell
        
        Inputs: 
        
            x_in: double, list of of n (2,) arrays of input points, either in 
                the physical cell (if mapsto='reference') or in the reference
                cell (if mapsto='physical'). 
                
            jacobian: bool, specify whether to return the Jacobian of the
                transformation.
                
            hessian: bool, specify whether to return the Hessian tensor of the
                transformation.
            
            mapsto: str, 'reference' if mapping onto the refence cell [0,1]^2
                or 'physical' if mapping onto the physical cell. Default is 
                'physical'
                
                
        Outputs:
        
            x_mapped: double, (n,2) array of mapped points
            
            jac: double, list of n (2,2) arrays of jacobian matrices 
            
            hess: double, list of n (2,2,2) arrays of hessian matrices           
        """
        #
        # Convert input to array
        # 
        x_in = convert_to_array(x_in, dim=2)
        n = x_in.shape[0]
        assert x_in.shape[1]==2, 'Input "x" has incorrect dimension.'
        
        #
        # Get cell corner vertices
        #  
        x_verts = convert_to_array(self.get_vertices())
        p_sw_x, p_sw_y = x_verts[0,:]
        p_se_x, p_se_y = x_verts[1,:]
        p_ne_x, p_ne_y = x_verts[2,:]
        p_nw_x, p_nw_y = x_verts[3,:]

        if mapsto=='physical':        
            #    
            # Map points from [0,1]^2 to the physical cell, using bilinear
            # nodal basis functions 
            #
            
            # Points in reference domain
            s, t = x_in[:,0], x_in[:,1] 
            
            # Mapped points
            x = p_sw_x*(1-s)*(1-t) + p_se_x*s*(1-t) +\
                p_ne_x*s*t + p_nw_x*(1-s)*t
            y = p_sw_y*(1-s)*(1-t) + p_se_y*s*(1-t) +\
                p_ne_y*s*t + p_nw_y*(1-s)*t
             
            # Store points in a list
            x_mapped = np.array([x,y]).T
            
        elif mapsto=='reference':
            #
            # Map from physical- to reference domain using Newton iteration
            #   
            
            # Points in physical domain
            x, y = x_in[:,0], x_in[:,1]
            
            # Initialize points in reference domain
            s, t = 0.5*np.ones(n), 0.5*np.ones(n) 
            n_iterations = 5
            for dummy in range(n_iterations):
                #
                # Compute residual
                # 
                rx = p_sw_x*(1-s)*(1-t) + p_se_x*s*(1-t) \
                     + p_ne_x*s*t + p_nw_x*(1-s)*t - x
                         
                ry = p_sw_y*(1-s)*(1-t) + p_se_y*s*(1-t) \
                     + p_ne_y*s*t + p_nw_y*(1-s)*t - y
                 
                #
                # Compute jacobian
                #              
                drx_ds = -p_sw_x*(1-t) + p_se_x*(1-t) + p_ne_x*t - p_nw_x*t  # J11 
                dry_ds = -p_sw_y*(1-t) + p_se_y*(1-t) + p_ne_y*t - p_nw_y*t  # J21
                drx_dt = -p_sw_x*(1-s) - p_se_x*s + p_ne_x*s + p_nw_x*(1-s)  # J12
                dry_dt = -p_sw_y*(1-s) - p_se_y*s + p_ne_y*s + p_nw_y*(1-s)  # J22 
                
                #
                # Newton Update: 
                # 
                Det = drx_ds*dry_dt - drx_dt*dry_ds
                s -= ( dry_dt*rx - drx_dt*ry)/Det
                t -= (-dry_ds*rx + drx_ds*ry)/Det
                
                #
                # Project onto [0,1]^2
                # 
                s = np.minimum(np.maximum(s,0),1)
                t = np.minimum(np.maximum(t,0),1)
                
            x_mapped = np.array([s,t]).T
        
        if jacobian:
            #
            # Compute Jacobian of the forward mapping 
            #
            xs = -p_sw_x*(1-t) + p_se_x*(1-t) + p_ne_x*t - p_nw_x*t  # J11 
            ys = -p_sw_y*(1-t) + p_se_y*(1-t) + p_ne_y*t - p_nw_y*t  # J21
            xt = -p_sw_x*(1-s) - p_se_x*s + p_ne_x*s + p_nw_x*(1-s)  # J12
            yt = -p_sw_y*(1-s) - p_se_y*s + p_ne_y*s + p_nw_y*(1-s)  # J22
              
            if mapsto=='physical':
                jac = [\
                       np.array([[xs[i], xt[i]],\
                                 [ys[i], yt[i]]])\
                       for i in range(n)\
                       ]
            elif mapsto=='reference':
                #
                # Compute matrix inverse of jacobian for backward mapping
                #
                Det = xs*yt-xt*ys
                sx =  yt/Det
                sy = -xt/Det
                tx = -ys/Det
                ty =  xs/Det
                jac = [ \
                       np.array([[sx[i], sy[i]],\
                                 [tx[i], ty[i]]])\
                       for i in range(n)\
                       ]
                
        if hessian:
            hess = []
            if mapsto=='physical':
                if self.is_rectangle():
                    for i in range(n):
                        hess.append(np.zeros((2,2,2)))
                else:
                    for i in range(n):
                        h = np.zeros((2,2,2))
                        xts = p_sw_x - p_se_x + p_ne_x - p_nw_x
                        yts = p_sw_y - p_se_y + p_ne_y - p_nw_y
                        h[:,:,0] = np.array([[0, xts], [xts, 0]])
                        h[:,:,1] = np.array([[0, yts], [yts, 0]])
                        hess.append(h)
            elif mapsto=='reference':
                if self.is_rectangle():
                    hess = [np.zeros((2,2,2)) for dummy in range(n)]
                else:
                    Dx = p_sw_x - p_se_x + p_ne_x - p_nw_x
                    Dy = p_sw_y - p_se_y + p_ne_y - p_nw_y
                    
                    dxt_dx = Dx*sx
                    dxt_dy = Dx*sy
                    dyt_dx = Dy*sx
                    dyt_dy = Dy*sy
                    dxs_dx = Dx*tx
                    dxs_dy = Dx*ty
                    dys_dx = Dy*tx
                    dys_dy = Dy*ty
                    
                    dDet_dx = dxs_dx*yt + dyt_dx*xs - dys_dx*xt - dxt_dx*ys
                    dDet_dy = dxs_dy*yt + dyt_dy*xs - dys_dy*xt - dxt_dy*ys
                    
                    sxx =  dyt_dx/Det - yt*dDet_dx/Det**2
                    sxy =  dyt_dy/Det - yt*dDet_dy/Det**2
                    syy = -dxt_dy/Det + xt*dDet_dy/Det**2
                    txx = -dys_dx/Det + ys*dDet_dx/Det**2
                    txy = -dys_dy/Det + ys*dDet_dy/Det**2
                    tyy =  dxs_dy/Det - xs*dDet_dy/Det**2
                    
                    for i in range(n):
                        h = np.zeros((2,2,2))
                        h[:,:,0] = np.array([[sxx[i], sxy[i]], 
                                             [sxy[i], syy[i]]])
                        
                        h[:,:,1] = np.array([[txx[i], txy[i]], 
                                             [txy[i], tyy[i]]])
                        hess.append(h)
        #
        # Return output
        #    
        if jacobian and hessian:
            return x_mapped, jac, hess
        elif jacobian and not hessian:
            return x_mapped, jac
        elif hessian and not jacobian:
            return x_mapped, hess
        else: 
            return x_mapped


class RVertex(Vertex):
    """
    Vertex on the reference cell
    """
    def __init__(self, coordinates):
        """
        Constructor
        """
        Vertex.__init__(self, coordinates)
        self.__pos = {0: None, 1: {0: None, 1: None, 2: None, 3: None}}
        self.__basis_index = None
    
    
    def set_pos(self, pos, level=0, child=None):
        """
        Set the position of the Dof Vertex
        
        Inputs: 
        
            pos: int, a number not exceeding the element's number of dofs
            
            level: int in {0,1}, number specifying the refinement level
                ( 0 = coarse, 1 = fine ).
                
            child: int in {0,1,2,3}, number specifying the child cell
        
        """
        assert level in [0,1], 'Level should be either 0 or 1.'
        if level==0:
            self.__pos[level] = pos
        if level==1:
            assert child in [0,1,2,3], 'Level=1. Child should be specified.'
            self.__pos[level][child] = pos


    def get_pos(self, level, child=None, debug=False):
        """
        Return the dof vertex's position at a given level for a given child
        """
        if debug:
            print(self.__pos)
        if level==1:
            assert child is not None, 'On fine level, child must be specified.'
            return self.__pos[level][child]
        else:
            return self.__pos[level]
        

    def set_basis_index(self, idx):
        self.__basis_index = idx
        

class RHalfEdge(HalfEdge):
    """
    HalfEdge for reference element
    """
    def __init__(self, base, head, dofs_per_edge, 
                 parent=None, position=None, twin=None):
        """
        Constructor
        """
        HalfEdge.__init__(self, base, head, parent=parent, \
                          position=position, twin=twin)
        #
        # Assign edge dof vertices
        #
        self.__dofs_per_edge = dofs_per_edge
        self.assign_edge_dof_vertices()
        
    
    def get_edge_dof_vertices(self, pos=None):
        """
        Returns all dof vertices associated with HalfEdge
        """
        if pos is None:
            return self.__edge_dof_vertices
        else:
            return self.__edge_dof_vertices[pos]
    
    
    def assign_edge_dof_vertices(self):
        if self.twin() is not None:
            #
            # Use RHalfEdge's twin's dof vertices
            # 
            assert isinstance(self.twin(),RHalfEdge), \
                'Twin should also be an RHalfEdge'
            edge_dofs = self.twin().get_edge_dof_vertices()
            edge_dofs.reverse()
        else:
            #
            # Make new dof Vertices
            # 
            dofs_per_edge = self.n_dofs()
            x0, y0 = self.base().coordinates()
            x1, y1 = self.head().coordinates()
            edge_dofs = []
            if dofs_per_edge!=0:
                h = 1/(dofs_per_edge+1)
                for i in range(dofs_per_edge):
                    #
                    # Compute coordinates for dof vertex
                    #
                    t = (i+1)*h
                    x = x0 + t*(x1-x0)
                    y = y0 + t*(y1-y0)
                    v = RVertex((x,y))
                    if self.has_parent():
                        #
                        # Check if vertex already exists
                        #
                        for v_p in self.get_parent().get_edge_dof_vertices():
                            if np.allclose(v.coordinates(),v_p.coordinates()):
                                v = v_p
                    edge_dofs.append(v)
        #
        # Store edge dof vertices
        #
        self.__edge_dof_vertices = edge_dofs
    
    
    def make_twin(self):
        """
        Returns the twin RHalfEdge
        """
        return RHalfEdge(self.head(), self.base(), self.n_dofs(), twin=self)
        
        
    def n_dofs(self):
        """
        Returns the number of dofs associated with the HalfEdge
        """
        return self.__dofs_per_edge
    
    
    def split(self):
        """
        Refine current half-edge (overwrite Tree.split)
        """
        #
        # Compute new midpoint vertex
        #
        x = convert_to_array([self.base().coordinates(),\
                              self.head().coordinates()]) 
        xm = 0.5*(x[0,:]+x[1,:]) 
        vm = RVertex(tuple(xm))
        for v in self.get_edge_dof_vertices():
            if np.allclose(vm.coordinates(), v.coordinates()):
                vm = v
        #
        # Define own children independently of neighbor
        # 
        c0 = RHalfEdge(self.base(), vm, self.n_dofs(), parent=self, position=0)
        c1 = RHalfEdge(vm, self.head(), self.n_dofs(), parent=self, position=1)  
        #
        # Save the babies
        # 
        self._children[0] = c0
        self._children[1] = c1
         
class RQuadCell(QuadCell):
    """
    Quadrilateral Reference Cell
    """
    def __init__(self, element, half_edges=None, parent=None, position=None):
        """
        Constructor 
        """
        #
        # Check if the element is correct
        #
        self.element = element
        
        # Extract numbers of degrees of freedom
        dofs_per_vertex = element.n_dofs('vertex') 
        assert dofs_per_vertex<=1, \
            'Only elements with at most one dof per vertex supported'
        #
        # Determine Cell's RHalfEdges
        # 
        if parent is None:
            #
            # Corner Vertices
            #
            vertices = [RVertex((0,0)), RVertex((1,0)), 
                        RVertex((1,1)), RVertex((0,1))]
            #
            # Reference HalfEdges
            #
            dofs_per_edge = element.n_dofs('edge')
            half_edges = []
            for i in range(4):
                he = RHalfEdge(vertices[i], vertices[(i+1)%4], dofs_per_edge)
                half_edges.append(he)
        else:
            assert half_edges is not None, 'Cell has parent. Specify RefHalfEdges.'
            
        # Define Quadcell
        QuadCell.__init__(self, half_edges, parent=parent, position=position)
        
        #
        # Assign cell dof vertices
        #
        self.assign_cell_dof_vertices()
        
        
        if not self.has_parent():
            #
            # Assign positions on coarse level
            #
            self.assign_dof_positions(0)
            
            #
            # Split
            #
            self.split()
            
            #
            # Assign positions
            # 
            self.assign_dof_positions(1)
        
        
    def split(self):
        """
        Split refQuadCell into 4 subcells
        """
        assert not self.has_children(), 'Cell already split.'
        
        #
        # Middle Vertex
        #
        xx = convert_to_array(self.get_vertices())
        v_m = RVertex((np.mean(xx[:,0]),np.mean(xx[:,1]))) 

        # Check if this vertex is contained in cell
        for v_p in self.get_dof_vertices():
            if np.allclose(v_m.coordinates(), v_p.coordinates()):
                
                # Vertex already exists
                v_m = v_p
                break
            
        
        dofs_per_edge = self.element.n_dofs('edge')
        interior_half_edges = []
        for half_edge in self.get_half_edges():
            #
            # Split each half_edge
            #
            if not half_edge.has_children():
                half_edge.split()     
            #
            # Form new HalfEdges to and from the center
            # 
            h_edge_up = RHalfEdge(half_edge.get_child(0).head(),v_m, dofs_per_edge)
            h_edge_down = h_edge_up.make_twin()
            
            # Add to list
            interior_half_edges.append([h_edge_up, h_edge_down])   
        #
        # Form new cells using new half_edges
        # 
        i = 0
        for half_edge in self.get_half_edges():
            #
            # Define Child's HalfEdges
            # key
            h1 = half_edge.get_child(0)
            h2 = interior_half_edges[i][0]
            h3 = interior_half_edges[(i-1)%self.n_half_edges()][1]
            h4 = half_edge.previous().get_child(1)
            hes = deque([h1, h2, h3, h4])
            hes.rotate(i)
            hes = list(hes)
            #hes = [h1, h2, h3, h4]
            #
            # Define new QuadCell
            # 
            self._children[i] = RQuadCell(self.element, hes, parent=self, position=i)

            # Increment counter
            i += 1
            
        
    def assign_cell_dof_vertices(self):
        """
        Assign interior dof vertices to cell
        """
        dofs_per_cell = self.element.n_dofs('cell')        
        cell_dofs = []
        if dofs_per_cell!=0:
            n = int(np.sqrt(dofs_per_cell))  # number of dofs per direction
            x0, x1, y0, y1 = self.bounding_box()
            h = 1/(n+1)  # subcell width
            for i in range(n):  # y-coordinates
                for j in range(n):  # x-coordinates
                    #
                    # Compute new Vertex
                    #
                    v_c = RVertex((x0+(j+1)*h*(x1-x0),y0+(i+1)*h*(y1-y0)))
                    
                    #
                    # Check if vertex exists within parent cell
                    # 
                    inherits_dof_vertex = False
                    if self.has_parent():
                        for v_p in self.get_parent().get_cell_dof_vertices():
                            if np.allclose(v_c.coordinates(), v_p.coordinates()):
                                cell_dofs.append(v_p)
                                inherits_dof_vertex = True
                                break
                    if not inherits_dof_vertex:
                        cell_dofs.append(v_c)
                
        self.__cell_dof_vertices = cell_dofs
    
    
    def get_cell_dof_vertices(self, pos=None): 
        """
        Return the interior dof vertices
        """                 
        if pos is None:
            return self.__cell_dof_vertices
        else:
            return self.__cell_dof_vertices[pos]
    
    
    def assign_dof_positions(self, level):
        """
        """
        
        if level==0:
            #
            # Level 0: Assign positions to vertices on coarse level
            #
            self.__dof_vertices = {0: [], 1: {0: [], 1: [], 2: [], 3: []}}
            count = 0
            
            # Corner dof vertices
            for vertex in self.get_vertices():
                if self.element.n_dofs('vertex')!=0:
                    vertex.set_pos(count, level)
                    self.__dof_vertices[level].append(vertex)
                    count += 1
            
            # HalfEdge dof vertices
            for half_edge in self.get_half_edges():
                for vertex in half_edge.get_edge_dof_vertices():
                    vertex.set_pos(count, level)
                    self.__dof_vertices[level].append(vertex)
                    count += 1
                    
            # Cell dof vertices        
            for vertex in self.get_cell_dof_vertices():
                vertex.set_pos(count, level)
                self.__dof_vertices[level].append(vertex)
                count += 1
        elif level==1:       
            #
            # Assign positions to child vertices
            #
            coarse_dofs = [i for i in range(self.element.n_dofs())]    
            for i_child in range(4):
                #
                # Add all dof vertices to one list
                #
                child = self.get_child(i_child) 
                child_dof_vertices = []
                
                # Dofs at Corners 
                for vertex in child.get_vertices():
                    if self.element.n_dofs('vertex')!=0:
                        child_dof_vertices.append(vertex)
                
                # Dofs on HalfEdges
                for half_edge in child.get_half_edges():
                    for vertex in half_edge.get_edge_dof_vertices():
                        child_dof_vertices.append(vertex)
                
                # Dofs in Cell
                for vertex in child.get_cell_dof_vertices():
                    child_dof_vertices.append(vertex)
                
                count = 0
                for vertex in child_dof_vertices: 
                    if not self.element.torn_element():
                        #
                        # Continuous Element (Dof Vertex can be inherited multiple times)
                        #
                        vertex.set_pos(count, level=level, child=i_child)
                        self.__dof_vertices[level][i_child].append(vertex)
                        count += 1
                    else:
                        #
                        # Discontinuous Element (Dof Vertex can be inherited once)
                        # 
                        if vertex in self.__dof_vertices[0]:
                            i_vertex = self.__dof_vertices[0].index(vertex)
                            if i_vertex in coarse_dofs:
                                #
                                # Use vertex within child cell
                                # 
                                vertex.set_pos(count, level=level, child=i_child)
                                self.__dof_vertices[level][i_child].append(vertex)
                                count += 1
                                
                                # Delete the entry (preventing reuse).
                                coarse_dofs.pop(coarse_dofs.index(i_vertex))
                            else:
                                #
                                # Vertex has already been used, make a new one
                                # 
                                vcopy = RVertex(vertex.coordinates())
                                vcopy.set_pos(count, level=level, child=i_child)
                                self.__dof_vertices[level][i_child].append(vcopy)
                                count += 1
                        else:
                            #
                            # Not contained in coarse vertex set
                            # 
                            vertex.set_pos(count, level=level, child=i_child)
                            self.__dof_vertices[level][i_child].append(vertex)
                            count += 1
                                
    
    def get_dof_vertices(self, level=0, child=None, pos=None):
        """
        Returns all dof vertices in cell
        """
        if level==0:
            return self.__dof_vertices[0]
        elif level==1:
            assert child is not None, 'On level 1, child must be specified.'
            if pos is None:
                return self.__dof_vertices[1][child]
            else:
                return self.__dof_vertices[1][child][pos]
        
        
class RInterval(Interval):
    def __init__(self, element, base=None, head=None, 
                 parent=None, position=None):
        """
        Constructor
        """    
        assert element.dim()==1, 'Element must be one dimensional'
        self.element = element
        
        if parent is None:
            base = RVertex(0)
            head = RVertex(1)
        else:
            assert isinstance(head, RVertex), 'Input "head" must be an RVertex.'
            assert isinstance(base, RVertex), 'Input "base" must be an RVertex.'
        
        Interval.__init__(self, base, head, parent=parent, position=position)
            
        
        #
        # Assign cell dof vertices
        #
        self.assign_cell_dof_vertices()
        
        
        if not self.has_parent():
            #
            # Assign positions on coarse level
            #
            self.assign_dof_positions(0)
            
            #
            # Split
            #
            self.split()
            
            #
            # Assign positions
            # 
            self.assign_dof_positions(1)
        
     
     
    
    def split(self):
        """
        Split a given interval into 2 subintervals
        """                 
        #
        # Determine interval endpoints
        # 
        x0, = self.base().coordinates()
        x1, = self.head().coordinates()
        n = self.n_children()
        #
        # Loop over children
        # 
        for i in range(n):
            #
            # Determine children base and head Vertices
            # 
            if i==0:
                base = self.base()  
                 
            if i==n-1:
                head = self.head()
            else:
                head = RVertex(x0+(i+1)*(x1-x0)/n)
                #
                # Check whether Vertex appears in parent
                # 
                for v_p in self.get_dof_vertices():
                    if np.allclose(head.coordinates(), v_p.coordinates()):
                        head = v_p
            #     
            # Define new child interval
            # 
            subinterval = RInterval(self.element, base, head, \
                                    parent=self, position=i)
            #
            # Store in children
            # 
            self._children[i] = subinterval
            #
            # The head of the current subinterval 
            # becomes the base of the next one 
            base = subinterval.head()
        #
        # Assign previous/next
        #
        for child in self.get_children(): 
            i = child.get_node_position()
            #
            # Assign previous
            #
            if i==0:
                # Leftmost child assign own previous
                child.assign_previous(self.previous())
            else:
                # Child in the middle
                #print(child.get_node_position(), child.base().coordinates())
                #print(self.get_child(i-1).get_node_position(), child.base().coordinates())
                child.assign_previous(self.get_child(i-1))
            #
            # Assign next
            # 
            if i==n-1:
                # Rightmost child, assign own right
                child.assign_next(self.next())
          
        
    def assign_cell_dof_vertices(self):
        dofs_per_cell = self.element.n_dofs('edge')
        cell_dofs = []
        if dofs_per_cell !=0:
            #
            # Compute coordinates for cell dof vertices
            # 
            x0, = self.base().coordinates()
            x1, = self.head().coordinates()
            h = 1/(dofs_per_cell+1)
            for i in range(dofs_per_cell):
                x = x0 + (i+1)*h*(x1-x0)
                v_c = RVertex(x)
                #
                # Check if vertex exists within parent cell
                # 
                inherits_dof_vertex = False
                if self.has_parent():
                    for v_p in self.get_parent().get_cell_dof_vertices():
                        if np.allclose(v_c.coordinates(), v_p.coordinates()):
                            cell_dofs.append(v_p)
                            inherits_dof_vertex = True
                            break
                if not inherits_dof_vertex:
                    cell_dofs.append(v_c)
        self.__cell_dof_vertices = cell_dofs
        
    
    def get_cell_dof_vertices(self, pos=None):
        """
        Returns the Dofs associated with the interior of the cell
        
        Note: This function is only used during construction
        """
        if pos is None:
            return self.__cell_dof_vertices
        else:
            return self.__cell_dof_vertices[pos]
 
    
    def get_dof_vertices(self, level=0, child=None, pos=None):
        """
        Returns all dof vertices in cell
        
        Inputs: 
        
            level: int 0/1, 0=coarse, 1=fine
            
            child: int, child node position within parent (0/1)
            
            pos: int, 0,...n_dofs-1, dof number within cell
        """
        if level==0:
            return self.__dof_vertices[0]
        elif level==1:
            assert child is not None, 'On level 1, child must be specified.'
            if pos is None:
                return self.__dof_vertices[1][child]
            else:
                return self.__dof_vertices[1][child][pos]
        
        
    def assign_dof_positions(self, level):
        """
        Assigns a number to each dof vertex in the interval.
        
        Note: We only deal with bisection
        """ 
        if level==0:
            #
            # Level 0: Assign position to vertices on coarse level
            #
            self.__dof_vertices = {0: [], 1: {0: [], 1: []}}
        
            count = 0
            #
            # Add endpoints
            #
            dpv = self.element.n_dofs('vertex')
            if dpv != 0: 
                for vertex in self.get_vertices():
                    vertex.set_pos(count, level)
                    self.__dof_vertices[level].append(vertex)
                    count += 1
            #
            # Add cell dof vertices
            # 
            for vertex in self.get_cell_dof_vertices():
                vertex.set_pos(count, level)
                self.__dof_vertices[level].append(vertex)
                count += 1
        
        elif level==1:
            #
            # Assign positions to child vertices
            # 
            coarse_dofs = [i for i in range(self.element.n_dofs())]
            for i_child in range(2):
                # 
                # Add all dof vertices to a list
                #
                child = self.get_child(i_child)
                child_dof_vertices = []
                
                # Dofs at corners
                for vertex in child.get_vertices():
                    if self.element.n_dofs('vertex')!=0:
                        child_dof_vertices.append(vertex)
                        
                # Dofs in Interval
                for vertex in child.get_cell_dof_vertices():
                    child_dof_vertices.append(vertex)
                #
                # Inspect each vertex in the child, to see 
                # whether it is duplicated in the parent.
                # 
                count = 0
                for vertex in child_dof_vertices:
                    if not self.element.torn_element():
                        #
                        # Continuous Element (Dof Vertex can be inherited multiple times)
                        # 
                        vertex.set_pos(count, level=level, child=i_child)
                        self.__dof_vertices[level][i_child].append(vertex)
                        count += 1
                    else:
                        #
                        # Discontinuous Element (Dof Vertex can be inherited once)
                        # 
                        if vertex in self.__dof_vertices[0]:
                            i_vertex = self.__dof_vertices[0].index(vertex)
                            if i_vertex in coarse_dofs:
                                #
                                # Use vertex within child cell
                                # 
                                vertex.set_pos(count, level=level, child=i_child)
                                self.__dof_vertices[level][i_child].append(vertex)
                                count += 1
                                
                                # Delete the entry (preventing reuse).
                                coarse_dofs.pop(coarse_dofs.index(i_vertex))
                            else:
                                #
                                # Vertex has already been used, make a new one
                                # 
                                vcopy = RVertex(vertex.coordinates())
                                vcopy.set_pos(count, level=level, child=i_child)
                                self.__dof_vertices[level][i_child].append(vcopy)
                                count += 1
                        else:
                            #
                            # Not contained in coarse vertex set
                            # 
                            vertex.set_pos(count, level=level, child=i_child)
                            self.__dof_vertices[level][i_child].append(vertex)
                            count += 1
                                
                                       
'''    


class Mesh(object):
    """
    Mesh class, consisting of a grid (a doubly connected edge list), as well
    as a list of root cells, -half-edges and vertices. 
    
    Attributes:
    
    Methods:

    
    """
    def __init__(self, grid):
        """
        Constructor
        

class Mesh(object):
    """
    Mesh class, consisting of a grid (a doubly connected edge list), as well
    as a list of root cells, -half-edges and vertices. 
    
    Attributes:
    
    Methods:

    
    """
    def __init__(self, grid):
        """
        Constructor
        
        Inputs:
        
            grid: DCEL object, doubly connected edge list specifying
                the mesh topology. 
             
        """
        self.__grid = grid
         
        # =====================================================================
        # Vertices 
        # =====================================================================
        n_vertices = grid.points['n']
        vertices = []
        for i in range(n_vertices):
            vertices.append(Vertex(grid.points['coordinates'][i]))
        
        # =====================================================================
        # Half-edges 
        # =====================================================================
        n_he = grid.half_edges['n']
        #
        # Define Half-Edges via base and head vertices
        #
        half_edges = []
        for i in range(n_he): 
            i_base, i_head = grid.half_edges['connectivity'][i]
            v_base = grid.points['coordinates'][i_base]
            v_head = grid.points['coordinates'][i_head]
            half_edges.append(HalfEdge(Vertex(v_base), Vertex(v_head)))    
        #
        # Specify relations among Half-Edges
        # 
        for i in range(n_he):
            he = half_edges[i]
            i_prev = grid.half_edges['prev'][i] 
            i_next = grid.half_edges['next'][i]
            i_twin = grid.half_edges['twin'][i]
            
            he.assign_next(half_edges[i_next])
            he.assign_prev(half_edges[i_prev])
            
            if i_twin != -1:
                he.assign_twin(half_edges[i_twin])
                
        # =====================================================================
        # Cells 
        # =====================================================================
        n_faces = grid.faces['n']
        cells = []
        for i in range(n_faces):
            cell_type = grid.faces['type'][i]
            if cell_type == 'interval':
                cell = BCell()
                pass
            elif cell_type == 'triangle':
                #cell = TriCell()
                pass
            elif cell_type == 'quadrilateral':
                cell = QuadCell()
            else:
                unknown_cell_type = 'Unknown cell type. Use "interval", '+\
                                    '"triangle", or "quadrilateral".'
                raise Exception(unknown_cell_type)
            cells.append(cell)
            
        if grid is not None:
            #
            # grid specified 
            # 
            #assert all(i is None for i in [node, cell, dim]),\
            #'Grid specified: All other inputs should be None.'
            
            #
            # ROOT node
            # 
            dim = grid.dim()
            if dim == 1:
                node = BiNode(grid=grid)
            elif dim == 2:
                node = QuadNode(grid=grid)
            else:
                raise Exception('Only dimensions 1 and 2 supported.')
            
            #
            # Cells
            # 
            node.split()
            for pos in node._child_positions:
                #
                # ROOT cells
                #         
                if dim == 1:
                    cell = BiCell(grid=grid, position=pos)
                elif dim == 2:
                    cell = QuadCell(grid=grid, position=pos)
                                  
                child = node.children[pos]
                child.link(cell)
            
            #
            # Mark nodes, edges, and vertices
            # 
            
        elif cell is not None:
            #
            # Cell specified
            # 
            assert all(i is None for i in [node, grid, dim]),\
            'Cell specified: All other inputs should be None.'
            #
            # ROOT node linked to cell
            # 
            dim = cell.dim()
            if dim == 1:
                node = BiNode(bicell=cell)
            elif dim == 2: 
                node = QuadNode(quadcell=cell)
            else:
                raise Exception('Only dimensions 1 and 2 supported.')
            
        elif node is not None: 
            #
            # Tree specified
            # 
            assert all(i is None for i in [cell, grid, dim]),\
            'Tree specified: All other inputs should be None.'
            #
            # Default cell
            # 
            dim = node.dim()
            if dim == 1:
                cnr_vtcs = [0,1]
                cell = BiCell(corner_vertices=cnr_vtcs)
            elif dim == 2:
                cnr_vtcs = [0,1,0,1]
                cell = QuadCell(corner_vertices=cnr_vtcs)
            node.link(cell)
            
        elif dim is not None:
            #
            # Dimension specified
            #
            assert all(i is None for i in [node, cell, grid]),\
            'Dimension specified: All other inputs should be None.'
            #
            # Default cell
            #
            if dim == 1:
                cnr_vtcs = [0,1]
                cell = BiCell(corner_vertices=cnr_vtcs)
            elif dim == 2:
                cnr_vtcs = [0,1,0,1]
                cell = QuadCell(corner_vertices=cnr_vtcs)
            #
            # Default node, linked to cell
            #
            if dim == 1: 
                node = BiNode(bicell=cell)
            elif dim==2:
                node = QuadNode(quadcell=cell)
            else:
                raise Exception('Only dimensions 1 or 2 supported.')      
        else:
            #
            # Default cell 
            # 
            cnr_vtcs = [0,1,0,1]
            cell = QuadCell(corner_vertices=cnr_vtcs)
            node = QuadNode(quadcell=cell)
            dim = 2
            
        self.__root_node = node
        self.grid = grid 
        self.__mesh_count = 0
        self.__dim = dim
         
    
    def dim(self):
        """
        Return the spatial dimension of the region
        """
        return self.__dim
    
    
    def depth(self):
        """
        Return the maximum refinement level
        """    
        return self.root_node().tree_depth()
    
        
    def n_nodes(self, flag=None):
        """
        Return the number of cells
        """
        if hasattr(self, '__n_cells'):
            return self.__n_cells
        else:
            self.__n_cells = len(self.__root_node.get_leaves(flag=flag))
            return self.__n_cells
    
            
    def root_node(self):
        """
        Return tree node used for mesh
        """
        return self.__root_node
     
        
    def boundary(self, entity, flag=None):
        """
        Returns a set of all boundary entities (vertices/edges)
        
        Input:
        
            entity: str, 'vertices', 'edges', or 'quadcells'
            
            flag: 
            
        TODO: Add support for tricells
        """
        boundary = set()
        print(entity)
        print(len(boundary))
        for node in self.root_node().get_leaves(flag=flag):
            cell = node.cell()
            for direction in ['W','E','S','N']:
                # 
                # Look in 4 directions
                # 
                if node.get_neighbor(direction) is None:
                    if entity=='quadcells':
                        boundary.add(cell)
                        break
                    
                    edge = cell.get_edges(direction)
                    if entity=='edges':
                        boundary.add(edge)
                        
                    if entity=='vertices':
                        for v in edge.vertices():
                            boundary.add(np.array(v.coordinates()))
        return boundary
                        
        
    def bounding_box(self):
        """
        Returns the mesh's bounding box
        
        Output:
        
            box: double,  [x_min, x_max, y_min, y_max] if mesh is 2d
                and [x_min, x_max] if mesh is 1d. 
        """
        root = self.root_node()
        if root.grid is not None:
            #
            # DCEL on coarsest level
            # 
            grid = root.grid
            if self.dim() == 1:
                x_min, x_max = grid.points['coordinates'][[0,-1]]
                return [x_min, x_max]
            elif self.dim() == 2:
                #
                # Determine bounding box from boundary points
                # 
                i_vbnd = grid.get_boundary_points()
                v_bnd = []
                for k in i_vbnd:
                    v_bnd.append( \
                        grid.points['coordinates'][i_vbnd[k]].coordinates())
                v_bnd = np.array(v_bnd) 
                x_min, x_max = v_bnd[:,0].min(), v_bnd[:,0].max()
                y_min, y_max = v_bnd[:,1].min(), v_bnd[:,1].max()
                return [x_min, x_max, y_min, y_max] 
        else:
            #
            # No DCEL: Use Cell
            # 
            cell = root.cell()
            if cell.dim()==1:
                x_min, x_max = cell.get_vertices(pos='corners', as_array=True)
                return [x_min, x_max]
            elif cell.dim()==2:
                vbnd = cell.get_vertices(pos='corners', as_array=True)
                x_min, x_max = vbnd[:,0].min(), vbnd[:,0].max()
                y_min, y_max = vbnd[:,1].min(), vbnd[:,1].max()
                return [x_min, x_max, y_min, y_max]
            else:
                raise Exception('Only 1D and 2D supported.')
                    
        
    def unmark_all(self, flag=None, nodes=False, cells=False, edges=False, 
                   vertices=False, all_entities=False):
        """
        Unmark all nodes, cells, edges, or vertices. 
        """
        if all_entities:
            # 
            # Unmark everything
            # 
            nodes = True
            cells = True
            edges = True
            vertices = True
               
        for node in self.root_node().traverse():
            if nodes:
                #
                # Unmark node
                #
                node.unmark(flag=flag, recursive=True)
            if cells:
                #
                # Unmark quad cell
                #
                node.cell().unmark(flag=flag, recursive=True)
            if edges:
                #
                # Unmark quad edges
                #
                for edge in node.cell().edges.values():
                    edge.unmark(flag=flag)
            if vertices:
                #
                # Unmark quad vertices
                #
                for vertex in node.cell().vertices.values():
                    vertex.unmark(flag=flag)
                
    
    def iter_quadedges(self, flag=None, nested=False):
        """
        Iterate over cell edges
        
        Output: 
        
            quadedge_list, list of all active cell edges
       
       
        """
        quadedge_list = []
        #
        # Unmark all edges
        # 
        self.unmark_all(quadedges=True)
        for cell in self.iter_quadcells(flag=flag, nested=nested):
            for edge_key in [('NW','SW'),('SE','NE'),('SW','SE'),('NE','NW')]:
                edge = cell.edges[edge_key]
                if not(edge.is_marked()):
                    #
                    # New edge: add it to the list
                    # 
                    quadedge_list.append(edge)
                    edge.mark()
        #
        # Unmark all edges again
        #             
        self.unmark_all(quadedges=True)
        return quadedge_list
        
                    
    def quadvertices(self, coordinate_array=True, flag=None, nested=False):
        """
        Iterate over quad cell vertices
        
        Inputs: 
        
            coordinate_array: bool, if true, return vertices as arrays 
            
            nested: bool, traverse tree depthwise
        
        Output: 
        
            quadvertex_list, list of all active cell vertices
        """
        quadvertex_list = []
        #
        # Unmark all vertices
        # 
        self.unmark_all(quadvertices=True)
        for cell in self.iter_quadcells(flag=flag, nested=nested):
            for direction in ['SW','SE','NW','NE']:
                vertex = cell.vertices[direction]
                if not(vertex.is_marked()):
                    #
                    # New vertex: add it to the list
                    #
                    quadvertex_list.append(vertex)
                    vertex.mark()
        self.unmark_all(quadvertices=True)
        if coordinate_array:
            return np.array([v.coordinates() for v in quadvertex_list])
        else:
            return quadvertex_list
        
        
    def refine(self, flag=None):
        """
        Refine mesh by splitting marked LEAF nodes
        """ 
        for leaf in self.root_node().get_leaves(flag=flag):
            leaf.split()
            
    
    
    def coarsen(self, flag=None):
        """
        Coarsen mesh by merging marked LEAF nodes. 
        
        Inputs: 
        
            flag: str/int, marker flag.
                
                If flag is specified, merge a node if all 
                of its children are flagged.
                
                If no flag is specified, merge nodes so that 
                mesh depth is reduced by 1.   
        """
        root = self.root_node()
        if flag is None:
            tree_depth = root.tree_depth()
            for leaf in root.get_leaves():
                if leaf.depth == tree_depth:
                    leaf.parent.merge()
        else:
            for leaf in root.get_leaves(flag=flag):
                parent = leaf.parent
                if all(child.is_marked(flag=flag) \
                       for child in parent.get_children()):
                    parent.merge()
    
    
    def record(self,flag=None):
        """
        Mark all mesh nodes with flag
        """
        count = self.__mesh_count
        for node in self.root_node().traverse(mode='breadth-first'):
            if flag is None:
                node.mark(count)
            else:
                node.mark(flag)
        self.__mesh_count += 1
    
    
    def n_meshes(self):
        """
        Return the number of recorded meshes
        """
        return self.__mesh_count 
    
'''


class DCEL(object):
    """
    Description: Doubly connected edge list
    
    Attributes:
    
            
        __dim: int, dimension of grid
    
        format: str, version of mesh file
        
        is_rectangular: bool, specifying whether 2D grid has rectangular faces
        
        subregions: struct, encoding the mesh's subregions, with fields:
        
            n: int, number of subregions
            
            dim: int, dimension of subregion
            
            tags: int, tags of subregions
            
            names: str, names of subregions

        
        points: struct, encoding the mesh's vertices, with fields:
        
            n: int, number of points
            
            n_dofs: int, number of dofs associated with point
            
            tags: tags associated with vertices 
            
                phys: int list, indicating membership to one of the
                    physical subregions listed above.
                    
                geom: int list, indicating membership to certain 
                    geometric entities. 
                    
                partition: int, list indicating membership to certain
                    mesh partitions. 
        
            half_edge: int array, pointing to a half-edge based at
                point.
                
            coordinates: double, list of tuples
            
                
        edges: struct, encoding the mesh's edges associated with 
            specific subregions, w. fields: 
            
            n: int, number of edges
            
            n_dofs: int, number of dofs associated with edge
            
            tags: struct, tags associated with edges (see points)
            
            connectivity: int, list of sets containing edge vertices
                
            half_edge: int, array pointing to associated half-edge
                                   
            Edges: Edge list in same order as connectivity
            
        
        half_edges: struct, encoding the mesh's half-edges
        
            n: int, number of half-edges
            
            n_dofs: int, number of dofs associated with half_edge
            
            tags: struct, tags associated with half-edges (see points)
            
            connectivity: int, list pointing to initial and final 
                vertices [v1,v2].
                
            prev: int, array pointing to the preceding half-edge
            
            next: int, array pointing to the next half-edge
            
            twin: int, array pointing to the reversed half-edge
            
            edge: int, array pointing to an associated edge
            
            face: int, array pointing to an incident face
            
            
        faces: struct, encoding the mesh's faces w. fields:
        
            n: int, number of faces
            
            n_dofs: int, list containing number of dofs per face
            
            type: str, type of face (interval, triangle, or quadrilateral)
            
            tags: tags associated with faces (same as for points)
            
            connectivity: int, list of indices of vertices that make 
                up faces.
            
            half_edge: int, array pointing to a half-edge on the boundary
                
       
    Methods:
    
        __init__
        
        initialize_grid_structure
        
        rectangular_grid
        
        grid_from_gmsh
        
        determine_half_edges
        
        dim
        
        get_neighbor
        
        contains_node
        
    Note: The grid can be used to describe the connectivity associated with a
        ROOT Tree. 
    
    """
    def __init__(self, box=None, resolution=None, periodic=None, dim=None, 
                 x=None, connectivity=None, file_path=None, file_format='gmsh'):
        """
        Constructor
        
        Inputs:
        
            box: list of endpoints for rectangular mesh
            
                1D     [x_min, x_max]
                2D     [x_min, x_max, y_min, y_max]  
            
            resolution: tuple, with number of cells in each direction 
               
            dim: int, spatial dimension of the grid
            
            x: double, (n,) array of points in for constructing a grid 
            
            connectivity: int, list of cell connectivities
            
            file_path: str, path to mesh file
            
            file_format: str, type of mesh file (currently only gmsh)
            
            periodic: int, set containing integers 0 and/or 1.
                0 in periodic: make periodic in x-direction
                1 in periodic: make periodic in y-direction
        """
        #
        # Initialize struct
        #     
        self.is_rectangular = False
        self.is_periodic = False
        self.resolution = resolution
        self.initialize_grid_structure() 
        if file_path is not None:
            # =================================================================
            # Import grid from gmsh
            # =================================================================
            assert file_format=='gmsh', \
            'For input file_format, use "gmsh".'
            #
            # Import grid from gmsh
            # 
            self.grid_from_gmsh(file_path)    
        
        elif x is not None:
            # =================================================================
            # Generate grid from connectivity
            # =================================================================
            self.grid_from_connectivity(x, connectivity)        
        else:
            # =================================================================
            # Rectangular Grid
            # =================================================================
            #
            # Determine dimension
            #
            if dim is None:
                if resolution is not None:
                    assert type(resolution) is tuple, \
                    'Input "resolution" should be a tuple.'
                    dim = len(resolution)
                elif box is not None:
                    assert type(box) is list, 'Input "box" should be a list.' 
                    if len(box) == 2:
                        dim = 1
                    elif len(box) == 4:
                        dim = 2
                    else:
                        box_length = 'Box should be a list of length 2 or 4.'
                        raise Exception(box_length)
                else:
                    raise Exception('Unable to verify dimension of grid')
            self.__dim = dim
            #
            # Specify box
            #     
            if box is None:
                #
                # Default boundary box
                # 
                if dim==1:
                    box = [0,1]
                elif dim==2:
                    box = [0,1,0,1]
            #
            # Specify resolution
            #
            if resolution is None:
                #
                # Default resolution
                # 
                if dim==1:
                    resolution = (1,)
                elif dim==2:
                    resolution = (1,1)  
            self.is_rectangular = True
            self.rectangular_grid(box=box, resolution=resolution)
           
                
        # =====================================================================
        # Generate doubly connected edge list 
        # =====================================================================
        self.determine_half_edges()
        
        #
        # Add periodicity
        # 
        self.periodic_coordinates = {}
        if periodic is not None:
            if self.dim()==2:
                assert self.is_rectangular, \
                    'Only rectangular meshes can be made periodic'
            self.make_periodic(periodic, box)
            self.is_periodic = True
            
                            
    def initialize_grid_structure(self):
        """
        Initialize empty grid. 
        """
        self.format = None
         
        # Subregions 
        self.subregions = {'dim': [], 'n': None, 'names': [], 'tags': []}
        
        # Points
        self.points = {'half_edge': [], 'n': None, 'tags': {}, 'n_dofs': None, 
                       'coordinates': []}
        
        # Edges 
        # TODO: Remove
        self.edges = {'n': None, 'tags': {}, 'n_dofs': None, 'connectivity': []}
        
        
        # Half-Edges
        self.half_edges = {'n': None, 'tags': {}, 'n_dofs': None, 
                           'connectivity': [], 'prev': [], 'next': [],
                           'twin': [], 'edge': [], 'face': [], 'position': []}
        
        # Faces
        self.faces = {'n': None, 'type': [], 'tags': {}, 'n_dofs': [], 
                      'connectivity': []}
        
    
    def rectangular_grid(self, box, resolution):
        """
        Construct a grid on a rectangular region
        
        Inputs:
        
            box: int, tuple giving bounding vertices of rectangular domain:
                (x_min, x_max) in 1D, (x_min, x_max, y_min, y_max) in 2D. 
            
            resolution: int, tuple giving the number of cells in each direction
            
        """
        assert type(resolution) is tuple, \
            'Input "resolution" should be a tuple.'
        dim = len(resolution)    
        if dim == 1:
            # =================================================================
            # One dimensional grid
            # =================================================================
            
            # Generate DCEL
            x_min, x_max = box
            n_points = resolution[0] + 1 
            x = np.linspace(x_min, x_max, n_points)
            
            # Store grid information
            self.__dim = 1
            self.points['coordinates'] = [(xi,) for xi in x]
            self.points['n'] = n_points
            
        elif dim  == 2:
            # =================================================================
            # Two dimensional grid
            # =================================================================
            self.__dim = 2
            x_min, x_max, y_min, y_max = box
            nx, ny = resolution
            n_points = (nx+1)*(ny+1)
            self.points['n'] = n_points
            #
            # Record vertices
            # 
            x = np.linspace(x_min, x_max, nx+1)
            y = np.linspace(y_min, y_max, ny+1)
            for i_y in range(ny+1):
                for i_x in range(nx+1):
                    self.points['coordinates'].append((x[i_x],y[i_y]))
            #
            # Face connectivities
            #         
            # Vertex indices
            idx = np.arange((nx+1)*(ny+1)).reshape(ny+1,nx+1).T
            for i_y in range(ny):
                for i_x in range(nx):
                    fv = [idx[i_x,i_y], idx[i_x+1,i_y], 
                          idx[i_x+1,i_y+1], idx[i_x,i_y+1]]
                    self.faces['connectivity'].append(fv)
            self.faces['n'] = nx*ny
            self.faces['type'] = ['quadrilateral']*self.faces['n']
            
        else:
            raise Exception('Only 1D/2D supported.') 
        
    
    def grid_from_connectivity(self, x, connectivity):
        """
        Construct grid from connectivity information
        """
        points = self.points        
        x = convert_to_array(x, dim=1)
        dim = x.shape[1]
        if dim==1:
            #
            # 1D 
            #
            self.__dim = 1
            #
            # Store points
            # 
            x = np.sort(x, axis=0)  # ensure the vector is sorted
            points['coordinates'] = [(xi[0],) for xi in x]
            points['n'] = len(x)
        elif dim==2:
            #
            # 2D
            # 
            self.__dim = 2
            # 
            # Store points 
            # 
            n_points = x.shape[0]
            points['coordinates'] = [(x[i,0],x[i,1]) for i in range(n_points)]
            points['n'] = n_points
            #
            # Store faces
            #  
            faces = self.faces
            assert connectivity is not None, 'Specify connectivity.'
            assert type(connectivity) is list, \
                'Connectivity should be passed as a list.'
            n_faces = len(connectivity)
            faces['n'] = n_faces
            for i in range(n_faces):
                assert type(connectivity[i]) is list, \
                    'Connectivity entries should be lists'
                faces['connectivity'].append(connectivity[i])
                faces['n_dofs'].append(len(connectivity[i]))
            
            
    def grid_from_gmsh(self, file_path):
        """
        Import computational mesh from a .gmsh file and store it in the grid.
        
        Input:
        
            file_path: str, path to gmsh file
             
        """     
        points = self.points
        edges = self.edges
        faces = self.faces
        subregions = self.subregions
        #
        # Initialize tag categories
        #
        for entity in [points, edges, faces]:
            entity['tags'] = {'phys': [], 'geom': [], 'partition': []}
            
        with open(file_path, 'r') as infile:
            while True:
                line = infile.readline()
                # 
                #  Mesh format
                # 
                if line == '$MeshFormat\n':
                    # Read next line
                    line = infile.readline()
                    self.format = line.rstrip()
                    # TODO: Put an assert statement here to check version
                    while line != '$EndMeshFormat\n':
                        line = infile.readline()
                
                line = infile.readline()
                # 
                #  Subregions
                # 
                
                if line == '$PhysicalNames\n':
                    #
                    # Record number of subregions
                    #
                    line = infile.readline()
                    subregions['n'] = int(line.rstrip())
                    line = infile.readline()
                    while True:
                        if line == '$EndPhysicalNames\n':
                            line = infile.readline()
                            break
                        #
                        # Record names, dimensions, and tags of subregions
                        # 
                        words = line.split()
                        name = words[2].replace('"','')
                        subregions['names'].append(name) 
                        subregions['dim'].append(int(words[0]))
                        subregions['tags'].append(int(words[1]))
                        line = infile.readline()
                # TODO: Is this necessary? 
                        
                # =============================================================
                # Cell Vertices
                # =============================================================
                if line == '$Nodes\n':              
                    #
                    # Record number of nodes
                    #
                    line = infile.readline()
                    points['n'] = int(line.rstrip())
                    line = infile.readline()

                    while True:
                        if line == '$EndNodes\n':
                            line = infile.readline()
                            break
                        #
                        # Record vertex coordinates
                        # 
                        words = line.split()
                        vtx = (float(words[1]),float(words[2]))
                        points['coordinates'].append(vtx)
                        line = infile.readline()
                
                # =============================================================
                #  Faces
                # =============================================================        
                if line == '$Elements\n':
                    next(infile)  # skip 'number of elements' line
                    line = infile.readline()
                    n_faces = 0  # count number of faces
                    while True:
                        """
                        General format for elements
                    
                        $Elements
                        n_elements
                        el_number | el_type* | num_tags** | ...
                        tag1 .. tag_num_tags |...
                        node_number_list
                    
                        *el_type: element type
                    
                            points: 15 (1 node point)
                    
                            lines: 1  (2 node line),            0 --------- 1
                                   8  (3 node 2nd order line),  0 --- 2 --- 1
                                   26 (4 node 3rd order line)   0 - 2 - 3 - 1
                    
                    
                            triangles: 2   (3 node 1st order triangle)
                                       9   (6 node 2nd order triangle)
                                       21  (9 node 3rd order triangle)
                                      
                            
                            quadrilateral: 3 (4 node first order quadrilateral)
                                          10 (9 node second order quadrilateral)
                                         
                    
                              
                        **num_tags: 
                           
                           1st tag - physical entity to which element belongs 
                                     (often 0)
                        
                           2nd tag - number of elementary geometrical entity to
                                     which element belongs (as defined in the 
                                     .geo file).
                        
                           3rd tag - number of the mesh partition to which the 
                                     element belongs.
                    
                        """
                        if line == '$EndElements\n':
                            faces['n'] = n_faces
                            line = infile.readline()
                            break
                        words = line.split()
                        #
                        # Identify entity
                        # 
                        element_type = int(words[1])
                        if element_type==15:
                            #
                            # Point (1 node)
                            #
                            dofs_per_entity = 1
                            entity = points
                        
                        if element_type==1:
                            #
                            # Linear edge (2 nodes)
                            #
                            dofs_per_entity = 2       
                            entity = edges
                        
                        elif element_type==8:
                            #
                            # Quadratic edge (3 nodes)
                            #
                            dofs_per_entity = 3
                            entity = edges
                                                        
                        elif element_type==26:
                            #
                            # Cubic edge (4 nodes)
                            #
                            dofs_per_entity = 4
                            entity = edges
                            
                        elif element_type==2:
                            #
                            # Linear triangular element (3 nodes)
                            #
                            dofs_per_entity = 3
                            entity = faces
                            entity['type'].append('triangle')
                            n_faces += 1
                            
                        elif element_type==9:
                            #
                            # Quadratic triangular element (6 nodes)
                            #
                            dofs_per_entity = 6
                            entity = faces
                            entity['type'].append('triangle')
                            n_faces += 1
                            
                        elif element_type==21:
                            #
                            # Cubic triangle (10 nodes)
                            #
                            dofs_per_entity = 10
                            entity = faces
                            entity['type'].append('triangle')
                            n_faces += 1
                            
                        elif element_type==3:
                            #
                            # Linear quadrilateral (4 nodes)
                            #
                            dofs_per_entity = 4
                            entity = faces
                            entity['type'].append('quadrilateral')
                            n_faces += 1
                            
                        elif element_type==10:
                            #
                            # Quadratic quadrilateral (9 nodes)
                            #
                            dofs_per_entity = 9
                            entity = faces
                            entity['type'].append('quadrilateral')
                            n_faces += 1
                            
                        entity['n_dofs'] = dofs_per_entity
                        #
                        # Record tags
                        # 
                        num_tags = int(words[2])
                        if num_tags > 0:
                            #
                            # Record Physical Entity tag
                            #
                            entity['tags']['phys'].append(int(words[3]))
                        else:
                            #
                            # Tag not included ... delete
                            # 
                            entity['tags'].pop('phys', None)
                            
                        if num_tags > 1:
                            #
                            # Record Geometrical Entity tag
                            #
                            entity['tags']['geom'].append(int(words[4]))
                        else:
                            #
                            # Tag not included ... delete
                            # 
                            entity['tags'].pop('geom', None)
                            
                        if num_tags > 2:
                            #
                            # Record Mesh Partition tag
                            # 
                            entity['tags']['partition'].append(int(words[5]))
                        else:
                            #
                            # Tag not included ... delete
                            # 
                            entity['tags'].pop('partition', None)
                            
                        if dofs_per_entity > 1:
                            #
                            # Connectivity
                            #
                            i_begin = 3 + num_tags
                            i_end   = 3 + num_tags + dofs_per_entity 
                            connectivity = [int(words[i])-1 for i in \
                                            np.arange(i_begin,i_end) ]
                            entity['connectivity'].append(connectivity)
                        line = infile.readline()        
                                        
                if line == '':
                    break
        #
        # Check for mixed Faces
        #         
        if len(set(faces['type']))>1:
            raise Warning('Face types are mixed')
        
        #
        # Turn Edge connectivities into sets
        # 
        for i in range(len(edges['connectivity'])):
            edges['connectivity'][i] = frozenset(edges['connectivity'][i])
        
        #
        # There are faces, dimension = 2
        #    
        if n_faces > 0:
            self.__dim = 2
            
    
    def determine_half_edges(self):
        """
        Returns a doubly connected edge list.
        
        The grid should already have the following specified:
        
            1D: points
            
            2D: points, faces
        
        Currently, 
        """
        #
        # Update Point Fields
        # 
        n_points = self.points['n']
        self.points['half_edge'] = np.full((n_points,), -1, dtype=np.int)
        
        # =====================================================================
        # Initialize Half-Edges
        # =====================================================================
        if self.dim()==1:
            #
            # 1D mesh
            #
            n_he = self.points['n']-1
        elif self.dim()==2:
            #
            # 2D mesh
            # 
            n_faces = self.faces['n']
            n_he = 0
            for i in range(n_faces):
                n_he += len(self.faces['connectivity'][i])
                
        self.half_edges['n'] = n_he
        self.half_edges['connectivity'] = np.full((n_he,2), -1, dtype=np.int)
        self.half_edges['prev'] = np.full((n_he,), -1, dtype=np.int)
        self.half_edges['next'] = np.full((n_he,), -1, dtype=np.int)
        self.half_edges['twin'] = np.full((n_he,), -1, dtype=np.int)
        self.half_edges['edge'] = np.full((n_he,), -1, dtype=np.int)
        self.half_edges['face'] = np.full((n_he,), -1, dtype=np.int)
        
        # =====================================================================
        # Define Half-Edges 
        # =====================================================================
        if self.dim()==1:
            #
            # 1D: Define HE's and link with others and points
            # 
            n_points = self.points['n']
            for i in range(n_points-1):

                # Connectivity
                self.half_edges['connectivity'][i] = [i,i+1]
                
                # Previous and next half_edge in the DCEL
                # NOTE: Here (unlike 2D), prev and next are used to 
                #     navigate in the grid.
                self.half_edges['prev'][i] = i-1
                self.half_edges['next'][i] = i+1 if i+1<n_points-1 else -1
                
                # Incident half_edge to left endpoint
                self.points['half_edge'][i] = i
                
                '''
                #
                # Twin
                # 
                # Define twin half-edge
                self.half_edges['connectivity'][n_points-1+i] = [i+1,i]
                self.half_edges['twin'][i] = n_points-1+i 
                self.half_edges['twin'][n_points-1+i] = i
                
                # Incident half-edge to right endpoint
                self.points['half_edge'][i+1] = n_points + i 
                
                # Next and previous
                self.half_edges['next'][n_points-1+i] = i-1
                self.half_edges['prev'][n_points-1+i] = \
                    i+1 if i+1<n_points else -1
                '''
        elif self.dim()==2:
            #
            # 2D: Define HE's and link with others, faces, and points
            # 
            n_faces = self.faces['n']
            self.faces['half_edge'] = np.full((n_faces,), -1, dtype=np.int)
            
            #
            # Loop over faces 
            # 
            half_edge_count = 0
            for i_fce in range(n_faces):
                fc = self.faces['connectivity'][i_fce]
                n_sides = len(fc)
                #
                # Face's half-edge numbers
                # 
                fhe = [half_edge_count + j for j in range(n_sides)]
                
                #
                # Update face information 
                # 
                self.faces['half_edge'][i_fce] = fhe[0]
                
                for i in range(n_sides):
                    # 
                    # Update half-edge information
                    # 
                    
                    #
                    # Connectivity
                    #
                    hec = [fc[i%n_sides], fc[(i+1)%n_sides]]
                    self.half_edges['connectivity'][fhe[i],:] = hec
                    
                    '''
                    DEBUG
                    if fhe[i] >= n_he:
                        print('Half-edge index exceeds matrix dimensions.')
                        print('Number of faces: {0}'.format(self.faces['n']))
                        print('Number of half-edges: 3x#faces =' + \
                              ' {0}'.format(3*self.faces['n']))
                        print('#Half-Edges recorded: {0}'+\
                              ''.format(self.half_edges['n']))
                    '''
                    #
                    # Previous Half-Edge
                    #    
                    self.half_edges['prev'][fhe[i]] = fhe[(i-1)%n_sides]
                    #
                    # Next Half-Edge
                    #
                    self.half_edges['next'][fhe[i]] = fhe[(i+1)%n_sides]
                    #
                    # Face
                    #
                    self.half_edges['face'][fhe[i]] = i_fce
                    
                    # 
                    # Points
                    #                   
                    self.points['half_edge'][fc[i%n_sides]] = fhe[i]
                #
                # Update half-edge count
                # 
                half_edge_count += n_sides    
        hec = self.half_edges['connectivity']
        # =====================================================================
        # Determine twin half_edges
        # =====================================================================
        for i in range(n_he):
            #
            # Find the row whose reversed entries match current entry
            # 
            row = np.argwhere((hec[:,0]==hec[i,1]) & (hec[:,1]==hec[i,0]))
            if len(row) == 1:
                #
                # Update twin field
                #
                self.half_edges['twin'][i] = int(row)
             
        """
        # =====================================================================
        # Link with Edges
        # =====================================================================
        #
        # Update Edge Fields 
        #
        # TODO: Delete when safe to do so!!
        edge_set = set(self.edges['connectivity']) 
        self.edges['half_edge'] = [None]*len(edge_set) 
        for i_he in range(n_he):
            #
            # Loop over half-edges
            # 
            hec = self.half_edges['connectivity'][i_he]
            '''
            DEBUG
            #print('Size of edge_set: {0}'.format(len(edge_set)))
            #print('Size of edge connectivity: {0}'.format(len(self.edges['connectivity'])))
            '''
            if set(hec) in edge_set:
                '''
                DEBUG
                print('Set {0} is in edge_set. Locating it'.format(hec))
                '''
                #
                # Edge associated with Half-Edge exists
                # 
                i_edge = self.edges['connectivity'].index(set(hec))
                '''
                DEBUG
                print('Location: {0}'.format(i_edge))
                print('Here it is: {0}'.format(self.edges['connectivity'][i_edge]))
                #print('Linking half edge with edge:')  
                #print('Half-edge: {0}'.format(self.edges['connectivity'][i_edge]))
                #print('Edge: {0}'.format(self.half_edges['connectivity'][fhe[i]]))
                #print(len(self.edges['half_edge']))
                #print('Length of edge_set {0}'.format(len(edge_set)))
                #print(edge_set)
                '''
                #
                # Link edge to half edge
                #
                self.edges['half_edge'][i_edge] = i_he
            else:
                #print('Set {0} is not in edge_set \n '.format(hec))
                #
                # Add edge
                #
                new_edge = frozenset(hec)
                self.edges['connectivity'].append(new_edge)
                edge_set.add(new_edge)
                i_edge =len(self.edges['connectivity'])-1
                #
                # Assign empty tags
                # 
                for tag in self.edges['tags'].values():
                    tag.append(None)
                #                
                # Link edge to half-edge
                #
                self.edges['half_edge'].append(i)
                #
                # Link half-edge to edge
                #     
                self.half_edges['edge'][i] = i_edge    
                #
                # Update size of edge list       
                #
                self.edges['n'] = len(self.edges['connectivity'])
        """
            
            
    def dim(self):
        """
        Returns the underlying dimension of the grid
        """ 
        return self.__dim
    
    
    def get_neighbor(self, i_entity, i_direction):
        """
        Returns the neighbor of an entity in a given direction 
        
        Inputs: 
        
            i_entity: int, index of the entity whose neighbor we seek 
                
                In 1D: i_entity indexes a half_edge
                In 2D: i_entity indexes a face
        
            
            i_direction: int, index of an entity specifying a direction
            
                In 1D: i_direction indexes an interval endpoint
                In 2D: i_direction indexes a half_edge
                
        """
        if self.dim() == 1:
            #
            # 1D grid
            #
            hec = self.half_edges['connectivity'][i_entity]
            assert i_direction in hec, \
                'Point index not in connectivity of this Half-Edge.'
            if i_direction == hec[0]:
                #
                # Left endpoint: go to previous half-edge 
                #
                i_nbr = self.half_edges['prev'][i_entity] 

            elif i_direction == hec[1]:
                #
                # Right endpoint: go to next Half-Edge
                # 
                i_nbr = self.half_edges['next'][i_entity]
        elif self.dim() == 2:
            #
            # 2D grid: use half_edges 
            # 
            assert self.half_edges['face'][i_direction] == i_entity,\
                'Cell not incident to Half-Edge.'
            
            i_nbr_he = self.half_edges['twin'][i_direction]
            i_nbr = self.half_edges['face'][i_nbr_he]
        
        if i_nbr != -1:
            return i_nbr
        else:
            return None
            
            
    def get_boundary_half_edges(self):
        """
        Returns a list of the boundary half_edge indices
        """
        assert self.dim()==2, 'Half edges only present in 2D grids.'
    
        bnd_hes_conn = []
        bnd_hes = []
        #
        # Locate half-edges on the boundary
        # 
        for i_he in range(self.half_edges['n']):
            if self.half_edges['twin'][i_he] == -1:
                bnd_hes.append(i_he)
                bnd_hes_conn.append(self.half_edges['connectivity'][i_he])
        #
        # Group and sort half-edges
        #
        bnd_hes_sorted = [deque([he]) for he in bnd_hes]
        
        while True:
            for g1 in bnd_hes_sorted:
                #
                # Check if g1 can add a deque in bnd_hes_sorted
                # 
                merger_activity = False
                for g2 in bnd_hes_sorted:
                    #
                    # Does g1's head align with g2's tail?
                    #
                    if self.half_edges['connectivity'][g1[-1]][1]==\
                       self.half_edges['connectivity'][g2[0]][0]:
                        # Remove g2 from list
                        if len(bnd_hes_sorted) > 1:
                            g2 = bnd_hes_sorted.pop(bnd_hes_sorted.index(g2))
                            g1.extend(g2)
                            merger_activity = True 
                    #
                    # Does g1's tail align with g2's head?
                    #
                    elif self.half_edges['connectivity'][g1[0]][0]==\
                         self.half_edges['connectivity'][g2[-1]][1]:
                        if len(bnd_hes_sorted) > 1:
                            g2 = bnd_hes_sorted.pop(bnd_hes_sorted.index(g2))
                            g1.extendleft(g2)
                            merger_activity = True
            if not merger_activity:
                break    
        #
        # Multiple boundary segments
        #     
        return [list(segment) for segment in bnd_hes_sorted]
        
        """
        bnd_hes_sorted = []
        i_he_left = bnd_hes.pop()
        i_he_right = i_he_left
        he_conn_left = bnd_hes_conn.pop()
        he_conn_right = he_conn_left
        subbnd_hes_sorted = deque([i_he])
        while len(bnd_hes)>0:
            added_to_left = False
            added_to_right = False
            for i in range(len(bnd_hes)):
                if bnd_hes_conn[i][0] == he_conn_right[1]:
                    #
                    # Base vertex of he in list matches 
                    # head vertex of popped he.
                    #
                    i_he_right = bnd_hes.pop(i)
                    he_conn_right = bnd_hes_conn.pop(i)
                    subbnd_hes_sorted.append(i_he_right)
                    added_to_right = True 
                elif bnd_hes_conn[i][1] == he_conn_left[0]:
                    #
                    # Head vertex of he in list matches
                    # base vertex of popped he.
                    #
                    i_he_left = bnd_hes_conn.pop(i)
                    he_conn_left = bnd_hes_conn.pop(i)
                    subbnd_hes_sorted.appendleft(i_he_left)
                    added_to_left = True
                if added_to_left and added_to_right:
                    break
            if not added_to_left and not added_to_right:
                # Could not find any half-edges to add
                #
                # Add boundary segment to sorted hes
                # 
                bnd_hes_sorted.extend(ihe for ihe in subbnd_hes_sorted)
                #
                # Reinitialize subbnd_hes_sorted
                # 
                i_he_left = bnd_hes.pop()
                i_he_right = i_he_left
                he_conn_left = bnd_hes_conn.pop()
                he_conn_right = he_conn_left
                subbnd_hes_sorted = deque([i_he])
        return bnd_hes_sorted
        """
        
    '''
    def get_boundary_edges(self):
        """
        Returns a list of the boundary edge indices
        
        TODO: Get rid of this 
        """
        bnd_hes_sorted = self.get_boundary_half_edges()
        #
        # Extract boundary edges
        # 
        bnd_edges = [self.half_edges['edge'][i] for i in bnd_hes_sorted]
        return bnd_edges
    '''
    
    def get_boundary_points(self):
        """
        Returns a list of boundary point indices
        """
        if self.dim() == 1:
            #
            # One dimensional grid (assume sorted)
            # 
            bnd_points = [0, self.points['n']-1]
        elif self.dim() == 2:
            #
            # Two dimensional grid
            # 
            bnd_points = []
            for i_he in self.get_boundary_half_edges():
                #
                # Add initial point of each boundary half edge
                # 
                bnd_points.append(self.half_edges['connectivity'][i_he][0])
        else: 
            raise Exception('Only dimensions 1 and 2 supported.')
        return bnd_points
    
    
    def make_periodic(self, coordinates, box):
        """
        Make a rectangular DCEL periodic by assigning the correct twins to
        HalfEdges on the boundary. 
        
        Inputs:
        
            Coordinates: set, containing 0 (x-direction) and/or 1 (y-direction). 
        
        TODO: Cannot make periodic (1,1) DCEL objects
        """
        
        if self.dim()==1:
            #
            # In 1D, first half-edge becomes "next" of last half-edge
            # 
            self.half_edges['next'][-1] = 0
            self.half_edges['prev'][0] = self.half_edges['n']-1
        elif self.dim()==2:
            #
            # In 2D, must align vertices on both side of the box
            # 
            x_min, x_max, y_min, y_max = box
            if 0 in coordinates:
                #
                # Make periodic in the x-direction
                #
                left_hes = []
                right_hes = []
                for segment in self.get_boundary_half_edges():
                    for he in segment:               
                        #
                        # Record coordinates of half-edge's base and head
                        # 
                        i_base, i_head = self.half_edges['connectivity'][he][:]
                        x_base, y_base = self.points['coordinates'][i_head]
                        x_head, y_head = self.points['coordinates'][i_base]
                        
                        if np.isclose(x_base,x_max) and np.isclose(x_head,x_max):
                            #
                            # If x-values are near x_max, it's on the right
                            # 
                            right_hes.append((he, y_base, y_head))
        
                        elif np.isclose(x_base,x_min) and np.isclose(x_head,x_min):
                            #
                            # If x-values are near x_min, it's on the left
                            # 
                            left_hes.append((he, y_base, y_head))                
                #
                # Look for twin half-edges
                # 
                n_right = len(left_hes)
                n_left = len(right_hes)
                assert n_right==n_left, \
                    'Number of half-edges on either side of domain differ.'+\
                    'Cannot make periodic.'
                while len(left_hes)>0:
                    l_he, l_ybase, l_yhead = left_hes.pop()
                    for ir in range(len(right_hes)):
                        #
                        # For each halfedge on the left, check if there is a 
                        # corresponding one on the right.
                        #         
                        r_he, r_ybase, r_yhead = right_hes[ir]
                        if np.isclose(l_ybase, r_yhead) and np.isclose(l_yhead, r_ybase):
                            self.half_edges['twin'][l_he] = r_he
                            self.half_edges['twin'][r_he] = l_he
                            del right_hes[ir]
                            break
                        
                assert len(right_hes)==0, \
                    'All HalfEdges on the left should be matched with '+\
                    'one on the right.'            
                
            if 1 in coordinates:
                #
                # Make periodic in the y-direction
                #coordinates
                top_hes = []
                bottom_hes = []
                for segment in self.get_boundary_half_edges():
                    for he in segment:
                        #
                        # Record coordinates of half-edge's base and head
                        # 
                        i_base, i_head = self.half_edges['connectivity'][he]
                        x_base, y_base = self.points['coordinates'][i_head]
                        x_head, y_head = self.points['coordinates'][i_base]
                        
                        if np.isclose(y_base,y_max) and np.isclose(y_head,y_max):
                            #
                            # If y-values are near y_max, it's on the top
                            # 
                            top_hes.append((he, x_base, x_head))
        
                        elif np.isclose(y_base,y_min) and np.isclose(y_head,y_min):
                            #
                            # If y-values are near y_min, it's on the bottom
                            # 
                            bottom_hes.append((he, x_base, x_head))
                #
                # Look for twin half-edges
                # 
                while len(bottom_hes)>0:
                    b_he, b_xbase, b_xhead = bottom_hes.pop()
                    for it in range(len(top_hes)):
                        #
                        # For each halfedge on the left, check if there is a 
                        # corresponding one on the right.
                        #         
                        t_he, t_xbase, t_xhead = top_hes[it]
                        if np.isclose(t_xbase, b_xhead) and np.isclose(t_xhead, b_xbase):
                            self.half_edges['twin'][b_he] = t_he
                            self.half_edges['twin'][t_he] = b_he
                            del top_hes[it]
                            break
                        
                assert len(top_hes)==0, \
                    'All HalfEdges on the left should be matched with '+\
                    'one on the right.'                 
        self.periodic_coordinates = coordinates
        

class Mesh(object):
    """
    Mesh class
    """
    def __init__(self, dcel=None, box=None, resolution=None, periodic=None, 
                 dim=None, x=None, connectivity=None, file_path=None, 
                 file_format='gmsh'):
        
        # =====================================================================
        # Doubly connected Edge List
        # =====================================================================
        if dcel is None:
            #
            # Initialize doubly connected edge list if None
            # 
            dcel = DCEL(box=box, resolution=resolution, periodic=periodic, 
                        dim=dim, x=x, connectivity=connectivity, 
                        file_path=file_path, file_format=file_format)
        else:
            assert isinstance(dcel,DCEL)
            
        self.dcel = dcel
        #
        # Determine mesh dimension
        # 
        dim = dcel.dim()
        self._dim = dim
        
        # =====================================================================
        # Vertices
        # =====================================================================
        vertices = []
        n_points = dcel.points['n']
        for i in range(n_points):
            vertices.append(Vertex(dcel.points['coordinates'][i]))
        self.vertices = vertices


    def dim(self):
        """
        Returns the dimension of the mesh (1 or 2)
        """
        return self._dim


class Mesh1D(Mesh):
    """
    1D Mesh Class
    """
    def __init__(self, dcel=None, box=None, resolution=None, periodic=False, 
                 x=None, connectivity=None, file_path=None, file_format='gmsh'):
        
        #
        # Convert input "periodic" to something intelligible for DCEL
        # 
        if periodic is True:
            periodic = {0}
        else:
            periodic = None
            
        Mesh.__init__(self, dcel=dcel, box=box, resolution=resolution, 
                      periodic=periodic, dim=1, x=x, connectivity=connectivity,
                      file_path=file_path, file_format=file_format)
        
        assert self.dim()==1, 'Mesh dimension not 1.'
        
        # =====================================================================
        # Intervals
        # =====================================================================
        intervals = []
        n_intervals = self.dcel.half_edges['n']
        for i in range(n_intervals):
            #
            # Make list of intervals
            # 
            i_vertices  = self.dcel.half_edges['connectivity'][i]
            v_base = self.vertices[i_vertices[0]]
            v_head = self.vertices[i_vertices[1]]
            interval = Interval(v_base, v_head) 
            intervals.append(interval)
        #
        # Align intervals (assign next)
        # 
        for i in range(n_intervals):
            i_nxt = self.dcel.half_edges['next'][i]
            if i_nxt!=-1:
                if intervals[i].head() != intervals[i_nxt].base():
                    assert self.dcel.is_periodic, 'DCEL should be periodic'
                    #
                    # Intervals linked by periodicity
                    # 
                    itv_1, vtx_1 = intervals[i], intervals[i].head()
                    itv_2, vtx_2 = intervals[i_nxt], intervals[i_nxt].base()
                    
                    # Mark intervals periodic
                    itv_1.set_periodic()
                    itv_2.set_periodic()
                    
                    # Mark vertices periodic
                    vtx_1.set_periodic()
                    vtx_2.set_periodic()
                    
                    # Associate vertices with one another
                    vtx_1.set_periodic_pair((itv_2, vtx_2))
                    vtx_2.set_periodic_pair((itv_1, vtx_1))
                else:        
                    intervals[i].assign_next(intervals[i_nxt])
                
        #
        # Store intervals in Forest
        # 
        self.cells = Forest(intervals)
        self.__periodic_coordinates = self.dcel.periodic_coordinates
        
    
    def is_periodic(self):
        """
        Returns true if the mesh is periodic
        """
        return 0 in self.__periodic_coordinates
    
    
    def locate_point(self, point, flag=None):
        """
        Returns the smallest (flagged) cell containing a given point 
        or None if current cell doesn't contain the point
        
        Input:
            
            point: Vertex
            
        Output:
            
            cell: smallest cell that contains x
                
        """
        for interval in self.cells.get_children(flag=flag):
            if interval.contains_points(point):
                if flag is not None:
                    if not interval.is_marked(flag):
                        return None
                    else: return interval.locate_point(point, flag=flag)
                else:
                    return interval.locate_point(point, flag=flag)
    
    
    def get_boundary_vertices(self):
        """
        Returns the mesh endpoint vertices
        """
        v0 = self.cells.get_child(0).base()
        v1 = self.cells.get_child(-1).head()
        return v0, v1
    
    
    def bounding_box(self):
        """
        Returns the interval endpoints
        """
        v0, v1 = self.get_boundary_vertices()
        x0, = v0.coordinates()
        x1, = v1.coordinates()
        return x0, x1
    

    def mark_boundary(self, flag, bnd_fun):
        """
        Marks left or right mesh endpoint with flag
        
        Inputs: 
        
            flag: str/int, vertex marker
            
            bnd_fun: bool function, used to locate boundary 
        """
        v0, v1 = self.get_boundary_vertices()
        x0, = v0.coordinates()
        x1, = v1.coordinates()
        
        if bnd_fun(x0):
            v0.mark(flag)
        
        if bnd_fun(x1):
            v1.mark(flag)
        
    
    '''
    def record(self, mesh_label, flag=None):
        """
        Mark (flagged) intervals in the current mesh (and their parents) with
        mesh_label.  
        """
        # 
        # Traverse the forest of flagged intervals
        # 
        for interval in self.cells.traverse(flag=flag, mode='breadth-first'):
            interval.mark(mesh_label)
            
    
    def refine(self, flag, mesh_label=None):
        """
        Split flagged LEAF cells of mesh (and label the refined mesh).
        """
        for interval in self.cells.get_leaves(flag):
            interval.split()
            for child in interval.get_children():
                child.mark(flag)
        if mesh_label is not None:
            self.record(mesh_label, flag=flag)
    
    
    def coarsen(self, flag, mesh_label=None):
        """
        Delete (or unmark) the children 
        """
        for interval in self.cells.get_leaves(flag=flag):
          pass  
    '''
    
    
class Mesh2D(Mesh):
    """
    2D Mesh class
    """
    def __init__(self, dcel=None, box=None, resolution=None, x=None, 
                 periodic=None, connectivity=None, file_path=None, 
                 file_format='gmsh'):
        
        Mesh.__init__(self, dcel=dcel, box=box, resolution=resolution, 
                      periodic=periodic, dim=2, x=x, connectivity=connectivity,
                      file_path=file_path, file_format=file_format)
        
        self._is_rectangular = self.dcel.is_rectangular
        self._periodic_coordinates = self.dcel.periodic_coordinates
        # ====================================================================
        # HalfEdges
        # ====================================================================
        half_edges = []
        n_hes = self.dcel.half_edges['n']            
        for i in range(n_hes):
            i_vertices  = self.dcel.half_edges['connectivity'][i]
            v_base = self.vertices[i_vertices[0]]
            v_head = self.vertices[i_vertices[1]]
            half_edge = HalfEdge(v_base, v_head)
            half_edges.append(half_edge)
        #
        # Assign twins (2D)
        #                 
        for i_he in range(n_hes):
            i_twin = self.dcel.half_edges['twin'][i_he]
            if i_twin!=-1:
                #
                # HalfEdge has twin
                #
                he_nodes = self.dcel.half_edges['connectivity'][i_he]
                twin_nodes = self.dcel.half_edges['connectivity'][i_twin]
                if not all(he_nodes == list(reversed(twin_nodes))):
                    #
                    # Heads and Bases don't align, periodic boundary
                    # 
                    assert self.is_periodic(), 'Mesh is not periodic.'\
                    'All HalfEdges should align.'
                    half_edges[i_he].set_periodic()
                    half_edges[i_twin].set_periodic()
                
                half_edges[i_he].assign_twin(half_edges[i_twin])
                half_edges[i_twin].assign_twin(half_edges[i_he])
                
        #
        # Store HalfEdges in Forest.
        #
        self.half_edges = Forest(half_edges)
                    
        # =====================================================================
        # Cells
        # =====================================================================
        cells = []
        n_cells = self.dcel.faces['n']
        is_quadmesh = True
        for ic in range(n_cells):
            i_he_pivot = self.dcel.faces['half_edge'][ic]
            i_he = i_he_pivot
            one_rotation = False 
            i_hes = []
            while not one_rotation:
                i_hes.append(i_he)
                i_he = self.dcel.half_edges['next'][i_he]
                if i_he==i_he_pivot:
                    one_rotation = True    
            if len(i_hes)==4:
                cells.append(QuadCell([half_edges[i] for i in i_hes]))
            else:
                cells.append(Cell([half_edges[i] for i in i_hes]))
                is_quadmesh = False                
        self._is_quadmesh = is_quadmesh
        self.cells = Forest(cells)    
        
        # =====================================================================
        # Pair Periodic Vertices
        # =====================================================================
        for half_edge in self.half_edges.get_children():
            # Pair periodic vertices
            #
            if half_edge.is_periodic():
                half_edge.pair_periodic_vertices()
                
                
    def is_rectangular(self):
        """
        Check whether the Mesh is rectangular
        """
        return self._is_rectangular
    
    
    def is_periodic(self, coordinates=None):
        """
        Check whether the Mesh is periodic in the x- and/or the y direction
        
        Input:
        
            *coordinates: int, set containing 0 (x-direction) and/or 1 (y-direction)
                if directions is None, check for periodicity in any direction 
            
        """
        if coordinates is None:
            return 0 in self._periodic_coordinates or 1 in self._periodic_coordinates
        else:
            is_periodic = True
            for i in coordinates:
                if i not in self._periodic_coordinates:
                    return False
            return is_periodic
        
    
    def is_quadmesh(self):
        """
        Check if the mesh is a quadmesh
        """
        return self._is_quadmesh
          
        
    def locate_point(self, point, flag=None):
        """
        Returns the smallest (flagged) cell containing a given point 
        or None if current cell doesn't contain the point
        
        Input:
            
            point: Vertex
            
        Output:
            
            cell: smallest cell that contains x
                
        """
        for cell in self.cells.get_children():
            if flag is None:
                if cell.contains_points(point):
                    return cell
            else:
                if cell.is_marked(flag) and cell.contains_points(point):
                    return cell
            

    def get_boundary_segments(self, subforest_flag=None, flag=None):
        """
        Returns a list of segments of boundary half edges
        
        Inputs: 
        
            subforest_flag: optional flag (int/str) specifying the submesh
                within which boundary segments are
        """
        bnd_hes = []
        #
        # Locate half-edges on the boundary
        # 
        for he in self.half_edges.get_leaves(subforest_flag=subforest_flag,\
                                             flag=flag):
            if he.twin() is None:
                bnd_hes.append(he)
        #
        # Group and sort half-edges
        #
        bnd_hes_sorted = [deque([he]) for he in bnd_hes]
        while True:
            merger_activity = False
            for g1 in bnd_hes_sorted:
                #
                # Check if g1 can add a deque in bnd_hes_sorted
                # 
                merger_activity = False
                for g2 in bnd_hes_sorted:
                    #
                    # Does g1's head align with g2's tail?
                    #
                    if g1[-1].head()==g2[0].base():
                        # Remove g2 from list
                        g2 = bnd_hes_sorted.pop(bnd_hes_sorted.index(g2))
                        g1.extend(g2)
                        merger_activity = True 
                    #
                    # Does g1's tail align with g2's head?
                    #
                    elif g1[0].base()==g2[-1].head():
                        g2 = bnd_hes_sorted.pop(bnd_hes_sorted.index(g2))
                        g2.reverse()
                        g1.extendleft(g2)
                        merger_activity = True
            if not merger_activity or len(bnd_hes_sorted)==1:
                break    
        #
        # Multiple boundary segments
        #     
        return [list(segment) for segment in bnd_hes_sorted]
    
    
    def mark_boundary_edges(self, flag, bnd_fun, subforest_flag=None):
        """
        Flag specific boundary edges identified by bnd_fun 
        
        Inputs:
        
            flag: str/int, marker for half_edge
            
            bnd_fun: boolean function whose input is a half-edge
        """
        for segment in self.get_boundary_segments(subforest_flag):
            #
            # Iterate over boundary segments
            #
            for he in segment:
                #
                # Iterate over half-edges within each segment
                # 
                if bnd_fun(he):
                    he.mark(flag)
    
    
    def bounding_box(self):
        """
        Returns the bounding box of the mesh
        """
        xy = convert_to_array(self.vertices, dim=2)
        x0, x1 = xy[:,0].min(), xy[:,0].max()
        y0, y1 = xy[:,1].min(), xy[:,1].max()
        return x0, x1, y0, y1
    '''
    def get_boundary_edges(self, flag=None):
        """
        Returns the half-nodes on the boundary
        """
        bnd_hes_unsorted = []
        #
        # Locate ROOT half-edges on the boundary
        # 
        for he in self.half_edges.get_children():
            if he.twin() is None:
                bnd_hes_unsorted.append(he)
        n_bnd = len(bnd_hes_unsorted)
        #
        # Sort half-edges
        #
        he = bnd_hes_unsorted.pop()
        bnd_hes_sorted = [he]
        while n_bnd>0:
            for i in range(n_bnd):
                nxt_he = bnd_hes_unsorted[i]
                if he.head()==nxt_he.base():
                    bnd_hes_sorted.append(nxt_he)
                    he = bnd_hes_unsorted.pop(i)
                    n_bnd -= 1
                    break
        #
        # Get LEAF half-edges
        # 
        bnd_hes = []
        for he in bnd_hes_sorted:
            bnd_hes.extend(he.get_leaves(flag=flag))
    '''                        

    def get_boundary_vertices(self, flag=None, subforest_flag=None):
        """
        Returns the Vertices on the boundary
        """
        vertices = []
        for segment in self.get_boundary_segments(subforest_flag=subforest_flag, 
                                                  flag=flag):
            for he in segment:
                vertices.append(he.base())
        return vertices
    
    
    def mark_boundary_vertices(self, flag, bnd_fun, subforest_flag=None):
        """
        Mark boundary vertices specified by bnd_fun with flag
        
        Inputs:
        
            flag: str/int, vertex marker
            
            bnd_fun: boolean bivariate function, used to specify boundary
                vertices
            
            *subforest_flag: optional flag used to specify submesh
        """
        for v in self.get_boundary_vertices(subforest_flag=subforest_flag):
            #
            # Iterate over all boundary vertices 
            #
            x0, x1 = v.coordinates()
            if bnd_fun(x0,x1):
                #
                # Mark identified vertices 
                # 
                v.mark(flag)
        
        
class QuadMesh(Mesh2D):
    """
    Two dimensional mesh with quadrilateral cells.
    
    
    Note:

        When coarsening and refining a QuadMesh, the HalfEdges are not deleted
        Rather use submeshes.
    """
    def __init__(self, dcel=None, box=None, resolution=None, x=None, 
                 periodic=None, connectivity=None, file_path=None, 
                 file_format='gmsh'):
        #
        # Initialize 2D Mesh.
        # 
        Mesh2D.__init__(self, dcel=dcel, box=box, resolution=resolution, 
                        periodic=periodic, x=x, connectivity=connectivity,
                        file_path=file_path, file_format=file_format)
        self.cells = Forest(self.cells.get_children())
         
        
    def locate_point(self, point, flag=None):
        """
        Returns the smallest (flagged) cell containing a given point 
        or None if current cell doesn't contain the point
        
        Input:
            
            point: Vertex
            
        Output:
            
            cell: smallest cell that contains x
                
        """
        for cell in self.cells.get_children(flag=flag):
            if cell.contains_points(point):
                if cell.has_children(flag=flag):
                    return cell.locate_point(point, flag=flag)
                else:
                    return cell

        
    def is_balanced(self, subforest_flag=None):
        """
        Check whether the mesh is balanced
        
        Inputs: 
            
            flag (optional): marker, allowing for the restriction to
                a submesh. 
        
        """
        for cell in self.cells.get_leaves(subforest_flag=subforest_flag):
            for half_edge in cell.get_half_edges():
                nb = cell.get_neighbors(half_edge, flag=subforest_flag)
                if nb is not None and nb.has_children(flag=subforest_flag):
                    twin = half_edge.twin()
                    for the_child in twin.get_children():
                        if the_child.cell().has_children(flag=subforest_flag):
                            return False 
        return True

        
    def balance(self, subforest_flag=None):
        """
        Ensure that subcells of current cell conform to the 2:1 rule
        """
        assert self.cells.is_forest_of_rooted_subtree(subforest_flag)
        #
        # Get all LEAF cells
        # 
        leaves = set(self.cells.get_leaves(subforest_flag=subforest_flag))  # set: no duplicates
        while len(leaves)>0:
            leaf = leaves.pop()
            #
            # For each Cell
            # 
            is_split = False
            for half_edge in leaf.get_half_edges():
                #
                # Look for neighbors in each direction
                # 
                nb = leaf.get_neighbors(half_edge, flag=subforest_flag)
                if nb is not None and nb.has_children(flag=subforest_flag):
                    #
                    # Check if neighbor has children (still fine)
                    # 
                    twin = half_edge.twin()
                    for the_child in twin.get_children():
                        if the_child.cell().has_children(flag=subforest_flag):
                            #
                            # Neighbor has grandchildren
                            # 
                            if not leaf.has_children(flag=subforest_flag):
                                #
                                # LEAF does not have any flagged children
                                #
                                if leaf.has_children():
                                    #
                                    # LEAF has children (just not flagged)
                                    #  
                                    for child in leaf.get_children():
                                        child.mark(subforest_flag)
                                else:
                                    #
                                    # LEAF needs new children.
                                    # 
                                    leaf.split(flag=subforest_flag)
                                #
                                # Add children to the leaf nodes to be considered
                                # 
                                for child in leaf.get_children():
                                    leaves.add(child)
                                
                                #
                                # If LEAF is split, add all its neighbors to leaves
                                # to be considered for splitting.
                                #     
                                for half_edge in leaf.get_half_edges():
                                    hep = half_edge.get_parent()
                                    if hep is not None:
                                        hep_twin = hep.twin()
                                        if hep_twin is not None:
                                            leaves.add(hep_twin.cell())                             
                            #
                            # Current LEAF cell has been split, move on to next one
                            #    
                            is_split = True
                            break                        
                if is_split:
                    #
                    # LEAF already split, no need to check other directions
                    # 
                    break
            
            
            
    
    def remove_supports(self, subforest_flag=None, coarsening_flag=None):
        """
        Given a submesh (subforest_flag) and a coarsening_flag,   
        
        Input: 
        
            subforest_flag: flag specifying the submesh to be considered
            
            coarsening_flag: flag specifying the cells to be removed 
            during coarsening
            
        TODO: Unfinished. Loop over cells to be coarsened. Check if it's 
        safe to coarsen neighbors.
        """    
        #
        # Get all flagged LEAF nodes
        # 
        leaves = self.get_leaves(subforest_flag=subforest_flag, 
                                 coarsening_flag=coarsening_flag)
        while len(leaves) > 0:
            #
            # For each LEAF
            # 
            leaf = leaves.pop()
            #
            # Check if leaf is a support leaf
            # 
            if subforest_flag is None:
                is_support = leaf.is_marked('support')
            else:
                is_support = leaf.is_marked((subforest_flag, 'support')) 
            if is_support:
                #
                # Check whether its safe to delete the support cell
                # 
                safe_to_coarsen = True
                for half_edge in leaf.get_half_edges():
                    nb = leaf.get_neighbor(half_edge, flag=subforest_flag)
                    if nb is not None and nb.has_children(flag=subforest_flag):
                        #
                        # Neighbor has (flagged) children, coarsening will lead
                        # to an unbalanced tree
                        # 
                        safe_to_coarsen = False
                        break
                
                if safe_to_coarsen:
                    #
                    # Remove support by marking self with coarsening flag
                    #
                    self.mark(coarsening_flag)
                    leaves.append(leaf.get_parent())
                    
    
'''
class TriCell(object):
    """
    TriCell object
    
    Attributes:
        
    
    Methods:
    
    """
    def __init__(self, vertices, parent=None):
        """
        Inputs:
        
            vertices: Vertex, list of three vertices (ordered counter-clockwise)
            
            parent: QuadCell that contains triangle
            
        """
        v = []
        e = []
        assert len(vertices) == 3, 'Must have exactly 3 vertices.'
        for i in range(3):
            #
            # Define vertices and Half-Edges with minimun information
            # 
            v.append(Vertex(vertices[i],2))        
        #
        # Some edge on outerboundary
        # 
        self.outer_component = e[0]
        
        for i in range(3):
            #
            # Half edge originating from v[i]
            # 
            v[i].incident_edge = e[i]
            #
            # Edges preceding/following e[i]
            # 
            j = np.remainder(i+1,3)
            e[i].next = e[j]
            e[j].previous = e[i]
            #
            #  Incident face
            # 
            e[i].incident_face = self
            
        self.parent_node = parent
        self.__vertices = v
        self.__edges = [
                        Edge(vertices[0], vertices[1], parent=self), \
                        Edge(vertices[1], vertices[2], parent=self), \
                        Edge(vertices[2], vertices[0], parent=self)
                        ]
        self.__element_no = None
        self._flags = set()
        
        
    def vertices(self,n):
        return self.__vertices[n]
    
    def edges(self):
        return self.__edges
    
        
    def area(self):
        """
        Compute the area of the triangle
        """
        v = self.__vertices
        a = [v[1].coordinates()[i] - v[0].coordinates()[i] for i in range(2)]
        b = [v[2].coordinates()[i] - v[0].coordinates()[i] for i in range(2)]
        return 0.5*abs(a[0]*b[1]-a[1]*b[0])
    
     
    def unit_normal(self, edge):
        #p = ((y1-y0)/nnorm,(x0-x1)/nnorm)
        pass    
    
    
    def number(self, num, overwrite=False):
        """
        Assign a number to the triangle
        """
        if self.__element_no == None or overwrite:
            self.__element_no = num
        else:
            raise Warning('Element already numbered. Overwrite disabled.')
            return
        
    def get_neighbor(self, edge, tree):
        """
        Find neighboring triangle across edge wrt a given tree   
        """
        pass
    

    def mark(self, flag=None):
        """
        Mark TriCell
        
        Inputs:
        
            flag: optional label used to mark cell
        """  
        if flag is None:
            self._flags.add(True)
        else:
            self._flags.add(flag)
            
        
    def unmark(self, flag=None, recursive=False):
        """
        Remove label from TriCell
        
        Inputs: 
        
            flag: label to be removed
        
            recursive: bool, also unmark all subcells
        """
        #
        # Remove label from own list
        #
        if flag is None:
            # No flag specified -> delete all
            self._flags.clear()
        else:
            # Remove specified flag (if present)
            if flag in self._flags: self._flags.remove(flag)
        
        #
        # Remove label from children if applicable   
        # 
        if recursive and self.has_children():
            for child in self.children.values():
                child.unmark(flag=flag, recursive=recursive)
                
 
         
    def is_marked(self,flag=None):
        """
        Check whether cell is marked
        
        Input: flag, label for QuadCell: usually one of the following:
            True (catchall), 'split' (split cell), 'count' (counting)
            
        TODO: Possible to add/remove set? Useful? 
        """ 
        if flag is None:
            # No flag -> check whether set is empty
            if self._flags:
                return True
            else:
                return False
        else:
            # Check wether given label is contained in cell's set
            return flag in self._flags
'''                    
