#import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import numbers
from math import isclose
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
            assert x.shape[1]==2,\
            'Dimension of array should be 2'
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
            
            # Ensure that the number of ROOT children is specified
            assert n_children is not None, \
                'ROOT node: Specify number of children.'
            
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
                assert n_children is not None, \
                    'Not a regular tree: Must specify number of children.'
            
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
        Remove node from parent's list of children
        """
        assert self.get_node_type() != 'ROOT', 'Cannot delete ROOT node.'
        self.get_parent()._children[self._node_position] = None    
    

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

        
    def split(self, n_grandchildren=None):
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
            # Not a regular tree: Must specify number of grandchildren
            # 
            assert type(n_grandchildren) is list, \
                'Input "n_grandchildren" must be a list.'
            assert len(n_grandchildren) == self.n_children(), \
                'Input "n_grandchildren" not the right length.'
            
            for i in range(self.n_children()):
                #
                # Instantiate Children
                # 
                self._children[i] = Tree(regular=self.is_regular(),\
                                         n_children = n_grandchildren[i],\
                                         parent=self, position=i)
                      
            
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
                 
                
    def get_leaves(self, flag=None, nested=False):
        """
        Return all LEAF sub-nodes (nodes with no children) of current node
        
        Inputs:
        
            flag: If flag is specified, return all leaf nodes within labeled
                submesh (or an empty list if there are none).
                
            nested: bool, indicates whether leaves should be searched for 
                in a nested way (one level at a time).
                
        Outputs:
        
            leaves: list, of LEAF nodes.
            
            
        Note: 
        
            For nested traversal of a node with flags, there is no simple way
            of determining whether any of the progeny are flagged. We therefore
            restrict ourselves to subtrees. If your children are not flagged, 
            then theirs will also not be. 
        """
        if nested:
            #
            # Nested traversal
            # 
            leaves = []
            for node in self.traverse(flag=flag, mode='breadth-first'):
                if not node.has_children(flag=flag):
                    leaves.append(node)
            return leaves
        else:
            #
            # Non-nested (recursive algorithm)
            # 
            leaves = []
            if flag is None:
                if self.has_children():
                    for child in self.get_children():
                        leaves.extend(child.get_leaves(flag=flag))
                else:
                    leaves.append(self)
            else:
                if self.has_children(flag=flag):
                    for child in self.get_children(flag=flag):
                        leaves.extend(child.get_leaves(flag=flag))
                elif self.is_marked(flag):
                    leaves.append(self)
            return leaves
        
    
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
        
    
    def get_leaves(self, flag=None, nested=False):
        """
        Return all LEAF sub-nodes (nodes with no children) of current node
        
        Inputs:
        
            flag: If flag is specified, return all leaf nodes within labeled
                submesh (or an empty list if there are none).
                
            nested: bool, indicates whether leaves should be searched for 
                in a nested way (one level at a time).
                
        Outputs:
        
            leaves: list, of LEAF nodes.
            
            
        Note: 
        
            For nested traversal of a node with flags, there is no simple way
            of determining whether any of the progeny are flagged. We therefore
            restrict ourselves to subtrees. If your children are not flagged, 
            then theirs will also not be. 
        """
        if nested:
            #
            # Nested traversal
            # 
            leaves = []
            for node in self.traverse(flag=flag, mode='breadth-first'):
                if not node.has_children(flag=flag):
                    leaves.append(node)
            return leaves
        else:
            #
            # Non-nested (recursive algorithm)
            # 
            leaves = []
            for child in self.get_children():
                leaves.extend(child.get_leaves(flag=flag))
            '''    
            if flag is None:
                if self.has_children():
                    for child in self.get_children():
                        #
                        # Loop over trees
                        #
                        if child.has_children():
                            #
                            # Loop over tree children
                            # 
                            for grandchild in child.get_children():
                                leaves.extend(grandchild.get_leaves(flag=flag))
                        else:
                            leaves.append(child)
            else:
                if self.has_children(flag=flag):
                    for child in self.get_children(flag=flag):
                        if child.has_children(flag=flag):
                            for grandchild in child.get_children():
                                
                        leaves.extend(child.get_leaves(flag=flag))
                elif self.is_marked(flag):
                    leaves.append(self)
            '''
            return leaves
        
    
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
        assert position >=0 and type(position) is np.int, \
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
    
        
    def coarsen(self, label=None, coarsening_flag=None):
        """
        Unmark leaves in the subforest 
        """
        for leaf in self.get_leaves(flag=label):
            #
            # Leaf is root node, cannot coarsen
            #
            if not leaf.has_parent(): 
                continue 
            #
            # Leaf's parent is not marked
            #
            elif leaf.has_parent() \
            and coarsening_flag is not None \
            and not leaf.get_parent().is_marked(coarsening_flag):     
                continue
            
            if label is not None:
                #
                # Label specified unmark 
                #  
                for child in leaf.get_parent().get_children():
                    child.unmark(label)
            else:
                #
                # No label specified, delete children
                # 
                leaf.get_parent().delete_children()
    
    
    def refine(self, label=None, refinement_flag=None):
        """
        Refine forest.
        """        
        for tree in self.get_leaves(flag=label):
            if refinement_flag is not None \
            and not tree.is_marked(refinement_flag):   
                continue
            
            if not tree.has_children():
                tree.split()
                
            if label is not None:
                # Mark children with the appropriate label    
                for child in tree.get_children():
                    child.mark(label)
                
        
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
                 regular=True, forest=None, flag=None):
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
            assert self.base()==parent.base() or self.head()==parent.head(),\
                'One of child endpoints should match that of parent.'
        
        
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
        # Assign twin half-edge
        #
        if twin is not None: 
            assert isinstance(twin, HalfEdge), \
                'Input "twin" should be a HalfEdge object.'
            assert self.head()==twin.base() and self.base()==twin.head(),\
                'Base and head vertices of "twin" should equal own head '\
                'and base vertices respectively.'
        self.__twin = twin
        
       
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
        assert self.base()==twin.head() and self.head()==twin.base(),\
            'Own head vertex should be equal to twin base vertex & vice versa.'
        self.__twin = twin
        
    
    def make_twin(self):
        """
        Construct a twin HalfEdge
        """
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
        twin = self.twin()
        if twin is not None and twin.has_children():
            #
            # Share twin's midpoint Vertex
            # 
            t0, t1 = twin.get_children()
            vm = t0.head()
            #
            # Define own children and combine with twin children 
            # 
            c0 = HalfEdge(self.base(), vm, parent=self, twin=t1, position=0)
            c1 = HalfEdge(vm, self.head(), parent=self, twin=t0, position=1)
            t0.assign_twin(c1)
            t1.assign_twin(c0)
        else:
            #
            # Compute new midpoint vertex
            #
            x = convert_to_array([self.base().coordinates(),\
                                  self.head().coordinates()]) 
            xm = 0.5*(x[0,:]+x[1,:]) 
            vm = Vertex(tuple(xm))
            #
            # Define own children independently of neighbor
            # 
            c0 = HalfEdge(self.base(), vm, parent=self, position=0)
            c1 = HalfEdge(vm, self.head(), parent=self, position=1)  
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
        if n_points==1:
            return in_half_edge[0]
        else:
            return in_half_edge
    
        
    
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


class Interval(HalfEdge):
    """
    Interval Class (1D equivalent of a Cell)
    """
    def __init__(self, vertex_left, vertex_right, n_children=2, \
                 regular=True, parent=None, position=None, forest=None):
        """
        Constructor
        """
        assert vertex_left.dim()==1 and vertex_right.dim()==1, \
            'Input "half_edge" should be one dimensional.'
        
        HalfEdge.__init__(self, vertex_left, vertex_right, \
                          n_children=n_children, regular=regular,\
                          parent=parent, position=position, forest=forest)

    
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
        assert isinstance(prev, Interval), \
            'Input "prev" should be an Interval.'
        HalfEdge.assign_previous(self, prev)
    
    
    def assign_next(self, nxt):
        """
        Assign the next interval
        """
        assert isinstance(nxt, Interval), \
            'Input "nxt" should be an Interval.'
        HalfEdge.assign_next(self,nxt)
     
     
    def get_neighbor(self, pivot, flag=None):
        """
        Returns the neighboring interval
        
        Input:
        
            pivot: int, 0 (=left) or 1 (=right)
            
            flag (optional): marker to specify 
        """
        #
        # Move left or right
        # 
        if pivot == 0:
            nb = self.previous()
        elif pivot==1:
            nb = self.next()          
        #
        # Account for flag
        # 
        if flag is not None:
                if nb.is_marked(flag):
                    return nb
                else:
                    return None
        else:
            return nb
    
    
    def split(self, n_children=None):
        """
        Split a given interval into subintervals
        """                 
        #
        # Compute midpoint
        # 
        x0, = self.base().coordinates()
        x1, = self.head().coordinates()
        n = self.n_children()
        for i in range(n+1):
            #
            # Store vertices
            # 
            if i==0:
                base = self.base()
            else:
                if i<n-1:
                    head = Vertex(i*(x1-x0)/2)
                else:
                    head = self.head()
                
                subinterval = \
                    Interval(base, head, parent=self,\
                             regular=self.is_regular(),\
                             position=i-1, n_children=n_children)
                self._children[i-1] = subinterval
                base = head
   
            
        for child in self.get_children():
            #
            # Assign previous/next
            # 
            i = child.get_node_position()
            if i==0:
                # Align own left child with neighbor's right
                nbr = self.get_neighbor(0)
                if nbr is not None and nbr.has_children():
                    child.assign_previous(nbr.get_child(-1))
            else:
                child.assign_previous(self.get_child(i-1))
                
            if i==n-1:
                # Align own right child with neighbor's left
                nbr = self.get_neighbor(1)
                if nbr is not None and nbr.has_children():
                    child.assign_next(nbr.get_child(0))
        
    
    def locate_point(self, point, flag=None):
        """
        Returns the smallest subinterval that contains a given point
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
    
            
    
        
    def reference_map(self, x, jacobian=True, hessian=False, mapsto='physical'):
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
        # Assess input
        # 
        if type(x) is list:
            # Convert list to array
            assert all(isinstance(xi, numbers.Real) for xi in x)
            x = np.array(x)
            
        #
        # Compute mapped points
        # 
        n = len(x)    
        x0, = self.get_vertex(0).coordinates()
        x1, = self.get_vertex(1).coordinates()
        if mapsto == 'physical':
            y = list(x0 + (x1-x0)*x)
        elif mapsto == 'reference':
            y = list((x-x0)/(x1-x0))
        
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
     
                        
class Cell(object):
    """
    Cell object: A two dimensional polygon 
    """
    def __init__(self, half_edges):            
        """
        Constructor
        
        Inputs:
        
            half_edges: HalfEdge, list of half-edges that determine the cell
 
        """    
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
        assert position < self.n_vertices(), 'Input "position" not '
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
                
                - If the pivot is a Vertex, then return a list 
            
            
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
                    # Full rotation
                    # 
                    return neighbors
                else:
                    #
                    # Got at neighbor!
                    # 
                    neighbors.append(neighbor)
                    cell = neighbor
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
                else:
                    #
                    # Got a neighbor
                    # 
                    neighbors_clockwise.append(neighbor)
                    cell = neighbor
            #
            # Combine clockwise and anticlockwise neighbors
            #
            neighbors.extend(reversed(neighbors_clockwise))
            if flag is not None:
                return [nb for nb in neighbors if nb.is_marked(flag)]
            else:
                return neighbors
            
            
    def contains_points(self, points):
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
            in_cell[pos_means_left<0] = False
        
        if len(in_cell)==1:
            return in_cell[0]
        else:
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
        
        Tree.__init__(self, n_children=4, parent=parent, 
                      position=position, forest=grid)
        Cell.__init__(self, half_edges)
        
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
    
        
    def split(self):
        """
        Split QuadCell into 4 subcells
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
            #
            # Define new QuadCell
            # 
            self._children[i] = QuadCell([h1, h2, h3, h4], parent=self, position=i)

            # Increment counter
            i += 1
    
    
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
            for _ in range(n_iterations):
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
                    hess = [np.zeros((2,2,2)) for _ in range(n)]
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
        
            half_edge: int array, pointing to an associated half-edge
        
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
        
        regular_grid
        
        grid_from_gmsh
        
        determine_half_edges
        
        dim
        
        get_neighbor
        
        contains_node
        
    Note: The grid can be used to describe the connectivity associated with a
        ROOT Tree. 
    
    """
    def __init__(self, box=None, resolution=None, dim=None, x=None,
                 connectivity=None, file_path=None, file_format='gmsh'):
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
        """
        #
        # Initialize struct
        #     
        self.is_rectangular = False
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
            self.regular_grid(box=box, resolution=resolution)

        
        # =====================================================================
        # Generate doubly connected edge list 
        # =====================================================================
        self.determine_half_edges()
        
                            
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
        
    
    def regular_grid(self, box, resolution):
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
        n_bnd = len(bnd_hes)
        #
        # Sort half-edges and extract edge numbers
        #
        i_he = bnd_hes.pop()
        he_conn = bnd_hes_conn.pop()
        bnd_hes_sorted = [i_he]
        while len(bnd_hes)>0:
            for i in range(n_bnd):
                if bnd_hes_conn[i][0] == he_conn[1]:
                    #
                    # Initial vertex of he in list matches 
                    # final vertex of popped he.
                    #
                    i_he = bnd_hes.pop(i)
                    he_conn = bnd_hes_conn.pop(i)
                    break
            bnd_hes_sorted.append(i_he) 
        return bnd_hes_sorted
    
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
    

class Mesh(Tree):
    """
    Mesh class
    """
    def __init__(self, dcel=None, box=None, resolution=None, dim=None, x=None,
                 connectivity=None, file_path=None, file_format='gmsh'):
        
        # =====================================================================
        # Doubly connected Edge List
        # =====================================================================
        if dcel is None:
            #
            # Initialize doubly connected edge list if None
            # 
            dcel = DCEL(box=box, resolution=resolution, dim=dim, 
                        x=x, connectivity=connectivity, 
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
    
    
    def locate_point(self):
        """
        Returns the smallest cell/interval in Mesh that contains a given point
        """
        pass
    
    
    

class Mesh1D(Mesh):
    """
    1D Mesh Class
    """
    def __init__(self, dcel=None, box=None, resolution=None, x=None,
                 connectivity=None, file_path=None, file_format='gmsh'):
        
        Mesh.__init__(self, dcel=dcel, box=box, resolution=resolution, dim=1,
                      x=x, connectivity=connectivity, file_path=file_path,
                      file_format=file_format)
        
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
            i_vertices  = dcel.half_edges['connectivity'][i]
            v_base = self.vertices[i_vertices[0]]
            v_head = self.vertices[i_vertices[1]]
            interval = Interval(v_base, v_head) 
            intervals.append(interval)
        #
        # Align intervals (assign next)
        # 
        for i in range(n_intervals):
                i_nxt = dcel.half_edges['next'][i]
                if i_nxt!=-1:
                    intervals[i].assign_next(intervals[i_nxt])
        #
        # Store intervals in Forest
        # 
        self.intervals = Forest(intervals)
    
    
    def locate_point(self, point, flag=None):
        """
        Returns the smallest (flagged) cell containing a given point 
        or None if current cell doesn't contain the point
        
        Input:
            
            point: Vertex
            
        Output:
            
            cell: smallest cell that contains x
                
        """
        for interval in self.intervals.get_children(flag=flag):
            if interval.contains_points(point):
                if not interval.is_marked(flag):
                    return None
                else:
                    return interval.locate_point(point, flag=flag)
    
        
class Mesh2D(Tree):
    """
    2D Mesh class
    """
    def __init__(self, dcel=None, box=None, resolution=None, x=None,
                 connectivity=None, file_path=None, file_format='gmsh'):
        
        Mesh.__init__(self, dcel=dcel, box=box, resolution=resolution, dim=2,
                      x=x, connectivity=connectivity, file_path=file_path,
                      file_format=file_format)
        # ====================================================================
        # HalfEdges
        # ====================================================================
        half_edges = []
        n_hes = dcel.half_edges['n']            
        for i in range(n_hes):
            i_vertices  = dcel.half_edges['connectivity'][i]
            v_base = self.vertices[i_vertices[0]]
            v_head = self.vertices[i_vertices[1]]
            half_edge = HalfEdge(v_base, v_head)
            half_edges.append(half_edge)
        #
        # Assign twins (2D)
        #         
        for i_he in range(n_hes):
            i_twin = dcel.half_edges['twin'][i_he]
            if i_twin!=-1:
                half_edges[i_he].assign_twin(half_edges[i_twin])
        #
        # Add HalfEdges to forest.
        #
        self.half_edges = Forest(half_edges)
                    
        # =====================================================================
        # Cells
        # =====================================================================
        cells = []
        n_cells = dcel.faces['n']
        is_quadmesh = True
        for ic in range(n_cells):
            i_he_pivot = dcel.faces['half_edge'][ic]
            i_he = i_he_pivot
            one_rotation = False 
            i_hes = []
            while not one_rotation:
                i_hes.append(i_he)
                i_he = dcel.half_edges['next'][i_he]
                if i_he==i_he_pivot:
                    one_rotation = True    
            if len(i_hes)==4:
                cells.append(QuadCell([half_edges[i] for i in i_hes]))
            else:
                cells.append(Cell([half_edges[i] for i in i_hes]))
                is_quadmesh = False                
        self.__is_quadmesh = is_quadmesh
        if is_quadmesh:
            #
            # If all cells are quadrilaterals, then its a quadmesh
            # 
            self.cells = Forest(cells)
        else:
            #
            # Otherwise it's just a normal mesh
            # 
            self.cells = cells
    

    def is_quadmesh(self):
        """
        Returns True if all cells in the mesh are QuadCells 
        """
        return self.__is_quadmesh
            
    
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
                if not cell.is_marked(flag):
                    return None
                else:
                    return cell.locate_point(point, flag=flag)
            
        
    def refine(self, label=None, indicator=None, \
               remove_indicator=True):
        """
        Refine the given mesh by splitting cells/intervals marked with a 
        refinement flag and store the tree structre using the mesh_flag.
        """
        Forest.refine(self.cells, label=label, \
                      indicator=indicator, \
                      remove_indicator=remove_indicator)
        
    
    def coarsen(self, label=None, coarsening_flag=None, \
                remove_coarsening_flag=True):
        """
        Coarsen the given mesh by merging cells/intervals marked with a 
        coarsening flag and store the tree structre using the mesh_flag.
        """
        Forest.coarsen(self.cells, label=label, \
                       coarsening_flag=coarsening_flag,\
                       remove_coarsening_flag=remove_coarsening_flag)
        
    
    def is_balanced(self, flag=None):
        """
        Check whether the mesh is balanced
        
        Inputs: 
            
            flag (optional): marker, allowing for the restriction to
                a submesh. 
        
        Note: 
        
            Balancing only works when a mesh contains only QuadCells  
        """
        assert self.is_quadmesh(), \
            'Can only verify 2:1 rule for meshes of QuadCells'
            
        for cell in self.cells.get_leaves(flag=flag):
            for half_edge in cell.get_half_edges():
                nb = cell.get_neighbors(half_edge, flag=flag)
                if nb is not None:
                    for child in nb.get_children(flag=flag):
                        if child.has_children(flag=flag):
                            return False
        return True
       
        
    def balance(self, flag=None):
        """
        Ensure that subcells of current cell conform to the 2:1 rule
        """
        
    
        leaves = set(self.cells.get_leaves(flag=flag))  # set: no duplicates
        while len(leaves)>0:
            leaf = leaves.pop()
            refine_self = False
            for half_edge in leaf.get_half_edges():
                nb = leaf.get_neighbor(half_edge, flag=flag)
                if nb is not None and nb.has_children(flag=flag):
                    twin = half_edge.get_twin()
                    for the_child in twin.get_children():
                        if the_child.cell().has_children(flag=flag):
                            leaf.split()
                            for child in leaf.get_children():
                                child.mark(flag)
                                leaves.add(child)
                                
            for half_edge in leaf.get_half_edges():
                nb = leaf.get_neighbor(half_edge, flag=flag)
                if twin is not None:
                    nb = twin.cell()
                    
                    
                
                    pass
        leaf_dict = {'N': ['SE', 'SW'], 'S': ['NE', 'NW'],
                     'E': ['NW', 'SW'], 'W': ['NE', 'SE']} 

        while len(leaves) > 0:            
            leaf = leaves.pop()
            flag = False
            #
            # Check if leaf needs to be split
            # 
            for direction1 in ['N', 'S', 'E', 'W']:
                nb = leaf.get_neighbor(direction1) 
                if nb == None:
                    pass
                elif nb.type == 'LEAF':
                    pass
                else:
                    for pos in leaf_dict[direction1]:
                        #
                        # If neighor's children nearest to you aren't LEAVES,
                        # then split and add children to list of leaves! 
                        #
                        if nb.children[pos].type != 'LEAF':
                            leaf.split()
                            for child in leaf.children.values():
                                child.mark('support')
                                leaves.add(child)
                                
                                    
                            #
                            # Check if there are any neighbors that should 
                            # now also be split.
                            #  
                            for direction2 in ['N', 'S', 'E', 'W']:
                                nb = leaf.get_neighbor(direction2)
                                if (nb is not None) and \
                                   (nb.type == 'LEAF') and \
                                   (nb.depth < leaf.depth):
                                    leaves.add(nb)
                                
                            flag = True
                            break
                if flag:
                    break
        self.__balanced = True
        
    
    def remove_supports(self):
        """
        Remove the supporting nodes. This is useful after coarsening
        
        """    
        leaves = self.get_leaves()
        while len(leaves) > 0:
            leaf = leaves.pop()
            if leaf.is_marked('support'):
                #
                # Check whether its safe to delete the support cell
                # 
                safe_to_coarsen = True
                for direction in ['N', 'S', 'E', 'W']:
                    nb = leaf.get_neighbor(direction)
                    if nb!=None and nb.has_children():
                        safe_to_coarsen = False
                        break
                if safe_to_coarsen:
                    parent = leaf.parent
                    parent.merge()
                    leaves.append(parent)
        self.__balanced = False
    
    
    def get_boundary_edges(self, flag=None):
        """
        Returns the half-nodes on the boundary
        """
        bnd_hes = []
        #
        # Locate half-edges on the boundary
        # 
        for he in self.half_edges.get_children():
            if he.twin() is None:
                bnd_hes.append(he)
        n_bnd = len(bnd_hes)
        #
        # Sort half-edges
        #
        he = bnd_hes.pop()
        bnd_hes_sorted = [he]
        while n_bnd>0:
            for i in range(n_bnd):
                nxt_he = bnd_hes[i]
                if he.head()==nxt_he.base():
                    bnd_hes_sorted.append(nxt_he)
                    he = bnd_hes.pop(i)
                    break
        return bnd_hes_sorted
 
 

    def get_boundary_vertices(self):
        """
        Returns the Vertices on the boundary
        """
        pass
    


class BiNode(Tree):
    """
    Binary tree Tree
    
    Attributes:
    
        address:
        
        children:
        
        depth:
        
        parent: 
        
        position:
        
        type:
        
        _cell:
        
        _flags:
        
        _support:
        
    """
    def __init__(self, parent=None, position=None, 
                 grid=None, bicell=None):
        """
        Constructor
        
        Inputs:
                    
            parent: BiNode, parental node
            
            position: position within parent 
                ['L','R'] if parent = Tree
                None if parent = None
                i if parent is a ROOT node with specified grid_size
                
            grid: DCEL object, used to store children of ROOT node
                
            bicell: BiCell, physical Cell associated with tree: 
        """
        super().__init__()
        self.parent = parent
        #
        # Types
        # 
        if parent is None:
            #
            # ROOT node
            #
            node_type = 'ROOT'
            node_address = []
            node_depth = 0
            if grid  is not None:
                assert isinstance(grid, DCEL),\
                'Input "grid" should be a DCEL object.'
                
                child_positions = list(range(grid.faces['n']))
                node_children = dict.fromkeys(child_positions)
            else:
                node_children = {'L':None, 'R':None}
                child_positions = ['L','R']
        else:
            #
            # LEAF node
            # 
            node_type = 'LEAF'
            node_address = parent.address + [parent.pos2id(position)]
            node_depth = parent.depth + 1
            node_children = {'L': None, 'R': None}
            if parent.type == 'LEAF':
                parent.type = 'BRANCH'  # modify parent to branch
            child_positions = ['L','R']  
        #
        # Record Attributes
        # 
        self.address = node_address
        
        self._cell = bicell
        self.children = node_children
        self._child_positions = child_positions
        self.depth = node_depth
        self._dim = 1
        self.grid = grid
        self.parent = parent
        self.position = position       
        self.type = node_type
    
    
    def get_neighbor(self, direction):
        """
        Description: Returns the deepest neighboring node, whose depth is at 
            most that of the given node, or 'None' if there aren't any 
            neighbors.
         
        Inputs: 
         
            direction: char, 'L'(left), 'R'(right)
             
        Output: 
         
            neighboring node
         
        """
        
        assert direction in ['L','R'], 'Invalid direction: use "L" or "R".'
        
        if self.type == 'ROOT':
            #
            # ROOT Cells have no neighbors
            # 
            return None
        #
        # For a node in a grid, do a brute force search (comparing vertices)
        #
        elif self.in_grid():
            i = self.address[0]
            p = self.parent
            nx = p.grid.faces['n']
            if direction == 'L':
                if i > 0:
                    return p.children[i-1]
                else:
                    return None
            elif direction == 'R':
                if i < nx-1:
                    return p.children[i+1]
                else:
                    return None
        #
        # Non-ROOT cells 
        # 
        else:
            #
            # Check for neighbors interior to parent cell
            # 
            opposite = {'L': 'R', 'R': 'L'}
            if self.position == opposite[direction]:
                return self.parent.children[direction]
            else: 
                #
                # Children 
                # 
                mu = self.parent.get_neighbor(direction)
                if mu is None or mu.type == 'LEAF':
                    return mu
                else:
                    return mu.children[opposite[direction]]    
                                        
    
    '''
    def has_children(self, position=None, flag=None):
        """
        Determine whether node has children
        
        TODO: Move to parent class
        """
        if position is None:
            # Check for any children
            if flag is None:
                return any(child is not None for child in self.children.values())
            else:
                # Check for flagged children
                for child in self.children.values():
                    if child is not None and child.is_marked(flag):
                        return True
                return False
        else:
            #
            # Check for child in specific position
            # 
            # Ensure position is valid
            pos_error = 'Position should be one of: "L", or "R".'
            assert position in ['L','R'], pos_error
            if flag is None:
                #
                # No flag specified
                #  
                return self.children[position] is not None
            else:
                #
                # With flag
                # 
                return (self.children[position] is not None) and \
                        self.children[position].is_marked(flag) 
    
    '''
    '''
    def get_children(self, flag=None):
        """
        Returns a list of (flagged) children, ordered 
        
        Inputs: 
        
            flag: [None], optional marker
        
        Note: Only returns children that are not None 
        
        TODO: move to parent class
        """
        if self.has_children(flag=flag):
            if self.type=='ROOT' and self.grid_size() is not None:
                #
                # Gridded root node - traverse from bottom to top, left to right
                # 
                nx = self.grid_size()
                for i in range(nx):
                    child = self.children[i]
                    if child is not None:
                        if flag is None:
                            yield child
                        elif child.is_marked(flag):
                            yield child
            #
            # Traverse in left-to-right order
            #  
            else:
                for pos in ['L','R']:
                    child = self.children[pos]
                    if child is not None:
                        if flag is None:
                            yield child
                        elif child.is_marked(flag):
                            yield child
    '''                

    
 
    
    def split(self):
        """
        Add new child nodes to current node
        """
        #
        # If node is linked to cell, split cell and attach children
        #
        assert not(self.has_children()),'Tree already has children.' 
        if self._cell is not None: 
            cell = self._cell
            #
            # Ensure cell has children
            # 
            if not(cell.has_children()):
                cell.split()
            for pos in self._child_positions:
                self.children[pos] = BiNode(parent=self, position=pos, \
                                            bicell=cell.children[pos])
        else:
            for pos in self._child_positions:
                self.children[pos] = BiNode(parent=self, position=pos)
                
    
class QuadNode(Tree):
    """
    Quadtree Tree
    """
    def __init__(self, parent=None, position=None, \
                 grid=None, quadcell=None):
        """
        Constructor
        
        Inputs:
                    
            parent: Tree, parental node
            
            position: position within parent 
                ['SW','SE','NE','NW'] if parent = Tree
                None if parent = None
                [i,j] if parent is a ROOT node with specified grid_size
                
            grid: DCEL object, specifying ROOT node's children
                
            cell: QuadCell, physical Cell associated with tree
            
        """  
        super().__init__()           
        #
        # Types
        # 
        self.parent = parent
        if parent is None:
            #
            # ROOT node
            #
            self.type = 'ROOT'
            node_address = []
            node_depth = 0
            if grid is not None:
                assert isinstance(grid, DCEL), \
                    'Input "grid" should be a DCEL object.'
                child_positions = list(range(grid.faces['n']))
                node_children = dict.fromkeys(child_positions)
            else:
                node_children = {'SW':None, 'SE':None, 'NW':None, 'NE':None}
                child_positions = ['SW', 'SE', 'NW', 'NE']
        else:
            #
            # LEAF node
            # 
            self.type = 'LEAF'
            node_address = parent.address + [self.pos2id(position)]
            node_depth = parent.depth + 1
            node_children = {'SW': None, 'SE': None, 'NW': None, 'NE': None}
            child_positions = ['SW', 'SE', 'NW', 'NE']
            if parent.type == 'LEAF':
                parent.type = 'BRANCH'  # modify parent to branch
            assert grid is None, 'LEAF nodes cannot have a grid.'

        #
        # Record Attributes
        # 
        self.address = node_address
        self._cell = quadcell
        self.children = node_children
        self._child_positions = child_positions
        self.depth = node_depth
        self._dim = 2
        self._flags  = set()
        self.grid = grid
        
        self.position = position
        self._support = False
        
           
        
    def get_neighbor(self, direction):
        """
        Description: Returns the deepest neighboring cell, whose depth is at 
            most that of the given cell, or 'None' if there aren't any 
            neighbors.
         
        Inputs: 
         
            direction: char, 'SW','S','SE','E','NE','N','NW','W'
             
        Output: 
         
            neighbor: QuadNode, neighboring node
                        
        """
        if self.type == 'ROOT':
            #
            # ROOT Cells have no neighbors
            # 
            return None
        #
        # For a node in a grid, use the grid 
        #
        elif self.in_grid():
            i_fc = self.position
            grid = self.parent.grid
            i_nb_fc = grid.get_neighbor(i_fc, direction)
            if i_nb_fc is None:
                return None
            else:
                return self.parent.children[i_nb_fc]
        #
        # Non-ROOT cells 
        # 
        else:
            #
            # Check for neighbors interior to parent cell
            # 
            if direction == 'N':
                interior_neighbors_dict = {'SW': 'NW', 'SE': 'NE'}
            elif direction == 'S':
                interior_neighbors_dict = {'NW': 'SW', 'NE': 'SE'}
            elif direction == 'E':
                interior_neighbors_dict = {'SW': 'SE', 'NW': 'NE'}
            elif direction == 'W':
                interior_neighbors_dict = {'SE': 'SW', 'NE': 'NW'}
            elif direction == 'SW':
                interior_neighbors_dict = {'NE': 'SW'}
            elif direction == 'SE':
                interior_neighbors_dict = {'NW': 'SE'}
            elif direction == 'NW':
                interior_neighbors_dict = {'SE': 'NW'}
            elif direction == 'NE':
                interior_neighbors_dict = {'SW': 'NE'}
            else:
                print("Invalid direction. Use 'N', 'S', 'E', "+\
                      "NE','SE','NW, 'SW', or 'W'.")
            
            if self.position in interior_neighbors_dict:
                neighbor_pos = interior_neighbors_dict[self.position]
                return self.parent.children[neighbor_pos]
            #
            # Check for (children of) parental neighbors
            #
            else:
                if direction in ['SW','SE','NW','NE'] \
                and direction != self.position:
                    # Special case
                    for c1,c2 in zip(self.position,direction):
                        if c1 == c2:
                            here = c1
                    mu = self.parent.get_neighbor(here)
                    if mu is not None \
                    and mu.depth == self.depth-1 \
                    and mu.has_children():
                        #
                        # Diagonal neighbors must share corner vertex
                        # 
                        opposite = {'N':'S', 'S':'N', 'W':'E', 'E':'W'}
                        nb_pos = direction
                        for i in range(len(direction)):
                            if direction[i] == here:
                                nb_pos = nb_pos.replace(here,opposite[here])
                        child = mu.children[nb_pos]
                        return child
                    else:
                        return None
                else:
                    mu = self.parent.get_neighbor(direction)
                    if mu == None or mu.type == 'LEAF':
                        return mu
                    else:
                        #
                        # Reverse dictionary to get exterior neighbors
                        # 
                        exterior_neighbors_dict = \
                           {v: k for k, v in interior_neighbors_dict.items()}
                            
                        if self.position in exterior_neighbors_dict:
                            neighbor_pos = \
                                exterior_neighbors_dict[self.position]
                            return mu.children[neighbor_pos] 
                        

    '''
    TODO: DELETE
    def has_children(self, position=None, flag=None):
        """
        Determine whether node has children
        """
        if position is None:
            # Check for any children
            if flag is None:
                return any(child is not None for child in self.children.values())
            else:
                # Check for flagged children
                for child in self.children.values():
                    if child is not None and child.is_marked(flag):
                        return True
                return False
        else:
            #
            # Check for child in specific position
            # 
            # Ensure position is valid
            pos_error = 'Position should be one of: "SW", "SE", "NW", or "NE"'
            assert position in ['SW','SE','NW','NE'], pos_error
            if flag is None:
                #
                # No flag specified
                #  
                return self.children[position] is not None
            else:
                #
                # With flag
                # 
                return (self.children[position] is not None) and \
                        self.children[position].is_marked(flag) 
    
    
    
    def get_children(self, flag=None):
        """
        Returns a list of (flagged) children, ordered 
        
        Inputs: 
        
            flag: [None], optional marker
        
        Note: Only returns children that are not None 
        """
        if self.has_children(flag=flag):
            if self.type=='ROOT' and self.grid_size() is not None:
                #
                # Gridded root node - traverse from bottom to top, left to right
                # 
                nx, ny = self.grid_size()
                for j in range(ny):
                    for i in range(nx):
                        child = self.children[(i,j)]
                        if child is not None:
                            if flag is None:
                                yield child
                            elif child.is_marked(flag):
                                yield child
            #
            # Usual cell division: traverse in bottom-to-top mirror Z order
            #  
            else:
                for pos in ['SW','SE','NW','NE']:
                    child = self.children[pos]
                    if child is not None:
                        if flag is None:
                            yield child
                        elif child.is_marked(flag):
                            yield child

    '''
    def info(self):
        """
        Displays relevant information about the QuadNode
        """
        print('-'*11)
        print('Tree Info')
        print('-'*11)
        print('{0:10}: {1}'.format('Address', self.address))
        print('{0:10}: {1}'.format('Type', self.type))
        if self.type != 'ROOT':
            print('{0:10}: {1}'.format('Parent', self.parent.address))
            print('{0:10}: {1}'.format('Position', self.position))
        print('{0:10}: {1}'.format('Flags', self._flags))
        if self.has_children():
            if self.type == 'ROOT' and self.grid is not None:
                n = self.grid_size()
                child_string = ''
                for i in range(n):
                    child_string += repr(i)
                    if self.children[i] is not None:
                        child_string += '1,  '
                    else:
                        child_string += '0,  '
                    if i%10==0:
                        child_string += '\n           '
                print('{0:10}:'.format('Children'))
            else:
                child_string = ''
                for key in ['SW','SE','NW','NE']:
                    child = self.children[key]
                    if child != None:
                        child_string += key + ': 1,  '
                    else:
                        child_string += key + ': 0,  '
                print('{0:10}: {1}'.format('Children',child_string))
        else:
            child_string = 'None'
            print('{0:10}: {1}'.format('Children',child_string))

   
                   
    def split(self):
        """
        Add new child nodes to current quadnode
        """
        #
        # If node is linked to cell, split cell and attach children
        #
        assert not(self.has_children()),\
        'QuadNode already has children.' 
        if self._cell is not None: 
            cell = self._cell
            #
            # Ensure cell has children
            # 
            if not(cell.has_children()):
                cell.split()
            for pos in self.children.keys():
                self.children[pos] = \
                    QuadNode(parent=self, position=pos, \
                             quadcell=cell.children[pos])
        else:
            for pos in self._child_positions:
                    self.children[pos] = QuadNode(parent=self, position=pos)
    
    
    
            
   
           
class BCell(Cell):
    """
    Binary tree of sub-intervals in a 1d mesh
    """       
    def __init__(self, half_edges, parent=None, position=None, grid=None):
        """
        Constructor
        """
        super.__init__(self, parent=parent, position=position, grid=grid)
        self.children = [None, None]
        self.cell_type = 'interval'
            

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
          
'''            
class QuadCell(Cell):
    """
    (Tree of) Rectangular cell(s) in mesh
        
    Attributes: 
    
        type - Cells can be one of the following types:
            
            ROOT   - cell on coarsest level
            BRANCH - cell has parent(s) as well as children
            LEAF   - cell on finest refinement level
         
        parent: cell/mesh of which current cell is a sub-cell
         
        children: list of sub-cells of current cell
        
        flag: boolean, used to mark cells
         
        neighbors: addresses of neighboring cells
         
        depth: int, current refinement level 0,1,2,...
        
        address: address within root cell/mesh.
         
        min_size: double, minimum possible mesh width
     
    
    Methods: 
       
    """ 
    
    
    def __init__(self, parent=None, position=None, grid=None, 
                 corner_vertices=None):
        """
        Constructor
        
        
        Inputs:
        
            parent: QuadCell, parental cell (must be specified for LEAF cells).
            
            position: str/tuple, position within parental cell (must be 
                specified for LEAF cells).
            
            grid: DCEL, object containing cell
            
            corner_vertices: vertices on the sw, se, ne, and nw corners
                passed as a list of tuples/vertices or a dict of vertices,
                or a rectangular box = [x_min, x_max, y_min, y_max]
            
        
        Modified: 12/27/2016
        
        TODO: We only need 4 corner vertices plus 4 half-edges for each cell 
        """
        super().__init__()
        
        # =====================================================================
        # Tree Attributes
        # =====================================================================
        in_grid = False
        if parent is None:
            #
            # ROOT Tree
            # 
            cell_type = 'ROOT'
            cell_depth = 0
            cell_address = []
            
            if grid is not None:
                #
                # Cell contained in a DCEL
                # 
                assert position is not None, \
                'Must specify "position" when ROOT QuadCell is in a grid.'  
                
                assert isinstance(grid, DCEL), \
                'Input grid must be an instance of DCEL class.' 
                
                in_grid = True
                self.grid = grid
                self.grid.faces['Cell'][position] = self
                
            else:
                #
                # Free standing ROOT cell
                # 
                assert position is None, \
                'Unattached ROOT cell has no position.'                
        else:
            #
            # LEAF Tree
            #  
            position_missing = 'Position within parent cell must be specified.'
            assert position is not None, position_missing
        
            cell_type = 'LEAF'
            # Change parent type (from LEAF)
            if parent.type == 'LEAF':
                parent.type = 'BRANCH'
            
            cell_depth = parent.depth + 1
            cell_address = parent.address + [self.pos2id(position)]    
        #
        # Set attributes
        # 
        self.type = cell_type
        self.parent = parent
        # TODO: Change to 0,1,2,3
        self.children = {'SW': None, 'SE': None, 'NE':None, 'NW':None} 
        self.depth = cell_depth
        self._dim = 2
        self.address = cell_address
        self.position = position
        # TODO: Unnecessary
        self._child_positions = ['SW','SE','NW','NE']
        
        # TODO: Only want 4 vertices
        self._vertex_positions = ['SW', 'S', 'SE', 'E', 
                                  'NE', 'N', 'NW', 'W','M']
        
        # TODO: unnecessary
        self._corner_vertex_positions = ['SW', 'SE', 'NE', 'NW']
        self._in_grid = in_grid
        
        # =====================================================================
        # Vertices and Edges
        # =====================================================================
        #
        # Initialize
        # 
        vertex_keys = ['SW','S','SE','E','NE','N','NW','W','M']
        vertices = dict.fromkeys(vertex_keys)
        
        # TODO: Don't need edges - use half_edges
        edge_keys = [('M','SW'), ('M','S'), ('M','SE'), ('M','E'),
                     ('M','NE'), ('M','N'), ('M','NW'), ('M','W'),
                     ('SW','NE'), ('NW','SE'), ('SW','S'), ('S','SE'),
                     ('SE','E'), ('E','NE'), ('NE','N'), ('N','NW'),
                     ('NW','W'), ('W','SW'), ('SW','SE'), ('SE','NE'), 
                     ('NE','NW'), ('NW','SW')] 
        edges = dict.fromkeys(edge_keys)
        
        # Classify cell
        is_free_root = self.type == 'ROOT' and not self.in_grid()
        is_leaf = self.type == 'LEAF'
        
        
        #
        # Check whether cell's parent is a rectangle
        # 
        if self.parent is not None:
            is_rectangle = self.parent.is_rectangle()
        elif self.in_grid() and self.grid.is_rectangular:
            is_rectangle = True
        else:
            is_rectangle = False    
        #
        # Corner vertices
        #         
        if is_free_root:
            #
            # ROOT Cell not in grid: Must specify the corner vertices
            # 
            if corner_vertices is None:
                #
                # Use default corner vertices
                #
                vertices['SW'] = Vertex((0,0))
                vertices['SE'] = Vertex((1,0))
                vertices['NE'] = Vertex((1,1))
                vertices['NW'] = Vertex((0,1))
                
                is_rectangle = True
            else:
                #
                # Determine the input type
                # 
                if type(corner_vertices) is list:
                    assert len(corner_vertices) == 4, \
                    '4 Vertices needed to specify QuadCell'
                    
                    if all([isinstance(v,numbers.Real) \
                            for v in corner_vertices]):
                        #
                        # Box [x_min, x_max, y_min, y_max]
                        # 
                        x_min, x_max, y_min, y_max = corner_vertices
                        
                        assert x_min<x_max and y_min<y_max, \
                        'Bounds of rectangular box should be ordered.'
                        
                        vertices['SW'] = Vertex((x_min, y_min))
                        vertices['SE'] = Vertex((x_max, y_min))
                        vertices['NE'] = Vertex((x_max, y_max))
                        vertices['NW'] = Vertex((x_min, y_max))
                        
                        #
                        # Cell is a rectangle: record it
                        # 
                        is_rectangle = True
                        
                    elif all([type(v) is tuple for v in corner_vertices]):
                        #
                        # Vertices passed as tuples 
                        #
                        
                        # Check tuple length
                        assert all([len(v)==2 for v in corner_vertices]), \
                            'Vertex tuples should be of length 2.'
                        
                        vertices['SW'] = Vertex(corner_vertices[0])
                        vertices['SE'] = Vertex(corner_vertices[1])
                        vertices['NE'] = Vertex(corner_vertices[2])
                        vertices['NW'] = Vertex(corner_vertices[3])
                        
                    elif all([isinstance(v, Vertex) for v in corner_vertices]):
                        #
                        # Vertices passed in list 
                        #             
                        vertices['SW'] = corner_vertices[0]
                        vertices['SE'] = corner_vertices[1] 
                        vertices['NE'] = corner_vertices[2]
                        vertices['NW'] = corner_vertices[3]  

                elif type(corner_vertices) is dict:
                    #
                    # Vertices passed in a dictionary
                    #  
                    for pos in ['SW', 'SE', 'NE', 'NW']:
                        assert pos in corner_vertices.keys(), \
                        'Dictionary should contain at least corner positions.'
                    
                    for pos in corner_vertices.keys():
                        assert isinstance(corner_vertices[pos],Vertex), \
                        'Dictionary values should be of type Vertex.'
                        if vertices[pos] is None:
                            vertices[pos] = corner_vertices[pos]
                            
        
                #
                # Check winding order by trying to compute 2x the area
                #
                cnr_positions = ['SW','SE','NE','NW']
                winding_error = 'Cell vertices not ordered correctly.'
                is_positive = 0
                for i in range(4):
                    v_prev = vertices[cnr_positions[(i-1)%4]].coordinates()
                    v_curr = vertices[cnr_positions[i%4]].coordinates()
                    is_positive += (v_curr[0]+v_prev[0])*(v_curr[1]-v_prev[1])
                assert is_positive > 0, winding_error
                
        elif in_grid:    
            #
            # ROOT Cell contained in grid
            # 
            assert corner_vertices is None, \
            'Input "cell_vertices" cannot be specified for cell in grid.'
            
            # Use the initial and final vertices of half-edges as cell
            # corner vertices.
            i_he = grid.faces['half_edge'][self.position]
            sub_dirs = {'S': ['SW','SE'], 'E': ['SE', 'NE'], 
                        'N': ['NE','NW'], 'W': ['NW','SW'] }
            for i in range(3):
                direction = grid.half_edges['position'][i_he]
                for j in range(2):
                    sub_dir = sub_dirs[direction][j]
                    i_vtx = grid.half_edges['connectivity'][i_he][j] 
                    vertices[sub_dir] = grid.points['coordinates'][i_vtx]
                # Proceed to next half-edge
                i_he = grid.half_edges['next'][i_he]
                
        elif is_leaf:
            #
            # LEAF cells inherit corner vertices from parents
            # 
            inherited_vertices = \
                {'SW': {'SW':'SW', 'SE':'S', 'NE':'M', 'NW':'W'},
                 'SE': {'SW':'S', 'SE':'SE', 'NE':'E', 'NW':'M'}, 
                 'NE': {'SW':'M', 'SE':'E', 'NE':'NE', 'NW':'N'}, 
                 'NW': {'SW':'W', 'SE':'M', 'NE':'N', 'NW':'NW'}}
            
            for cv,pv in inherited_vertices[position].items():
                vertices[cv] = parent.vertices[pv]
        
        #
        # Record corner vertex coordinates
        #
        x_sw, y_sw = vertices['SW'].coordinates()
        x_se, y_se = vertices['SE'].coordinates()
        x_ne, y_ne = vertices['NE'].coordinates()
        x_nw, y_nw = vertices['NW'].coordinates()    
    
        
        #
        # Check one last time whether cell is rectangular
        # 
        if self.parent is None and not is_rectangle:
            is_rectangle = (isclose(x_sw,x_nw) and isclose(x_se,x_ne) and \
                            isclose(y_sw,y_se) and isclose(y_nw,y_ne))
        
        #
        # Edge midpoint vertices
        #       
        opposite = {'N':'S', 'S':'N', 'W':'E', 'E':'W'}
        sub_directions = {'S': ['SW','SE'], 'N': ['NE','NW'], 
                          'E': ['SE','NE'], 'W': ['NW','SW']}
        for direction in ['N','E','S','W']:
            #
            # Check neighbors
            # 
            nbr = self.get_neighbor(direction)
            if nbr is not None and nbr.depth==self.depth:
                vertices[direction] = nbr.vertices[opposite[direction]]
            else:
                x0, y0 = vertices[sub_directions[direction][0]].coordinates()
                x1, y1 = vertices[sub_directions[direction][1]].coordinates()
                vertices[direction] = Vertex((0.5*(x0+x1), 0.5*(y0+y1)))    
        #
        # Middle vertex
        # 
        vertices['M'] = vertices['M'] = Vertex((0.25*(x_sw+x_se+x_ne+x_nw),\
                                                0.25*(y_sw+y_se+y_ne+y_nw)))
        
        #
        # Half-Edges 
        # 
        if in_grid:
            #
            # Cell in grid, obtain half-edges from doubly connected edge list
            #
            half_edges = [None, None, None, None] 
            i_he = grid.faces['half_edge'][self.position]
            for i in range(4):
                #
                # New half-edge using base and head vertices
                #
                i_base, i_head = grid.half_edges['connectivity'][i_he]
                v_base = grid.points['coordinates'][i_base]
                v_head = grid.points['coordinates'][i_head]
                half_edges[i] = HalfEdge(v_base, v_head, cell=self)
                #
                # Next half-edge
                # 
                i_he = grid.half_edges['next'][i_he]
            
            #
            # Assign previous and next half-edges
            # 
            
                    
        #
        # Edges
        # 
        if is_leaf:
            #
            # LEAF cells inherit edges from parents
            # 
            inherited_edges = \
                {'SW': { ('SW','SE'):('SW','S'), ('SE','NE'):('M','S'), 
                         ('NE','NW'):('M','W'), ('NW','SW'):('W','SW'),
                         ('SW','NE'):('M','SW') }, 
                 'SE': { ('SW','SE'):('S','SE'), ('SE','NE'):('SE','E'),
                         ('NE','NW'):('M','E'), ('NW','SW'):('M','S'), 
                         ('NW','SE'):('M','SE') },
                 'NE': { ('SW','SE'):('M','E'), ('SE','NE'):('E','NE'), 
                         ('NE','NW'):('NE','N'), ('NW','SW'):('M','N'),
                         ('SW','NE'):('M','NE') }, 
                 'NW': { ('SW','SE'):('M','W'), ('SE','NE'):('M','N'), 
                         ('NE','NW'):('N','NW'), ('NW','SW'):('NW','W'),
                         ('NW','SE'):('M','NW') } }
                 
            for ce,pe in inherited_edges[position].items():
                edges[ce] = parent.edges[pe]
        #
        # New interior Edges
        # 
        edges[('M','SW')] = Edge(vertices['M'],vertices['SW'])
        edges[('M','S')]  = Edge(vertices['M'],vertices['S'])
        edges[('M','SE')] = Edge(vertices['M'],vertices['SE'])
        edges[('M','E')]  = Edge(vertices['M'],vertices['E'])
        edges[('M','NE')] = Edge(vertices['M'],vertices['NE'])
        edges[('M','N')]  = Edge(vertices['M'],vertices['N'])
        edges[('M','NW')] = Edge(vertices['M'],vertices['NW'])
        edges[('M','W')]  = Edge(vertices['M'],vertices['W'])
        
        if is_free_root:
            #
            # Specify all interior edges of free cell
            # 
            edges[('SW','NE')] = Edge(vertices['SW'],vertices['NE']) 
            edges[('NW','SE')] = Edge(vertices['NW'],vertices['SE'])                        
            edges[('SW','S') ] = Edge(vertices['SW'],vertices['S'])
            edges[('S','SE') ] = Edge(vertices['S'],vertices['SE']) 
            edges[('SE','E') ] = Edge(vertices['SE'],vertices['E'])
            edges[('E','NE') ] = Edge(vertices['E'],vertices['NE'])
            edges[('NE','N') ] = Edge(vertices['NE'],vertices['N'])
            edges[('N','NW') ] = Edge(vertices['N'],vertices['NW'])
            edges[('NW','W') ] = Edge(vertices['NW'],vertices['W'])
            edges[('W','SW') ] = Edge(vertices['W'],vertices['SW'])
            edges[('SW','SE')] = Edge(vertices['SW'],vertices['SE'])
            edges[('SE','NE')] = Edge(vertices['SE'],vertices['NE'])
            edges[('NE','NW')] = Edge(vertices['NE'],vertices['NW'])
            edges[('NW','SW')] = Edge(vertices['NW'],vertices['SW'])      
                                        
        else:   
            edge_keys = {'N': [('NE','N'),('N','NW')], 
                         'S': [('SW','S'),('S','SE')],
                         'E': [('SE','E'),('E','NE')],
                         'W': [('NW','W'),('W','SW')] }
            
            for direction in ['N','S','E','W']:
                nbr = self.get_neighbor(direction)
                if nbr is None or nbr.depth < self.depth:
                    #
                    # No/too big neighbor, specify new edges
                    #   
                    for edge_key in edge_keys[direction]:
                        v1, v2 = edge_key
                        edges[edge_key] = Edge(vertices[v1], vertices[v2])
                            
                    if nbr is not None and nbr.depth < self.depth-1:
                        #
                        # Enforce the 2-1 rule
                        # 
                        nbr.split()
                             
                elif nbr.depth == self.depth:
                    #
                    # Neighbor on same level use neighboring edges
                    #            
                    for edge_key in edge_keys[direction]:
                        e0 = edge_key[0].replace(direction,opposite[direction])
                        e1 = edge_key[1].replace(direction,opposite[direction])
                        opp_edge_key = (e1,e0)
                        edges[edge_key] = nbr.edges[opp_edge_key]
                else:
                    raise Exception('Cannot parse neighbor')
            
            #
            # Possibly new diagonal edges
            #
            for edge_key in [('SW','NE'), ('NW','SE')]:
                if edges[edge_key] is None:
                    v1, v2 = edge_key
                    edges[edge_key] = Edge(vertices[v1],vertices[v2])
        #
        # Store vertices and edges
        #  
        self.vertices = vertices
        self.edges = edges    
        
        #
        self._is_rectangle = is_rectangle
    
    
    def area(self):
        """
        Compute the area of the quadcell
        """
        V = self.get_vertices(pos='corners', as_array=True)
        n_points = V.shape[0]
        area = 0
        for i in range(n_points):
            j = (i-1)%4
            area += (V[i,0]+V[j,0])*(V[i,1]-V[j,1])
        return area/2


    def is_rectangle(self):
        """
        Is the cell a rectangle?
        """
        return self._is_rectangle
    
               
    def get_edges(self, pos=None):
        """
        Returns edge with a given position or all
        
        TODO: Modify to get_half_edges 
        """
        edge_dict = {'W':('NW','SW'),'E':('SE','NE'),'S':('SW','SE'),'N':('NE','NW')}   
        if pos == None:
            return [self.edges[edge_dict[key]] for key in ['W','E','S','N']]
        else:
            return self.edges[edge_dict[pos]] 
    
    
    def get_neighbor(self, direction):
        """
        Returns the deepest neighboring cell, whose depth is at most that of the
        given cell, or 'None' if there aren't any neighbors.
         
        Inputs: 
         
            direction: char, 'N'(north), 'S'(south), 'E'(east), or 'W'(west)
             
        Output: 
         
            neighbor: QuadCell, neighboring cell
            
        TODO: Use half-edge to get directions
        """
        #
        # Free-standing ROOT
        # 
        if self.type == 'ROOT' and not self.in_grid():
            return None
        #
        # ROOT cell in grid
        #
        elif self.in_grid():
            i_fc = self.position
            i_nb_fc = self.grid.get_neighbor(i_fc, direction)
            if i_nb_fc is None:
                return None
            else:
                return self.grid.faces['Cell'][i_nb_fc]
            
        #
        # Non-ROOT cells 
        # 
        else:
            #
            # Check for neighbors interior to parent cell
            # 
            if direction == 'N':
                interior_neighbors_dict = {'SW': 'NW', 'SE': 'NE'}
            elif direction == 'S':
                interior_neighbors_dict = {'NW': 'SW', 'NE': 'SE'}
            elif direction == 'E':
                interior_neighbors_dict = {'SW': 'SE', 'NW': 'NE'}
            elif direction == 'W':
                interior_neighbors_dict = {'SE': 'SW', 'NE': 'NW'}
            else:
                print("Invalid direction. Use 'N', 'S', 'E', or 'W'.")
            
            if self.position in interior_neighbors_dict:
                neighbor_pos = interior_neighbors_dict[self.position]
                return self.parent.children[neighbor_pos]
            #
            # Check for (children of) parental neighbors
            #
            else:
                mu = self.parent.get_neighbor(direction)
                if mu is None or mu.type == 'LEAF':
                    return mu
                else:
                    #
                    # Reverse dictionary to get exterior neighbors
                    # 
                    exterior_neighbors_dict = \
                       {v: k for k, v in interior_neighbors_dict.items()}
                        
                    if self.position in exterior_neighbors_dict:
                        neighbor_pos = exterior_neighbors_dict[self.position]
                        return mu.children[neighbor_pos]                       

    
    
    def contains_point(self, points):
        """
        Determine whether the given cell contains a point
        
        Input: 
        
            point: tuple (x,y), list of tuples, or (n,2) array
            
        Output: 
        
            contains_point: boolean array, True if cell contains point, 
            False otherwise
        
        
        Note: Points on the boundary between cells belong to both adjoining
            cells.
            
        """          
        is_single_point = False
        if type(points) is tuple:
            x, y = points
            in_cell = True
            is_single_point = True
        elif type(points) is list:
            for pt in points:
                assert type(pt) is tuple or type(pt) is np.ndarray,\
                'List entries should be 2-tuples or numpy arrays.'
                 
            assert len(pt)==2, \
                'Points passed in list should have 2 components '
                  
            points = np.array(points)
        elif type(points) is np.ndarray:
            assert points.shape[1] == 2,\
                'Array should have two columns.'
            
        if not is_single_point:    
            x, y = points[:,0], points[:,1]
            n_points = len(x)
            in_cell = np.ones(n_points, dtype=np.bool)
          
        for i in range(4):
            #
            # Traverse vertices in counter-clockwise order
            # 
            pos_prev = self._corner_vertex_positions[(i-1)%4]
            pos_curr = self._corner_vertex_positions[i%4]
            
            x0, y0 = self.vertices[pos_prev].coordinates()
            x1, y1 = self.vertices[pos_curr].coordinates()

            # Determine which points lie outside cell
            pos_means_left = (y-y0)*(x1-x0)-( x-x0)*(y1-y0) 
            if is_single_point:
                in_cell = False
                break
            else:
                in_cell[pos_means_left<0] = False
            
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
        if self.contains_point(line[0]) and self.contains_point(line[1]):
            return True
        
        #
        # Check whether line intersects with any cell edge
        # 
        for edge in self.edges.values():
            if edge.intersects_line_segment(line):
                return True
            
        #
        # If function has not terminated yet, there is no intersection
        #     
        return False
    
               
    def locate_point(self, point):
        """
        Returns the smallest cell containing a given point or None if current 
        cell doesn't contain the point
        
        Input:
            
            point: tuple (x,y)
            
        Output:
            
            cell: smallest cell that contains (x,y)
                
        """
        # TESTME: locate_point
        
        if self.contains_point(point):
            if self.type == 'LEAF': 
                return self
            else:
                #
                # If cell has children, find the child containing the point and continue looking from there
                # 
                for child in self.children.values():
                    if child.contains_point(point):
                        return child.locate_point(point)                     
        else:
            return None    
    
    
    def unit_normal(self, edge):
        """
        Return the cell's outward unit normal vector along given edge
        
        Input:
        
            edge: Edge, along which the unit normal is to be computed
        
        TODO: Modify using half-edges    
        """ 
        #
        # Get vertex coordinates
        # 
        xm, ym = self.vertices['M'].coordinates()
        v0,v1 = edge.vertices()
        x0,y0 = v0.coordinates() 
        x1,y1 = v1.coordinates()
        #
        # Form the vector along edge
        #
        v_01 = np.array([x1-x0, y1-y0])
        #
        # Form vector from initial to center
        #
        v_0m = np.array([x0-xm, y0-ym])
        #
        # Construct a normal 
        # 
        n = np.array([v_01[1], -v_01[0]])
        #
        # Adjust if it isn't outward pointing
        # 
        is_outward = np.dot(n, v_0m) > 0
        if is_outward:
            return n/np.linalg.norm(n)
        else:
            return -n/np.linalg.norm(n)
        
    
    def reference_map(self, x, jacobian=False, mapsto='physical'):
        """
        Map a list of points between a physical quadrilateral and a reference 
        cell.
        
        
        Inputs: 
        
            x: double, list of of n (2,) arrays of input points, either in 
                the physical cell (if mapsto='reference') or in the reference
                cell (if mapsto='physical'). 
                
            jacobian: bool, specify whether to return the Jacobian of the
                transformation. 
                
            mapsto: str, 'reference' if mapping onto the refence cell [0,1]^2
                or 'physical' if mapping onto the physical cell. Default is 
                'physical'
            
            
        Outputs:
        
            y: double, list of n (2,) arrays of mapped points
            
            jac: double, list of n (2,2) arrays of jacobian matrices
            
            
        Reference: "Perspective Mappings", David Eberly (2012). 
        """
        #
        # Check whether input x are in the right format
        # 
        assert type(x) is list, 'Input "x" should be a list of arrays.'
        for xi in x:
            assert type(xi) is np.ndarray,\
                'Each entry of input "x" must be an array.'
            assert xi.shape == (2,),\
                'Each entry of input "x" must be a (2,) array.'
        
        #
        # Get cell corner vertices
        #  
        x_verts = self.get_vertices(pos='corners', as_array=True)
        x_sw = x_verts[0,:]
        x_se = x_verts[1,:]
        x_ne = x_verts[2,:]
        x_nw = x_verts[3,:]
        
        #
        # Translated 1,0 and 0,1 vertices form basis
        # 
        Q = np.array([x_se-x_sw, x_nw-x_sw]).T
    
        #
        # Express upper right corner i.n terms of (1,0) and (0,1)
        # 
        b = x_ne-x_sw
        a = np.linalg.solve(Q,b)
        
        if mapsto=='reference':
            #
            # Map from physical cell to [0,1]^2
            #
            y = []
            jac = []
            for xi in x:
                #
                # Express centered point in terms of Q basis
                # 
                xii = np.linalg.solve(Q, xi-x_sw)
                #
                # Common denominator
                #  
                c = a[0]*a[1] + a[1]*(a[1]-1)*xii[0] + a[0]*(a[0]-1)*xii[1]
                #
                # Reference coordinates
                # 
                y0 = a[1]*(a[0]+a[1]-1)*xii[0]/c
                y1 = a[0]*(a[0]+a[1]-1)*xii[1]/c
                y.append(np.array([y0,y1]))
                
                if jacobian:
                    #
                    # Compute the Jacobian for each point
                    #
                    dy0_dx0 = a[1]*(a[0]+a[1]-1)*(1/c+a[1]*(1-a[1])*xii[0]/c**2)
                    dy0_dx1 = a[1]*(a[0]+a[1]-1)*a[0]*(1-a[0])*xii[0]/c**2
                    dy1_dx0 = a[0]*(a[0]+a[1]-1)*a[1]*(1-a[1])*xii[1]/c**2
                    dy1_dx1 = a[0]*(a[0]+a[1]-1)*(1/c+a[0]*(1-a[0])*xii[1]/c**2)
                    dydx = np.array([[dy0_dx0, dy0_dx1],[dy1_dx0, dy1_dx1]])
                    jac.append(dydx.dot(np.linalg.inv(Q)))
            #
            # Return mapped points (and jacobian)
            # 
            if jacobian:        
                return y, jac
            else:
                return y
            
        elif mapsto=='physical':
            #
            # Map from reference cell [0,1]^2 to physical cell
            # 
            y = []
            jac = []
            for xi in x:
                #
                # Common denominator 
                # 
                c = a[0] + a[1] - 1 + (1-a[1])*xi[0] + (1-a[0])*xi[1]
                #
                # Physical coordinates in terms of Q basis
                #
                y0 = a[0]*xi[0]/c
                y1 = a[1]*xi[1]/c
                #
                # Transform, translate, and store physical coordinates 
                # 
                yi = x_sw + np.dot(Q,np.array([y0,y1]))
                y.append(yi)
                    
                if jacobian:
                    #
                    # Compute the jacobian for each point
                    # 
                    dy0_dx0 = a[0]/c + (a[1]-1)*a[0]*xi[0]/c**2
                    dy0_dx1 = (a[0]-1)*a[0]*xi[0]/c**2
                    dy1_dx0 = (a[1]-1)*a[1]*xi[1]/c**2
                    dy1_dx1 = a[1]/c + (a[0]-1)*a[1]*xi[1]/c**2 
                    dydx = np.array([[dy0_dx0, dy0_dx1],[dy1_dx0, dy1_dx1]])
                    jac.append(np.dot(Q,dydx))
            #
            # Return mapped points (and jacobian)
            #
            if jacobian:
                return y, jac
            else:
                return y
    
    def map_from_reference(self, x_ref, jacobian=False):
        """
        Map point from reference cell [0,1]^2 to physical cell
        
        Inputs: 
        
            x_ref: double, (n_points, dim) array of points in the reference
                cell. 
                
        Output:
        
            x: double, (n_points, dim) array of points in the physical cell
            
            
        Note: Older version that only works with rectangles
        TODO: Delete
        """
        x0,x1,y0,y1 = self.box()
        x = np.array([x0 + (x1-x0)*x_ref[:,0], 
                      y0 + (y1-y0)*x_ref[:,1]]).T
        return x
    
    
    
    
    def derivative_multiplier(self, derivative):
        """
        Determine the 
        
        TODO: Delete
        """
        c = 1
        if derivative[0] in {1,2}:
            # 
            # There's a map and we're taking derivatives
            #
            x0,x1,y0,y1 = self.box()
            for i in derivative[1:]:
                if i==0:
                    c *= 1/(x1-x0)
                elif i==1:
                    c *= 1/(y1-y0)
        return c
    
                
    
        
                                
    def split(self):
        """
        Split cell into subcells
        """
        assert not self.has_children(), 'Cell is already split.'
        for pos in self.children.keys():
            self.children[pos] = QuadCell(parent=self, position=pos)
            

    def is_balanced(self):
        """
        Check whether the tree is balanced
        
        """
        children_to_check = {'N': ['SE', 'SW'], 'S': ['NE', 'NW'],
                             'E': ['NW', 'SW'], 'W': ['NE', 'SE']}        
        for leaf in self.get_leaves():
            for direction in ['N','S','E','W']:
                nb = leaf.get_neighbor(direction)
                if nb is not None and nb.has_children():
                    for pos in children_to_check[direction]:
                        child = nb.children[pos]
                        if child.type != 'LEAF':
                            return False
        return True
    
        
    def balance(self):
        """
        Ensure that subcells of current cell conform to the 2:1 rule
        
        TODO: Modify to work with labels, half-edges
        """
        leaves = set(self.get_leaves())  # set: no duplicates
        leaf_dict = {'N': ['SE', 'SW'], 'S': ['NE', 'NW'],
                     'E': ['NW', 'SW'], 'W': ['NE', 'SE']} 

        while len(leaves) > 0:            
            leaf = leaves.pop()
            flag = False
            #
            # Check if leaf needs to be split
            # 
            for direction1 in ['N', 'S', 'E', 'W']:
                nb = leaf.get_neighbor(direction1) 
                if nb == None:
                    pass
                elif nb.type == 'LEAF':
                    pass
                else:
                    for pos in leaf_dict[direction1]:
                        #
                        # If neighor's children nearest to you aren't LEAVES,
                        # then split and add children to list of leaves! 
                        #
                        if nb.children[pos].type != 'LEAF':
                            leaf.split()
                            for child in leaf.children.values():
                                child.mark('support')
                                leaves.add(child)
                                
                                    
                            #
                            # Check if there are any neighbors that should 
                            # now also be split.
                            #  
                            for direction2 in ['N', 'S', 'E', 'W']:
                                nb = leaf.get_neighbor(direction2)
                                if (nb is not None) and \
                                   (nb.type == 'LEAF') and \
                                   (nb.depth < leaf.depth):
                                    leaves.add(nb)
                                
                            flag = True
                            break
                if flag:
                    break
        self.__balanced = True
        
    
    def remove_supports(self):
        """
        Remove the supporting cell. This is useful after coarsening
        
        TODO: Modify to work with cells/flags
        """    
        leaves = self.get_leaves()
        while len(leaves) > 0:
            leaf = leaves.pop()
            if leaf.is_marked('support'):
                #
                # Check whether its safe to delete the support cell
                # 
                safe_to_coarsen = True
                for direction in ['N', 'S', 'E', 'W']:
                    nb = leaf.get_neighbor(direction)
                    if nb!=None and nb.has_children():
                        safe_to_coarsen = False
                        break
                if safe_to_coarsen:
                    parent = leaf.parent
                    parent.merge()
                    leaves.append(parent)
        self.__balanced = False    
    
    ============
    OLD VERSION
    ============    
    def split(self):
        """
        Split cell into subcells.

        """
        assert not self.has_children(), 'QuadCell has children and cannot be split.'
        if self.type == 'ROOT' and self.grid_size != None:
            #
            # ROOT cell's children may be stored in a grid 
            #
            nx, ny = self.grid_size
            cell_children = {}
            xmin, ymin = self.vertices['SW'].coordinates()
            xmax, ymax = self.vertices['NE'].coordinates()
            x = np.linspace(xmin, xmax, nx+1)
            y = np.linspace(ymin, ymax, ny+1)
            for i in range(nx):
                for j in range(ny):
                    if i == 0 and j == 0:
                        # Vertices
                        v_sw = Vertex((x[i]  ,y[j]  ))
                        v_se = Vertex((x[i+1],y[j]  ))
                        v_ne = Vertex((x[i+1],y[j+1]))
                        v_nw = Vertex((x[i]  ,y[j+1]))
                        
                        # Edges
                        e_w = Edge(v_sw, v_nw, parent=self)
                        e_e = Edge(v_se, v_ne, parent=self)
                        e_s = Edge(v_sw, v_se, parent=self)
                        e_n = Edge(v_nw, v_ne, parent=self)
                        
                    elif i > 0 and j == 0:
                        # Vertices
                        v_se = Vertex((x[i+1],y[j]  ))
                        v_ne = Vertex((x[i+1],y[j+1]))
                        v_sw = cell_children[i-1,j].vertices['SE']
                        v_nw = cell_children[i-1,j].vertices['NE']
                        
                        # Edges
                        e_w = cell_children[i-1,j].edges['E']
                        e_e = Edge(v_se, v_ne, parent=self)
                        e_s = Edge(v_sw, v_se, parent=self)
                        e_n = Edge(v_nw, v_ne, parent=self)
                        
                    elif i == 0 and j > 0:
                        # Vertices
                        v_ne = Vertex((x[i+1],y[j+1]))
                        v_nw = Vertex((x[i]  ,y[j+1]))
                        v_sw = cell_children[i,j-1].vertices['NW']
                        v_se = cell_children[i,j-1].vertices['NE']
                        
                        # Edges
                        e_w = Edge(v_sw, v_nw, parent=self)
                        e_e = Edge(v_se, v_ne, parent=self)
                        e_s = cell_children[i,j-1].edges['N']
                        e_n = Edge(v_nw, v_ne, parent=self)
                                            
                    elif i > 0 and j > 0:
                        # Vertices
                        v_ne = Vertex((x[i+1],y[j+1]))
                        v_nw = cell_children[i-1,j].vertices['NE']
                        v_sw = cell_children[i,j-1].vertices['NW']
                        v_se = cell_children[i,j-1].vertices['NE']
                        
                        # Edges
                        e_w = cell_children[i-1,j].edges['E']
                        e_e = Edge(v_se, v_ne, parent=self)
                        e_s = cell_children[i,j-1].edges['N']
                        e_n = Edge(v_nw, v_ne, parent=self)
                        
                    child_vertices = {'SW': v_sw, 'SE': v_se, \
                                      'NE': v_ne,'NW': v_nw}
                    child_edges = {'W':e_w, 'E':e_e, 'S':e_s, 'N':e_n}
                                        
                    child_position = (i,j)
                    cell_children[i,j] = QuadCell(vertices=child_vertices, \
                                              parent=self, edges=child_edges, \
                                              position=child_position)
            self.children = cell_children
        else: 
            if self.type == 'LEAF':    
                #
                # Reclassify LEAF cells to BRANCH (ROOTS remain as they are)
                #  
                self.type = 'BRANCH'
            #
            # Add cell vertices
            #
            x0, y0 = self.vertices['SW'].coordinates()
            x1, y1 = self.vertices['NE'].coordinates()
            hx = 0.5*(x1-x0)
            hy = 0.5*(y1-y0)
             
            if not 'M' in self.vertices:
                self.vertices['M'] = Vertex((x0 + hx, y0 + hy))        
            #
            # Add edge midpoints to parent
            # 
            mid_point = {'N': (x0 + hx, y1), 'S': (x0 + hx, y0), 
                         'W': (x0, y0 + hy), 'E': (x1, y0 + hy)}
            
            #
            # Add dictionary for edges
            #  
            directions = ['N', 'S', 'E', 'W']
            edge_dict = dict.fromkeys(directions)
            sub_edges = dict.fromkeys(['SW','SE','NE','NW'],edge_dict)
            
            opposite_direction = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
            for direction in directions:
                #
                # Check wether we already have a record of this
                # MIDPOINT vertex
                #
                if not (direction in self.vertices):
                    neighbor = self.get_neighbor(direction)
                    opposite_dir = opposite_direction[direction]
                    if neighbor == None: 
                        #
                        # New vertex - add it only to self
                        # 
                        v_new =  Vertex(mid_point[direction])
                        self.vertices[direction] = v_new
                        
                        #
                        # New sub-edges - add them to dictionary
                        # 
                        for edge_key in sub_edges.keys():
                            if direction in list(edge_key):
                                sub_edges[edge_key][direction] = \
                                    Edge(self.vertices[direction], \
                                         self.vertices[edge_key])
                        
                    elif neighbor.type == 'LEAF':
                        #
                        # Neighbor has no children add new vertex to self and 
                        # neighbor.
                        #  
                        v_new =  Vertex(mid_point[direction])
                        self.vertices[direction] = v_new
                        neighbor.vertices[opposite_dir] = v_new 
                        
                        #
                        # Add new sub-edges to dictionary (same as above)
                        #
                        for edge_key in sub_edges.keys():
                            if direction in list(edge_key):
                                sub_edges[edge_key][direction] = \
                                    Edge(self.vertices[direction], \
                                         self.vertices[edge_key]) 
                    else:
                        #
                        # Vertex exists already - get it from neighoring Tree
                        # 
                        self.vertices[direction] = \
                            neighbor.vertices[opposite_dir]
                        
                        #
                        # Edges exist already - get them from the neighbor    
                        # 
                        for edge_key in sub_edges.keys():
                            if direction in list(edge_key):
                                nb_ch_pos = edge_key.replace(direction,\
                                                 opposite_dir)
                                nb_child = neighbor.children[nb_ch_pos]
                                sub_edges[edge_key][direction] = \
                                    nb_child.edges[opposite_dir]
            #            
            # Add child cells
            # 
            sub_vertices = {'SW': ['SW', 'S', 'M', 'W'], 
                            'SE': ['S', 'SE', 'E', 'M'], 
                            'NE': ['M', 'E', 'NE', 'N'],
                            'NW': ['W', 'M', 'N', 'NW']}   
     
              
            for i in sub_vertices.keys():
                child_vertices = {}
                child_vertex_pos = ['SW', 'SE', 'NE', 'NW'] 
                for j in range(4):
                    child_vertices[child_vertex_pos[j]] = self.vertices[sub_vertices[i][j]] 
                child = QuadCell(child_vertices, parent=self, position=i)
                self.children[i] = child
    
                    
      
    def merge(self):
        """
        Delete child nodes
        """
        #
        # Delete unnecessary vertices of neighbors
        # 
        opposite_direction = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
        for direction in ['N','S','E','W']:
            neighbor = self.get_neighbor(direction)
            if neighbor==None:
                #
                # No neighbor on this side delete midpoint vertices
                # 
                del self.vertices[direction]
                
            elif not neighbor.has_children():
                #
                # Neighbouring cell has no children - delete edge midpoints of both
                # 
                del self.vertices[direction]
                op_direction = opposite_direction[direction]
                del neighbor.vertices[op_direction] 
        #
        # Delete all children
        # 
        self.children.clear()
        self.type = 'LEAF'
    

    
    =================================
    OBSOLETE: TREE IS ALwAYS BALANCED
    =================================
    def balance_tree(self):
        """
        Ensure that subcells of current cell conform to the 2:1 rule
        """
        leaves = self.get_leaves()
        leaf_dict = {'N': ['SE', 'SW'], 'S': ['NE', 'NW'],
                     'E': ['NW', 'SW'], 'W': ['NE', 'SE']} 

        while len(leaves) > 0:
            leaf = leaves.pop()
            flag = False
            #
            # Check if leaf needs to be split
            # 
            for direction in ['N', 'S', 'E', 'W']:
                nb = leaf.get_neighbor(direction) 
                if nb == None:
                    pass
                elif nb.type == 'LEAF':
                    pass
                else:
                    for pos in leaf_dict[direction]:
                        #
                        # If neighor's children nearest to you aren't LEAVES,
                        # then split and add children to list of leaves! 
                        #
                        if nb.children[pos].type != 'LEAF':
                            leaf.mark()
                            leaf.split()
                            for child in leaf.children.values():
                                child.mark_support_cell()
                                leaves.append(child)
                            
                            #
                            # Check if there are any neighbors that should 
                            # now also be split.
                            #  
                            for direction in ['N', 'S', 'E', 'W']:
                                nb = leaf.get_neighbor(direction)
                                if nb != None and nb.depth < leaf.depth:
                                    leaves.append(nb)
                                
                            flag = True
                            break
                if flag:
                    break
    
                
        
    def pos2id(self, pos):
        """ 
        Convert position to index: 'SW' -> 0, 'SE' -> 1, 'NW' -> 2, 'NE' -> 3 
        """
        if type(pos) is int:
            return pos
        elif pos in ['SW','SE','NW','NE']:
            pos_to_id = {'SW': 0, 'SE': 1, 'NW': 2, 'NE': 3}
            return pos_to_id[pos]
        else:
            raise Exception('Unidentified format for position.')
    
    
    def id2pos(self, idx):
        """
        Convert index to position: 0 -> 'SW', 1 -> 'SE', 2 -> 'NW', 3 -> 'NE'
        """
        if type(idx) is tuple:
            #
            # DCEL index and positions coincide
            # 
            assert len(idx) == 2, 'Expecting a tuple of integers.'
            return idx
        
        elif idx in ['SW', 'SE', 'NW', 'NE']:
            #
            # Input is already a position
            # 
            return idx
        elif idx in [0,1,2,3]:
            #
            # Convert
            # 
            id_to_pos = {0: 'SW', 1: 'SE', 2: 'NW', 3: 'NE'}
            return id_to_pos[idx]
        else:
            raise Exception('Unrecognized format.')
    '''    

       
       
        
              
class Edge(object):
    '''
    Description: Edge object in quadtree
    
    
    Attributes:
    
    v_begin: Vertex, vertex where edge begins
    
    v_end: Vertex, vertex where edge ends
    
    children: Edge, list of Edge's between [v_begin,v_middle], and between 
              [v_middle,v_end].

    incident_face: Cell, lying to the left of the edge
    
    on_boundary: bool, True if edge lies on boundary 
    
    
    TODO: Delete 
    '''
    
    def __init__(self, v1, v2, parent=None):
        """
        Description: Constructor
        
        Inputs: 
        
            v1, v2: Vertex, two vertices that define the edge
            
            parent: One QuadCell/TriCell containing the edge (not necessary?)
            
            on_boundary: Either None (if not set) or Boolean (True if edge lies on boundary)
        """
        self.__vertices = set([v1,v2])
        
        dim = len(v1.coordinates())
        if dim == 1:
            x0, = v1.coordinates()
            x1, = v2.coordinates()
            nnorm = np.abs(x1-x0)
        elif dim == 2:
            x0,y0 = v1.coordinates()
            x1,y1 = v2.coordinates()
            nnorm = np.sqrt((y1-y0)**2+(x1-x0)**2)
        self.__length = nnorm
        self._flags = set()
        self.__parent = parent 
     
     
    def info(self):
        """
        Display information about edge
        """
        v1, v2 = self.vertices()
        print('{0:10}: {1} --> {2}'.format('Vertices', v1.coordinates(), v2.coordinates()))
        #print('{0:10}: {1}'.format('Length', self.length()))
    
    
    def box(self):
        """
        Return the edge endpoint coordinates x0,y0,x1,y1, where 
        edge: (x0,y0) --> (x1,y1) 
        To ensure consistency, the points are sorted, first in the x-component
        then in the y-component.
        """    
        verts = self.vertex_coordinates()
        verts.sort()
        x0,y0 = verts[0]
        x1,y1 = verts[1]
        return x0,x1,y0,y1
        
        
    def mark(self, flag=None):
        """
        Mark Edge
        
        Inputs:
        
            flag: optional label used to mark edge
        """  
        if flag is None:
            self._flags.add(True)
        else:
            self._flags.add(flag)
            
        
    def unmark(self, flag=None):
        """
        Unmark Edge
        
        Inputs: 
        
            flag: label to be removed
            
        """
        if flag is None:
            # No flag specified -> delete all
            self._flags.clear()
        else:
            # Remove specified flag (if present)
            if flag in self._flags: self._flags.remove(flag)         
 
         
    def is_marked(self,flag=None):
        """
        Check whether Edge is marked
        
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
      
       
    def vertices(self):
        """
        Returns the set of vertices
        """
        return self.__vertices

    
    def vertex_coordinates(self):
        """
        Returns the vertex coordinates as list of tuples
        """        
        v1,v2 = self.__vertices
        return [v1.coordinates(), v2.coordinates()]

        
    def length(self):
        """
        Returns the length of the edge
        """
        return self.__length
    
    
    def intersects_line_segment(self, line):
        """
        Determine whether the edge intersects with a given line segment
        
        Input: 
        
            line: double, list of two tuples
            
        Output:
        
            boolean, true if intersection, false otherwise.
        """        
        # Express edge as p + t*r, t in [0,1]
        v1,v2 = self.vertices()
        p = np.array(v1.coordinates())
        r = np.array(v2.coordinates()) - p
        
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
                 
    
