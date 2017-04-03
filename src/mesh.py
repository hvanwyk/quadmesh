import matplotlib.pyplot as plt
import numpy as np

"""
Created on Jun 29, 2016
@author: hans-werner

"""

class Mesh(object):
    """
    Description: Mesh Class, consisting of a quadcell (background mesh), together with a tree, 
        from which a specific mesh instance can be constructed without deleting previous 
        mesh parameters.
    
    Attributes:
    
    Methods:
    """
    def __init__(self, quadcell=None, root_node=None):
        """
        Description: Constructor
        """
        if root_node.is_linked():
            Warning('Tree node is linked to a cell. Unlinking')    
        root_node.unlink()
        root_node.link(quadcell)
        self.__quadcell = quadcell
        self.__root_node = root_node
        self.__triangulated = False 
     
    @classmethod 
    def copymesh(cls, mesh):
        """
        Copy existing mesh
        """
        quadcell = mesh.quadcell()
        root_node = mesh.root_node().copy()
        return cls(quadcell=quadcell, root_node=root_node)

        
    @classmethod
    def submesh(cls, mesh):
        """
        Construct new mesh from existing mesh 
        """
        quadcell = mesh.quadcell()
        root_node = mesh.root_node().copy()
        return cls(quadcell=quadcell, root_node=root_node) 
    
    
    @classmethod
    def newmesh(cls, box=[0.,1.,0.,1.], grid_size=None):
        """
        Construct new mesh from bounding box and initial grid
        """
        quadcell = QuadCell(box=box, grid_size=grid_size)
        root_node = Node(grid_size=grid_size)
        return cls(quadcell=quadcell, root_node=root_node)
    
     
    def box(self):
        """
        Return the dimensions of the rectangular domain
        """
        return self.root_quadcell().box()
    
        
    def grid_size(self):
        """
        Return grid size on coarsest level
        """
        return self.__quadcell.grid_size
    
        
    def get_number_of_cells(self):
        """
        Return the number of cells
        """
        if hasattr(self, '__n_quadcells'):
            return self.__n_quadcells
        else:
            self.__n_quadcells = len(self.__root_node.find_leaves())
            return self.__n_quadcells
    
            
    def root_node(self):
        """
        Return tree node used for mesh
        """
        return self.__root_node
        
        
    def unmark(self, nodes=False, quadcells=False, quadedges=False, quadvertices=False,
               tricells=False, triedges=False, trivertices=False, all_entities=False):
        """
        Unmark all nodes and/or quadcells, -edges, and -vertices 
        and/or tricells, -edges, and -vertices(recursively)
        """
        if all_entities:
            # 
            # Unmark everything
            # 
            nodes = True
            quadcells = True
            quadedges = True
            quadvertices = True
            tricells = True
            triedges = True
            trivertices = True
            
        node_list = self.root_node().traverse_tree()
        for node in node_list:
            if nodes:
                #
                # Unmark node
                #
                node.unmark(recursive=True)
            if quadcells:
                #
                # Unmark quad cell
                #
                node.quadcell().unmark()
            if quadedges:
                #
                # Unmark quad edges
                #
                for edge in node.quadcell().edges.values():
                    edge.unmark()
            if quadvertices:
                #
                # Unmark quad vertices
                #
                for vertex in node.quadcell().vertices.values():
                    vertex.unmark()
            if tricells or triedges or trivertices:
                if node.has_tricells():
                    for triangle in node.tricells():
                        if tricells:
                            #
                            # Unmark triangular cells
                            # 
                            triangle.unmark()
                        if triedges:
                            #
                            # Unmark triangle edges
                            # 
                            for edge in triangle.edges.values():
                                edge.unmark()
                        if trivertices:
                            #
                            # Unmark triangle vertices
                            #
                            for vertex in triangle.vertices.values():
                                vertex.unmark()
     
    def balance(self):
        """
        Balance the tree associated with the mesh
        """            
        self.__root_node.balance()
        
        
    def root_quadcell(self):
        """
        Return the root_quadcell
        """    
        return self.__quadcell
     
    
    def is_triangulated(self):
        """
        Determine whether triangular mesh is present
        """
        return self.__triangulated
    
    
    def triangulate(self):
        """
        Generate a triangulation
        
        TODO: Incomplete 
        """
        node = self.root_node()
        if not(node.is_balanced()):
            self.balance()
            
        self.__triangulated = True
    
    
    def iter_quadcells(self):
        """
        Iterate over active quad cells
        
        Output:
        
            quadcell_list, list of all active quadrilateral cells
        """ 
        quadcell_list = []
        node = self.root_node()
        for leaf in node.find_leaves():
            quadcell_list.append(leaf.quadcell())
        return quadcell_list
    
    
    def iter_quadedges(self):
        """
        Iterate over quadcell edges
        
        Output: 
        
            quadedge_list, list of all active quadcell edges
        """
        
        quadedge_list = []
        #
        # Unmark all edges
        # 
        self.unmark(quadedges=True)
        for cell in self.iter_quadcells():
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
        self.unmark(quadedges=True)
        return quadedge_list
        
                    
    def iter_quadvertices(self):
        """
        Iterate over quad cell vertices
        
        Output: 
        
            quadvertex_list, list of all active quadcell vertices
        """
        quadvertex_list = []
        #
        # Unmark all vertices
        # 
        self.unmark(quadvertices=True)
        for cell in self.iter_quadcells():
            for direction in ['SW','SE','NW','NE']:
                vertex = cell.vertices[direction]
                if not(vertex.is_marked()):
                    #
                    # New vertex: add it to the list
                    #
                    quadvertex_list.append(vertex)
                    vertex.mark()
        return quadvertex_list
    
    
    def iter_tricells(self):
        """
        Iterate over triangles
        
        Output: 
        
            tricell_list, list of all active triangular cells
        """
        tricell_list = []
        #
        # Unmark all triangle cells
        #
        self.unmark(tricells=True)
        for leaf in self.node().find_leaves():
            for triangle in leaf.tricells():
                tricell_list.append(triangle) 
        return tricell_list
    
    
    def iter_triedges(self):
        """
        Iterate over triangle edges
        
        Output: 
        
            triedge_list, list of all active edges
        """
        triedge_list = []
        self.unmark(triedges=True)
        for triangle in self.iter_tricells():
            for edge in triangle.edges():
                if not(edge.is_marked()):
                    triedge_list.append(edge)
                    edge.mark()
        return triedge_list
    
    
    def iter_trivertices(self):
        """
        Iterate over Triangle vertices
        
        Output: 
        
            trivertex_list, list of all active nodes
        """
        trivertex_list = []
        self.unmark(trivertices=True)
        for triangle in self.iter_tricells():
            for vertex in triangle.vertices():
                if not(vertex.is_marked()):
                    trivertex_list.append(vertex)
                    vertex.mark()
        return trivertex_list
    
        
    def refine(self, flag=None):
        """
        Refine mesh by splitting marked LEAF nodes
        """ 
        for leaf in self.root_node().find_leaves():
            if flag is None:
                # Non-selective refinement
                leaf.split()
            else:
                # Refine selectively according to flag
                if leaf.is_marked(flag=flag):
                    leaf.split()
                    leaf.unmark(flag=flag)
    
    
    def coarsen(self):
        """
        Coarsen mesh by merging marked LEAF nodes
        """
        pass
    
    
    def plot_quadmesh(self, ax, name=None, show=True, set_axis=True, 
                      vertex_numbers=False, edge_numbers=False,
                      cell_numbers=False):
        """
        Plot Mesh of QuadCells
        """
        node = self.root_node()
        if set_axis:
            x0, x1, y0, y1 = node.quadcell().box()          
            hx = x1 - x0
            hy = y1 - y0
            ax.set_xlim(x0-0.1*hx, x1+0.1*hx)
            ax.set_ylim(y0-0.1*hy, y1+0.1*hy)
            rect = plt.Polygon([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],fc='b',alpha=0.5)
            ax.add_patch(rect)
        #
        # Plot QuadCells
        #                       
        for cell in self.iter_quadcells():
             
            x0, y0 = cell.vertices['SW'].coordinate()
            x1, y1 = cell.vertices['NE'].coordinate() 

            # Plot current cell
            # plt.plot([x0, x0, x1, x1],[y0, y1, y0, y1],'r.')
            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            if cell.is_marked():
                rect = plt.Polygon(points, fc='r', edgecolor='k')
            else:
                rect = plt.Polygon(points, fc='w', edgecolor='k')
            ax.add_patch(rect)
        
        #
        # Plot Vertex Numbers
        #    
        if vertex_numbers:
            vertices = self.iter_quadvertices()
            v_count = 0
            for v in vertices:
                x,y = v.coordinate()
                x += 0.01
                y += 0.01
                ax.text(x,y,str(v_count),size='smaller')
                v_count += 1
                
        #
        # Plot Edge Numbers
        #
        if edge_numbers:
            edges = self.iter_quadedges()
            e_count = 0
            for e in edges:
                if not(e.is_marked()):
                    v1, v2 = e.vertices()
                    x0,y0 = v1.coordinate()
                    x1,y1 = v2.coordinate()
                    x_pos, y_pos = 0.5*(x0+x1),0.5*(y0+y1)
                    if x0 == x1:
                        # vertical
                        ax.text(x_pos,y_pos,str(e_count),rotation=-90,
                                size='smaller',verticalalignment='center')
                    else:
                        # horizontal
                        y_offset = 0.05*np.abs((x1-x0))
                        ax.text(x_pos,y_pos+y_offset,str(e_count),size='smaller',
                                horizontalalignment='center')                 
                    e_count += 1
                e.mark()
        
        #
        # Plot Cell Numbers
        #
        if cell_numbers:
            cells = self.iter_quadcells()
            c_count = 0
            for c in cells:
                x0,x1,y0,y1 = c.box()
                x_pos, y_pos = 0.5*(x0+x1), 0.5*(y0+y1)
                ax.text(x_pos,y_pos,str(c_count),horizontalalignment='center',
                        verticalalignment='center',size='smaller')
                c_count += 1
        return ax
    
    
    def plot_trimesh(self):
        """
        Plot Mesh of Triangles
        """
        pass
        
        
class Node(object):
    """
    Description: Tree object for storing and manipulating adaptively
        refined quadtree meshes.
    
    Attributes:
    
        type: str, specifying node's relation to parents and/or children  
            'ROOT' (no parent node), 
            'BRANCH' (parent & children), or 
            'LEAF' (parent but no children)
        
        address: int, list allowing access to node's location within the tree
            General form [[i,j],k1,k2,k3,...kd] where d is the depth and
            [i,j] is the (x,y) position in the original mesh (if parent = MESH)
            kd = [0,1,2,3], where 0->SW, 1->SE, 2->NE, 3->NW
            address = [] if ROOT node. 
        
        depth: int, depth within the tree (ROOT nodes are at depth 0).
        
        parent: Node/Mesh whose child this is
        
        children: dictionary of child nodes. 
            If parent = MESH, keys are [i,j],
            Otherwise, keys are SW, SE, NE, NW. 
        
        marked: bool, flag used to refine or coarsen tree
        
        support: bool, indicates whether given node exists only to enforce the 
            2:1 rule. These nodes can be deleted when the mesh is coarsened.
    
    Methods:
    
        pos2id, id2pos
    """
    def __init__(self, parent=None, position=None, \
                 grid_size=None, quadcell=None):
        """
        Constructor
        
        Inputs:
                    
            parent: Node, parental node
            
            position: position within parent 
                ['SW','SE','NE','NW'] if parent = Node
                None if parent = None
                [i,j] if parent is a ROOT node with specified grid_size
                
            grid_size: int, tuple (nx,ny) specifying shape of a
                ROOT node's child array (optional).
                
            quadcell: QuadCell, physical Cell associated with tree
            
        """             
        #
        # Types
        # 
        if parent == None:
            #
            # ROOT node
            #
            node_type = 'ROOT'
            node_address = []
            node_depth = 0
            if grid_size != None:
                assert type(grid_size) is tuple \
                and all(type(i) is int for i in grid_size), \
                'Child grid size should be a tuple of integers'
                nx,ny = grid_size
                node_children = {}
                for i in range(nx):
                    for j in range(ny):
                        node_children[i,j] = None
            else:
                node_children = {'SW':None, 'SE':None, 'NW':None, 'NE':None}
            self.__grid_size = grid_size
        else:
            #
            # LEAF node
            # 
            node_type = 'LEAF'
            node_address = parent.address + [self.pos2id(position)]
            node_depth = parent.depth + 1
            node_children = {'SW': None, 'SE': None, 'NW': None, 'NE': None}
            if parent.type == 'LEAF':
                parent.type = 'BRANCH'  # modify parent to branch
            
        #
        # Record Attributes
        # 
        self.type = node_type
        self.position = position
        self.address = node_address
        self.depth = node_depth
        self.parent = parent
        self.children = node_children
        self.__quadcell = quadcell
        self.__tricells = None
        self.__flags  = set()
        self.__support = False
    
    
    def info(self):
        """
        Display essential information about Node
        """
        print('-'*11)
        print('Node Info')
        print('-'*11)
        print('{0:10}: {1}'.format('Address', self.address))
        print('{0:10}: {1}'.format('Type', self.type))
        if self.type != 'ROOT':
            print('{0:10}: {1}'.format('Parent', self.parent.address))
            print('{0:10}: {1}'.format('Position', self.position))
        if self.has_children():
            if self.type == 'ROOT' and self.grid_size() != None:
                nx, ny = self.grid_size()
                for iy in range(ny):
                    str_row = ''
                    for ix in range(nx):
                        str_row += repr((ix,iy)) + ': ' 
                        if self.children[(ix,iy)] != None:
                            str_row += '1,  '
                        else:
                            str_row += '0,  '
                    if iy == 0:
                        print('{0:10}: {1}'.format('Children', str_row))
                    else:
                        print('{0:11} {1}'.format(' ', str_row))
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
            
            
    def copy(self, position=None, parent=None):
        """
        Copy existing Node without attached cell or parental node
        """
        if self.type == 'ROOT':
            #
            # As ROOT, only copy grid_size
            # 
            node_copy = Node(grid_size=self.grid_size())
        else:
            #
            # Copy parent node and position
            # 
            node_copy = Node(position=position, parent=parent)
            
        if self.has_children():
            for child in self.children.values():
                if child != None:
                    node_copy.children[position] = \
                        child.copy(position=child.position, parent=node_copy) 
        return node_copy
            
        
    def grid_size(self):
        """
        Return the grid size of root node
        """
        assert self.type == 'ROOT', 'Only ROOT nodes have children in grid.'
        return self.__grid_size
        
        
    def find_neighbor(self, direction):
        """
        Description: Returns the deepest neighboring cell, whose depth is at 
            most that of the given cell, or 'None' if there aren't any 
            neighbors.
         
        Inputs: 
         
            direction: char, 'N'(north), 'S'(south), 'E'(east), or 'W'(west)
             
        Output: 
         
            neighboring cell
            
        """
        if self.type == 'ROOT':
            #
            # ROOT Cells have no neighbors
            # 
            return None
        #
        # For a node in a grid, do a brute force search (comparing vertices)
        #
        elif self.in_grid():
            p = self.parent
            nx, ny = p.grid_size()

            i,j = self.address[0]
            if direction == 'N':
                if j < ny-1:
                    return p.children[i,j+1]
                else:
                    return None
            elif direction == 'S':
                if j > 0:
                    return p.children[i,j-1]
                else:
                    return None
            elif direction == 'E':
                if i < nx-1:
                    return p.children[i+1,j]
                else:
                    return None
            elif direction == 'W':
                if i > 0:
                    return p.children[i-1,j]
                else:
                    return None
            elif direction == 'SW':
                if i > 0 and j > 0:
                    return p.children[i-1,j-1]
                else:
                    return None
            elif direction == 'SE':
                if i < nx-1 and j > 0:
                    return p.children[i+1,j-1]
                else:
                    return None
            elif direction == 'NW':
                if i > 0 and j < ny-1:
                    return p.children[i-1,j+1]
                else: 
                    return None
            elif direction == 'NE':
                if i < nx-1 and j < ny-1:
                    return p.children[i+1,j+1]
                else:
                    return None 
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
                print("Invalid direction. Use 'N', 'S', 'E', 'NE','SE','NW, 'SW', or 'W'.")
            
            if self.position in interior_neighbors_dict:
                neighbor_pos = interior_neighbors_dict[self.position]
                return self.parent.children[neighbor_pos]
            #
            # Check for (children of) parental neighbors
            #
            else:
                if direction in ['SW','SE','NW','NE'] and direction != self.position:
                    # Special case
                    for c1,c2 in zip(self.position,direction):
                        if c1 == c2:
                            here = c1
                    mu = self.parent.find_neighbor(here)
                    if mu != None and mu.depth == self.depth-1 and mu.has_children():
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
                    mu = self.parent.find_neighbor(direction)
                    if mu == None or mu.type == 'LEAF':
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
    
    
    def max_depth(self):
        """
        Return the maximum depth of sub-nodes 
        """
        depth = self.depth
        if self.has_children():
            for child in self.children.values():
                d = child.max_depth()
                if d > depth:
                    depth = d 
        return depth
             
    
    def traverse_tree(self):
        """
        Return list of current node and ALL of its sub-nodes        
        """
        
        all_nodes = []
        #
        # Add self to list
        #
        all_nodes.append(self)
        if self.has_children():
            #
            # Iterate over children
            #
            if self.type == 'ROOT' and self.grid_size() is not None:
                #
                # Gridded root node: iterate from left to right, bottom to top
                # 
                nx, ny = self.grid_size()
                for j in range(ny):
                    for i in range(nx):
                        child = self.children[(i,j)]
                        if child is not None:
                            all_nodes.extend(child.traverse_tree())
            else:
                #
                # Usual quadcell division: traverse in bottom-to-top mirror Z order
                # 
                for key in ['SW','SE','NW','NE']:
                    child = self.children[key]
                    if child != None:
                        all_nodes.extend(child.traverse_tree())
            
        return all_nodes
    
    
    def traverse_depthwise(self, childlist=None):
        """
        Return node and all sub-nodes, ordered by depth
        TODO: UNFINISHED
        """
        all_nodes = []
        #
        # Add self to list
        # 
        all_nodes.append(self)
        node = self
        has_children = node.has_children()
        while has_children:
            children = self.get_children()
        return all_nodes
        
        
    def find_leaves(self):
        """
        Return all LEAF sub-nodes of current node
        """
        leaves = []    
        if self.type == 'LEAF' or not(self.has_children()):
            # 
            # LEAF or childless ROOT
            # 
            leaves.append(self)
        else:
            if self.has_children():
                #
                # Iterate
                #
                if self.type == 'ROOT' and self.grid_size() is not None:
                    #
                    # Gridded root node: iterate from left to right, bottom to top
                    # 
                    nx, ny = self.grid_size()
                    for j in range(ny):
                        for i in range(nx):
                            child = self.children[(i,j)]
                            leaves.extend(child.find_leaves())
                else:
                    #
                    # Usual quadcell division: traverse in bottom-to-top mirror Z order
                    #
                    for key in ['SW','SE','NW','NE']:
                        child = self.children[key]
                        if child != None:
                            leaves.extend(child.find_leaves())
                    
        return leaves
    
    
    def find_root(self):
        """
        Return root node
        """
        if self.type == 'ROOT':
            return self
        else:
            return self.parent.find_root()
    
    
    def find_node(self, address):
        """
        Locate node by its address
        TODO: THIS DOESN'T LOOK LIKE IT WILL WORK
        """
        node = self.find_root()
        if address != []:
            #
            # Not the ROOT node
            # 
            for a in address:
                idx = self.id2pos(a)
                node = node.children[idx]
        return node
        
                
    def has_children(self, position=None):
        """
        Determine whether node has children
        """
        if position == None:
            return any(child != None for child in self.children.values())
        else:
            pos_error = 'Position should be one of: "SW", "SE", "NW", or "NE"'
            assert position in ['SW','SE','NW','NE'], pos_error
            return self.children[position] != None 

    
    def get_children(self):
        """
        Returns a list of children, ordered 
        
        TODO: Test 
        """
        children = []
        if self.has_children():
            if self.type=='ROOT' and self.grid_size() is not None:
                #
                # Gridded root node - traverse from bottom to top, left to right
                # 
                nx, ny = self.grid_size()
                for j in range(ny):
                    for i in range(nx):
                        child = self.children[(i,j)]
                        if child is not None:
                            children.append(child)
                #
                # Usual quadcell division: traverse in bottom-to-top mirror Z order
                #  
                else:
                    for pos in ['SW','SE','NW','NE']:
                        children.append(self.children[pos])
        return children

        
    def has_parent(self):
        """
        Determine whether node has parents
        """
        return self.type != 'ROOT'
    
    
    def in_grid(self):
        """
        Determine whether node position is given by coordinates or directions
        """
        return type(self.position) is tuple
    
    
    def mark(self, flag=None):
        """
        Mark node with keyword. 
        
        Recognized keys: 
                
            True, catchall 
            'split', split node
            'merge', delete children
            'support', mark as support node
            'count', mark for counting
        """
        if flag is None:
            self.__flags.add(True)
        else:
            self.__flags.add(flag)
        
    
    def unmark(self, flag=None, recursive=False):
        """
        Unmark node (and sub-nodes)
        
        Inputs: 
        
            flag: 
        
            recursive (False): boolean, unmark all progeny
            
        """
        # Remove tag
        if flag is None:
            self.__flags.clear()
        else:
            self.__flags.remove(flag)
        # Remove tag from children
        if recursive and self.has_children():
            for child in self.children.values():
                child.unmark(flag=flag, recursive=recursive)
     
    
    def is_marked(self,flag=None):
        """
        Check whether a node is marked.
        """
        if flag is None:
            # No flag specified check whether there is any mark
            if self.__flags:
                return True
            else:
                return False
        else:
            # Check for the presence of given flag
            return flag in self.__flags           
    
    
    def is_linked(self):
        """
        Determine whether node is linked to a cell
        """
        return not self.__quadcell is None
    
        
    def link(self,quadcell,recursive=True):
        """
        Link node with QuadCell
        
        Inputs: 
        
            Quadcell: QuadCell object, rectangular cell linked to node
            
            recursive: bool, if True - link entire tree with cells 
        """
        self.__quadcell = quadcell
        if recursive:
            #
            # Link child nodes to appropriate child cells
            #
            assert self.children.keys() == quadcell.children.keys(), \
            'Keys of tree and cell incompatible.'
            
            if self.has_children():
                if not(quadcell.has_children()):
                    #
                    # Cell must be split first
                    #
                    quadcell.split()
             
                for pos in self.children.keys():
                    tree_child = self.children[pos]
                    if tree_child.cell == None:
                        cell_child = quadcell.children[pos]
                        tree_child.link(cell_child,recursive=recursive) 
    
        
    def unlink(self, recursive=True):
        """
        Unlink node from cell
        """
        self.__quadcell = None
        if recursive and self.has_children():
            #
            # Unlink child nodes from cells
            # 
            for child in self.children.values():
                if child != None:
                    child.unlink()
        
    
    def quadcell(self, position=None):
        """
        Return associated quadcell
        """
        return self.__quadcell
       
    
    
    def add_tricells(self, tricells):
        """
        Associate a list of triangular cells with node
        """
        self.__tricells = tricells
        
        
    def tricells(self):
        """
        Return associated tricells 
        """
        return self.__tricells
    
    
    def has_tricells(self):
        """
        Return true if node is associated with list of tricells
        """
        return self.__tricells != None
  
    
    def merge(self):
        """
        Delete all sub-nodes of given node
        """
        for key in self.children.keys():
            self.children[key] = None
        self.type = 'LEAF'
    
    
    def remove(self):
        """
        Remove node from parent's list of children
        """
        assert self.type != 'ROOT', 'Cannot delete ROOT node.'
        self.parent.children[self.position] = None
        
        
    def split(self):
        """
        Add new child nodes to current node
        """
        #
        # If node is linked to cell, split cell and attach children
        #
        assert not(self.has_children()),'Node already has children.' 
        if self.__quadcell != None: 
            cell = self.__quadcell
            #
            # Ensure cell has children
            # 
            if not(cell.has_children()):
                cell.split()
            for pos in self.children.keys():
                self.children[pos] = Node(parent=self, position=pos, \
                                          quadcell=cell.children[pos])
        else:
            for pos in self.children.keys():
                self.children[pos] = Node(parent=self, position=pos)
            
                    
    def is_balanced(self):
        """
        Check whether the tree is balanced
        """
        for leaf in self.find_leaves():
            for direction in ['N','S','E','W']:
                nb = leaf.find_neighbor(direction)
                if nb != None and nb.has_children():
                    for child in nb.children.values():
                        if child.type != 'LEAF':
                            return False
        return True
    
        
    def balance(self):
        """
        Ensure that subcells of current cell conform to the 2:1 rule
        """
        leaves = set(self.find_leaves())  # set: no duplicates
        leaf_dict = {'N': ['SE', 'SW'], 'S': ['NE', 'NW'],
                     'E': ['NW', 'SW'], 'W': ['NE', 'SE']} 

        while len(leaves) > 0:            
            leaf = leaves.pop()
            flag = False
            #
            # Check if leaf needs to be split
            # 
            for direction1 in ['N', 'S', 'E', 'W']:
                nb = leaf.find_neighbor(direction1) 
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
                                nb = leaf.find_neighbor(direction2)
                                if (nb != None) and \
                                   (nb.type == 'LEAF') and \
                                   (nb.depth < leaf.depth):
                                    leaves.add(nb)
                                
                            flag = True
                            break
                if flag:
                    break
    
    
    def remove_supports(self):
        """
        Remove the supporting nodes. This is useful after coarsening
        """    
        leaves = self.find_leaves()
        while len(leaves) > 0:
            leaf = leaves.pop()
            if leaf.is_marked('support'):
                #
                # Check whether its safe to delete the support cell
                # 
                safe_to_coarsen = True
                for direction in ['N', 'S', 'E', 'W']:
                    nb = leaf.find_neighbor(direction)
                    if nb!=None and nb.has_children():
                        safe_to_coarsen = False
                        break
                if safe_to_coarsen:
                    parent = leaf.parent
                    parent.merge()
                    leaves.append(parent)
        self.__balanced = False
                
                
    def pos2id(self, pos):
        """ 
        Convert position to index: 'SW' -> 0, 'SE' -> 1, 'NW' -> 2, 'NE' -> 3 
        """
        if type(pos) is tuple:
            assert len(pos) == 2, 'Expecting a tuple of integers.'
            return pos 
        elif type(pos) is int and 0 <= pos and pos <= 3:
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
            # Grid index and positions coincide
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
            
    def plot(self):
        """
        Plot tree
        """
        pass
    

class QuadCell(object):
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
    
    
    '''
    ================
    OLD CONSTRUCTOR
    ================
    def __init__(self, vertices, parent=None, edges=None,
                 position=None, grid_size=None):
        """
        Description: Initializes the cell (sub-)tree
        
        Inputs: 
            
            vertices: corner points of cell in one of the following formats
                list             - [x0, x1, y0, y1]
                dict of tuples   - {'SW': (x0,y0), 'SE': (x1,y0), 
                                    'NE': (x1,y1), 'NW', (x0,y1) }
                dict of vertices - {'SW': Vertex((x0,y0)), 'SE': Vertex((x1,y0)), 
                                    'NE': Vertex((x1,y1)), 'NW': Vertex((x0,y1))}        
            parent: parental cell
                
            position: own position in parent cell. Formats:
                list - [i,j] i=0...nx, j=0...ny (when grid_size!=None). 
                    left bottom = (0,0) -> right top = (nx,ny).         
                str - NW, SW, NE, or SE
                
            grid_size: array size for children as integer tuple 
                (only for ROOT).
        """
        #
        # Vertices
        # 
        if type(vertices) is list:
            #
            # Vertices in the form [xmin, xmax, ymin, ymax]
            # 
            assert len(vertices) == 4, 'Vertex list must contain 4 entries.'
            x0, x1, y0, y1 = vertices
            cell_vertices = {'SW':Vertex((x0,y0)), 'SE': Vertex((x1,y0)), \
                             'NW':Vertex((x0,y1)), 'NE': Vertex((x1,y1)), \
                             'M': Vertex((0.5*(x0+x1),0.5*(y0+y1)))}
        else:
            #
            # Vertices in dictionary form
            cell_vertices = {}
            for k in ['SW', 'SE', 'NE', 'NW']:
                v = vertices[k]
                #
                # Convert tuple to Vertex if necessary
                #
                if type(v) is tuple:
                    v = Vertex(v)
                    cell_vertices[k] = v 
                elif type(v) is Vertex:
                    cell_vertices[k] = v
                else:
                    raise Exception('Only Vertex or tuple allowed.')
            x0, y0 = cell_vertices['SW'].coordinate()
            x1, y1 = cell_vertices['NE'].coordinate()
            cell_vertices['M'] = Vertex((0.5*(x0+x1),0.5*(y0+y1)))
        self.vertices = cell_vertices
        #
        # Edges
        #
        if edges == None:
            #
            # Edges not specified - define new ones using vertices
            # 
            e_we = Edge(self.vertices['SW'], self.vertices['SE'], parent=self)
            e_sn = Edge(self.vertices['SE'], self.vertices['NE'], parent=self) 
            e_ew = Edge(self.vertices['NE'], self.vertices['NW'], parent=self)
            e_ns = Edge(self.vertices['NW'], self.vertices['SW'], parent=self)
            self.edges = {'S': e_we, 'E': e_sn, 'N': e_ew, 'W': e_ns}
        else:
            #
            # Edges given: Incorporate after some checks
            # 
            assert type(edges) is dict,\
                'Type: %s - should be a dictionary' %(type(edges))
            
            assert all([direction in edges.keys() \
                        for direction in ['S','E','N','W']]), \
                   'Keys: %s- should be N, S, E, W.' %(repr(edges.keys()))
                   
            assert all([type(edge) is Edge for edge in edges.values() ]), \
                   'Values should be in class Edge.'
                   
            self.edges = edges
            
        if parent == None:
            #
            # ROOT cell
            #
            cell_type = 'ROOT'
            cell_depth = 0
            cell_address = []
            if grid_size != None:
                nx,ny = grid_size
                cell_children = {}
                for i in range(nx):
                    for j in range(ny):
                        cell_children[i,j] = None
            else:
                cell_children = {'SW':None, 'SE':None, 'NE':None, 'NW':None}
            self.grid_size = grid_size
        else:
            #
            # LEAF cell
            #
            cell_type = 'LEAF'
            if parent.type == 'LEAF':
                parent.type = 'BRANCH'  # update parent's type
            cell_depth = parent.depth + 1
            cell_address = parent.address + [self.pos2id(position)]
            cell_children = {'SW':None, 'SE':None, 'NE':None, 'NW':None}
        #
        # Set attributes
        #     
        self.parent = parent
        self.children = cell_children
        self.type = cell_type
        self.position = position
        self.address = cell_address
        self.depth = cell_depth
        self.__flag = False
        self.support_cell = False   
        '''
        
    def __init__(self, parent=None, position=None, grid_size=None, box=None):
        """
        Constructor
        
        
        Inputs:
        
            parent: QuadCell, parental cell (must be specified for LEAF cells).
            
            position: str/tuple, position within parental cell (must be 
                specified for LEAF cells).
            
            grid_size: tuple, dimensions of ROOT cell's grid
            
            box: double, list [x0,x1,y0,y1] bnd of cell (default [0,1,0,1])
            
            
        Modified: 12/27/2016
        """
        # =====================================================================
        # Tree Attributes
        # =====================================================================
        if parent == None:
            #
            # ROOT Node
            # 
            cell_type = 'ROOT'
            cell_depth = 0
            cell_address = []
            
            if grid_size == None:
                children = {'SW': None, 'SE': None, 'NE':None, 'NW':None}
            else:
                nx, ny = grid_size
                children = {}
                for i in range(nx):
                    for j in range(ny):
                        children[i,j] = None
            self.grid_size = grid_size
        else:
            #
            # LEAF Node
            #  
            position_missing = 'Position within parent cell must be specified.'
            assert position != None, position_missing
        
            cell_type = 'LEAF'
            # Change parent type (from LEAF)
            if parent.type == 'LEAF':
                parent.type = 'BRANCH'
            
            cell_depth = parent.depth + 1
            cell_address = parent.address + [self.pos2id(position)]    
            children = {'SW': None, 'SE': None, 'NE':None, 'NW':None}
            
        #
        # Set attributes
        # 
        self.type = cell_type
        self.parent = parent
        self.children = children
        self.depth = cell_depth
        self.address = cell_address
        self.position = position
        self.__flags = set()  
        
        
        # =====================================================================
        # Vertices and Edges
        # =====================================================================
        if parent == None:
            #
            # ROOT Cell
            # 
            if box == None:
                # Use default
                box = [0.,1.,0.,1.]                

            box_format = 'The box variable must be a list with 4 entries.'
            assert (type(box) is list) and (len(box)==4), box_format 
            
            x0, x1, y0, y1 = box
            if grid_size == None:
                #
                # 4 subcells
                # 
                xm = 0.5*(x0+x1)
                ym = 0.5*(y0+y1)
                vertices = {'SW': Vertex((x0,y0)),
                            'S' : Vertex((xm,y0)), 
                            'SE': Vertex((x1,y0)),
                            'E' : Vertex((x1,ym)),
                            'NE': Vertex((x1,y1)),
                            'N' : Vertex((xm,y1)),
                            'NW': Vertex((x0,y1)),
                            'W' : Vertex((x0,ym)),
                            'M' : Vertex((xm,ym))}
                                                      
                edges = {('M','SW') : Edge(vertices['M'],vertices['SW']),
                         ('M','S')  : Edge(vertices['M'],vertices['S']),
                         ('M','SE') : Edge(vertices['M'],vertices['SE']),
                         ('M','E')  : Edge(vertices['M'],vertices['E']),
                         ('M','NE') : Edge(vertices['M'],vertices['NE']),
                         ('M','N')  : Edge(vertices['M'],vertices['N']),
                         ('M','NW') : Edge(vertices['M'],vertices['NW']),
                         ('M','W')  : Edge(vertices['M'],vertices['W']),
                         ('SW','NE'): Edge(vertices['SW'],vertices['NE']),
                         ('NW','SE'): Edge(vertices['NW'],vertices['SE']),                         
                         ('SW','S') : Edge(vertices['SW'],vertices['S']),
                         ('S','SE') : Edge(vertices['S'],vertices['SE']), 
                         ('SE','E') : Edge(vertices['SE'],vertices['E']),
                         ('E','NE') : Edge(vertices['E'],vertices['NE']),
                         ('NE','N') : Edge(vertices['NE'],vertices['N']),
                         ('N','NW') : Edge(vertices['N'],vertices['NW']),
                         ('NW','W') : Edge(vertices['NW'],vertices['W']),
                         ('W','SW') : Edge(vertices['W'],vertices['SW']),
                         ('SW','SE'): Edge(vertices['SW'],vertices['SE']),
                         ('SE','NE'): Edge(vertices['SE'],vertices['NE']),
                         ('NE','NW'): Edge(vertices['NE'],vertices['NW']),
                         ('NW','SW'): Edge(vertices['NW'],vertices['SW'])}             
            else:
                #
                # Grid of sub-cells
                #
                nx, ny = grid_size                
                x = np.linspace(x0,x1,nx+1)
                y = np.linspace(y0,y1,ny+1)
                vertices = {}
                edges = {}
                for i in range(nx+1):
                    for j in range(ny+1):
                        # Vertices
                        vertices[i,j] = Vertex((x[i],y[j]))
                        
                        # Children
                        if i<nx and j<ny:
                            children[i,j] = None
                        
                        # Edges
                        if i>0:
                            # Horizontal edges
                            edges[((i-1,j),(i,j))] = \
                                Edge(vertices[i-1,j],vertices[i,j])
                        
                        if j>0:
                            # Vertical edges
                            edges[((i,j-1),(i,j))] = \
                                Edge(vertices[i,j-1],vertices[i,j])
        else: 
            #
            # LEAF Node
            # 
            vertex_keys = ['SW','S','SE','E','NE','N','NW','W','M']
            vertices = dict.fromkeys(vertex_keys)
            edge_keys = [('M','SW'), ('M','S'), ('M','SE'), ('M','E'),
                         ('M','NE'), ('M','N'), ('M','NW'), ('M','W'),
                         ('SW','NE'), ('NW','SE'), ('SW','S'), ('S','SE'),
                         ('SE','E'), ('E','NE'), ('NE','N'), ('N','NW'),
                         ('NW','W'), ('W','SW'), ('SW','SE'), ('SE','NE'), 
                         ('NE','NW'), ('NW','SW')] 
            edges = dict.fromkeys(edge_keys)
            #
            # Inherited Vertices and Edges
            # 
            if parent.type == 'ROOT' and parent.grid_size != None:
                #
                # Cell lies in grid
                #
                i,j = position
                vertices['SW'] = parent.vertices[i,j]
                vertices['SE'] = parent.vertices[i+1,j]
                vertices['NE'] = parent.vertices[i+1,j+1]
                vertices['NW'] = parent.vertices[i,j+1]
                
                x0,y0 = vertices['SW'].coordinate()
                x1,y1 = vertices['NE'].coordinate()
                
                xm = 0.5*(x0+x1)
                ym = 0.5*(y0+y1)     
                vertices['M'] = Vertex((xm,ym))
                
                edges[('SW','SE')] = parent.edges[((i,j),(i+1,j))] 
                edges[('SE','NE')] = parent.edges[((i+1,j),(i+1,j+1))]
                edges[('NE','NW')] = parent.edges[((i,j+1),(i+1,j+1))]
                edges[('NW','SW')] = parent.edges[((i,j),(i,j+1))]
                
            else:
                
                #
                # Parent not gridded
                #
                                                   
                inherited_vertices = \
                    {'SW': {'SW':'SW', 'SE':'S', 'NE':'M', 'NW':'W'},
                     'SE': {'SW':'S', 'SE':'SE', 'NE':'E', 'NW':'M'}, 
                     'NE': {'SW':'M', 'SE':'E', 'NE':'NE', 'NW':'N'}, 
                     'NW': {'SW':'W', 'SE':'M', 'NE':'N', 'NW':'NW'}}
                
                for cv,pv in inherited_vertices[position].items():
                    vertices[cv] = parent.vertices[pv]
                
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
            
            x0,y0 = vertices['SW'].coordinate()
            x1,y1 = vertices['NE'].coordinate()
            xm = 0.5*(x0+x1)
            ym = 0.5*(y0+y1)
            vertices['M'] = Vertex((xm,ym))        
            #
            # Neighboring Vertices and Edges 
            #
            opposite = {'N':'S', 'S':'N', 'W':'E', 'E':'W'}
            midv = {'N': (xm,y1), 'S':(xm,y0), 'W':(x0,ym), 'E':(x1,ym)}
            e_dir = {'N': [('NE','N'),('N','NW')], 
                     'S': [('SW','S'),('S','SE')],
                     'E': [('SE','E'),('E','NE')],
                     'W': [('NW','W'),('W','SW')] }                              
            for direction in ['N','S','E','W']:
                neighbor = self.find_neighbor(direction)
                if neighbor == None or neighbor.depth < self.depth:
                    #
                    # No/too big neighbor, specify vertices and edges
                    #
                    vertices[direction] = \
                        Vertex(midv[direction])
                    
                        
                    for edge_key in e_dir[direction]:
                        v1, v2 = edge_key
                        x0,y0 = vertices[v1].coordinate()
                        x1,y1 = vertices[v2].coordinate()
                        edges[edge_key] = Edge(vertices[v1],vertices[v2])
                            
                    if neighbor != None and neighbor.depth < self.depth-1:
                        #
                        # Enforce the 2-1 rule
                        # 
                        neighbor.split()
                            
                        
                elif neighbor.depth == self.depth:
                    #
                    # Neighbor on same level use neighboring vertices/edges
                    #            
                    vertices[direction] = \
                        neighbor.vertices[opposite[direction]]
                            
                    for edge_key in e_dir[direction]:
                        e0 = edge_key[0].replace(direction,opposite[direction])
                        e1 = edge_key[1].replace(direction,opposite[direction])
                        opp_edge_key = (e1,e0)
                        edges[edge_key] = neighbor.edges[opp_edge_key]
                else:
                    raise Exception('Cannot parse neighbor')
            #
            # New interior edges
            #             
            edges[('M','SW')] = Edge(vertices['M'],vertices['SW'])
            edges[('M','S')]  = Edge(vertices['M'],vertices['S'])
            edges[('M','SE')] = Edge(vertices['M'],vertices['SE'])
            edges[('M','E')]  = Edge(vertices['M'],vertices['E'])
            edges[('M','NE')] = Edge(vertices['M'],vertices['NE'])
            edges[('M','N')]  = Edge(vertices['M'],vertices['N'])
            edges[('M','NW')] = Edge(vertices['M'],vertices['NW'])
            edges[('M','W')]  = Edge(vertices['M'],vertices['W'])
            #
            # Possibly new diagonal edges
            #
            for edge_key in [('SW','NE'), ('NW','SE')]:
                if edges[edge_key] == None:
                    v1, v2 = edge_key
                    edges[edge_key] = Edge(vertices[v1],vertices[v2])
        #
        # Store vertices and edges
        #  
        self.vertices = vertices
        self.edges = edges
        
        
    def box(self):
        """
        Returns the coordinates of the cell's bounding box [x0,x1,y0,y1]
        """
        if self.type == 'ROOT' and self.grid_size != None:
            nx, ny = self.grid_size
            x0, y0 = self.vertices[0,0].coordinate()
            x1, y1 = self.vertices[nx,ny].coordinate()
        else:
            x0, y0 = self.vertices['SW'].coordinate()
            x1, y1 = self.vertices['NE'].coordinate()
        return x0, x1, y0, y1
           
            
    def get_edges(self, pos=None):
        """
        Returns edge with a given position or all 
        """
        edge_dict = {'W':('NW','SW'),'E':('SE','NE'),'S':('SW','SE'),'N':('NE','NW')}   
        if pos == None:
            return [self.edges[edge_dict[key]] for key in ['W','E','S','N']]
        else:
            return self.edges[edge_dict[pos]] 
        
        
    def find_neighbor(self, direction):
        """
        Returns the deepest neighboring cell, whose depth is at most that of the
        given cell, or 'None' if there aren't any neighbors.
         
        Inputs: 
         
            direction: char, 'N'(north), 'S'(south), 'E'(east), or 'W'(west)
             
        Output: 
         
            neighboring cell
            
        """
        if self.parent == None:
            return None
        #
        # For cell in a MESH, do a brute force search (comparing vertices)
        #
        elif self.parent.type == 'ROOT' and self.parent.grid_size != None:
            m = self.parent
            nx, ny = m.grid_size
            i,j = self.position
            if direction == 'N':
                if j < ny-1:
                    return m.children[i,j+1]
                else:
                    return None
            elif direction == 'S':
                if j > 0:
                    return m.children[i,j-1]
                else:
                    return None
            elif direction == 'E':
                if i < nx-1:
                    return m.children[i+1,j]
                else:
                    return None
            elif direction == 'W':
                if i > 0:
                    return m.children[i-1,j]
                else:
                    return None 

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
                mu = self.parent.find_neighbor(direction)
                if mu == None or mu.type == 'LEAF':
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


    def find_leaves(self, with_depth=False):
        """
        Returns a list of all 'LEAF' type sub-cells (and their depths) of a given cell 
        """
        leaves = []
        if self.type == 'LEAF':
            if with_depth:
                leaves.append((self,self.depth))
            else:
                leaves.append(self)
        elif self.has_children():
            for child in self.children.values():
                leaves.extend(child.find_leaves(with_depth))    
        return leaves

   
    def find_cells_at_depth(self, depth):
        """
        Return a list of cells at a certain depth
        """
        cells = []
        if self.depth == depth:
            cells.append(self)
        elif self.has_children():
            for child in self.children.values():
                cells.extend(child.find_cells_at_depth(depth))
        return cells
    
    
    def find_root(self):
        """
        Find the ROOT cell for a given cell
        """
        if self.type == 'ROOT' or self.type == 'MESH':
            return self
        else:
            return self.parent.find_root()
        
        
    def has_children(self):
        """
        Returns True if cell has any sub-cells, False otherwise
        """    
        return any([self.children[pos]!=None for pos in self.children.keys()])
    
    
    def has_parent(self):
        """
        Returns True if cell has a parent cell, False otherwise
        """
        return not self.parent == None
    
    
    def contains_point(self, points, interior=False):
        """
        Determine whether the given cell contains a point
        
        Input: 
        
            point: tuple (x,y), list of tuples, or (n,2) array
            
        Output: 
        
            contains_point: boolean array, True if cell contains point, 
            False otherwise
              
        """
        # TODO: What about points on the boundary? They will be counted double.          
        xy = np.array(points)
        x_min, x_max, y_min, y_max = self.box()
        
        in_box = (x_min <= xy[:,0]) & (xy[:,0] <= x_max) & \
                 (y_min <= xy[:,1]) & (xy[:,1] <= y_max)
        return in_box
            

    
    def intersects_line_segment(self, line):
        """
        Determine whether cell intersects with a given line segment
        
        Input: 
        
            line: double, list of two tuples (x0,y0) and (x1,y1)
            
        Output:
        
            intersects: bool, true if line segment and quadcell intersect
            
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
    
    
    def normal(self, edge):
        """
        Return the cell's outward normal vector along edge
        """    
        xm,ym = self.vertices['M'].coordinate()
        v0,v1 = edge.vertices()
        x0,y0 = v0.coordinate(); x1 = v1.coordinate()[0]
        if np.abs(x0-x1) < 1e-12:
            #
            # Vertical 
            # 
            return np.sign(x0-xm)*np.array([1.,0.])
        else:
            #
            # Horizontal
            # 
            return np.sign(y0-ym)*np.array([0.,1.])
                
     
    def mark(self, flag=None):
        """
        Mark QuadCell
        
        Inputs:
        
            flag: int, optional label used to mark cell
        """  
        if flag is None:
            self.__flags.add(True)
        else:
            self.__flags.add(flag)
            
        
    def unmark(self, flag=None, recursive=False):
        """
        Unmark QuadCell
        
        Inputs: 
        
            flag: label to be removed
        
            recursive: bool, also unmark all subcells
        """
        #
        # Remove label from own list
        #
        if flag is None:
            # No flag specified -> delete all
            self.__flags.clear()
        else:
            # Remove specified flag (if present)
            if flag in self.__flags: self.__flags.remove(flag)
        
        #
        # Remove label from children if applicable   
        # 
        if recursive and self.has_children():
            for child in self.children.values():
                child.unmark(flag=flag, recursive=recursive)
                
 
         
    def is_marked(self,flag=None):
        """
        Check whether quadcell is marked
        
        Input: flag, label for QuadCell: usually one of the following:
            True (catchall), 'split' (split cell), 'count' (counting)
        """ 
        if flag is None:
            # No flag -> check whether set is empty
            if self.__flags:
                return True
            else:
                return False
        else:
            # Check wether given label is contained in quadcell's set
            return flag in self.__flags
                    
                                
    def split(self):
        """
        Split cell into subcells
        """
        assert not self.has_children(), 'Cell is already split.'
        for pos in self.children.keys():
            self.children[pos] = QuadCell(parent=self, position=pos)        
    '''
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
            xmin, ymin = self.vertices['SW'].coordinate()
            xmax, ymax = self.vertices['NE'].coordinate()
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
            x0, y0 = self.vertices['SW'].coordinate()
            x1, y1 = self.vertices['NE'].coordinate()
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
                    neighbor = self.find_neighbor(direction)
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
                        # Vertex exists already - get it from neighoring Node
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
    '''
                    
    '''  
    def merge(self):
        """
        Delete child nodes
        """
        #
        # Delete unnecessary vertices of neighbors
        # 
        opposite_direction = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
        for direction in ['N','S','E','W']:
            neighbor = self.find_neighbor(direction)
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
    '''

    '''
    =================================
    OBSOLETE: TREE IS ALwAYS BALANCED
    =================================
    def balance_tree(self):
        """
        Ensure that subcells of current cell conform to the 2:1 rule
        """
        leaves = self.find_leaves()
        leaf_dict = {'N': ['SE', 'SW'], 'S': ['NE', 'NW'],
                     'E': ['NW', 'SW'], 'W': ['NE', 'SE']} 

        while len(leaves) > 0:
            leaf = leaves.pop()
            flag = False
            #
            # Check if leaf needs to be split
            # 
            for direction in ['N', 'S', 'E', 'W']:
                nb = leaf.find_neighbor(direction) 
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
                                nb = leaf.find_neighbor(direction)
                                if nb != None and nb.depth < leaf.depth:
                                    leaves.append(nb)
                                
                            flag = True
                            break
                if flag:
                    break
    '''
                
        
    def pos2id(self, pos):
        """ 
        Convert position to index: 'SW' -> 0, 'SE' -> 1, 'NE' -> 2, 'NW' -> 3 
        """
        if type(pos) is tuple:
            assert len(pos) == 2, 'Expecting a tuple of integers.'
            return pos 
        elif type(pos) is int and 0 <= pos and pos <= 3:
            return pos
        elif pos in ['SW','SE','NE','NW']:
            pos_to_id = {'SW': 0, 'SE': 1, 'NE': 2, 'NW': 3}
            return pos_to_id[pos]
        else:
            raise Exception('Unidentified format for position.')
    
    
    def id2pos(self, idx):
        """
        Convert index to position: 0 -> 'SW', 1 -> 'SE', 2 -> 'NE', 3 -> 'NW'
        """
        if type(idx) is tuple:
            #
            # Grid index and positions coincide
            # 
            assert len(idx) == 2, 'Expecting a tuple of integers.'
            return idx
        
        elif idx in ['SW', 'SE', 'NE', 'NW']:
            #
            # Input is already a position
            # 
            return idx
        elif idx in [0,1,2,3]:
            #
            # Convert
            # 
            id_to_pos = {0: 'SW', 1: 'SE', 2: 'NE', 3: 'NW'}
            return id_to_pos[idx]
        else:
            raise Exception('Unrecognized format.')
        
        
    def plot(self, ax, show=True, recursive=True, set_axis=True, edges=False):
        """
        Plot the current cell with all of its sub-cells
        """
        if set_axis:
            x0,x1,y0,y1 = self.box()                
            hx = x1-x0
            hy = y1-y0
            ax.set_xlim(x0-0.1*hx, x1+0.1*hx)
            ax.set_ylim(y0-0.1*hy, y1+0.1*hy)    
        
        if edges:
                #
                # Plot all edges   
                # 
                for edge in self.edges.values():
                    v1, v2 = edge.vertices()
                    x_v1, y_v1 = v1.coordinate()
                    x_v2, y_v2 = v2.coordinate()
                    plt.plot([x_v1,x_v2],[y_v1,y_v2],'r')
                        
        if self.has_children() and recursive:            
            for child in self.children.values():
                ax = child.plot(ax, set_axis=False) 
        else:
            x0,x1,y0,y1 = self.box()
            # Plot current cell            
            for vertex in self.vertices.values():
                plt.plot(vertex.coordinate()[0],vertex.coordinate()[1],'ow')
            plt.plot([x0, x0, x1, x1],[y0, y1, y0, y1],'k.')
            
            
            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            if self.__flag:
                rect = plt.Polygon(points, fc='#FA5858', alpha=1, edgecolor='k')
                #elif self.support_cell:
                #    rect = plt.Polygon(points, fc='#64FE2E', alpha=1, edgecolor='k')
            else:
                rect = plt.Polygon(points, fc='w', edgecolor='k')
            ax.add_patch(rect)         
    
        return ax



# ==============
# TriCell Class
# ==============

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
            #e.append(riHalfEdge(v[i]))
        
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
        self.__flags = set()
        
        
    def vertices(self,n):
        return self.__vertices[n]
    
    def edges(self):
        return self.__edges
    
        
    def area(self):
        """
        Compute the area of the triangle
        """
        v = self.__vertices
        a = [v[1].coordinate()[i] - v[0].coordinate()[i] for i in range(2)]
        b = [v[2].coordinate()[i] - v[0].coordinate()[i] for i in range(2)]
        return 0.5*abs(a[0]*b[1]-a[1]*b[0])
    
     
    def normal(self, edge):
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
        
    def find_neighbor(self, edge, tree):
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
            self.__flags.add(True)
        else:
            self.__flags.add(flag)
            
        
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
            self.__flags.clear()
        else:
            # Remove specified flag (if present)
            if flag in self.__flags: self.__flags.remove(flag)
        
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
            if self.__flags:
                return True
            else:
                return False
        else:
            # Check wether given label is contained in quadcell's set
            return flag in self.__flags
        
        
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
    
    
    Methods:
    
    '''
    
    def __init__(self, v1, v2, 
                 parent=None, on_boundary=None):
        """
        Description: Constructor
        
        Inputs: 
        
            v1, v2: Vertex, two vertices that define the edge
            
            parent: One QuadCell/TriCell containing the edge (not necessary?)
            
            on_boundary: Either None (if not set) or Boolean (True if edge lies on boundary)
        """
        self.__vertices = set([v1,v2])
        
        x0,y0 = v1.coordinate()
        x1,y1 = v2.coordinate()
        nnorm = np.sqrt((y1-y0)**2+(x1-x0)**2)
        self.__length = nnorm
        self.__flags = set()
        self.__parent = parent 
     
     
    def info(self):
        """
        Display information about edge
        """
        v1, v2 = self.vertices()
        print('{0:10}: {1} --> {2}'.format('Vertices', v1.coordinate(), v2.coordinate()))
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
            self.__flags.add(True)
        else:
            self.__flags.add(flag)
            
        
    def unmark(self, flag=None):
        """
        Unmark Edge
        
        Inputs: 
        
            flag: label to be removed
            
        """
        if flag is None:
            # No flag specified -> delete all
            self.__flags.clear()
        else:
            # Remove specified flag (if present)
            if flag in self.__flags: self.__flags.remove(flag)         
 
         
    def is_marked(self,flag=None):
        """
        Check whether Edge is marked
        
        Input: flag, label for QuadCell: usually one of the following:
            True (catchall), 'split' (split cell), 'count' (counting)
        """ 
        if flag is None:
            # No flag -> check whether set is empty
            if self.__flags:
                return True
            else:
                return False
        else:
            # Check wether given label is contained in quadcell's set
            return flag in self.__flags     
      
       
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
        return [v1.coordinate(), v2.coordinate()]

        
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
        p = np.array(v1.coordinate())
        r = np.array(v2.coordinate()) - p
        
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
         
            
class Vertex(object):
    """
    Description:
    
    Attributes:
    
        coordinate: double, tuple (x,y)
        
        flag: boolean
    
    Methods: 
    """


    def __init__(self, coordinate):
        """
        Description: Constructor
        
        Inputs: 
        
            coordinate: double tuple, x- and y- coordinates of vertex
            
            on_boundary: boolean, true if on boundary
              
        """
        assert type(coordinate) is tuple, 'Vertex coordinate should be a tuple.'
        self.__coordinate = coordinate
        self.__flags = set()
    
    def coordinate(self):
        """
        Return coordinate tuple
        """
        return self.__coordinate
    
    def mark(self, flag=None):
        """
        Mark Vertex
        
        Inputs:
        
            flag: int, optional label
        """  
        if flag is None:
            self.__flags.add(True)
        else:
            self.__flags.add(flag)
            
        
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
            self.__flags.clear()
        else:
            # Remove specified flag (if present)
            if flag in self.__flags: self.__flags.remove(flag)
        
         
    def is_marked(self,flag=None):
        """
        Check whether Vertex is marked
        
        Input: flag, label for QuadCell: usually one of the following:
            True (catchall), 'split' (split cell), 'count' (counting)
        """ 
        if flag is None:
            # No flag -> check whether set is empty
            if self.__flags:
                return True
            else:
                return False
        else:
            # Check wether given label is contained in quadcell's set
            return flag in self.__flags
