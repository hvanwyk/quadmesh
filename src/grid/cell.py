from grid.edge import Edge
from grid.vertex import Vertex
import matplotlib.pyplot as plt

class Cell(object):
    """
    Description: (Tree of) Rectangular cell(s) in mesh
    
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
    
    DEBUG = True
    
    def __init__(self, vertices, parent=None, position=None):
        """
        Description: Initializes the cell (sub-)tree
        
        Inputs: 
                    
            parent: parental cell/mesh
            
            vertices: dictionary of coordinates {'SW': (x0,y0), 'SE': (x1,y0), 'NE': (x1,y1), 'NW', (x0,y1) }
            
            position: own position in parent cell/mesh
             
                      - if parent == 'MESH', position is a list [i,j], i=0...nx, j=0...ny 
                        left bottom = (0,0) -> right top = (nx,ny).
                          
                      - else, position is one of (NW, SW, NE, SE)
            

        """  
        self.parent = parent
        self.children = {}
        self.flag = False
        self.support_cell = False
        
        self.position = position
        #
        # Classify cell as ROOT or LEAF
        #
        if parent == None:
            #
            # Unanchored cell
            # 
            cell_type = 'ROOT'
            cell_depth = 0
            cell_address = []
            self.depth = 0
            if position != None:
                raise Exception('ROOT cell cannot have a position.')
        else:
            #
            # Anchored cell
            #
            cell_type = 'LEAF'
            #
            # Use parent to set depth and address 
            # 
            if parent.type == 'MESH':
                cell_depth = 1
                if type(position) == str:
                    raise Exception('Position within MESH is a list [i,j] of two integers.')
                else:
                    cell_address = [] + position  # check whether position is a list
            else:
                cell_depth = parent.depth + 1
                if type(position) is list:
                    raise Exception('Position within Cell is one of SW, SE, NE, or NW.')
                else:
                    cell_address = parent.address + [self.pos2id(position)] 

        self.address = cell_address
        self.depth = cell_depth
        self.type = cell_type
        #
        # Position vertices within Cell
        #
        self.vertices = {}
        for k in ['SW', 'SE', 'NE', 'NW']:
            v = vertices[k]
            #
            # Convert tuple to Vertex if necessary
            #
            if type(v) is tuple:
                v = Vertex(v)
                self.vertices[k] = v 
            elif type(v) is Vertex:
                self.vertices[k] = v
            else:
                raise Exception('Vertices should either be of type Vertex or a tuple.')                        
        #
        # Define edges
        # 
        e_we = Edge(self.vertices['SW'], self.vertices['SE'], self)
        e_sn = Edge(self.vertices['SE'], self.vertices['NE'], self) 
        e_ew = Edge(self.vertices['NE'], self.vertices['NW'], self)
        e_ns = Edge(self.vertices['NW'], self.vertices['SW'], self)
        self.edges = {'S': e_we, 'E': e_sn, 'N': e_ew, 'W': e_ns}


    def box(self):
        """
        Description: Returns the coordinates of the cell's bounding box x_min, x_max, y_min, y_max
        """
        x_min, y_min = self.vertices['SW'].coordinate
        x_max, y_max = self.vertices['NE'].coordinate
        return x_min, x_max, y_min, y_max
            
                                     
    def find_neighbor(self, direction):
        """
        Description: Returns the deepest neighboring cell, whose depth is at most that of the given cell, or
                     'None' if there aren't any neighbors.
         
        Inputs: 
         
            direction: char, 'N'(north), 'S'(south), 'E'(east), or 'W'(west)
             
        Output: 
         
            neighboring cell
            
        """
        # TESTME: find_neighbor
        
        if self.parent == None:
            return None
        #
        # For cell in a MESH, do a brute force search (comparing vertices)
        #
        elif self.parent.type == 'MESH':
            m = self.parent
            nx, ny = m.children_array_size
            i,j = self.address
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
                print "Invalid direction. Use 'N', 'S', 'E', or 'W'."
            
            if interior_neighbors_dict.has_key(self.position):
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
                       {v: k for k, v in interior_neighbors_dict.iteritems()}
                        
                    if exterior_neighbors_dict.has_key(self.position):
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
        else:
            for child in self.children.itervalues():
                leaves.extend(child.find_leaves(with_depth))    
        return leaves

   
    def find_cells_at_depth(self, depth):
        """
        Return a list of cells at a certain depth
        """
        cells = []
        if self.depth == depth:
            cells.append(self)
        else:
            for child in self.children.itervalues():
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


    def contains_point(self, point):
        """
        Determine whether the given cell contains a point
        
        Input: 
        
            point: tuple (x,y)
            
        Output: 
        
            contains_point: boolean, True if cell contains point, False otherwise
              
        """
        # TODO: What about points on the boundary?            
        x,y = point
        x_min, y_min = self.vertices['SW'].coordinate
        x_max, y_max = self.vertices['NE'].coordinate 
        
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        else:
            return False
    
    
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
        for edge in self.edges.itervalues():
            if edge.intersects_line_segment(line):
                return True
            
        #
        # If function has not terminated yet, there is no intersection
        #     
        return False
    
               
    def locate_point(self, point):
        """
        Returns the smallest cell containing a given point or None if current cell doesn't contain the point
        
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
                for child in self.children.itervalues():
                    if child.contains_point(point):
                        return child.locate_point(point)                     
        else:
            return None    
            
     
    def mark(self):
        """
        Mark cell
        """   
        self.flag = True
    
    
    def mark_support_cell(self):
        """
        Classify LEAF as support cell. These cells are created to maintain the 2:1 rule. 
        Support cells can then be deleted if they are no longer needed.  
        """
        self.support_cell = True
        
        
    def unmark(self):
        """
        Unmark cell
        """
        self.flag = False
    
    
    def unmark_all(self):
        """
        Umark cell and all children
        """    
        #
        # Unmark self
        # 
        self.unmark()
        
        if self.has_children():
            #
            # Unmark children
            #
            for child in self.children.itervalues():
                child.unmark_all()
        
        
    def split(self):
        """
        Split cell into 4 subcells
        """
        if self.has_children():
            #
            # Not a LEAF or ROOT cell 
            # 
            print 'Cell ', self.address, 'has already been split.'
            pass
        else: 
            if self.type == 'LEAF':    
                #
                # Reclassify LEAF cells to BRANCH (ROOTS remain as they are)
                #  
                self.type = 'BRANCH'
            #
            # Add cell vertices
            #
            x0, y0 = self.vertices['SW'].coordinate
            x1, y1 = self.vertices['NE'].coordinate
            hx = 0.5*(x1-x0)
            hy = 0.5*(y1-y0)
             
            if not self.vertices.has_key('M'):
                self.vertices['M'] = Vertex((x0 + hx, y0 + hy))        
            #
            # Add edge midpoints to parent
            # 
            mid_point = {'N': (x0 + hx, y1), 'S': (x0 + hx, y0), 
                         'W': (x0, y0 + hy), 'E': (x1, y0 + hy)}
            opposite_direction = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
            for direction in ['N', 'S', 'E', 'W']:
                #
                # Check wether we already have a record of this vertex
                #
                if not self.vertices.has_key(direction):
                    neighbor = self.find_neighbor(direction)
                    if neighbor == None or neighbor.type == 'LEAF':
                        #
                        # New vertex - add it
                        # 
                        self.vertices[direction] = Vertex(mid_point[direction])
                    else:
                        #
                        # Vertex exists already - get it from neighouring Node
                        # 
                        opposite_dir = opposite_direction[direction]
                        self.vertices[direction] = neighbor.vertices[opposite_dir]     
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
                child = Cell(child_vertices, parent=self, position=i)
                self.children[i] = child

        
    def merge(self):
        '''
        Delete child nodes
        '''
        self.children.clear()
        self.type = 'LEAF'

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
                            for child in leaf.children.itervalues():
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


    def number_vertices(self, n_vertex):
        """
        Number cell vertices and add vertices to list
        """
        vertex_list = []            
        for pos in ['SW', 'SE', 'NE', 'NW']:
            vertex = self.vertices[pos]
            #
            # Number own corner vertices
            # 
            if vertex.node_number == None:
                vertex.set_node_number(n_vertex)
                vertex_list.append(vertex)
                n_vertex += 1
                   
        if not self.has_children():
            #
            # If no children, return results
            #
            return vertex_list
        else:
            #
            # If cell has children, first number their vertices then return results
            #
            for pos in ['SW', 'SE', 'NE', 'NW']:
                child = self.children[pos]
                v = child.number_vertices(n_vertex)
                vertex_list.extend(v)
                n_vertex = n_vertex + len(v) 
            return vertex_list
        
        
    def pos2id(self, pos):
        """ 
        Convert position to index: 'SW' -> 0, 'SE' -> 1, 'NE' -> 2, 'NW' -> 3 
        """
        if type(pos) == int and 0 <= pos and pos <= 3:
            return pos
        else:
            pos_to_id = {'SW': 0, 'SE': 1, 'NE': 2, 'NW': 3}
            return pos_to_id[pos]
    
    
    def id2pos(self, idx):
        """
        Convert index to position: 0 -> 'SW', 1 -> 'SE', 2 -> 'NE', 3 -> 'NW'
        """
        if any(idx == direction for direction in ['SW', 'SE', 'NE', 'NW']):
            return idx
        else:
            id_to_pos = {0: 'SW', 1: 'SE', 2: 'NE', 3: 'NW'}
            return id_to_pos[idx]
        
        
    def plot(self, ax, show=True, set_axis=True):
        """
        Plot the current cell with all of its sub-cells
        """
            
        if self.has_children():
            if set_axis:
                x0, y0 = self.vertices['SW'].coordinate
                x1, y1 = self.vertices['NE'].coordinate 
                
                hx = x1 - x0
                hy = y1 - y0
                ax.set_xlim(x0-0.1*hx, x1+0.1*hx)
                ax.set_ylim(y0-0.1*hy, y1+0.1*hy)
            
            for child in self.children.itervalues():
                ax = child.plot(ax, set_axis=False) 
        else:
            x0, y0 = self.vertices['SW'].coordinate
            x1, y1 = self.vertices['NE'].coordinate 

            # Plot current cell
            plt.plot([x0, x0, x1, x1],[y0, y1, y0, y1],'r.')
            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            if self.flag:
                rect = plt.Polygon(points, fc='#FA5858', alpha=1, edgecolor='k')
            elif self.support_cell:
                rect = plt.Polygon(points, fc='#64FE2E', alpha=1, edgecolor='k')
            else:
                rect = plt.Polygon(points, fc='w', edgecolor='k')
            ax.add_patch(rect)         
        return ax
