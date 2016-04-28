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
         
        min_size: double, minimum possible mesh width
     
    
    Methods: 
     
        refine, coarsen, has_children,  
    """
    # TESTME: 
    # TODO: --plot_grid
    # TODO: --coarsen
    # TODO: --balance tree 
    # TODO: Global node and edge_lists?  
    
    # Globals
    NODE_NUM = 0
    VERTEX_LIST = [] 
    EDGE_LIST = []
    DEBUG = True
    
    def __init__(self, vertices, parent=None, position=None):
        """
        Description: Initializes the cell (sub-)tree
        
        Inputs: 
                    
            parent: parental cell/mesh
            
            vertices: dictionary of coordinates {'SW': (x0,y0), 'SE': (x1,y0), 'NE': (x1,y1), 'NW', (x0,y1) }
            
            position: own position in parent cell (NW, SW, NE, SE) or None if N/A

        """       
        self.parent = parent
        self.children = {}
        self.flag = False
        self.position = position
        #
        # Classify Node as ROOT or LEAF
        # 
        if self.parent == None or parent.type == 'MESH':
            cell_type = 'ROOT'
            cell_depth = 0               
        else:
            cell_type = 'LEAF'
            cell_depth = parent.depth + 1
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
            if type(v) is Vertex:
                self.vertices[k] = v
                         
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
        # For the ROOT cell in a MESH, do a brute force search (comparing vertices)
        #
        elif self.parent.type == 'MESH':
            x = self.vertices
            for sibling in self.parent.children:
                xx = sibling.vertices
                if direction == 'N':
                    is_neighbor = ( x['NW'] == xx['SW'] and x['NE'] == xx['SE'] )
                elif direction == 'S':
                    is_neighbor = ( x['SW'] == xx['NW'] and x['SE'] == xx['NE'] )
                elif direction == 'E':
                    is_neighbor = ( x['SE'] == xx['SW'] and x['NE'] == xx['NW'] )
                elif direction == 'W':
                    is_neighbor = ( x['SW'] == xx['SE'] and x['NW'] == xx['NE'] ) 
                else:
                    print "Invalid direction. Use 'N', 'S', 'E', or 'W'."
                    return None
                    
                if is_neighbor:
                    return sibling
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


    def find_leaves(self):
        """
        Returns a list of all 'LEAF' type sub-cells of a given cell
        """
        leaves = []
        if self.type == 'LEAF':
            leaves.append(self)
        else:
            for pos in self.children.keys():
                child = self.children[pos]
                if child != None:
                    leaves.extend(child.find_leaves())
            
        return leaves
   
    
    def find_root(self):
        """
        Find the ROOT cell for a given cell
        """
        if self.type == 'ROOT':
            return self
        else:
            return self.parent.find_root()
        
        
    def has_children(self):
        """
        Returns True if cell has any sub-cells, False otherwise
        """    
        return any([self.children[pos]!=None for pos in self.children.keys()])
    
        
    def balance_tree(self):
        """
        Ensure that the quadtree conforms to the 2:1 rule
        """
        root = self.find_root()
        leaves = root.find_leaves()
        leaf_dict = {'N': ['SE', 'SW'], 'S': ['NE', 'NW'],
                     'E': ['NW', 'SW'], 'W': ['NE', 'SE']} 
        print "There are ", len(leaves), " leaves in total."             
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
                            if self.DEBUG:
                                print "node", id(leaf), "must be split"
                            leaf.mark()
                            leaf.refine()
                            for pos in ['NW', 'NE', 'SW', 'SE']:
                                leaves.append(leaf.children[pos])
                            
                            #
                            # Check if there are any neighbors that should 
                            # now also be split.
                            #  
                            for direction in ['N', 'S', 'E', 'W']:
                                if self.DEBUG:
                                    print "Looking in the ", direction
                                nb = leaf.find_neighbor(direction)
                                if nb != None and nb.depth < leaf.depth:
                                    if self.DEBUG:
                                        print "Neighbor needs to be split"
                                    leaves.append(nb)
                                
                            flag = True
                            break
                if flag:
                    break
    
        
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
                rect = plt.Polygon(points, fc='r', edgecolor='k')
            else:
                rect = plt.Polygon(points, fc='w', edgecolor='k')
            ax.add_patch(rect)         
        return ax
    
    
    def contains_point(self, point):
        """
        Determine whether the given cell contains a point
        
        Input: 
        
            point: tuple (x,y)
            
        Output: 
        
            contains_point: boolean, True if cell contains point, False otherwise
              
        """
        # TODO: What about points on the boundary?
        # TESTME: contains_point
            
        x,y = point
        x_min, y_min = self.vertices['SW'].coordinate
        x_max, y_max = self.vertices['NE'].coordinate 
        
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
        else:
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
    
    
    def unmark(self):
        """
        Unmark cell
        """
        self.flag = False
        
        
    def refine(self):
        """
        Subdivide marked cells in cell
        """
        
        if self.has_children():         
            #
            # Refine all progeny that are 'LEAF' cells  
            # 
            leaves = self.find_leaves()
            for leaf in leaves:
                leaf.mark()
                leaf.refine()
        else:
            
            if self.type == 'LEAF':
                # Change type to 'BRANCH'
                self.type = 'BRANCH'
            
            #
            # Add cell vertices
            #
            #
            # Add center vertex to global- and parent vertex lists
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
        
        #
        # Unmark yourself once refinement is done
        #             
        self.unmark()
        
        
    def coarsen(self):
        '''
        Delete all marked sub-cells
        '''
        if self.flag:
            #
            # Cell is flagged
            # 
            if not self.has_children():
                print 'Cell has no children and cannot delete itself.'
                return
            else:
                #
                # Delete all children
                # 
                del self.children
        else:
            if self.DEBUG:
                print 'Cell [%f,%f,%f,%f]' % self.box() 
                     
            for pos in self.children.keys():
                #
                # Delete only flagged children
                # 
                if self.children[pos].flag:
                    if self.DEBUG:
                        print '%s child flagged: deleted' % (pos)
                    
                    del self.children[pos] 
                    
                    if self.DEBUG:
                        print 'updated list of children:', self.children
                else: 
                    self.children[pos].coarsen()
        
    