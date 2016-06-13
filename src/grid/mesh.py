from grid.cell import Cell
from grid.vertex import Vertex
from grid.triangle import Triangle
import numpy
import matplotlib.pyplot as plt

class Mesh(object):
    '''
    Description: (Quad) Mesh object
    
    Attributes:
    
        bounding_box: [xmin, xmax, ymin, ymax]
        
        children: Cell, list of cells contained in mesh 
        
        vertex_list: Vertex, list of vertices (run number_vertices)
        
        connectivity: int, numpy array - element connectivity matrix (run build_connectivity)
        
        max_depth: int, maximum number of times each of the mesh's cell can be refined
        
        balanced: bool, true if mesh is balanced. 
    
    Methods:
    
    '''
    def __init__(self, box=[0.,1.,0.,1.], nx=2, ny=2):
        '''
        Description: Constructor, initialize rectangular grid
        
        Inputs: 
            
            box: double, boundary vertices of rectangular grid, box = [x_min, x_max, y_min, y_max]
            
            nx: int, number of cells in x-direction
            
            ny: int, number of cells in y-direction
            
            type: 'MESH'
            
        '''
        self.bounding_box = box
        self.type = 'MESH'
        self.children_array_size = (nx,ny)
        #
        # Define cells in mesh
        # 
        xmin, xmax, ymin, ymax = box
        x = numpy.linspace(xmin, xmax, nx+1)
        y = numpy.linspace(ymin, ymax, ny+1)
        
        mesh_cells = {}
        for i in range(nx):
            for j in range(ny):
                if i == 0 and j == 0:
                    v_sw = Vertex((x[i]  ,y[j]  ))
                    v_se = Vertex((x[i+1],y[j]  ))
                    v_ne = Vertex((x[i+1],y[j+1]))
                    v_nw = Vertex((x[i]  ,y[j+1]))
                elif i > 0 and j == 0:
                    v_se = Vertex((x[i+1],y[j]  ))
                    v_ne = Vertex((x[i+1],y[j+1]))
                    v_sw = mesh_cells[i-1,j].vertices['SE']
                    v_nw = mesh_cells[i-1,j].vertices['NE']                    
                elif i == 0 and j > 0:
                    v_ne = Vertex((x[i+1],y[j+1]))
                    v_nw = Vertex((x[i]  ,y[j+1]))
                    v_sw = mesh_cells[i,j-1].vertices['NW']
                    v_se = mesh_cells[i,j-1].vertices['NE']                    
                elif i > 0 and j > 0:
                    v_ne = Vertex((x[i+1],y[j+1]))
                    v_nw = mesh_cells[i-1,j].vertices['NE']
                    v_sw = mesh_cells[i,j-1].vertices['NW']
                    v_se = mesh_cells[i,j-1].vertices['NE']
                    
                cell_vertices = {'SW': v_sw, 'SE': v_se, 'NE': v_ne, 'NW': v_nw}                    
                cell_address = [i,j]
                mesh_cells[i,j] = Cell(cell_vertices, self, cell_address)
        
        self.children = mesh_cells
        self.vertex_list = []
        self.connectivity = None
        self.max_depth = 0
        self.__num_vertices = 0
        self.__num_cells = 0
        self.__balanced = False
        self.__triangles = []
        

    def leaves(self):
        """
        Description: Returns a list of all leaf sub-cells of the mesh
        
        Input: 
        
            group: string, optional sorting criterium (None, or 'depth') 
            
        Output:
        
            leaves: list of LEAF cells  
        """
        #
        # All leaves go in a long list
        # 
        leaves = []
        for child in self.children.itervalues():
            leaves.extend(child.find_leaves())
        self.__num_cells = len(leaves)        
        return leaves
    
    def triangles(self):
        """
        Returns a list of triangles 
        """
        if len(self.__triangles) == 0:
            #
            # Mesh has not been triangulated yet
            #  
            self.triangulate()
            return self.__triangles
        else:
            #
            # Mesh triangulated
            # 
            return self.__triangles
            
    def vertices(self):
        """
        Returns a list of vertices.
        
        POSSIBLE BUG: if vertex has been marked outside of 
            this function, it will not show up in the list.  
        """ 
        n_vertices = -1
        vertices = []
        for leaf in self.leaves():
            for v in leaf.vertices.itervalues():
                if not v.is_marked():
                    n_vertices += 1
                    vertices.append(v)
                    v.set_node_number(n_vertices)
                    #
                    # Mark vertices in the list
                    # 
                    v.mark()
        self.__num_vertices = n_vertices            
        #
        # Unmark all vertices again
        # 
        for v in vertices:
            v.unmark()
            
        
    def cells_at_depth(self, depth):
        """
        Return all cells at a given depth > 0
        """
        cells = []
        for child in self.children.itervalues():
            cells.extend(child.cells_at_depth(depth))  
        return cells
    
     
    def has_children(self):
        """
        Determine whether the mesh has children
        """
        return any(child != None for child in self.children.itervalues())
        
        
    def get_max_depth(self):
        """
        Determine the maximum depth of the mesh
        """
                    
    
    def unmark_all(self):
        """
        Unmark all cells in mesh
        """
        if self.has_children():
            for child in self.children.itervalues():
                child.unmark_all()
                
            
    def refine(self):
        """
        Refine mesh by splitting marked cells.            
        """
        leaves = self.leaves()
        for leaf in leaves:
            if leaf.flag:
                leaf.split()
                leaf.unmark()
        self.__balanced = False
        
    
    def coarsen(self):
        """
        Coarsen mesh by collapsing marked cells
        """
        leaves = self.leaves()
        for leaf in leaves:
            parent = leaf.parent
            if parent.flag:
                parent.children.clear()
        self.remove_supports()    
        self.__balanced = False
                
     
    def balance_tree(self):
        """
        Ensure the 2:1 rule holds 
        """
        leaves = self.leaves()
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
        self.__balanced = True
        
    
    def remove_supports(self):
        """
        Remove the supporting cells
        """    
        leaves = self.leaves()
        while len(leaves) > 0:
            leaf = leaves.pop()
            if leaf.support_cell:
                #
                # Check whether its safe to delete the support cell
                # 
                safe_to_coarsen = True
                for direction in ['N', 'S', 'E', 'W']:
                    nb = leaf.find_neighbor(direction)
                    if nb.has_children():
                        safe_to_coarsen = False
                        break
                if safe_to_coarsen:
                    parent = leaf.parent
                    for child in parent.children.itervalues():
                        #
                        # Delete cells individually
                        # 
                        del child
                    parent.children.clear()
                    leaves.append(parent)
        self.__balanced = False
        
     
    def triangulate(self):
        """
        Generate triangulation of mesh:
            balance if necessary
            populate cells with triangles
            generate connectivity matrix. 
            
        #TODO: unfinished
        """
        triangles = []
        if not self.__balanced:
            #
            # Balance mesh first
            # 
            self.balance_tree()
        for leaf in self.leaves():
            v = leaf.vertices
            #
            # Determine whether Steiner Point is necessary
            # 
            if any([v.has_key(direction) for direction in ['N','S','E','W']]):
                #
                # Add Steiner vertex
                # 
                x0, x1, y0, y1 = leaf.box()
                vm = Vertex((0.5*(x0 + x1), 0.5*(y0 + y1)))
                leaf.vertices['M'] = vm
                
                sub_edge_dict = {'S': ['SW','S','SE'], \
                                 'E': ['NE','E','SE'], \
                                 'N': ['NE','N','NW'], \
                                 'W': ['NW','W','SW']}
                for direction in ['S','E','N','W']:
                    se = sub_edge_dict[direction]
                    if v.has_key(direction):
                        #
                        # Midpoint on this edge
                        #
                        tri = [Triangle([v[se[0]],v[se[1]],vm],parent_cell=leaf), 
                               Triangle([v[se[1]],v[se[2]],vm],parent_cell=leaf)]
                    else:
                        #
                        # No midpoint
                        # 
                        tri = [Triangle([v[se[0]],v[se[2]],vm],parent_cell=leaf)]
                        
                    triangles.extend(tri)
            else:
                #
                # No Steiner vertex - simple triangulation
                # 
                tri = [Triangle([v['SW'],v['SE'],v['NE']], parent_cell=leaf), \
                       Triangle([v['NE'],v['NW'],v['SW']], parent_cell=leaf)]
                triangles.extend(tri)
        self.__triangles = triangles
                
               
    def build_connectivity(self):
        """
        Returns the connectivity matrix for the tree
        """
        # TODO: FIX build_connectivity
        
        econn = []
        num_vertices = len(self.vertex_list)
        
        #
        # Balance tree first
        # 
        #self.balance_tree()
        for leaf in self.leaves():
            add_steiner_pt = False
            #
            # Get global indices for each corner vertex
            # 
            gi = {}
            for pos in ['NW', 'SW', 'NE', 'SE']:
                gi[pos] = leaf.vertices[pos].node_number
            
            edges = {'S': [[gi['SW'], gi['SE']]], 'N': [[gi['NE'], gi['NW']]], 
                     'W': [[gi['NW'], gi['SW']]], 'E': [[gi['SE'], gi['NE']]] }
                     
            opposite_direction = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
            for direction in ['S', 'N', 'E', 'W']:
                neighbor = leaf.find_neighbor(direction)
                if neighbor != None and neighbor.type != 'LEAF':
                    # If neighbor has children, then add the midpoint to
                    # your list of vertices, update the list of edges and
                    # remember to add the Steiner point later on. 
                    #
                    od = opposite_direction[direction]
                    leaf.vertices[direction] = neighbor.vertices[od]
                    gi[direction] = leaf.vertices[direction].node_number
                    add_steiner_pt = True
                                        
                    edges[direction] = [[edges[direction][0][0], gi[direction]],
                                        [gi[direction], edges[direction][0][1]]]
        
            # 
            # Add the Triangles to connectivity
            # 
            if not add_steiner_pt:
                #
                # Simple Triangulation
                #
                econn.extend([[gi['SW'], gi['SE'], gi['NE']], 
                              [gi['NE'], gi['NW'], gi['SW']]] )
                              
            elif not leaf.vertices.has_key('M') or leaf.vertices['M'] == None:
                #
                # Add Steiner Vertex
                # 
                x0, x1, y0, y1 = leaf.box()
                vm = Vertex((0.5*(x0 + x1), 0.5*(y0 + y1)), node_number=num_vertices)
                leaf.vertices['M'] = vm
                gi['M'] = vm.node_number
                self.vertex_list.append(vm)
                num_vertices += 1
                for direction in ['N', 'S', 'E', 'W']:
                    for sub_edge in edges[direction]:
                        econn.append([sub_edge[0], sub_edge[1], gi['M']])                                    
        return econn
            
            
    def plot_quadmesh(self, ax, name=None, show=True, set_axis=True):
        '''
        Plot the current quadmesh
        '''
                   
        if self.has_children():
            if set_axis:
                x0, x1, y0, y1 = self.bounding_box          
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
    
    
    def plot_trimesh(self, ax):
        """
        Plot triangular mesh
        """
        e_conn = self.build_connectivity()
        for element in e_conn:
            points = []
            for node_num in element:
                x, y = self.vertex_list[node_num].coordinate
                points.append([x,y])
            triangle = plt.Polygon(points, fc='w', ec='k')
            ax.add_patch(triangle)
    