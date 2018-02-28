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
            x = [xi.coordinate() for xi in x]
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
        
    elif type(x) is np.ndarray:
        #
        # Points in numpy array
        #
        if len(x.shape)==1:
            #
            # x is a one-dimensional vector
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
                
    
class Mesh(object):
    """
    Mesh Class, consisting of a cell (background mesh), together with a tree, 
        from which a specific mesh instance can be constructed without deleting
        previous mesh parameters.
    
    Attributes:
    
    Methods:

    
    """
    def __init__(self, cell=None, node=None, grid=None, dim=None):
        """
        Constructor
        
        
        Inputs:
        
            cell: Cell object, a single root BiCell/QuadCell 
            
            node: Node object, a single root BiNode/QuadNode
            
            grid: Grid object, specifying grid associated with
                root node. 
             
            dim: int, dimension of the mesh   
            
            
        NOTE: Specify at most one of the above inputs 
        """
        if grid is not None:
            #
            # grid specified 
            # 
            assert all(i is None for i in [node, cell, dim]),\
            'Grid specified: All other inputs should be None.'
            
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
            # Node specified
            # 
            assert all(i is None for i in [cell, grid, dim]),\
            'Node specified: All other inputs should be None.'
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
                            boundary.add(np.array(v.coordinate()))
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
            # Grid on coarsest level
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
                        grid.points['coordinates'][i_vbnd[k]].coordinate())
                v_bnd = np.array(v_bnd) 
                x_min, x_max = v_bnd[:,0].min(), v_bnd[:,0].max()
                y_min, y_max = v_bnd[:,1].min(), v_bnd[:,1].max()
                return [x_min, x_max, y_min, y_max] 
        else:
            #
            # No Grid: Use Cell
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
            return np.array([v.coordinate() for v in quadvertex_list])
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
            vertices = self.quadvertices()
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
    '''
    
    
    
class Grid(object):
    """
    Description: Class used for storing Nodes on coarsest refinement level
    
    Attributes:
    
            
        __dim: int, dimension of grid
    
        format: str, version of mesh file
        
        subregions: struct, encoding the mesh's subregions, with fields:
        
            n: int, number of subregions
            
            dim: int, dimension of subregion
            
            tags: int, tags of subregions
            
            names: str, names of subregions
         
        node: Node, root node associated with grid
        
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
        
            coordinates: double, list of Vertices
            
                
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
            
            position: str, 'N', 'W', 'S', 'E'
            
            
        faces: struct, encoding the mesh's faces w. fields:
        
            n: int, number of faces
            
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
        
        doubly_connected_edge_list
        
        dim
        
        get_neighbor
        
        contains_node
        
    Note: The grid can be used to describe the connectivity associated with a
        ROOT Node. 
    
    """
    def __init__(self, box=None, resolution=None, dim=None,
                 file_path=None, file_format='gmsh'):
        """
        Constructor
        
        Inputs:
        
            box: list of endpoints for rectangular mesh
                1d: [x_min, x_max]
                2d: [x_min, x_max, y_min, y_max]  
            
            resolution: tuple, with number of cells in each direction 
                
            file_path: str, path to mesh file
            
            file_format: str, type of mesh file (currently only gmsh)
        """
        #
        # Initialize struct
        #     
        self.is_rectangular = False
        self.resolution = resolution
        self.initialize_grid_structure() 
        if file_path is None:
            #
            # Rectangular Grid
            #
            # Determine dimension
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
                
            if box is None:
                #
                # Default boundary box
                # 
                if dim==1:
                    box = [0,1]
                elif dim==2:
                    box = [0,1,0,1]
            if resolution is None:
                #
                # Default resolution
                # 
                if dim==1:
                    resolution = (1,)
                elif dim==2:
                    resolution = (1,1)
            self.__dim = dim          
            self.is_rectangular = True
            self.regular_grid(box=box, resolution=resolution)
        else:
            #
            # Import grid from gmsh
            # 
            assert file_format=='gmsh', \
            'For input file_format, use "gmsh".'
            #
            # Import grid from gmsh
            # 
            self.grid_from_gmsh(file_path)
        
        
        if self.dim() == 1:
            #
            # Add BiNodes 
            # 
            pass
            
        elif self.dim() == 2:
            #
            # Generate doubly connected edge list 
            # 
            self.doubly_connected_edge_list()
            #
            # Add Edges
            # 
            self.edges['Edges'] = []
            for i in range(self.edges['n']):
                i_v1, i_v2 = self.edges['connectivity'][i]
                v1, v2 = self.points['coordinates'][i_v1], \
                         self.points['coordinates'][i_v2]
                self.edges['Edges'].append(Edge(v1, v2))
               
        else:
            raise Exception('Only dimensions 1 and 2 supported.')
        
        #
        # Initialize Nodes, and Cells
        # 
        self.faces['Cell'] = [None]*self.faces['n']       
               
                        
    def initialize_grid_structure(self):
        """
        Initialize empty grid. 
        """
        self.format = None
         
        # Subregions 
        self.subregions = {'dim': [], 'n': None, 'names': [], 'tags': []}
        
        # Points
        self.points = {'half_edge': [], 'n': None, 'tags': {}, 'n_dofs': None, 
                       'coordinates': [], 'connectivity': []}
        
        # Edges 
        self.edges = {'n': None, 'tags': {}, 'n_dofs': None, 'connectivity': []}
        
        
        # Half-Edges
        self.half_edges = {'n': None, 'tags': {}, 'n_dofs': None, 
                           'connectivity': [], 'prev': [], 'next': [],
                           'twin': [], 'edge': [], 'face': [], 'position': []}
        
        # Faces
        self.faces = {'n': None, 'type': [], 'tags': {}, 'n_dofs': None, 
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
            #
            # One dimensional grid
            # 
            
            # Generate Grid
            x_min, x_max = box
            n_points = resolution[0] + 1 
            x = np.linspace(x_min, x_max, n_points)
            
            # Store grid information
            self.__dim = 1
            self.points['coordinates'] = [Vertex(xi) for xi in x]
            self.points['n'] = n_points
            self.faces['connectivity'] = [[i,i+1] for i in range(n_points-1)]
            self.faces['n'] = len(self.faces['connectivity'])
            self.faces['type'] = ['interval']*self.faces['n']
            
        elif dim  == 2:
            #
            # Two dimensional grid
            #
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
                    self.points['coordinates'].append(Vertex((x[i_x],y[i_y])))
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
                        
                # 
                # Cell Vertices
                #     
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
                        vtx = Vertex((float(words[1]),float(words[2])))
                        points['coordinates'].append(vtx)
                        line = infile.readline()
                
                # --------------------------------------------------------------
                #  Faces
                # --------------------------------------------------------------        
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
                                      
                            
                            quadrilateral: 3   (4 node first order quadrilateral)
                                          10   (9 node second order quadrilateral)
                                         
                    
                              
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
            
    
    def doubly_connected_edge_list(self):
        """
        Returns a doubly connected edge list.
                
        """
        #
        # Update Point Fields
        # 
        n_points = self.points['n']
        self.points['half_edge'] = np.full((n_points,), -1, dtype=np.int)
        
        #
        # Initialize Half-Edges
        # 
        if all(f_type == 'triangle' for f_type in self.faces['type']):
            n_sides = 3
        elif all(f_type == 'quadrilateral' for f_type in self.faces['type']):
            n_sides = 4
        
        n_he = n_sides*self.faces['n']
        
        self.half_edges['n'] = n_he
        self.half_edges['connectivity'] = np.full((n_he,2), -1, dtype=np.int)
        self.half_edges['prev'] = np.full((n_he,), -1, dtype=np.int)
        self.half_edges['next'] = np.full((n_he,), -1, dtype=np.int)
        self.half_edges['twin'] = np.full((n_he,), -1, dtype=np.int)
        self.half_edges['edge'] = np.full((n_he,), -1, dtype=np.int)
        self.half_edges['face'] = np.full((n_he,), -1, dtype=np.int)
        
        #
        # Update Edge Fields 
        #
        edge_set = set(self.edges['connectivity']) 
        self.edges['half_edge'] = [None]*len(edge_set)
        
        #
        # Update Face Fields 
        # 
        n_faces = self.faces['n']
        self.faces['half_edge'] = np.full((n_faces,), -1, dtype=np.int)
        
        #
        # Initialize 
        # 
        half_edge_count = 0
        for i_fce in range(self.faces['n']):
            fc = self.faces['connectivity'][i_fce]
            #
            # Face's half-edge numbers
            # 
            fhe = [half_edge_count + j for j in range(n_sides)]
            
            #
            # Update face information 
            # 
            self.faces['half_edge'][i_fce] = fhe[0]
            
            for i in range(n_sides):
                # .............................................................
                # Update half-edge information
                # .............................................................
                # Connectivity
                #
                hec = [fc[i%n_sides], fc[(i+1)%n_sides]]
                if fhe[i] >= n_he:
                    print('Half-edge index exceeds matrix dimensions.')
                    print('Number of faces: {0}'.format(self.faces['n']))
                    print('Number of half-edges: 3x#faces = {0}'.format(3*self.faces['n']))
                    print('#Half-Edges recorded: {0}'.format(self.half_edges['n']))
                self.half_edges['connectivity'][fhe[i],:] = hec
                    
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
                
                # -------------------------------------------------------------
                # Update point information
                # -------------------------------------------------------------                  
                self.points['half_edge'][fc[i%n_sides]] = fhe[i]
                
                # -------------------------------------------------------------
                # Update edge information
                # -------------------------------------------------------------
                #print('Size of edge_set: {0}'.format(len(edge_set)))
                #print('Size of edge connectivity: {0}'.format(len(self.edges['connectivity'])))
                if set(hec) in edge_set:
                    #print('Set {0} is in edge_set. Locating it'.format(hec))
                    #
                    # Edge associated with Half-Edge exists
                    # 
                    i_edge = self.edges['connectivity'].index(set(hec))
                    #print('Location: {0}'.format(i_edge))
                    #print('Here it is: {0}'.format(self.edges['connectivity'][i_edge]))
                    
                    # Link edge to half edge  
                    #print('Linking half edge with edge:')  
                    #print('Half-edge: {0}'.format(self.edges['connectivity'][i_edge]))
                    #print('Edge: {0}'.format(self.half_edges['connectivity'][fhe[i]]))
                    #print(len(self.edges['half_edge']))
                    #print('Length of edge_set {0}'.format(len(edge_set)))
                    #print(edge_set)
                    self.edges['half_edge'][i_edge] = fhe[i]
                    
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
                
                    # Link edge to half-edge
                    self.edges['half_edge'].append(fhe[i])
                    
                    
                #
                # Link half-edge to edge
                #     
                self.half_edges['edge'][fhe[i]] = i_edge    
                
            #
            # Update half-edge count
            # 
            half_edge_count += n_sides
        #
        # Update size of edge list       
        #
        self.edges['n'] = len(self.edges['connectivity'])
        # ---------------------------------------------------------------------
        # Find twin 1/2 edges
        # ---------------------------------------------------------------------
        hec = self.half_edges['connectivity']
        for i in range(n_he):
            #
            # Find the row whose reversed entries match current entry
            # 
            row = np.argwhere( (hec[:,0]==hec[i,1]) & (hec[:,1]==hec[i,0]) )
            if len(row) == 1:
                #
                # Update twin field
                #
                self.half_edges['twin'][i] = int(row)
        
        # ---------------------------------------------------------------------
        # Assign Directions to half-edges (only for quadrilaterals)
        # ---------------------------------------------------------------------
        # TODO: This is folly, it won't work.
        first_half_edge = True
        directions = ['S','E','N','W']
        opposite  = {'S':'N', 'N':'S', 'W':'E', 'E':'W'}
        he_dirs = [None]*n_he
        faces_visited = set()
        faces_to_do = deque([0])
        if n_sides == 4:
            #
            # Ensure grid consists of quadrilaterals
            #
            while len(faces_to_do) > 0:
                #
                # Assign directions to all half-edges in current face
                #
                i_f  = faces_to_do.popleft() 
                i_he = self.faces['half_edge'][i_f]
                if first_half_edge:
                    #
                    # Assign 'S' to first half edge
                    #
                    he_dirs[i_he] = 'S'
                    cdir = 'S'
                    first_half_edge = False
                else:
                    #
                    # Look for half-edge with an assigned direction
                    #
                    direction_assigned = False
                    while not direction_assigned: 
                        if he_dirs[i_he] is None:
                            i_he = self.half_edges['next'][i_he]
                        else:
                            cdir = he_dirs[i_he]
                            direction_assigned = True
                #
                # Assign directions to other he's if necessary
                # 
                i_cdir = directions.index(cdir) 
                for i in np.arange(1,4):
                    i_he = self.half_edges['next'][i_he]
                    if he_dirs[i_he] is None:
                        he_dirs[i_he] = directions[(i_cdir+i)%4]
                 
                faces_visited.add(i_f)
                
                #
                # Assign directions to twin half-edges and add neighbor to "todo"
                # 
                for i in range(4):
                    i_the = self.half_edges['twin'][i_he]
                    if i_the != -1 and \
                    self.half_edges['face'][i_the] not in faces_visited:
                        #
                        # Add neighbor to list
                        # 
                        faces_to_do.append(self.half_edges['face'][i_the])
                        #
                        # Assign opposite direction to twin half_edge
                        # 
                        if he_dirs[i_the] is None:
                            he_dirs[i_the] = opposite[he_dirs[i_he]]
                        else:
                            assert he_dirs[i_the] == opposite[he_dirs[i_he]],\
                            'Twin half edge should have opposite direction.'
                    i_he = self.half_edges['next'][i_he]
            self.half_edges['position'] = he_dirs
                 
            
    def dim(self):
        """
        Returns the underlying dimension of the grid
        """ 
        return self.__dim
    
    
    def get_neighbor(self, i_f, direction):
        """
        Returns the neighbor of a Node in the Grid
        
        Inputs: 
        
            i_f: int, face index
            
            direction: str, ['L','R'] for a 1D grid or 
                ['N','S','E','W'] (or combinations) for a 2D grid
                
        """
        if self.dim() == 1:
            #
            # One dimensional mesh (assume faces are ordered)
            # 
            assert direction in ['L', 'R'], 'Direction not recognized.'
            if direction == 'L':
                if i_f - 1 >= 0:
                    return i_f-1
                else:
                    return None
            elif direction == 'R':
                if i_f + 1 <= self.faces['n']:
                    return i_f+1
                else:
                    return None
        elif self.dim() == 2:
            #
            # Two dimensional mesh
            #
            assert direction in ['N', 'S', 'E', 'W', 'NE', 'SE', 'NW', 'SW'],\
                'Direction not recognized.'
            if len(direction)==2:
                #
                # Composite direction
                #     
                direction_1, direction_2 = direction
                i_nf = self.get_neighbor(i_f, direction_1)
                if i_nf is not None:
                    return self.get_neighbor(i_nf, direction_2)
                else:
                    i_nf = self.get_neighbor(i_f, direction_2)
                    if i_nf is not None:
                        return self.get_neighbor(i_nf, direction_1)
                    else: 
                        #
                        # There is still a possibility that the diagonal
                        # neighbor exists, although it is not reachable 
                        # via edges (only via the vertex). 
                        # 
                        return None
            else:
                #
                # Find half edge in given direction
                # 
                i_he = self.faces['half_edge'][i_f]
                for _ in range(4):
                    if self.half_edges['position'][i_he] == direction:
                        break
                    i_he = self.half_edges['next'][i_he]
                i_the = self.half_edges['twin'][i_he]
                if i_the != -1: 
                    #
                    # Neighbor exists, return it
                    #  
                    return self.half_edges['face'][i_the]
                else:
                    #
                    # No neighbor exists, return None
                    # 
                    return None
    
    
    def contains_node(self, node):
        """
        Determine whether a given Node is contained in the grid
        
        Inputs:
        
            Node: Node, 
        """
        return node in self.faces['Nodes']
    
    
    def get_boundary_edges(self):
        """
        Returns a list of the boundary edge indices
        """
        if self.dim == 1:
            raise Exception('Boundary edges only present in 2D grids.')
        
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
        #
        # Extract boundary edges
        # 
        bnd_edges = [self.half_edges['edge'][i] for i in bnd_hes_sorted]
        return bnd_edges
    
    
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
            for i_e in self.get_boundary_edges():
                i_he = self.edges['half_edge'][i_e]
                #
                # Add initial point of each boundary half edge
                # 
                bnd_points.append(self.half_edges['connectivity'][i_he][0])
        else: 
            raise Exception('Only dimensions 1 and 2 supported.')
        return bnd_points
    
    
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
    def __init__(self):
        """
        Constructor
        """            
        #
        # Record Attributes
        # 
        self._flags  = set()
        self._support = False
            
    '''  
    TODO: If we really need it, move it to subclass      
    def copy(self, position=None, parent=None):
        """
        Copy existing Node without attached cell or parental node
        
        TODO: Unnecessary? 
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
            for child in self.get_children():
                if child is None:
                    node_copy.children[position] = \
                        child.copy(position=child.position, parent=node_copy) 
        return node_copy
    '''        
        
    def grid_size(self):
        """
        Return the grid size of root node
        """
        assert self.type == 'ROOT', \
        'Only ROOT nodes have a grid.'
        
        return self.grid.faces['n']
        
    
        
    def dim(self):
        """
        Return cell dimension
        """
        return self._dim
    
    '''    
    def get_neighbor(self, direction):
        """
        Description: Returns the deepest neighboring cell, whose depth is at 
            most that of the given cell, or 'None' if there aren't any 
            neighbors.
         
        Inputs: 
         
            direction: char, 'N'(north), 'S'(south), 'E'(east), or 'W'(west)
             
        Output: 
         
            neighboring cell
         
        TODO: DELETE  
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
                    mu = self.parent.get_neighbor(here)
                    if mu is not None and mu.depth == self.depth-1 and mu.has_children():
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
                            neighbor_pos = exterior_neighbors_dict[self.position]
                            return mu.children[neighbor_pos] 
    '''
    
    def tree_depth(self, flag=None):
        """
        Return the maximum depth of sub-nodes 
        """
        depth = self.depth
        if self.has_children():
            for child in self.get_children(flag=flag):
                d = child.tree_depth()
                if d > depth:
                    depth = d 
        return depth
             
    
    def traverse(self, flag=None, mode='depth-first'):
        """
        Iterator: Return current node and all its flagged sub-nodes         
        
        Inputs: 
        
            flag [None]: node flag
            
            mode: str, type of traversal 
                'depth-first' [default]: Each cell's progeny is visited before 
                    proceeding to next cell.
                 
                'breadth-first': All cells at a given depth are returned before
                    proceeding to the next level.
        
        Output:
        
            all_nodes: list, of all nodes in tree (marked with flag).
            
            
        TODO: Delete other traverse methods
        """
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
                    if child is not None:
                        queue.append(child)
            if flag is not None: 
                if node.is_marked(flag):
                    yield node
            else:
                yield node    
                
    '''            
    def traverse_tree(self, flag=None):
        """
        Return list of current node and ALL of its sub-nodes         
        
        Inputs: 
        
            flag [None]: node flag
        
        Output:
        
            all_nodes: list, of all nodes in tree (marked with flag).
        
        Note:
        
            Each node's progeny is visited before proceeding to next node 
            (compare traverse depthwise). 
            
        TODO: DELETE
        """
        all_nodes = []
        #
        # Add self to list
        #
        if flag is not None:
            if self.is_marked(flag):
                all_nodes.append(self)
        else:
            all_nodes.append(self)
            
        #
        # Add (flagged) children to list
        # 
        if self.has_children():
            for child in self.get_children():
                all_nodes.extend(child.traverse_tree(flag=flag))
                 
        return all_nodes
    
    
    def traverse_depthwise(self, flag=None):
        """
        Iterate node and all sub-nodes, ordered by depth
        
        TODO: DELETE
        """
        queue = deque([self]) 
        while len(queue) != 0:
            node = queue.popleft()
            if node.has_children():
                for child in node.get_children():
                    if child is not None:
                        queue.append(child)
            if flag is not None:
                if node.is_marked(flag):
                    yield node
            else:
                yield node
    '''    
    
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
                              

            """
            if flag is None:
                #
                # No flag specified
                #
                if not self.has_children():
                    leaves.append(self)
                else:
                    for child in self.get_children():
                        leaves.extend(child.get_leaves(flag=flag))
            else:
                #
                # Flag specified
                # 
                if not any([child.is_marked(flag) for child in self.get_children()]):
                    if self.is_marked(flag=flag):
                        leaves.append(self)
                else:
                    for child in self.get_children():
                        leaves.extend(child.get_leaves(flag=flag))
            return leaves
            """
        
    '''    
    def get_leaves(self, flag=None):
        """
        Return all LEAF sub-nodes of current node
        
        Inputs:
        
            flag: If flag is specified, return all leaf nodes within labeled
                submesh (or an empty list if there are none).
        """
        leaves = []    
        if self.type == 'LEAF' or not(self.has_children()):
            # 
            # LEAF or childless ROOT
            # 
            if flag is not None:
                #
                # Extra condition imposed by flag
                # 
                if self.is_marked(flag):
                    leaves.append(self)
            else:
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
                            leaves.extend(child.get_leaves(flag=flag))
                else:
                    #
                    # Usual quadcell division: traverse in bottom-to-top mirror Z order
                    #
                    for child in self.get_children():
                        if child != None:
                            leaves.extend(child.get_leaves(flag=flag))
                    
        return leaves
    '''
    
    
    def get_root(self):
        """
        Return root node
        """
        if self.type == 'ROOT':
            return self
        else:
            return self.parent.get_root()
    
    
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
                if node.grid is not None:
                    pos = a
                else:
                    pos = self.id2pos(a)
                node = node.children[pos]
        return node
        
    '''            
    def has_children(self, position=None, flag=None):
        """
        Determine whether node has children
        
        TODO: Replace this version in future
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
    
    '''
    def has_children(self, position=None, flag=None):
        """
        Determine whether node has children
        
        TODO: Use this version in future
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
            pos_error = 'Position should be one of: %s' %self._child_positions
            assert position in self._child_positions, pos_error
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
    def get_children(self, flag=None):
        """
        Returns a list of (flagged) children, ordered 
        
        Inputs: 
        
            flag: [None], optional marker
        
        Note: Only returns children that are not None 
        
        TODO: Replace with version below
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
            # Usual quadcell division: traverse in bottom-to-top mirror Z order
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
    def get_children(self, flag=None, reverse=False):
        """
        Returns (flagged) children, ordered 
        
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
                for pos in self._child_positions:
                    child = self.children[pos]
                    if child is not None:
                        if flag is None:
                            yield child
                        elif child.is_marked(flag):
                            yield child
            else: 
                #
                # Go in reverse order
                # 
                for pos in reversed(self._child_positions):
                    child = self.children[pos]
                    if child is not None:
                        if flag is None:
                            yield child
                        elif child.is_marked(flag):
                            yield child
    
        
    def has_parent(self, flag=None):
        """
        Determine whether node has parents (with a given flag)
        """
        if flag is None:
            return self.type != 'ROOT'
        else:
            if self.type != 'ROOT':
                parent = self.parent
                if parent.is_marked(flag):
                    return True
                else:
                    return parent.has_parent(flag=flag)
            else:
                return False 
    
    
    def get_parent(self, flag=None):
        """
        Return node's parent, or first ancestor with given flag (None if there
        are none).
        """
        if flag is None:
            if self.has_parent():
                return self.parent
        else:
            if self.has_parent(flag):
                parent = self.parent
                if parent.is_marked(flag):
                    return parent
                else:
                    return parent.get_parent(flag=flag)
        
    
    def in_grid(self):
        """
        Determine whether node position is given by coordinates or directions        
        """
        if self.parent is None:
            return False
        elif self.parent.grid is not None:
            return True
        else:
            return False
            
    
    def mark(self, flag=None, recursive=False):
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
            self._flags.add(True)
        else:
            self._flags.add(flag)
        
        #
        # Mark children as well
        # 
        if recursive and self.has_children():
            for child in self.get_children():
                child.mark(flag, recursive=recursive)
                
    
    def unmark(self, flag=None, recursive=False):
        """
        Unmark node (and sub-nodes)
        
        Inputs: 
        
            flag: 
        
            recursive (False): boolean, unmark all progeny
            
        """
        # Remove tag
        if flag is None:
            self._flags.clear()
        else:
            self._flags.remove(flag)
        # Remove tag from children
        if recursive and self.has_children():
            for child in self.children.values():
                child.unmark(flag=flag, recursive=recursive)
     
    
    def is_marked(self, flag=None):
        """
        Check whether a node is marked.
        
        Input: 
        
            flag: str/int/double
        """
        if flag is None:
            # No flag specified check whether there is any mark
            if self._flags:
                return True
            else:
                return False 
        else:
            # Check for the presence of given flag
            return flag in self._flags           
    
    
    def is_linked(self):
        """
        Determine whether node is linked to a cell
        """
        return not self._cell is None
    
        
    def link(self, cell, recursive=True):
        """
        Link node with Cell
        
        Inputs: 
        
            Quadcell: Cell object to be linked to node
            
            recursive: bool, if True - link entire tree with cells
            
        """
        self._cell = cell
        if recursive:
            #
            # Link child nodes to appropriate child cells
            #
            assert self.children.keys() == cell.children.keys(), \
            'Keys of tree and cell incompatible.'
            
            if self.has_children():
                if not(cell.has_children()):
                    #
                    # Cell must be split first
                    #
                    cell.split()
             
                for pos in self._child_positions:
                    tree_child = self.children[pos]
                    if tree_child.cell is None:
                        cell_child = cell.children[pos]
                        tree_child.link(cell_child,recursive=recursive) 
    
        
    def unlink(self, recursive=True):
        """
        Unlink node from cell
        """
        self._cell = None
        if recursive and self.has_children():
            #
            # Unlink child nodes from cells
            # 
            for child in self.children.values():
                if child is not None:
                    child.unlink()
        
    
    def cell(self):
        """
        Return associated cell
        """
        return self._cell
       
    '''
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
    '''
    
    def merge(self):
        """
        Delete all sub-nodes of given node
        """
        for key in self.children.keys():
            self.children[key] = None
        if self.type == 'BRANCH':
            self.type = 'LEAF'
    
    
    def remove(self):
        """
        Remove node from parent's list of children
        """
        assert self.type != 'ROOT', 'Cannot delete ROOT node.'
        self.parent.children[self.position] = None
            
                    
    
                
    '''            
    def pos2id(self, pos):
        """ 
        Convert position to index: 'SW' -> 0, 'SE' -> 1, 'NW' -> 2, 'NE' -> 3 
        
        TODO: move to subclass
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
        
        TODO: move to subclass
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
    '''


class BiNode(Node):
    """
    Binary tree Node
    
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
                ['L','R'] if parent = Node
                None if parent = None
                i if parent is a ROOT node with specified grid_size
                
            grid: Grid object, used to store children of ROOT node
                
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
                assert isinstance(grid, Grid),\
                'Input "grid" should be a Grid object.'
                
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
        if bicell is not None:
            assert isinstance(bicell, BiCell), 'Cell must be in BiCell class'
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
        print('{0:10}: {1}'.format('Flags', self._flags))
        if self.has_children():
            if self.type == 'ROOT' and self.grid_size() is not None:
                str_row = ''
                nx = self.grid_size()
                for ix in range(nx):
                    str_row += repr(ix) + ': ' 
                    if self.children[ix] is not None:
                        str_row += '1,  '
                    else:
                        str_row += '0,  '
                print('{0:10}: {1}'.format('Children', str_row))
                
            else:
                child_string = ''
                for key in ['L','R']:
                    child = self.children[key]
                    if child is not None:
                        child_string += key + ': 1,  '
                    else:
                        child_string += key + ': 0,  '
                print('{0:10}: {1}'.format('Children',child_string))
        else:
            child_string = 'None'
            print('{0:10}: {1}'.format('Children',child_string))
    
    
    def pos2id(self, position):
        """
        Convert 'L' and 'R' to 0 and 1 respectively
        
        Input:
        
            position: str, 'L', or 'R'
            
        Output:
        
            position: int, 0 (for 'L') or 1 (for 'R')
        """
        if type(position) is int:
            if position in [0,1]:
                return position
            
            root = self.get_root()
            if root.grid is not None:
                assert position < root.grid.faces['n'], \
                'Input "position" incompatible with number of faces.'
                return position 
        else:
            assert position in ['L','R'], \
            'Position is %s. Use "R" or "L" for position' % position
            if position == 'L':
                return 0
            else:
                return 1
            
    
    def id2pos(self, idx):
        """
        Convert index to position: 0 -> 'L', 1 -> 'R'
        """
        if idx in ['L','R']:
            #
            # Input is already a position
            # 
            return idx
        elif type(idx) is int:
            if idx in [0,1]: 
                #
                # Convert to suitable direction
                # 
                id_to_pos = {0: 'SW', 1: 'SE', 2: 'NW', 3: 'NE'}
                return id_to_pos[idx]
            else:
                #
                # 
                # 
                root = self.get_root()
                assert idx < root.faces['n'], \
                'Input "idx" exceeds total number of faces.'
                
                return idx
        else:
            raise Exception('Unrecognized format.')
    
    
    def split(self):
        """
        Add new child nodes to current node
        """
        #
        # If node is linked to cell, split cell and attach children
        #
        assert not(self.has_children()),'Node already has children.' 
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
                
    
class QuadNode(Node):
    """
    Quadtree Node
    """
    def __init__(self, parent=None, position=None, \
                 grid=None, quadcell=None):
        """
        Constructor
        
        Inputs:
                    
            parent: Node, parental node
            
            position: position within parent 
                ['SW','SE','NE','NW'] if parent = Node
                None if parent = None
                [i,j] if parent is a ROOT node with specified grid_size
                
            grid: Grid object, specifying ROOT node's children
                
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
                assert isinstance(grid, Grid), \
                    'Input "grid" should be a Grid object.'
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
        print('Node Info')
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


    def pos2id(self, pos):
        """ 
        Convert position to index: 'SW' -> 0, 'SE' -> 1, 'NW' -> 2, 'NE' -> 3 
        
        Input: 
        
            pos: str, 'SW', 'SE', 'NW', or 'NE' or integer 
            
        """
        if type(pos) is int:
            #
            # Position already specified as an integer
            #
            if  0 <= pos and pos <= 3:
                #
                # Position between 0 and 3 -> don't ask questions
                # 
                return pos
            else:
                #
                # Position exceeds 3, check wether it is consistent with
                # grid size.  
                # 
                root = self.get_root()
                assert root.grid is not None, 'Position index exceeds 3.'
                assert pos >= 0 and pos < root.grid.faces['n'],\
                    'Position exceeds number of grid faces.' 
                return pos
            
        elif pos in ['SW','SE','NW','NE']:
            #
            # Position a valid string
            #
            pos_to_id = {'SW': 0, 'SE': 1, 'NW': 2, 'NE': 3}
            return pos_to_id[pos]
        
        else:
            raise Exception('Unidentified format for position.')
         
         
    
    def id2pos(self, idx):
        """
        Convert index to position: 0 -> 'SW', 1 -> 'SE', 2 -> 'NW', 3 -> 'NE'
        
        Input:
        
            idx: int, node index
            
        Note: It is impossible to tell without context whether idx in [0,1,2,3]
            should be converted or returned.
        """
        if type(idx) is int:
            #
            # Position already specified as an integer
            #
            if idx in [0,1,2,3]:
                #
                # Position is valid integer: convert
                # 
                idx_to_pos = {0: 'SW', 1: 'SE', 2: 'NW', 3: 'NE'}
                return idx_to_pos[idx]
            else:
                #
                # Position exceeds 3, check wether it is consistent with
                # grid size.  
                # 
                root = self.get_root()
                assert root.grid is not None, 'Position index exceeds 3.'
                assert idx in list(range(root.grid.faces['n'])),\
                    'Position exceeds number of grid faces.' 
                return idx
            
        elif idx in ['SW','SE','NW','NE']:
            #
            # Position a valid string
            #
            return idx
        
        else:
            raise Exception('Unidentified format for index.')
                   
                   
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
        
            
class Cell(object):
    """
    Cell object
    """
    def __init__(self):
        """
        Almost Empty Constructor
        """
        self._flags = set()
        self._vertex_positions = []
        self._corner_vertex_positions = []
            
    
    def dim(self):
        """
        Return cell dimension
        """
        return self._dim
    
        
    def tree_depth(self, flag=None):
        """
        Return the maximum depth of sub-nodes 
        """
        depth = self.depth
        if self.has_children():
            for child in self.get_children(flag=flag):
                d = child.tree_depth()
                if d > depth:
                    depth = d 
        return depth
                      
            
    def traverse(self, flag=None, mode='depth-first'):
        """
        Iterator: Return current cell and all its flagged sub-cells         
        
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
        queue = deque([self])
        while len(queue) != 0:
            if mode == 'depth-first':
                cell = queue.pop()
            elif mode == 'breadth-first':
                cell = queue.popleft()
            else:
                raise Exception('Input "mode" must be "depth-first"'+\
                                ' or "breadth-first".')
            if cell.has_children():
                reverse = True if mode=='depth-first' else False    
                for child in cell.get_children(reverse=reverse):
                    if child is not None:
                        queue.append(child)
            
            if flag is not None: 
                if cell.is_marked(flag):
                    yield cell
            else:
                yield cell             
                 
                
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
            for cell in self.traverse(flag=flag, mode='breadth-first'):
                if not cell.has_children(flag=flag):
                    leaves.append(cell)
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

    
    def get_root(self):
        """
        Find the ROOT cell for a given cell
        """
        if self.type == 'ROOT' or self.type == 'MESH':
            return self
        else:
            return self.parent.get_root()
        
        
    
    def find_cell(self, address):
        """
        Locate node by its address
        """
        cell = self.get_root()
        if address != []:
            #
            # Not the ROOT node
            # 
            for a in address:
                if cell.grid is not None:
                    pos = a
                else:
                    pos = self.id2pos(a)
                cell = cell.children[pos]
        return cell  
    
    
    
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
            pos_error = 'Position should be one of: %s' %self._child_positions
            assert position in self._child_positions, pos_error
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
    
    
    def get_children(self, flag=None, reverse=False):
        """
        Returns (flagged) children, ordered 
        
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
                for pos in self._child_positions:
                    child = self.children[pos]
                    if child is not None:
                        if flag is None:
                            yield child
                        elif child.is_marked(flag):
                            yield child
            else: 
                #
                # Go in reverse order
                # 
                for pos in reversed(self._child_positions):
                    child = self.children[pos]
                    if child is not None:
                        if flag is None:
                            yield child
                        elif child.is_marked(flag):
                            yield child
                            
    
    def has_parent(self):
        """
        Returns True if cell has a parent cell, False otherwise
        """
        return self.parent is not None
    
    
    def get_parent(self, flag=None):
        """
        Return cell's parent, or first ancestor with given flag (None if there
        are none).
        """
        if flag is None:
            if self.has_parent():
                return self.parent
        else:
            if self.has_parent(flag):
                parent = self.parent
                if parent.is_marked(flag):
                    return parent
                else:
                    return parent.get_parent(flag=flag)
        
            
    def in_grid(self):
        """
        Determine whether a (ROOT)cell lies within a grid        
        """
        return self.__in_grid 
    

    def mark(self, flag=None):
        """
        Mark Cell
        
        Inputs:
        
            flag: int, optional label used to mark cell
        """  
        if flag is None:
            self._flags.add(True)
        else:
            self._flags.add(flag)
      
    
    def unmark(self, flag=None, recursive=False):
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
        if recursive and self.has_children():
            for child in self.children.values():
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
    
    
    def remove(self):
        """
        Remove cell from parent's list of children
        """
        assert self.type != 'ROOT', 'Cannot delete ROOT node.'
        self.parent.children[self.position] = None    
    

    def merge(self):
        """
        Delete all sub-nodes of given node
        """
        for key in self.children.keys():
            self.children[key] = None
        if self.type == 'BRANCH':
            self.type = 'LEAF'

    
    def get_vertices(self, pos=None, as_array=True):
        """
        Returns the vertices of the current cell. 
        
        Inputs:
        
            pos: str, position of vertex within the cell: 
                SW, S, SE, E, NE, N, NW, or W. 
                If pos='corners', return vertices ['L','R'] 1d, 
                    or ['SW', 'SE', 'NE', 'NW'] 2d
                If pos is not specified, return all vertices.
                
            as_array: bool, if True, return vertices as a numpy array.
                Otherwise return a list of Vertex objects. 
             
                
        Outputs: 
        
            vertices: 
                    
        """            
        single_vertex = False
        if pos is None: 
            #
            # Return all vertices
            # 
            vertices = [self.vertices[p] for p in \
                        self._vertex_positions]
        elif pos=='corners':
            #
            # Return corner vertices
            # 
            vertices = [self.vertices[p] for p in \
                        self._corner_vertex_positions]
        elif type(pos) is list:
            #
            # Positions specified in list
            #
            for p in pos:
                assert p in self._vertex_positions, \
                'Valid inputs for pos are None, or %s' % self._vertex_positions
            vertices = [self.vertices[p] for p in pos]
        elif type(pos) is str and pos !='corners':
            #
            # Single position specified
            # 
            assert pos in self._vertex_positions, \
            'Valid inputs for pos are None, or %s' % self._vertex_positions
            single_vertex = True
            vertices = [self.vertices[pos]]
            
        if as_array:
            #
            # Convert to array
            #  
            v = [vertex.coordinate() for vertex in vertices]
            if single_vertex:
                return np.array(v[0])
            else:
                return np.array(v)
        else:
            #
            # Return as list of Vertex objects
            #
            if single_vertex:
                return vertices[0]
            else:
                return vertices
           
           
class BiCell(Cell):
    """
    Binary tree of sub-intervals in a 1d mesh
    
    Attributes:
    
        type: str, specifies cell's position in the binary tree
        
            ROOT - cell on coarsest level
            BRANCH - cell has a parent as well as children
            LEAF - cell on finest refinement level
            
        parent: BiCell, of which current cell is a child
        
        children: dict, of sub-cells of current cell
        
        flags: set, of flags (numbers/strings) associated with cell
        
        position: str, position within parent 'L' (left) or 'R' (right)
        
        address: list, specifying address within tree
        
        vertices: double, dictionary of left and right vertices
    
    
    Methods:
    
    
    Notes: 
    
        There are many similarities between BiCells (1d) and Edges (2d) 
        
        Once we're done with this, we have to modify 'Node'
    """
    
    
    
    
    def __init__(self, parent=None, position=None, grid=None, corner_vertices=None):
        """
        Constructor
        
        Inputs:
        
            parent: BiCell, parental cell
            
            position: str, position within parental cell
            
            grid: Grid, 
            
            corner_vertices: interval endpoints, pass either as a list
                [x_min, x_max], or a list of Vertices. 
        """
        super().__init__()

        # =====================================================================
        # Tree Attributes
        # =====================================================================
        in_grid = False
        if parent is None:
            #
            # ROOT Node
            # 
            cell_type = 'ROOT'
            cell_depth = 0
            cell_address = []
            
            if grid is not None:
                #
                # Cell contained in a Grid
                # 
                assert position is not None, \
                'Must specify "position" when ROOT QuadCell is in a grid.'  
                
                assert isinstance(grid, Grid), \
                'Input grid must be an instance of Grid class.' 
                
                in_grid = True
                self.grid = grid
                
            else:
                #
                # Free standing ROOT cell
                # 
                assert position is None, \
                'Unattached ROOT cell has no position.'                
        else:
            #
            # LEAF Node
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
        self.children = {'L': None, 'R': None} 
        self.depth = cell_depth
        self._dim = 1
        self.address = cell_address
        self.position = position
        self._child_positions = ['L','R']
        self._vertex_positions = ['L', 'R', 'M']
        self._corner_vertex_positions = ['L', 'R']
        self.__in_grid = in_grid
        
        
        # =====================================================================
        # Vertices
        # =====================================================================
        #
        # Initialize Vertices and Edges
        #
        vertex_keys = ['L','M','R']
        vertices = dict.fromkeys(vertex_keys)
        edge_keys = [('L','R'),('L','M'), ('M','R')] 
        edges = dict.fromkeys(edge_keys)
        
        
        # Classify cell
        is_free_root = self.type == 'ROOT' and not self.in_grid()
        is_leaf = self.type == 'LEAF'
        
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
                vertices['L'] = Vertex((0,))
                vertices['R'] = Vertex((1,))
            else:
                #
                # Determine the input type
                # 
                if type(corner_vertices) is list:
                    assert len(corner_vertices) == 2, \
                    '4 Vertices needed to specify QuadCell'
                    
                    if all([isinstance(v,numbers.Real) \
                            for v in corner_vertices]):
                        #
                        # Box [x_min, x_max]
                        # 
                        x_min, x_max = corner_vertices
                        
                        assert x_min<x_max, \
                        'Interval endpoints should be ordered.'
                        
                        vertices['L'] = Vertex((x_min,))
                        vertices['R'] = Vertex((x_max,))
                        
                    elif all([type(v) is tuple for v in corner_vertices]):
                        #
                        # Vertices passed as tuples 
                        #
                        
                        # Check tuple length
                        assert all([len(v)==1 for v in corner_vertices]), \
                            'Vertex tuples should be of length 2.'
                        
                        vertices['L'] = Vertex(corner_vertices[0])
                        vertices['R'] = Vertex(corner_vertices[1])
                        
                    elif all([isinstance(v, Vertex) for v in corner_vertices]):
                        #
                        # Vertices passed in list 
                        #             
                        vertices['L'] = corner_vertices[0]
                        vertices['R'] = corner_vertices[1]   

                elif type(corner_vertices) is dict:
                    #
                    # Vertices passed in a dictionary
                    #  
                    for pos in ['L', 'R']:
                        assert pos in corner_vertices.keys(), \
                        'Dictionary should contain at least corner positions.'
                    
                    for pos in corner_vertices.keys():
                        assert isinstance(corner_vertices[pos],Vertex), \
                        'Dictionary values should be of type Vertex.'
                        if vertices[pos] is None:
                            vertices[pos] = corner_vertices[pos]                
        elif in_grid:    
            #
            # ROOT Cell contained in grid
            # 
            assert corner_vertices is None, \
            'Input "cell_vertices" cannot be specified for cell in grid.'
            
            # Extract interval endpoints from connectivity of grid
            i0,i1 = grid.faces['connectivity'][self.position]
            vertices['L'] = grid.points['coordinates'][i0]
            vertices['R'] = grid.points['coordinates'][i1]
                
        elif is_leaf:
            #
            # LEAF cells inherit corner vertices from parents
            # 
            inherited_vertices = \
                    {'L': {'L':'L', 'R':'M'},
                     'R': {'L':'M', 'R':'R'}}
                
            for cv,pv in inherited_vertices[position].items():
                vertices[cv] = parent.vertices[pv]
        #
        # Record corner vertex coordinates
        #
        x_l, = vertices['L'].coordinate()
        x_r, = vertices['R'].coordinate()    
        #
        # Middle vertex
        # 
        vertices['M'] = vertices['M'] = Vertex((0.5*(x_l+x_r),))
                
        #
        # Edges
        # 
        if is_leaf:
            #
            # LEAF cells inherit edges from parents
            # 
            inherited_edges = \
                    {'L': { ('L','R'):('L','M')}, 
                     'R': { ('L','R'):('M','R')}} 
                     
            for ce,pe in inherited_edges[position].items():
                edges[ce] = parent.edges[pe]  
        elif is_free_root:
            #
            # Specify all interior edges of free cell
            #
            edges[('L','R')] = Edge(vertices['L'],vertices['R'])      
        #
        # New interior edges
        #             
        edges[('L','M')] = Edge(vertices['L'],vertices['M'])
        edges[('M','R')] = Edge(vertices['M'],vertices['R'])
               
        #
        # Store vertices and edges
        #  
        self.vertices = vertices
        self.edges = edges    
              
        
    
    def area(self):
        """
        Return the length of the BiCell
        """
        V = self.get_vertices(pos='corners', as_array=True)
        return np.abs(V[1,0]-V[0,0])


            
    def box(self):
        """
        Return the cell's interval endpoints
        """
        x0, = self.vertices['L'].coordinate()
        x1, = self.vertices['R'].coordinate()
        return x0, x1
    
    
    
    def get_neighbor(self, direction):
        """
        Returns the deepest neighboring cell, whose depth is at most that of the
        given cell, or 'None' if there aren't any neighbors.
         
        Inputs: 
         
            direction: char, 'L', 'R'
             
        Output: 
         
            neighboring cell    
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
            return self.grid['faces']['Cell'][i_nb_fc]     
        #
        # Non-ROOT cells 
        # 
        else:
            #
            # Check for neighbors interior to parent cell
            # 
            if direction == 'L':
                if self.position == 'R':
                    return self.parent.children['L']
            elif direction == 'R':
                if self.position == 'L':
                    return self.parent.children['R']
            else:
                raise Exception('Invalid direction. Use "L", or "R".')    
            #
            # Check for (children of) parental neighbors
            #
            mu = self.parent.get_neighbor(direction)
            if mu == None or mu.type == 'LEAF':
                return mu
            else:
                if direction == 'L':
                    return mu.children['R']
                elif direction == 'R':
                    return mu.children['L']
               
    '''
    TODO: Remove 
    def get_leaves(self, with_depth=False):
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
                leaves.extend(child.get_leaves(with_depth))    
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
  
    
    def get_root(self):
        """
        Find the ROOT cell for a given cell
        """
        if self.type == 'ROOT' or self.type == 'MESH':
            return self
        else:
            return self.parent.get_root()
      
    def has_children(self):
        """
        Returns True if cell has any sub-cells, False otherwise
        """    
        return any([self.children[pos]!=None for pos in self.children.keys()])
    '''
    
    
    
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
        x = np.array(points)
        x_min, = self.vertices['L'].coordinate() 
        x_max, = self.vertices['R'].coordinate()
        
        in_box = (x_min <= x) & (x <= x_max)
        return in_box
     
                      
    def locate_point(self, point):
        """
        Returns the smallest cell containing a given point or None if current 
        cell doesn't contain the point
        
        Input:
            
            point: double, x
            
        Output:
            
            cell: smallest cell that contains x
                
        """
        # TESTME: locate_point
        
        if self.contains_point(point):
            if self.type == 'LEAF': 
                return self
            else:
                #
                # If cell has children, find the child containing 
                # the point and continue looking from there.
                # 
                for child in self.children.values():
                    if child.contains_point(point):
                        return child.locate_point(point)                     
        else:
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
        x0, = self.vertices['L'].coordinate()
        x1, = self.vertices['R'].coordinate()
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
        hess = np.zeros(n)
         
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
        
    '''
    def derivative_multiplier(self, derivative):
        """
        Let y = l(x) be the mapping from the physical to the reference element.
        Then, if a (shape) function f(x) = g(l(x)), its derivative f'(x) = g'(x)l'(x)
        This method returns the constant l'(x) = 1/(b-a).   
        """
        c = 1
        if derivative[0] in {1,2}:
            # 
            # There's a map and we're taking derivatives
            #
            x0, = self.vertices['L'].coordinate()
            x1, = self.vertices['R'].coordinate()
            for _ in range(derivative[0]):
                c *= 1/(x1-x0)
        return c
    '''                
                                
    def split(self):
        """
        Split cell into subcells
        """
        assert not self.has_children(), 'Cell is already split.'
        for pos in self._child_positions:
            self.children[pos] = BiCell(parent=self, position=pos) 
        
        
    def pos2id(self, position):
        """
        Convert 'L' and 'R' to 0 and 1 respectively
        
        Input:
        
            position: str, 'L', or 'R', or position in grid
            
        Output:
        
            position: int, 0 (for 'L') or 1 (for 'R'), or position within grid
        """
        if type(position) is int:
            #
            # Position is already an integer
            # 
            if self.in_grid():
                assert position < self.grid.faces['n'], \
                'Position exceeds total number of faces in grid.'
                return position
            elif position in [0,1]:
                return position 
        else:
            assert position in ['L','R'], \
            'Position is %s. Use "R" or "L" for position' % position
            if position == 'L':
                return 0
            else:
                return 1

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
        a = [v[1].coordinate()[i] - v[0].coordinate()[i] for i in range(2)]
        b = [v[2].coordinate()[i] - v[0].coordinate()[i] for i in range(2)]
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
            
            grid: Grid, object containing cell
            
            corner_vertices: vertices on the sw, se, ne, and nw corners
                passed as a list of tuples/vertices or a dict of vertices,
                or a rectangular box = [x_min, x_max, y_min, y_max]
            
            
        Modified: 12/27/2016
        """
        super().__init__()
        
        # =====================================================================
        # Tree Attributes
        # =====================================================================
        in_grid = False
        if parent is None:
            #
            # ROOT Node
            # 
            cell_type = 'ROOT'
            cell_depth = 0
            cell_address = []
            
            if grid is not None:
                #
                # Cell contained in a Grid
                # 
                assert position is not None, \
                'Must specify "position" when ROOT QuadCell is in a grid.'  
                
                assert isinstance(grid, Grid), \
                'Input grid must be an instance of Grid class.' 
                
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
            # LEAF Node
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
        self.children = {'SW': None, 'SE': None, 'NE':None, 'NW':None} 
        self.depth = cell_depth
        self._dim = 2
        self.address = cell_address
        self.position = position
        self._child_positions = ['SW','SE','NW','NE']
        self._vertex_positions = ['SW', 'S', 'SE', 'E', 
                                  'NE', 'N', 'NW', 'W','M']
        self._corner_vertex_positions = ['SW', 'SE', 'NE', 'NW']
        self.__in_grid = in_grid
        
        # =====================================================================
        # Vertices and Edges
        # =====================================================================
        #
        # Initialize
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
                    v_prev = vertices[cnr_positions[(i-1)%4]].coordinate()
                    v_curr = vertices[cnr_positions[i%4]].coordinate()
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
        x_sw, y_sw = vertices['SW'].coordinate()
        x_se, y_se = vertices['SE'].coordinate()
        x_ne, y_ne = vertices['NE'].coordinate()
        x_nw, y_nw = vertices['NW'].coordinate()    
    
        
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
                x0, y0 = vertices[sub_directions[direction][0]].coordinate()
                x1, y1 = vertices[sub_directions[direction][1]].coordinate()
                vertices[direction] = Vertex((0.5*(x0+x1), 0.5*(y0+y1)))    
        #
        # Middle vertex
        # 
        vertices['M'] = vertices['M'] = Vertex((0.25*(x_sw+x_se+x_ne+x_nw),\
                                                0.25*(y_sw+y_se+y_ne+y_nw)))
        
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

    '''   
    def find_cells_at_depth(self, depth):
        """
        Return a list of cells at a certain depth
        
        TODO: Is this necessary? 
        TODO: Move to Cell class
        """
        cells = []
        if self.depth == depth:
            cells.append(self)
        elif self.has_children():
            for child in self.children.values():
                cells.extend(child.find_cells_at_depth(depth))
        return cells
    '''       
        

    
    
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
            
            x0, y0 = self.vertices[pos_prev].coordinate()
            x1, y1 = self.vertices[pos_curr].coordinate()

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
        xm, ym = self.vertices['M'].coordinate()
        v0,v1 = edge.vertices()
        x0,y0 = v0.coordinate() 
        x1,y1 = v1.coordinate()
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
        
    '''
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
    '''
    
    
    '''    
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
    '''
                
    def reference_map(self, x_in, jacobian=False, 
                      hessian=False, mapsto='physical'):
        """
        Bilinear map between reference cell [0,1]^2 and physical cell
        
        Inputs: 
        
            x: double, list of of n (2,) arrays of input points, either in 
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
        x_verts = self.get_vertices(pos='corners', as_array=True)
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
    '''

    '''
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
    '''
                
        
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
        

class HalfEdge(object):
    """
    Description: Half-Edge in Quadtree mesh
    
    Attributes:
    
        base: Vertex, at base 
        
        head: Vertex, at head
        
        twin: HalfEdge, in adjoining cell pointing from head to base 
        
        cell: QuadCell, lying to half edge's left 
        
    Methods:
    
    
    """ 
    def __init__(self, base, head, parent=None, cell=None, 
                 previous=None, nxt=None, twin=None):
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
        """
        self.__base = base
        self.__head = head
        self.__parent = parent
        self.__cell = cell
        self.__previous = previous
        self.__next = nxt
        self.__twin = twin
        self.__children = [None, None]
    
    
       
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
        self.__twin = twin
    
    
    def next(self):
        """
        Returns the next half-edge, whose base is current head
        """
        return self.__next
    
    
    def assign_next(self, nxt):
        """
        Assigns half edge to next
        """
        self.__next = nxt
        
    
    def previous(self):
        """
        Returns previous half-edge, whose head is current base
        """
        return self.__previous
        
    
    def assign_previous(self, previous):
        """
        Assigns half-edge to previous
        """
        self.__previous = previous
    
        
    def has_parent(self):
        """
        Returns true if HE has parent
        """
        return self.__parent is not None

    
    def get_parent(self):
        """
        Returns half-edge's parent edge
        """
        return self.__parent
        
    
    def has_children(self):
        """
        Determine whether half-edge has been split 
        """
        return all([child is not None for child in self.__children])
    
    
    def get_children(self):
        """
        Returns the half-edge's children
        """    
        return self.__children
    
        
    def split(self):
        """
        Refine current half-edge
        """
        x0, y0 = self.base().coordinate()
        x1, y1 = self.head().coordinate()
        vm = Vertex(0.5*(x0+x1), 0.5*(y0+y1))
        self.__children[0] = HalfEdge(self.base(), vm, parent=self)
        self.__children[1] = HalfEdge(vm, self.head(), parent=self)
        
               
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
    
    def __init__(self, v1, v2, parent=None):
        """
        Description: Constructor
        
        Inputs: 
        
            v1, v2: Vertex, two vertices that define the edge
            
            parent: One QuadCell/TriCell containing the edge (not necessary?)
            
            on_boundary: Either None (if not set) or Boolean (True if edge lies on boundary)
        """
        self.__vertices = set([v1,v2])
        
        dim = len(v1.coordinate())
        if dim == 1:
            x0, = v1.coordinate()
            x1, = v2.coordinate()
            nnorm = np.abs(x1-x0)
        elif dim == 2:
            x0,y0 = v1.coordinate()
            x1,y1 = v2.coordinate()
            nnorm = np.sqrt((y1-y0)**2+(x1-x0)**2)
        self.__length = nnorm
        self._flags = set()
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
        if isinstance(coordinate, numbers.Real):
            #
            # Coordinate passed as a real number 1D
            # 
            dim = 1
            coordinate = (coordinate,)  # recast coordinate as tuple
        elif type(coordinate) is tuple:
            #
            # Coordinate passed as a tuple
            # 
            dim = len(coordinate)
            assert dim <= 2, 'Only 1D and 2D meshes supported.'
        else:
            raise Exception('Enter coordinate as a number or a tuple.')
        self.__coordinate = coordinate
        self._flags = set()
        self.__dim = dim
    
    def coordinate(self):
        """
        Return coordinate tuple
        """
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
