from grid.cell import Cell
from grid.vertex import Vertex
import numpy

class Mesh(object):
    '''
    Description: (Quad) Mesh object
    
    Attributes:
    
        bounding_box: [xmin, xmax, ymin, ymax]
        
        children: Cell, list of cells contained in mesh 
        
        vertex_list: Vertex, list of vertices (run number_vertices)
        
        connectivity: int, numpy array - element connectivity matrix (run build_connectivity)
        
        max_refinement_level: int, maximum number of times each of the mesh's cell can be refined
    
    Methods:
    '''


    def __init__(self, box=[0.,1.,0.,1.], nx=10, ny=10):
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
        #
        # Define cells in mesh
        # 
        xmin, xmax, ymin, ymax = box
        x = numpy.linspace(xmin, xmax, nx+1)
        y = numpy.linspace(ymin, ymax, ny+1)
        xx, yy = numpy.meshgrid(x, y, sparse=True)
        mesh_cells = []
        node_number = 0
        for i in range(nx):
            for j in range(ny):
                cell_vertices = {'SW': Vertex((xx(i)  ,yy(j)  )),  
                                 'SE': Vertex((xx(i+1),yy(j)  )),
                                 'NE': Vertex((xx(i+1),yy(j+1))),
                                 'NW': Vertex((xx(i)  ,yy(j+1)))
                                 }
                node_number += 1
                mesh_cells.append(Cell(self, cell_vertices, node_number))
        self.children = mesh_cells
        self.vertex_list = None
        self.connectivity = None
    
    
    def find_leaves(self):
        '''
        Returns a list of all leaf sub-cells of the mesh
        '''
        leaves = []
        for child in self.children:
            leaves.extend(child.find_leaves())        
        return leaves
    
    def number_vertices(self):
        """
        Numbers all vertices and stores them in a list
        """
        #
        # Empty vertex list
        # 
        self.vertex_list = []
        num_vertices = 0
        for cell in self.children:
            num_vertices, cell_vertex_list = cell.number_vertices()
            
        
    def build_connectivity(self):
        """
        Returns the connectivity matrix for the tree
        """
        # TODO: FIX build_connectivity
        self.balance_tree()
        root = self.find_root()
        leaves = root.find_leaves()
    
        econn = []
        
        for leaf in leaves:
            print "Leaf Number:", leaves.index(leaf)
            print "Coordinates:"
            leaf.print_coordinates()
            add_steiner_pt = False
            #
            # Get global indices for each corner vertex
            # 
            gi = {}
            for pos in ['NW', 'SW', 'NE', 'SE']:
                gi[pos] = leaf.vertices[pos].idx
                
            edges = {'S': [[gi['SW'], gi['SE']]], 'N': [[gi['NE'], gi['NW']]], 
                     'W': [[gi['NW'], gi['SW']]], 'E': [[gi['SE'], gi['NE']]] }
                     
            opposite_direction = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W'}
            for direction in ['S', 'N', 'E', 'W']:
                neighbor = leaf.find_neighbor(direction)
                if neighbor == None:
                    print "Neighbor in the", direction, neighbor
                else:
                    if neighbor.type != 'LEAF':
                        print "Neighbor", direction, "has children"
                        # If neighbor has children, then add the midpoint to
                        # your list of vertices, update the list of edges and
                        # remember to add the Steiner point later on. 
                        #
                        od = opposite_direction[direction]
                        leaf.vertices[direction] = neighbor.vertices[od]
                        gi[direction] = leaf.vertices[direction].idx
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
                              
            elif leaf.vertices['M'] == None:
                #
                # Add Steiner Vertex
                # 
                x0, x1, y0, y1 = leaf.rectangle
                Cell.NODE_NUM += 1
                vm = Vertex((0.5*(x0 + x1), 0.5*(y0 + y1)), Cell.NODE_NUM)
                Cell.VERTEX_LIST.append(vm)
                leaf.vertices['M'] = vm
                gi['M'] = vm.idx
                
                for direction in ['N', 'S', 'E', 'W']:
                    for sub_edge in edges[direction]:
                        econn.append([sub_edge[0], sub_edge[1], gi['M']])                           
            
        return econn
    
    def get_maxdepth(self):
        """
        Determine the maximum depth of the mesh
        """
        for child in self.children:
            child.get_
            
    def plot_quadmesh(self, name):
        '''
        Plot the current quadmesh
        '''
        pass
    
    
    def plot_trimesh(self):
        pass
    