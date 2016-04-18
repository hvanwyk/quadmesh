from grid.cell import Cell
from grid.vertex import Vertex
import numpy

class Mesh(object):
    '''
    Description: (Quad) Mesh object
    
    Attributes:
    
        bounding_box: [xmin, xmax, ymin, ymax]
        
        children: Cell, list of cells contained in mesh 
    
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
    
    
    def find_leaves(self):
        '''
        Returns a list of all leaf sub-cells of the mesh
        '''
        leaves = []
        for child in self.children:
            leaves.extend(child.find_leaves())        
        return leaves
      
        
    def plot_quadmesh(self, name):
        '''
        Plot the current quadmesh
        '''
        pass
    
    def plot_trimesh(self):
        pass
    