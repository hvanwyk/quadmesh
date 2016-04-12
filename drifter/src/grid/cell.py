from drifter.grid.edge import Edge
from drifter.grid.vertex import Vertex

class Cell(object):
    """
    Description: (Tree of) Rectangular cell(s) in mesh
    
    
    Attributes: 
    
        type - Cells can be one of the following types:
            
            ROOT   - cell on coarsest level
            BRANCH - cell has parent(s) as well as children
            LEAF   - cell on finest refinement level
         
        parent: cell of which current cell is a sub-cell
         
        children: list of sub-cells of current cell
         
        neighbors: addresses of neighboring cells
         
        depth: int, current refinement level 0,1,2,...
         
        rectangle: (x0, x1, y0, y1), where cell = [x0, x1]x[y0, y1]
         
        min_size: double, minimum possible mesh width
     
    
    Methods: 
     
        refine, coarsen, has_children,  
    """
    # TODO: Is rectangle necessary? 
    
    # Globals
    NODE_NUM = 0
    VERTEX_LIST = [] 
    EDGE_LIST = []
    
    def __init__(self, parent, vertices, pos=None):
        """
        Description: Initializes the cell (sub-)tree
        
        Inputs: 
                    
            parent: parental Node
            
            rectangle: rectangle defining the cell boundary
            
            pos: own position in parent cell (NW, SW, NE, SE)
        """
 
        self.parent = parent
        self.flag = False
        self.position = pos
        #
        # Classify Node as ROOT or LEAF
        # 
        if self.parent == None:
            self.type = 'ROOT'
            self.children = []
            self.depth = 0            
        else:
            self.type = 'LEAF'
            self.children = {'NE': None, 'NW': None, 'SE': None, 'SW': None}
            self.depth = parent.depth + 1

            #
            # Position vertices within Node
            # 
            v_sw, v_se, v_ne, v_nw = vertices
            self.vertices = {'SW': v_sw, 'SE': v_se, 'NW': v_nw, 'NE': v_ne, 
                             'N': None, 'S': None, 'E': None, 'W': None, 
                             'M': None}
           
            #
            # Define bounding rectangle
            #
            coordinates = [vertex.coordinates for vertex in vertices]
            x0, y0 = min(coordinates)
            x1, y1 = max(coordinates)
            self.rectangle = (x0, x1, y0, y1)         
                             
            #
            # Define edges
            # 
            e_we = Edge(v_sw, v_se, self)
            e_sn = Edge(v_se, v_ne, self)
            e_ew = Edge(v_ne, v_nw, self)
            e_ns = Edge(v_nw, v_sw, self)
            self.edges = {'S': e_we, 'E': e_sn, 'N': e_ew, 'W': e_ns}
                 
            if parent.type == 'ROOT':
                self.position = None
            else:
                self.position = pos

        