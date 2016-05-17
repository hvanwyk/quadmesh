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
    
    def __init__(self, vb, ve, face=None, on_boundary=False):
        """
        Description: Constructor
        
        Inputs: 
        
            vb: Vertex, vertex where edge begins
            
            ve: Vertex, vertex where edge ends
            
            face: Incident face (lying to the left of the edge)
            
            on_boundary: boolean, true if edge lies on boundary
        """
        # TODO: Change incident face to dictionary with either 'N','S' or 'E', 'W' ? 
        
        self.v_begin = vb
        self.v_end = ve
        self.incident_face = face
        self.on_boundary = on_boundary