import numpy as np

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
     
        
    def intersects_line_segment(self, line):
        """
        Determine whether the edge intersects with a given line segment
        
        Input: 
        
            line: double, list of two tuples
            
        Output:
        
            boolean, true if intersection, false otherwise.
        """        
        # Express edge as p + t*r, t in [0,1]
        p = np.array(self.v_begin.coordinate)
        r = np.array(self.v_end.coordinate) - p
        
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
