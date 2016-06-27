"""
Created on Apr 11, 2016

@author: hans-werner
"""

class Vertex(object):
    """
    Description:
    
    Attributes:
    
        coordinate: double, tuple (x,y)
        
        node_number: int, index of vertex in mesh
    
    Methods: 
    """


    def __init__(self, coordinate, node_number=None):
        """
        Description: Constructor
        
        Inputs: 
        
            coordinate: double tuple, x- and y- coordinates of vertex
            
            node_number: int, index for vertex
            
            on_boundary: boolean, true if on boundary
            
            address: int, address within root mesh (computed based on highest refinement level)  
        """
        if type(coordinate) is not tuple:
            raise Exception('Vertex coordinate should be a tuple.')
        else:
            self.coordinate = coordinate
        self.node_number = node_number
        self.on_boundary = None 
        
        
    def set_node_number(self, node_num, overwrite=False):
        """
        Assign node number
        """
        if self.node_number == None or overwrite:
            #
            # No existing node number/overwrite
            # 
            self.node_number = node_num
        else:
            raise Warning('Node number already assigned to this vertex')
            return
       
        
    def mark(self):
        """
        Mark vertex
        """
        self.__flag = True
    
    
    def unmark(self):
        """
        Unmark vertex
        """
        self.__flag = False
    
        
    def is_marked(self):
        """
        Check whether vertex is marked
        """
        return self.__flag