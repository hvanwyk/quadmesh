from grid.edge import Edge
import numpy as np

class Triangle(object):
    """
    Triangle object
    
    Attributes:
        
    
    Methods:
    
    """
    def __init__(self, vertices, parent_cell=None, parent_triangle=None):
        """
        Inputs:
        
            vertices: Vertex, list of three vertices (ordered counter-clockwise)
            
            parent_cell: Cell, cell containing triangle
            
            parent_triangle: Triangle, supertriangle containing triangle
            
        """
        self.__vertices = vertices
        self.__parent_cell = parent_cell
        self.__parent_triangle = parent_triangle
        self.__edges = [
                        Edge(vertices[0], vertices[1]), \
                        Edge(vertices[1], vertices[2]), \
                        Edge(vertices[2], vertices[0])
                        ]
    
    def vertices(self):
        return self.__vertices
    
    def area(self):
        """
        Compute the area of the triangle
        """
        v = self.__vertices
        a = [v[1].coordinate[i] - v[0].coordinate[i] for i in range(2)]
        b = [v[2].coordinate[i] - v[0].coordinate[i] for i in range(2)]
        return 0.5*abs(a[0]*b[1]-a[1]*b[0])
        
    def number(self, num):
        """
        Assign a number to the triangle
        """
    
    
    