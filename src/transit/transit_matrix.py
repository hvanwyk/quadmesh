from datetime import datetime, timedelta
from scipy import sparse as sp
from grid.mesh import Mesh

class TransitMatrix(object):
    '''
    Description: A transit matrix is a possibly periodic or even time dependent operator
    
    Attributes:
    
    Methods:
    '''
    def __init__(self, mesh, data_file):
        '''
        Constructor
        
        Inputs:
        
            date_range: specify date range
            
            spatial_range: 'auto'
            
            dt
            
            mesh
            
            seasonality
               
        '''
        self.__mesh = mesh
        
        