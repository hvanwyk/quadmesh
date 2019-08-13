from mesh import QuadMesh
from mesh import Mesh1D
from fem import DofHandler
from fem import Map
from fem import QuadFE
from fem import Kernel
from fem import Form
from fem import Basis
from fem import Assembler
from fem import LinearSystem
from plot import Plot
import numpy as np
import matplotlib.pyplot as plt

"""
Apply an upwinding scheme for a 1D advection diffusion problem
"""
def xi(alpha, method):
    """
    Compute the quantity xi(alpha) according to one of the standard methods
    
    Inputs: 
    
        method: str, specifying the upwind scheme
            
            'classical': sign(alpha) 
            
            'Ilin': coth(alpha)-1/alpha
            
            'double_asymptotic': (alpha/3     if |alpha|<3, 
                                 (sign(alpha) otherwise
            
            'critical': ( -1-1/alpha, if alpha <= -1
                        < 0         , if -1<=alpha<=1
                        ( 1-1/alpha,  if alpha >= 1 
        
    """
    if method=='classical':
        xi = np.sign(alpha)
    elif method=='ilin':
        xi = np.coth(alpha)-1/alpha
    elif method=='double_asymptotic':
        if np.abs(alpha)<=3:
            xi = alpha/3
        else:
            xi = np.sign(alpha)
    elif method=='critical':
        if alpha <= -1:
            xi = -1-1/alpha
        elif np.abs(alpha) < 1:
            xi = 0
        elif alpha >= 1:
            xi = 1 - 1/alpha  
            
            