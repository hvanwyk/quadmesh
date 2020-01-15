from assembler import Form, Kernel, Assembler
from fem import QuadFE, Basis, DofHandler
from function import Nodal, Explicit, Constant
from mesh import Mesh1D, Interval
from plot import Plot
from solver import LinearSystem

import numpy as np

"""
Simulate the time dependent advection-diffusion-reaction system 

    u_t - div*(D*grad(u)) + div(v*u) + R(u) = 0
    
subject to the appropriate initial and boundary conditions, using
SUPG and 
 
"""
# Computational mesh
