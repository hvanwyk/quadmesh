from assembler import Assembler
from fem import Element
from fem import Basis
from fem import DofHandler
from fem import QuadFE
from function import Nodal, Explicit, Constant
from mesh import QuadMesh
from plot import Plot

import numpy as np

# Mesh 
mesh = QuadMesh(box=[-1,1,-1,1], resolution=(20,20))
Q1 = QuadFE(2,'Q1')  # element for pressure
Q2 = QuadFE(2,'Q2')  # element for velocity

# Dofhandler
DQ1 = DofHandler(mesh, Q1)
DQ2 = DofHandler(mesh, Q2)

# Problem Parameters
g = 10
gma = 0.1
U = 0.1
tht = 1
nu = Explicit(f=lambda x: np.sin(x[:,0])**2+1, dim=2)

# Explicit solution 
u1 = Explicit(f=lambda x,t: t[:,0]**3*np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1]),\
              dim=2, n_variables=2)
u2 = Explicit
plot = Plot()
plot.contour(nu, mesh=mesh)

