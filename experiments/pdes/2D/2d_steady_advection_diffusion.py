"""
Solve the steady advection-diffusion equation in 2D:

    -div(q*grad(y)) + a . grad(y) = f  in  D, 

 with boundary conditions
 
    y = g_D       on dD
    q*dy/dn = g_N on dN

We are interested in computing the following quantities of interest (QoI):

1. The value of y at a specific point in the domain.
2. The flux of y across a specific boundary.
3. The average value of y over the entire domain.
"""
# Check if quadmesh is in the path
import sys

if '/home/hans-werner/git/quadmesh/src' not in sys.path:
    sys.path.append('/home/hans-werner/git/quadmesh/src')

# built-in modules
import numpy as np
import matplotlib.pyplot as plt


# Quadmesh modules
from mesh import QuadMesh
from fem import QuadFE, DofHandler, Basis
from function import Nodal, Constant, Explicit
from assembler import Assembler, Form, Kernel
from plot import Plot

# -----------------------------------------------------------------------------
# Geometry and Mesh
mesh = QuadMesh(box=[0, 2, 0, 1], resolution=(64, 64))

# Mark boundaries
mesh.mark_region('left', lambda x,y: abs(x) < 1e-6, 
                 entity_type='vertex', on_boundary=True)

mesh.mark_region('right', lambda x,y: abs(x-2) < 1e-6, 
                 entity_type='vertex',   on_boundary=True)

mesh.mark_region('bottom', lambda x,y: abs(y) < 1e-6, 
                 entity_type='half_edge', on_boundary=True)

mesh.mark_region('top', lambda x,y: abs(y-1) < 1e-6, 
                 entity_type='half_edge', on_boundary=True)


plot = Plot(quickview=False)
fig, ax = plt.subplots(figsize=(6,3))
regions = [('left','vertex'),('right','vertex'), ('bottom','edge'),('top','edge')]
ax = plot.mesh(mesh,regions=regions,axis=ax)
ax.set_title('Mesh')
plt.tight_layout()
plt.show()

# Elements and Basis
element = QuadFE(2, 'Q1')  # Linear elements

# Degree of freedom handler (mesh + element)
dofhandler = DofHandler(mesh, element)  
dofhandler.distribute_dofs() 

# Basis functions (mesh + element + function)
v = Basis(dofhandler, 'v')    # Test/trial function
vx = Basis(dofhandler, 'vx')  # x-derivative of test function
vy = Basis(dofhandler, 'vy')  # y-derivative of test function

# Coefficients and Source Terms
q = Constant(0.1)  # Diffusion coefficient
a0 = Constant(1)
a1 = Constant(0)
#f = Explicit(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]),dim=1)  # Source term

#

q = 1  # diffusion coefficient
a0,a1 = 2, 1.5 # advection coefficients
b0,b1 = 0.1*np.pi, 1*np.pi # frequency in x1 and x2 directions

# Reference solution
y_ref = Explicit(lambda x: np.sin(b0*x[:,0])*np.sin(b1*x[:,1]),dim=1)

# Corresponding source term
f = Explicit(lambda x: q*(b0**2+b1**2)*np.sin(b0*x[:,0])*np.sin(b1*x[:,1]) + \
                       a0*b0*np.cos(b0*x[:,0])*np.sin(b1*x[:,1]) + \
                       a1*b1*np.sin(b0*x[:,0])*np.cos(b1*x[:,1]),dim=1)

g_D = Constant(0.0)  # Dirichlet BC
g_N = Constant(0.0)  # Neumann BC

# Variational Forms
problems = [Form(kernel=q, test=vx, trial=vx),
            Form(kernel=q, test=vy, trial=vy),
            Form(kernel=a0, test=v, trial=vx),
            Form(kernel=a1, test=v, trial=vy)]


#assembler = Assembler(problems,mesh)
#assembler.assemble()

