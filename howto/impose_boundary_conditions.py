"""
This is how to implement Neumann boundary conditions in the variational formulation. 

 using the dmu argument of the Form class to specify integration over the boundary and a Kernel to specify the Neumann flux function.
"""

from assembler import Assembler, Form
from fem import Basis, DofHandler, QuadFE
from function import Nodal, Constant, Explicit
from mesh import Mesh1D, QuadMesh

# Geometry and Mesh
mesh = QuadMesh(box=[0, 2, 0, 1], resolution=(64, 64))

# Specify Neumann regions
mesh.mark_region('top', lambda x,y: abs(y-1) < 1e-6, 
                 entity_type='half_edge', on_boundary=True)
mesh.mark_region('bottom', lambda x,y: abs(y) < 1e-6, 
                 entity_type='half_edge', on_boundary=True)

# Elements and Basis
element = QuadFE(2, 'Q1')  # Linear elements
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()
v = Basis(dofhandler, 'v')    # Test/trial function
vx = Basis(dofhandler, 'vx')  # x-derivative of test function
vy = Basis(dofhandler, 'vy')  # y-derivative of test function

