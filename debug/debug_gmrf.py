"""

"""
from mesh import QuadMesh
from fem import QuadFE
from plot import Plot
from fem import DofHandler 
from parameter_identification import elliptic_adjoint
from mesh import Mesh1D
from mesh import QuadMesh
from fem import DofHandler
from fem import Function
from fem import QuadFE
from fem import Kernel
from fem import Form
from fem import Basis
from fem import Assembler
from fem import LinearSystem
from plot import Plot
import numpy as np
from mesh import HalfEdge
import matplotlib.pyplot as plt





kernel = lambda x,y: sgm**2*np.exp(-(x**2+y**2)/(2*l**2))

mesh = QuadMesh(resolution=(10,10))

element = QuadFE(2, 'DQ0')

u = Basis(element)

form = Form(trial=u, test=u)
assembler = Assembler(form, mesh)
assembler.assemble()
A = assembler.af[0]['bilinear'].get_matrix()
plt.imshow(A.toarray())
plt.show()


dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()
dofhandler.set_dof_vertices()
x = dofhandler.get_dof_vertices()

areas = []
for cell in mesh.cells.get_leaves(subforest_flag=None):
    areas.append(cell.area())

plot = Plot()
plot.mesh(mesh, dofhandler=dofhandler, dofs=True)

#
# Collocation
# 
