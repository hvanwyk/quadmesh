from mesh import QuadMesh
from fem import DofHandler, System, QuadFE
from fem import Kernel, Form, Basis, Function
from plot import Plot
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np

mesh = QuadMesh(resolution=(2,2))
cell = mesh.cells.get_child(0)
cell.mark(1)
mesh.cells.refine(refinement_flag=1)
element = QuadFE(2, 'Q1')
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plot = Plot(quickview=False)
plot.mesh(mesh, ax=ax, dofhandler=dofhandler, dofs=True)
plt.show()


hanging_nodes = dofhandler.get_hanging_nodes()
#print(hanging_nodes.keys())

#
# Set up system
# 
ux = Basis(element, 'ux')
uy = Basis(element, 'uy')
u  = Basis(element, 'u')

one = Kernel(Function(1,'constant'))
problem = [Form(one, ux, ux), Form(one, u)]

system = System(problem, mesh)
system.assemble()
rows = system.af[0]['bilinear']['rows']
cols = system.af[0]['bilinear']['cols']
dofs = system.af[0]['bilinear']['row_dofs']
vals = system.af[0]['bilinear']['vals']

A = sp.coo_matrix((vals,(rows,cols)))


x = np.arange(4)
y = np.array([1,3,6,2])
ii = np.array([0,1,1,1,0,3,2,2])
print(y[ii])
z = np.array([3,2])
dirichlet = np.zeros(4, dtype=np.bool)
for zi in z:
    dirichlet[y==zi] = True
print(x[dirichlet])
print(x[~dirichlet])
print(y)
print(dirichlet)
