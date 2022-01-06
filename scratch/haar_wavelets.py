from mesh import QuadMesh, Mesh1D
from plot import Plot
from fem import QuadFE, DofHandler
from function import Explicit
import numpy as np

plot = Plot()
mesh = Mesh1D()
Q0 = QuadFE(1,'DQ0')
dh0 = DofHandler(mesh,Q0)
n_levels = 10

for l in range(n_levels):
    mesh.cells.refine(new_label=l)
    dh0.distribute_dofs(subforest_flag=l)


f = Explicit(lambda x: np.abs(x-0.5), dim=1)
fQ = f.interpolant(dh0, subforest_flag=3)


plot.line(fQ, mesh) 
plot.mesh(mesh, dofhandler=dh0, subforest_flag=0)

mesh = QuadMesh(resolution=(10,10))
plot.mesh(mesh)