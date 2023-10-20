from mesh import QuadMesh
from fem import DofHandler, Basis, QuadFE
from plot import Plot
import matplotlib.pyplot as plt

plot = Plot(quickview=False)
mesh = QuadMesh()
mesh.record(0)
mesh.cells.refine(new_label=1)
mesh.cells.refine(new_label=2)

Q1 = QuadFE(2,'DQ0')
dh = DofHandler(mesh,Q1)
dh.distribute_dofs()
fig, ax = plt.subplots(1,1)
ax = plot.mesh(mesh, axis=ax, dofhandler=dh,dofs=True, doflabels=True,subforest_flag=2)
plt.show() 




