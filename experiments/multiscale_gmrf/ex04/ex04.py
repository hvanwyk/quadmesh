from fem import DofHandler, Basis, QuadFE
from gmrf import GaussianField, Covariance
from mesh import Mesh1D, QuadMesh
from plot import Plot
from function import Nodal
import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt


#
# Nested Mesh
# 
mesh = Mesh1D()
for i in range(8):
    mesh.record(i)
    mesh.cells.refine()

# Finite Element Space
DQ0 = QuadFE(1,'DQ0')
dh_0 = DofHandler(mesh,DQ0)
dh_0.distribute_dofs()
n = dh_0.n_dofs() 

phi_0 = Basis(dh_0)
psi_0 = Basis(dh_0, subforest_flag=7)
    
#plot.mesh(mesh, dofhandler=dh)

C = Covariance(dh_0, name='gaussian', parameters={'l':0.05})
eta = GaussianField(n,K=C)
eta_path = Nodal(data=eta.sample(), basis=phi_0)


#
# Coarsening 
#

rows = []
cols = []
vals = []
for leaf in mesh.cells.get_leaves():
    rows.extend(dh_0.get_cell_dofs(leaf.get_parent()))
    cols.extend(dh_0.get_cell_dofs(leaf)) 
    vals.append(0.5)

#
# Map to index 
#

# Rows
coarse_dofs = list(set(rows))
dof2idx = dict()
for (dof,i) in zip(coarse_dofs,range(len(coarse_dofs))):
    dof2idx[dof] = i 
rows = [dof2idx[dof] for dof in rows]

# Columns
fine_dofs = list(set(cols))
dof2idx = dict()
for (dof,i) in zip(fine_dofs,range(len(fine_dofs))):
    dof2idx[dof] = i 
cols = [dof2idx[dof] for dof in cols]

# Local averaging matrix
R = sp.coo_matrix((vals,(rows,cols))).tocsc()

# Average data
ave_data = R.dot(eta_path.data())
eta_ave = Nodal(data=ave_data, basis=psi_0)


#
# Plots
# 
plot = Plot(quickview=False)
ax = plt.subplot(111)
ax = plot.line(eta_path, axis=ax, mesh=mesh)
ax = plot.line(eta_ave, axis=ax, mesh=mesh)

plt.show()