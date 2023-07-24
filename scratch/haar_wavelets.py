from mesh import QuadMesh, Mesh1D
from plot import Plot
from fem import QuadFE, DofHandler, Basis
from function import Explicit, Nodal
import numpy as np
from gmrf import Covariance, GaussianField
from diagnostics import Verbose

vb = Verbose()
plot = Plot()
mesh = QuadMesh()
Q0 = QuadFE(2,'DQ0')
dh0 = DofHandler(mesh,Q0)

n_levels = 6

for l in range(n_levels):
    mesh.cells.refine(new_label=l)
    dh0.distribute_dofs(subforest_flag=l)
    #plot.mesh(mesh, dofhandler=dh0, subforest_flag=l)
    print(l,dh0.n_dofs(subforest_flag=l))
phi = Basis(dh0)
#f = Explicit(lambda x: np.abs(x-0.5), dim=1)
#fQ = f.interpolant(dh0, subforest_flag=3)

"""
plot.line(fQ, mesh) 
plot.mesh(mesh, dofhandler=dh0, subforest_flag=1)

mesh = QuadMesh(resolution=(10,10))
"""

#plot.mesh(mesh)
vb.tic('Assembling Covariance')
cov = Covariance(dh0, discretization='interpolation', name='exponential')
vb.toc()

vb.tic('Plotting row of covariance')
g = Nodal(data=cov.get_matrix()[:,2000], basis=phi)
plot.contour(g)
vb.toc()

verbose = False
if verbose: print('verbose') 

vb.tic('Plotting realization')
q = GaussianField(dh0.n_dofs(), K=cov)
#plot.contour(Nodal(data=q.sample(),basis=phi))
vb.toc()

wfn = lambda x: np.exp(-((x[:,0]-0.5)**2 + (x[:,1]-0.5)**2)/0.01)
w = Nodal(f=wfn, basis=phi)
plot.contour(w)

wq = Nodal(data=w.data()*q.sample(),basis=phi)
plot.contour(wq)