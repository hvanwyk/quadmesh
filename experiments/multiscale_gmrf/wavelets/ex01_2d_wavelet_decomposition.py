"""
Multiresolution analysis of random field using Haar wavelets
"""
from mesh import QuadMesh
from function import Nodal 
from plot import Plot
from fem import DofHandler, QuadFE, Basis
from gmrf import Covariance, GaussianField
from diagnostics import Verbose

v = Verbose()
plt = Plot(time=100)
msh = QuadMesh(box=[1,2,0,2],resolution=(50,100))
P0 = QuadFE(msh.dim(),'DQ0')
dh = DofHandler(msh, P0)
dh.distribute_dofs()
phi = Basis(dh)
#plt.mesh(msh,dofhandler=dh,dofs=True)

K = Covariance(dh,name='exponential')
x = GaussianField(dh.n_dofs(), K=K)
xs = Nodal(data=x.sample(),basis=phi)
plt.contour(xs)