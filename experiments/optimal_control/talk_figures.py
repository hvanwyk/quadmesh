from fem import DofHandler
from fem import QuadFE

from function import Nodal

from mesh import QuadMesh, Mesh1D

from gmrf import GaussianField
from gmrf import SPDMatrix
from gmrf import Covariance

from plot import Plot

import numpy as np
from scipy import linalg
from scipy import sparse as sp
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
savedir = '/home/hans-werner/Dropbox/work/presentations/2019_chengdu_siam_ct/'

"""
#
# Plot simple quadratric example
#
x = np.linspace(-2,1.5, 100)
#u = 2*np.random.rand(10)-1
u = np.linspace(-1,1,7)
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111) 
for ui in u:
    ax.plot(x,(x-ui)**2, 'C0', alpha=0.5)

xticks = np.array([-1.5,-1,0, 1])
plt.xticks(xticks,(r'$u_0$', '-1', r'$u*$','1'))
plt.yticks([],[])
plt.savefig(savedir+'parabolas.pdf')


#
# Gaussian fields of different resolution
# 
names = ['low_res.jpeg', 'high_res.jpeg']
for resolution, name in zip([(10,10), (40,40)],names):
    mesh = QuadMesh(resolution=resolution)
    Q0 = QuadFE(mesh.dim(), 'DQ0')
    Q1 = QuadFE(mesh.dim(), 'Q1')
    
    dofhandler = DofHandler(mesh, Q0)
    dofhandler.distribute_dofs()
    n_dofs = dofhandler.n_dofs()
    cov = Covariance(dofhandler, name='gaussian', parameters={'l':0.05})
    n_samples = 1
    k = GaussianField(n_dofs, K=cov)
    k.update_support()
    f = Nodal(dofhandler=dofhandler, data=k.sample())
    
    fig, ax = plt.subplots(1,1, figsize=(3,3))
    plot = Plot(quickview=False)
    ax = plot.contour(f, axis=ax, colorbar=False)
    ax.set_axis_off()
    plt.savefig(savedir+name, dpi=300, bbox_inches='tight', pad_inches=0)
"""    
 
#
# 
# 
epsilon = np.linspace(1e-6,1,500)
n = np.arange(1000)
plt.loglog(epsilon, -np.log(epsilon)/epsilon)
plt.show()