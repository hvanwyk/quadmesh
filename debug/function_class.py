from fem import QuadFE, DofHandler, Function
from mesh import Mesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


mesh = Mesh.newmesh()
for _ in range(5):
    mesh.refine()

element = QuadFE(2,'Q3')
f = Function( lambda x,y: np.sin(2*np.pi*x)*np.cos(3*np.pi*y), 'explicit')
fn = f.interpolate(mesh, element)
g = fn.derivative((2,0,0))

fig = plt.figure()

ax = fig.add_subplot(1,1,1, projection='3d')
for node in mesh.root_node().find_leaves():
    x0, x1, y0, y1 = node.quadcell().box()
    xx, yy = np.meshgrid(np.linspace(x0,x1,5),np.linspace(y0,y1,5))
    xy = np.array([xx.ravel(),yy.ravel()]).T
    zz = g.eval(xy, node).reshape(xx.shape)
    ax.plot_surface(xx,yy,zz, cmap='viridis', vmin=-30, vmax=30, alpha=0.9)
plt.show()
'''
dofhandler = DofHandler(mesh,element)
for node in mesh.root_node().traverse_depthwise():
    node.info()
    dofhandler.fill_dofs(node)
    print('Dofs: {0}'.format(dofhandler.get_global_dofs(node)))
    
    
#dofhandler.distribute_dofs(nested=True)
''' 