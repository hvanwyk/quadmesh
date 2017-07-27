"""
Experiment with representation, simulation and conditioning of Gaussian fields
using the KL expansion. 
"""
# Local imports
from mesh import Mesh
from finite_element import System, QuadFE
from plot import Plot
from gmrf import sqr_exponential_cov, distance

# General imports
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

#
# Generate mesh and finite elements
# 
mesh = Mesh.newmesh(grid_size=(10,10))
mesh.refine()
mesh.record(0)

box = [0.25,0.5,0.25,0.5]
for i in range(3):    
    for node in mesh.root_node().find_leaves():
        cell = node.quadcell()
        outside_box = False
        for v in cell.vertices.values():
            x,y = v.coordinate()
            if x < box[0] or x > box[1] or y < box[2] or y > box[3]:
                outside_box = True
                
        if not outside_box:
            node.mark('refine')
                    
    mesh.refine('refine')
    mesh.balance()
mesh.record(1)

#
# Assemble the covariance Matrix
# 
element = QuadFE(2,'Q2')
system = System(mesh, element)

C = lambda x,y: sqr_exponential_cov(x, y, sgm=25, l=0.1)
M = np.array([[4,1],[1,2]])
x = np.array([1,2]).reshape(-1,2)
print(x)
y = np.zeros(x.shape)
distance(x,y,M)
system.assemble(bilinear_forms=[(1,'u','v')])
n_nodes = system.get_n_nodes() 
CC = np.zeros((n_nodes, n_nodes))
for node in mesh.root_node().find_leaves():
    node_dofs = system.get_global_dofs(node)
    n_dofs = len(node_dofs) 
    cell = node.quadcell()
    x_loc = system.x_loc(cell)
    phi = system.shape_eval(cell=cell)
    rule = system.cell_rule()
    weights = rule.jacobian(cell)*rule.weights()
    x_gauss = rule.map(cell, x=rule.nodes())
    n_gauss = rule.n_nodes() 
    i,j = np.meshgrid(np.arange(n_gauss), np.arange(n_gauss))
    C_loc = C(x_gauss[i.ravel(),:],x_gauss[j.ravel(),:]).reshape(n_gauss,n_gauss)
    W = np.diag(weights)
    WPhi = W.dot(phi)
    CC_loc = np.dot(WPhi.T,C_loc.dot(WPhi))
    for i in range(n_dofs):
        for j in range(n_dofs):
            CC[node_dofs[i],node_dofs[j]] = CC_loc[i,j]
    
U,S,VT = np.linalg.svd(CC)
Z = np.random.normal(size=(n_nodes,))
print(Z.shape)
X = U.dot(np.diag(np.sqrt(S)).dot(Z))
  
fig, ax = plt.subplots(1,3)
plot = Plot()
ax[0] = plot.mesh(ax[0], mesh, node_flag=0)
ax[1] = plot.mesh(ax[1], mesh, node_flag=1)
ax[2] = plot.contour(ax[2], fig, X, mesh, element)
plt.show()