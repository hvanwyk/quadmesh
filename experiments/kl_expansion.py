"""
Experiment with representation, simulation and conditioning of Gaussian fields
using the KL expansion.

Note: The eigenvalue problem on an adaptively refined mesh yields modes 
    that behave erratically near the local refinement. 
    
"""
# Local imports
from mesh import Mesh
from fem import System, QuadFE
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

for i in range(2):    
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

print('Assembling Mass matrix')
Cfn = lambda x,y: sqr_exponential_cov(x, y, sgm=0.5, l=0.01)
M = system.assemble(bilinear_forms=[(1,'u','v')])
n_nodes = system.n_dofs() 
C = np.zeros((n_nodes, n_nodes))
rule = system.cell_rule()
n_gauss = rule.n_nodes()
print('Assembling Covariance Operator')
for node_1 in mesh.root_node().find_leaves():
    node_dofs_1 = system.get_global_dofs(node_1)
    n_dofs_1 = len(node_dofs_1)
    cell_1 = node_1.quadcell()
    
    
    weights_1 = rule.jacobian(cell_1)*rule.weights()
    x_gauss_1 = rule.map(cell_1, x=rule.nodes())
    phi_1 = system.shape_eval(cell=cell_1)    
    WPhi_1 = np.diag(weights_1).dot(phi_1)
    for node_2 in mesh.root_node().find_leaves():
        node_dofs_2 = system.get_global_dofs(node_2)
        n_dofs_2 = len(node_dofs_2)
        cell_2 = node_2.quadcell()
        
        x_gauss_2 = rule.map(cell_2, x=rule.nodes())
        weights_2 = rule.jacobian(cell_2)*rule.weights()
        phi_2 = system.shape_eval(cell=cell_2)
        WPhi_2 = np.diag(weights_2).dot(phi_2)
        
        i,j = np.meshgrid(np.arange(n_gauss),np.arange(n_gauss))
        x1, x2 = x_gauss_1[i.ravel(),:],x_gauss_2[j.ravel(),:]
        C_loc = Cfn(x1,x2).reshape(n_gauss,n_gauss)
    
        CC_loc = np.dot(WPhi_2.T,C_loc.dot(WPhi_1))
        for i in range(n_dofs_1):
            for j in range(n_dofs_2):
                C[node_dofs_1[i],node_dofs_2[j]] += CC_loc[i,j]


print('Computing eigen-decomposition')
lmd, V = la.eigh(C,M.toarray())
lmd = np.real(lmd)
lmd[lmd<0] = 0

K = la.solve(M.toarray(),C)    
#U,D,UT = la.svd(CCC)
Z = np.random.normal(size=(n_nodes,))

#plt.semilogy(lmd)
#plt.show()


X = V.dot(np.diag(np.sqrt(lmd)).dot(Z))
  

print('Plotting field')
fig = plt.figure()


plot = Plot()
ax = fig.add_subplot(1,4,1)
ax = plot.mesh(ax, mesh, node_flag=0)

ax = fig.add_subplot(1,4,2)
ax = plot.mesh(ax, mesh, node_flag=1)

#ax = fig.add_subplot(1,4,3, projection='3d')
#ax = plot.surface(ax, X, mesh, element)

#ax = fig.add_subplot(1,4,4, projection='3d')
#ax = plot.surface(ax, V[:,-3], mesh, element)

ax = fig.add_subplot(1,4,3)
ax = plot.contour(ax, fig, X, mesh, element)

ax = fig.add_subplot(1,4,4)
ax = plot.contour(ax, fig, V[:,-3], mesh, element)

plt.show()