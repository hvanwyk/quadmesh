from mesh import QuadMesh, Vertex, HalfEdge, QuadCell, Mesh1D, Interval
from fem import DofHandler, QuadFE, GaussRule, Function
from mesh import convert_to_array
from plot import Plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


print('')


n_ders = 3
I = Interval(Vertex(1),Vertex(3))
x = np.linspace(1,3,100)
etypes = ['DQ0', 'Q1', 'Q2', 'Q3']
derivatives = [(0,),(1,0), (2,0)]
for etype in etypes:
    element = QuadFE(1, etype)
    n_dofs = element.n_dofs()

    fig = plt.figure()
    phi = element.shape(x, cell=I, derivatives=derivatives)
    
    for i_dof in range(n_dofs):
        for i_der in range(len(derivatives)):
            ax = fig.add_subplot(n_dofs, n_ders, i_dof*n_ders + i_der + 1)
            ax.plot(x, phi[i_der][:,i_dof])
    
plt.show()

rule1d = GaussRule(2, element)

x_ref, w_ref = rule1d.nodes(), rule1d.weights()
x, jac = I.reference_map(x_ref, jacobian=True)
print(type(jac))
w = w_ref*np.array(jac)

phi = element.shape(x, cell=I, derivatives=derivatives)
print(phi)
xx, jac = I.reference_map(x, jacobian=True, mapsto='reference')
print(x,w)

"""
print('')
# 1D 
mesh = Mesh1D(resolution=(2,))
mesh.cells.record(0)
mesh.cells.get_child(0).mark('r')
mesh.cells.refine(refinement_flag='r', new_label=1)
plot = Plot()
plot.mesh(mesh,mesh_flag=1)
element = QuadFE(1,'Q1')
dofhandler = DofHandler(mesh, element)

fn = lambda x: x**2
f = Function(fn, 'nodal', dofhandler=dofhandler, subforest_flag=1)
print('Nodal values', f.fn())
x = convert_to_array(np.linspace(0,1,5),1)
for interval in mesh.cells.get_leaves(subforest_flag=0):
    in_cell = interval.contains_points(x)
    x_ref = interval.reference_map(x[in_cell,:], mapsto='reference')
    print(x_ref)
print(f.eval(x))
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.linspace(0,1,100)
plt.plot(x,f.eval(x),'k--')
element = QuadFE(1,'DQ0')
x_ref = element.reference_nodes()
print('reference nodes',x_ref)
fi = f.interpolant(mesh, element, subforest_flag=1)
plt.plot(x,fi.eval(x),'b--')

plt.show()

"""