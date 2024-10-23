

from mesh import QuadMesh, Mesh1D, QuadCell, HalfEdge, Vertex
from fem import QuadFE, DofHandler, Basis
from assembler import Assembler, Form, Kernel, GaussRule
from function import Constant, Nodal
import numpy as np
from plot import Plot
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
from mpl_toolkits import mplot3d

"""
Test the evaluation of basis functions over sub-meshes.

"""
# Test something with a dictionary
a = {}
a[None] = 1
a['other'] = 0
print(a[None])
print(a['other']) 
print(a)
#%%

mesh = QuadMesh(box=[0,1,0,2])
mesh.record('coarse')
mesh.cells.refine(new_label='fine')


Q0 = QuadFE(mesh.dim(), 'DQ0')

dhQ0 = DofHandler(mesh, Q0)
dhQ0.distribute_dofs()

phi_0 = Basis(dhQ0, subforest_flag='coarse')
phi_1 = Basis(dhQ0, subforest_flag='fine')

problem = [Form(trial=phi_0,test=phi_0), Form(trial=phi_1, test=phi_0)]
assembler = Assembler(problem, mesh=mesh)
for c in mesh.cells.get_leaves(subforest_flag='fine'):
    for form in assembler.problems:
        for f in form:
            shape_info = assembler.shape_info(c)
            xi_g, wi_g, phii, dofsi = assembler.shape
    shape_info = assembler.shape_info(c)
    xi_g, wi_g, phii, dofsi = assembler

# Try to mimic an assembly


cell = mesh.cells.get_leaves(subforest_flag='coarse')[0]
cell.info()

print(mesh.cells.is_contained_in('fine','coarse'))
# Evaluate phi_1 on cell
sub_dofs = []
for child in cell.get_leaves(flag='fine'):
    sub_dofs.extend(dhQ1.get_cell_dofs(child))

sub_dofs = list(set(sub_dofs))
print(sub_dofs)
x = dhQ1.get_dof_vertices(dofs=sub_dofs)
x_ref =cell.reference_map(x, mapsto='reference')
print(x_ref)


data = phi_1.eval(x_ref,cell)


fig = plt.figure()
ax = plt.axes(projection='3d')

print(data.shape, x.shape)
ax.scatter3D(x[:,0],x[:,1],data[:,0])
plt.show()
print(data)

phi_2 = Basis(dhQ1, subforest_flag='fine')
phi_0 = Basis(dhQ0)


k = Nodal(data=np.array([1,2,3,4]), basis=phi_0)


plot = Plot(quickview=False)
fig, ax = plt.subplots(1,1)
plot.contour(k, axis=ax)
plt.show()


problem = [Form(kernel=k, test=phi_1)]

assembler = Assembler(problem, mesh=mesh, n_gauss=(2,4))
assembler.assemble()
l = assembler.get_vector()
print('This should be 10', np.sum(l))
f = Nodal(f=lambda x: x[:,0]*(2-3*x[:,1]), basis=phi_1)



#print(l.dot(f.data()))
# First basis function on the [0,1]x[0,2]
g = lambda x,y: 0.5*(2-y)*(1-x)
for cell in mesh.cells.get_leaves():
    shape_info = assembler.shape_info(cell)

    xi_g, wi_g, phii, dofsi = assembler.shape_eval(shape_info, cell)
    print('phi_1(x,y)=', g(xi_g[cell][:,0],xi_g[cell][:,1]),'\n')
    print(phii[cell][phi_1])

    f_loc = problem[0].eval(cell, xi_g, wi_g, phii, dofsi)
    #print(f_loc,'\n')



plot = Plot(quickview=False)
fig, ax = plt.subplots(1,1)
ax = plot.mesh(mesh, axis=ax, show_axis=True)
plt.show()

#%% Mapping to a rectangular cell


# Define a QuadCell
v1, v2, v3, v4 = Vertex((1,1)), Vertex((2,1)), Vertex((2,5)), Vertex((1,5))
halfedges = [HalfEdge(v1,v2), HalfEdge(v2,v3), HalfEdge(v3,v4), HalfEdge(v4,v1)]
cell = QuadCell(halfedges)
cell.split()

plot = Plot(quickview=False)
fig, ax = plt.subplots(1,1)

# Plot cell and children
for c in cell.traverse():
    vertices = [v.coordinates() for v in c.get_vertices()]
    poly = plt.Polygon(vertices, fc=clrs.to_rgba('w'), edgecolor=(0,0,0,1))
    ax.add_patch(poly)


# Define Gauss Rule
child = cell.get_child(0)
gauss = GaussRule(4, shape='quadrilateral')

# Map rule to sw child
xg, wg, mg = gauss.mapped_rule(child, jac_p2r=True, hess_p2r=True)

plt.plot(xg[:,0],xg[:,1],'.k')

# Subdivide reference cell with same tree structure
r0, r1, r2, r3 = Vertex((0,0)), Vertex((1,0)), Vertex((1,1)), Vertex((0,1))
half_edges = [HalfEdge(r0,r1),
              HalfEdge(r1,r2),
              HalfEdge(r2,r3),
              HalfEdge(r3,r0)]

rcell = QuadCell(half_edges=half_edges)
rcell.split()

# Plot cell and children
for c in rcell.traverse():
    vertices = [v.coordinates() for v in c.get_vertices()]
    poly = plt.Polygon(vertices, fc=clrs.to_rgba('w'), edgecolor='r')
    ax.add_patch(poly)

ax.set_xlim(0,3)
ax.set_ylim(0,6)
plt.show()

#%% Mapping to a skew cell


#%%
# Define a mesh with two levels
mesh = QuadMesh()
mesh.record(0)
mesh.cells.refine(new_label=1)


# Elements
Q1 = QuadFE(2, 'Q1')  # linear
Q2 = QuadFE(2, 'Q2')  # quadratic

# DofHandlers
dQ1 = DofHandler(mesh, Q1)  #
dQ2 = DofHandler(mesh, Q2)

# Distribute the Dofs
for dQ in [dQ1,dQ2]: dQ.distribute_dofs()

# Basis functions
phi_1 = Basis(dQ1, subforest_flag=0)
phi_2 = Basis(dQ2, subforest_flag=0)

plot = Plot(quickview=False)

fig, ax = plt.subplots(1,2)
for i in range(2):
    ax[i] = plot.mesh(mesh, axis=ax[i],subforest_flag=i, dofhandler=dQ2,
      dofs=True, doflabels=True)
plt.show()

problems = [Form(trial=phi_1, test=phi_2)]
assembler = Assembler(problems, mesh=mesh)

"""
Iterate over fine leaves and get coarse scale dofs
"""


"""
mesh = Mesh1D(resolution=(2,))
element = QuadFE(1,'Q1')
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()
phif = Basis(dofhandler, 'u')
q = np.random.rand(phif.n_dofs())
qfn = Nodal(data=q, basis=phif)
kernel = Kernel(qfn)

mesh = Mesh1D(resolution=(2,))
dh = DofHandler(mesh, element)
dh.distribute_dofs()

phi = Basis(dh,'v')
problem = Form(kernel=kernel, test=phi, trial=phi)
assembler = Assembler(problem)

for cell in mesh.cells.get_leaves():
    shape_info = assembler.shape_info(cell)
    print(phif in shape_info[cell])
    xg, wg, basis, dofs = assembler.shape_eval(shape_info, cell)
    for problem in assembler.problems:
        for form in problem:
            # Get form
            #form.eval(cell, xg, wg, phi, dofs)

            # Determine regions over which form is defined
            regions = form.regions(cell)

            for region in regions:
                # Get Gauss points in region
                x = xg[region]

                print(basis[region][phif])
                #print(dofs[region])
                #
                # Compute kernel, weight by quadrature weights
                #
                Ker = kernel.eval(x=x, region=region, cell=cell,
                                  phi=basis[region], dofs=dofs[region])

"""