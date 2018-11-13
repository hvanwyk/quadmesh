from fem import Function
from fem import QuadFE
from fem import DofHandler
from fem import Kernel
from fem import Basis
from fem import Form
from fem import Assembler
from fem import LinearSystem
from plot import Plot
from mesh import convert_to_array
from mesh import QuadMesh
from mesh import Mesh1D
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy
import numpy as np



#
# Define mesh
#  
mesh = QuadMesh(resolution=(2,1))
mesh.cells.get_child(1).mark(1)
mesh.cells.refine(refinement_flag=1)
    
# 
# Define element
# 
Q1 = QuadFE(2,'Q1')
dofhandler = DofHandler(mesh, Q1)
dofhandler.distribute_dofs()
dofhandler.set_hanging_nodes()

print(dofhandler.constraints)
print(dofhandler.get_hanging_nodes())
#
# Define exact solution
# 
ue = Function(lambda x,y: x, 'nodal', dofhandler=dofhandler)

#
# Assemble weak form
# 
one = Function(1, 'constant')
zero = Function(0, 'constant')
u = Basis(Q1, 'u')
ux = Basis(Q1, 'ux')
uy = Basis(Q1, 'uy')

ax = Form(kernel=Kernel(one), trial=ux, test=ux)
ay = Form(kernel=Kernel(one), trial=uy, test=uy)
L = Form(kernel=Kernel(zero), test=u)

assembler = Assembler([ax, ay, L], mesh)
assembler.assemble()

rows = assembler.af[0]['bilinear']['rows']
cols = assembler.af[0]['bilinear']['cols']
vals = assembler.af[0]['bilinear']['vals']
dofs = assembler.af[0]['bilinear']['row_dofs']

b = assembler.af[0]['linear']['vals']

n = len(dofs)

#
# Check bilinear form by integrating
# 
A = sp.coo_matrix((vals, (rows, cols)))

assert np.allclose(np.sum(np.ones(n)*A.dot(np.ones(n))),0)

u = Function(lambda x,y: 2*x+x*y-y, 'nodal', dofhandler=dofhandler)
v = Function(lambda x,y: -x+2*y, 'nodal', dofhandler=dofhandler)

assert np.allclose(np.sum(u.fn()*A.dot(v.fn())), -3.5)


f = Function(lambda x,y: x + y, 'nodal', dofhandler=dofhandler)
fx = f.fn()
uex = ue.fn()
assert np.allclose(np.sum(uex*A.dot(fx)),1)


print(12*A.todense())

print('look here')
AD = A.todense()
p = np.zeros(uex.shape)
p[:] = AD[2][:] + 0.5*AD[8][:]


print(p)
print(uex.dot(p))

#
# Form linear system
# 
system = LinearSystem(assembler, compressed=False)
#x = system.dofhandler.get_dof_vertices(dofs)
#x = convert_to_array(x)


print(A.dot(ue.fn())-b)


system.extract_hanging_nodes()

print('*'*60)
print(uex)
print(system.A().dot(uex)-system.b())
print('*'*60)


print('marking dirichlet vertices')

# Mark Dirichlet Regions
f_left = lambda x, dummy: np.abs(x)<1e-9
f_right = lambda x, dummy: np.abs(x-1)<1e-9
f_top = lambda dummy, y: np.abs(y-1)<1e-9
f_bottom = lambda dummy, y: np.abs(y)<1e-9


mesh.mark_region('left', f_left, on_boundary=True)
mesh.mark_region('right', f_right, on_boundary=True)
mesh.mark_region('top', f_top, on_boundary=True)
mesh.mark_region('bottom', f_bottom, on_boundary=True)

# Getting mesh 
print(system.has_hanging_nodes())
#print(system.hanging_nodes)

#print('*'*60)
#print(uex)
#print(system.A().dot(uex)-system.b())
#print('*'*60)
    
# Extract Dirichlet conditions 
system.extract_dirichlet_nodes('left', 0)


#print(6*system.b())
#print(6*system.A().todense())

system.extract_dirichlet_nodes('right',1)
print(system.dofhandler.constraints)

system.compress()


#system.extract_dirichlet_nodes('top', ue)
#system.extract_dirichlet_nodes('bottom', ue)

print(12*system.b())
print(12*system.A().todense())

print('*'*60)
print(uex)
print(system.A().dot(uex)-system.b())
print(np.linalg.det(system.A().todense()))
print('*'*60)


A = system.A().todense()
ua = np.linalg.solve(A,system.b())


print(ua)

#print('*'*60)
#print(uex)
#print(system.A().dot(uex)-system.b())
#print('*'*60)

print(12*system.A().todense())
print(12*system.b())

#plt.imshow(A.todense()) 
#plt.show()
A = system.A().todense()
print(np.linalg.det(A))

#plot = Plot(quickview=False)
system.solve()
u = system.sol(as_function=True)

plot = Plot(quickview=False)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax = plot.wire(u, axis=ax)
#ax = plot.mesh(mesh, axis=ax, dofhandler=system.dofhandler, dofs=True)
plt.show()
"""
DQ0 = QuadFE(2,'DQ0')
f = Function(lambda x, y: y*x**2, 'nodal', mesh=mesh, element=DQ0)
plot.wire(f, mesh)
"""