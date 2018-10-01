from mesh import QuadMesh
from fem import DofHandler, Assembler, QuadFE
from fem import Kernel, Form, Basis, Function
from fem import LinearSystem
from plot import Plot
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np

def bnd_fn_1(he):
    """
    
    """
    x1, y1 = he.base().coordinates()
    x2, y2 = he.head().coordinates()
    if np.abs(x1)<1e-9 and np.abs(x2)<1e-9:
        return True
    else:
        return False
    

def bnd_fn_2(he):
    """
    """
    x1, y1 = he.base().coordinates()
    x2, y2 = he.head().coordinates()
    if np.abs(x1-1)<1e-9 and np.abs(x2-1)<1e-9:
        return True
    else:
        return False
    
# 
# Define mesh
# 
mesh = QuadMesh(resolution=(2,2))
cell = mesh.cells.get_child(2)
cell.mark(1)
mesh.cells.refine(refinement_flag=1)

mesh.mark_boundary_vertices('D1',bnd_fn_1)
mesh.mark_boundary_vertices('D2',bnd_fn_2)
for he in mesh.half_edges.get_leaves():
    if he.is_marked('D1'):
        print('D1', [he.base().coordinates(),he.head().coordinates()])
    elif he.is_marked('D2'):
        print('D2', [he.base().coordinates(),he.head().coordinates()])
        
        
#
# Define Elements
# 
element = QuadFE(2, 'Q1')
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()
dofhandler.set_hanging_nodes()
hn = dofhandler.get_hanging_nodes()
print(hn=={})
#
# Plot Dofs
#
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plot = Plot(quickview=False)
plot.mesh(mesh, ax=ax, dofhandler=dofhandler, dofs=True)
#plt.show()

#
# Set up system
# 
ux = Basis(element, 'ux')
uy = Basis(element, 'uy')
u  = Basis(element, 'u')

one = Kernel(Function(1,'constant'))
problem = [Form(one, trial=ux, test=ux), Form(one, test=u)]

assembler = Assembler(problem, mesh)
assembler.assemble()

equation = LinearSystem(assembler, compressed=True)
equation.extract_dirichlet_nodes('D1')
equation.extract_dirichlet_nodes('D2')
print(equation.A().todense().shape)
u = equation.solve()

i_free = equation.free_indices()
print(sum(i_free))
for d in equation.dirichlet:
    print(int(d['mask']))
#print([d['mask'] for d in equation.dirichlet])

rows = assembler.af[0]['bilinear']['rows']
cols = assembler.af[0]['bilinear']['cols']
dofs = assembler.af[0]['bilinear']['row_dofs']
vals = assembler.af[0]['bilinear']['vals']

problem = assembler.af[0]
print(problem['linear'].keys())
A = sp.coo_matrix((vals,(rows,cols)))
Ad = A.todense()

#A, b = system.extract_dirichlet_nodes('D1')




x = np.arange(4)
y = np.array([1,3,6,2])
ii = np.array([0,1,1,1,0,3,2,2])
print(y[ii])
z = np.array([3,2])
dirichlet = np.zeros(4, dtype=np.bool)
for zi in z:
    dirichlet[y==zi] = True
print(x[dirichlet])
print(x[~dirichlet])
print(y)
print(dirichlet)
