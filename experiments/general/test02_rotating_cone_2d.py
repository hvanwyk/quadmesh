from assembler import Assembler
from assembler import Form
from assembler import Kernel
from fem import DofHandler
from fem import QuadFE
from fem import Basis
from function import Explicit
from mesh import Mesh1D
from mesh import QuadMesh
from mesh import Vertex
from mesh import HalfEdge
from solver import LinearSystem as LS
from plot import Plot
import numpy as np

    
# =============================================================================
# Computational mesh
# =============================================================================
mesh = QuadMesh(box=[-0.5,0.5,-0.5,0.5], resolution=(20,20))


# Mark slit region
slit = HalfEdge(Vertex((0,0)), Vertex((0,-0.5)))
sf = lambda x,y: slit.contains_points(np.array([x,y]))[0]
mesh.mark_region('slit',sf, entity_type='half_edge')

# Mark perimeter
tol = 1e-9
pf = lambda x,y: np.abs(x+0.5)<tol or np.abs(x-0.5)<tol or \
                 np.abs(y+0.5)<tol or np.abs(y-0.5)<tol
mesh.mark_region('perimeter', pf, entity_type='half_edge')

    
# Get rid of neighbors of half-edges on slit
for he in mesh.half_edges.get_leaves('slit'):
    if he.unit_normal()[0] < 0:
        he.mark('B')    
mesh.tear_region('B')


# =============================================================================
# Functions
# =============================================================================
Q1 = QuadFE(2, 'Q1')
dofhandler = DofHandler(mesh, Q1)
dofhandler.distribute_dofs()

u = Basis(dofhandler, 'u')
ux = Basis(dofhandler, 'ux')
uy = Basis(dofhandler, 'uy')

epsilon = 1e-6
vx = Explicit(f=lambda x: -x[:,1], dim=2)
vy = Explicit(f=lambda x:  x[:,0], dim=2)

uB = Explicit(f=lambda x: np.cos(2*np.pi*(x[:,1]+0.25)), dim=2)
problem = [Form(epsilon, trial=ux, test=ux),
           Form(epsilon, trial=uy, test=uy),
           Form(vx, trial=ux, test=u),
           Form(vy, trial=uy, test=u),
           Form(0, test=u)]

print('assembling', end=' ')
assembler = Assembler(problem, mesh)
assembler.assemble()
print('done')

print('solving',  end=' ')
A = assembler.get_matrix()
b = np.zeros(u.n_dofs())
system = LS(u, A=A, b=b)
system.add_dirichlet_constraint('B',uB)
system.add_dirichlet_constraint('perimeter',0)

system.solve_system()
ua = system.get_solution()
print('done')


plot = Plot()
plot.wire(ua)    
