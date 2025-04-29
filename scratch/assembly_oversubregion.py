import sys
sys.path.append('/home/hans-werner/git/quadmesh/src/')

from mesh import QuadMesh
from fem import QuadFE, Basis, DofHandler
from function import Explicit, Nodal, Constant
from assembler import Assembler, Form, Kernel
from plot import Plot
from diagnostics import Verbose
import matplotlib.pyplot as plt

# Initialize plot
plot = Plot(quickview=False)
comment = Verbose()

# Computational domain
domain = [-1,1,-1,1]
mesh  = QuadMesh(box=domain, resolution=(2,2))

# Integration region
mesh.mark_region('Omega', lambda x,y: 0<=x and 0<=y, entity_type='cell')

# Degrees of freedom
#
element = QuadFE(mesh.dim(), 'Q1')
dh = DofHandler(mesh,element)
dh.distribute_dofs()
v = Basis(dh, 'v')


fig, ax = plt.subplots()
ax = plot.mesh(mesh,axis=ax, dofhandler=dh, dofs=True, doflabels=True, regions=[('Omega','cell')])
#plt.show()

# Assembler
problem = Form(Constant(1), trial=v, test=v, flag='Omega')
assembler = Assembler(problem, mesh)
for cell in mesh.cells.get_leaves():
    #print(problem.regions(cell))
    print(assembler.shape_eval(cell))
    dofs_tst = problem.test.dofs(cell)
    dofs_trl = problem.trial.dofs(cell)
    print([dofs_tst, dofs_trl])
    #print('test dofs', problem.test.dofs(cell))
    #print('trial dofs', problem.trial.dofs(cell))


assembler.assemble()
A = assembler.get_matrix()
#print(A)
