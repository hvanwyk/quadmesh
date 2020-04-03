from assembler import Form, Assembler
from mesh import QuadMesh
from fem import DofHandler, QuadFE, Basis
from function import Nodal
from diagnostics import Verbose
import time 
import numpy as np

comment = Verbose()

# Mesh 
mesh = QuadMesh(resolution=(100,100))

comment.tic('marking dirichlet boundary')
# Mark Dirichlet Region
bm_left = lambda x,dummy: np.abs(x)<1e-9
bm_right = lambda x, dummy: np.abs(1-x)<1e-9
mesh.mark_region('left', bm_left, entity_type='half_edge')
mesh.mark_region('right', bm_right, entity_type='half_edge')
comment.toc()

# Element 
element = QuadFE(2,'Q1')

# Dofhandler
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()

comment.tic('Getting hanging nodes')
hn = dofhandler.get_hanging_nodes()
comment.toc()


comment.tic('Getting dirichlet nodes')
dirichlet_dofs = dofhandler.get_region_dofs(entity_type='half_edge', 
                                            entity_flag='left', 
                                            interior=False, 
                                            on_boundary=False, \
                                            subforest_flag=None) 
comment.toc()


x = dofhandler.get_dof_vertices()

# Basis
basis = Basis(dofhandler)

f = lambda x:x[:,0]*(1-x[:,0])*x[:,1]*(1-x[:,1])
fNodal = Nodal(f=f, basis=basis)

comment.tic('evaluating nodal function at dirichlet dofs')
idx = fNodal.dof2idx(dirichlet_dofs)
dir_vals = fNodal.data()[idx,:]
print(dir_vals.shape)
comment.toc()


problem = [Form(fNodal, test=basis),
           Form(1, test=basis, trial=basis)]
"""
assembler = Assembler(problem, mesh)
assembler.assemble(clear_cell_data=False)

a_forms = assembler.af[0]
bilinear = a_forms['bilinear']
linear = a_forms['linear']
print(bilinear.cell_address)
print(linear.cell_address)

cols, rows, vals = [], [], []

"""
