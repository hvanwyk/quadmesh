from mesh import QuadMesh, Vertex, HalfEdge, QuadCell, Mesh1D, Interval, Tree
from fem import DofHandler, QuadFE, GaussRule, Function
from mesh import convert_to_array
from plot import Plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


mtags = Tree(regular=False)
mesh = Mesh1D(resolution=(1,))

flag = tuple(mtags.get_node_address())
mesh.cells.record(flag)

mtags.add_child()
new_flag = tuple(mtags.get_child(0).get_node_address())
mesh.cells.refine(subforest_flag=flag, \
                  new_label=new_flag)

for leaf in mesh.cells.get_leaves(subforest_flag=flag):
    leaf.info()
    
    
print('=='*20)

for leaf in mesh.cells.get_leaves(subforest_flag=new_flag):
    leaf.info()


element = QuadFE(1, 'Q2')
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()
dofhandler.set_dof_vertices()

dofs = dofhandler.get_global_dofs(subforest_flag=flag)
print(dofs)

dofs = dofhandler.get_global_dofs(subforest_flag=new_flag)
print(dofs)

dv = dofhandler.get_dof_vertices(dofs)
print(dv)

