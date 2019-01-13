from parameter_identification import elliptic_adjoint
from mesh import Mesh1D
from fem import DofHandler
from fem import Function
from fem import QuadFE
from fem import Kernel
from fem import Form
from fem import Basis
from fem import Assembler
from fem import LinearSystem
from plot import Plot
import numpy as np


#
# Define mesh
# 
mesh = Mesh1D(resolution=(2,))
element = QuadFE(1,'Q1')

u = Function(np.array([1,2,1]), 'nodal', mesh=mesh, element=element)
plot = Plot(3)
plot.line(u)

kernel = Kernel([u], dfdx=['fx'], F=lambda u: np.abs(u))
form = Form(kernel=kernel)
assembler = Assembler(form, mesh)
assembler.assemble()
cf = assembler.af[0]['constant']
#print(cf.get_matrix())


for cell in mesh.cells.get_leaves():
    for pivot in cell.get_vertices():
        nb = cell.get_neighbor(pivot)
        if nb is not None:
            print('pivot=',pivot.coordinates())
            print(u.eval(pivot, cell=cell))
            print(u.eval(pivot, cell=nb))
            