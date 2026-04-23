from operator import not_

from function import Constant, Explicit, Nodal
from fem import DofHandler, Basis, QuadFE
from assembler import Form, Kernel, Assembler
from mesh import Mesh1D, QuadMesh
from plot import Plot

import numpy as np
import matplotlib.pyplot as plt

import plot

def flux(q,ux,uy,region=None):
    n = region.normal()
    return q*(ux*n[0]+uy*n[1])
    
#
# Test Boundary Assembly Over a Sub-Region
# 
mesh = QuadMesh(box=[0,1,0,1], resolution=(2,2))
outflow_indicator = lambda x,y: np.abs(x-1)<1e-6 and (0.5<=y) and (y<=1)
mesh.mark_region('outflow_edge', outflow_indicator, 
                 entity_type='half_edge', on_boundary=True)
mesh.mark_region('outflow_cell', outflow_indicator, 
                 entity_type='cell', strict_containment=False, on_boundary=True)

Q1 = QuadFE(mesh.dim(), 'Q1')
dhQ1 = DofHandler(mesh, Q1)
dhQ1.distribute_dofs()
v = Basis(dhQ1, 'v')
vx = Basis(dhQ1, 'vx')
vy = Basis(dhQ1, 'vy')

u = Nodal(f=lambda x: 1 - x[:,0], basis=v)

k_flux = Kernel(f=[Constant(1),u,u], derivatives=[(0,),(1,0), (1,1)], F=flux)
f_flux = Form(kernel=k_flux, flag='outflow_edge', dmu='ds')
assembler = Assembler(f_flux, mesh)

# Iterate over cells at the outflow boundary
for ci in mesh.cells.get_leaves(flag='outflow_cell'):
    for edge in ci.get_half_edges():
        print(edge.is_marked('outflow_edge'))
    #shape_info = assembler.shape_info(ci)
    #print(f"Cell {ci} shape info: {shape_info}")