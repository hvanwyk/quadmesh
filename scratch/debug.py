from function import Constant, Explicit, Nodal
from fem import DofHandler, Basis, QuadFE
from assembler import Form, Kernel, Assembler
from mesh import Mesh1D, QuadMesh
from plot import Plot

import numpy as np
import matplotlib.pyplot as plt

import plot


mesh = QuadMesh(box=[0,1,0,1], resolution=(2,2))
mesh.record(0)
mesh.cells.refine()
mesh.record(1)

DQ0 = QuadFE(mesh.dim(), 'DQ0')
dhQ0 = DofHandler(mesh, DQ0)
dhQ0.distribute_dofs()

Q1 = QuadFE(mesh.dim(), 'Q1')
dhQ1 = DofHandler(mesh, Q1)
dhQ1.distribute_dofs()

v_x = Basis(dhQ1,'vx')
v_y = Basis(dhQ1,'vy')

# Basis functions at each level
w = [Basis(dhQ0, subforest_flag=l) for l in range(2)]

q0 = Nodal(data=1+np.arange(dhQ0.n_dofs(subforest_flag=0)), basis=w[0])
q1 = Nodal(data=1+np.arange(dhQ0.n_dofs(subforest_flag=1)), basis=w[1])

"""plot = Plot(quickview=False)
fig, ax = plt.subplots(1,2)
ax[0] = plot.contour(q0, axis=ax[0])
ax[1] = plot.contour(q1, axis=ax[1])
plt.show()
"""
k0 = Kernel(q0)
k1 = Kernel(q1)

#print('k0 basis mesh flag:', k0.basis()[0].subforest_flag())
#print('k1 basis mesh flag:', k1.basis()[0].subforest_flag())

assembler = Assembler(Form(kernel=k0, test=v_x, trial=v_y), mesh, subforest_flag=1, n_gauss=(4,4))
i = 0
for ci in mesh.cells.get_leaves(subforest_flag=1):
    #print(i)
    xg, wg, phi, dofs = assembler.shape_eval(ci)
    if i == 8:
        #print('x:', x)
        #print('w:', w)
        #print('phi:', phi)
        #print('dofs:', dofs)
        form = assembler.problems[0][0]
        #print(form)
        region = form.regions(ci)[0]
        #print(region.bounding_box())
        x = xg[region]
        ker = form.kernel
        k0_loc = ker.eval(x, phi=phi[region], region=region, cell=ci, dofs=dofs[region])
        print('k0_loc:', k0_loc)
        
        q0_loc = q0.eval(x)
        print('q0_loc:', q0_loc)
        
        f_loc = form.eval(ci, xg, wg, phi=phi, dofs=dofs)
        print('form_loc:', f_loc)
        #print('xg:', xg)

        #form_loc = form.eval(ci, x, w, phi, dofs)
        #print('form_loc:', form_loc)

    
    i += 1

assembler.assemble()
A = assembler.get_matrix()
print('A:', A)