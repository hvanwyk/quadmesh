from mesh import QuadMesh, Mesh1D, QuadCell
from fem import QuadFE, DofHandler, Basis
from assembler import Assembler, Form, Kernel
from function import Constant, Nodal
import numpy as np


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

