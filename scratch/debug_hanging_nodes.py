from mesh import QuadMesh
from fem import QuadFE
from fem import DofHandler
from fem import Form
from fem import Basis
from fem import Kernel
from fem import GaussRule
from fem import Assembler
from fem import Function
from plot import Plot
from scipy import sparse as sp
import numpy as np

#
# Define Mesh
# 
mesh = QuadMesh(resolution=(2,1))
mesh.cells.get_child(1).mark('1')
mesh.cells.refine(refinement_flag='1')


#
# Define element
# 
Q1 = QuadFE(2,'Q1')

#
# Basis Functions 
#
u = Basis(Q1, 'u')
ux = Basis(Q1, 'ux')
uy = Basis(Q1, 'uy')

rule = GaussRule(9, Q1)

ue = Function(lambda x,dummy: x, 'nodal', mesh=mesh, element=Q1)
zero = Function(0, 'constant')

ax = Form(trial=ux, test=ux)
ay = Form(trial=uy, test=uy)
L = Form(kernel=Kernel(zero), test=u)
assembler = Assembler([ax,ay,L], mesh)
for dofhandler in assembler.dofhandlers.values():
    dofhandler.set_hanging_nodes()

plot = Plot()
plot.mesh(mesh, dofhandler=dofhandler, dofs=True)

n_equations = dofhandler.n_dofs()
rows = []
cols = []
vals = []
b = np.zeros(n_equations)

for cell in mesh.cells.get_leaves():
    #
    # Get global cell dofs for each element type  
    #
    cell_dofs = assembler.cell_dofs(cell)
    #
    # Determine what shape functions and Gauss rules to compute on current cells
    # 
    shape_info = assembler.shape_info(cell)
    
    # 
    # Compute Gauss nodes and weights on cell
    # 
    xg, wg = assembler.gauss_rules(shape_info)

    #
    # Compute shape functions on cell
    #  
    phi = assembler.shape_eval(shape_info, xg, cell)
    
    problem = assembler.problems[0]
    for form in problem:
        form_loc = form.eval(cell, xg, wg, phi, cell_dofs)
        if form.type=='bilinear':
            # Assemble bilinear form
            R,C = np.meshgrid(cell_dofs['Q1'], cell_dofs['Q1'])
            rows.extend(list(R.ravel()))
            cols.extend(list(C.ravel()))
            vals.extend(list(form_loc.ravel(order='F')))
        elif form.type=='linear':
            for i in range(4):
                dof = cell_dofs['Q1'][i]
                b[dof] += form_loc[i]
    
    A = sp.coo_matrix((vals,(rows,cols)))
    A = A.toarray()

#hn = dofhandler.get_hanging_nodes()
# Manually impose dirichlet conditions 
# At x=0
A[0,:] = 0
A[:,0] = 0
A[0,0] = 1
b[0] = 0 

A[3,:] = 0 
A[:,3] = 0
A[3,3] = 1
b[3] = 0

# At x = 1
one = np.zeros(n_equations)
one[[4,5,9]] = 1
b -= A.dot(one)
b[[4,5,9]] = 1

A[[4,5,9],:] = 0 
A[:,[4,5,9]] = 0
A[4,4] = 1
A[5,5] = 1
A[9,9] = 1 


A[:,1] += A[:,8]*0.5 
A[:,2] += A[:,8]*0.5 
A = A[:,[0,1,2,3,4,5,6,7,9,10]]

A[1,:] += A[8,:]*0.5
A[2,:] += A[8,:]*0.5
A = A[[0,1,2,3,4,5,6,7,9,10],:]

b[1] += b[8]*0.5
b[2] += b[8]*0.5
b = b[[0,1,2,3,4,5,6,7,9,10]]
print(np.linalg.matrix_rank(A))

#print(hn[8])
ue_vec = ue.fn()  
print(A.dot(ue_vec[[0,1,2,3,4,5,6,7,9,10]])-b)
ua = np.zeros(n_equations)

ua[[0,1,2,3,4,5,6,7,9,10]] = np.linalg.solve(A,b)
ua[8] = 0.5*ua[1] + 0.5*ua[2]

ua_fn = Function(ua, 'nodal', dofhandler=dofhandler)
plot = Plot()
plot.wire(ua_fn)
    
"""

shape_info = assembler.shape_info(cell)
xg, wg = assembler.gauss_rules(shape_info)
phi = assembler.shape_eval(shape_info, xg, cell)
b_loc, rows, cols = form.eval(cell, xg, wg, phi)
B = np.empty((4,4))
B[np.ix_(rows,cols)] = b_loc
print(B)
print(rows)
print(cols)
B = b_loc.reshape((4,4),order='F')
print(l2g)
test_etype = test_element.element_type()
trial_etype = trial_element.element_type()
test_l2g = np.array(list(l2g[test_etype].values()))
trial_l2g = np.array(list(l2g[trial_etype].values()))

print(test_l2g)
print(trial_l2g)
A = sp.coo_matrix((b_loc, (rows,cols)))
A = A.todense()
print(test_l2g.dot(A).dot(trial_l2g.T))
"""