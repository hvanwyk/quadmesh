from mesh import Mesh1D
from fem import DofHandler
from function import Function
from fem import QuadFE
from assembler import Kernel
from assembler import Form
from fem import Basis
from assembler import Assembler
from solver import LinearSystem
from plot import Plot
import numpy as np
import matplotlib.pyplot as plt

def qfn(x):
    """
    Diffusion coefficient 
    """
    q = 1 + 0.2*np.cos(2*np.pi*x) + 0.1*np.cos(3*np.pi*x) + 0.1*np.cos(4*np.pi*x)
    return q

"""
Test 02

Parameter idenfication of continuous diffusion parameter 
"""
#
# Define Computational Mesh
# 
mesh = Mesh1D(resolution=(200,))
mesh.mark_region('left', lambda x: np.abs(x)<1e-9)
mesh.mark_region('right', lambda x: np.abs(x-1)<1e-9)

#
# Elements
#
Q0 = QuadFE(1, 'DQ0')
Q1 = QuadFE(1, 'Q1')

#
# Exact diffusion coefficient
# 
qe = Function(qfn, 'explicit', dim=1)
one  = Function(1, 'constant')
k1 = 1e-9
k2 = 1000


#
# Basis functions 
# 
u  = Basis(Q1, 'u')
ux = Basis(Q1, 'ux')
q  = Basis(Q1, 'q')

#
# Forms
# 
a_qe  = Form(kernel=Kernel(qe), trial=ux, test=ux)
a_one = Form(kernel=Kernel(one), trial=ux, test=ux)

L = Form(kernel=Kernel(one), test=u)

# 
# Problems
# 
problems = [[a_qe,L], [a_one]]

#
# Assembly
# 
assembler = Assembler(problems, mesh)
assembler.assemble()

# =============================================================================
# Linear system for generating observations
# =============================================================================
system = LinearSystem(assembler,0)
f = system.get_rhs()

#
# Incorporate constraints
# 
system.add_dirichlet_constraint('left',0)
system.add_dirichlet_constraint('right',0)

#
# Compute model output
# 
system.solve_system()
ue = system.get_solution(as_function=True)
ue_x = ue.derivative((1,0))

plot = Plot()
plot.line(qe, mesh=mesh)

# ==============================================================================
# Linear system for generating the inverse laplacian C
# ==============================================================================
C = LinearSystem(assembler, 1)

#
# Incorporate constraints
# 
C.add_dirichlet_constraint('left',0)
C.add_dirichlet_constraint('right',0)


#
# Compute penalty coefficients gamma and r
# 
r = 1e-9
gamma = 1
beta = 1e-9

k_max = 20
n_max = 20

#
# Initial guesses
# 
u_iter = [ue]  # State
q_iter = [Function(1,'constant')]  # Parameter
p_iter = [Function(0,'constant')]  # Lagrange multiplier
for n in range(n_max):
    #
    # Augmented Lagrangian Subproblem
    # 
    for k in range(k_max):
        #
        # Solve q = argmin L(q,u_{k-1}, lmd)
        # 
        
        # Assemble Bu
        u = u_iter[-1]
        bu = Form(kernel=Kernel([u],['ux']), trial=ux, test=q)
        assembler = Assembler([bu], mesh)
        assembler.assemble()
        Bu = assembler.af[0]['bilinear'].get_matrix()
        
        # Compute B        
        # Incorporate boundary conditions
        
        
        
        #
        # Solve u_n^k = argmin L(q_n^k, u, lm_[n-1]) 
        # 
        aq = Form(kernel=Kernel(qo), test=ux, trial=ux)
        pass
    
    #
    # Update p 
    # 
    #p0 = p0 + r*eqn
    
    #
    # Record updates
    # 
    u_iter.append(u0)
    q_iter.append(q0)
    p_iter.append(p0)
    
#
# Define elements
#
Q0 = QuadFE(1, 'DQ0')
Q1 = QuadFE(1, 'Q1')

#
# Forms
# 
q = Function(qfn, 'nodal', mesh=mesh, element=Q0)
#q = Function(1, 'constant')
zero = Function(0, 'constant')
one = Function(1, 'constant')
#
# Trial and test functions
# 
u = Basis(Q1,'u')
ux = Basis(Q1,'ux')

#
#  
#
a = Form(kernel=Kernel(q), trial=ux, test=ux)
L = Form(kernel=Kernel(one), test=u)
problem = [a,L]

assembler = Assembler(problem, mesh)
assembler.assemble()

system = LinearSystem(assembler)

# Boundary functions 
bm_left = lambda x: np.abs(x)<1e-9
bm_rght = lambda x: np.abs(x-1)<1e-9

# Mark boundary regions
mesh.mark_region('left', bm_left, on_boundary=True)
mesh.mark_region('right',bm_rght, on_boundary=True)

# Add Dirichlet constraints
system.add_dirichlet_constraint('left',0)
system.add_dirichlet_constraint('right',0)

system.set_constraint_matrix()

system.incorporate_constraints()
system.solve()
system.resolve_constraints()
u = system.sol(as_function=True)

plot = Plot()
plot.line(u)

 
 
#
# Define the exact parameter as a piecewise constant function
# 
#qex_vec = np.array([0.5,2,0.5,2,0.5])
#qex_fn = Function(qex_vec, 'nodal', mesh=mesh, element=Q0, subforest_flag=0)

#
# Right hand side 
# 
#f = Function(1, 'constant')



