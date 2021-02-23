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
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def qfn(x):
    """
    Diffusion coefficient 
    
           { 0.5, if x in [0,0.2) u [0.4,0.6) u [0.8,1]
    q(x) = { 2  , if x in [0.2,0.4) u [0.6,0.8)
           { 0  , otherwise
    
    """
    #
    # Define regions
    # 
    r1 = np.logical_and(x>=0,   x<0.2)
    r2 = np.logical_and(x>=0.2, x<0.4)
    r3 = np.logical_and(x>=0.4, x<0.6)
    r4 = np.logical_and(x>=0.6, x<0.8)
    r5 = np.logical_and(x>=0.8, x<=1)
    
    #
    # Compute the union of regions on which q=0.5
    # 
    s1 = None
    for r in [r1, r3, r5]:
        if s1 is None:
            s1 = r 
        else:
            s1 = np.logical_or(s1, r)
    
    #
    # Compute union of regions on which q=2
    # 
    s2 = None
    for r in [r2, r4]:
        if s2 is None:
            s2 = r
        else:
            s2 = np.logical_or(s2, r)
    
    #
    # Define q
    # 
    q = np.zeros(x.shape)
    q[s1] = 0.5 # + 1/100*np.sin(100*np.pi*x[s1]) 
    q[s2] = 2

    return q

def dLdu(u):
    pass

def dLdq(q):
    pass


"""
Test 01 

Parameter idenfication of diffusion parameter 
"""
#
# Define Computational Mesh
# 
mesh = Mesh1D(resolution=(100,))
mesh.mark_region('left', lambda x: np.abs(x)<1e-9)
mesh.mark_region('right', lambda x: np.abs(x-1)<1e-9)

#
# Elements
#
Q0 = QuadFE(1, 'DQ0')
Q1 = QuadFE(1, 'Q1')


qe = Function(qfn, 'explicit', dim=1)
one  = Function(1, 'constant')
#
# Basis functions 
# 
u  = Basis(Q1, 'u')
ux = Basis(Q1, 'ux')
q  = Basis(Q0, 'q')

#
# Forms
# 
a_qe  = Form(kernel=Kernel(qe), trial=ux, test=ux)
a_one = Form(kernel=Kernel(one), trial=ux, test=ux)
m     = Form(kernel=Kernel(one), trial=u,  test=u)
L = Form(kernel=Kernel(one), test=u)

# 
# Problems
# 
problems = [[a_qe,L], [a_one], [m]]

#
# Assembly
# 
assembler = Assembler(problems, mesh)
assembler.assemble()

# =============================================================================
# Linear system for generating observations
# =============================================================================
system = LinearSystem(assembler,0)
b = system.get_rhs()

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
ue_data = ue.fn()
n_points = ue_data.shape[0]
ue_noisy = ue_data + 1e-2*np.random.rand(n_points)
ue.set_rules(ue_noisy)
#ue_x = ue.derivative((1,0))
#print(ue.dofhandler().element.element_type())
#plot = Plot(3)
#plot.line(ue)
#plot.line(qe, mesh=mesh)
#plot.line(ue_x)

# ==============================================================================
# Linear system for generating the inverse laplacian C
# ==============================================================================
D = LinearSystem(assembler, 1)

#
# Incorporate constraints
# 
D.add_dirichlet_constraint('left',0)
D.add_dirichlet_constraint('right',0)
D.set_constraint_relation()
D.constrain_matrix()
Dlt = D.get_matrix()
D.set_rhs(b)
D.constrain_rhs()
D.solve()
print(spla.spsolve(Dlt, b))

#
# Initial guesses
# 
u_iter = [ue]  # State
q_iter = [Function(1,'constant')]  # Parameter
p_iter = [Function(1,'constant')]  # Lagrange multiplier
r = 1e-9 
k_max = 1
n_max = 1
for n in range(n_max):
    #
    # Augmented Lagrangian Subproblem
    # 
    for k in range(k_max):
        #
        # Solve q = argmin L(q,u_{k-1}, lmd)
        # 
        
        # Assemble system
        uo = u_iter[-1]
        bu = Form(kernel=Kernel([uo],['ux']), trial=ux, test=q)
        
        assembler = Assembler([bu], mesh)
        assembler.assemble()
        Bu = assembler.af[0]['bilinear'].get_matrix()
        
        #Bu = LinearSystem(assembler, 0)
        
        #
        # Solve u_n^k = argmin L(q_n^k, u, lm_[n-1]) 
        # 
        
        # 
        aq = Form(kernel=Kernel(qe), test=ux, trial=ux)
        assembler = Assembler([aq], mesh)
        assembler.assemble()
        Aq = assembler.af[0]['bilinear'].get_matrix()
        
        Aq = LinearSystem(assembler, 0)
         
        # Add Dirichlet constraints
        Aq.add_dirichlet_constraint('left',0)
        Aq.add_dirichlet_constraint('right',0)
        Aq.set_constraint_relation()
        Aq.constrain_matrix()
        AQ = Aq.get_matrix().toarray()
        plt.spy(Aq.get_matrix(), markersize=1)
        plt.show()
    

        
        
        
    #
    # Update p 
    # 
    #p0 = p0 + r*eqn
    
    #
    # Record updates
    # 
    #u_iter.append(u0)
    #q_iter.append(q0)
    #p_iter.append(p0)

'''    
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

system = LinearSystem(assembler,0)

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



'''