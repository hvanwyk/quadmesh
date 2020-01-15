from assembler import Form, Kernel, Assembler
from fem import QuadFE, Basis, DofHandler
from function import Nodal, Explicit, Constant
from mesh import Mesh1D, Interval
from plot import Plot
from solver import LinearSystem

import numpy as np
"""
Solve the problem 

    -eps*u_xx + a*u_x = 0
    u(0) = 0, u(1) = 1

by stabilized finite elements, i.e. use test functions of the form 

    w = v + p, where p = 0.5*h*xi*v_x
    
and xi is given by the `upwind scheme' formula. The corresponding weak forms 
are given by

    (eps*u_x,v_x)+(a*u_x,v_x) + (-eps*u_xx + a*u_x, p) = 0
    
"""
class SUPGKernel(Kernel):
    """
    Define a Kernel for assembling a SUPG stabilizer
    
         (-eps*u_xx,p) + (a*u_x,p) 
     
     where 
     
         p = 0.5*h*xi*v_x
         
    The resulting kernels are 
    
        -eps*0.5*h*xi 
        
    or 
    
        a*0.5*h*xi
    """
    def __init__(self,f,u,eps,upwind_type='classical'):
        """
        Constructor
        
        Inputs:
        
            f: Map, explicit kernel function
            
            u: Map, velocity function
            
            eps: double >0, diffusivity
            
            upwind_scheme: str, type of upwind scheme used 
        """
        Kernel.__init__(self,f)
        self.__upwind_type = upwind_type
        self.__vel = u
        self.__eps = eps
        
        
    def eval(self, x, region=None):
        """
        Modified evaluation
        """
        assert isinstance(region,Interval), \
        'Integration region should be an interval.'
        
        u = self.__vel.eval(x)  # velocity
        Pe = u/eps
        h = region.length() 
        a = 0.5*h*Pe
        xi = upwind_indicator(a,self.__upwind_type)
        
        # Evaluate function f
        f_vals = []
        for f in self.f():
            f_vals.append(f.eval(x=x))
        Fv = self.F()(*f_vals)
        
        # Combine function with kernel estimate
        return 0.5*Fv*h*xi


def upwind_indicator(a,name):
    """
    Returns the indicator xi = xi(Pe,h)
    """ 
    if name=='classical':
        return np.sign(a)
    elif name=='ilin':
        return 1/tanh(a) - 1/a
    elif name=='double_asymptotic':
        if np.abs(a)<=3:
            return a/3
        else:
            return np.sign(a)
    elif name=='critical_approximation':
        if a <= -1:
            return -1-1/a 
        elif np.abs(a)<=1:
            return 0
        elif a >= 1:
            return 1-1/a


# Parameters 
eps = 1e-3
a = 1

# Computational mesh
mesh = Mesh1D(resolution=(200,))
mesh.mark_region('left', lambda x: np.abs(x)<1e-9, entity_type='vertex')
mesh.mark_region('right', lambda x: np.abs(x-1)<1e-9, entity_type='vertex')

# Element
element = QuadFE(1,'Q1')
dofhandler = DofHandler(mesh,element)
dofhandler.distribute_dofs()

# Kernels
k_eps = SUPGKernel(Constant(-eps),Constant(a),eps)
k_a  = SUPGKernel(Constant(a),Constant(a),eps)


# Forms 
u = Basis(dofhandler,'u')
ux = Basis(dofhandler,'ux')
uxx = Basis(dofhandler, 'uxx')

problem = [Form(eps,test=ux,trial=ux),
           Form(1,trial=ux,test=u), 
           Form(0,test=u)]

problem = [Form(eps,test=ux,trial=ux),
           Form(1,trial=ux,test=u), 
           Form(0,test=u),
           Form(k_eps,trial=uxx,test=ux),
           Form(k_a,trial=ux,test=ux)]
        
# Assemble forms
assembler = Assembler(problem, mesh)
assembler.assemble()
A = assembler.af[0]['bilinear'].get_matrix()
b = assembler.af[0]['linear'].get_matrix()

# Define linear system
system = LinearSystem(u)
system.set_matrix(A)
system.set_rhs(b)

 # Add Dirichlet constraints
system.add_dirichlet_constraint('left', 0)
system.add_dirichlet_constraint('right', 1)
system.set_constraint_relation()

system.solve_system()
u = system.get_solution()

plot = Plot()
plot.line(u)