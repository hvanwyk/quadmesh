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
mesh = Mesh1D(resolution=(10,))

for etype in ['Q1','Q2','Q3']:
    element = QuadFE(1,etype)

    #
    # Exact solution 
    # 
    ue = Function(lambda x: x, 'nodal', mesh=mesh, element=element) 
    
    #
    # Define Basis functions 
    #  
    u = Basis(element, 'u')
    ux = Basis(element, 'ux')
    
    #
    # Define bilinear form
    #
    one = Function(1, 'constant')
    zero = Function(0, 'constant')
    a = Form(kernel=Kernel(one), trial=ux, test=ux)
    L = Form(kernel=Kernel(zero), test=u)
    problem = [a,L]
    
    #
    # Assemble 
    # 
    assembler = Assembler(problem, mesh)
    assembler.assemble()
    
    #
    # Form linear system
    # 
    system = LinearSystem(assembler, 0)
    
    #
    # Dirichlet conditions 
    # 
    
    # Boundary functions 
    bm_left = lambda x: np.abs(x)<1e-9
    bm_rght = lambda x: np.abs(x-1)<1e-9
    
    # Mark boundary regions
    mesh.mark_region('left', bm_left, on_boundary=True)
    mesh.mark_region('right',bm_rght, on_boundary=True)
    
    # Add Dirichlet constraints
    system.add_dirichlet_constraint('left', ue)
    system.add_dirichlet_constraint('right', ue)
    
    #
    # Solve system
    # 
    system.solve_system()
    
    #
    # Return solution
    # 
    ua = system.get_solution()
    
    # Compare with exact solution
    assert np.allclose(ua.fn(), ue.fn()), 'not close'
    
