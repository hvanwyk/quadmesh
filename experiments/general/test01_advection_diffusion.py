#
# Internal
# 
from assembler import Kernel
from assembler import Form
from assembler import Assembler

from fem import QuadFE
from fem import DofHandler
from fem import Basis

from function import Explicit
from function import Constant
from function import Nodal

from mesh import QuadMesh
from mesh import Mesh1D

from plot import Plot

from solver import LinearSystem

#
# External
# 
import numpy as np
import matplotlib.pyplot as plt

"""
Error analysis for advection diffusion equation

 -epsilon*(d^2u_dx^2 + d^2u_dy^2) + vx*du_dx + vy*du_dy = f 

subject to dirichlet boundary conditions.
"""
#
# Define functions
# 


# Exact solution
ue = Explicit(f = lambda x: np.sin(x[:,0])*np.sin(x[:,1]), n_variables=1, dim=2)

# Forcing function
ffn = lambda x: 2*eps*np.sin(x[:,0])*np.sin(x[:,1]) + \
                    x[:,0]*np.cos(x[:,0])*np.sin(x[:,1]) + \
                    x[:,1]*np.sin(x[:,0])*np.cos(x[:,1])
f = Explicit(ffn, dim=2)

# Velocity function
vx = Explicit(lambda x: x[:,0], dim=2) 
vy = Explicit(lambda x: x[:,1], dim=2)


errors = {}
for resolution in [(5,5), (10,10), (20,20), (40,40)]:
    #
    # Define new mesh
    # 
    mesh = QuadMesh(resolution=resolution)
    
    errors[resolution] = {}
    for eps in [1, 1e-3, 1e-6]:
        
        errors[resolution][eps] = {}
        
        for etype in ['Q1', 'Q2']:
            #
            # Define element
            # 
            element = QuadFE(2, etype)
            dofhandler = DofHandler(mesh, element)
            #
            # Define Basis Functions
            #
            u  = Basis(dofhandler, 'u')
            ux = Basis(dofhandler, 'ux')
            uy = Basis(dofhandler, 'uy') 
         
            #
            # Define weak form
            # 
            a_diff_x = Form(eps, trial=ux, test=ux)
            a_diff_y = Form(eps, trial=uy, test=uy)
            a_adv_x = Form(vx, trial=ux, test=u)
            a_adv_y = Form(vy, trial=uy, test=u)
            b = Form(f, test=u)
    
            problem = [a_diff_x, a_diff_y, a_adv_x, a_adv_y, b]
            
            #
            # Assembler system
            # 
            assembler = Assembler([problem], mesh)
            assembler.add_dirichlet_constraint(None, ue)
            assembler.assemble()
            
            #
            # Get solution
            # 
            ua = assembler.solve()
            
            #
            # Compute the error
            # 
            e_vec = ua.data() - ue.interpolant(mesh, element).data()
            efn = Nodal(data=e_vec, dofhandler=dofhandler)
    
            #
            # Record error           
            # 
            errors[resolution][eps][etype] = max(np.abs(e_vec))


headers = ('Resolution','Q1:1', 'Q1:1e-3', 'Q1:1e-6','Q2:1', 'Q2:1e-3', 'Q2:1e-6')
print('{:<12} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(*headers))

for resolution in errors.keys():
    er = errors[resolution]
    row = [resolution[0],er[1]['Q1'], er[1e-3]['Q1'], er[1e-6]['Q1'], \
           er[1]['Q2'], er[1e-3]['Q2'], er[1e-6]['Q2']]
    print('{:<12} {:<10.3e} {:<10.3e} {:<10.3e} {:<10.3e} {:<10.3e} {:<10.3e}'.format(*row))








