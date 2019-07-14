from mesh import Mesh1D
from mesh import QuadMesh
from fem import DofHandler
from function import Function
from function import Nodal
from fem import QuadFE
from assembler import Kernel
from assembler import Form
from fem import Basis
from assembler import Assembler
from solver import LinearSystem
from plot import Plot
import numpy as np
from mesh import HalfEdge
class JumpKernel(Kernel):
    """
    Special Kernel to compute jumps across half-edges
    """
    def __init__(self, u, dfdx=None, samples='all'): 
        """
        Constructor 
        """
        Kernel.__init__(self, u, dfdx=dfdx, F=None, samples=samples)
        
        
    def eval(self, x=None, cell=None, region=None, phi=None, 
             dofs=None, compatible_functions=set()):
        """
        Evaluate Kernel
        """ 
        assert isinstance(region, HalfEdge), \
            'Input "region" must be a "HalfEdge".'
        
        dfdx = self.dfdx[0]
        #
        # Compute u on current cell
        # 
        u_in = u.eval(x=x, cell=cell, derivative=dfdx, samples=self.samples)
        
        #
        # Compute u on neighboring cell
        # 
        nb = cell.get_neighbors(region)
        if nb is not None:
            u_out = u.eval(x=x, cell=nb, derivative=dfdx, samples=self.samples)
            #
            # Compute the jump
            # 
            u_jump = 0.5*np.abs(u_out-u_in)
            print(u_jump)
        else:
            u_jump = 0*u_in
        
        
        n_samples = self.n_samples
        if n_samples is not None:
            if u.n_samples() is None:
                #
                # Deterministic function
                # 
                u_jump = (np.ones((n_samples,n_points))*u_jump).T
        
        #
        # Return jump
        # 
        return u_jump
    
    

"""
Task: 

Modify the kernel class to allow for the computation of forms involving

(i) normal derivatives or
(ii) jumps 

across cell edges. 

i.e. I n.A*grad(u) ds or 
"""


#
# Define mesh
# 
mesh = QuadMesh(resolution=(20,20))
element = QuadFE(2,'DQ1')

u = Nodal(f=lambda x: -1+x[:,0]**5 + x[:,1]**8, mesh=mesh, element=element)
print(u.data())
plot = Plot(3)
plot.wire(u)
 
kernel = JumpKernel(u, dfdx='fx')
form = Form(kernel=kernel, dmu='ds')
assembler = Assembler(form, mesh)
assembler.assemble()
cf = assembler.af[0]['constant']
print(cf.get_matrix())

