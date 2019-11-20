from mesh import Mesh1D
from function import Nodal, Explicit, Constant
from assembler import Assembler
from fem import Basis
from fem import Kernel 
from plot import Plot

"""
Solve the problem 

-c*u_xx + v u_x = 0
u(0) = 0, u(1) = 1
"""

mesh = Mesh1D(resolution=(20,))

