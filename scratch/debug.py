from function import Constant, Explicit, Nodal
from fem import DofHandler, Basis, QuadFE
from assembler import Kernel
from mesh import Mesh1D

import numpy as np
import matplotlib.pyplot as plt


mesh = Mesh1D(resolution=(1,))
Q1 = QuadFE(1,'Q1')
Q2 = QuadFE(1,'Q2')
# 
# Sampling 
#
one = Constant(1)
f1 = Explicit(lambda x: x**2 + 1, dim=1)

# Sampled function
a = np.linspace(0,1,11)
n_samples = len(a)

# Define Dofhandler
dh = DofHandler(mesh, Q2)
dh.distribute_dofs()
dh.set_dof_vertices()
xv = dh.get_dof_vertices()
n_dofs = dh.n_dofs()

phi = Basis(dh, 'u')

# #
# Sample multiple constant functions
# 
f2_m  = np.empty((n_dofs, n_samples))
for i in range(n_samples):
    f2_m[:,i] = xv.ravel() + a[i]*xv.ravel()**2

f1 = Constant(data=a)
f2 = Explicit(lambda x: 1 + x**2, dim=1)
f3 = Nodal(data=f2_m[:,-1], basis=phi)

F = lambda f1, f2, f3: f1 + f2 + f3
k = Kernel([f1,f2,f3], F=F)

x = np.linspace(0,1,100)

plt.plot(x, k.eval(x)[:,2], x, f1.eval(x)[:,2]+f2.eval(x)+f3.eval(x),'.-')
plt.show()
for i in range(n_samples):
    print(np.allclose(k.eval(x)[:,[i]], f1.eval(x)[:,[i]]+f2.eval(x) + f3.eval(x)))
#self.assertTrue(np.allclose(k.eval(x)[:,i], \
#                a[i] + f2.eval(x) + f3.eval(x)))