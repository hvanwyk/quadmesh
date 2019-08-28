from assembler import Assembler
from assembler import Form
from fem import DofHandler
from fem import QuadFE
from fem import Basis
from function import Nodal
from gmrf import Covariance
from gmrf import GaussianField
from mesh import Mesh1D
from plot import Plot
from solver import LS

# Built-in modules
import numpy as np
"""
Implement Reduced order Model 

-d/dx(q(x,w)d/dx u(x)) = f(x)
u(0) = 1
u(1) = 0
"""
mesh = Mesh1D(resolution=(100,))
mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)

element = QuadFE(mesh.dim(), 'Q1')
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()
n = dofhandler.n_dofs()

# =============================================================================
# Random field
# =============================================================================
n_samples = 1000
cov = Covariance(dofhandler, name='gaussian', parameters={'l':0.1})
log_q = GaussianField(n, K=cov)
log_q.update_support()
qfn = Nodal(dofhandler=dofhandler, 
            data=np.exp(log_q.sample(n_samples=n_samples)))
plot = Plot()
plot.line(qfn, i_sample=np.arange(n_samples))

# =============================================================================
# Generate Snapshot Set
# ============================================================================= 
phi = Basis(dofhandler, 'u')
phi_x = Basis(dofhandler, 'ux')

problems = [[Form(kernel=qfn, trial=phi_x, test=phi_x), Form(1, test=phi)],
            [Form(1, test=phi, trial=phi)]]

assembler = Assembler(problems, mesh)
assembler.assemble()

A = assembler.af[0]['bilinear'].get_matrix()
b = assembler.af[0]['linear'].get_matrix()

linsys = LS(phi)
linsys.add_dirichlet_constraint('left',1)
linsys.add_dirichlet_constraint('right',0)

u_snap = Nodal(dofhandler=dofhandler, 
               data=np.empty((n,n_samples)))
u_data = np.empty((n,n_samples))
for i in range(n_samples):
    linsys.set_matrix(A[i])
    linsys.set_rhs(b.copy())
    linsys.solve_system()
    u_data[:,[i]] = linsys.get_solution(as_function=False)
u_snap.set_data(u_data)
plot = Plot()
plot.line(u_snap, i_sample=np.arange(n_samples))
  
  
# =============================================================================
# Compute Reduced Order Model
# ============================================================================= 
M = assembler.af[1]['bilinear'].get_matrix()


# =============================================================================
# Predict output using ROM
# ============================================================================= 