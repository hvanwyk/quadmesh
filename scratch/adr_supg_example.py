from assembler import Form, Kernel, Assembler
from fem import QuadFE, Basis, DofHandler
from function import Nodal, Explicit, Constant
from mesh import QuadMesh
from plot import Plot
from solver import LinearSystem
from diagnostics import Verbose
import numpy as np

"""
Simulate the time dependent advection-diffusion-reaction system 

    u_t - div*(D*grad(u)) + div(v*u) + R(u) = 0
    
subject to the appropriate initial and boundary conditions, using
SUPG and 
 
"""
comment = Verbose()
# Computational mesh
mesh = QuadMesh(box=[0,10,0,10], resolution=(25,25))
left = mesh.mark_region('left', lambda x,y: abs(x)<1e-10)

# Finite elements

# Piecewise constants
E0 = QuadFE(mesh.dim(),'DQ0')
V0 = DofHandler(mesh,E0)

# Piecewise linears
E1 = QuadFE(mesh.dim(),'Q1')
V1 = DofHandler(mesh,E1)
v = Basis(V1,'v')
v_x = Basis(V1,'vx')
v_y = Basis(V1,'vy')

V1.distribute_dofs()
print(V1.n_dofs())


# Time discretization
t0, t1, dt = 0, 1, 0.025
nt = np.int((t1-t0)/dt)

# Initial condition
u0 = Constant(0)

# Left Dirichlet condition
def u_left(x):
    n_points = x.shape[0]
    u = np.zeros((n_points,1))
    left_strip = (x[:,0]>=4)*(x[:,0]<=6)*(abs(x[:,1])<1e-9)
    u[left_strip,0]=1
    return u
u_left = Explicit(f=u_left,mesh=mesh)

# Define problems
Dx = 1
Dy = 0.1*Dx

Form(1,trial=v,test=v)

k_max = 1
for t in np.linspace(t0,t1,nt):
    print(t)
    for k in range(k_max):
        up = Nodal(f=lambda x:x[:,1]**2, dofhandler=V1)
        um = Nodal(f=lambda x:x[:,0], dofhandler=V1)
        residual = [Form(Kernel(f=up),test=v),
                    Form(Kernel(f=um,F=lambda um:-um), test=v),
                    Form(Kernel(f=[up],derivatives=['ux'],F=lambda ux,dt=dt,Dx=Dx:dt*Dx*ux),test=v_x)]
        
        comment.tic('Assembly')
        assembler = Assembler(residual,mesh)
        assembler.assemble()
        comment.toc()
        
        # Form residual 
        """
        (up, v) - (um,v) 
        + dt*(Dx*up_x,v_x) + dt*(Dy*up_y,v_y) 
        + dt*(v1*up_x,v) + dt*(v2*up_y,v) 
        + dt*(k*up**2,v)
        
        """
        #residual = [Form(u_np,test=v), 
        #            Form(-u_nm,test=v)]
                

 