"""
Flow-Transport Equation with Random Permeability

Flow Equation: Seek pressure p = p(x,y)

    -div(K grad(p(x,y))) = 0,    (x,y) in [0,1]^2
    p(0,y) = 1,                  outflow
    p(1,y) = 0,                  inflow
    K*grad(p(x,y))*n = 0,        y = 0, or y = 1  (no flow)

Velocity
    
    u = -K*grad(p)

Transport Equation: Seek concentration c = c(x,y,t)

    dc/dt + u*grad(c) - div(D*grad(c)) = 0
    c(x,y,0) = 1            initial data
    c(0,y,t) = 0            homogenous dirichlet conditions
    D grad(c(x,y,t)*n=0     (x,y) in {(x,y) in dD: x != 0}


Quantity of Interest: Average Breakthrough Curve

    Q = \int_{dD_out} c(x,y,t)[u*n]ds

Source: 

Ossiander et. al. 2014, Conditional Stochastic Simulations of Flow and
Transport with Karhunen-Loève Expansions, Stochastic Collocation, and 
Sequential Gaussian Simulation
"""
# Imports 
from assembler import Form
from assembler import Kernel
from assembler import Assembler

from fem import QuadFE
from fem import DofHandler
from fem import Basis

from mesh import QuadMesh

from function import Nodal
from function import Explicit
from function import Constant

from plot import Plot

import numpy as np
import scipy
from scipy import linalg
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
from matplotlib import animation

from diagnostics import Verbose

def test_ft():
    plot = Plot()
    vb = Verbose()
    
    # =============================================================================
    # Parameters
    # =============================================================================
    #
    # Flow
    # 
    
    # permeability field
    phi = Constant(1)  # porosity
    D   = Constant(0.0252)  # dispersivity
    K   = Constant(1)  # permeability
    
    # =============================================================================
    # Mesh and Elements
    # =============================================================================
    # Mesh
    mesh = QuadMesh(resolution=(30,30))
    
    # Mark left and right regions
    mesh.mark_region('left',lambda x,y: np.abs(x)<1e-9, entity_type='half_edge')
    mesh.mark_region('right',lambda x,y: np.abs(x-1)<1e-9, entity_type='half_edge')
    
    
    # Elements
    p_element = QuadFE(2,'Q1')  # element for pressure
    c_element = QuadFE(2,'Q1')  # element for concentration
    
    # Dofhandlers
    p_dofhandler = DofHandler(mesh, p_element)
    c_dofhandler = DofHandler(mesh, c_element)
    
    p_dofhandler.distribute_dofs()
    c_dofhandler.distribute_dofs()
    
    # Basis functions
    p_ux = Basis(p_dofhandler, 'ux')
    p_uy = Basis(p_dofhandler, 'uy')
    p_u =  Basis(p_dofhandler, 'u')
    
    
    p_inflow = lambda x,y: np.ones(shape=x.shape)
    p_outflow = lambda x,y: np.zeros(shape=x.shape)
    c_inflow = lambda x,y: np.zeros(shape=x.shape)
    
    # =============================================================================
    # Solve the steady state flow equations
    # =============================================================================
    vb.comment('Solving flow equations')
    
    # Define problem
    flow_problem = [Form(1,test=p_ux,trial=p_ux), 
                    Form(1,test=p_uy,trial=p_uy), 
                    Form(0,test=p_u)] 
    
    # Assemble 
    vb.tic('assembly')
    assembler = Assembler(flow_problem)
    assembler.add_dirichlet('left',1)
    assembler.add_dirichlet('right',0)
    assembler.assemble()
    vb.toc()
    
    # Solve linear system
    vb.tic('solve')
    A = assembler.get_matrix().tocsr()
    b = assembler.get_vector()
    x0 = assembler.assembled_bnd()
    
    # Interior nodes
    pa = np.zeros((p_u.n_dofs(),1))
    int_dofs = assembler.get_dofs('interior')
    pa[int_dofs,0] = spla.spsolve(A,b-x0)
    
    # Resolve Dirichlet conditions
    dir_dofs, dir_vals = assembler.get_dirichlet(asdict=False)
    pa[dir_dofs] = dir_vals
    vb.toc()
    
    # Pressure function
    pfn = Nodal(data=pa, basis=p_u)
    
 
    px = pfn.differentiate((1,0))
    py = pfn.differentiate((1,1))
    
    #plot.contour(px)
    #plt.show()
        
    # =============================================================================
    # Transport Equations
    # =============================================================================
    # Specify initial condition
    c0 = Constant(1)
    dt = 1e-1
    T  = 6
    N  = int(np.ceil(T/dt))
    
    c = Basis(c_dofhandler, 'c')
    cx = Basis(c_dofhandler, 'cx')
    cy = Basis(c_dofhandler, 'cy')
    
    print('assembling transport equations')
    k_phi = Kernel(f=phi)
    k_advx = Kernel(f=[K,px], F=lambda K,px: -K*px)
    k_advy = Kernel(f=[K,py], F=lambda K,py: -K*py)
    tht = 1
    m = [Form(kernel=k_phi, test=c, trial=c)]
    s = [Form(kernel=k_advx, test=c, trial=cx),
         Form(kernel=k_advy, test=c, trial=cy),
         Form(kernel=Kernel(D), test=cx, trial=cx),
         Form(kernel=Kernel(D), test=cy, trial=cy)]
    
    problems = [m,s]
    assembler = Assembler(problems)
    assembler.add_dirichlet('left',0, i_problem=0)
    assembler.add_dirichlet('left',0, i_problem=1)
    assembler.assemble()
    
    x0 = assembler.assembled_bnd()
    
    # Interior nodes
    int_dofs = assembler.get_dofs('interior')
    
    # Dirichlet conditions
    dir_dofs, dir_vals = assembler.get_dirichlet(asdict=False)
    
    # System matrices
    M = assembler.get_matrix(i_problem=0)
    S = assembler.get_matrix(i_problem=1)
    
    # Initialize c0 and cp
    c0 = np.ones((c.n_dofs(),1))
    cp = np.zeros((c.n_dofs(),1))
    c_fn = Nodal(data=c0, basis=c) 
    
    #
    # Compute solution 
    #  
    print('time stepping')
    for i in range(N):
        
        # Build system
        A = M + tht*dt*S
        b = M.dot(c0[int_dofs])-(1-tht)*dt*S.dot(c0[int_dofs])
                
        # Solve linear system
        cp[int_dofs,0] = spla.spsolve(A,b)
        
        # Add Dirichlet conditions
        cp[dir_dofs] = dir_vals
        
        # Record current iterate
        c_fn.add_samples(data=cp)
        
        # Update c0
        c0 = cp.copy()
        
        #plot.contour(c_fn, n_sample=i)
    
    #
    # Quantity of interest
    #
    def F(c,px,py, entity=None):
        """
        Compute c(x,y,t)*(grad p * n)
        """
        n = entity.unit_normal()
        return c*(px*n[0]+py*n[1])
    
    px.set_subsample(i=np.arange(41))
    py.set_subsample(i=np.arange(41))
    
    #kernel = Kernel(f=[c_fn,px,py], F=F)
    kernel = Kernel(c_fn)
    
    #print(kernel.n_subsample())
    form = Form(kernel, flag='right', dmu='ds')
    assembler = Assembler(form, mesh=mesh)
    assembler.assemble()
    QQ = assembler.assembled_forms()[0].aggregate_data()['array']
    
    Q = np.array([assembler.get_scalar(i_sample=i) for i in np.arange(N+1)])
    t = np.linspace(0,T,N+1)
    plt.plot(t,Q)
    plt.show()
    print(Q)
if __name__ == '__main__':
    test_ft()
    





"""

# Boundary conditions
bc_flow = {'dirichlet': [(bnd_inflow, p_inflow),(bnd_outflow, p_outflow)]}

A, b = p_system.assemble(bilinear_forms = bf_flow, 
                         linear_forms = lf_flow, 
                         boundary_conditions = bc_flow)

p = spla.spsolve(A.tocsc(),b)
p_fn = Function(p, 'nodal', mesh, p_element)


# Use Darcy's law to construct the velocity functions
vx_fn = p_fn.derivative((1,0))
vy_fn = p_fn.derivative((1,1))



dt = Function(0.01, 'constant')
T  = 0.5
n_time = np.int(T/0.01)
c0 = Function(1, 'constant').interpolate(mesh=mesh, element=c_element)

#
# Transport
# 
c_vals = np.empty((c_system.dofhandler().n_dofs(),n_time))
c_fn = Function(c_vals, 'nodal', mesh, c_element)
c_fn.assign(c0.fn(),pos=0)
for i in range(n_time-1):
    print(i)
    # Assembly
    bf_transport = [(1,'u','v'), 
                    (dt.times(vx_fn), 'ux','v'), (dt.times(vy_fn), 'uy','v'), 
                    (dt.times(D),'ux','vx'), (dt.times(D),'uy','vy')]
    lf_transport = [(c0,'v')]
    
    bc_transport = {'dirichlet': [(bnd_inflow, c_inflow)]}
    A,b = c_system.assemble(bilinear_forms=bf_transport, 
                            linear_forms=lf_transport, 
                            boundary_conditions = bc_transport)
    c = spla.spsolve(A.tocsc(),b)
    c_fn.assign(c, pos=i+1)
    c0.assign(c)
    
# ============================================================================
# Plot flow
# ============================================================================
def update_contour_plot(i, data,  ax, fig, xi, yi):
    ax.cla()
    im = ax.contourf(xi, yi, data[:,i].reshape(xi.shape), 200, cmap='viridis')
    plt.title(str(i))
    return im,

x0,x1,y0,y1 = mesh.box()
x = np.linspace(x0,x1,100)
y = np.linspace(y0,y1,100)
X, Y = np.meshgrid(x,y)
xy = np.array([X.ravel(),Y.ravel()]).T
c_data = c_fn.eval(xy)

fig = plt.figure()
im = plt.contourf(X, Y, c_data[:,0].reshape(X.shape), 200, cmap='viridis')
numframes = c_data.shape[1]
ax = fig.gca()
ani = animation.FuncAnimation(fig, update_contour_plot, frames=range(numframes), fargs=(c_data, ax, fig, X, Y))
plt.colorbar(im)
plt.show()





print(c_data.shape)
'''
def update_contour_plot(c_fn, pos, ax_c, fig, mesh, element):
    ax_c.cla()
    
    fig, ax_c, cm = plot.contour(ax_c, fig, c_fn, mesh, element=None, derivative=(0,), \
                                 colorbar=True, resolution=(100,100), flag=None) 
    
    return im,
'''
print('done')
"""