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
from fem import Function, QuadFE, System
from mesh import Mesh
from plot import Plot
import numpy as np
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt

# =============================================================================
# Parameters
# =============================================================================
#
# Flow
# 

# permeability field
phi = Function(1, 'constant')  # porosity
D   = Function(0.0252, 'constant')  # dispersivity
K   = Function(1, 'constant')  # permeability

# =============================================================================
# Mesh and Elements
# =============================================================================
# Finite element mesh
mesh = Mesh.newmesh(grid_size=(20,20))
mesh.refine()
 
p_element = QuadFE(2,'Q1')  # element for pressure
c_element = QuadFE(2,'Q1')  # element for concentration

p_system = System(mesh, p_element)
c_system = System(mesh, c_element)


# =============================================================================
# Boundary Conditions
# =============================================================================
def bnd_inflow(x,y):
    """
    Inflow boundary: x = 0
    """
    return np.abs(x)<1e-10

def bnd_outflow(x,y):
    """
    Outflow boundary: x = 1
    """
    return np.abs(x-1)<1e-10

p_inflow = lambda x,y: np.ones(shape=x.shape)
p_outflow = lambda x,y: np.zeros(shape=x.shape)
c_inflow = lambda x,y: np.zeros(shape=x.shape)

# =============================================================================
# Assembly
# =============================================================================
#
# Flow
# 

# Bilinear forms
bf_flow = [(K,'ux','vx'),(K,'uy','vy')]

# Linear forms
lf_flow = [(0,'v')]

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
T  = 2
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
# 
# ============================================================================

print('done')