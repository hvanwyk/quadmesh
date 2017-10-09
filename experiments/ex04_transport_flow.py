"""
Flow-Transport Equation with Random Permeability

Flow Equation: Seek pressure p = p(x,y)

    -div(K grad(p(x,y))) = 0,    (x,y) in [0,1]^2
    p(0,y) = 1,                  outflow
    p(1,y) = 0,                  inflow
    K*grad(p(x,y))*n = 0,        y = 0, or y = 1  (no flow)

Velocity
    
    u = grad(p)

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
from finite_element import Function, QuadFE, System
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
phi = 1  # porosity
D = 0.0252  # dispersivity
K = 1  # permeability


# =============================================================================
# Mesh and Elements
# =============================================================================
# Finite element mesh
mesh = Mesh.newmesh(grid_size=(50,50))
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

print('assembly')
A, b = p_system.assemble(bilinear_forms = bf_flow, 
                         linear_forms = lf_flow, 
                         boundary_conditions = bc_flow)

print('solve')
p = spla.spsolve(A.tocsc(),b)
p_fn = Function(p, mesh, p_element)
# TODO: Construct new function by (i) taking derivatives or (ii) multiplying by a constant or another function ...


print('plot')
fig = plt.figure() 
ax = fig.add_subplot(1,1,1, projection='3d')
plot = Plot()
#plot.contour(ax, fig, p_fn, mesh, p_element, derivatives=(1,0))
ax = plot.surface(ax, p_fn, mesh, p_element)
plt.show()

#
# Transport
# 
bf_trans = [(1,'u','v'), (dt*vx_fn, 'ux','v'), (dt*vy_fn, 'uy','v'), 
            (dt*D,'ux','vx'), (dt*D,'uy','vy')]
lf_trans = []

# ============================================================================
# 
# ============================================================================