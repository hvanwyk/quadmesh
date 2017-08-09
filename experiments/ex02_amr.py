"""
Example 2: Adaptive mesh refinement for a simple diffusion dominated 
    steady state advection-diffusion problem
    
    
"""
from finite_element import System, QuadFE
from mesh import Mesh
from plot import Plot
import matplotlib.pyplot as plt
from scipy.sparse import linalg as spla
import numpy as np


mesh = Mesh.newmesh(grid_size=(8,8))
mesh.refine()
element = QuadFE(2,'Q1')
system = System(mesh, element, n_gauss=(4,16))

tol = 1e-9
k = 3
v = 3.1
def m_dirichlet_inflow(x,y):
    return (np.abs(x-0)<1e-7)

def m_dirichlet_homogeneous(x,y):
    return np.logical_or(np.abs(y-0)<1e-9, np.abs(y-1)<1e-9)


def g_dirichlet_homogeneous(x,y):
    return 0

def g_dirichlet_inflow(x,y):
    return np.sin(np.pi*y)       
        
bc = {'dirichlet': [(m_dirichlet_homogeneous, g_dirichlet_homogeneous), 
                    (m_dirichlet_inflow, g_dirichlet_inflow)]}
bf = [(k,'ux','vx'),(k,'uy','vy'),(v,'ux','v')]
lf = [(0,'v')]

A,b = system.assemble(bilinear_forms=bf, 
                      linear_forms=lf, 
                      boundary_conditions=bc)

#
# Solve system 
# 
u = spla.spsolve(A.tocsc(), b)

#
# Plot results
# 
fig = plt.figure()
#ax = fig.add_subplot(1,1,1, projection='3d')
ax = fig.add_subplot(1,1,1)
plot = Plot()
#ax = plot.surface(ax, u, mesh, element)
ax = plot.contour(ax, fig, u, mesh, element)
plt.show()