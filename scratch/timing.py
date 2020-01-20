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
from mesh import Mesh1D

from function import Nodal
from function import Explicit
from function import Constant

from plot import Plot
from solver import LinearSystem

import numpy as np
import scipy
from scipy import linalg
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
from matplotlib import animation

import time
from diagnostics import Verbose
# =============================================================================
# Parameters
# =============================================================================
#
# Flow
# 
comment = Verbose()

# permeability field
phi = Constant(1)  # porosity
D   = Constant(0.0252)  # dispersivity
K   = Constant(1)  # permeability

# =============================================================================
# Mesh and Elements
# =============================================================================
# Mesh
comment.tic('initializing mesh')
mesh = QuadMesh(resolution=(200,200))
comment.toc()

comment.tic('iterating over mesh cells')
for cell in mesh.cells.get_leaves():
    pass
comment.toc()

# Elements
element = QuadFE(2,'Q1')  # element for pressure

# Dofhandlers
dofhandler = DofHandler(mesh, element)

comment.tic('distribute dofs')
dofhandler.distribute_dofs()
comment.toc()

print('number of dofs:',dofhandler.n_dofs())

# Basis functions
p_u  = Basis(dofhandler, 'u')
p_ux = Basis(dofhandler, 'ux')
p_uy = Basis(dofhandler, 'uy')


p_inflow = lambda x,y: np.ones(shape=x.shape)
p_outflow = lambda x,y: np.zeros(shape=x.shape)
c_inflow = lambda x,y: np.zeros(shape=x.shape)

# =============================================================================
# Solve the steady state flow equations
# =============================================================================

# Define problem
flow_problem = [Form(1,test=p_ux,trial=p_ux), 
                Form(1,test=p_uy,trial=p_uy), 
                Form(0,test=p_u)] 

# Assembler

tic = time.time()
assembler = Assembler(flow_problem, mesh)
toc = time.time()-tic

print('initializing assembler', toc)

tic = time.time()
assembler.assemble()
toc = time.time()-tic
print('assembly time', toc)

A = assembler.af[0]['bilinear'].get_matrix()
b = assembler.af[0]['linear'].get_matrix()
