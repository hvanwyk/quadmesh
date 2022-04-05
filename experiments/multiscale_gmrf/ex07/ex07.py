#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Sampling for Advection Diffusion
=======================================

Local Average

Q = 1/A*∫ u(x,θ) dx, 

where

u satisfies the advection-diffusion equation

-∇·(exp(θ) ∇u) + v·∇u = 0, for x in [0,1]^2
  u(x_in) = 1,   on inflow
  u(x_out) = 0   on outflow
  ∇u·n = 0       otherwise
    

θ ~ Gaussian Random Field

Created on Mon Apr  4 16:17:45 2022

@author: hans-werner
"""

from mesh import QuadMesh
from fem import DofHandler, Basis, QuadFE
from function import Nodal
from gmrf import GaussianField, SPDMatrix, Covariance
from plot import Plot
from assembler import Assembler, Form, Kernel
import numpy as np

# Computational mesh 
mesh = QuadMesh(resolution=(50,50))

# Mark Dirichlet boundary regions 
out_fn = lambda x,y: abs(x-1)<1e-8 and 0.8<=y and y<=1
mesh.mark_region('out', out_fn, entity_type='half_edge', on_boundary=True)

in_fn = lambda x,y: abs(x)<1e-8 and 0<=y and y<=0.2
mesh.mark_region('in', in_fn, entity_type='half_edge', on_boundary=True )


x_min, x_max = 0.7, 0.8
y_min, y_max = 0.4, 0.5
reg_fn = lambda x,y: x>=x_min and x<=x_max and y>=y_min and y<=y_max
mesh.mark_region('reg', reg_fn, entity_type='cell')

# Elements
Q1 = QuadFE(mesh.dim(), 'Q1')
dQ1 = DofHandler(mesh, Q1)
dQ1.distribute_dofs()

phi = Basis(dQ1)
phi_x = Basis(dQ1, derivative='vx')
phi_y = Basis(dQ1, derivative='vy')

#
# Diffusion Parameter
# 

# Covariance Matrix
K = Covariance(dQ1, name='exponential', parameters={'sgm':1, 'l': 0.1})

# Gaussian random field θ
tht = GaussianField(dQ1.n_dofs(),K=K)

# Sample from field
tht_fn = Nodal(data=tht.sample(n_samples=3), basis=phi)
plot = Plot()
plot.contour(tht_fn)

#
# Advection 
# 
v = [0.1,-0.1]

plot.mesh(mesh,regions=[('in','edge'),('out','edge'),('reg','cell')])

k = Kernel(tht_fn, F=lambda tht: np.exp(tht))
adv_diff = [Form(k, trial=phi_x, test=phi_x), 
            Form(k, trial=phi_y, test=phi_y),
            Form(0, test=phi),
            Form(v[0], trial=phi_x, test=phi),
            Form(v[1], trial=phi_y, test=phi)] 

average = [Form(1, test=phi, flag='reg'), Form(1, flag='reg')]
assembler = Assembler([adv_diff, average], mesh)
assembler.add_dirichlet('out', dir_fn=0)
assembler.add_dirichlet('in', dir_fn=10)


assembler.assemble()

u_vec = assembler.solve(i_problem=0, i_matrix=0, i_vector=0)
plot.contour(Nodal(data=u_vec, basis=phi))