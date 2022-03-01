#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYBRID SAMPLING FOR FUNCTIONAL
==============================
Quantity of interest

 Q(y) =  ∫ f(θ(x,y)) dx
 
where θ(x,y) is a given Gaussian random field. 

Experiment with 

    1. The covariance structure of θ(x,y)
    2. The smoothness of f. 

Observe 

    A. Accuracy of the sparse grid
    B. Truncation Error
    C. Monte Carlo Error / Conditional Variance


Created on Wed Feb 16 11:32:00 2022

@author: hans-werner
"""

from mesh import Mesh1D
from assembler import Assembler, Form
from fem import QuadFE, DofHandler, Basis
from function import Nodal
from gmrf import Covariance, GaussianField
from plot import Plot
import numpy as np

#
# Mesh
#
mesh = Mesh1D(resolution=(20,))
x_min, x_max = 0.9, 1
region_fn = lambda x: x>=x_min and x<=x_max
mesh.mark_region('region',region_fn,entity_type='cell')

#
# Elements
#

# Piecewise Constant
Q0 = QuadFE(mesh.dim(), 'DQ0')
dQ0 = DofHandler(mesh, Q0)
dQ0.distribute_dofs()
phi_0 = Basis(dQ0)

# Piecewise Linear
Q1  = QuadFE(mesh.dim(), 'Q1')
dQ1 = DofHandler(mesh, Q1)
dQ1.distribute_dofs()
phi_1 = Basis(dQ1)

#
# Random Field
# 
# Covariance kernel
K = Covariance(dQ1,name='exponential', parameters={'sgm':1, 'l':0.1})
D, V = K.get_eig_decomp()

# Random Field
theta = GaussianField(dQ1.n_dofs(),K=K)

# Samples of random field
theta_fn = Nodal(data=np.exp(theta.sample()),basis=phi_1)
 
# Assembler
problem = Form(flag='region')
#assembler = Assembler(kernel=Jfn,mesh)

plot = Plot()
plot.mesh(mesh,regions=[('region','cell')])
#plot.line(theta_fn)


