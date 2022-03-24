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
# Quadmesh Modules
from mesh import Mesh1D
from assembler import Assembler, Form
from fem import QuadFE, DofHandler, Basis
from function import Nodal
from gmrf import Covariance, GaussianField
from plot import Plot

# External modules
import numpy as np
import Tasmanian
import matplotlib.pyplot as plt


def hermite_rule(dimension, depth, type='level'):
    """
    Return the quadrature nodes and weights associated with the Hermite rule. 
    
    Parameters
    ----------
    dimension: int, 
        Dimension of the quadrature rule
        
    depth: int, 
        The interpolation 'degree' of the rule
        
    type: {level}, 
        The type of tensorization used 
        
        
    Returns
    -------
    z : double, 
        Quadrature nodes for rule.
        
    w: double, 
        Quadrature weights           
    """
    grid = Tasmanian.TasmanianSparseGrid()
    k = 4
    outputs = 0
    type = 'level'  # can be changed
    rule = 'gauss-hermite'  # appropriate for Gaussian fields
    grid.makeGlobalGrid(dimension, outputs, depth, type, rule)
    
    # Sample Points
    zzSG = grid.getPoints()
    z = np.sqrt(2)*zzSG                # transform to N(0,1)
    
    # Quadrature weights
    w = grid.getQuadratureWeights()
    w /= np.sqrt(np.pi)**k     # normalize weights
    
    
    return z, w 

def experiment():
    """
    
    """
    pass

# -----------------------------------------------------------------------------
# Spatial Approximation
# -----------------------------------------------------------------------------
#
# Mesh
#
mesh = Mesh1D(resolution=(500,))
x_min, x_max = 0.9, 1
region_fn = lambda x: x>=x_min and x<=x_max
mesh.mark_region('region',region_fn,entity_type='cell')

#
# Finite Elements
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

# -----------------------------------------------------------------------------
# Stochastic Approximation
# -----------------------------------------------------------------------------
#
# Random Field
# 
# Covariance kernel
K = Covariance(dQ1,name='gaussian', parameters={'sgm':1, 'l':0.1})
D, V = K.get_eig_decomp()

# Random Field
n_dofs = dQ1.n_dofs()
eta = GaussianField(n_dofs,K=K)

#
# Construct Sparse Grid
# 
k = 20
depth = 4
z, w = hermite_rule(k, depth)
n_sg = len(w)
print(n_sg)

# Generate truncated field at the sparse grid points
eta_trunc_sg = V[:,:k].dot(np.diag(np.sqrt(D[:k])).dot(z.T))

# Generate a Monte Carlo sample on top of sparse grid 
n_mc = 20
zz = np.random.randn(n_dofs-k, n_mc)
eta_tail_mc = V[:,k:].dot(np.diag(np.sqrt(D[k:]))).dot(zz)
              

# -----------------------------------------------------------------------------
# Sample and Integrate
# -----------------------------------------------------------------------------
# Samples of random field
theta_trunc = Nodal(data=eta_trunc_sg[:,[20]]+eta_tail_mc,basis=phi_1)
 
# Assembler
problem = Form(flag='region')

plot = Plot(quickview=False)
fig, ax = plt.subplots()

#plot.mesh(mesh,regions=[('region','cell')])


ax = plot.line(theta_trunc, axis=ax, i_sample=np.arange(n_mc), 
               plot_kwargs={'linewidth':0.2, 
                            'color':'k'})

plt.show()

