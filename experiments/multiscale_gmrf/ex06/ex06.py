#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYBRID SAMPLING FOR FUNCTIONAL
==============================
Quantity of interest

 Q(y) =  ∫ f(θ(x,y)) dx
 
where θ(x,y) is a given Gaussian random field, x is a spatial variable, and 
y is a Gaussian random vector. 

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
from assembler import Assembler, Form, Kernel
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

def reference_qoi(f, tht, basis, region, n=1000000, verbose=True):
    """
    Parameters
    ----------
    f : lambda function,
        Function of θ to be integrated.
        
    tht : GaussianField, 
        Random field θ defined on mesh in terms of its mean and covariance
        
    basis : Basis, 
        Basis function defining the nodal interpolant. It incorporates the
        mesh, the dofhandler, and the derivative.
        
    region : meshflag, 
        Flag indicating the region of integration
    
    n : int, default=1000000 
        Sample size
    
    Returns
    -------
    Q_ref : double, 
        Reference quantity of interest
        
    err : double, 
        Expected RMSE given by var(Q)/n. 
    """
     
    #
    # Assemble integral 
    # 
    batch_size = 100000
    n_batches = n//batch_size + (0 if (n%batch_size)==0 else 1)
    
    if verbose:
        print('Computing Reference Quantity of Interest')
        print('========================================')
        print('Sample size: ', n)
        print('Batch size: ', batch_size)
        print('Number of batches: ', n_batches)
        
    Q_smpl = np.empty(n)
    for k in range(n_batches):
        
        # Determine sample sizes for each batch
        if k < n_batches-1:
            n_sample = batch_size
        else:
            # Last sample may be smaller than batch_size
            n_sample = n - k*batch_size
        
        if verbose:
            print(' - Batch Number ', k)
            print(' - Sample Size: ', n_sample)
            print(' - Sampling random field')    

        # Sample from field
        tht_smpl = tht.sample(n_sample)
        
        # Define kernel
        tht_n = Nodal(data=tht_smpl, basis=basis)
        kf = Kernel(tht_n, F=f)
    
        # Define forms
        if k == 0:
            problems = [ [Form(kernel=kf, flag=region)], [Form(flag=region)]]
        else:
            problems = [Form(kernel=kf, flag=region)]
        
        
        if verbose:
            print(' - Assembling.')
            
        # Compute the integral
        assembler = Assembler(problems, basis.mesh())
        assembler.assemble()
        
        #
        # Compute statistic
        # 
        
        # Get samples
        if k == 0:
            dx = assembler.get_scalar(i_problem=1)  
        
        if verbose:
            print(' - Updating samples \n')
            
        batch_sample = assembler.get_scalar(i_problem=0, i_sample=None)
        Q_smpl[k*batch_size:k*batch_size + n_sample] = batch_sample/dx
    
    # Compute mean and MSE
    Q_ref = np.mean(Q_smpl)
    err = np.var(Q_smpl)/n
    
    # Return reference 
    return Q_ref, err



def sg_convergence(f,):
    """
    """
    pass


def plot_heuristics(f, tht, basis, region, condition):
    """
    Parameters
    ----------
    f : lambda, 
        Function of θ to be integrated.
        
    n : int, 
        Sample size for Monte Carlo sample
    
    """
    tht.update_support()
    #
    # Plot samples of the random field
    #
    n = 10000
    tht_sample = tht.sample(n)
    tht_fn = Nodal(data=tht_sample, basis=basis)
    
    #
    # Compute the quantity of interest
    #
    
    # Define the kernel
    kf = Kernel(tht_fn, F=f)
    
    # Assemble over the mesh
    problems = [ [Form(kernel=kf, flag=region)], [Form(flag=region)]]
    assembler = Assembler(problems, basis.mesh())
    assembler.assemble()
    
    # Extract sample
    dx = assembler.get_scalar(i_problem=1)
    q_sample = assembler.get_scalar(i_sample=None)/dx
    
    #
    # Compute correlation coefficients of q with spatial data
    # 
    
    plot = Plot(quickview=False)
    fig, ax = plt.subplots()
    plt_args = {'linewidth':0.5, 'color':'k'}
    ax = plot.line(tht_fn, axis=ax, i_sample=list(range(100)),
                   plot_kwargs=plt_args)
    fig.savefig('ex01_sample_paths.eps')
    
    fig, ax = plt.subplots()
    ftht = Nodal(data=f(tht_sample), basis=basis)
    ax = plot.line(ftht, axis=ax, i_sample=list(range(100)),
                   plot_kwargs=plt_args)
    fig.savefig('ex01_integrand.eps')
    
    fig, ax = plt.subplots(1,1)
    plt.hist(q_sample, bins=50, density=True)
    ax.set_title(r'Histogram $Q(\theta)$')
    fig.savefig('ex01_histogram.eps')
    plt.close('all')
    
    dh = basis.dofhandler()
    n_dofs = dh.n_dofs()
    
    # Extract the region on which we condition
    cnd_dofs = dh.get_region_dofs(entity_flag=condition)
    I = np.eye(n_dofs)
    I = I[cnd_dofs,:]
    
    # Measured tht
    tht_msr = tht_sample[cnd_dofs,0][:,None]
    
    n_cnd = 30
    cnd_tht = tht.condition(I, tht_msr, n_samples=n_cnd)
    
    #cnd_tht_data = np.array([tht_sample[:,0] for dummy in range(n_cnd)])
    #cnd_tht_data[cnd_dofs,:] = cnd_tht
    
    cnd_tht_fn = Nodal(data=f(cnd_tht), basis=basis)
    fig, ax = plt.subplots()
    ax = plot.line(cnd_tht_fn, axis=ax, i_sample=np.arange(n_cnd), 
                   plot_kwargs=plt_args)
    fig.tight_layout()
    plt.show()
    
    #rho = np.corrcoef(np.vstack((q_sample, tht_sample)))
    #fig, ax = plt.subplots()
    #plt.plot(rho[0,1:])
    
# -----------------------------------------------------------------------------
# Spatial Approximation
# -----------------------------------------------------------------------------
#
# Mesh
#
mesh = Mesh1D(resolution=(500,))
x_min, x_max = 0.8, 1  # Integration limits
xx_min, xx_max = 0, 0.7  # region on which to condition

f_int_region = lambda x: x>=x_min and x<=x_max
f_cnd_region = lambda x: x>=xx_min and x<=xx_max

mesh.mark_region('integration', f_int_region,entity_type='cell')
mesh.mark_region('condition', f_cnd_region, entity_type='cell')

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

#f = lambda x: np.exp(-x**2)
#f = lambda x: x**2
#f = lambda x: np.arctan(10*x)
f = lambda x: np.exp(-np.abs(x))

plot_heuristics(f, eta, phi_1, 'integration', 'condition')

#Q_ref, err = reference_qoi(f, eta, phi_1, 'region')
#print(Q_ref, err)

#
# Construct Sparse Grid
# 
k = 20
depth = 4
z, w = hermite_rule(k, depth)
n_sg = len(w)


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
theta_trunc = Nodal(data=eta_trunc_sg[:,[50]]+eta_tail_mc,basis=phi_1)
 
# Assembler
k = Kernel(theta_trunc, F=lambda tht: tht**2)
problem = Form(flag='integration', kernel=k)
assembler = Assembler(problem, mesh)
assembler.assemble()
v = assembler.get_scalar(i_sample=3)


#plot = Plot(quickview=False)
#fig, ax = plt.subplots()

#plot.mesh(mesh,regions=[('region','cell')])

"""
ax = plot.line(theta_trunc, axis=ax, i_sample=np.arange(n_mc), 
               plot_kwargs={'linewidth':0.2, 
                            'color':'k'})
"""
#plt.show()

