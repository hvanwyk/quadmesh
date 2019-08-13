from assembler import Assembler
from assembler import Form
from assembler import Kernel

from diagnostics import Verbose

from fem import QuadFE
from fem import DofHandler
from fem import Basis

from function import Constant
from function import Explicit
from function import Map
from function import Nodal

from gmrf import Covariance
from gmrf import GaussianField

from mesh import QuadMesh
from mesh import Mesh1D

from plot import Plot

from solver import LS

import TasmanianSG

import numpy as np
import matplotlib.pyplot as plt
import gc
import scipy.sparse as sp
import multiprocessing

"""
Perform convergence tests for the problem

-(q(x,w)u'(x))' = 1,  0 < x < 1
u(0) = u(1) = 0

where

    q(x,w) = sum_{i=0}^3 alpha_i*cos(i*pi*x)*yi
    
and yi ~ unif(-1,1)
"""
def make_grid(dimensions=4, outputs=1, depth=4, 
              grid_type='tensor',rule='gauss-legendre'):
    """
    Generate a sparse grid
    """
    # Sparse grid
    tasmanian_library=\
        "/home/hans-werner/bin/TASMANIAN-6.0/libtasmaniansparsegrid.so"
    grid = TasmanianSG.TasmanianSparseGrid(tasmanian_library=\
                                           tasmanian_library)
    dimensions = dimensions
    outputs = outputs
    depth = depth
    grid_type = grid_type
    rule = 'gauss-legendre'
    grid.makeGlobalGrid(dimensions, outputs, depth, grid_type, rule)
    return grid


def get_points(mode='mc', n_samples=None, grid=None):
    """
    Sample the random vector y = [y1, y2, y3, y4]
    
    Inputs:
    
        n: int, level (if mode='sg') or sample size (if mode='mc')
        
        mode: str, type of sample ('mc'=Monte Carlo, 'sg'=Sparse Grid)
        
    
    Output:
    
        y: double, (n_samples,n_dim) array of sample nodes 
    """
    n_dimensions = 4
    #
    # Generate 'standard set' in [-1,1]
    # 
    if mode=='sg':
        assert grid is not None, 'Sparse grid must be specified'
        z = grid.getPoints()
        
    elif mode=='mc':
        assert n_samples is not None, 'Must specify sample size "n_samples".'
        z = np.random.rand(n_samples,n_dimensions)
        
        #
        # Modify to [-1,1]
        # 
        z = 2*z-1
    return z


def get_quadrature_weights(n, mode='mc',grid=None):
    """
    Return the quadrature weights for computing QoI's related to y's
    
    Inputs:
    
        n: int, level (if mode='sg') or sample size (if mode='mc')
        
        mode: str, type of sample ('mc'=Monte Carlo, 'sg'=Sparse Grid)
        
    
    Output:
    
        w: double, (n_samples,) array of quadrature weights     
    """
    if mode=='sg':
        assert grid is not None, 'Sparse grid should be specified.'
        w = grid.getQuadratureWeights()
        w *= 2**4
    elif mode=='mg':
        w = np.ones(n)/n

    return w


def set_diffusion(dQ, z):
    """
    Generate sample of diffusion coefficient
    
    Inputs:
    
        dQ: DofHandler, dofhandler object
    
        z: (n_samples, n_dim) array of sample points in [0,1]^2
    """
    x = dQ.get_dof_vertices()
    q_vals = 1 + 0.1  *np.outer(np.cos(1*np.pi*x[:,0]),z[:,0])+\
                 0.05 *np.outer(np.cos(2*np.pi*x[:,0]),z[:,1])+\
                 0.01 *np.outer(np.cos(3*np.pi*x[:,0]),z[:,2])+\
                 0.005*np.outer(np.cos(4*np.pi*x[:,0]),z[:,3])

    q_fn = Nodal(dofhandler=dQ, data=q_vals)
    return q_fn

    
def sample_state(mesh,dQ,z,mflag,reference=False):
    """
    Compute the sample output corresponding to a given input
    """
    n_samples = z.shape[0]
    q = set_diffusion(dQ,z)
    
    phi = Basis(dQ,'u', mflag)
    phi_x = Basis(dQ, 'ux', mflag)
    
    if reference:
        problems = [[Form(q,test=phi_x,trial=phi_x), Form(1,test=phi)], 
                    [Form(1,test=phi, trial=phi)],
                    [Form(1,test=phi_x, trial=phi_x)]]
    else:
        problems = [[Form(q,test=phi_x,trial=phi_x), Form(1,test=phi)]]
    
    assembler = Assembler(problems, mesh, subforest_flag=mflag)
    assembler.assemble()
    
    A = assembler.af[0]['bilinear'].get_matrix()
    b = assembler.af[0]['linear'].get_matrix()
    if reference:
        M = assembler.af[1]['bilinear'].get_matrix()
        K = assembler.af[2]['bilinear'].get_matrix()
        
    system = LS(phi)
    system.add_dirichlet_constraint('left')
    system.add_dirichlet_constraint('right')
    
    n_dofs = dQ.n_dofs(subforest_flag=mflag)
    y = np.empty((n_dofs,n_samples))
    for n in range(n_samples):
        system.set_matrix(A[n])
        system.set_rhs(b.copy())
        system.solve_system()
        y[:,n] = system.get_solution(as_function=False)[:,0]
    y_fn = Nodal(dofhandler=dQ,subforest_flag=mflag,data=y)
    
    if reference:
        return y_fn, M, K
    else:
        return y_fn


def finite_element_error(y_ref, y_app, M, K):
    """
    Compute the finite element errors 
    
        ||y_ref - y_app||^2 
        
    in the L2-norm and the H1-norm.
    """
    n_samples = y_ref.data().shape[1]
    
    dh = y_ref.dofhandler()
    x = dh.get_dof_vertices()
    n_dofs = dh.n_dofs()
    
    v_ref = y_ref.data()
    v_app = y_app.eval(x)
    v_err = v_ref - v_app
    norm_ref = np.sqrt(np.sum(v_ref*M.dot(v_ref)))
    
    l2_error = np.empty(n_samples)
    h1_error = np.empty(n_samples)
    for n in range(n_samples):
        l2_error[n] = np.sqrt(np.sum(v_err*M.dot(v_err)))/norm_ref
        h1_error[n] = np.sqrt(np.sum(v_err*K.dot(v_err)))/norm_ref
    
    return l2_error, h1_error


def assembly_time():
    """
    Test the amount of time
    """
    pass


def finite_element_convergence():
    """
    Test the finite element error for a representative sample
    """
    level_max = 10

    """
    c = Verbose()
    c.comment('Testing finite element convergence')
    
    n_samples = 100
    
    #
    # Computational mesh
    # 
    c.comment('Initializing finite element mesh.')
    mesh = Mesh1D()
    mesh.mark_region('left', lambda x:np.abs(x)<1e-10)
    mesh.mark_region('right', lambda x:np.abs(x-1)<1e-10)
    for level in range(level_max):
        mesh.cells.record(level)
        mesh.cells.refine()
    #
    # Finite element space
    # 
    eQ1 = QuadFE(mesh.dim(),'Q1')
    dQ1 = DofHandler(mesh, eQ1)
    dQ1.distribute_dofs()
    dQ1.set_dof_vertices()
    
    #
    # Sample random input
    #
    z = get_points(n_samples=n_samples)
    
    #
    # Compute reference solution
    #
    c.tic('Computing reference solution')
    y_ref, M, K = sample_state(mesh,dQ1,z,None,True)
    c.toc()
    
    l2_errors = np.empty((n_samples,level_max))
    h1_errors = np.empty((n_samples,level_max))
    
    plot = Plot()
    c.comment('Computing samples of finite element solutions')
    for level in range(level_max):
        c.tic('  Level %d'%(level))
        y_fem = sample_state(mesh,dQ1,z,level)
        c.toc()
        
        
        c.tic('Computing error')
        l2_error, h1_error = finite_element_error(y_ref, y_fem, M, K)
        l2_errors[:,level] = np.sqrt(l2_error)
        h1_errors[:,level] = np.sqrt(h1_error)
        c.toc()
        
    np.save('ex01a_l2_errors',l2_errors)
    np.save('ex01a_h1_errors',h1_errors)
    """
    
    l2_errors = np.load('ex01a_l2_errors.npy')
    h1_errors = np.load('ex01a_h1_errors.npy')
    
    # Plot Errors
    plt.figure(figsize=(4,3))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12) 
    
    h = 1/2**(np.arange(level_max))
    plt.loglog(h,h1_errors.transpose(),'.-k',linewidth=0.1)
    plt.loglog(h,l2_errors.transpose(),'.-b',linewidth=0.1)
    
    plt.plot([], '.-k', label=r'$\|\cdot\|_{H^1}$')
    plt.plot([], '.-b', label=r'$\|\cdot\|_{L^2}$')
    plt.legend()
    
    plt.grid()
    
    plt.xlabel(r'$h$')
    plt.ylabel(r'$\|y-\hat y\|$')
    
    plt.tight_layout()
    
    plt.savefig('ex01_fem_errors.pdf')
    
    
def sampling_error():
    """
    Test the sampling error
    """
    


if __name__=='__main__':
    finite_element_convergence()
    grid = make_grid(depth=9)
    z = get_points(mode='sg',grid=grid)
    
    
    n_samples = 1000
    z = get_points(n_samples=n_samples)
