"""

"""


import sys
if '/home/hans-werner/git/quadmesh/src/' not in sys.path:
    sys.path.append('/home/hans-werner/git/quadmesh/src/')

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats as stats

# Mesh
from mesh import Mesh1D

# Finite Elements
from fem import Basis, DofHandler, QuadFE
from function import Nodal, Explicit

# Plotting
from plot import Plot

# Gaussian random field
from gmrf import Covariance, GaussianField

from diagnostics import Verbose
from assembler import Form, Assembler, Kernel
from solver import LinearSystem

if __name__ == "__main__":
    comment = Verbose()
    comment.comment("Create Hierarchical Mesh")
    #
    # Hierarchical Mesh
    # 
    nl = 10; 
    mesh = Mesh1D(resolution=(1,), box = [0,1])
    mesh.record(0)
    for l in range(nl):
        print(f'Number of cells: {len(mesh.cells.get_leaves(subforest_flag=l))}')
        mesh.cells.refine()
        mesh.record(l+1)
    print(f'Number of cells: {len(mesh.cells.get_leaves(subforest_flag=nl))}')
    comment.tic("Plotting mesh")
    # Plot mesh
    plot = Plot(quickview=False)
    fig, ax = plt.subplots(nl+1,1, figsize=(4, 6))
    
    for  l in range(nl+1):
        ax[l] = plot.mesh(mesh,ax[l],subforest_flag=l)
        ax[l].set_title(f"Mesh Level {l}")
    plt.tight_layout()
    comment.toc()
    plt.show()
     
    
    # Create finite element system
    Q = QuadFE(1, 'Q1')
    dh = DofHandler(mesh, Q)
    dh.distribute_dofs() 
    phi = Basis(dh,'v')
    phi_x = Basis(dh,'vx')


    # Compute projection matrices
    comment.tic("Compute projection matrices")
    P = []
    for l in range(nl+1):
        problems = []

    # Define Gaussian random field
    comment.tic("Create Covariance")
    cov = Covariance(dh,name='gaussian',parameters={'sgm':1,'l':0.01})
    comment.toc()

    # Create Gaussian random field
    comment.tic("Create Gaussian random field")
    eta = GaussianField(dh.n_dofs(), covariance=cov)
    comment.toc()

    # Sample from the Gaussian random field
    comment.tic("Sample from Gaussian random field")
    n_samples = 100000
    eta_smpl = eta.sample(n_samples=n_samples)
    comment.toc()

    comment.tic("Define eta function")
    eta_fn = Nodal(basis=phi, data=eta_smpl[:,:10],dim=1)
    comment.toc()

    comment.tic("Plot eta function")
    plot = Plot(quickview=False)
    fig, ax = plt.subplots(1,1, figsize=(8, 4))
    for n in range(10):
        ax = plot.line(eta_fn, axis=ax, i_sample=n, plot_kwargs={'color':'black','alpha':0.5})
    ax.set_ylim(-4, 4)
    comment.toc()
    #plt.tight_layout()
    plt.show()

    # Compute the spatial average of the samples
    a = 0.4
    b = 0.6
    f_region = lambda x: (x >= a) & (x <= b)
    mesh.mark_region('integration_region',f_region,entity_type='cell')
    
    comment.tic("Compute spatial average")
    sgm = 1/60
    mu = 0.5
    norm = stats.norm(0,1)
    f_weight = lambda x,mu,sgm: np.exp(-(x-mu)**2/(2*sgm**2))    
    I_norm = np.sqrt(2*np.pi)*sgm*(norm.cdf((1-0)/2/sgm))-norm.cdf(-(1-0)/2/sgm)
    problem = [[Form(kernel=Kernel(Explicit(lambda x: f_weight(x,mu,sgm)/I_norm,dim=1)),test=phi)]]
    assembler = Assembler(problem, mesh)
    assembler.assemble()
    L = assembler.get_vector()
    comment.toc()

    plot = Plot(quickview=False)
    fig, ax = plt.subplots(1,1, figsize=(8, 4))
    ax = plot.mesh(mesh,axis=ax, regions=[('integration_region','cell')])
    ax.set_title("Integration region")
    #plt.show()



    x = dh.get_dof_vertices()
    print(x)
    print('Should be 1', np.sum(L))
    print('Should be 0.5', L.dot(x))
    fig, ax = plt.subplots(1,1, figsize=(8, 4))
    plt.plot(x,L,'.',color='black',alpha=0.5)
    #x = np.linspace(0,1,1000)
    #plt.plot(x,f_weight(x,mu,sgm),color='black',alpha=0.5)
    #plt.show()

    # Compute a histogram of the spatial averages
    fig, ax = plt.subplots(1,1, figsize=(8, 4))
    pi = L.dot(eta_smpl)
    plt.hist(pi, bins=50, density=True, alpha=0.5, color='black')
    plt.show()
