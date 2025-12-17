"""
One-dimensional advection-diffusion equation with hierarchical random parameters. 

    - d/dx (q(x) du/dx) + b(x) du/dx = f(x),  x in [0,1], 

with boundary conditions:

    u(0) = 0, u(1) = 1

where q(x) is a random diffusion coefficient and b(x) is a random advection coefficient. The random coefficients are modeled as Gaussian random fields with a given covariance structure. 

We sample from quantities of interest related to the solution, such as 

    (i) the spatial average over a given region [a,b] in the domain, or 
    (ii) the flux at the boundary.

TODO: Sample from the solution 
TODO: Sample low-complexity parameter system and compare distribution of the solution with the one from the high-complexity parameter system.
"""


import sys
if '/home/hans-werner/Documents/git/quadmesh/src/' not in sys.path:
    sys.path.append('/home/hans-werner/Documents/git/quadmesh/src/')

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import scipy.stats as stats

# Mesh
from mesh import Mesh1D

# Finite Elements
from fem import Basis, DofHandler, QuadFE
from function import Nodal, Explicit, Constant

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
     
    # Mark boundaries
    left_bnd =  lambda x: abs(x) < 1e-6
    right_bnd = lambda x: abs(x-1) < 1e-6
    mesh.mark_region('left', left_bnd, entity_type='vertex', on_boundary=True)
    mesh.mark_region('right', right_bnd, entity_type='vertex', on_boundary=True)

    # Create finite element system
    Q = QuadFE(1, 'Q1')
    dh = DofHandler(mesh, Q)
    dh.distribute_dofs() 
    phi = Basis(dh,'v')
    phi_x = Basis(dh,'vx')
    phi_l = Basis(dh,'v',subforest_flag=2)

    # Compute projection matrices
    comment.tic("Compute projection matrices")
    P = []
    for l in range(nl+1):
        problems = []

    # Define Gaussian random field
    comment.tic("Create Covariance")
    cov = Covariance(dh,name='exponential',parameters={'sgm':1,'l':0.01},subforest_flag=2)
    comment.toc()

    # Create Gaussian random field
    comment.tic("Create Gaussian random field")
    eta = GaussianField(dh.n_dofs(subforest_flag=2), covariance=cov)
    comment.toc()

    # Sample from the Gaussian random field
    comment.tic("Sample from Gaussian random field")
    n_samples = 100
    eta_smpl = eta.sample(n_samples=n_samples)
    comment.toc()


    # 
    # Plot specifications
    #
    plot = Plot(quickview=False)
    fig = plt.figure(figsize=(8, 8))


    comment.tic("Define eta function")
    eta_fn = Nodal(basis=phi_l, data=eta_smpl[:,:n_samples],dim=1,subforest_flag=2)
    comment.toc()

    #comment.tic("Plot q function")
    #ax_eta = plt.subplot2grid((6,6), (0,0), colspan=4, rowspan=3)
    #fig, ax = plt.subplots(1,1, figsize=(8, 4))
    #for n in range(50):
    #    ax_eta = plot.line(eta_fn, axis=ax_eta, i_sample=n, 
    #                   plot_kwargs={'color':'black','alpha':0.1})
    #ax.set_ylim(-4, 4)
    #comment.toc()
    #plt.tight_layout()

    # 
    # Compute the solution of the advection-diffusion equation
    # 

    # Problem parameters
    f =  Constant(1.0)  # Right-hand side function
    b =  Constant(0.5)  # Advection coefficient
    q =  Nodal(data=0.1 + np.exp(eta_smpl),basis=phi_l,dim=1,subforest_flag=2)  # Diffusion coefficient

    comment.tic("Plot q function")
    ax_q = plt.subplot2grid((6,6), (0,0), colspan=4, rowspan=3)
    for n in range(50):
        ax_q = plot.line(q, axis=ax_q, i_sample=n, 
                       plot_kwargs={'color':'black','alpha':0.1})
    ax_q.set_title(r"$q(x,\omega)$")
    ax_q.set_xlabel("x")
    comment.toc()

    # Define the weak form of the advection-diffusion equation
    FDiff = Form(Kernel(q), test=phi_x, trial=phi_x)
    FAdv = Form(Kernel(b), test=phi_x, trial=phi)
    FSource = Form(Kernel(f), test=phi)

    # Assemble the finite element system
    comment.tic("Assemble the finite element system")   
    problems = [FDiff, FAdv, FSource]
    assembler = Assembler(problems, mesh)

    # Add Dirichlet boundary conditions
    assembler.add_dirichlet('left', Constant(0.0))
    assembler.add_dirichlet('right', Constant(1.0))

    assembler.assemble()
    comment.toc()

    # Solve the linear system
    comment.tic("Solve the linear system")
    u = []
    for i in range(n_samples):
        print((i+1)/n_samples, end='\r')
        u.append(assembler.solve(i_matrix=i))
    u = np.array(u).transpose()
    comment.toc()

    u_fn = Nodal(basis=phi, data=u, dim=1)
    comment.tic("Plot solution")
    ax_u = plt.subplot2grid((6,6), (3,0), colspan=4, rowspan=3)
    
    for n in range(100):
        ax_u = plot.line(u_fn, axis=ax_u, i_sample=n,
                       plot_kwargs={'color':'black','alpha':0.1})
    #ax.set_ylim(0, 1)
    ax_u.set_title(r"u")
    ax_u.set_xlabel("x")
    comment.toc()
    #plt.tight_layout()
    #plt.show()

    #
    # Sample quantities of interest
    #
    
    # 1. Point evaluation of the solution at x = 0.5
    comment.tic("Point evaluation of the solution at x = 0.5")
    x_eval = 0.5
    qoi_mid = u_fn.eval(x_eval).ravel()    
    comment.toc()

    # Plot the point evaluation
    ax_qoi_mid = plt.subplot2grid((6,6), (0,4), colspan=2, rowspan=2)
    ax_qoi_mid.hist(qoi_mid, bins=50, density=True, alpha=0.5, color='black')
    ax_qoi_mid.set_title(r"$u(0.5,\omega)$")
    ax_qoi_mid.set_xlabel("u(0.5)")
    ax_qoi_mid.set_ylabel("Density")

    # 2. flux at the right boundary
    comment.tic("Flux at the right boundary")
    q_flux = q.eval(1.0)*u_fn.eval(1.0,derivative='vx') + b.eval(1.0)*u_fn.eval(1.0)
    q_flux = q_flux.ravel()  # Flatten the array for plotting
    comment.toc()

    # Plot the flux at the right boundary
    ax_flux = plt.subplot2grid((6,6), (2,4), colspan=2, rowspan=2)
    ax_flux.hist(q_flux, bins=50, density=True, alpha=0.5, color='black')
    ax_flux.set_title(r"Flux at the right boundary")
    ax_flux.set_xlabel(r"Flux at $x=1$")
    ax_flux.set_ylabel(r"Density")


    # Weighted spatial average of the solution over a given region [a,b]

    #a = 0.4
    #b = 0.6
    #f_region = lambda x: (x >= a) & (x <= b)
    #mesh.mark_region('integration_region',f_region,entity_type='cell')
    
    comment.tic("Weighted spatial average")
    sgm = 1/60
    mu = 0.5
    norm = stats.norm(0,1)
    f_weight = lambda x,mu,sgm: np.exp(-(x-mu)**2/(2*sgm**2))    
    I_norm = np.sqrt(2*np.pi)*sgm*(norm.cdf((1-0)/2/sgm))-norm.cdf(-(1-0)/2/sgm)
    problem = [[Form(kernel=Kernel(Explicit(lambda x: f_weight(x,mu,sgm)/I_norm,dim=1)),test=phi)]]
    assembler = Assembler(problem, mesh)
    assembler.assemble()
    L = assembler.get_vector()
    q_ave = L.dot(u)
    comment.toc()

    # Plot the weighted spatial average
    ax_qoi_ave = plt.subplot2grid((6,6), (4,4), colspan=2, rowspan=2)
    ax_qoi_ave.hist(q_ave, bins=50, density=True, alpha=0.5, color='black')
    ax_qoi_ave.set_title("Weighted spatial average")
    ax_qoi_ave.set_xlabel("Weighted spatial average of u")
    ax_qoi_ave.set_ylabel("Density")    
    plt.tight_layout()
    plt.show()
    #plot = Plot(quickview=False)
    #fig, ax = plt.subplots(1,1, figsize=(8, 4))
    #ax = plot.mesh(mesh,axis=ax, regions=[('integration_region','cell')])
    #ax.set_title("Integration region")
    #plt.show()

    """
    x = dh.get_dof_vertices()
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
    plt.show()#
    """