from assembler import Assembler
from assembler import Kernel
from assembler import Form
from fem import DofHandler
from fem import QuadFE
from fem import Basis
from function import Nodal
from gmrf import Covariance
from gmrf import GaussianField
from mesh import Mesh1D
from plot import Plot
import TasmanianSG
import time
from diagnostics import Verbose

# Built-in modules
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

 

"""
Consider the elliptic equation

-d/dx(e^q dy/dx) = f
y(0) = 1
y(1) = 0

on (0,1), where q is a normal gaussian field.

Split the diffusion coefficient into a low- and a high dimensional component

Use sparse grids to integrate the low dimensional approximation and Monte Carlo
for the high dimensional region. 

TODO: Finish
"""
comment = Verbose()

def sample_q0(V, lmd, d0, z0):
    """
    Inputs:
    
        V: double (n_dofs,n_dofs) eigenvectors of covariance matrix
        
        lmd: double, (n_dofs,) eigenvalues of covariance matrix
        
        d0: int, dimension of the low dimensional subspace
        
        z0: int, (d0,n_sample) Gaussian random vector
    """
    # Get KL basis
    V0 = V[:,:d0]
    Lmd0 = np.diag(np.sqrt(lmd[:d0]))
    
    # Form log of q0
    log_q0 = V0.dot(Lmd0.dot(z0)) 
    
    # Return result
    return np.exp(log_q0)


def sample_q_given_q0(q0, V, lmd, d0, z1):
    """
    Inputs:
    
        q0: double, single sample of in coarse parameter space
        
        V: double, eigenvectors of covariance
        
        lmd: double, eigenvalues of covariance
        
        d0: int, dimension of low dimensional q0
        
        z1: double, (d-d0,n_samples) samples of N(0,1)
    """
    # Get remaining expansion coefficients
    V1 = V[:,d0:]
    Lmd1 = np.diag(np.sqrt(lmd[d0:]))   
    
    # Form log of q1
    log_q1 = V1.dot(Lmd1.dot(z1))

    # Form log q given q0
    log_q0 = np.log(q0)
    log_q = log_q0[:,None] + log_q1
    
    # Return q given q0
    return np.exp(log_q)


def dJdq_per(q, dq, dofhandler, eps=1e-6):
    """
    Compute the directional derivative 
    
        dJ(q,dq) ~= (J(q+eps*dq)-J(q))/eps
    
    where 
    
        J(q) = exp(q(1))*u'(1)
        
        
    u solves 
    
        -(exp(q)*u'(x))' = 1, x in (0,1)
        u(0)=0, u(1)=1
        
    
    Inputs:
    
        q: vector, reference parameter
        
        dq: vector, parameter perturbation
        
        eps: 1e-7 finite difference parameter
    """
    # Compute qoi at reference parameter
    J_ref = sample_qoi(q, dofhandler)
    
    # Compute qoi at perturbed parameter
    J_per = sample_qoi(q+eps*dq, dofhandler)
    
    # Compute finite difference
    dJ = (J_per - J_ref)/eps
    
    return dJ 
    

def dJdq_adj(q, u, dq, dofhandler):
    """
    Compute the directional derivative dJ(q,dq) by solving the adjoint equation 
    
        -(exp(q)v')' = 0, 
        v(0)=0, v(1)=-1 
        
    and computing
    
        ( v, (exp(q)*dq(1)*u')' ) + exp(q(1))*dq(1)*u'(1) = -(exp(q)*dq u', v')
    
    """
    # Define finite element basis
    phi = Basis(dofhandler, 'u')
    phi_x = Basis(dofhandler, 'ux')
    
    # Define finite element functions
    expq_fn = Nodal(data=np.exp(q), basis=phi)
    u_fn = Nodal(data=u, basis=phi)
    ux_fn = Nodal(data=u, basis=phi_x)
    
    # Define adjoint equations
    adjoint_eqn = [Form(exp_qref, test=phi_x, trial=phi_x), Form(0, test=phi)]
    
    # Assembler
    assembler = Assembler(adjoint_eqn)
    
    # Apply Dirichlet BC
    assembler.add_dirichlet('left', 0)
    assembler.add_dirichlet('right',-1)
    
    # Assemble
    assembler.assemble()
    
    # Solve for adjoint 
    v = assembler.solve()
    
    # Get derivative
    vx_fn = Nodal(data=v, basis=phi)
    
    # Assemble 
    
    # Adjoint qoi
    k_adj = Kernel(f=[expq_fn, dq_fn, ux_fn, vx_fn], F=lambda eq,dq,ux,vx: -eq*dq*ux*vx)
    adj_qoi = [Form(k_adj)]
    
    assembler = Assembler(adj_qoi, mesh=dofhandler.mesh)

def dJdq_sen(q, u, dq, dofhandler):
    """
    Compute the directional derivative dJ(q,dq) by means of the sensitivity 
    equation.

        -(exp(q)*s')' = (exp(q)*dq*u')'
        s(0) = s(1) = 0
        
    and computing 
    
        dJ(q,dq) = exp(q(1))*dq(1)*u'(1) + exp(q(1))*s'(1)
        
    """
    # Finite Element Basis  
    phi = Basis(dofhandler, 'u')
    phi_x = Basis(dofhandler, 'ux')
    
    # Define finite element functions
    u_fn = Nodal(data=u, basis=phi)
    ux_fn = Nodal(data=u, basis=phi_x)
    expq_fn = Nodal(data=np.exp(q), basis=phi)
    dq_fn = Nodal(data=dq, basis=phi)
    
    # Define sensitivity equation
    ker_sen = Kernel(f=[expq_fn, dq, ux_fn], F=lambda eq, dq, ux: -eq*dq*ux)
    sensitivity_eqn = [Form(expq_fn, test=phi_x, trial=phi_x), 
                       Form(ker_sen, test=phi_x)]

    # Assembler
    assembler = Assembler(sensitivity_eqn, n_gauss=(6,36))
    
    # Apply Dirichlet Boundary conditions
    assembler.add_dirichlet('left',0)
    assembler.add_dirichlet('right',0)
    
    # Assemble system
    assembler.assemble()
    
    # Solve for sensitivity
    s_fn = Nodal(basis=phi)
    for i in range(dq_fn.n_samples()):
        # Solve for ith sensitivity
        s = assembler.solve(i_vector=i)
        s_fn.add_samples(s) 
        
    # Derivative of sensitivity
    sx_fn = Nodal(data=s_fn.data(), basis=phi_x) 
    
     # Sensitivity
    k_sens = Kernel(f=[expq_fn, dq_fn, ux_fn, sx_fn], 
                    F=lambda eq,dq,ux,sx: eq*dq*ux+eq*sx)
    sens_qoi = Form(k_sens, dmu='dv', flag='right')
    
    # Assemble 
    assembler = Assembler(sens_qoi, mesh=mesh)
    assembler.assemble()

    # Differential
    dJ = np.array([assembler.get_scalar(i_sample=i) \
                   for i in range(dq_fn.n_samples())])
    
    return dJ 
     
     
def gradient(q, u):
    """
    Compute the gradient dJdq at q and state u, using the adjoint formulation
    
        dJdq = (v,f), 
        
    where v is the solution of the adjoint equation
    
        -(q v')' = 0, v(0)=0, v(1)=-1
        
    and f = (q u')'.
    
    Inputs:
    
        q: Nodal, diffusion coefficient (possibly sampled)
        
        u: Nodal, corresponding state (possibly sampled)
        
    
    Output:
    
        dJdq: double, (nx,n_samples) array of (sampled) gradient vectors
    """
    #
    # Finite element basis 
    # 
    phi = u.basis()
    dh  = phi.dofhandler()
    phi_x = Basis(dh,'ux')
    
    adjoint_eqn = [Form(q, test=phi_x, trial=phi_x), Form(0, test=phi)]


def sample_qoi(q, dofhandler, return_state=False):
    """
    Compute the Quantity of Interest 
    
        J(u) = q(1)*u'(1),
        
    where u solves 
    
        -d/dx ( q du/dx) = 1
        u(0) = 0,  u(1) = 1
        
    for a sample of q's. 
    
    Inputs:
    
        q: Nodal, (n_dofs, n_samples) function representing the log porosity
        
        dofhandler: DofHandler
        
    """    
    # Basis
    phi   = Basis(dofhandler, 'v')
    phi_x = Basis(dofhandler, 'vx')
    n_dofs = phi.n_dofs()
        
    # Define problem
    qfn = Nodal(data=q, basis=phi)
    
    problem = [[Form(qfn, test=phi_x, trial=phi_x), Form(1, test=phi)],
               [Form(qfn, test=phi_x, dmu='dv', flag='right')]]
    
    # Define assembler
    assembler = Assembler(problem)
    
    # Incorporate Dirichlet conditions 
    assembler.add_dirichlet('left',0)
    assembler.add_dirichlet('right',1)
    
    n_samples = qfn.n_subsample()
    
    # Assemble system
    assembler.assemble()
    
    if return_state:
        U = np.empty((n_dofs,n_samples))
        
    J = np.zeros(n_samples)
    for i in range(n_samples):
        # Solve system
        u = assembler.solve(i_problem=0, i_matrix=i, i_vector=0)
        
        # Compute quantity of interest
        J[i] = u.dot(assembler.get_vector(1,i))
        
        if return_state:
            U[:,i] = u
    
    if return_state:
        return J,U 
    else:
        return J


def sensitivity_sample_qoi(exp_q,dofhandler):
    """
    Sample QoI by means of Taylor expansion
    
        J(q+dq) ~= J(q) + dJdq(q)dq
    """
    # Basis
    phi   = Basis(dofhandler, 'v')
    phi_x = Basis(dofhandler, 'vx')
        
    # Define problem
    exp_q_fn = Nodal(data=exp_q, basis=phi)
    
    primal = [Form(exp_q_fn, test=phi_x, trial=phi_x), Form(1, test=phi)]
    adjoint = [Form(exp_q_fn, test=phi_x, trial=phi_x), Form(0, test=phi)]
    qoi = [Form(exp_q_fn, test=phi_x)]
    problems = [primal, adjoint, qoi] 
    
    # Define assembler
    assembler = Assembler(problems)
    
    #
    # Dirichlet conditions for primal problem 
    #
    assembler.add_dirichlet('left', 0, i_problem=0)
    assembler.add_dirichlet('right', 1, i_problem=0)
    
    # Dirichlet conditions for adjoint problem
    assembler.add_dirichlet('left',0, i_problem=1)
    assembler.add_dirichlet('right',-1,i_problem=1)
        
    # Assemble system
    assembler.assemble()
    
    # Compute solution and qoi at q (primal)
    u = assembler.solve(i_problem=0)
    
    # Compute solution of the adjoint problem
    v = assembler.solve(i_problem=1)
    
    # Evaluate J
    J = u.dot(assembler.get_vector(2))
    
    #
    # Assemble gradient 
    # 
    ux_fn = Nodal(data=u, basis=phi_x)
    vx_fn = Nodal(data=v, basis=phi_x)
    
    
    k_int = Kernel(f=[exp_q_fn, ux_fn, vx_fn],
                   F=lambda exp_q, ux, vx: exp_q*ux*vx)
    
    problem = [Form(k_int, test=phi)]
    
    assembler = Assembler(problem)
    assembler.assemble()
    dJ = -assembler.get_vector()
    return dJ
    
        
def test00_finite_elements():
    """
    
    """
    # 
    # Construct reference solution
    #
    plot = Plot(quickview=False)
    
    # Mesh 
    mesh = Mesh1D(resolution=(2**11,))
    mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
    mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)
    
    # Element
    Q1 = QuadFE(mesh.dim(), 'Q1')
    dQ1 = DofHandler(mesh, Q1)
    dQ1.distribute_dofs()
    
    # Basis
    phi = Basis(dQ1, 'v')
    phi_x = Basis(dQ1, 'vx')
    
    #
    # Covariance
    # 
    cov = Covariance(dQ1, name='gaussian', parameters={'l':0.05})
    cov.compute_eig_decomp()
    lmd, V = cov.get_eig_decomp()
    d = len(lmd)
    
    #
    # Sample and plot full dimensional parameter and solution  
    # 
    n_samples = 1
    z = np.random.randn(d,n_samples)
    q_ref = sample_q0(V, lmd, d, z)
    
    print(q_ref.shape)
    
    # Define finite element function
    q_ref_fn = Nodal(data=q_ref, basis=phi)
    problem = [[Form(q_ref_fn, test=phi_x, trial=phi_x), Form(1, test=phi)],
               [Form(q_ref_fn, test=phi_x, dmu='dv', flag='right')]]
    
    # Define assembler
    assembler = Assembler(problem)
    
    # Incorporate Dirichlet conditions 
    assembler.add_dirichlet('left',0)
    assembler.add_dirichlet('right',1)
    
    # Assemble system
    assembler.assemble()
    
    # Solve system
    u_ref = assembler.solve()
        
    # Compute quantity of interest
    J_ref = u_ref.dot(assembler.get_vector(1))
    
    # Plot 
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    u_ref_fn = Nodal(basis=phi,data=u_ref)
    
    ax = plot.line(u_ref_fn, axis=ax)
    
    n_levels = 10
    J = np.zeros(n_levels)
    for l in range(10):
        comment.comment('level: %d'%(l))

        #
        # Mesh
        #
        mesh = Mesh1D(resolution=(2**l,))
        mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
        mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)
    
        #
        # Element
        #
        Q1 = QuadFE(mesh.dim(), 'Q1')
        dQ1 = DofHandler(mesh, Q1)
        dQ1.distribute_dofs()
        
        #
        # Basis
        # 
        phi = Basis(dQ1, 'v')
        phi_x = Basis(dQ1, 'vx')
        
        # Define problem
        problem = [[Form(q_ref_fn, test=phi_x, trial=phi_x), Form(1, test=phi)],
                   [Form(q_ref_fn, test=phi_x, dmu='dv', flag='right')]]

        assembler = Assembler(problem)
           
        # Incorporate Dirichlet conditions 
        assembler.add_dirichlet('left',0)
        assembler.add_dirichlet('right',1)
    
        assembler.assemble()
        A = assembler.get_matrix()
        print('A shape', A.shape)
        
        u = assembler.solve()
        J[l] = u.dot(assembler.get_vector(1))
        
        print(u.shape)
        print(phi.n_dofs())
        ufn = Nodal(basis=phi, data=u)
        ax = plot.line(ufn, axis=ax)
        
        
    plt.show()
    #
    # Plots
    #
    # Formatting
    plt.rc('text', usetex=True)
    
    # Figure sizes
    fs2 = (3,2)
    fs1 = (4,3)
    
    print(J_ref)
    print(J)
    
    #
    # Plot truncation error for mean and variance of J
    # 
    
    fig = plt.figure(figsize=fs2)
    ax = fig.add_subplot(111)
    
    err = np.array([np.abs(J[i]-J_ref) for i in range(n_levels)])
    h = np.array([2**(-l) for l in range(n_levels)])
    plt.loglog(h, err,'.-')
    
    ax.set_xlabel(r'$h$')
    ax.set_ylabel(r'$|J-J^h|$')
    plt.tight_layout()
    fig.savefig('fig/ex02_gauss_fem_error.eps')
     

def test00_sensitivity():
    """
    Check that the sensitivity calculation works. Compare 
    
        J(q+eps*dq) - J(q) ~= eps*dJ^T dq
    """  
    #  
    # Mesh
    #
    mesh = Mesh1D(resolution=(20,))
    mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
    mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)
    
    #
    # Element
    #
    Q = QuadFE(mesh.dim(), 'Q3')
    dQ = DofHandler(mesh, Q)
    dQ.distribute_dofs()
    nx = dQ.n_dofs()
    x = dQ.get_dof_vertices() 

    #
    # Basis
    # 
    phi = Basis(dQ, 'v')
    phi_x = Basis(dQ, 'vx')
    
    #
    # Parameters 
    # 
    
    # Reference q
    q_ref = np.zeros(nx)
    
    # Perturbation
    dq = np.ones(nx)
    
    # Perturbed q
    n_eps = 10  # Number of refinements
    epsilons = [10**(-l) for l in range(n_eps)]
    q_per = np.empty((nx,n_eps)) 
    for i in range(n_eps):
        q_per[:,i] = q_ref + epsilons[i]*dq
    
    # Define finite element function
    exp_qref = Nodal(data=np.exp(q_ref), basis=phi)
    exp_qper = Nodal(data=np.exp(q_per), basis=phi)
    
    #
    # PDEs 
    # 
    
    # 1. State Equation
    state_eqn = [Form(exp_qref, test=phi_x, trial=phi_x), Form(1, test=phi)]
    state_dbc = {'left':0, 'right':1}
    
    # 2. Perturbed Equation
    perturbed_eqn = [Form(exp_qper, test=phi_x, trial=phi_x), Form(1, test=phi)]
    perturbed_dbc = {'left':0, 'right':1}
    
    # 3. Adjoint Equation
    adjoint_eqn = [Form(exp_qref, test=phi_x, trial=phi_x), Form(0, test=phi)]
    adjoint_dbc = {'left':0,'right':-1}
    
    # Combine 
    eqns = [state_eqn, perturbed_eqn, adjoint_eqn]
    bcs  = [state_dbc, perturbed_dbc, adjoint_dbc]
    
    #
    # Assembly
    # 
    assembler = Assembler(eqns)
    
    # Boundary conditions
    for i, bc in zip(range(3), bcs):
        for loc, val in bc.items():
            assembler.add_dirichlet(loc, val, i_problem=i)
    
    # Assemble
    assembler.assemble()
    
    #
    # Solve
    # 
    
    # Solve state
    ur = assembler.solve(i_problem=0)
    u_ref = Nodal(data=ur, basis=phi)
    ux_ref = Nodal(data=ur, basis=phi_x)
    
    # Solve perturbed state
    u_per = Nodal(basis=phi)
    ue_per = Nodal(basis=phi)
    for i in range(n_eps):
        # FEM solution
        up = assembler.solve(i_problem=1, i_matrix=i)
        u_per.add_samples(up)
        
        # Exact perturbed solution
        eps = epsilons[i]
        ue_per.add_samples(0.5*np.exp(-eps)*(x-x**2)+x)
        
    ux_per = Nodal(data=u_per.data(), basis=phi_x)
    
    # Solve adjoint equation
    v = assembler.solve(i_problem=2)
    v_adj = Nodal(data=v, basis=phi)
    vx_adj = Nodal(data=v, basis=phi_x)
    
    #
    # Check against exact solution
    #
    ue = -0.5*x**2 + 1.5*x 
    ve = -x
    
    assert np.allclose(ue, u_ref.data())
    assert np.allclose(ve, v_adj.data())
    assert np.allclose(ue_per.data(), u_per.data())
    
    
    #
    # Quantities of Interest
    #
    
    # Reference
    k_ref = Kernel(f=[exp_qref, ux_ref], F=lambda eq,ux: eq*ux)
    ref_qoi = [Form(k_ref, dmu='dv', flag='right')]
    
    # Perturbed
    k_per = Kernel(f=[exp_qper, ux_per], F=lambda eq,ux: eq*ux)
    per_qoi = [Form(k_per, dmu='dv', flag='right')]
    
    # Adjoint
    k_adj = Kernel(f=[exp_qref, ux_ref, vx_adj], F=lambda eq,ux,vx: -eq*ux*vx)
    adj_qoi = [Form(k_adj, test=phi)]
    
    qois = [ref_qoi, per_qoi, adj_qoi]
    
    # Assemble 
    assembler = Assembler(qois)
    assembler.assemble()
    
    # Evaluate
    J_ref = assembler.get_scalar(0)
    J_per = []
    for i in range(n_eps):
        J_per.append(assembler.get_scalar(1,i))
    
    # Finite difference approximation
    dJ = []
    for eps, J_p in zip(epsilons, J_per):
        dJ.append((J_p-J_ref)/eps)
    
    # Adjoint differential
    dJ_adj = assembler.get_vector(2).dot(dq)
    
    #
    # Check against exact qois
    #
    
    # Check reference sol
    Je_ref = 0.5
    assert np.allclose(Je_ref,J_ref)
    
    # Check perturbed cost 
    Je_per = -0.5 + np.exp(np.array(epsilons))
    assert np.allclose(Je_per,J_per)
    
    # Check derivative by the adjoint equation
    dJdq = 1  
    assert np.allclose(dJ_adj,dJdq)  
    
    """
    print(J_ref)
    print(J_per)
    print(dJ)
    print(dJ_adj)
    """
    """
    #
    # Covariance
    # 
    cov = Covariance(dQ1, name='gaussian', parameters={'l':0.05})
    cov.compute_eig_decomp()
    lmd, V = cov.get_eig_decomp()
    d = len(lmd)
    
    #
    # Two samples of q  
    # 
    z = np.random.randn(d,2)
    expq = sample_q0(V, lmd, d, z)
    q = np.log(expq)
    
    #
    # Reference and perturbations
    #
    q_ref = q[:,0]  # reference log permeability  
    #q_ref = np.zeros(nx)
    dq = q[:,1]-q[:,0]  # perturbation direction
    
    # Generate perturbed problems
    
    """
    
    """
    #
    # Plot
    # 
    q_min, q_max = np.min(q), np.max(q)
    
    q_ref_fn = Nodal(basis=phi, data=q_ref)
    q_per_fn = Nodal(basis=phi, data=q_per)
    dq_fn = Nodal(basis=phi, data=dq)
    
    fig, ax = plt.subplots(2,1)
    plt.autoscale(enable=True,tight=True)
    plot = Plot(quickview=False)
    ax[0] = plot.line(q_ref_fn, axis=ax[0], 
                   plot_kwargs={'color':'k', 'linestyle':'solid'})
    ax[0] = plot.line(q_per_fn, axis=ax[0], i_sample=np.arange(n_eps), 
                   plot_kwargs={'color':'k', 'linestyle':'dashed'})
    ax[0].set_ylim([q_min, q_max])
    
    ax[1] = plot.line(dq_fn, axis=ax[1], 
                      plot_kwargs={'color':'k','linestyle':'solid'})
    plt.show()
    """
    
    """
    
    problem = [[Form(exp_qref, test=phi_x, trial=phi_x), Form(1, test=phi)],
               [Form(exp_qref, test=phi_x, dmu='dv', flag='right')],
               [Form(exp_qper, test=phi_x, trial=phi_x), Form(1, test=phi)],
               [Form(exp_qper, test=phi_x, dmu='dv', flag='right')],
               [Form(exp_qref, test=phi_x, trial=phi_x), Form(0, test=phi)]]
    
    # Define assembler
    assembler = Assembler(problem)
    
    # Incorporate Dirichlet conditions 
    for i in [0,2]:
        assembler.add_dirichlet('left',0, i_problem=i)
        assembler.add_dirichlet('right',1, i_problem=i)
    
    # Boundary conditions for the adjoint equation
    assembler.add_dirichlet('left',0, i_problem=4)
    assembler.add_dirichlet('right',-1, i_problem=4)
    
    # Assemble system
    assembler.assemble()

    # Compute reference solution
    u_ref = Nodal(basis=phi)
    u_ref.add_samples(assembler.solve(i_problem=0))
    
    # Compute reference qoi
    Ibnd = assembler.get_vector(i_problem=1)
    J_ref = u_ref.data().ravel().dot(Ibnd)
    
    # Compute adjoint solution
    u_adj = Nodal(basis=phi)
    u_adj.add_samples(assembler.solve(i_problem=4))
    
    
    #
    # Solve for each perturbation
    # 
    u_per = Nodal(basis=phi)
    J_per = np.zeros(n_eps)
    dJ = np.zeros(n_eps)
    for i in range(n_eps):
        
        # Compute perturbed solution
        up = assembler.solve(i_problem=2, i_matrix=i)
        u_per.add_samples(up)
        
        # Compute perturbed quantity of interest
        Ibnd = assembler.get_vector(i_problem=3, i_sample=i)
        J_per[i] = up.dot(Ibnd)
        
        # Compute the finite difference gradient
        dJ[i] = (J_per[i]-J_ref)/epsilons[i]
    
    plt.semilogx(epsilons, dJ,'.-')
    
    print(dJ)
    #plot = Plot()
    x = np.linspace(0,1,201)
    #assert np.allclose(u_ref.data().ravel(),-0.5*x**2 + 1.5*x)
    
    
    
    #plt.plot(x,ue)
    #plt.show()
    #plot.line(u_ref)
    #plot.line(u_adj)
    
    # Assemble
    ux_ref = Nodal(data=u_ref.data(), basis=phi_x)
    ux_adj = Nodal(data=u_adj.data(), basis=phi_x) 
    
    k = Kernel(f=[exp_qref, ux_ref, ux_adj],  
               F=lambda eq,ux,vx: -eq*ux*vx)
    
    #plt.plot(x,exp_qref.data(), x,u_ref.data(), x, ue, '--', x,u_adj.data())
    plt.show()
    
    problem = [Form(kernel=k, test=phi)]
    assembler = Assembler(problem, mesh=mesh)
    assembler.assemble()
    
    #print(assembler.get_scalar())
    dJa = dq.dot(assembler.get_vector())
    print(dJa)    
    """
    
def test01_problem():
    """
    Illustrate the problem:  Plot sample paths of the input q, of the output, 
        and histogram of the QoI.
    """
    
    #
    # Computational Mesh
    #
    mesh = Mesh1D(resolution=(100,))
    mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
    mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)
    
    #
    # Element
    #
    Q1 = QuadFE(mesh.dim(), 'Q1')
    dQ1 = DofHandler(mesh, Q1)
    dQ1.distribute_dofs()
    
    #
    # Basis
    # 
    phi = Basis(dQ1, 'v')
    phi_x = Basis(dQ1, 'vx')
    
    
    #
    # Covariance
    # 
    cov = Covariance(dQ1, name='gaussian', parameters={'l':0.05})
    cov.compute_eig_decomp()
    lmd, V = cov.get_eig_decomp()
    d = len(lmd)
    
    #
    # Sample and plot full dimensional parameter and solution  
    # 
    n_samples = 20000
    z = np.random.randn(d,n_samples)
    q = sample_q0(V, lmd, d, z)
    
    # Define finite element function
    qfn = Nodal(data=q, basis=phi)
    problem = [[Form(qfn, test=phi_x, trial=phi_x), Form(1, test=phi)],
               [Form(qfn, test=phi_x, dmu='dv', flag='right')]]
    
    # Define assembler
    assembler = Assembler(problem)
    
    # Incorporate Dirichlet conditions 
    assembler.add_dirichlet('left',0)
    assembler.add_dirichlet('right',1)
    
    comment.tic('assembly')
    # Assemble system
    assembler.assemble()
    comment.toc()


    comment.tic('solver')
    ufn = Nodal(basis=phi,data=None)
    J = np.zeros(n_samples) 
    for i in range(n_samples):
        # Solve system
        u = assembler.solve(i_problem=0, i_matrix=i, i_vector=0)
        
        # Compute quantity of interest
        J[i] = u.dot(assembler.get_vector(1,i))
        
        # Update sample paths
        ufn.add_samples(u)
    comment.toc()
    
    #
    # Plots
    # 
    """
    # Formatting
    plt.rc('text', usetex=True)
    
    # Figure sizes
    fs2 = (3,2)
    fs1 = (4,3)
    
    plot = Plot(quickview=False)
    plot_kwargs = {'color':'k', 'linewidth':0.05}
    
    #
    # Plot qfn
    # 
    
    # Figure 
    fig = plt.figure(figsize=fs2)
    ax = fig.add_subplot(111)
    ax = plot.line(qfn, axis=ax, 
                   i_sample=np.arange(100), 
                   plot_kwargs=plot_kwargs)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$q$')
    plt.tight_layout()
    fig.savefig('fig/ex02_gauss_qfn.eps')
    plt.close()
    
    #
    # Plot ufn
    # 
    fig = plt.figure(figsize=fs2)
    ax = fig.add_subplot(111)
    ax = plot.line(ufn, axis=ax, 
                   i_sample=np.arange(100), 
                   plot_kwargs=plot_kwargs)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u$')
    plt.tight_layout()
    fig.savefig('fig/ex02_gauss_ufn.eps')
    plt.close()
    """
    
    # Formatting
    plt.rc('text', usetex=True)
    
    # Figure sizes
    fs2 = (3,2)
    fs1 = (4,3)
    
    fig = plt.figure(figsize=fs2)
    ax = fig.add_subplot(111)
    plt.hist(J, bins=100, density=True)
    ax.set_xlabel(r'$J(u)$')
    plt.tight_layout()
    fig.savefig('fig/ex02_gauss_jhist.eps')
    
    
def test02_reference():
    """
    Convergence rate of MC
    """
    generate = False
    #
    # Computational Mesh
    #
    mesh = Mesh1D(resolution=(100,))
    mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
    mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)
    
    #
    # Element
    #
    Q1 = QuadFE(mesh.dim(), 'Q1')
    dQ1 = DofHandler(mesh, Q1)
    dQ1.distribute_dofs()
    
    #
    # Covariance
    # 
    cov = Covariance(dQ1, name='gaussian', parameters={'l':0.05})
    cov.compute_eig_decomp()
    lmd, V = cov.get_eig_decomp()
    d = len(lmd)
    
    #
    # Generate random sample for J
    #  
    
    n_samples = 1000000
    
    if generate:
        n_batches = 1000
        batch_size = n_samples//n_batches
        J = np.empty(n_samples)
        for i in range(n_batches):
            
            # Sample diffusion coefficient
            z = np.random.randn(d,n_samples//n_batches)
            q = sample_q0(V,lmd,d,z)
            
            # Evaluate quantity of interest
            J[(i)*batch_size:(i+1)*batch_size] = sample_qoi(q,dQ1)
            
            # Save current update to file
            np.save('./data/j_mc.npy',J)
        
    #
    # Process data 
    # 
    
    # Load MC samples
    J = np.load('data/j_mc.npy')
    
    # Compute sample mean and variance of J
    EX = np.mean(J)
    VarX = np.var(J)
    
    print(EX, VarX)


def test03_truncation():
    """
    Investigate the error in truncation level
    """
    generate = False
    
    mesh = Mesh1D(resolution=(100,))
    mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
    mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)
    
    #
    # Element
    #
    Q1 = QuadFE(mesh.dim(), 'Q1')
    dQ1 = DofHandler(mesh, Q1)
    dQ1.distribute_dofs()
    
    #
    # Basis
    # 
    phi = Basis(dQ1, 'v')
    phi_x = Basis(dQ1, 'vx')
    
    
    #
    # Covariance
    # 
    cov = Covariance(dQ1, name='gaussian', parameters={'l':0.05})
    cov.compute_eig_decomp()
    lmd, V = cov.get_eig_decomp()
    d = len(lmd)
    
    # Truncation levels
    truncation_levels = [1,5,10,20,50]
    
    n_samples = 1000000
    if generate:
        n_batches = 1000
        batch_size = n_samples//n_batches
        
        for d0 in truncation_levels:
            comment.tic('d = %d'%(d0))
            J = np.empty(n_samples)
            for i in range(n_batches):
                # Print progress
                #print('.',end='')
                
                # Sample diffusion coefficient
                z = np.random.randn(d0,batch_size)
                q = sample_q0(V,lmd,d0,z)
                
                # Evaluate quantity of interest
                J[(i)*batch_size:(i+1)*batch_size] = sample_qoi(q,dQ1)
                
                # Save current update to file
                np.save('./data/j_%d_mc.npy'%(d0),J)
            comment.toc()
    
    #
    # Compute estimates and errors 
    # 
    n_levels = len(truncation_levels)
    mean = []
    var  = []
    for d0 in truncation_levels:
        J = np.load('data/j_%d_mc.npy'%(d0))
        
        # Compute mean and variance
        mean.append(np.mean(J))
        var.append(np.var(J))
    
    # Load reference
    J = np.load('data/j_mc.npy')
    mean_ref = np.mean(J)
    var_ref = np.var(J)
    
    #truncation_levels.append(101)
    err_mean = [np.abs(mean[i]-mean_ref) for i in range(n_levels)]
    err_var  = [np.abs(var[i]-var_ref) for i in range(n_levels)]
    
    
    #
    # Plots
    #
    # Formatting
    plt.rc('text', usetex=True)
    
    # Figure sizes
    fs2 = (3,2)
    fs1 = (4,3)
    
    #
    # Plot truncation error for mean and variance of J
    # 
    
    fig = plt.figure(figsize=fs2)
    ax = fig.add_subplot(111)
    
    plt.semilogy(truncation_levels, err_mean,'.-', label='mean')
    plt.semilogy(truncation_levels, err_var, '.--', label='variance')    
    plt.legend()

    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$\mathrm{Error}$')
    plt.tight_layout()
    fig.savefig('fig/ex02_gauss_trunc_error.eps')
    
    #
    # Plot estimated mean and variance
    #
    
    fig = plt.figure(figsize=fs2)
    ax = fig.add_subplot(111)
    
    truncation_levels.append(101)
    mean.append(mean_ref)
    var.append(var_ref)
    plt.plot(truncation_levels, mean,'k.-', label='mean')
    plt.plot(truncation_levels, var, 'k.--', label='variance')    
    plt.legend()

    ax.set_xlabel(r'$k$')
    plt.tight_layout()
    fig.savefig('fig/ex02_gauss_trunc_stats.eps')
    
    
def test04_sparse_grid():
    """
    Test sparse grid
    """
    #
    # Computational mesh
    # 
    mesh = Mesh1D(resolution=(100,))
    mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
    mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)
    
    #
    # Element
    #
    Q1 = QuadFE(mesh.dim(), 'Q1')
    dQ1 = DofHandler(mesh, Q1)
    dQ1.distribute_dofs()
    
    #
    # Covariance
    # 
    cov = Covariance(dQ1, name='gaussian', parameters={'l':0.05})
    cov.compute_eig_decomp()
    lmd, V = cov.get_eig_decomp()
    
    # Truncation levels
    truncation_levels = [1,5,10, 20]
    
    # Formatting
    plt.rc('text', usetex=True)
    
    # Set figure and axis
    fs2 = (3,2)
    fs1 = (4,3)
    
    # For mean
    fig1 = plt.figure(figsize=fs1)
    ax1 = fig1.add_subplot(111)
    
    # For variance
    fig2 = plt.figure(figsize=fs1)
    ax2 = fig2.add_subplot(111)
    
    for d0 in truncation_levels:
        J = []
        mean = []
        var = []
        n = []
        for depth in range(5):
            #
            # Construct Sparse Grid
            #
            grid = TasmanianSG.TasmanianSparseGrid()
            dimensions = d0
            outputs = 1
            type = 'level'
            rule = 'gauss-hermite'
            grid.makeGlobalGrid(dimensions, outputs, depth, type, rule)
            
            # Get Sample Points
            zzSG = grid.getPoints()
            zSG = np.sqrt(2)*zzSG                # transform to N(0,1)
            
            wSG = grid.getQuadratureWeights()
            wSG /= np.sqrt(np.pi)**d0     # normalize weights

            n0 = grid.getNumPoints()
            n.append(n0)
            
            #
            # Sample input parameter
            # 
            q0 = sample_q0(V,lmd,d0,zSG.T)
            J = sample_qoi(q0, dQ1)
            
            EJ = np.sum(wSG*J)
            VJ = np.sum(wSG*(J**2)) - EJ**2
            mean.append(EJ)
            var.append(VJ)
            
        J_mc = np.load('data/j_%d_mc.npy'%(d0))
        
        # Compute mean and variance
        mean_mc = np.mean(J_mc)
        var_mc = np.var(J_mc)
        
        # Plot mean error
        mean_err = [np.abs(mean[i]-mean_mc) for i in range(5)]
        ax1.loglog(n, mean_err, '.-.', label=r'$k=%d$'%(d0))
        ax1.set_xlabel(r'$n$')
        ax1.set_ylabel(r'$\mathrm{Error}$')
        ax1.legend()
        fig1.tight_layout()
        
        # Plot variance error
        var_err = [np.abs(var[i]-var_mc) for i in range(5)]
        ax2.loglog(n, var_err, '.-.', label=r'k=%d'%(d0))
        ax2.set_xlabel(r'$n$')
        ax2.set_ylabel(r'$\mathrm{Error}$')
        ax2.legend()
        fig2.tight_layout()
        
        
    fig1.savefig('fig/ex02_gauss_sg_mean_error.eps')
    fig2.savefig('fig/ex02_gauss_sg_var_error.eps')    
        
    
def test05_conditioning():
    """
    Obtain an estimate of J using sparse grids on the coarse scale and MC as a
    correction. 
    """     
    #
    # Computational mesh
    # 
    mesh = Mesh1D(resolution=(100,))
    mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
    mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)
    
    #
    # Element
    #
    Q1 = QuadFE(mesh.dim(), 'Q1')
    dQ1 = DofHandler(mesh, Q1)
    dQ1.distribute_dofs()
    
    #
    # Covariance
    # 
    cov = Covariance(dQ1, name='gaussian', parameters={'l':0.05})
    cov.compute_eig_decomp()
    lmd, V = cov.get_eig_decomp()
    d = len(lmd)
    
    # Fix coarse truncation level
    d0 = 10
    
    #
    # Build Sparse Grid
    #
    grid = TasmanianSG.TasmanianSparseGrid()
    dimensions = d0
    outputs = 1
    depth = 2
    type = 'level'
    rule = 'gauss-hermite'
    grid.makeGlobalGrid(dimensions, outputs, depth, type, rule)
    
    # Sample Points
    zzSG = grid.getPoints()
    zSG = np.sqrt(2)*zzSG                # transform to N(0,1)
    
    # Quadrature Weights
    wSG = grid.getQuadratureWeights()
    wSG /= np.sqrt(np.pi)**d0     # normalize weights
    
    # Number of grid points
    n0 = grid.getNumPoints()
    
    #
    # Sample low dimensional input parameter
    # 
    q0 = sample_q0(V,lmd,d0,zSG.T)
    J0 = sample_qoi(q0, dQ1)
    
    # Compute sparse grid mean and variance
    EJ0 = np.sum(wSG*J0)
    VJ0 = np.sum(wSG*(J0**2)) - EJ0**2
    
    J = np.load('data/j_mc.npy')
    mean_ref = np.mean(J)
    var_ref = np.var(J)
    
    # Record errors
    mean_err = [np.abs(EJ0-mean_ref)]
    var_err = [np.abs(VJ0-var_ref)]
    
    for n_samples in [10,100,1000]:
        mean_Jg0 = 0
        var_Jg0 = 0
        for i in range(n0):
            z  = np.random.randn(d-d0,n_samples)
            qg0 = sample_q_given_q0(q0[:,i], V, lmd, d0, z)
            Jg0 = sample_qoi(qg0, dQ1)
            
            mean_Jg0 += wSG[i]*np.mean(Jg0)
            
        mean_err.append(np.abs(mean_Jg0-mean_ref))
    
    
    # Formatting
    plt.rc('text', usetex=True)
    
    # Figure sizes
    fs2 = (3,2)
    fs1 = (4,3)
    
    fig = plt.figure(figsize=fs2)
    ax = fig.add_subplot(111)
    ax.semilogy([0,10,100,1000], mean_err,'.-')
    ax.set_xlabel(r'$n$')
    ax.set_ylabel(r'$\mathrm{Error}$')
    
    fig.tight_layout()
    fig.savefig('fig/ex02_gauss_hyb_mean_err.eps')
    
    """
    #
    # Plot conditional variances
    #
    fig = plt.figure(figsize=fs2)
    ax = fig.add_subplot(111)
    ax.hist(varJg,bins=30, density=True)
    ax.set_xlabel(r'$\sigma_{J|q_0}^2$')
    fig.tight_layout()
    fig.savefig('fig/ex02_gauss_cond_var.eps')
    """
    
    """     
    d0 = 20
    n_samples = 1
    z0 = np.random.randn(d0,n_samples)
    
    
    d = len(lmd)
    z1 = np.random.randn(d-d0,50000)
    q1 = sample_q_given_q0(q0, V, lmd, d0, z1)
    
    m = dQ1.n_dofs()
    x = dQ1.get_dof_vertices()
    plt.plot(x,q0,'k',linewidth=1)
    plt.plot(x,q1,'k', linewidth=0.1)
    plt.show()
    
    J = sample_qoi(q1,dQ1)
    """
def test06_sensitivity_sample():
    """
    Test function sensitivity sample
    """
    pass

def test06_linearization():
    """
    Compute samples on fine grid via the linearization
    """
    plot = Plot()
    #
    # Computational mesh
    # 
    mesh = Mesh1D(resolution=(100,))
    mesh.mark_region('left', lambda x: np.abs(x)<1e-10)
    mesh.mark_region('right', lambda x: np.abs(x-1)<1e-10)
    
    #
    # Element
    #
    Q1 = QuadFE(mesh.dim(),'Q1')
    dQ1 = DofHandler(mesh, Q1)
    dQ1.distribute_dofs()
    nx = dQ1.n_dofs()
    x = dQ1.get_dof_vertices()
    
    
    Q3 = QuadFE(mesh.dim(),'Q3')
    dQ3 = DofHandler(mesh,Q3)
    dQ3.distribute_dofs()
    
    
    #
    # Basis
    # 
    phi = Basis(dQ1,'u')
    phi_x = Basis(dQ1, 'ux')
    psi = Basis(dQ3, 'u')
    psi_x = Basis(dQ3, 'ux')
    
    #
    # Covariance
    # 
    cov = Covariance(dQ1, name='gaussian', parameters={'l':0.05})
    cov.compute_eig_decomp()
    lmd, V = cov.get_eig_decomp()
    d = len(lmd)
    
    # Fix coarse truncation level
    d0 = 10
    
    #
    # Build Sparse Grid
    #
    grid = TasmanianSG.TasmanianSparseGrid()
    dimensions = d0
    outputs = 1
    depth = 2
    type = 'level'
    rule = 'gauss-hermite'
    grid.makeGlobalGrid(dimensions, outputs, depth, type, rule)
    
    # Sample Points
    zzSG = grid.getPoints()
    zSG = np.sqrt(2)*zzSG                # transform to N(0,1)
    
    # Quadrature Weights
    wSG = grid.getQuadratureWeights()
    wSG /= np.sqrt(np.pi)**d0     # normalize weights
    
    # Number of grid points
    n0 = grid.getNumPoints()
    
    #
    # Sample low dimensional input parameter
    # 
    q0 = sample_q0(V,lmd,d0,zSG.T)
    J0 = sample_qoi(q0, dQ1)
    
    #
    # Sample conditional expectation
    # 
    
    # Pick a single coarse sample to check
    i0 = np.random.randint(0,high=n0)
    
    # Sample fine, conditional on coarse
    n_samples = 1
    z1 = np.random.randn(d-d0,n_samples)
    q = sample_q_given_q0(q0[:,i0], V, lmd, d0, z1)
    
    # Perturbation
    log_qref = np.log(q0[:,i0])
    dlog_q = np.log(q.ravel()) - log_qref
    dlog_qfn = Nodal(data=dlog_q, basis=phi)
    
    # Perturbed q
    n_eps = 12  # Number of refinements
    epsilons = [10**(-l) for l in range(n_eps)]
    log_qper = np.empty((nx,n_eps)) 
    for i in range(n_eps):
        log_qper[:,i] = log_qref + epsilons[i]*dlog_q
    
    """
    plt.plot(x, log_qref, label='ref')
    for i in range(n_eps):
        plt.plot(x, log_qper[:,i],label='%d'%(i))
    """
    
    assert np.allclose(log_qper[:,0], np.log(q.ravel()))
      
    plt.legend()    
    plt.show()
    
    # Define finite element function
    exp_qref = Nodal(data=q0[:,i0], basis=phi)
    exp_qper = Nodal(data=np.exp(log_qper), basis=phi)
    
    #
    # PDEs 
    # 
    
    # 1. State Equation
    state_eqn = [Form(exp_qref, test=phi_x, trial=phi_x), Form(1, test=phi)]
    state_dbc = {'left':0, 'right':1}
    
    # 2. Perturbed Equation
    perturbed_eqn = [Form(exp_qper, test=phi_x, trial=phi_x), Form(1, test=phi)]
    perturbed_dbc = {'left':0, 'right':1}
    
    # 3. Adjoint Equation
    adjoint_eqn = [Form(exp_qref, test=psi_x, trial=psi_x), Form(0, test=psi)]
    adjoint_dbc = {'left':0,'right':-1}
    
    # Combine 
    eqns = [state_eqn, perturbed_eqn, adjoint_eqn]
    bcs  = [state_dbc, perturbed_dbc, adjoint_dbc]
    
    #
    # Assembly
    # 
    assembler = Assembler(eqns, n_gauss=(6,36))
    
    # Boundary conditions
    for i, bc in zip(range(3), bcs):
        for loc, val in bc.items():
            assembler.add_dirichlet(loc, val, i_problem=i)
    
    # Assemble
    assembler.assemble()
    
    #
    # Solve
    # 
    
    # Solve state
    ur = assembler.solve(i_problem=0)
    u_ref = Nodal(data=ur, basis=phi)
    ux_ref = Nodal(data=ur, basis=phi_x)
    
    
    # Solve perturbed state
    u_per = Nodal(basis=phi)
    for i in range(n_eps):
        # FEM solution
        up = assembler.solve(i_problem=1, i_matrix=i)
        u_per.add_samples(up) 
        
        plt.plot(x,up-ur)
    plt.show()    
    ux_per = Nodal(data=u_per.data(), basis=phi_x)
    
    # Solve adjoint equation
    v = assembler.solve(i_problem=2)
    v_adj = Nodal(data=v, basis=psi)
    vx_adj = Nodal(data=v, basis=psi_x)
    
    
    #
    # Sensitivity
    #

    # Sensitivity Equation
    ker_sen = Kernel(f=[exp_qref, dlog_qfn, ux_ref], F=lambda eq, dq, ux: -eq*dq*ux)
    sensitivity_eqn = [Form(exp_qref, test=phi_x, trial=phi_x), 
                       Form(ker_sen, test=phi_x)]
    
    sensitivity_dbc = {'left': 0, 'right': 0}
    assembler = Assembler(sensitivity_eqn, n_gauss=(6,36))
    for loc in sensitivity_dbc:
        assembler.add_dirichlet(loc, sensitivity_dbc[loc])
    assembler.assemble()
    s = assembler.solve() 
    sx = Nodal(data=s, basis=phi_x)
    
    plt.plot(x,s)
    plt.show()

    #
    # Quantities of Interest
    #
    
    # Reference
    k_ref = Kernel(f=[exp_qref, ux_ref], F=lambda eq,ux: eq*ux)
    ref_qoi = [Form(k_ref, dmu='dv', flag='right')]
    
    # Perturbed
    k_per = Kernel(f=[exp_qper, ux_per], F=lambda eq,ux: eq*ux)
    per_qoi = [Form(k_per, dmu='dv', flag='right')]
    
    # Adjoint
    k_adj = Kernel(f=[exp_qref, dlog_qfn, ux_ref, vx_adj], F=lambda eq,dq,ux,vx: -eq*dq*ux*vx)
    adj_qoi = [Form(k_adj)]
    
    # Sensitivity
    k_sens = Kernel(f=[exp_qref,dlog_qfn, ux_ref,sx], F=lambda eq,dq,ux,sx: eq*dq*ux+eq*sx)
    sens_qoi = Form(k_sens, dmu='dv', flag='right')
    
    qois = [ref_qoi, per_qoi, adj_qoi, sens_qoi]
    
    # Assemble 
    assembler = Assembler(qois, mesh=mesh)
    assembler.assemble()
    
    # Evaluate
    J_ref = assembler.get_scalar(0)
    J_per = []
    for i in range(n_eps):
        J_per.append(assembler.get_scalar(1,i))
    
    # Finite difference approximation
    dJ = []
    for eps, J_p in zip(epsilons, J_per):
        dJ.append((J_p-J_ref)/eps)
    
    # Adjoint differential
    dJ_adj = assembler.get_scalar(2)
    
    # Sensitivity differential
    dJ_sen = assembler.get_scalar(3)
    
    print(dJ_adj)
    print(dJ_sen)
    print(dJ)
    """
    
    #for i in range(n0):
        #  
        # Cycle over sparse grid points
        # 
    
    #dq = np.log(exp_q) - np.log(exp_q0[:,[i0]])
    
    # Plot log(q|q0)
    dq_fn = Nodal(data=np.log(exp_q), basis=Basis(dQ1,'u'))
    kwargs = {'color':'k', 'linewidth':0.1}
    plot.line(dq_fn,i_sample=np.arange(n_samples),plot_kwargs=kwargs)
    
    # 
    J, U = sample_qoi(exp_q, dQ1, return_state=True)         
    ufn = Nodal(data=U, basis=phi)
    plot.line(ufn, i_sample=np.arange(n_samples),plot_kwargs=kwargs)
    
    #  
    dJ = sensitivity_sample_qoi(exp_q0[:,[i0]], dQ1)
    JJ = J0[i0] + dJ.T.dot(dq)
    
    #
    #plt.hist(np.abs(J-JJ), density=True)
    
    print(np.corrcoef(J, JJ))
    
    plt.hist(J, bins=100, density=True, alpha=0.5, label='actual')
    plt.hist(JJ, bins=100, density=True, alpha=0.5, label='sensitivity')
    
    plt.show()
    
    # Compute sparse grid mean and variance
    EJ0 = np.sum(wSG*J0)
    VJ0 = np.sum(wSG*(J0**2)) - EJ0**2
    """
    
    
"""
# =============================================================================
# Random field
# =============================================================================

n_samples = 5


#cov = Covariance(dofhandler, name='exponential', parameters={'l':0.1})

 

# Plot low dimensional field
d0 = 10
d  = len(lmd)
Lmd0 = np.diag(np.sqrt(lmd[:d0]))
V0 = V[:,:d0]
Z0  = np.random.randn(d0,n_samples)
log_q0 = V0.dot(Lmd0.dot(Z0))
plt.plot(x,log_q0)
plt.show()

# Plot high dimensional field conditional on low
Dc = np.diag(np.sqrt(lmd[d0:]))
Vc = V[:,d0:]
for n in range(n_samples):
    Zc = np.random.randn(d-d0,100)
    log_qc = Vc.dot(Dc.dot(Zc))
    plt.plot(x,log_q0[:,n],'k',linewidth=1.5)
    plt.plot(x,(log_q0[:,n].T+log_qc.T).T, 'k', linewidth=0.1, alpha=0.5)
plt.show()

# =============================================================================
# Sparse Grid Loop
# =============================================================================
grid = TasmanianSG.TasmanianSparseGrid()
dimensions = d0
outputs = m
depth = 4
type = 'level'
rule = 'gauss-hermite'
grid.makeGlobalGrid(dimensions, outputs, depth, type, rule)

# Get Sample Points
zzSG = grid.getPoints()
zSG = np.sqrt(2)*zzSG                # transform to N(0,1)

wSG = grid.getQuadratureWeights()
n0 = grid.getNumPoints()

# Sample low resolution parameter
log_qSG = V0.dot(Lmd0.dot(zSG.T))
log_q0 = Nodal(data=log_qSG, dofhandler=dofhandler)

# Sample state
qfn = Nodal(dofhandler=dofhandler, data=np.exp(log_qSG))

# =============================================================================
# Compute Sparse Grid Expectation
# ============================================================================= 
print('1. Low dimensional sparse grid')
print('  -Number of Dofs: %d'%(m))
print('  -SG sample size: %d'%(n0))

comment.tic(' a) assembly: ')
phi = Basis(dofhandler, 'u')
phi_x = Basis(dofhandler, 'ux')

problems = [[Form(kernel=qfn, trial=phi_x, test=phi_x), Form(1, test=phi)],
            [Form(1, test=phi, trial=phi)]]

assembler = Assembler(problems, mesh)
assembler.assemble()
comment.toc()

comment.tic(' b) solver: ')
A = assembler.af[0]['bilinear'].get_matrix()
b = assembler.af[0]['linear'].get_matrix()

linsys = LinearSystem(phi)
linsys.add_dirichlet_constraint('left',1)
linsys.add_dirichlet_constraint('right',0)

y_data = np.empty((m,n0))
for n in range(n0):        
    linsys.set_matrix(A[n].copy())
    linsys.set_rhs(b.copy())
    linsys.solve_system()
    y_data[:,[n]] = linsys.get_solution(as_function=False)
comment.toc()

comment.tic(' c) saving SG:')
np.save('y_SG',y_data)
comment.toc()

comment.tic(' d) loading SG:')
y_SG = np.load('y_SG.npy')
comment.toc()

comment.tic(' e) computing SG average:')
c_norm = np.sqrt(np.pi)**d0     # normalization constant
y_ave_SG = np.zeros(m)
for n in range(n0):
    y_ave_SG += wSG[n]*y_SG[:,n]/c_norm
comment.toc()
"""
'''
print('2. Enrich with MC')
n1 = 100
print('  -number of sg samples: %d'%(n0))
print('  -number of mc per sg: %d'%(n1))
print('  -total number of samples: %d'%(n0*n1))


# Plot high dimensional field conditional on low
Dc = np.diag(np.sqrt(lmd[d0:]))
Vc = V[:,d0:]

yc_ave_MC = np.empty((m,n0))
k = 0
comment.comment(' a) iterating over sparse grid points')
for i in range(n0):
    
    comment.tic('  i. sampling mc conditional input')
    Zc = np.random.randn(d-d0,m)
    log_qc = Vc.dot(Dc.dot(Zc))
    qfn = Nodal(dofhandler=dofhandler, data=np.exp(log_qc))
    comment.toc()
    
    comment.tic('  ii. assembling')
    assembler.assemble()
    comment.toc()
    
    comment.tic('  iii. solver')
    # Compute conditional expectation
    yc_data = np.empty((m,n1))
    for j in range(n1):
        linsys.set_matrix(A[j].copy())
        linsys.set_rhs(b.copy())
        linsys.solve_system()
        yc_data[:,[j]] = linsys.get_solution(as_function=False)
    comment.toc()
    
    """
    if i==5:
        plt.plot(x,yc_data,'k', linewidth=0.1, alpha=0.5)
        plt.title('Solution conditional on q0')
        plt.show()
    """
    # Compute conditional average using Monte Carlo
    yc_ave_MC[:,i] = 1/n1*np.sum(yc_data,axis=1)
np.save('yc_ave_MC',yc_ave_MC)
'''

"""
y_ave_MC = np.load('yc_ave_MC.npy')

y_ave_HYB =  np.zeros(m)
for n in range(n0):
    y_ave_HYB += wSG[n]*y_ave_MC[:,n]/c_norm
plt.plot(x,y_ave_SG, 'k', label='coarse')
plt.plot(x,y_ave_HYB, 'k--',label='hybrid')
plt.legend()
plt.show()
"""
"""
  
# =============================================================================
# Compute Reduced Order Model
# ============================================================================= 
M = assembler.af[1]['bilinear'].get_matrix()
y_train = y_data[:,i_train]
y_test = y_data[:,i_test]
U,S,Vt = la.svd(y_train)

x = dofhandler.get_dof_vertices()

m = 8
d = 7

Um = U[:,:m]
plt.plot(x,Um,'k')

# Test functions
i_left = dofhandler.get_region_dofs(entity_flag='left', entity_type='vertex')
B = Um[i_left,:].T

plt.plot(np.tile(x[i_left],B.shape),B,'r.')
plt.show()

Q,R = la.qr(B, mode='full')
psi = Um.dot(Q[:,1:])
plt.plot(x,psi)
plt.show()


rom_tol = 1e-10
rom_error = 1-np.cumsum(S)/np.sum(S)
n_rom = np.sum(rom_error>=rom_tol)
print(n_rom)
Ur = U[:,:n_rom]

Am = np.empty((m,m))
Am[:d,:] = Q[:,1:].T.dot(Um.T.dot(A[0].dot(Um)))
Am[-1,:] = B.ravel()

bm = np.zeros((m,1))
bm[:d,:] = Q[:,1:].T.dot(Um.T.dot(b.toarray()))
bm[-1,:] = 1


c = la.solve(Am,bm)
plt.plot(x,y_data[:,[0]],'k',x,Um.dot(c),'r') 
plt.show()

print(Am.shape)
#plt.plot(x,Ur)
#plt.show()

# =============================================================================
# Predict output using ROM
# ============================================================================= 
u_rom = np.empty((n,n_train))
br = b.T.dot(Ur).T 
for i in np.arange(n_train):
    Ar = Ur.T.dot(A[i_train[i]].dot(Ur)) 
    cr = la.solve(Ar, br)
    u_rom[:,[i]] = Ur.dot(cr)


# =============================================================================
# Compare ROM output with direct numerical simulation
# ============================================================================= 

#plt.plot(x,u_rom,'k',x,y_data[:,i_train])
#plt.show()

du = np.empty((n,n_train))
for i in range(n_train):
    du[:,i] = u_rom[:,i]-y_train[:,i]
    #du[:,i] = Ur.dot(Ur.T.dot(u_test[:,i])) - u_test[:,i]


u_error = Nodal(dofhandler=dofhandler, data=du)
#u_error = np.dot(du.T, M.dot(du))
#plot.line(u_error, i_sample=np.arange(0,n_train))
"""

if __name__ == '__main__':
    #test00_finite_elements()
    #test00_sensitivity()
    #test01_problem()
    #test02_reference()
    #test03_truncation()
    #test04_sparse_grid()
    #test05_conditioning()
    test06_linearization()