"""
Random Iterative Solver:

1. Generate a random parameter field on a coarse mesh.
    1.1 Assemble the system matrix
    1.2 Solve the coarse linear system -> uhat 
2. Generate samples from a conditional random parameter field on a fine mesh, 
   conditioned on the coarse parameter qhat.
3. Try to solve the fine linear system iteratively, using the coarsely
   assembled system matrix as a preconditioner.
   
"""

from assembler import Assembler, Form, Kernel
from diagnostics import Verbose
from fem import Basis, DofHandler, QuadFE
from function import Constant, Nodal
from gmrf import Covariance, GaussianField
from mesh import Mesh1D
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import matplotlib.pyplot as plt                                                                                                                                                                                                                                                                                                                                                                                     

from plot import Plot
from solver import LinearSystem


def assemble_projection_matrix(v_fne, v_crs):
    """
    Description: 

        Assemble the projection matrix from a fine basis to a coarse basis.
    
    Inputs:

        v_fne: Basis, Fine basis functions.

        v_crs: Basis, Coarse basis functions.

    Outputs:

        P: np.ndarray, Projection matrix from fine to coarse basis.

            (v_crs, v_crs)P = (v_crs, v_fne)
    """
    mesh = v_fne.mesh()
    subff = v_fne.subforest_flag()

    # Define the forms
    problem_cc = [Form(trial=v_crs, test=v_crs)]
    problem_cf = [Form(trial=v_fne, test=v_crs)]

    # Define the assembler
    assembler = Assembler([problem_cc, problem_cf],
                          mesh=mesh, subforest_flag=subff)
    
    # Assemble the matrices
    assembler.assemble()

    # Extract the mass matrices
    M_cc = assembler.get_matrix(0).tocsc()
    M_cf = assembler.get_matrix(1).tocsc()

    # Solve for projection matrix
    P = spla.spsolve(M_cc, M_cf)

    return P

def restrict_to_coarse_mesh(v_fne, v_crs):
    """
    Description: 

        Restrict a fine mesh basis function to a coarse submesh.

    Inputs:

        v_fne: Basis, Fine mesh basis function.

        v_crs: Basis, Coarse mesh basis function.

    
    Outputs:

        I_f2c : ndarray, Restriction matrix from fine to coarse mesh.
    """
    # Get fine dofs 
    #d_fne = v_fne.dofs()  # this is the number of columns
    d_crs = v_crs.dofs()  # this is the number of rows


    cols = v_fne.d2i(d_crs)  # map coarse dofs to fine indices
    rows = v_crs.d2i(d_crs)  # map coarse dofs to coarse indices
    vals = np.ones_like(rows, dtype=float)

    n_crs = v_crs.n_dofs()
    n_fne = v_fne.n_dofs()

    I_f2c = sp.coo_matrix( (vals, (rows, cols)), \
                          shape=(n_crs, n_fne) )   

    return I_f2c

def assemble_system_matrix(q, mesh, vx, subforest_flag):
    """
    Assemble the system matrix for the given parameters and mesh.

    Inputs:

        q: Nodal, Diffusion coefficient.

        mesh: Mesh, Finite element mesh.

        vx: Basis, Basis functions for the gradient term.

        subforest_flag: int, Subforest flag for the basis functions.
    """
    # Define the form
    F_stiff = Form(Kernel(q), test=vx, trial=vx)

    # Define the assembler
    assembler = Assembler([F_stiff], mesh=mesh, subforest_flag=subforest_flag)

    # Assemble the matrix
    assembler.assemble()

    # Determine the number of systems assembled
    n_samples = assembler.n_samples(i_problem=0, form_type='bilinear')

    print(f"Assembled {n_samples} system matrices.")

    # Collect the system matrices
    A = [assembler.get_matrix(i_sample=i).tocsc() for i in range(n_samples)]

    return A

def assemble_rhs(f, mesh, v, subforest_flag):
    """
    Assemble the right-hand side vector for the given parameters and mesh.

    Inputs:

        f: Constant, Right-hand side function.

        mesh: Mesh, Finite element mesh.

        v: Basis, Basis functions for the source term.

        subforest_flag: int, Subforest flag for the basis functions.
    """
    # Define the form
    F_source = Form(Kernel(f), test=v)

    # Define the assembler
    assembler = Assembler([F_source], mesh=mesh, subforest_flag=subforest_flag)

    # Assemble the vector
    assembler.assemble()

    # Collect the right-hand side vectors
    b = assembler.get_vector()

    return b

def iterative_solver(Ac, Af, b, uc, vf, maxiter=10):
    """
    Solve the linear system Af x = b using an iterative solver with Ac as a preconditioner.

    Inputs:

        Ac: sp.csc_matrix, Coarse system matrix (preconditioner).

        Af: sp.csc_matrix, Fine system matrix.

        b: np.ndarray, Right-hand side vector.

        vf: Basis, Finite element basis for the solution.

        tol: float, Tolerance for convergence.

        maxiter: int, Maximum number of iterations.
    """
    # Define the linear system solver
    solver = LinearSystem(vf, Ac, b)

    # Define the preconditioner
    M = spla.LinearOperator(shape=Ac.shape, matvec=lambda x: linear_solve(solver, Ac, x),dtype=np.float64)
    
    comment = Verbose()
    # Solve the system using GMRES with the preconditioner
    comment.tic('GMRES with preconditioner')
    x, info = spla.gmres(Af, b, x0=uc, M=M, maxiter=maxiter)
    comment.toc()

    if info == 0:
        print("Convergence achieved.")
    elif info > 0:
        print(f"Maximum iterations reached without convergence. Iterations: {info}")
    else:
        print("Illegal input or breakdown.")

    return x

def linear_solve(solver, A, b):
    """
    Solve the linear system A x = b using the provided solver.

    Inputs:

        solver: LinearSystem, Linear solver object.

        A: sp.csc_matrix, System matrix.

        b: np.ndarray, Right-hand side vector.
    """
    # Determine matrix and rhs
    solver.set_matrix(A)
    solver.set_rhs(b)

    # Apply Dirichlet boundary conditions
    solver.add_dirichlet_constraint('left', Constant(0.0))
    solver.add_dirichlet_constraint('right', Constant(0.0))

    # Solve
    solver.solve_system()
    x = solver.get_solution(as_function=False)

    return x
#
# Mesh 
# 

# Create hierarchical mesh
mesh = Mesh1D(box=[0,1], resolution=(16,))  # initial mesh
mesh.record(0)  # coarse-level flag
for l in range(2): mesh.cells.refine()   # refine once 
mesh.record(1) # fine-level flag

# Mark boundaries
left_bnd =  lambda x: abs(x) < 1e-6
right_bnd = lambda x: abs(x-1) < 1e-6
mesh.mark_region('left', left_bnd, entity_type='vertex', on_boundary=True)
mesh.mark_region('right', right_bnd, entity_type='vertex', on_boundary=True)

#
# Finite Element Space
#  

# Define a finite element space on the mesh
Q1 = QuadFE(mesh.dim(), 'Q1')
dh = DofHandler(mesh,Q1)
dh.distribute_dofs()

# Define coarse and fine basis functions 
vc = Basis(dh, 'v', subforest_flag=0)  # Coarse basis
vf = Basis(dh, 'v', subforest_flag=1)  # Fine basis
vf_x = Basis(dh, 'vx', subforest_flag=1)

# 
# Projection and Interpolation Matrices
#  

# Assemble the projection matrix from fine to coarse basis
P = assemble_projection_matrix(vf, vc)
I_f2c = restrict_to_coarse_mesh(vf, vc)

#
# Random Parameter Field
# 

# -----------------------------------------------------------------------------
# Coarse Gaussian random field
# -----------------------------------------------------------------------------
# Covariance
cov_crs = Covariance(dh, name='gaussian', 
                        parameters = {'l':0.05, 'sgm':1.0}, subforest_flag=0)

# Random field
eta_crs = GaussianField(vc.n_dofs(), covariance=cov_crs)

# Sample coarse random field
eta_crs_smpl = eta_crs.sample(n_samples=1)

# -----------------------------------------------------------------------------
# Conditional Gaussian random field on fine mesh
# -----------------------------------------------------------------------------
n_samples = 100  # sample size

# Covariance with same parameters as coarse field
cov_fne = Covariance(dh, name='gaussian', 
                        parameters = {'l':0.05, 'sgm':1.0})

# Gaussian field on fine mesh
eta_fne = GaussianField(vf.n_dofs(), covariance=cov_fne)

# Conditional sample on fine mesh, conditioned on coarse sample
eta_fne_smpl = eta_fne.condition(P, eta_crs_smpl,n_samples=n_samples) 


#
# Define Problem Parameters
# 
# Single sample of coarse-scale parameter
qc = Nodal(data=np.exp(eta_crs_smpl), basis=vc) 

# n_sample samples of fine-scale parameter
qf = Nodal(data=np.exp(eta_fne_smpl), basis=vf)

# Right-hand side function
f = Constant(1.0)

#
# Assemble system matrices for coarse and fine parameters on the fine mesh
# 
Ac = assemble_system_matrix(qc, mesh, vf, subforest_flag=1)
Af = assemble_system_matrix(qf, mesh, vf, subforest_flag=1)
b  = assemble_rhs(f, mesh, vf, subforest_flag=1)


#
# Solves 
#
solver = LinearSystem(vf, Ac[0], b)
solver.add_dirichlet_constraint('left', Constant(0.0))
solver.add_dirichlet_constraint('right', Constant(0.0))
solver.solve_system(b)

uc_vec = solver.get_solution(as_function=False)
uc_fn = solver.get_solution(as_function=True)

uf_fns = []
for i in range(n_samples):
    solver.set_matrix(Af[i])
    solver.set_rhs(b)
    solver.add_dirichlet_constraint('left', Constant(0.0))
    solver.add_dirichlet_constraint('right', Constant(0.0))
    solver.solve_system(b)
    uf_fns.append(solver.get_solution(as_function=True))


#
# Test iterative solver with coarse matrix as preconditioner
# 
uf_vec = iterative_solver(Ac[0], Af[0], b, uc_vec, vf, maxiter=10)
uf_apx = Nodal(data=uf_vec, basis=vf)
#
# Plots 
#

# Plot the coarse and fine parameter samples
eta_crs_fn = Nodal(data=eta_crs_smpl, basis=vc) 
eta_fne_fn = Nodal(data=eta_fne_smpl, basis=vf)
plot = Plot(quickview=False)
fig, ax = plt.subplots()
plot.line(eta_crs_fn, axis=ax, i_sample=0)
for i in range(n_samples):
    ax = plot.line(eta_fne_fn, axis=ax, i_sample=i, 
                   plot_kwargs={'color':'gray', 'alpha':0.1} )
plt.show()


# Plot the solution for the coarse parameter sample
fig, ax = plt.subplots()
ax = plot.line(uc_fn, axis=ax)
for i in range(n_samples):
    ax = plot.line(uf_fns[i], axis=ax, 
                   plot_kwargs={'color':'gray', 'alpha':0.1} )
plt.show()

uf_ref = uf_fns[0] 
fig, ax = plt.subplots()
ax = plot.line(uf_ref, axis=ax, plot_kwargs={'color':'blue', 'label':'Reference Solution'})
ax = plot.line(uf_apx, axis=ax, plot_kwargs={'color':'red', 'linestyle':'dashed', 'label':'Iterative Solution'})
plt.legend() 
plt.show()