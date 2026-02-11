"""
This is how to sample from a conditional Gaussian field using projection
or pointwise conditioning on hierarchical meshes.
"""

from assembler import Assembler, Form, Form
from fem import Basis, DofHandler, QuadFE
from function import Nodal
from gmrf import Covariance, GaussianField
from mesh import Mesh1D, QuadMesh
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
import matplotlib.pyplot as plt

from plot import Plot

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

def test01_1D_condition_on_projection():
    """
    Test sampling from a conditional Gaussian field using projection between
    coarse and fine 1D hierarchical meshes.
    """
    # Create hierarchical mesh
    mesh = Mesh1D(box=[0,1], resolution=(4,))  # initial mesh
    mesh.record(0)  # coarse-level flag
    for l in range(2): mesh.cells.refine()   # refine twice
    mesh.record(1) # fine-level flag

    # Define a finite element space on the mesh
    Q1 = QuadFE(mesh.dim(), 'Q1')
    dh = DofHandler(mesh,Q1)
    dh.distribute_dofs()

    # Define coarse and fine basis functions 
    vc = Basis(dh, 'v', subforest_flag=0)  # Coarse basis
    vf = Basis(dh, 'v', subforest_flag=1)  # Fine basis

    # Assemble the projection matrix from fine to coarse basis
    P = assemble_projection_matrix(vf, vc)

    x_crs = dh.get_dof_vertices(subforest_flag=0)
    q_crs = np.sin(2*np.pi*x_crs)  # Coarse observations

    n_fne = vf.n_dofs()
    cov_fne = Covariance(dh, name='gaussian', 
                         parameters = {'l':0.2, 'sgm':1.0})
    q_fne = GaussianField(n_fne, covariance=cov_fne)
    q_fne_smpl = q_fne.condition(P, q_crs,n_samples=10)
    q_fne_nodal = Nodal(data=q_fne_smpl, basis=vf)
    x_fne = dh.get_dof_vertices(subforest_flag=1)[:,0]

    # Plot the results
    fig, ax = plt.subplots()
    plot = Plot(quickview=False)
    for i in range(10):
        ax = plot.line(q_fne_nodal, axis=ax, i_sample=i, 
                       plot_kwargs = {'color':'gray', 'alpha':0.5})
    plt.plot(x_crs, q_crs, 'ro', label='Coarse Observations')
    plt.legend()
    plt.show()


def test02_1D_condition_on_pointwise():
    """
    Test sampling from a conditional Gaussian field using pointwise
    conditioning on a fine 1D hierarchical mesh.

    TODO: Unfinished!
    """
    # Create a Mesh
    mesh = Mesh1D(box=[0,1], resolution=(5,))  # initial mesh
    mesh.record(0)  # coarse-level flag
    mesh.cells.refine()   # refine once
    mesh.record(1) # fine-level flag
    
    # Define a finite element space on the mesh
    Q1 = QuadFE(mesh.dim(), 'Q1')
    dh = DofHandler(mesh,Q1)
    dh.distribute_dofs()

    # Define fine basis functions 
    vf = Basis(dh, 'v', subforest_flag=1)  # Fine basis
    vc = Basis(dh, 'v', subforest_flag=0)  # Coarse basis

    n_fne = vf.n_dofs()
    cov = Covariance(dh, name='gaussian', 
                     parameters = {'l':0.2, 'sgm':1.0})
    q = GaussianField(n_fne, covariance=cov)

    # Define the restriction mapping
    I_f2c = restrict_to_coarse_mesh(vf, vc)
   
    # Define pointwise observations on coarse mesh
    x_dofs = dh.get_dof_vertices(subforest_flag=0)
    q_obs = np.sin(2*np.pi*x_dofs)  # observations at coarse
    print(q_obs.shape)

    q_smpl = q.condition(I_f2c, q_obs, n_samples=10)
    q_nodal = Nodal(data=q_smpl, basis=vf)

    # Plot the results
    fig, ax = plt.subplots()
    plot = Plot(quickview=False)
    for i in range(10):
        ax = plot.line(q_nodal, axis=ax, i_sample=i, 
                       plot_kwargs = {'color':'gray', 'alpha':0.5})
    plt.plot(x_dofs, q_obs, 'ro', label='Point Observations')
    plt.legend()
    plt.show()  
    

if __name__ == "__main__":
    test01_1D_condition_on_projection()
    test02_1D_condition_on_pointwise()