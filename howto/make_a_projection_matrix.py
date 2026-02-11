"""
This is how you construct a projection matrix from a larger subspace onto a
smaller subspace contained in it. The subspaces are defined through their 
basis functions.

"""

import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from mesh import Mesh1D, QuadMesh
from fem import Basis, DofHandler, QuadFE
from function import Nodal
from assembler import Form, Assembler, Kernel
from plot import Plot   

def assemble_ortho_projection(v_fne, v_crs):
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


def test01_1D_hrefinement():
    """
    Test the projection matrix assembly on a two-level hierarchical mesh.
    """
    # Create hierarchical mesh
    mesh = Mesh1D(resolution=(5,), box=[0, 1])
    mesh.record(0)
    mesh.cells.refine()
    mesh.record(1)

    # Define finite element spaces
    Q = QuadFE(1, 'Q1')
    dh = DofHandler(mesh, Q)
    dh.distribute_dofs()

    v_fne = Basis(dh, 'v', subforest_flag=1)  # Fine basis
    v_crs = Basis(dh, 'v', subforest_flag=0)  # Coarse basis

    # Assemble projection matrix
    P = assemble_ortho_projection(v_fne, v_crs)

    f_fne = Nodal(f=lambda x: np.sin(2 * np.pi * x), basis=v_fne)
    data = f_fne.data()
    print("Fine basis function data:", data)

    plot = Plot(quickview=False)
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0] = plot.line(f_fne, axis = ax[0])
    ax[0].set_title("Function in Fine Basis")   
    
    f_prj = Nodal(data=P @ f_fne.data(), basis=v_crs)
    ax[1] = plot.line(f_prj, axis = ax[1])
    ax[1].set_title("Projected Function in Coarse Basis")
    plt.tight_layout()
    plt.show()


def test02_2D_hrefinement():
    """
    Test the projection matrix assembly on a two-level hierarchical mesh in 2D.
    """
    # Create hierarchical mesh
    mesh = QuadMesh(resolution=(4,4), box=[0, 1, 0, 1])
    mesh.record(0)
    for i in range(2):
        mesh.cells.refine()
    mesh.record(1)

    # Define finite element spaces
    Q = QuadFE(2, 'Q1')
    dh = DofHandler(mesh, Q)
    dh.distribute_dofs()

    v_fne = Basis(dh, 'v', subforest_flag=1)  # Fine basis
    v_crs = Basis(dh, 'v', subforest_flag=0)  # Coarse basis

    # Assemble projection matrix
    P = assemble_ortho_projection(v_fne, v_crs)

    f_fne = Nodal(f=lambda x: np.sin(2 * np.pi * x[:,0]) * np.sin(2 * np.pi * x[:,1]), 
                  basis=v_fne)


    plot = Plot(quickview=False)
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0] = plot.contour(f_fne, axis = ax[0], mesh=mesh)
    ax[0].set_title("Function in Fine Basis")   
    
    f_prj = Nodal(data=P @ f_fne.data(), basis=v_crs)
    ax[1] = plot.contour(f=f_prj, axis = ax[1], mesh=mesh)
    ax[1].set_title("Projected Function in Coarse Basis")
    plt.tight_layout()
    plt.show()

def test03_1D_p_refinement():
    """
    Test the projection matrix assembly on a two-level p-refined mesh.
    """
    # Create mesh
    mesh = Mesh1D(resolution=(5,), box=[0, 1])

    # Define finite element spaces
    Q_fne = QuadFE(mesh.dim(), 'Q2')  # Fine basis (p=2)
    Q_crs = QuadFE(mesh.dim(), 'Q1')  # Coarse basis (p=1)
    dh_fne = DofHandler(mesh, Q_fne)
    dh_fne.distribute_dofs()
    dh_crs = DofHandler(mesh, Q_crs)
    dh_crs.distribute_dofs()

    v_fne = Basis(dh_fne, 'v')  # Fine basis
    v_crs = Basis(dh_crs, 'v')  # Coarse basis

    # Assemble projection matrix
    P = assemble_ortho_projection(v_fne, v_crs)

    f_fne = Nodal(f=lambda x: np.sin(2 * np.pi * x), basis=v_fne)

    plot = Plot(quickview=False)
    fig, ax = plt.subplots(2, 1, figsize=(6, 4))
    ax[0] = plot.line(f_fne, axis = ax[0])
    ax[0].set_title("Function in Fine Basis")   
    
    f_prj = Nodal(data=P @ f_fne.data(), basis=v_crs)
    ax[1] = plot.line(f_prj, axis = ax[1])
    ax[1].set_title("Projected Function in Coarse Basis")
    plt.tight_layout()
    plt.show()


def test04_2D_p_refinement():
    """
    Test the projection matrix assembly on a two-level p-refined mesh in 2D.
    """
    # Create mesh
    mesh = QuadMesh(resolution=(4,4), box=[0, 1, 0, 1])

    # Define finite element spaces
    Q_fne = QuadFE(mesh.dim(), 'Q2')  # Fine basis (p=2)
    Q_crs = QuadFE(mesh.dim(), 'Q1')  # Coarse basis (p=1)
    dh_fne = DofHandler(mesh, Q_fne)
    dh_fne.distribute_dofs()
    dh_crs = DofHandler(mesh, Q_crs)
    dh_crs.distribute_dofs()

    v_fne = Basis(dh_fne, 'v')  # Fine basis
    v_crs = Basis(dh_crs, 'v')  # Coarse basis

    # Assemble projection matrix
    P = assemble_ortho_projection(v_fne, v_crs)

    f_fne = Nodal(f=lambda x: np.sin(2 * np.pi * x[:,0]) * np.sin(2 * np.pi * x[:,1]), 
                  basis=v_fne)


    plot = Plot(quickview=False)
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0] = plot.contour(f_fne, axis = ax[0], mesh=mesh)
    ax[0].set_title("Function in Fine Basis")   
    
    f_prj = Nodal(data=P @ f_fne.data(), basis=v_crs)
    ax[1] = plot.contour(f=f_prj, axis = ax[1], mesh=mesh)
    ax[1].set_title("Projected Function in Coarse Basis")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #test01_1D_hrefinement()
    #test02_2D_hrefinement()
    test03_1D_p_refinement()
    test04_2D_p_refinement()