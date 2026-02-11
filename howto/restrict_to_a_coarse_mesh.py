"""
This is how evaluate a fine mesh function on a coarse submesh 
"""

import numpy as np
import matplotlib.pyplot as plt
from mesh import Mesh1D, QuadMesh
from fem import Basis, DofHandler, QuadFE
from function import Nodal
from assembler import Form, Assembler, Kernel
from plot import Plot
import scipy.sparse as sp

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

def test01_1D_hrefinement():
    """
    Test the restriction to a coarse mesh on a two-level hierarchical mesh.
    """
    # Create hierarchical mesh
    nl = 2
    mesh = Mesh1D(resolution=(2,), box = [0,1])
    mesh.record(0)
    for l in range(nl):
        mesh.cells.refine()
        mesh.record(l+1)

    sf_crs = 1  # coarse subforest flag

    # Define finite element space
    Q = QuadFE(mesh.dim(), 'Q1')
    dh = DofHandler(mesh, Q)
    dh.distribute_dofs()
    v_fne = Basis(dh, 'v')
    v_crs = Basis(dh, 'v', subforest_flag=sf_crs)

    # Define a fine mesh function
    f_fne = Nodal(basis=v_fne)
    f_fne.set_data( f= lambda x: np.sin(2 * np.pi * x[:,0]) )

    # Restrict to coarse mesh (level sf_crs)
    I_f2c = restrict_to_coarse_mesh(v_fne, v_crs)

    f_crs = Nodal(basis=v_crs, data=I_f2c @ f_fne.data())

    # Plotting
    plot = Plot(quickview=False)
    fig, ax = plt.subplots(2,1, figsize=(6, 4))

    ax[0] = plot.line(f_fne, axis=ax[0])
    ax[0].set_title("Fine Mesh Function")
    ax[1] = plot.line(f_crs, axis=ax[1])
    ax[1].set_title("Restricted Coarse Mesh Function")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test01_1D_hrefinement()