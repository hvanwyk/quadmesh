"""
Generate Matern random fields using the PDE filter 

    (k^2(x) - nabla * (H(x) nabla) u(x) = W(x)
    
This series of numerical experiments should test the 
implementation and effect of  

1. the scaling parameter k 
2. the diffusion tensor
3. boundary conditions 

"""

# Import 
from mesh import Mesh
from finite_element import QuadFE, System, DofHandler, Function, GaussRule
from plot import Plot
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import linalg as spla

 
def test01():
    """
    """
    pass


def test02():
    """
    Spatially varying variance
    """
    pass


def test03():
    """
    Constant Anisotropy
    """
    # Mesh
    mesh = Mesh.newmesh([0,20,0,20], grid_size=(200,200))
    mesh.refine()
    element = QuadFE(2, 'Q2')
    system = System(mesh, element)
    
    gma = 1 
    bta = 8
    tht = np.pi/4 
    v = np.array([np.cos(tht), np.sin(tht)])
    H = gma*np.eye(2,2) + bta*np.outer(v,v)
    Z = np.random.normal(size=system.n_dofs())
    dW = Function(Z,mesh,element)
    
    # Bilinear forms
    bf = [(1,'u','v'), \
          (H[0,0],'ux','vx'), (H[0,1],'uy','vx'),\
          (H[1,0],'ux','vy'), (H[1,1],'uy','vy')]
                                
    A = system.assemble(bilinear_forms=bf, linear_forms=None)
    M = system.assemble(bilinear_forms=[(1,'u','v')])
    m_lumped = np.array(M.sum(axis=1)).squeeze()
    X = spla.spsolve(A.tocsc(), np.sqrt(m_lumped)*Z)
    
    fig, ax = plt.subplots()
    plot = Plot()
    ax = plot.contour(ax, fig, X, mesh, element)
    plt.show()

def test04():
    """
    Boundary Conditions
    """
    pass


if __name__ == '__main__':
    test03()
    