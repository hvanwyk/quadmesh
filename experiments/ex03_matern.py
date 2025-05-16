"""
Generate Matern random fields using the PDE filter 

    (k^2(x) - nabla * (H(x) nabla) u(x) = W(x)
    
This series of numerical experiments should test the 
implementation and effect of  

1. the scaling parameter k 
2. the diffusion tensor
3. boundary conditions 

TODO: Update to latest source code
TODO: Add boundary conditions
"""
import sys
if '/home/hans-werner/git/quadmesh/src' not in sys.path:
    sys.path.append('/home/hans-werner/git/quadmesh/src')


# Import 
from mesh import Mesh
from function import Function
from fem import QuadFE, DofHandler, Function, GaussRule
from gmrf import Gmrf
from plot import Plot
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import linalg as spla


 
def test01():
    """
    Condition on coarse realization
    """
    print('Test 1:')
    grid = Grid(box=[0,20,0,20], resolution=(10,10))
    mesh = Mesh.newmesh(grid=grid)
    mesh.record(flag=0)
    for _ in range(3):
        mesh.refine()
    
    mesh.record(flag=1)
    
    element = QuadFE(2,'Q1')
    system = System(mesh, element, nested=True)
    #system = System(mesh, element)
    
    gma = 1 
    bta = 8
    tht = np.pi/4 
    v = np.array([np.cos(tht), np.sin(tht)])
    H = gma*np.eye(2,2) + bta*np.outer(v,v)
    Z = 10*np.random.normal(size=system.n_dofs())
    
    # Bilinear forms
    bf = [(1,'u','v'), \
          (H[0,0],'ux','vx'), (H[0,1],'uy','vx'),\
          (H[1,0],'ux','vy'), (H[1,1],'uy','vy')]
    
    print('assembly')
    A = system.assemble(bilinear_forms=bf, linear_forms=None)
    M = system.assemble(bilinear_forms=[(1,'u','v')])
    m_lumped = np.array(M.sum(axis=1)).squeeze()
    
    print('generating realizations')
    X = spla.spsolve(A.tocsc(), np.sqrt(m_lumped)*Z)
    fX = Function(X,'nodal', mesh, element, flag=1)
    R01 = system.restrict(0,1)
    Xr = Function(R01.dot(X), 'nodal', mesh, element, flag=0)
    
    print('plotting')
    plot = Plot()
    fig, ax = plt.subplots(2,2)
    ax[0][0] = plot.mesh(ax[0][0], mesh, element,node_flag=0)
    ax[0][1] = plot.mesh(ax[0][1], mesh, element,node_flag=1)
    ax[1][0] = plot.contour(ax[1][0], fig, Xr, mesh, element, flag=0)
    ax[1][1] = plot.contour(ax[1][1], fig, fX, mesh, element, flag=1)
    plt.show()
    
    
    


def test02():
    """
    Spatially varying anisotropy
    """
    print('Test 2:')
    grid = Grid(box = [0,20,0,20], resolution=(100,100))
    mesh = Mesh(grid=grid)
    element = QuadFE(2,'Q1')
    system = System(mesh, element)
    
    
    alph = 2
    kppa = 1
    
    # Symmetric tensor gma T + bta* vv^T
    gma = 0.1
    bta = 25
    v2 = lambda x,y: -0.75*np.cos(np.pi*x/10)
    v1 = lambda x,y: 0.25*np.sin(np.pi*y/10)
    
    h11 = lambda x,y: gma + v1(x,y)*v1(x,y)
    h12 = lambda x,y: v1(x,y)*v2(x,y)
    h22 = lambda x,y: v2(x,y)*v2(x,y)
    
    X = Gmrf.from_matern_pde(alph, kppa, mesh, element, tau=(h11,h12,h22))
    x = X.sample(1).ravel()
    
    fig, ax = plt.subplots()
    plot = Plot()
    ax = plot.contour(ax, fig, x, mesh , element, resolution=(200,200))
    plt.show()

  
def test03():
    """
    Constant Anisotropy
    """
    print('Test 3:')
    # Mesh
    mesh = Mesh.newmesh([0,20,0,20], grid_size=(40,40))
    mesh.refine()
    element = QuadFE(2, 'Q1')
    system = System(mesh, element)
    
    gma = 1 
    bta = 8
    tht = np.pi/4 
    v = np.array([np.cos(tht), np.sin(tht)])
    H = gma*np.eye(2,2) + bta*np.outer(v,v)
    Z = 10*np.random.normal(size=system.n_dofs())
    
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
    ax = plot.contour(ax, fig, X, mesh, element, resolution=(200,200))
    plt.show()
    
    

def test04():
    """
    Boundary Conditions
    """
    pass


if __name__ == '__main__':
    test01()
    test02()
    test03()