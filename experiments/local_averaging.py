'''
Created on Feb 2, 2017

Multiscale Gaussian random field with standard anisotropic, homogeneous 
exponential covariance matrix. Coarse scale field obtained from local averaging.

@author: hans-werner

Strategy for obtaining covariance matrix: 

    On finest mesh level: Compute the covariance for all relative distances
    Loop over the cells: 
'''
from mesh import Mesh
import matplotlib.pyplot as plt
from numpy.random import rand, randint
import numpy as np
from itertools import count
from scipy.linalg import norm, svd
import numpy.random as random
from plot import Plot

def covh(mesh,a,b):
    """
    Compute the covariance of the lower left point with every other point 
    on the finest mesh.
    """
    nx_0,ny_0 = mesh.grid_size()
    max_depth = mesh.root_node().max_depth()
    nx = nx_0*2**(max_depth-1)
    ny = ny_0*2**(max_depth-1)
    x0,x1,y0,y1 = mesh.box()
    dx = (x1-x0)/nx
    dy = (y1-y0)/ny
    ii,jj = np.meshgrid(range(nx),range(ny))
    h = np.sqrt((ii.flatten()*dx)**2/a + (jj.flatten()*dy)**2/b)
    return np.reshape(np.exp(-h),(ny,nx)) 


def cov_fn(x,y,A=None):
    """
    Return the covariance Cov(xi,yi) for xi in x, yi in y.
    
    Inputs: 
    
        x,y: double, two (n,2) arrays of points in a 2D grid.
        
        A: double, optional (2,2) tensor array 
        
    Outputs:
    
        cov: (n,) array of covariances 
    """  
    h = y - x
    if A is not None:
        assert A.shape == (2,2), 'A must be a 2x2 matrix.'
        d = norm(A.dot(h.transpose()), ord=2, axis=0)
    else:
        d = norm(h.transpose(), ord=2, axis=0)
    return np.exp(-d)
    
    
def get_fine_cell_range(node_address, max_level):
    """
    Determine the indices of the fine scale cells that cover a node with a
    given address
    """
    #
    # Determine indices of cell on own level
    # 
    i = 0
    j = 0
    pos_to_idx = {0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)}
    for pos in node_address:
        i *= 2 
        j *= 2
        if type(pos) is tuple:
            i,j = pos
        else:
            i += pos_to_idx[pos][0] 
            j += pos_to_idx[pos][1]
    #
    # Translate to finest level
    # 
    cell_level = len(node_address)
    dilation = 2**(max_level-cell_level) 
    return i*dilation, (i+1)*dilation, j*dilation, (j+1)*dilation


def fine_mesh_size(mesh):
    """
    Returns the number of cells in the x- and y-directions for the fine mesh 
    """  
    if mesh.grid_size() is None:
        """
        No initial grid
        """
        nx_0, ny_0 = 1,1
        l_max = mesh.depth()
    else:
        nx_0, ny_0 = mesh.grid_size()
        l_max = mesh.depth()-1
    
    nx, ny = nx_0*2**l_max, ny_0*2**l_max
    return nx, ny
   
    
def covariance_matrix(mesh,cov_function):
    """
    Construct the covariance matrix on the given mesh
    
    Inputs:
    
        mesh: 
        
        fs_cov: (n,n) fine scale covariance matrix
        
    Output:
    
        ms_cov: (m,m) multiscale covariance
    """ 
    l_max = mesh.depth()
    
    n_cells = mesh.get_number_of_cells()
    nx, ny = fine_mesh_size(mesh)
    x_min, x_max, y_min, y_max = mesh.box()
    dx, dy = (x_max-x_min)/nx, (y_max-y_min)/ny
    ms_cov = np.empty((n_cells,n_cells))
    nodes = mesh.root_node().find_leaves()
    for i in range(n_cells):
        node_i = nodes[i]
        for j in range(i+1):
            
            node_j = nodes[j]
            
            # Get cells contained in cell i
            i0, j0, i1, j1 = get_fine_cell_range(node_i.address, l_max)
            ii, jj = np.meshgrid(np.arange(i0,i1+1), np.arange(j0,j1+1))
            x = np.array([x_min + dx*(0.5+ii.ravel()), \
                          y_min + dy*(0.5+jj.ravel())])
            lx = node_i.depth
            
            # Do the same for cell j
            i0, j0, i1, j1 = get_fine_cell_range(node_j.address, l_max)
            ii,jj = np.meshgrid(np.arange(i0,i1+1),np.arange(j0,j1+1))
            y = np.array([x_min + dx*(0.5+ii.ravel()), \
                          y_min + dy*(0.5+jj.ravel())])
            ly = node_j.depth
            
            X, Y = np.meshgrid(x,y)
            ms_cov[i,j] = 0.25**(2*l_max-lx-ly)*np.sum(cov_function(X.ravel(),Y.ravel()))
            ms_cov[j,i] = ms_cov[i,j]
            
    return ms_cov

    
    
if __name__ == '__main__':
    #
    # Test covariance function
    # 
    # Isotropic
    x, y = np.array([0,0]), np.array([1,1])
    assert np.abs(cov_fn(x, y)-np.exp(-np.sqrt(2))) < 1e-10, \
        'Incorrect value for covariance function'
    # Anisotropic
    A = np.array([[3,-1],[-1,2]])
    #assert np.abs(cov_fn(x, y, A)-np.exp(-np.sqrt(5))) < 1e-10,\
    #    'Anisotropic test fails.'
    
    """
    nx, ny = 20, 20 
    x = np.linspace(-5,5,nx)
    y = np.linspace(0,6,ny)
    X,Y = np.meshgrid(x,y)
    xy = np.array([X.ravel(),Y.ravel()]).transpose()
    
    n_cells = xy.shape[0]
    I,J = np.meshgrid(np.arange(n_cells),np.arange(n_cells))
    XY1 = xy[I.ravel(),:]
    XY2 = xy[J.ravel(),:]
    cov = cov_fn(XY1, XY2, A).reshape((n_cells,n_cells))
    U,D,UT = np.linalg.svd(cov)
    plt.semilogy(D,'.')
    plt.show()
    
    Z = random.normal(size=(n_cells,))
    YY = U.dot(np.diag(np.sqrt(D)).dot(Z)).reshape((nx,ny))
    c = plt.contourf(X,Y,YY,50)
    #c = plt.imshow(cov[:,150].reshape((nx,ny)))
    plt.colorbar(c)
    plt.show()
    #Z = cov_fn(X.ravel(), Y.ravel()).reshape(X.shape)
    fig, ax = plt.subplots()
    
    #ax.contourf(X,Y,Z)
    """
    
        
    m = Mesh.newmesh([0,20,0,8],(10,4))
    fig,ax = plt.subplots()
    
    m.root_node().mark()
    m.refine()
    print(m.depth())
    #m.plot_quadmesh(ax, show=True, set_axis=True, cell_numbers=False)
    #plt.show()
    #covariance_matrix(m, 1)
    
    for i in range(4):
        for leaf in m.root_node().find_leaves():
            if rand() < 0.1:
                leaf.mark('s')
        m.refine('s')
    m.plot_quadmesh(ax, show=True, set_axis=True, cell_numbers=False)
    #plt.show()
    
    
    # Mesh on finest level
    l_max = m.depth()
    x0, x1, y0, y1 = m.box()
    nx, ny = m.grid_size()
    dx,dy = (x1-x0)*2**(-l_max)/nx, (y1-y0)*2**(-l_max)/ny
    print(l_max)
    print(dx,dy)
    # Pick a random node
    n_cells = m.get_number_of_cells()
    i = randint(n_cells)
    leaves = m.root_node().find_leaves()
    node = leaves[i]
    
    # Plot it
    x_min, x_max, y_min, y_max = node.quadcell().box()
    rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, facecolor='r', alpha=0.5)
    ax.add_patch(rect)
    
    # Get all fine scale boxes 
    i0,i1,j0,j1 = get_fine_cell_range(node.address,l_max)
    print(i0,i1,j0,j1)
    for i in np.arange(i0,i1+1):
        for j in np.arange(j0,j1+1):
            
            x_min, x_max = x0 + dx*i, x0 + dx*(i+1)
            y_min, y_max = y0 + dy*j, y0 + dy*(j+1)
            
            #print(x_min,y_min)
            plt.plot(np.array([x_min,x_max]), np.array([y_min,y_max]),'.b')
            #rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, facecolor='b')
            #ax.add_patch(rect)
    plt.show()
    '''
    S = covariance_matrix(m, cov_fn)
    nodes = m.root_node().find_leaves()
    S1 = S[:,0]
    print(S1)
    fig, ax = plt.subplots()
    plot = Plot()
    plot.function(ax, S1, m)
    plt.show()
    U,D,UT = svd(S)
    
    
    
    print('Number of cells = %d'%(m.get_number_of_cells()))
    
    nx,ny = m.grid_size() 
    x0,x1,y0,y1 = m.box()
    print('Box: x_min=%f, x_max=%f, y_min=%f, y_max=%f'%(x0,x1,y0,y1))
    max_depth = m.root_node().max_depth()
    print('Gridsize: nx=%i, ny=%i'%(nx,ny))
    
    dx = (x1-x0)/nx*2**(-max_depth+1)
    dy = (y1-y0)/ny*2**(-max_depth+1)
    
    print('dx=%g, dy=%g'%(dx,dy))
    a = 3.0
    b = 1.0
    ch = covh(m,a,b)
    print(ch.shape)
    #plt.imshow(ch)
    #plt.colorbar(orientation='horizontal')
    #plt.show()
    
    
    node1_address = [(2,0),3]
    node2_address = [(4,1)]
    i1_min, i1_max, j1_min, j1_max = get_fine_cell_range(node1_address, max_depth)
    i2_min, i2_max, j2_min, j2_max = get_fine_cell_range(node2_address, max_depth)
    
    print('Cell 1: (%i,%i)->(%i,%i)'%(i1_min,j1_min,i1_max,j1_max))
    
    P = np.zeros((5,3))
    count = 0
    for i in range(5):
        for j in range(3):
            P[i,j] = count
            count += 1
            
    print(P)
    print('Cell 2: (%i,%i)->(%i,%i)'%(i2_min,j2_min,i2_max,j2_max))
    
    mesh = Mesh.newmesh()
    print('Number of cells: %d'%(mesh.get_number_of_cells()))
    print('Depth: %d'%(mesh.depth()))
    print(fine_mesh_size(mesh))
    
    mesh = Mesh.newmesh(grid_size=(3,2))
    mesh.refine()
    mesh.refine()
    print(mesh.depth())
    print(mesh.get_number_of_cells())
    print(fine_mesh_size(mesh))
    '''