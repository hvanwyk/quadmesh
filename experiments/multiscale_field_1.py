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
   
    
def covariance_matrix(mesh, fs_cov):
    """
    Construct the covariance matrix on the given mesh
    
    Inputs:
    
        mesh: 
        
        fs_cov: (n,n) fine scale covariance matrix
        
    Output:
    
        ms_cov: (m,m) multiscale covariance
    """ 
    l_max = mesh.depth()
    n_cells_fine = 2**l_max 
    n_cells = mesh.get_number_of_cells()
    x_min, x_max, y_min, y_max = mesh.box()
    ms_cov = np.empty((n_cells,n_cells))
    nodes = mesh.root_node().find_leaves()
    for i in range(n_cells):
        node_i = nodes[i]
        for j in range(i+1):
            node_j = nodes[j]
            i_xmin, i_ymin, i_xmax, i_ymax = \
                get_fine_cell_range(node_i.address, l_max)
            # Convert to linear index
            ny = i_ymax - i_ymin  # TODO: must be the number of fine cells in the y-direction 
            ii_min = i_ymin*ny + i_xmin  
            ii_max = i_ymax*ny + i_xmax
                
            j_xmin, j_ymin, j_xmax, j_ymax = \
                get_fine_cell_range(node_j.address, l_max)
            # Convert to linear index
            ny = j_ymax - j_ymin 
            ii_min = i_ymin*ny + i_xmin  
            ii_max = i_ymax*ny + i_xmax
            
                
            ms_cov[i,j] = 0


m = Mesh.newmesh([0,20,0,8],(5,2))
_,ax = plt.subplots()

m.root_node().mark()
m.refine()
#m.plot_quadmesh(ax, show=True, set_axis=True, cell_numbers=False)
#plt.show()
#covariance_matrix(m, 1)

for i in range(5):
    for leaf in m.root_node().find_leaves():
        if rand() < 0.5:
            leaf.mark('s')
    m.refine('s')
#m.plot_quadmesh(ax, show=True, set_axis=True, cell_numbers=False)
#plt.show()

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