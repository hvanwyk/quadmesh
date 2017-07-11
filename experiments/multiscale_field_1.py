'''
Created on Feb 2, 2017

Multiscale Gaussian random field with standard anisotropic, homogeneous 
exponential covariance matrix. 

@author: hans-werner

Strategy for obtaining covariance matrix: 

    On finest mesh level: Compute the covariance for all relative distances
    Loop over the cells: 
'''
from mesh import Mesh
import matplotlib.pyplot as plt
from numpy.random import rand, randint
import numpy as np

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
  
    
def get_fine_cell_range(node_address,max_level):
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


def fine_scale_midpoints():
    """
    Return the midpoints of 
    """   
    pass
    
def covariance_matrix(mesh, cov):
    """
    Construct the covariance matrix on the given mesh 
    """ 
    n = mesh.depth()
    x_min, x_max, y_min, y_max = mesh.box()
    


m = Mesh.newmesh([0,20,0,8],(5,2))
_,ax = plt.subplots()

m.root_node().mark()
m.refine()
#m.plot_quadmesh(ax, show=True, set_axis=True, cell_numbers=False)
#plt.show()

for i in range(5):
    for leaf in m.root_node().find_leaves():
        if rand() < 0.8:
            leaf.mark('s')
    m.refine('s')
m.plot_quadmesh(ax, show=True, set_axis=True, cell_numbers=False)
plt.show()

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
plt.imshow(ch)
plt.colorbar(orientation='horizontal')
plt.show()

node1_address = [(2,0),3]
node2_address = [(4,1)]
i1_min, i1_max, j1_min, j1_max = get_fine_cell_range(node1_address, max_depth)
i2_min, i2_max, j2_min, j2_max = get_fine_cell_range(node2_address, max_depth)

print('Cell 1: (%i,%i)->(%i,%i)'%(i1_min,j1_min,i1_max,j1_max))
print('Cell 2: (%i,%i)->(%i,%i)'%(i2_min,j2_min,i2_max,j2_max))