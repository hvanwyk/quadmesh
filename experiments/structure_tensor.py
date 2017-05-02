# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:20:08 2017

@author: hans-werner
"""

import os
from skimage import io, feature
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm
from scipy.interpolate import interp2d
from finite_element import System, QuadFE
from mesh import Mesh
from gmrf import Gmrf
from scipy.interpolate.fitpack2 import RectBivariateSpline

def two_by_two_eig(a,b,c):
    """
    Compute the eigenvalues and -vectors for a symmetric positive definite
    matrix 
    
        A = | a b | 
            | b c |
    
    """
    T = a + c
    D = a*c - b**2
    #
    # Eigenvalues
    # 
    l1 = T/2 + np.sqrt(T**2/4 - D)
    l2 = T/2 - np.sqrt(T**2/4 - D)
    D = np.array([l1,l2])   
    #
    # Eigenvectors
    # 
    v1 = np.array([l1-c,b])
    v2 = np.array([l2-c,b])
    V = np.array([v1/np.linalg.norm(v1),v2/np.linalg.norm(v2)]).transpose()
    
    return D,V


def structure_ellipse(point, Axx, Ayy, Axy):
    d,V = two_by_two_eig(Axx, Axy, Ayy)
    w, h = 2*d[0], 2*d[1]
    angle = -np.arctan2(V[0,1],V[1,1])*180/np.pi
    return Ellipse(point, h, w, angle, ec='y', fc='none')
#
# Load Image
# 
#filename = os.getcwd() + '/shale.png'
#filename = os.getcwd() + '/picture8a-4x3-swim.png'
#filename = '/home/hans-werner/desktop/shale.jpg'
filename = '/home/hans-werner/Dropbox/work/projects/' + \
           'spatially_indexed_noise/code/structure_tensor' + \
           '/picture8a-4x3-swim.png'
shale = io.imread(filename)
shale = shale[:,:,0]  # There is only 1 layer
#
# Plot image
# 
fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.imshow(shale, cmap=cm.Greys_r);

#
# Get structure tensor
# 
Axx, Axy, Ayy = feature.structure_tensor(shale)
nx, ny = Axx.shape
dx, dy = 2, 2
xi = np.arange(0, nx)
yi = np.arange(0, ny)
axx = RectBivariateSpline(xi, yi, Axx)
x = np.array([1,2,3])
y = np.array([4,5,6])
print(axx.ev(x,y))
axx = interp2d(xi, yi, Axx.T) 
axy = interp2d(xi, yi, Axy.T)
ayy = interp2d(xi, yi, Ayy.T)
tau = (axx, axy, ayy)

mesh = Mesh.newmesh(box=[0,nx,0,ny], grid_size=(60,80))
mesh.refine()
element = QuadFE(2,'Q1')
alpha = 2
kappa = 9


#X = Gmrf.from_matern_pde(alpha,kappa,mesh,element,tau)


for ix in np.arange(0,nx,dx):
    for iy in np.arange(0,ny,dy):
        e = structure_ellipse((iy,ix), Axx[ix,iy],Axy[ix,iy],Ayy[ix,iy])
        ax.add_artist(e)    
        
plt.show()


"""

l1, l2 = feature.structure_tensor_eigvals(Axx, Axy, Ayy)
#
# Test Case
#
a,b,c = Axx[20,20], Axy[20,20], Ayy[20,20]
P = np.array([[Axx[20,20],Axy[20,20]],[Axy[20,20],Ayy[20,20]]])
D,V = np.linalg.eig(P)
D1,V1 = two_by_two_eig(a,b,c)
print('Eigs: D = \n{0} \n\n V = \n{1}'.format(np.diag(D),V))
print('-----------------------------------------')
print('Eigs: D1 = \n{0} \n\n V1 = \n{1}'.format(np.diag(D1),V1))

#
# Make ellipses
#

P = np.array([[Axx[20,20],Axy[20,20]],[Axy[20,20],Ayy[20,20]]])

# Direct
tht = np.linspace(0,2*np.pi)
xy = np.array([np.cos(tht),np.sin(tht)])
z = P.dot(xy)

# Eigendecomposition
D,V = np.linalg.eig(P)
h, w = 2*D[0], 2*D[1]
angle = np.arctan2(-V[1,1],V[0,1])*180/np.pi


e = Ellipse((0,0), w, h, angle, ec='r', fc='none')


fig = plt.figure(0)
ax = fig.add_subplot(111)
ax.plot(z[0,:],z[1,:],'y-.')
ax.add_artist(e)
ax.plot([0,D[0]*V[1,0]],[0,D[0]*V[0,0]])

# Plot any ellipse
xy = (20,20)
w = 20
h = 10
angle = -30
fig = plt.figure(2)
ax = fig.add_subplot(111)
ax.imshow(shale, cmap=cm.Greys_r);
e = Ellipse(xy, w, h, angle, ec='y', fc='none')
fig = plt.figure(0);
#x = fig.add_subplot(111, aspect='equal')
ax.add_artist(e);
e.set_clip_box(ax.bbox)
#ax.set_xlim(0,4)
#ax.set_ylim(0,4)

plt.show();
"""