"""
Basic AMR using the DWR method 
"""
import numpy as np
from numpy import exp as e
from mesh import Mesh
from finite_element import QuadFE, System, DofHandler
from plot import Plot
import matplotlib.pyplot as plt

u = lambda x,y: 5*x**2*(1-x)**2*(e(10*x**2)-1)*y**2*(1-y)**2*(e(10*y**2)-1)
f = lambda x,y: 10*((e(10*x**2)-1)*(x-1)**2*x**2* (e(10*y**2)-1)*(y-1)**2\
                    + (e(10*x**2)-1)*(x-1)**2*x**2*(e(10*y**2)-1)*y**2\
                    + 50*(e(10*x**2)-1)*(x-1)**2*x**2*e(10*y**2)*(y-1)**2*y**2 
                    + 50*e(10*x**2)*(x-1)**2*x**2*(e(10*y**2)-1)*(y-1)**2*y**2\
                    + (e(10*x**2)-1)*x**2*(e(10*y**2)-1)*(y-1)**2*y**2\
                    + 4*(e(10*x**2)-1)*(x-1)**2*x**2*(e(10*y**2)-1)*(y-1)*y\
                    + 4*(e(10*x**2)-1)*(x-1)*x*(e(10*y**2)-1)*(y-1)**2*y**2\
                    + (e(10*x**2)-1)*(x-1)**2*(e(10*y**2)-1)*(y-1)**2*y**2\
                    + 200*(e(10*x**2)-1)*(x-1)**2*x**2*e(10*y**2)*(y-1)**2*y**4\
                    + 40*(e(10*x**2)-1)*(x-1)**2*x**2*e(10*y**2)*(y-1)*y**3\
                    + 200*e(10*x**2)*(x-1)**2*x**4*(e(10*y**2)-1)*(y-1)**2*y**2\
                    + 40*e(10*x**2)*(x-1)*x**3*(e(10*y**2)-1)*(y-1)**2*y**2)
    
mesh = Mesh.newmesh(grid_size=(8,8))
mesh.refine()
element = QuadFE(2,'Q1')
fig,ax = plt.subplots()
plot = Plot()
plot.function(ax, f, mesh, element)
plt.show()