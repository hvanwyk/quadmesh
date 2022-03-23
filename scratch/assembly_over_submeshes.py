#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Assemble a problem over a finer mesh than that over which the basis is defined.
Created on Thu Jan 27 14:00:16 2022

@author: hans-werner
"""
from mesh import QuadMesh
from fem import QuadFE, DofHandler, Basis
from plot import Plot 
from assembler import Assembler
from assembler import Form
from mesh import Vertex, HalfEdge, QuadCell
import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot(111)
print(ax)

#
# Define vertices
#
v1 = Vertex((1,1))
v2 = Vertex((2,3))
v3 = Vertex((1.5,5))
v4 = Vertex((0,2))
vertices = [v1,v2,v3,v4]

#
# Define HalfEdges
# 
h12 = HalfEdge(v1,v2)
h23 = HalfEdge(v2,v3)
h34 = HalfEdge(v3,v4)
h41 = HalfEdge(v4,v1)
halfedges = [h12,h23,h34,h41]
#
# Define QuadCell
# 
cell = QuadCell(halfedges)
print(cell.is_rectangle())
for v in vertices:
    x,y = v.coordinates()
    plt.plot(x,y,'.k')
    
for he in halfedges:
    x0, y0 = he.base().coordinates()
    x1, y1 = he.head().coordinates()         
    ax.plot([x0,x1],[y0,y1], linewidth=1, color='blue')

cell.split()
for child in cell.get_children():
    for he in child.get_half_edges():
        x0, y0 = he.base().coordinates()
        x1, y1 = he.head().coordinates()         
        ax.plot([x0,x1],[y0,y1], linewidth=0.5, color='red')
    
child = cell.get_child(2)
child.split()
for grandchild in child.get_children():
    for he in grandchild.get_half_edges():
        x0, y0 = he.base().coordinates()
        x1, y1 = he.head().coordinates()         
        ax.plot([x0,x1],[y0,y1], linewidth=0.5, color='red')
        


#
# Subdivide the Reference Unit Square into SubCells and Map onto Cell
#

#
# Reference point corresponding to child cell
# 
x_ref = np.array([[0,0],[0.5,0],[0.5,0.5],[0,0.5]])
x_map = cell.reference_map(x_ref)
poly = plt.Polygon(x_map)
ax.add_patch(poly)

#
# Reference point corresponding to grandchild cell
# 
x_ref = np.array([[0.5,0.5],[0.75,0.5],[0.75,0.75],[0.5,0.75]])
x_map = cell.reference_map(x_ref)
poly = plt.Polygon(x_map)
ax.add_patch(poly)


#
# It works!! 
# 


#
# For a given subcell, determine the sub-rectangle
# 

# Pick a grandchild
#address = grandchild.get_node_address()
address = [2,0]
print(address)
ax = plt.subplot(111)
he = cell.get_half_edge(0)
x0, y0 = he.base().coordinates()
x1, y1 = he.head().coordinates()

#ax.arrow(x0,y0, x1-x0, y1-y0, width=0.01)

x0, y0 = cell.get_vertex(0).coordinates()
ax.plot(x0,y0,'*r')

for child in cell.get_children():
    x0, y0 = child.get_vertex(0).coordinates()
    ax.plot(x0,y0,'*r')

child = cell.get_child(2)
for gchild in child.get_children():
    x0, y0 = gchild.get_vertex(0).coordinates()
    ax.plot(x0,y0,'*b')

gchild = child.get_child(1)
gchild.split()
for ggchild in gchild.get_children():
    x0, y0 = ggchild.get_vertex(0).coordinates()
    ax.plot(x0,y0,'+k')

print('great grandchild address', ggchild.get_node_address())
print('cell address', cell.get_node_address())
address = [2,1,2]
sign = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1]])
relative_position = (0,0)
for level,pos in enumerate(address):
    for i in range(pos+1):
        relative_position += 2**(-(level+1))*sign[i]
print('Relative position',relative_position)
h = 2**(-(level+1))
r0, s0 = relative_position
x_ref = np.array([[r0,s0],[r0+h,s0],[r0+h,s0+h],[r0,s0+h]])
print(x_ref)

x_map = cell.reference_map(x_ref)
poly = plt.Polygon(x_map)
ax.add_patch(poly)


parent_address = [0,2]
progeny_address = [0,2,2,1,0,1]

print('Cell address', cell.get_node_address())

# Check whether progeny is contained in parent
for i,pos in enumerate(parent_address):
    assert progeny_address[i]==pos, 'WHAT?!'
rel_address = progeny_address[i+1:]
print(rel_address)

subcell = cell.find_node([2,1,2])
rel_position, width = cell.subcell_position(subcell)
print('Position using method', rel_position)
print('Width using method', width)
"""
for pos in address:
    cell = cell.get_child(pos)
    he = cell.get_half_edge(0)
    x0, y0 = he.base().coordinates()
    x1, y1 = he.head().coordinates()

    ax.arrow(x0,y0, x1-x0, y1-y0, width=0.01, color='red', edgecolor=None)
    
    x0, y0 = cell.get_vertex(0).coordinates()
    ax.plot(x0,y0,'*r')
"""
plt.show()
 

"""
# Mesh 
mesh = QuadMesh(resolution=(2,2))
mesh.cells.record(0)
mesh.cells.refine(new_label=1)

#
# Define Basis on coarse mesh
#
element = QuadFE(mesh.dim(), 'Q1')

# Dofhandler 
dofhandler = DofHandler(mesh, element)
dofhandler.distribute_dofs()

# Basis function
phi = Basis(dofhandler, subforest_flag=0)

#
# Set up assembler
# 
problem = Form(test=phi)
assembler = Assembler(problem, mesh)
assembler.assemble()
v = assembler.get_vector()
print(np.sum(v))
#
# Do the assembly explicitly
# 
shapes = np.zeros((16,4,16))
i = 0
for cell in assembler.mesh().cells.get_leaves(subforest_flag=1):
    # Get cell info
    shape_info_i = assembler.shape_info(cell)
    xg, wg, shape, dofs = assembler.shape_eval(shape_info_i, cell)
    shapes[:,:,i] =shape[cell][phi]
    i += 1
    
    # Check whether cell 
    for cell, bases in shape_info_i.items():
        v = cell.get_vertices()
        for basis in bases:
            if basis.subforest_flag()!=1:
                parent = cell.get_parent(1)
                y = basis.eval(v, cell, location='physical')
                
        
for i in range(15):
    assert np.allclose(shapes[:,:,i+1]-shapes[:,:,i],0), 'Difference not 0.'
"""