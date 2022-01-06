#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Investigate local error estimates on the resolution of a random field

Created on Thu Oct 28 16:13:19 2021

@author: hans-werner
"""


from assembler import Assembler
from assembler import Kernel
from assembler import Form
from fem import DofHandler
from fem import QuadFE
from fem import Basis
from function import Nodal
from gmrf import Covariance
from gmrf import GaussianField
from mesh import QuadMesh
from plot import Plot
import TasmanianSG
import time
from diagnostics import Verbose

# Built-in modules
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

plot = Plot(quickview=False)

# Computational Mesh 
mesh = QuadMesh(resolution=(5,5))

# Mark boundary
bnd_fn = lambda x,y: abs(x)<1e-6 or abs(1-x)<1e-6 or abs(y)<1e-6 or abs(1-y)<1e-6 
mesh.mark_region('bnd', bnd_fn, entity_type='half_edge', on_boundary=True)

# Mark averaging region
dmn_fn = lambda x,y: x>=0.75 and x<=1 and y>=0.75 and y<=1
mesh.mark_region('dmn', dmn_fn, entity_type='cell', 
                 strict_containment=True, on_boundary=False) 
#cells = mesh.get_region(flag='dmn', entity_type='cell', on_boundary=False, subforest_flag=None)
#plot.mesh(mesh, regions=[('bnd','edge'),('dmn','cell')])

# Define Elements
Q0 = QuadFE(mesh.dim(),'DQ0')
dQ0 = DofHandler(mesh,Q0)
dQ0.distribute_dofs()

phi = Basis(dQ0)

q = Nodal(f=lambda x: np.sin(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1]), dim=2, 
          basis=phi)


problem = Form(q)

mesh.cells.refine(new_label='fine_mesh')
assembler = Assembler(problem, mesh, subforest_flag='fine_mesh')
assembler.assemble()
 