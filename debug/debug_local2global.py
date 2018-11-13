from fem import QuadFE
from fem import DofHandler
from fem import Function
from mesh import QuadMesh
from mesh import RQuadCell
from mesh import RHalfEdge
from mesh import RVertex
from plot import Plot

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TODO: Add global dofs to

mesh = QuadMesh(resolution=(2,1))
mesh.cells.get_child(1).mark('refine')
mesh.cells.refine(refinement_flag='refine')

Q = QuadFE(2, 'Q1')
dofhandler = DofHandler(mesh, Q)
dofhandler.distribute_dofs()

l2g = dict.fromkeys(mesh.cells.get_leaves())
for cell in mesh.cells.get_leaves():
    n_dofs = Q.n_dofs()
    gdofs = dofhandler.get_cell_dofs(cell)
    #
    # Initialize local-to-global map
    # 
    if l2g[cell] is None:
        #
        # Dictionary keys are global dofs in cell
        # 
        l2g[cell] = dict.fromkeys(gdofs)
        #
        # Values are expansion coefficients ito local basis
        # 
        I = np.identity(n_dofs)
        for i in range(n_dofs):
            l2g[cell][gdofs[i]] = I[i,:]
    
    #
    # Search for hanging nodes
    #     
    for he in cell.get_half_edges():
        if he.twin() is not None and he.twin().has_children():
            #
            # Edge with hanging nodes
            # 
            
            # Collect dofs on long edge
            le_ldofs = dofhandler.get_cell_dofs(cell, entity=he, doftype='local')
            le_gdofs = [gdofs[ldof] for ldof in le_ldofs]
            
            #
            # Iterate over subtending cells
            # 
            twin = he.twin()
            for che in twin.get_children():
                subcell = che.cell()
                #
                # Initialize mapping for sub-cell
                # 
                if l2g[subcell] is None:
                    #
                    # Dictionary keys are global dofs in cell
                    # 
                    sc_gdofs = dofhandler.get_cell_dofs(subcell)
                    l2g[subcell] = dict.fromkeys(sc_gdofs)
                    #
                    # Values are expansion coefficients ito local basis
                    # 
                    I = np.identity(n_dofs)
                    for i in range(n_dofs):
                        l2g[subcell][sc_gdofs[i]] = I[i,:]
                        
                # =============================================================
                # Expansion coefficients of global basis function on sub-cell 
                # =============================================================
            
                #    
                # Local dofs on sub-edge
                #    
                se_ldofs = dofhandler.get_cell_dofs(subcell, entity=che, \
                                                    doftype='local')
                
                #
                # Get vertices associated with these local dofs
                # 
                rsub = dofhandler.element.reference_nodes()[se_ldofs,:]
                x = subcell.reference_map(rsub, mapsto='physical')
                
                #
                # Evaluate coarse scale basis functions at fine scale vertices
                # 
                r = cell.reference_map(x, mapsto='reference')
                
                for le_ldof, le_gdof in zip(le_ldofs, le_gdofs):
                    #
                    # Evaluate global basis function at all sub-edge dof-verts
                    # 
                    vals = dofhandler.element.phi(le_ldof,r)
                    coefs = np.zeros(n_dofs)
                    coefs[se_ldofs] = vals
                    l2g[subcell][le_gdof] = coefs
    

ig = 1  # global basis 1
n_dofs = dofhandler.n_dofs()
fn = np.zeros(n_dofs)
for cell in mesh.cells.get_leaves():
    if ig in l2g[cell].keys():
        ldof = 0
        for gdof in dofhandler.get_cell_dofs(cell):
            fn[gdof] = l2g[cell][ig][ldof]
            ldof += 1
phi = Function(fn, 'nodal', dofhandler=dofhandler)       
"""    
#
# Modify l2g mapping
# 
for ldof,i in zip(le_ldofs,range(len(le_ldofs))):
    gdof = l2g[cell]['global'][ldof]
    try:
        #
        # Check if global long edge dof appears in sub-cell,
        # 
        i_gdof = l2g[sub_cell]['global'].index(gdof)
        
        #
        # Modify the corresponding row of the l2g matrix
        #
        l2g[sub_cell]['local'][i_gdof][se_ldofs] = vals[i] 
    except ValueError:
        continue

print(l2g[sub_cell]['local'])
"""
"""
    
# Plot mesh
plot = Plot()
plot.mesh(mesh, dofhandler=dofhandler, dofs=True)

# Plot global basis function at one of the coarse nodes
coarse_cell = mesh.cells.get_child(1)
dof = l2g[coarse_cell]['global'][le_ldofs[0]]
print('plotting global basis function ', dof)

fn = np.zeros(dofhandler.n_dofs())
for cell in mesh.cells.get_leaves():
    gdofs = l2g[cell]['global']
    print('In cell with global dofs', gdofs)
    try:
        i_dof = gdofs.index(dof)
        fn[gdofs] = l2g[cell]['local'][i_dof,:]
    except ValueError:
        continue

b = Function(fn, 'nodal', dofhandler=dofhandler)
print(fn)

"""
plot = Plot()
plot.wire(phi)
plot.mesh(mesh, dofhandler=dofhandler, dofs=True)
 