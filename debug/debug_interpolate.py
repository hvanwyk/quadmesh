from mesh import Mesh
from finite_element import QuadFE, System
import scipy.sparse as sp

mesh = Mesh.newmesh()
# Coarse mesh (2,2)
mesh.refine()
mesh.record()  # label 0

# Fine mesh 4,4
mesh.refine()
mesh.record()  # label 1
element = QuadFE(2,'Q1')
system = System(mesh, element, nested=True)
rows = []
cols = []
vals = []
for node in mesh.root_node().find_leaves(1):
    if node.has_parent(0):
        parent = node.get_parent(0)
        node_dofs = system.dofhandler().get_global_dofs(node)
        parent_dofs = system.dofhandler().get_global_dofs(parent)
        x = system.dofhandler().dof_vertices(node)
        phi = system.shape_eval(cell=parent.quadcell(), x=x)
        for i in range(len(node_dofs)):
            fine_dof = node_dofs[i]
            if fine_dof not in rows:
                #
                # New fine dof
                # 
                for j in range(len(parent_dofs)):
                    coarse_dof = parent_dofs[j]
                    phi_val = phi[i,j] 
                    if abs(phi_val) > 1e-9:
                        rows.append(fine_dof)
                        cols.append(coarse_dof)
                        vals.append(phi_val)
                        
n_dofs_coarse =  system.dofhandler().n_dofs(0)
n_dofs_fine = system.dofhandler().n_dofs(1)
A = sp.coo_matrix((vals,(rows,cols)),shape=(n_dofs_fine,n_dofs_coarse))
