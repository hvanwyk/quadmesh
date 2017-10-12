from fem import QuadFE, DofHandler
from mesh import Mesh

"""
TODO: Quick fix for piecewise constant functions 
Must resolve the "sharing of dofs with children...
"""
mesh = Mesh.newmesh()
mesh.refine()
element = QuadFE(2,'DQ0')
dofhandler = DofHandler(mesh,element)
for node in mesh.root_node().traverse_depthwise():
    node.info()
    dofhandler.fill_dofs(node)
    print('Dofs: {0}'.format(dofhandler.get_global_dofs(node)))
    
    
#dofhandler.distribute_dofs(nested=True)