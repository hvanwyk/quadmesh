from mesh import Mesh
from finite_element import QuadFE, DofHandler
import matplotlib.pyplot as plt

# TODO: Add this all to the test file.
        
mesh = Mesh.newmesh()
mesh.root_node().mark()
mesh.refine()

mesh.root_node().children['SE'].mark()
mesh.refine()

mesh.root_node().children['SE'].children['SW'] = None
_,ax = plt.subplots()


element_type = 'Q2'
V = QuadFE(2,element_type)
d = DofHandler(mesh,V)
d.distribute_dofs()
order = int(list(element_type)[1])

x = [i*1.0/order for i in range(order+1)]
xy = []
for xi in x:
    for yi in x:
        xy.append((xi,yi))
zlist = []
for n in range(order+1):
    print('\n\n')
    for yi in x:
        zline = []
        for xi in x:
            zline.append(V.phi(n,(xi,yi)))
        print([v for v in zline])       
"""    
print('Evaluating shape functions at hanging nodes.')
x = [i*0.5/order for i in range(2*order+1)]
xy = [(xi,1.0) for xi in x]
for xx in xy:
    print('Point {0}'.format(xx))
    zline = []
    for n in range(12):
        zline.append(V.phi(n,xx))
    print([v for v in zline])
    print('\n')
"""    
        
            

C = d.make_hanging_node_constraints()
print(C)
c = V.constraint_coefficients()
mesh.plot_quadmesh(ax, cell_numbers=True, vertex_numbers=True )
plt.show()
'''
# Number nodes Q1

count = 0
for node in mesh.root_node().find_leaves():
    node.dofs = dict.fromkeys(['SW','SE','NW','NE'])
    for key in ['SW','SE','NW','NE']:
        if node.dofs[key] == None:
            node.dofs[key] = count
            #
            # Shared cells
            # 
            for direction in list(key).append(key):
                nb = node.find_neighbor(direction)
                if nb != None
                    if nb.has_children():
                        nb = nb.children[opposite[direction]]
                    
                    
            no_neighbor = dict.fromkeys(list(key), False)
            print(no_neighbor)
            for direction in list(key):
                nb = node.find_neighbor(direction)
                if nb != None:
                    pos = key.replace(direction,opposite[direction])
                    if nb.has_children():
                        child = nb.children[pos]
                        shared_cells[direction] = {pos: child}
                    else:
                        shared_cells[direction] = {pos: nb}
                else:
                    no_neighbor[direction] = True
            # Diagonal neighbor
            if all(no_neighbor.values()):
                # No neighbors in either direction
                
                
                    """
                    if nb.has_children():
                        new_key = 
                        nb = nb.children[new_key]
                        if nb != None:
                            if not hasattr(nb,'dofs'):
                                nb.dofs = dict.fromkeys(['SW','SE','NW','NE'])
                                
                            if nb.dofs[key] != None:
                                nb.dofs().vertices[new_key] = count
                    else:
                        if nb.dofs[key] != None:
                            nb.dofs[key.replace(direction,opposite[direction])] = count
                    
                    if not hasattr(nb,'dofs'):
                        nb.dofs = dict.fromkeys(['SW','SE','NW','NE'])
                    """
            print(no_neighbor)
        count += 1
        
    
'''      