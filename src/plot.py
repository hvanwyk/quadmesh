'''
Created on Feb 8, 2017

@author: hans-werner
'''

import matplotlib.pyplot as plt
from matplotlib import colors, cm 
import numpy as np
from finite_element import DofHandler, System

class Plot(object):
    """
    classdocs
    """


    def __init__(self, ax=None):
        """
        Constructor
        """
        if ax is None:
            _, ax = plt.subplots()
        self.__ax = ax
        
        
    def mesh(self, mesh, element=None, name=None, show=True, set_axis=True,
             vertex_numbers=False, edge_numbers=False, cell_numbers=False, 
             dofs=False):
            """
            Plot Mesh of QuadCells
            """
            node = mesh.root_node()
            if set_axis:
                x0, x1, y0, y1 = node.quadcell().box()          
                hx = x1 - x0
                hy = y1 - y0
                self.__ax.set_xlim(x0-0.1*hx, x1+0.1*hx)
                self.__ax.set_ylim(y0-0.1*hy, y1+0.1*hy)
                rect = plt.Polygon([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],
                                   fc='b',alpha=0.5)
                self.__ax.add_patch(rect)
            #
            # Plot QuadCells
            #                       
            for cell in mesh.iter_quadcells():
                 
                x0, y0 = cell.vertices['SW'].coordinate()
                x1, y1 = cell.vertices['NE'].coordinate() 
    
                # Plot current cell
                # plt.plot([x0, x0, x1, x1],[y0, y1, y0, y1],'r.')
                points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                if cell.is_marked():
                    rect = plt.Polygon(points, fc='r', edgecolor='k')
                else:
                    rect = plt.Polygon(points, fc='w', edgecolor='k')
                self.__ax.add_patch(rect)
            
            #
            # Plot Vertex Numbers
            #    
            if vertex_numbers:
                vertices = mesh.iter_quadvertices()
                v_count = 0
                for v in vertices:
                    x,y = v.coordinate()
                    self.__ax.text(x,y,str(v_count),size='7',
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   backgroundcolor='w')
                    v_count += 1
                    
            #
            # Plot Edge Numbers
            #
            if edge_numbers:
                edges = mesh.iter_quadedges()
                e_count = 0
                for e in edges:
                    if not(e.is_marked()):
                        v1, v2 = e.vertices()
                        x0,y0 = v1.coordinate()
                        x1,y1 = v2.coordinate()
                        x_pos, y_pos = 0.5*(x0+x1),0.5*(y0+y1)
                        if x0 == x1:
                            # vertical
                            self.__ax.text(x_pos,y_pos,str(e_count),
                                           rotation=-90, size='smaller',
                                           verticalalignment='center',
                                           backgroundcolor='w')
                        else:
                            # horizontal
                            y_offset = 0.05*np.abs((x1-x0))
                            self.__ax.text(x_pos,y_pos+y_offset,str(e_count),
                                           size='7',
                                           horizontalalignment='center',
                                           backgroundcolor='w')                 
                        e_count += 1
                    e.mark()
            
            #
            # Plot Cell Numbers
            #
            if cell_numbers:
                cells = mesh.iter_quadcells()
                c_count = 0
                for c in cells:
                    x0,x1,y0,y1 = c.box()
                    x_pos, y_pos = 0.5*(x0+x1), 0.5*(y0+y1)
                    self.__ax.text(x_pos,y_pos,str(c_count),\
                                   horizontalalignment='center',
                                   verticalalignment='center',
                                   size='smaller')
                    c_count += 1
    
            #
            # Degrees of freedom
            # 
            if dofs:
                assert element is not None, \
                'Require element information to plot dofs'
                x_ref = element.reference_nodes()
                n_dofs = element.n_dofs()
                dofhandler = DofHandler(mesh, element)
                dofhandler.distribute_dofs()
                for node in mesh.root_node().find_leaves():
                    cell = node.quadcell()
                    x0,x1,y0,y1 = cell.box()
                    x_pos = x0 + x_ref[:,0]*(x1-x0)
                    y_pos = y0 + x_ref[:,1]*(y1-y0)
                    cell_dofs = dofhandler.get_node_dofs(node)
                    for i in range(n_dofs):
                        self.__ax.text(x_pos[i],y_pos[i],\
                                       str(cell_dofs[i]), size = '7',\
                                       horizontalalignment='center',
                                       verticalalignment='center',
                                       backgroundcolor='w')
                    

    def function(self,f,mesh,element=None,resolution=(100,100)):
        """
        Plot a function defined at the element nodes
        
        Loop over cells
            get local dofs
            evaluate shapefunctions
            
        """
        #
        # Initialize grid
        # 
        x0,x1,y0,y1 = mesh.box()
        nx, ny = resolution
        x_range = np.linspace(x0,x1,nx)
        y_range = np.linspace(y0,y1,ny)
        x,y = np.meshgrid(x_range,y_range)
        if callable(f):
            #
            # A function 
            # 
            z = f(x,y)  # TODO: Wrong plot if domain has gaps.
            plt.contourf(x,y,z.reshape(ny,nx),100)
        else:
            #
            # A vector
            #
            if len(f)==mesh.get_number_of_cells():
                #
                # Mesh function 
                #
                #my_map = cm.viridis
                normal = colors.Normalize(f.min(), f.max())
                color = plt.cm.viridis(normal(f))
                for leaf,c in zip(mesh.root_node().find_leaves(),color):
                    cell = leaf.quadcell()
                    x0,x1,y0,y1 = cell.box()
                    hx,hy = x1-x0, y1-y0
                    rect = plt.Rectangle((x0,y0),hx,hy, facecolor=c)
                    self.__ax.add_patch(rect)
            else:
                #
                # A Node function
                #  
                assert element is not None, \
                'Require element information for node functions'
                
                system = System(mesh,element)
                
                assert len(f)==system.get_n_nodes(), \
                'Functions vectors should have length n_cells or n_dofs' 
                      
                xy = np.array([x.flatten(),y.flatten()]).T
                z = np.empty((nx*ny,))
                z[:] = np.nan
        
                for node in mesh.root_node().find_leaves():
                    f_loc = f[system.get_node_dofs(node)]
                    cell = node.quadcell()
                    in_cell = cell.contains_point(xy)
                    xy_loc = xy[in_cell,:]
                    z[in_cell] = system.f_eval_loc(f_loc,cell,x=xy_loc)     
                self.__ax = plt.contourf(x,y,z.reshape(ny,nx),100)
