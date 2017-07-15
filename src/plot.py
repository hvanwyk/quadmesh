'''
Created on Feb 8, 2017

@author: hans-werner
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import *
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # @UnresolvedImport
import numpy as np
from finite_element import DofHandler, System


class Plot(object):
    """
    classdocs
    """


    def __init__(self):
        """
        Constructor
        """
        
        
        
    def mesh(self, ax, mesh, element=None, name=None, show=True, set_axis=True,
             vertex_numbers=False, edge_numbers=False, cell_numbers=False, 
             dofs=False, node_flag=None, nested=False):
        """
        Plot Mesh of QuadCells
        
        # TODO: With node_flag, we have to adjust vertex numbers, edge numbers, 
                cell numbers ...
        """
        node = mesh.root_node()
        if set_axis:
            x0, x1, y0, y1 = node.quadcell().box()          
            hx = x1 - x0
            hy = y1 - y0
            ax.set_xlim(x0-0.1*hx, x1+0.1*hx)
            ax.set_ylim(y0-0.1*hy, y1+0.1*hy)
            rect = plt.Polygon([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],
                               fc='k',alpha=0.1)
            ax.add_patch(rect)
        #
        # Plot QuadCells
        #                       
        for node in mesh.root_node().find_leaves(flag=node_flag):
            
            cell = node.quadcell()
            x0, x1, y0, y1 = cell.box()
             
            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            if node.is_marked():
                rect = plt.Polygon(points, fc='darkorange', edgecolor='k')
            else:
                rect = plt.Polygon(points, fc='w', edgecolor='k')
            ax.add_patch(rect)
        
        #
        # Plot Vertex Numbers
        #    
        if vertex_numbers:
            vertices = mesh.quadvertices()
            v_count = 0
            for v in vertices:
                x,y = v.coordinate()
                ax.text(x,y,str(v_count),size='7',
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
                        ax.text(x_pos,y_pos,str(e_count),
                                rotation=-90, size='smaller',
                                verticalalignment='center',
                                backgroundcolor='w')
                    else:
                        # horizontal
                        y_offset = 0.05*np.abs((x1-x0))
                        ax.text(x_pos,y_pos+y_offset,str(e_count),
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
                ax.text(x_pos,y_pos,str(c_count),\
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
            dofhandler.distribute_dofs(nested=True)
            for node in mesh.root_node().find_leaves():
                cell = node.quadcell()
                x0,x1,y0,y1 = cell.box()
                x_pos = x0 + x_ref[:,0]*(x1-x0)
                y_pos = y0 + x_ref[:,1]*(y1-y0)
                cell_dofs = dofhandler.get_global_dofs(node)
                for i in range(n_dofs):
                    ax.text(x_pos[i],y_pos[i],\
                            str(cell_dofs[i]), size = '7',\
                            horizontalalignment='center',
                            verticalalignment='center',
                            backgroundcolor='w')
        return ax



    def contour(self,ax, fig, f, mesh, element=None, resolution=(100,100)):
        """
        Plot a contour defined at the element nodes
        
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
            z = f(x,y)  
            cm = ax.contourf(x,y,z.reshape(ny,nx),100)
        else:
            #
            # A vector
            #
            if len(f)==mesh.get_number_of_cells():
                print('Plotting mesh function.')
                #
                # Mesh function 
                #
                patches = []
                for node in mesh.root_node().find_leaves():
                    cell = node.quadcell()
                    x0,x1,y0,y1 = cell.box()
                    rectangle = Rectangle((x0,y0), x1-x0, y1-y0)
                    patches.append(rectangle)
                    
                p = PatchCollection(patches)
                p.set_array(f)
                cm = ax.add_collection(p)
                """
                normal = colors.Normalize(f.min(), f.max())
                color = plt.cm.Greys_r(normal(f))
                count = 0
                for leaf, c in zip(mesh.root_node().find_leaves(),f):
                    cell = leaf.quadcell()
                    x0,x1,y0,y1 = cell.box()
                    X,Y = np.meshgrid(np.array([x0,x1]),np.array([y0,y1]))
                    #ax.contourf(X,Y,f_node*np.ones(X.shape), cmap='viridis',\
                    #            vmin=f.min(), vmax=f.max(), origin='lower')
                    hx,hy = x1-x0, y1-y0
                    rect = plt.Rectangle((x0,y0),hx,hy, facecolor=c)
                    ax.add_patch(rect)
                #plt.colorbar(c)
                """
            else:
                #
                # A Node contour
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
                    f_loc = f[system.get_global_dofs(node)]
                    cell = node.quadcell()
                    in_cell = cell.contains_point(xy)
                    xy_loc = xy[in_cell,:]
                    z[in_cell] = system.f_eval_loc(f_loc,cell,x=xy_loc)     
                cm = ax.contourf(x,y,z.reshape(ny,nx),100, cmap='viridis_r')
        fig.colorbar(cm, ax=ax)
        return ax
    
    
    def surface(self, ax, fig, f, mesh, element, derivatives=(0,), 
                shading=True, grid=True, resolution=(50,50),
                edge_resolution=10):
        """
        Plot the surface of a function defined on the finite element mesh
        
        Inputs: 
        
            ax, fig: axis and figure
            
            f: function, 
        """
        x0,x1,y0,y1 = mesh.box()
        hx = x1 - x0
        hy = y1 - y0
        ax.set_xlim(x0-0.1*hx, x1+0.1*hx)
        ax.set_ylim(y0-0.1*hy, y1+0.1*hy)
        system = System(mesh,element)
        if shading:
            #
            # 
            #
            
            # Define Grid
            nx, ny = resolution
            x,y = np.linspace(x0,x1,nx), np.linspace(y0,y1,ny)
            xx, yy = np.meshgrid(x,y)
            xy = np.array([xx.ravel(),yy.ravel()]).transpose()
            
            # Evaluate function
            
            zz = system.f_eval(f, xy, derivatives)
        
            ax.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                            linewidth=0, antialiased=False, alpha=0.4)
            
            
        if grid:
            #
            # Wirefunction
            # 
            ne = edge_resolution
            x_list, y_list, z_list = [], [], []
            lines = []
            for node in mesh.root_node().find_leaves():
                cell = node.quadcell()
                for edge in cell.get_edges():
                    v = edge.vertex_coordinates()
                    x0, y0 = v[0]
                    x1, y1 = v[1] 
                    
                    t = np.linspace(0,1,ne)
                    xx = (1-t)*x0 + t*x1 
                    yy = (1-t)*y0 + t*y1
                    zz = system.f_eval_loc(f, cell, x=np.array([xx,yy]).T)
                    
                    for i in range(ne-1):
                        lines.append([(xx[i],yy[i],zz[i]),(xx[i+1],yy[i+1],zz[i+1])])
                        #x_ends = [xx[i],xx[i+1]]
                        #y_ends = [yy[i],yy[i+1]]
                        #z_ends = [zz[i],zz[i+1]]
                        #x_list.extend([xx[i],xx[i+1]])
                        #y_list.extend([yy[i],yy[i+1]])
                        #z_list.extend([zz[i],zz[i+1]])
                        #lines.append((x_ends,y_ends,z_ends)) 
                        
                    #x_list.append(None)
                    #y_list.append(None)
                    #z_list.append(None)
                    
            ax.add_collection(Line3DCollection(lines, colors='k', linewidth=1))        
            #ax.plot(x_list,y_list,z_list,'k')
            