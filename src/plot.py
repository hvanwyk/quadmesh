'''
Created on Feb 8, 2017

@author: hans-werner
'''

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # @UnresolvedImport
from mpl_toolkits.mplot3d import axes3d # @UnresolvedImport
import numpy as np
from fem import DofHandler, System, Function


class Plot(object):
    """
    Plots related to finite element mesh and functions
    """
    
    def __init__(self):
        """
        Constructor
        """
        
        
    def mesh(self, ax, mesh, element=None, show_axis=False, color_marked=None,
             vertex_numbers=False, edge_numbers=False, cell_numbers=False, 
             dofs=False, node_flag=None, nested=False):
        """
        Plot computational mesh
        
        Inputs: 
        
            ax: current axes
            
            mesh: Mesh, computational mesh
            
            *element: QuadFE, element
            
            *show_axis: boolean, set axis on or off
            
            *color_marked: list of flags for cells that must be colored
            
            *vertex/edge/cell_numbers: bool, display vertex/edge/cell numbers.
            
            *dofs: boolean, display degrees of freedom
            
            *node_flag: boolean, plot only cells with the given flag
            
            *nested: boolean, traverse grid in a nested fashion. 
        
        
        Outputs:
        
            ax: axis, 
            
        """
        node = mesh.root_node()
        
        if mesh.dim() == 2:
            #
            # Two dimensional mesh
            # 
            v_cnr = node.cell().get_vertices(pos='corners', as_array=True)            
            x0, x1 = v_cnr[:,0].min(), v_cnr[:,0].max()
            y0, y1 = v_cnr[:,0].min(), v_cnr[:,0].max()
        
            
        hx = x1 - x0
        hy = y1 - y0
        ax.set_xlim(x0-0.1*hx, x1+0.1*hx)
        ax.set_ylim(y0-0.1*hy, y1+0.1*hy) 
        
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        rect = plt.Polygon(points, fc='darkgrey', edgecolor='k', alpha=0.1)
        ax.add_patch(rect)
        
        #
        # Plot QuadCells
        # 
        color_list = ['gold', 'darkorange','r']                      
        for node in mesh.root_node().get_leaves(flag=node_flag, \
                                                 nested=nested):
            cell = node.cell()
            x0, x1, y0, y1 = cell.box()
            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            if color_marked is not None:
                count = 0
                if not any(node.is_marked(flag) for flag in color_marked):
                    # Node contains none of the listed flags -> make it white
                    rect = plt.Polygon(points, fc='w', ec='k')
                else:
                    for flag in color_marked:
                        if node.is_marked(flag):  
                            # Color cell                      
                            rect = plt.Polygon(points, fc=color_list[count],\
                                               edgecolor='k')    
                        count += 1
            else:
                rect = plt.Polygon(points, fc='w', edgecolor='k')
            ax.add_patch(rect)
        
        #
        # Plot Vertex Numbers
        #    
        if vertex_numbers:
            vertices = mesh.quadvertices(flag=node_flag, nested=nested, \
                                         coordinate_array=False)
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
            edges = mesh.iter_quadedges(flag=node_flag, nested=nested)
            e_count = 0
            for e in edges:
                if not(e.is_marked()):
                    v1, v2 = e.vertices()
                    x0,y0 = v1.coordinate()
                    x1,y1 = v2.coordinate()
                    x_pos, y_pos = 0.5*(x0+x1),0.5*(y0+y1)
                    if x0 == x1:
                        # vertical
                        x_offset = 0*np.abs(x1-x0)
                        ax.text(x_pos,y_pos-x_offset,str(e_count),
                                rotation=-90, size='7',
                                verticalalignment='center',
                                horizontalalignment='center',
                                backgroundcolor='w')
                    else:
                        # horizontal
                        y_offset = 0*np.abs(y1-y0)
                        #y_offset = 0
                        ax.text(x_pos,y_pos-y_offset,str(e_count),
                                size='7',
                                horizontalalignment='center',
                                verticalalignment='center',
                                backgroundcolor='w')                 
                    e_count += 1
                e.mark()
        
        #
        # Plot Cell Numbers
        #
        if cell_numbers:
            cells = mesh.iter_quadcells(flag=node_flag, nested=nested)
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
            dofhandler.distribute_dofs(nested=nested)
            for node in mesh.root_node().get_leaves(nested=nested,\
                                                     flag=node_flag):
                cell = node.cell()
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
        if not show_axis:
            ax.axis('off')
            
        return ax



    def contour(self, ax, fig, f, mesh, element=None, derivative=(0,), \
                colorbar=True, resolution=(100,100), flag=None):
        """
        Returns a contour plot of a function f
        
        
        Inputs:
        
            ax: Axis, current axes
            
            fig: Figure, current figure
            
            f: Function, function to be plotted
            
            mesh: Mesh, computational mesh
            
            *element [None]: TODO: Not necessary if plotting a Function 
            
            *derivative [(0,)]: int, tuple specifying the function's derivative
            
            *colorbar [True]: bool, add a colorbar?
            
            *resolution [(100,100)]: int, tuple resolution of contour plot.
            
            *flag [None]: str/int, specifying submesh on which to evaluate f
                TODO: Unnecessary.
            
            
        Outputs: 
        
            ax
            
            fig
                    
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
        elif isinstance(f, Function):
            xy = [(xi,yi) for xi,yi in zip(x.ravel(),y.ravel())]
            z = f.eval(xy)
            cm = ax.contourf(x,y,z.reshape(ny,nx),100)
        else:
            #
            # A vector
            #
            if len(f)==mesh.n_nodes():
                #
                # Mesh function 
                #
                patches = []
                for node in mesh.root_node().get_leaves(flag=flag):
                    cell = node.cell()
                    x0,x1,y0,y1 = cell.box()
                    rectangle = Rectangle((x0,y0), x1-x0, y1-y0)
                    patches.append(rectangle)
                    
                p = PatchCollection(patches)
                p.set_array(f)
                cm = ax.add_collection(p)
            else:
                #
                # A Node contour
                #  
                assert element is not None, \
                'Require element information for node functions'
                
                xy = np.array([x.ravel(), y.ravel()]).transpose()
                system = System(mesh,element)
                z = system.f_eval(f, xy, derivatives=derivative)
                cm = ax.contourf(x,y,z.reshape(ny,nx),200, cmap='viridis')
                
        if colorbar:
            fig.colorbar(cm, ax=ax, format='%g')
            
        return fig, ax, cm
    
    
    def surface(self, ax, f, mesh=None, element=None, derivatives=(0,), 
                shading=True, grid=False, resolution=(100,100),
                edge_resolution=10, flag=None):
        """
        Plot the surface of a function defined on the finite element mesh
        
        Inputs: 
        
            ax: axis (don't forget to initialize it using projection='3d')
            
            f: Function, function to be plotted
            
            mesh: Mesh, on which to plot the function 
            
            *derivatives [(0,)]: int, tuple specifying what derivatives to
                plot (see Function.eval for details).
            
            *shading [True]: bool, shade surface or use wire plot? 
            
            *grid [False]: bool, display grid? 
            
            *resolution [(100,100)]: int, tuple (nx,ny) number of points 
                in the x- and y directions. 
            
            *edge_resolution: int, number of points along each each edge
            
            *flag [None]: str/int marker for submesh TODO: Not implemented
            
        
        Output:
        
            ax: Axis, containing plot.
        
        """
        if mesh is None:
            if isinstance(f, Function) and f.mesh is not None:
                mesh = f.mesh
            else:
                mesh_error = 'Mesh must be specified, either explicitly, '+\
                    'or as part of the Function.'
                raise Exception(mesh_error)
            
        x0,x1,y0,y1 = mesh.box()        
        system = System(mesh,element)
        if shading:
            #
            # Colormap
            #
            
            # Define Grid
            nx, ny = resolution
            x,y = np.linspace(x0,x1,nx), np.linspace(y0,y1,ny)
            xx, yy = np.meshgrid(x,y)
            xy = np.array([xx.ravel(),yy.ravel()]).transpose()
        
            # Evaluate function
            zz = system.f_eval(f, xy, derivatives)
            z_min, z_max = zz.min(), zz.max()
            
            if grid:
                alpha = 0.5
            else:
                alpha = 1
            ax.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                            linewidth=1, antialiased=True, alpha=alpha)
            
            
        if grid:
            #
            # Wirefunction
            # 
            ne = edge_resolution
            lines = []
            node_count = 0
            initialize_min_max = True
            for node in mesh.root_node().get_leaves():                
                #
                # Function type  
                # 
                if callable(f):
                    #
                    # Explicit function
                    #
                    assert derivatives==(0,),\
                        'Discretize before plotting derivatives.'
                    f_loc = f
                elif isinstance(f, Function):
                    #
                    # Function object
                    #
                    f_loc = f    
                elif len(f)==system.n_dofs():
                    #
                    # Nodal function
                    #
                    f_loc = f[system.get_global_dofs(node)]
            
                elif len(f)==mesh.n_nodes():
                    #
                    # Mesh function
                    #
                    f_loc = f[node_count] 
                
                cell = node.cell()
                for edge in cell.get_edges():
                    # Points on edges
                    v = edge.vertex_coordinates()
                    x0, y0 = v[0]
                    x1, y1 = v[1] 
                    t = np.linspace(0,1,ne)
                    xx = (1-t)*x0 + t*x1 
                    yy = (1-t)*y0 + t*y1
                    
                    # Evaluate function at edge points 
                    zz = system.f_eval_loc(f_loc, node, x=np.array([xx,yy]).T, \
                                           derivatives=derivatives)
                    if initialize_min_max:
                        z_min = zz.min()
                        z_max = zz.max()
                        initialize_min_max = False
                    else:
                        z_max = max(zz.max(),z_max) 
                        z_min = min(zz.min(),z_min)
             
                    for i in range(ne-1):
                        lines.append([(xx[i],yy[i],zz[i]),(xx[i+1],yy[i+1],zz[i+1])])
                node_count += 1   
            ax.add_collection(Line3DCollection(lines, colors='k', linewidth=0.5))
        
        x0,x1,y0,y1 = mesh.box()
        hx = x1 - x0
        hy = y1 - y0
        hz = z_max - z_min
        spc = 0.1
        #print(z_min,z_max)
        ax.set_xlim(x0-spc*hx, x1+spc*hx)
        ax.set_ylim(y0-spc*hy, y1+spc*hy)
        ax.set_zlim(z_min-spc*hz, z_max+spc*hz)
                
        return ax    
            