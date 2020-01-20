'''
Created on Feb 8, 2017

@author: hans-werner
'''

import matplotlib.pyplot as plt
from mesh import QuadCell, Cell, Interval
from function import Map, Constant, Nodal, Explicit
from matplotlib import colors as clrs
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D 
import time
import numpy as np
from assembler import Assembler
from function import Function


class Plot(object):
    """
    Plots related to finite element mesh and functions
    """
    
    def __init__(self, time=5, quickview=True):
        """
        Constructor
        """
        self.__quickview = quickview
        self.__time= time
        self.__color_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                              '#bcbd22', '#17becf']
    
    def check_axis(self, axis, plot_dim=2):
        """
        Check whether axis is specified or else if quickview is turned on
        
        Input:
        
            axis: Axes (or None), current axes
        """
        if axis is None:
            #
            # No axis specified: Quickview mode
            # 
            assert self.__quickview, 'No axis specified.'
            fig = plt.figure()
            if plot_dim==2:
                #
                # 2D plot
                # 
                axis = fig.add_subplot(111)
            elif plot_dim==3:
                #
                # 3D plot
                # 
                axis = fig.add_subplot(111, projection="3d")
        #
        # Return axis
        #
        return axis
    
    
    def set_bounding_box(self, axis, mesh):
        """
        Determine the axis limits
        
        Inputs:
        
            axis: Axes, current axes
            
            mesh: Mesh, defining the computational domain
        """
        if mesh.dim()==1:
            #
            # 1D Mesh
            #
            x0, x1 = mesh.bounding_box()
            l = x1 - x0
            axis.set_xlim([x0-0.1*l, x1+0.1*l])
            axis.set_ylim([-0.1,0.1])
        elif mesh.dim()==2:
            #
            # 2D Mesh
            # 
            x0, x1, y0, y1 = mesh.bounding_box()    
            hx = x1 - x0
            hy = y1 - y0
            axis.set_xlim(x0-0.1*hx, x1+0.1*hx)
            axis.set_ylim(y0-0.1*hy, y1+0.1*hy)
        else:
            raise Exception('Only 1D and 2D meshes supported.')
          
        return axis
    
    
    def mesh(self, mesh, axis=None, dofhandler=None, show_axis=False, 
             regions=None, vertex_numbers=False, 
             edge_numbers=False, cell_numbers=False, dofs=False, doflabels=False,
             subforest_flag=None):
        """
        Plot computational mesh
        
        Inputs: 
            
            mesh: Mesh, computational mesh
            
            *ax: current axes
            
            *dofhandler: DofHandler associated with mesh
            
            *show_axis: boolean, set axis on or off
            
            *regions: list of tuples consisting of (flag, entity_type), where 
                flag specifies the region to be plotted, and entity_type
                specifies whether the entity is a 'vertex', 'half_edge', or
                'cell'   
            
            *vertex/edge/cell_numbers: bool, display vertex/edge/cell numbers.
            
            *dofs: boolean, display degrees of freedom
            
            *mesh_flag: boolean, plot only cells with the given flag
                    
        
        Outputs:
        
            ax: axis, 
            
        """
        #
        # Check axis
        # 
        axis = self.check_axis(axis)
        
        #
        # Set axis limits
        # 
        axis = self.set_bounding_box(axis, mesh)
        
        #
        # Format background
        #
        if mesh.dim()==1:
            #
            # 1D Mesh
            # 
            axis.get_yaxis().set_ticks([])
        if mesh.dim() == 2:     
            #
            # 2D Mesh: Plot background rectangle
            # 
            x0, x1, y0, y1 = mesh.bounding_box()
            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            rect = plt.Polygon(points, fc='darkgrey', edgecolor='k', alpha=0.1)
            axis.add_patch(rect)
        
        #
        # Plot mesh cells
        # 
        cells = mesh.get_region(entity_type='cell', subforest_flag=subforest_flag)
        axis = self.cells(axis, cells)
        
        #
        # Plot regions
        # 
        if regions is not None:
            # 
            # Plot additional regions
            # 
            assert type(regions) is list, 'Regions should be passed as list'
            colors = self.__color_cycle[:len(regions)]
            
            for region, color in zip(regions,colors):
                flag, entity_type = region
                                
                #
                # Cycle over regions
                # 
                if entity_type=='vertex':
                    #
                    # Plot vertices
                    # 
                    axis = self.vertices(axis, mesh, 
                                         subforest_flag=subforest_flag, 
                                         vertex_flag=flag, color=color)
                    
                elif entity_type=='half_edge':
                    #
                    # Plot half-edges
                    # 
                    axis = self.half_edges(axis, mesh, 
                                           subforest_flag=subforest_flag,
                                           half_edge_flag=flag, color=color)
                    
                elif entity_type=='edge':
                    #
                    # Plot edges
                    # 
                    axis = self.edges(axis, mesh, subforest_flag=subforest_flag,
                                      edge_flag=flag, color=color)
                    
                elif entity_type=='cell':
                    #
                    # Plot cells
                    # 
                    axis = self.cells(axis, mesh, 
                                      subforest_flag=subforest_flag,
                                      cell_flag=flag, color=color, 
                                      alpha=0.3)
        #
        # Plot Dofs  
        # 
        if dofs:
            assert dofhandler is not None, 'Plotting Dofs requires dofhandler.'
            
            axis = self.dofs(axis, dofhandler)
    
        
        if not show_axis:
            axis.axis('off')
        
        #    
        # Plot immediately and/or save
        # 
        if self.__quickview:
            if False:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(self.__time)
                plt.close()
        else:  
            return axis
             
        """
        elif mesh.dim()==1
            #
            # Plot Cells
            # 
            cells = mesh.cells.get_leaves(subforest_flag=mesh_flag)    
            for cell in cells:
                #
                # Plot cells
                # 
                vertices = [v.coordinates() for v in cell.get_vertices()]
                rect = plt.Polygon(vertices, fc='w', edgecolor='k')
                axis.add_patch(rect)
                #
                # Plot half-edges
                #
                if False: 
                    for he in cell.get_half_edges():
                        axis.annotate(s='', xy=he.head().coordinates(), \
                                    xytext=he.base().coordinates(),\
                                    arrowprops=dict(arrowstyle="->",\
                                                    connectionstyle="arc3" )) 
                #
                # Plot vertices
                # 
                if False:
                    for v in vertices:
                        axis.plot(*v, '.k')
                        
            
        elif mesh.dim()==1:
            
            
            for interval in mesh.cells.get_leaves(flag=mesh_flag):
                a, = interval.base().coordinates()
                b, = interval.head().coordinates()
                axis.plot([a,b], [0,0], '-|k')
            
                                
        #
        # Degrees of freedom
        # 
        if dofs:
            self.dofs(axis, dofhandler)
        
        

        
        
        """
         
        """
        
        #
        # Plot Vertex Numbers
        #    
        if vertex_numbers:
            vertices = mesh.quadvertices(flag=mesh_flag, nested=nested, \
                                         coordinate_array=False)
            v_count = 0
            for v in vertices:
                x,y = v.coordinates()
                ax.text(x,y,str(v_count),size='7',
                        horizontalalignment='center',
                        verticalalignment='center',
                        backgroundcolor='w')
                v_count += 1
                
        #
        # Plot Edge Numbers
        #
        if edge_numbers:
            edges = mesh.iter_quadedges(flag=mesh_flag, nested=nested)
            e_count = 0
            for e in edges:
                if not(e.is_marked()):
                    v1, v2 = e.vertices()
                    x0,y0 = v1.coordinates()
                    x1,y1 = v2.coordinates()
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
        """
        
    def element(self, element, axis=None):
        """
        Plot reference element
        """            
        #
        # Check axis
        # 
        self.check_axis(axis)
        
        
        
        
    def dofs(self, axis, dofhandler, doflabels=False, subforest_flag=None):
        """
        Plot a mesh's dofs
        
        Inputs:
        
            axis: Axes, 
            
            dofhandler: DofHandler object used to store 
        """
        assert dofhandler is not None, \
            'Require dofhandler information to plot dofs'
            
        element = dofhandler.element
        mesh = dofhandler.mesh
        x_ref = element.reference_nodes()
        n_dofs = element.n_dofs()
        for cell in mesh.cells.get_leaves(subforest_flag=subforest_flag):
            if mesh.dim()==2:
                assert isinstance(cell, QuadCell), 'Can only map QuadCells'
            x = cell.reference_map(x_ref)
            cell_dofs = dofhandler.get_global_dofs(cell)
            if cell_dofs is not None:
                for i in range(n_dofs):
                    if cell_dofs[i] is not None:
                        if mesh.dim()==1:
                            xx, yy = x[i], 0
                        elif mesh.dim()==2:
                            xx, yy = x[i,0], x[i,1]
                        
                        if doflabels:
                            #
                            # Print the Doflabels
                            #
                            axis.text(xx,yy,\
                                    str(cell_dofs[i]), size = '12',\
                                    horizontalalignment='center',
                                    verticalalignment='center',
                                    backgroundcolor='w')
                        else:
                            #
                            #  Plot dof locations     
                            # 
                            axis.plot(xx,yy,'.k')
        return axis
    
    
    def vertices(self, axis, mesh, subforest_flag=None, 
                 vertex_flag=None, color='k'):
        """
        Plot (selected) vertices in a mesh
        
        Inputs:
        
            axis: Axes, current axes
            
            mesh: Mesh, whose vertices are being plotted
            
            subforest_flag: str/int/tuple specifying submesh
            
            cell_flag: str/int/tuple specifying vertices 
            
            color: str, vertex color
        """
        #
        # Iterate over region vertices
        # 
        for vertex in mesh.get_region(flag=vertex_flag, entity_type='vertex', 
                                      subforest_flag=subforest_flag):
            #
            # Determine the dimension
            # 
            if mesh.dim()==1:
                #
                # 1D Mesh
                # 
                x, = vertex.coordinates()
                axis.plot(x,0, '.', color=color)
            elif mesh.dim()==2:
                #
                # 2D Mesh
                # 
                x,y = vertex.coordinates()
                axis.plot(x,y,'.', color=color)
        return axis
    
    
    def half_edges(self, axis, mesh, subforest_flag=None, 
                   half_edge_flag=None, color='k'):
        """
        Plot (selected) half-edges in a mesh. Half-edges are drawn with arrows.
        (Also see self.edges)
        
        Inputs:
        
            axis: Axes, current axes
            
            mesh: Mesh, whose half-edges are to be plotted
            
            subforest_flag: str/int/tuple, submesh flag
            
            half_edge_flag: str/int/tuple, flag specifying subset of half-edges
            
            color: str, color of half-edges
        """ 
        assert mesh.dim()==2, 'Can only plot half-edges in a 2D mesh.'
        for he in mesh.get_region(flag=half_edge_flag, entity_type='half_edge',
                                  subforest_flag=subforest_flag):
            #
            # Draw arrow
            # 
            axis.annotate(s='', xy=he.head().coordinates(), \
                          xytext=he.base().coordinates(),\
                          color = color, \
                          arrowprops=dict(arrowstyle="->", \
                                          connectionstyle="arc3" ))
        return axis


    def edges(self, axis, mesh, subforest_flag=None, 
               edge_flag=None, color='k'):
        """
        Plot (selected) edges in a mesh. 
        
        Note: Edges are simply lines (See also self.half_edges)
        
        Inputs:
        
            axis: Axes, current axes
            
            mesh: Mesh, whose half-edges are to be plotted
            
            subforest_flag: str/int/tuple, submesh flag
            
            half_edge_flag: str/int/tuple, flag specifying subset of half-edges
            
            color: str, color of half-edges
            
        Output:
        
            axis: Axes, current axis
        
        """
        assert mesh.dim()==2, 'Can only plot half-edges in a 2D mesh.'
        for he in mesh.get_region(flag=edge_flag, entity_type='half_edge',
                                  subforest_flag=subforest_flag):
            #
            # Draw line
            #
            x0, y0 = he.base().coordinates()
            x1, y1 = he.head().coordinates()
             
            axis.plot([x0,x1],[y0,y1], linewidth=2, color=color)
            
        return axis

    
           
    def cells(self, axis, cells, color='w', alpha=1):
        """
        Plot (selected) cells in a mesh
        
        Inputs:
        
            axis: Axes, current axes
            
            cells: cells are to be plotted
            
            subforest_flag: str/int/tuple, submesh flag
            
            cell_flag: str/int/tuple, subregion flag
            
            color: str, cell color
            
        
        Outputs:
        
            axis: Axes, current axes
        """
        #
        # Iterate over cells
        # 
        for cell in cells: 
            if isinstance(cell, Interval):
                #
                # 1D Mesh
                # 
                a, = cell.base().coordinates()
                b, = cell.head().coordinates()
                axis.plot([a,b], [0,0], '-|k')
            elif isinstance(cell, Cell):
                #
                # 2D Mesh 
                #
                # 
                vertices = [v.coordinates() for v in cell.get_vertices()]
                poly = plt.Polygon(vertices, 
                                   fc=clrs.to_rgba(color,alpha=alpha),
                                   edgecolor=(0,0,0,0.4))
                axis.add_patch(poly) 
            else: 
                raise Exception('Only 1D and 2D meshes supported.')
        return axis
    
    
    def contour(self, f, n_sample=0, colorbar=True, derivative=(0,), 
                resolution=(500,500), axis=None, mesh=None):
        """
        Returns a contour plot of a function f
        
        
        Inputs:
        
            ax: Axis, current axes
                        
            f: Function, function to be plotted
                                    
            *derivative [(0,)]: int, tuple specifying the function's derivative
            
            *colorbar [True]: bool, add a colorbar?
            
            *resolution [(100,100)]: int, tuple resolution of contour plot.
                        
            
        Outputs: 
        
            ax
            
            fig
                    
        """
        if mesh is None:
            mesh = f.mesh()
        
        #
        # Check function
        # 
        assert isinstance(f, Map), 'Can only plot "Map" objects.'
        
        #
        # Check axis
        # 
        axis = self.check_axis(axis)
        
        #
        # Set axis limits
        #     
        axis = self.set_bounding_box(axis, mesh)
        
        #
        # Initialize grid
        # 
        x0,x1,y0,y1 = mesh.bounding_box()
        nx, ny = resolution 
        x,y = np.meshgrid(np.linspace(x0,x1,nx),np.linspace(y0,y1,ny))
        xy = np.array([x.ravel(), y.ravel()]).T
        if isinstance(f, Explicit):
            ff = f.eval(xy)
        else:
            ff = f.eval(xy, derivative=derivative)
        z  = ff[:,n_sample].reshape(x.shape)
        
        cm = axis.contourf(x,y,z,100)
        
        if colorbar:
            plt.colorbar(cm, ax=axis, format='%g')
            
        #    
        # Plot immediately and/or save
        # 
        if self.__quickview:
            if False:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(self.__time)
                plt.close()
        else:  
            return axis
    
    
    def surface(self, f, axis=None, mesh=None, derivative=(0,), 
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
        #
        # Check if input is a Function object
        # 
        assert isinstance(f, Function), 'Can only plot Function objects.'
        
        if mesh is None:
            if f.mesh is not None:
                mesh = f.mesh
            else:
                mesh_error = 'Mesh must be specified, either explicitly, '+\
                    'or as part of the Function.'
                raise Exception(mesh_error)
            
        x0,x1,y0,y1 = mesh.bounding_box()        
        system = Assembler()
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
            zz = f.eval(xy, derivative=derivative)
            z_min, z_max = zz.min(), zz.max()
            
            if grid:
                alpha = 0.5
            else:
                alpha = 1
            axis.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                              linewidth=1, antialiased=True, alpha=alpha)
            
        self.exit(axis=axis)
        
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
                    assert derivative==(0,),\
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
                                           derivatives=derivative)
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
            axis.add_collection(Line3DCollection(lines, colors='k', linewidth=0.5))
        
        x0,x1,y0,y1 = mesh.box()
        hx = x1 - x0
        hy = y1 - y0
        hz = z_max - z_min
        spc = 0.1
        #print(z_min,z_max)
        axis.set_xlim(x0-spc*hx, x1+spc*hx)
        axis.set_ylim(y0-spc*hy, y1+spc*hy)
        axis.set_zlim(z_min-spc*hz, z_max+spc*hz)
                
        return axis 
    
    def wire(self, f, n_sample=0, mesh=None, resolution=10, axis=None): 
        """
        Wire plot of 2D function
        """  
        #
        # Check function
        # 
        assert isinstance(f, Map), 'Can only plot "Map" objects.'
        
        #
        # Make sure there's a mesh
        # 
        if isinstance(f, Explicit) or isinstance(f, Constant):
            assert mesh is not None, \
            'For "explicit" or "constant" functions, mesh must be given.'
        else:
            mesh = f.mesh()    
        
        #
        # Check axis
        # 
        if self.__quickview:
            fig = plt.figure()
            axis = fig.gca(projection="3d")
        else:
            assert axis is not None, 'Axis not specified.'
            assert axis.name=="3d", 'Axis required to be 3D.'
    
        #
        # Set axis bounding box 
        #                
        x0, x1, y0, y1 = mesh.bounding_box()    
        hx = x1 - x0
        hy = y1 - y0
        axis.set_xlim(x0-0.1*hx, x1+0.1*hx)
        axis.set_ylim(y0-0.1*hy, y1+0.1*hy)
         
        #
        # Evaluate function
        # 
        z0, z1 = None, None
        for cell in mesh.cells.get_leaves(subforest_flag=f.subforest_flag()):
            x = []
            y = []
            z = []
            for he in cell.get_half_edges():
                x0, y0 = he.base().coordinates()
                x1, y1 = he.head().coordinates()
                t = np.linspace(0,1,resolution)
                
                xx = x0 + t*(x1-x0)
                yy = y0 + t*(y1-y0)
                xy  = np.array([xx,yy]).T
                # TODO: How do we incorporate cell information into the function? 
                #zz = f.eval(xy, cell=cell)
                zz = f.eval(xy)
                x.extend(list(xx))
                y.extend(list(yy))
                z.extend(list(zz))
            
            verts = [list(zip(x,y,z))]
            poly = Poly3DCollection(verts, edgecolor="black", linewidth=0.5,
                                    facecolor="white")
            axis.add_collection3d(poly, zs='z')
            
            # 
            # Update minimum function value
            # 
            if z0 is None:
                z0 = min(z)
            else:
                z0 = min(z0, min(z))
                
            #
            # Update maximum function value
            # 
            if z1 is None:
                z1 = max(z)
            else:
                z1 = max(z1, max(z))
            
        #
        # Set z limits
        #         
        hz = z1 - z0
        axis.set_zlim(z0-0.1*hz, z1+0.1*hz)
        
        #    
        # Plot immediately and/or save
        # 
        if self.__quickview:
            if False:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(self.__time)
                plt.close()
        else:  
            return axis
            
        
                
    def line(self, f, mesh=None, resolution=10, axis=None, i_sample=0, plot_kwargs={}):
        """
        Plot graph of 1D function
        """
        #
        # Check function properties
        # 
        assert isinstance(f, Map), 'Can only plot "Map" objects.'
        
        #
        # Ensure there's a mesh
        #
        if isinstance(f, Explicit) or isinstance(f, Constant):
            assert mesh is not None, \
            'For "explicit" or "constant" functions, mesh must be specified.'  
        else:
            mesh = f.dofhandler().mesh
            
        assert mesh.dim()==1, 'Line plots are for 1D functions'
        
        #
        # Check axis
        #  
        if self.__quickview:
            fig = plt.figure()
            axis = fig.gca()
        else:
            assert axis is not None, 'Axis not specified.'
             
        #
        # Evaluate function
        # 
        x = []
        fx = []
        for interval in mesh.cells.get_leaves(subforest_flag=f.subforest_flag()):
            #
            # Form grid on local interval
            # 
            x0, = interval.get_vertex(0).coordinates()
            x1, = interval.get_vertex(1).coordinates()
            xx = np.linspace(x0, x1, resolution)
            
            #
            # Evaluate function on local interval
            # 
            ff = f.eval(x=xx, cell=interval)
            #
            # Add x and y-values to list
            #  
            x.extend(xx.tolist())
            fx.extend(ff[:,i_sample].tolist())
            
            #
            # For discontinous elements, add a np.nan value
            # 
            if isinstance(f, Nodal):
                if f.dofhandler().element.torn_element():
                    x.append(x1)
                    fx.append(np.nan)
        
        #
        # Plot graph
        # 
        x = np.array(x)
        fx = np.array(fx)
        axis.plot(x, fx, **plot_kwargs)
        
        #
        # Axis limits
        # 
        spc = 0.1

        x0, x1 = mesh.bounding_box()
        hx = x1 - x0
        axis.set_xlim(x0-spc*hx, x1+spc*hx)        
        
        y0, y1 = np.nanmin(fx), np.nanmax(fx)        
        hy = y1-y0
        axis.set_ylim(y0-spc*hy, y1+spc*hy)
        
        #    
        # Plot immediately and/or save
        # 
        if self.__quickview:
            if False:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(self.__time)
                plt.close()
        else:  
            return axis
        
    