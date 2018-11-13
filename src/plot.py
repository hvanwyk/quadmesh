'''
Created on Feb 8, 2017

@author: hans-werner
'''

import matplotlib.pyplot as plt
from mesh import QuadCell
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from fem import Assembler, Function


class Plot(object):
    """
    Plots related to finite element mesh and functions
    """
    
    def __init__(self, quickview=True):
        """
        Constructor
        """
        self.__quickview = quickview
        
            
    def mesh(self, mesh, axis=None, dofhandler=None, show_axis=False, 
             color_marked=None, vertex_numbers=False, edge_numbers=False, 
             cell_numbers=False, dofs=False, mesh_flag=None):
        """
        Plot computational mesh
        
        Inputs: 
            
            mesh: Mesh, computational mesh
            
            *ax: current axes
            
            *dofhandler: DofHandler associated with mesh
            
            *show_axis: boolean, set axis on or off
            
            *color_marked: list of flags for cells that must be colored
            
            *vertex/edge/cell_numbers: bool, display vertex/edge/cell numbers.
            
            *dofs: boolean, display degrees of freedom
            
            *mesh_flag: boolean, plot only cells with the given flag
                    
        
        Outputs:
        
            ax: axis, 
            
        """
        if self.__quickview:
            fig = plt.figure()
            axis = fig.add_subplot(111)
        else:
            assert axis is not None, 'Axis not specified.'
        #
        # Two dimensional mesh
        #
        if mesh.dim() == 2:
            #
            # Set bounding box
            # 
            x0, x1, y0, y1 = mesh.bounding_box()    
            hx = x1 - x0
            hy = y1 - y0
            axis.set_xlim(x0-0.1*hx, x1+0.1*hx)
            axis.set_ylim(y0-0.1*hy, y1+0.1*hy) 
            #
            # Plot background rectangle
            # 
            points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            rect = plt.Polygon(points, fc='darkgrey', edgecolor='k', alpha=0.1)
            axis.add_patch(rect)
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
                for v in vertices:
                    axis.plot(*v, '.k')
                        
                    
        elif mesh.dim()==1:
            #
            # Bounding box
            # 
            x0, x1 = mesh.bounding_box()
            l = x1 - x0
            axis.set_xlim([x0-0.1*l, x1+0.1*l])
            axis.set_ylim([-0.1,0.1])
            axis.get_yaxis().set_ticks([])
            for interval in mesh.cells.get_leaves(flag=mesh_flag):
                a, = interval.base().coordinates()
                b, = interval.head().coordinates()
                axis.plot([a,b], [0,0], '-|k')
            
                                
        #
        # Degrees of freedom
        # 
        if dofs:
            self.dofs(axis, dofhandler)
        
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
                plt.pause(20)
                plt.close()
        else:  
            return axis
        

         
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
        
    def element(self, element, ax=None):
        """
        Plot reference element
        """            
        #
        # Check axis
        # 
        
        
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
    def dofs(self, ax, dofhandler, subforest_flag=None):
        """
        Plot a mesh's dofs
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
            x = cell.reference_map(x_ref, jacobian=False)
            cell_dofs = dofhandler.get_global_dofs(cell)
            if cell_dofs is not None:
                for i in range(n_dofs):
                    if cell_dofs[i] is not None:
                        if mesh.dim()==1:
                            xx, yy = x[i], 0
                        elif mesh.dim()==2:
                            xx, yy = x[i,0], x[i,1]
                        
                        ax.text(xx,yy,\
                                str(cell_dofs[i]), size = '12',\
                                horizontalalignment='center',
                                verticalalignment='center',
                                backgroundcolor='w')   
        
                    

    def contour(self, f, mesh=None, derivative=(0,),
                colorbar=True, resolution=(100,100), 
                axis=None):
        """
        Returns a contour plot of a function f
        
        
        Inputs:
        
            ax: Axis, current axes
                        
            f: Function, function to be plotted
            
            mesh: Mesh, computational mesh
                        
            *derivative [(0,)]: int, tuple specifying the function's derivative
            
            *colorbar [True]: bool, add a colorbar?
            
            *resolution [(100,100)]: int, tuple resolution of contour plot.
                        
            
        Outputs: 
        
            ax
            
            fig
                    
        """
        #
        # Check function
        # 
        assert isinstance(f, Function), 'Can only plot "Function" objects.'
        
        #
        # Initialize Axes
        # 
        if self.__quickview:
            fig = plt.figure()
            axis = fig.gca()
        else:
            assert axis is not None, 'Axis not specified.'
            assert axis.name=="2d", 'Axis required to be 2D.'
            
        #
        # Initialize grid
        # 
        x0,x1,y0,y1 = mesh.bounding_box()
        nx, ny = resolution 
        x,y = np.meshgrid(np.linspace(x0,x1,nx),np.linspace(y0,y1,ny))
        xy = np.array([x.ravel(), y.ravel()]).T
        ff = f.eval(xy, derivative=derivative)
        z  = ff.reshape(x.shape)
        
        cm = axis.contourf(x,y,z,30)
        
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
                system = Assembler(mesh,element)
                z = system.f_eval(f, xy, derivatives=derivative)
                cm = ax.contourf(x,y,z.reshape(ny,nx),200, cmap='viridis')
                
        if colorbar:
            fig.colorbar(cm, ax=ax, format='%g')
            
        return fig, ax, cm
    
    
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
    
    def wire(self, f, mesh=None, resolution=10, axis=None): 
        """
        Wire plot of 2D function
        """  
        #
        # Check function
        # 
        assert isinstance(f, Function), 'Can only plot "Function" objects.'
        
        #
        # Make sure there's a mesh
        # 
        if f.fn_type() in ['explicit', 'constant']:
            assert mesh is not None, \
            'For "explicit" or "constant" functions, mesh must be given.'
        else:
            mesh = f.dofhandler.mesh    
        
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
        for cell in mesh.cells.get_leaves(subforest_flag=f.flag()):
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
                zz = f.eval(xy, cell=cell)
                
                x.extend(list(xx))
                y.extend(list(yy))
                z.extend(list(zz))
            
            verts = [list(zip(x,y,z))]
            poly = Poly3DCollection(verts, edgecolor="black", linewidth=0.5,
                                    facecolor="white")
            axis.add_collection3d(poly, zs='z')
        #    
        # Plot immediately and/or save
        # 
        if self.__quickview:
            if False:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(10)
                plt.close()
        else:  
            return axis
            
        
                
    def line(self, f, mesh=None, resolution=10, axis=None):
        """
        Plot graph of 1D function
        """
        #
        # Check function properties
        # 
        assert isinstance(f, Function), 'Can only plot "Function" objects.'
        
        #
        # Ensure there's a mesh
        # 
        if f.fn_type() in ['explicit', 'constant']:
            assert mesh is not None, \
            'For "explicit" or "constant" functions, mesh must be specified.'  
        else:
            mesh = f.dofhandler.mesh
            
        assert mesh.dim()==1, 'Line plots are for 1D functions'
        
        #
        # Check axis
        #  
        if self.__quickview:
            fig = plt.figure()
            axis = fig.gca()
        else:
            assert axis is not None, 'Axis not specified.'
            assert axis.name=="2d", 'Axis required to be 2D.'
             
        #
        # Evaluate function
        # 
        x = []
        fx = []
        for interval in mesh.cells.get_leaves(subforest_flag=f.flag()):
            #
            # Form grid on local interval
            # 
            x0, = interval.get_vertex(0).coordinates()
            x1, = interval.get_vertex(1).coordinates()
            xx = np.linspace(x0, x1, resolution)
            
            #
            # Evaluate function on local interval
            # 
            ff = f.eval(xx, cell=interval)
                        
            #
            # Add x and y-values to list
            #  
            x.extend(xx.tolist())
            fx.extend(ff.tolist())
            
            #
            # For discontinous elements, add a np.nan value
            # 
            if f.fn_type()=='nodal':
                if f.dofhandler.element.torn_element():
                    x.append(x1)
                    fx.append(np.nan)
        
        #
        # Plot graph
        # 
        x = np.array(x)
        fx = np.array(fx)
        axis.plot(x, fx, linewidth=1.5)
        
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
                plt.pause(2)
                plt.close()
        else:  
            return axis
        
    