import unittest
from mesh import Mesh, convert_to_array
from mesh import Mesh1D
from mesh import Mesh2D
from mesh import QuadMesh
from mesh import DCEL

from fem import QuadFE
from fem import Function
from fem import DofHandler

from plot import Plot

import numpy as np
import matplotlib.pyplot as plt

class TestFunction(unittest.TestCase):
    """
    Test Function class
    
    TODO:
    
    a. Define function on a submesh, refine it
    """
    def test_constructor_and_eval(self):
        """
        Test Constructor
        """     
        mesh = QuadMesh()
        mesh.cells.record(0)
        mesh.cells.refine(new_label=1)
        mesh.cells.refine(new_label=2)
        
        
        
        f = lambda x,y: np.sin(np.pi*x)*np.cos(2*np.pi*y) + np.cos(np.pi*y)
        
        fig = plt.figure()
        
        #
        # Mesh points at which to plot function
        #
        vtcs = mesh.cells.get_child(0).get_vertices() 
        vtcs = convert_to_array(vtcs)
        x0, y0 = vtcs[0,:]
        x1, y1 = vtcs[2,:] 
        nx, ny = 30,30
        x,y = np.linspace(x0,x1,nx), np.linspace(y0,y1,ny)
        xx, yy = np.meshgrid(x,y)
        xy = np.array([xx.ravel(),yy.ravel()]).transpose()
        
        #
        # Test interpolant and eval
        #
        fn = Function(f,'explicit')
        ax = fig.add_subplot(3,4,1, projection='3d')
        #for node in mesh.root_node().get_leaves():
        #    cell = node.cell()
        #    print(cell.contains_point(xy).shape)
        zz = fn.eval(xy)  
        ax.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                        linewidth=0, antialiased=True)
        ax.set_title('f')
        #
        # Interpolate function: continous elements
        # 
        continuous_etype = ['Q1','Q2','Q3']
        for i in range(3):
            etype = continuous_etype[i]
            element = QuadFE(2,etype)
            fn_interp = fn.interpolant(mesh, element)
            
            ax = fig.add_subplot(3,4,2+i, projection='3d')
            zz = fn_interp.eval(xy)
            ax.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                            linewidth=0, antialiased=True)
            ax.set_title(etype)
        #
        # Interpolate function: discontinuous elements
        #
        '''
        Note: The plots in row 2 should be the same as those in row 1, with
            the exception of DQ0.
        '''
        discontinuous_etype = ['DQ0','DQ1','DQ2','DQ3']
        for i in range(4):
            etype = discontinuous_etype[i]
            element = QuadFE(2,etype)
            
            dh = DofHandler(mesh, element)
            dh.distribute_dofs()
            dh.set_dof_vertices()
            x = dh.get_dof_vertices()
        
            fn_interp = fn.interpolant(mesh, element)
            
            ax = fig.add_subplot(3,4,5+i, projection='3d')
            zz = fn_interp.eval(xy)
            ax.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                            linewidth=0, antialiased=True)
            ax.set_title(etype)
        
        
        #
        # Differentiate the function with respect to y
        # 
        etype_list = ['DQ0','Q1','Q2','Q3']
        for i in range(4):
            etype = etype_list[i]
            element = QuadFE(2,etype)
            fn_interp = fn.interpolant(mesh, element)
            
            ax = fig.add_subplot(3,4,9+i, projection='3d')
            df_dx = fn_interp.derivative((1,1))
            zz = df_dx.eval(xy)
            ax.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                            linewidth=0, antialiased=True)
            ax.set_title(etype)
        
        plt.tight_layout(pad=1, w_pad=2, h_pad=2)
            
        #
        # Function defined on meshes of various resolution, evaluated on 
        # the same cell.
        # 
        mesh = Mesh1D(resolution=(1,))
        mesh.cells.record(1)
        mesh.cells.refine(subforest_flag=1, new_label=2)
        
        element = QuadFE(1,'Q1')
        dofhandler = DofHandler(mesh, element)
        
        # Function on mesh coarser than cell
        f1 = Function(lambda x:x**2, 'nodal', \
                      dofhandler=dofhandler, \
                      subforest_flag=1)
        
        # Function on mesh containing cell
        f2 = Function(lambda x:x**2, 'nodal',\
                      dofhandler=dofhandler,\
                      subforest_flag=2)
        
        mesh.cells.refine(subforest_flag=2, new_label=3)
        
        # Function on mesh finer than cell
        f3 = Function(lambda x:x**2, 'nodal',\
                      dofhandler=dofhandler,\
                      subforest_flag=3)
        
        # Evaluate on [0,0.5] at x=1/3 
        cell = mesh.cells.get_child(0).get_child(0)
        x = [(1/3,)]
        
        # Check answers
        self.assertEqual(f1.eval(x=x, cell=cell), np.array([1/3]))
        self.assertEqual(f2.eval(x=x, cell=cell), np.array([1/6]))
        self.assertTrue(np.allclose(f3.eval(x=x, cell=cell), np.array([1/8])))
        
        
        """
        fig = plt.figure()
        plot = Plot()
        element = QuadFE(2,'DQ0')
        fn.interpolant(mesh, element)
        ax = fig.add_subplot(1,3,1)
        ax = plot.mesh(mesh, ax, mesh_flag=0, color_marked=[0])
        ax = fig.add_subplot(1,3,2)
        ax = plot.mesh(mesh, ax, mesh_flag=1, color_marked=[1])
        ax = fig.add_subplot(1,3,3)
        ax = plot.mesh(mesh, ax, mesh_flag=2, color_marked=[2])
        plt.show()
        
        fig, ax = plt.subplots(2,4)
        
        count = 0
        for etype in etype_list:
            print(etype)            
            element = QuadFE(2,etype)
            dh = DofHandler(mesh,element)
            dh.distribute_dofs(nested=True)
            x = dh.dof_vertices(flag=1)
            fv = f(x[:,0],x[:,1])
                 
            fn = Function(fv, mesh=mesh, element=element, flag=1)
            
            # Plot 
            i,j = count//4, count%4
            ax[i,j] = plot.contour(ax[i,j],fig,fn,mesh,element)
            count += 1
        plt.show()
        
        
        n_flags = 3
        n_element_types = 3
        for j in range(n_flags):
            for i in range(n_element_types):
                ax[i,j] = plot.mesh(ax[i,j], mesh, cell_numbers=True, node_flag=j)
                etype = etype_list[i]
                element = QuadFE(2,etype)
                dh = DofHandler(mesh, element)
                dh.distribute_dofs(nested=True)
                x = dh.dof_vertices(flag=j)
                ax[i,j].plot(x[:,0],x[:,1],'.b')
        plt.show()
        """
                
        # Explicit function
        
        # Nodal continuous function
        
        # Nodal discontinuous function
    
    
    def test_nodal_on_mesh(self):
        """
        Test whether a function is correctly identified as a nodal and 
        compatible with given (sub)mesh
        """
        # Define mesh
        mesh = QuadMesh(resolution=(1,1))
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh, element)
        
        # Nodal Function compatible with mesh
        f1 = Function(lambda x,y: x*y, 'nodal', dofhandler=dofhandler) 
        self.assertTrue(f1.mesh_compatible(mesh))
        
        # Explicit function: incompatible with mesh
        f2 = Function(lambda x,y: x*y, 'explicit')
        self.assertFalse(f2.mesh_compatible(mesh))
        
        # Hierarchical mesh
        mesh.cells.record(flag=1)
        mesh.cells.refine(new_label=2)
        dofhandler.distribute_dofs()
        
        # Function defined over coarse mesh
        f3 = Function(lambda x,y:x*y, 'nodal', dofhandler=dofhandler, \
                      subforest_flag=1)
        # Function compatible with coarse mesh, but not with fine one
        self.assertTrue(f3.mesh_compatible(mesh, subforest_flag=1))
        self.assertFalse(f3.mesh_compatible(mesh, subforest_flag=2))
        
        # Interpolate function on finer mesh now compatible
        element_2 = QuadFE(2,'Q2')
        f4 = f3.interpolant(mesh, element_2, subforest_flag=2)
        self.assertTrue(f4.mesh_compatible(mesh, subforest_flag=2))
        
        # Define another mesh -> incompatible
        mesh2 = QuadMesh(resolution=(1,1))
        self.assertFalse(f1.mesh_compatible(mesh2))
        
        
    def test_assign(self):
        """
        Test the assign method
        """
        #
        # New nodal function
        #  
        # Define Mesh and elements
        mesh = QuadMesh(resolution=(2,2))
        element = QuadFE(2,'DQ0')
        
        # Initialize
        vf = np.empty(4,)
        dh = DofHandler(mesh, element)
        dh.distribute_dofs()
        dofs = dh.get_global_dofs()
        x_mpt = convert_to_array(dh.get_dof_vertices(dofs),2)
        f = Function(vf, 'nodal', mesh=mesh, element=element)
        
        #
        # Assign new value to function (vector)
        #
        f_det = np.arange(1,5)
        f.assign(f_det)
        
        self.assertTrue(np.allclose(f.eval(x_mpt),f_det))
        
        #
        # Assign random sample to function
        #
        f_rand = np.random.rand(4,10)
        f.assign(f_rand) 
        
        self.assertTrue(np.allclose(f.eval(x_mpt),f_rand),\
                        'Function value assignment incorrect.')
        
        
        #
        # Now assign in specific position
        #
        n_samples = f.n_samples()
        f.assign(np.arange(1,5), pos=0)
        
        f_eval_0 = f.eval(x_mpt, samples=np.array([0]))
        f_eval_1ton = f.eval(x_mpt, samples=np.arange(1,n_samples))
        self.assertTrue(np.allclose(f_eval_0.ravel(), f_det.T), 
                        'Function value assignment incorrect.')
        self.assertTrue(np.allclose(f_eval_1ton, f_rand[:,1:]), \
                        'Function value assignment incorrect.')
                                    

        f.assign(np.arange(1,5),pos=0)
        self.assertTrue(np.allclose(f.eval(x_mpt)[:,1:],f_rand[:,1:]),\
                        'Function value assignment incorrect.')
        self.assertTrue(np.allclose(f.eval(x_mpt, samples=0).ravel(),\
                                    np.arange(1,5)),\
                        'Function value assignment incorrect.')
        

    def test_global_dofs(self):
        #
        # Check that global dofs are returned
        # 
        # Define mesh
        mesh = QuadMesh()
        mesh.cells.refine()
        element = QuadFE(2,'Q1')
        
        # Initialize nodal function
        vf = np.empty(9,)
        f = Function(vf, 'nodal', mesh=mesh, element=element)
        self.assertTrue(np.allclose(f.global_dofs(), np.arange(9)), \
                        'Incorrect global dofs returned')
    
        # Initialize explicit function
        vf = lambda x,y: x+y
        f = Function(vf, 'explicit', mesh=mesh, element=element)
        self.assertRaises(Exception, f.global_dofs, f)
    
    
    def test_dictionary(self):
        mesh = QuadMesh(resolution=(1,1))
        element_1 = QuadFE(2,'Q1')
        element_2 = QuadFE(2,'Q2')
        dofhandler = {}
        dofhandler['Q1'] = DofHandler(mesh, element_1)
        dofhandler['Q2'] = DofHandler(mesh, element_2)
        
        
    
    def test_flag(self):
        mesh = QuadMesh()
        mesh.cells.refine()
        node = mesh.cells.get_child(0)
        node.mark('1')
        for pos in [0,1]:
            node.get_child(pos).mark('1')
        element = QuadFE(2,'Q1')
        fn = lambda x,y: x+y
        f = Function(fn, 'nodal', mesh=mesh, element=element, subforest_flag='1')
        self.assertEqual(f.flag(),'1', 'Flag incorrect.')
    
    
    def test_input_dim(self):
        # Explicit
        f = Function(lambda x:x**2, 'explicit')
        self.assertEqual(f.input_dim(), 1, \
                         'The function should have one input.')
        
        f = Function(lambda x,y: x+y, 'explicit')
        self.assertEqual(f.input_dim(), 2, \
                         'The function should have two inputs.')
        
        # Nodal 
        #2D 
        mesh = QuadMesh()
        mesh.cells.refine()
        element = QuadFE(2,'Q1')
        vf = np.empty(9,)
        f = Function(vf, 'nodal', mesh=mesh, element=element)
        self.assertEqual(f.input_dim(),2,\
                         'Function should have two inputs.')
        
        # TODO: 1D
    
    def test_n_samples(self): 
        mesh = QuadMesh()
        mesh.cells.refine()
        element = QuadFE(2, 'Q1')
        # Assign 1d vector to function values -> sample size should be None
        vf = np.empty(9,)
        f = Function(vf, 'nodal', mesh=mesh, element=element)
        self.assertEqual(f.n_samples(),None, \
                         'Number of samples should be None.')
        # Assign column vector to function values -> sample size should be 1
        f = Function(vf[:,np.newaxis], 'nodal', mesh=mesh, element=element)
        self.assertEqual(f.n_samples(),1, \
                         'Number of samples should be 1.')
        
        # Assign (9,10) array to function values -> sample size should be 10 
        f = Function(np.empty((9,10)), 'nodal', mesh=mesh, element=element)
        self.assertEqual(f.n_samples(), 10, \
                         'Number of samples should be 10.')
        
        
    def test_interpolate(self):
        #
        # Interpolate a coarse function on a fine mesh
        # 
        mesh = Mesh1D(resolution=(2,))
        mesh.cells.record(0)
        mesh.cells.get_child(0).mark()
        mesh.cells.refine(refinement_flag=True, new_label=1)
        element = QuadFE(1,'Q1')
        dofhandler = DofHandler(mesh, element)
        
        fn = lambda x: x**2
        f = Function(fn, 'nodal', dofhandler=dofhandler, subforest_flag=1)
        x = np.linspace(0,1,5)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.linspace(0,1,1000)
        plt.plot(x,f.eval(x),'k--')
        
        element = QuadFE(1, 'DQ0')
        fi = f.interpolant(mesh, element, subforest_flag=0)
        plt.plot(x,fi.eval(x),'b--')
        
        
    def test_derivative(self):
        pass
    
    def test_times(self):
        pass
        
