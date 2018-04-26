import unittest
from mesh import Mesh
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
    """
    def test_constructor_and_eval(self):
        """
        Test Constructor
        """ 
        
    
        mesh = Mesh()
        mesh.record(0)
        mesh.refine()
        mesh.record(1)
        mesh.refine()
        mesh.record(2)
        
        f = lambda x,y: np.sin(np.pi*x)*np.cos(2*np.pi*y) + np.cos(np.pi*y)
        
        fig = plt.figure()
        
        #
        # Mesh points at which to plot function
        #
        vtcs = mesh.root_node().cell().get_vertices(pos='corners', \
                                                    as_array=True) 
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
        
            
        #plt.show()    
        
        
        plot = Plot()
        element = QuadFE(2,'DQ0')
        fn.interpolant(mesh, element)
        ax = fig.add_subplot(1,3,1)
        ax = plot.mesh(ax, mesh, node_flag=0, color_marked=[0])
        ax = fig.add_subplot(1,3,2)
        ax = plot.mesh(ax, mesh, node_flag=1, color_marked=[1])
        ax = fig.add_subplot(1,3,3)
        ax = plot.mesh(ax, mesh, node_flag=2, color_marked=[2])
        """
               
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
        
    def test_assign(self):
        #
        # New nodal function
        #  
        # Define Mesh and elements
        mesh = Mesh(grid=DCEL(resolution=(2,2)))
        #mesh.refine()
        element = QuadFE(2,'DQ0')
        
        # Initialize
        vf = np.empty(4,)
        dh = DofHandler(mesh, element)
        dh.distribute_dofs()
        f = Function(vf, 'nodal', mesh=mesh, element=element)
        
        # Assign new value to function (vector)
        f_det = np.arange(1,5)
        f.assign(f_det)
        
        # Assign random sample to function
        f_rand = np.random.rand(4,10)
        f.assign(f_rand) 
        
        # Get cell midpoints
        cell_midpoints = []
        for leaf in mesh.root_node().get_leaves():
            cell = leaf.cell()
            cell_midpoints.append(cell.get_vertices(pos='M'))
        x_mpt = np.array(cell_midpoints)
        self.assertTrue(np.allclose(f.eval(x_mpt),f_rand),\
                        'Function value assignment incorrect.')
        
        
        # Now assign in specific position
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
        mesh = Mesh()
        mesh.refine()
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
    
    
    def test_flag(self):
        mesh = Mesh()
        mesh.refine()
        node = mesh.root_node()
        for pos in ['SW','SE']:
            node.children[pos].mark('1')
        element = QuadFE(2,'Q1')
        fn = lambda x,y: x+y
        f = Function(fn, 'nodal', mesh=mesh, element=element, flag='1')
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
        mesh = Mesh()
        mesh.refine()
        element = QuadFE(2,'Q1')
        vf = np.empty(9,)
        f = Function(vf, 'nodal', mesh=mesh, element=element)
        self.assertEqual(f.input_dim(),2,\
                         'Function should have two inputs.')
        
        # TODO: 1D
    
    def test_n_samples(self): 
        mesh = Mesh()
        mesh.refine()
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
        
        
    def test_fn_type(self):
        # Simple
        pass
    
    def test_fn(self):
        # Simple
        pass
    
    def test_interpolate(self):
        pass
    
    def test_derivative(self):
        pass
    
    def test_times(self):
        pass
        
