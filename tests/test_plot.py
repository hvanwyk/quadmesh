'''
Created on Feb 24, 2017

@author: hans-werner
'''

import unittest
from plot import Plot
from mesh import Mesh
from finite_element import System
from finite_element import QuadFE
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import *  # @UnresolvedImport
import numpy as np

class TestPlot(unittest.TestCase):


    def test_plot_mesh(self):
        """
        Plot the computational mesh
        """
        plt.close('all')

        #
        # Initialize
        #
        fig, ax = plt.subplots(3,3)
        plot = Plot()
        #
        # Define mesh
        # 
        mesh = Mesh.newmesh(grid_size=(2,2))
        mesh.refine()      
        mesh.root_node().children[1,1].mark(1)
        mesh.refine(1)
        
        # Plot simple mesh
        ax[0,0] = plot.mesh(ax[0,0], mesh)
        
        #
        # Flag a few cells
        # 
        mesh.unmark(nodes=True)
        mesh.root_node().children[0,0].mark(2)
        mesh.root_node().children[1,0].mark(1)
        mesh.root_node().children[1,1].children['SW'].mark(3)
        mesh.root_node().children[1,1].children['NE'].mark(3)
        
        # Color flagged cells
        ax[0,1] = plot.mesh(ax[0,1], mesh, color_marked=[1,2,3], nested=True)
        
        # Plot vertex numbers
        ax[0,2] = plot.mesh(ax[0,2], mesh, vertex_numbers=True)
        
        # Plot edge numbers
        ax[1,0] = plot.mesh(ax[1,0], mesh, edge_numbers=True)
        
        # Plot cell numbers nested off
        mesh.refine(2)
        ax[1,1] = plot.mesh(ax[1,1], mesh, cell_numbers=True)
        
        # Plot cell numbers nested on
        ax[1,2] = plot.mesh(ax[1,2], mesh, cell_numbers=True, nested=True)

        # Plot dofs
        element = QuadFE(2,'Q1')
        ax[2,0] = plot.mesh(ax[2,0], mesh, element=element, dofs=True)
        
        # Assign dofs in a nested way
        ax[2,1] = plot.mesh(ax[2,1], mesh, element=element, dofs=True, \
                            nested=True)
        
        # Display only dofs of flagged nodes 
        ax[2,2] = plot.mesh(ax[2,2], mesh, element=element, dofs=True, \
                            node_flag=3, nested=True, show_axis=True)

    
    def test_plot_contour(self):
       
        f = lambda x,y: np.sin(3*np.pi*x*y)
        
        fig, ax = plt.subplots(3,3)
        plot = Plot()
        mesh = Mesh.newmesh(grid_size=(5,5))
        mesh.refine()
        
        #
        # Explicit function
        # 
        fig, ax[0,0] = plot.contour(ax[0,0], fig, f, mesh, colorbar=False)
        ax[0,0].axis('off')
        
        #
        # Nodal function
        #
        element = QuadFE(2,'Q1') 
        system = System(mesh, element)
        x = system.dof_vertices()
        fn = f(x[:,0],x[:,1])
        fig, ax[0,1] = plot.contour(ax[0,1], fig, fn, mesh, element, \
                                    colorbar=False)
        ax[0,1].axis('off')
        #
        # Mesh function
        #
        # Refine mesh 
        mesh = Mesh.newmesh(grid_size=(5,5))
        mesh.refine()
        for _ in range(4):
            for leaf in mesh.root_node().find_leaves():
                if np.random.rand() < 0.5:
                    leaf.mark('refine')
            mesh.refine('refine')
            mesh.unmark(nodes='True')
        mesh.balance()
        # Meshfunction
        fm = []
        for cell in mesh.iter_quadcells():
            xi,yi = cell.vertices['M'].coordinate()
            fm.append(f(xi,yi))
        fm = np.array(fm)
        # Plot
        fig, ax[0,2] = plot.contour(ax[0,2], fig, fm, mesh, colorbar=False)
        ax[0,2].axis('off')
        
        #
        # Plot x-derivative, using piecewise linear, quadratic, cubic
        #  
        element_list = ['Q1','Q2','Q3']
        for j in range(2):
            for i in range(3):
                etype = element_list[i]
                element = QuadFE(2,etype)
                system = System(mesh, element)
                x = system.dof_vertices()
                fn = f(x[:,0],x[:,1])
                fig, ax[1+j,i] = plot.contour(ax[1+j,i], fig, fn, mesh, element,\
                                            derivatives=(1,j), colorbar=False)
                ax[1+j,i].axis('off')
                
                
    def test_plot_surface(self):
        """
        Surface plots
        """
        f = lambda x,y: np.exp(-x**2-y**2)
        mesh = Mesh.newmesh(box=[-1,1,-1,1], grid_size=(2,2))
        mesh.refine()
        for _ in range(4):
            for leaf in mesh.root_node().find_leaves():
                if np.random.rand() < 0.5:
                    leaf.mark('s')
            mesh.refine('s')
            mesh.unmark(nodes=True)
        mesh.balance()   
        element = QuadFE(2,'Q1')
        
        fig = plt.figure()
        plot = Plot()
        
        #
        # Explicit function
        # 
        ax = fig.add_subplot(3,3,1, projection='3d')
        ax = plot.surface(ax, f, mesh, element)
        ax.set_title('explicit function')
        
        #
        # Nodal function
        #
        ax = fig.add_subplot(3,3,2, projection='3d')
        element = QuadFE(2,'Q1') 
        system = System(mesh, element)
        x = system.dof_vertices()
        fn = f(x[:,0],x[:,1])
        # Plot
        ax = plot.surface(ax, fn, mesh, element)
        ax.set_title('nodefunction')
        #
        # Mesh function
        #
        fm = []
        for cell in mesh.iter_quadcells():
            xi,yi = cell.vertices['M'].coordinate()
            fm.append(f(xi,yi))
        fm = np.array(fm)
        # Plot
        ax = fig.add_subplot(3,3,3, projection='3d')
        ax = plot.surface(ax, fm, mesh, element)
        ax.set_title('meshfunction')
        #
        # Plot x-derivative, using piecewise linear, quadratic, cubic
        #  
        element_list = ['Q1','Q2','Q3']
        for j in range(2):
            for i in range(3):
                etype = element_list[i]
                element = QuadFE(2,etype)
                system = System(mesh, element)
                x = system.dof_vertices()
                fn = f(x[:,0],x[:,1])
                ax = fig.add_subplot(3,3,3*(j+1)+(i+1), projection='3d')
                ax = plot.surface(ax, fn, mesh, element,\
                                  derivatives=(1,j))
                ax.set_title('%s: df_dx%d'%(etype,j))
        
        
        plt.show()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    