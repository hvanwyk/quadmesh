'''
Created on Feb 24, 2017

@author: hans-werner
'''

import unittest
from plot import Plot
from mesh import Mesh
from finite_element import QuadFE, DofHandler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl 

class TestPlot(unittest.TestCase):


    def testPlotMesh(self):
        mesh = Mesh.newmesh(grid_size=(4,4))
        mesh.refine()
        mesh.root_node().children[2,2].mark('smaller')
        mesh.refine(flag='smaller')
        fig, ax = plt.subplots() 
        plot = Plot(ax)
        plot.mesh(mesh,cell_numbers=True, vertex_numbers=True)
        #plt.show()
        element = QuadFE(2,'Q3')
        
        _, ax = plt.subplots()
        plot = Plot(ax)
        plot.mesh(mesh,element=element,dofs=True)
        #plt.show()
    
    def test_plot_function(self):
        print("Matplotlib version: {0}".format(mpl.__version__))
        mesh = Mesh.newmesh(grid_size=(40,40))
        mesh.refine()
        element = QuadFE(2,'Q3')
        _,ax = plt.subplots() 
        plot1 = Plot(ax)
        f = lambda x,y: np.sin(3*np.pi*x*y)
        dof_handler = DofHandler(mesh,element)
        dof_handler.distribute_dofs()
        x = dof_handler.mesh_nodes()
        f_vec = f(x[:,0],x[:,1])
        plot1.function(f, mesh, element)
        #plot.function(f_vec,mesh, element)
        
        mesh = Mesh.newmesh(grid_size=(5,5))
        mesh.refine()
        for i in range(4):
            for leaf in mesh.root_node().find_leaves():
                if np.random.rand() < 0.5:
                    leaf.mark('refine')
            mesh.refine('refine')
            mesh.unmark(nodes='True')
        fm = []
        for cell in mesh.iter_quadcells():
            xi,yi = cell.vertices['M'].coordinate()
            fm.append(f(xi,yi))
        fm = np.array(fm)
        _,ax = plt.subplots()
        plot2 = Plot(ax)
        plot2.function(fm,mesh)
        plt.show()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()