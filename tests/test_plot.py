'''
Created on Feb 24, 2017

@author: hans-werner
'''

import unittest
from plot import Plot
from mesh import Mesh
from finite_element import QuadFE

import matplotlib.pyplot as plt

class TestPlot(unittest.TestCase):


    def testPlotMesh(self):
        mesh = Mesh.newmesh(grid_size=(4,4))
        mesh.refine()
        mesh.root_node().children[2,2].mark('smaller')
        mesh.refine(flag='smaller')
        fig, ax = plt.subplots() 
        plot = Plot(ax)
        plot.mesh(mesh,cell_numbers=True, vertex_numbers=True)
        plt.show()
        element = QuadFE(2,'Q3')
        
        _, ax = plt.subplots()
        plot = Plot(ax)
        plot.mesh(mesh,element=element,dofs=True)
        plt.show()
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()