'''
Created on Mar 11, 2017

@author: hans-werner
'''

import unittest

from gmrf import Gmrf
from mesh import Mesh
from finite_element import QuadFE, DofHandler

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scikits.sparse.cholmod import cholesky  # @UnresolvedImport

class TestGmrf(unittest.TestCase):


    def test_constructor(self):
        Q = np.array([[6,-1,0,-1],[-1,6,-1,0],[0,-1,6,-1],[-1,0,-1,6]])
        S = np.linalg.inv(Q)
        mu = np.zeros(4)
        X = Gmrf(mu=mu, precision=Q, covariance=S)
        X.Q()

    def test_sample(self):
        import scipy.sparse

        F = scipy.sparse.rand(100, 100, density=0.05)
        M = F.transpose() * F + sp.eye(100, 100)
        M = sp.csc_matrix(M)
        factor = cholesky(M)
        
        #C = decomp_cholesky.cho_factor(M.toarray())

        
        adjacency_list = [M.getrow(i).indices for i in range(M.shape[0])]
        #node_nd = pymetis.nested_dissection(adjacency=adjacency_list)
        #perm, iperm = np.array(node_nd[0]), np.array(node_nd[1])
        #plt.show()
        #assert np.all(perm[iperm] == np.array(range(perm.size)))
    
    
    def test_matern_precision(self):
        #
        # Define mesh and element    
        # 
        mesh = Mesh.newmesh(grid_size=(20,20))
        mesh.refine()
        element = QuadFE(2,'Q3')
        dofhandler = DofHandler(mesh,element)
        dofhandler.distribute_dofs()
        #kappa = lambda x,y: 1 + x**2*y;
        kappa = 1 
        alpha = 2
        
        X = Gmrf()
        Q = X.matern_precision(mesh, element, alpha, kappa)
        Q = Q.tocsc()
        factor = cholesky(Q)
        P = factor.P()
        plt.spy(Q[P[:, np.newaxis], P[np.newaxis, :]], markersize=0.2)
        #plt.spy(Q, markersize=0.5)
        plt.show()
        print(Q.nnz)
        print('Number of rows: {0}'.format(Q.shape[0]))
        print('Number of dofs: {0}'.format(dofhandler.n_dofs()))
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()