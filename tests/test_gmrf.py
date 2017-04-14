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
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from sksparse.cholmod import cholesky  # @UnresolvedImport


def laplacian_precision(n, sparse=True):
    """
    Return the laplace precision matrix
    """
    a = np.array([1] + [2]*(n-2) + [1])
    b = np.array([-1]*(n-1))
    Q = np.diag(a,0) + np.diag(b, -1) + np.diag(b, 1) + np.eye(n,n)
    if sparse:
        return sp.coo_matrix(Q)
    else:
        return Q
    
class TestGmrf(unittest.TestCase):


    def test_constructor(self):
        Q = np.array([[6,-1,0,-1],[-1,6,-1,0],[0,-1,6,-1],[-1,0,-1,6]])
        S = np.linalg.inv(Q)
        mu = np.zeros(4)
        X = Gmrf(mu=mu, precision=Q, covariance=S)
        
    
    def test_Q(self):
        # 
        # Full
        #
        Q = laplacian_precision(10, sparse=False)
        X = Gmrf(precision=Q)
        self.assertTrue(np.allclose(X.Q(),Q,1e-9),\
                        'Precision matrix not returned')
        self.assertFalse(sp.isspmatrix(X.Q()),\
                         'Precision matrix should not be sparse')
        #
        # Sparse
        #
        Q = laplacian_precision(10)
        X = Gmrf(precision=Q)
        self.assertTrue(np.allclose(X.Q().toarray(),Q.toarray(),1e-9),\
                        'Precision matrix not returned.')
        self.assertTrue(sp.isspmatrix(X.Q()),\
                         'Precision matrix should not be sparse')
        
        #
        # Q not given
        # 
        X = Gmrf(covariance=Q)
        self.assertEqual(X.Q(), None, 'Should return None.')
        
        
    def test_Sigma(self):
        # 
        # Full
        #
        S = laplacian_precision(10, sparse=False)
        X = Gmrf(covariance=S)
        self.assertTrue(np.allclose(X.Sigma(),S,1e-9),\
                        'Covariance matrix not returned')
        self.assertFalse(sp.isspmatrix(X.Sigma()),\
                         'Covariance matrix should not be sparse')
        #
        # Sparse
        #
        S = laplacian_precision(10)
        X = Gmrf(covariance=S)
        self.assertTrue(np.allclose(X.Sigma().toarray(),S.toarray(),1e-9),\
                        'Covariance matrix not returned.')
        self.assertTrue(sp.isspmatrix(X.Sigma()),\
                         'Covariance matrix should not be sparse')
        
        #
        # Q not given
        # 
        X = Gmrf(precision=S)
        self.assertEqual(X.Sigma(), None, 'Should return None.')
        
        
    
    def test_L(self):
        pass
    
    
    def test_mu(self):
        Q = laplacian_precision(10)
        X = Gmrf(precision=Q)
        self.assertTrue(np.allclose(X.mu(),np.zeros(10),1e-10),\
                        'Mean should be the zero vector.')
        
        mu = np.random.rand(10)
        X = Gmrf(precision=Q,mu=mu)
        self.assertTrue(np.allclose(X.mu(),mu,1e-10),\
                        'Mean incorrect.')
        self.assertTrue(np.allclose(X.b(),spla.spsolve(Q.tocsc(),mu),1e-10),\
                        'Mean incorrect.')
    
    
    def test_n(self):
        pass
    
    
    def test_Q_solve(self):
        n = 10
        for sparse in [True, False]:
            Q = laplacian_precision(n, sparse=sparse)
            b = np.random.rand(n)
            X = Gmrf(precision=Q)
            
            self.assertTrue(np.allclose(Q.dot(X.Q_solve(b)),b,1e-10),\
                            'Q*Q^{-1}b should equal b.')
        
        
    
    
    def test_L_solve(self):
        """
        n = 3
        for sparse in [False, True]:
            print('Sparsity = {0}'.format(sparse))
            Q = laplacian_precision(n, sparse)
            b = np.random.rand(n)
            X = Gmrf(precision=Q)
            print('b={0}'.format(b))
            print('L^(-1)b = {0}'.format(X.L_solve(b)))
            print('L*L^(-1)b = {0}'.format(X.L(X.L_solve(b))))
            self.assertTrue(np.allclose(X.L(X.L_solve(b)),b,1e-10),\
                            'L*L^{-1}b should equal b.')
        """
        Q = laplacian_precision(3, sparse=True)
        X = Gmrf(precision=Q)
        f = cholesky(Q) 
           
        L = X.L().toarray()
        print('L={0}'.format(L))
        Qh = L.dot(L.transpose())
        P = f.P()
        print('Permutation = {0}'.format(f.P()))
        PQPt = P.dot(Q.toarray().dot(P.transpose()))
        print('Q-Qh={0}'.format(PQPt-Qh))    
    
    def test_Lt_solve(self):
        pass
    
    
    def test_sample_covariance(self):
        pass
    
    
    def test_sample_precision(self):
        pass
    
    
    def test_sample_canonical(self):
        pass
    
    
    def test_condition(self):
        pass


    
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
        """
        #
        # Define mesh and element    
        # 
        mesh = Mesh.newmesh(grid_size=(50,50))
        mesh.refine()
        element = QuadFE(2,'Q3')
        dofhandler = DofHandler(mesh,element)
        dofhandler.distribute_dofs()
        #kappa = lambda x,y: 1 + x**2*y;
        kappa = 1 
        alpha = 2
        
        X = Gmrf.from_matern_pde(mesh,element)
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
        """
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()