'''
Created on Mar 11, 2017

@author: hans-werner
'''

import unittest

from gmrf import Gmrf
from mesh import QuadMesh, DCEL
from fem import QuadFE, DofHandler, Assembler, Function
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sksparse.cholmod import cholesky  # @UnresolvedImport
import matplotlib.pyplot as plt
from plot import Plot

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

    def test_distance(self):
        """
        Test distance function
        """
        # ---------------------------
        # 1D
        # ---------------------------
        M = 2
        box = (0,1)
        x = np.array([0.5,0.75])
        y = np.array([0.25,0.125])
        d_xy = distance(x,y)
        d_xMy = distance(x,y,M=M)
        d_xy_tau = distance(x,y, periodic=True, box=box)
        d_xMy_tau = distance(x, y, M=M, periodic=True, box=box)
        self.assertTrue(np.allclose(d_xy, np.array([0.25,0.625])),\
                        'Unweighted distance incorrect.')
        self.assertTrue(np.allclose(d_xMy, np.sqrt(2)*np.array([0.25,0.625])),\
                        'Weighted distance incorrect.')
        self.assertTrue(np.allclose(d_xy_tau, np.array([0.25,0.375])),\
                        'Unweighted toroidal distance incorrect.')
        self.assertTrue(np.allclose(d_xMy_tau, np.sqrt(2)*np.array([0.25,0.375])),\
                        'Weighted toroidal distance incorrect.') 
        # --------------------------
        # 2D
        # ---------------------------
        M = np.array([[3,1],[1,2]])
        box = (0,1,0,1)
        x = np.array([[0.5,0.5],[0.75, 0.75]])
        y = np.array([[0.25,0.25],[0.125,0.5]])
        d_xy = distance(x, y)
        d_xMy = distance(x, y, M=M)
        d_xy_tau = distance(x, y, periodic=True, box=box)
        d_xMy_tau = distance(x, y, M=M, periodic=True, box=box)
        self.assertTrue(np.allclose(d_xy, \
                        np.array([np.sqrt(2)/4, np.sqrt(29)/8])), \
                        'Distance incorrect')
        
        self.assertTrue(np.allclose(d_xMy, \
                        np.array([np.sqrt(7)/4, np.sqrt(103)/8])),\
                        'Weighted distance incorrect.')
        self.assertTrue(np.allclose(d_xy_tau,\
                        np.array([np.sqrt(2)/4, np.sqrt(13)/8])),\
                        'Toroidal distance incorrect')
        self.assertTrue(np.allclose(d_xMy_tau,\
                        np.array([np.sqrt(7)/4, np.sqrt(47)/8])),\
                        'Weighted toroidal distance incorrect')
             
        
    def test_constructor(self):
        """
        TODO: This test doesn't test any values as yet. 
        """
        #
        # Precision specified
        # 
        Q = np.array([[6,-1,0,-1],[-1,6,-1,0],[0,-1,6,-1],[-1,0,-1,6]])
        S = np.linalg.inv(Q)
        mu = np.zeros(4)
        X = Gmrf(mu=mu, precision=Q, covariance=S)
        
        #
        # From covariance kernel
        #
        
        cov_names = ['constant', 'linear', 'gaussian', 
                     'exponential', 'matern', 'rational']
        anisotropy = [None, np.diag([2,1])]
        
        mesh = QuadMesh(resolution=(20,20))
        mesh.refine()
        element = QuadFE(2,'Q1')
        fig = plt.figure()
        plot = Plot()
        for M in anisotropy:
            cov_pars = {'constant': {'sgm':1},   
                        'linear': {'sgm': 1}, 
                        'gaussian': {'sgm': 1, 'l': 0.1}, 
                        'exponential': {'l': 0.1}, 
                        'matern': {'sgm': 1, 'nu': 2, 'l': 0.5}, 
                        'rational': {'a': 3}}
            count = 1
            for cov_name in cov_names:
                
                S = Gmrf.covariance_matrix(cov_name, cov_pars[cov_name], mesh=mesh, M=M)                
                X = Gmrf(mesh=mesh, covariance=S, discretization='finite_differences')
                
                
                Xi = X.sample(n_samples=1, mode='covariance')
                print(Xi.shape)
                ax = fig.add_subplot(2,3,count)
                plot.contour(ax, fig, Xi, mesh, element)
                count += 1
                #Xi = Function(Xi, 'nodal', mesh=mesh, element=element)            
                
                #
                # Finite Difference
                # 
                #X_fd = Gmrf.from_covariance_kernel(cov_name, cov_par, mesh)
                
                #
                # Finite Elements
                # 
                #X_fe = Gmrf.from_covariance_kernel(cov_name, cov_par, mesh, \
                #                                   element=element)
            plt.show()
    
    def test_covariance_matrix(self):
        """
        
        """ 
        mesh = QuadMesh(resolution=(10,10))
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        #x = dofhandler.dof_vertices()
        #n = dofhandler.n_dofs()
        n = 11
        x = np.linspace(0,1,n)
        
        i,j = np.triu_indices(n)
        #M = np.array([[2,1],[1,2]])
        M = 3
        #X, Y = x[i,:], x[j,:]
        X,Y = x[i], x[j]
        
        cov_fn = Gmrf.linear_cov
        cov_par = {'sgm':1}
        S = cov_fn(X, Y, **cov_par, M=M)   
        
        
        
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
        L = sp.csc_matrix([[1,0,0],[0,2,0],[1,2,3]])
        x = np.array([1,2,3])      
        b = L*x
        X = Gmrf(precision=L*L.T)
        self.assertTrue(np.allclose(X.L(x),b,1e-10),\
                        'L*x incorrect.')
        self.assertTrue(np.allclose(X.L().toarray(),L.toarray(),1e-10),\
                        'L incorrect.')
        self.assertRaises(AssertionError,X.L,b,mode='covariance')
        
        X = Gmrf(precision=(L*L.T).toarray())
        self.assertTrue(np.allclose(X.L(x),b,1e-10),\
                        'L*x incorrect.')
        self.assertRaises(AssertionError,X.L,b,mode='covariance')
        
        
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
        # ====================================================================
        # Precision
        # =====================================================================
        L = sp.csc_matrix([[1,0,0],[0,2,0],[1,2,3]])
        x = np.array([1,2,3])
        b = L*x
        Q = L*L.T
        
        #
        # Sparse
        # 
        X = Gmrf(precision=Q)
        self.assertTrue(np.allclose(X.L_solve(b),x,1e-10),\
                        'L solve returns incorrect result.')
        self.assertRaises(AssertionError, X.L_solve,b, mode='covariance')
        
        #
        # Dense
        # 
        X = Gmrf(precision=Q.toarray())
        self.assertTrue(np.allclose(X.L_solve(b),x,1e-10),\
                        'L solve returns incorrect result.')
        
        # =====================================================================
        # Covariance 
        # =====================================================================
        #
        # Sparse 
        # 
        X = Gmrf(covariance=Q)
        self.assertTrue(np.allclose(X.L_solve(b,mode='covariance'),x,1e-10),\
                        'L solve returns incorrect result.')
        self.assertRaises(AssertionError, X.L_solve,b, mode='precision')
        
        #
        # Dense
        # 
        X = Gmrf(covariance=Q.toarray())
        self.assertTrue(np.allclose(X.L_solve(b,mode='covariance'),x,1e-10),\
                        'L solve returns incorrect result.')
        self.assertRaises(AssertionError, X.L_solve,b, mode='precision')
        
        
    def test_Lt_solve(self):
        # ====================================================================
        # Precision
        # =====================================================================
        L = sp.csc_matrix([[1,0,0],[0,2,0],[1,2,3]])
        x = np.array([1,2,3])
        b = L.transpose()*x
        Q = L*L.T
        #
        # Sparse
        # 
        X = Gmrf(precision=Q)
        self.assertTrue(np.allclose(X.Lt_solve(b),x,1e-10),\
                        'L solve returns incorrect result.')
        self.assertRaises(AssertionError, X.Lt_solve,b, mode='covariance')
        
        #
        # Dense
        # 
        X = Gmrf(precision=Q.toarray())
        self.assertTrue(np.allclose(X.Lt_solve(b),x,1e-10),\
                        'L solve returns incorrect result.')
        
        # =====================================================================
        # Covariance 
        # =====================================================================
        #
        # Sparse 
        # 
        X = Gmrf(covariance=Q)
        self.assertTrue(np.allclose(X.Lt_solve(b,mode='covariance'),x,1e-10),\
                        'L solve returns incorrect result.')
        self.assertRaises(AssertionError, X.Lt_solve,b, mode='precision')
        
        #
        # Dense
        # 
        X = Gmrf(covariance=Q.toarray())
        self.assertTrue(np.allclose(X.Lt_solve(b,mode='covariance'),x,1e-10),\
                        'L solve returns incorrect result.')
        self.assertRaises(AssertionError, X.Lt_solve,b, mode='precision')    
    
    
    def test_sample(self):
        #
        # TODO Don't know how to test this routine yet
        # 
       
        L = sp.csc_matrix([[1,0,0],[0,2,0],[1,2,3]])
        x = np.array([1,2,3])
        b = L.transpose()*x
        Q = L*L.T
        S = sp.csc_matrix(np.linalg.inv(Q.toarray()))
        X = Gmrf(precision=Q, covariance=S)
        #print(X.L(b,mode='covariance')-x)
        #Ltilde = X.L(mode='covariance')
        #print((Ltilde).toarray())
        #print(np.linalg.inv(L.T.toarray()))
        #print((Ltilde*Ltilde.T*L*L.T).toarray())
        #print((Ltilde*Ltilde.T - S).toarray())
        #print(np.linalg.inv(L.toarray()).dot((Ltilde.T).toarray()))
        #print('{0}'.format(X.sample(z=b, mode='precision')))
        #print('{0}'.format(X.sample(z=b, mode='covariance')))
        
        n = 5
        Q = laplacian_precision(n, sparse=True)
        S = sp.csc_matrix(np.linalg.inv(Q.toarray()))
        X = Gmrf(precision=Q, covariance=S)
        z = np.random.normal(size=(X.n(),))
        
        x_prec = X.sample(z=z, mode='precision')
        x_cov = X.sample(z=z, mode='covariance')
        x_can = X.sample(z=z, mode='canonical')
        #for x in [x_prec,x_cov, x_can]:
        #    print(x)
        #self.assertTrue(np.allclose(x_prec,x_cov,1e-10), \
        #                'Precision samples differ from covariance samples.')
        #self.assertTrue(np.allclose(x_cov,x_can,1e-10), \
        #                'Covariance samples differ from canonical samples.')
        
        
    
    def test_condition(self):
        """
        Condition using (a) precision, (b) covariance, and (c) svd
        Condition on (i) pointwise data, (ii) hard constraints, 
        (iii) soft constraints. (1) finite elements, (2) finite
        differences.
        """
        mesh = QuadMesh(resolution=(10,10))
        mesh.cells.refine()
        mesh.cells.record(0)
        for dummy in range(2):
            for leaf in mesh.cells.get_leaves():
                x = leaf.get_vertices()
                if all(x[:,0]>=0.25) and all(x[:,0]<=0.75) and \
                   all(x[:,1]>=0.25) and all(x[:,1]<=0.75):
                    leaf.mark('refine')
            mesh.cells.refine(refinement_flag='refine')
            mesh.balance()
            
            #mesh.root_node().unmark(flag='refine',recursive=True)
            
        mesh.record(1)
        element = QuadFE(2,'Q1')
        
        """
        fig, ax = plt.subplots(1,2)
        plot = Plot()
        ax[0] = plot.mesh(ax[0], mesh, element=element, color_marked=[0,1], nested=True)
        ax[1] = plot.mesh(ax[1], mesh, element=element, node_flag=1, nested=True)
        plt.show()
        """
        
        cov_names = ['linear', 'sqr_exponential', 'exponential', 
                     'matern', 'rational']
        
        M = None
        cov_pars = {'linear': {'sgm': 1, 'M': M}, 
                    'sqr_exponential': {'sgm': 1, 'l': 0.1 ,'M': M}, 
                    'exponential': {'l': 0.1, 'M': M}, 
                    'matern': {'sgm': 1, 'nu': 2, 'l': 0.5, 'M': M}, 
                    'rational': {'a': 3, 'M': M}}
        for cov_name in cov_names:
            cov_par = cov_pars[cov_name]
            #X = Gmrf.from_covariance_kernel(cov_name, cov_par, mesh,\
            #                               element=element)
            
    
    def test_matern_precision(self):
        
        #
        # Define mesh and element    
        # 
        mesh = QuadMesh(resolution=(40,40), box=[0,20,0,20])
        mesh.refine()
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh,element)
        dofhandler.distribute_dofs()
        #kappa = lambda x,y: np.log(2+5*x**2 + 2*y**3);
        kappa = 3
        alpha =3
        system = Assembler(mesh,element)
        X = Gmrf.from_matern_pde(alpha, kappa, mesh, element)
        """
        Xsmpl = X.sample(n_samples=1)
        from plot import Plot
        fig, ax = plt.subplots()
        plot = Plot()
        plot.contour(ax, fig, Xsmpl.ravel(), mesh, element)
        plt.show()
        #Q = X.matern_precision(mesh, element, alpha, kappa)
        #Q = Q.tocsc()
        #factor = cholesky(Q)
        #P = factor.P()
        #plt.spy(Q[P[:, np.newaxis], P[np.newaxis, :]], markersize=0.2)
        #plt.spy(Q, markersize=0.5)
        #plt.show()
        #print(Q.nnz)
        #print('Number of rows: {0}'.format(Q.shape[0]))
        #print('Number of dofs: {0}'.format(dofhandler.n_dofs()))
        """
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()