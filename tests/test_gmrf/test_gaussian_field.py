import unittest

from gmrf import GaussianField
from gmrf import SPDMatrix

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def fbm_cov(x,y,H):
    """
    Compute the fractional Brownian motion covariance function
    """
    return 0.5*np.abs(x)**(2*H) + 0.5*np.abs(y)**(2*H) - 0.5*np.abs(x-y)**(2*H)


class TestGaussianField(unittest.TestCase):
    def test_constructor(self):
        #
        # Initialize Gaussian Random Field
        # 
        n = 21  # size
        H = 0.5  # Hurst parameter in [0.5,1]
        
        # Form covariance and precision matrices
        x = np.arange(1,n)
        X,Y = np.meshgrid(x,x)
        K = fbm_cov(X,Y,H)
        
        # Compute the precision matrix
        I = np.identity(n-1)
        Q = linalg.solve(K,I)
        
        # Define SPD matrices
        KK = SPDMatrix(K)
        QQ = SPDMatrix(Q)
        
        # Define mean
        mean = np.random.rand(n-1,1)
        
        # Define Gaussian field
        u_cov = GaussianField(n-1, mean=mean, K=K) 
        u_prc = GaussianField(n-1, mean=mean, K=Q, mode='precision')
        
        # Check vitals
        self.assertTrue(np.allclose(u_cov.covariance().get_matrix(),\
                                    KK.get_matrix()))
        self.assertEqual(u_cov.size(),n-1)
       
        self.assertTrue(np.allclose(u_prc.precision().get_matrix(), Q))
        self.assertTrue(np.allclose(u_cov.mean(),mean))
        self.assertEqual(u_prc.mean(n_copies=2).shape[1],2)
        self.assertTrue(np.allclose(u_prc.b(), QQ.dot(u_prc.mean())))
        
    
    def test_chol_sample(self):
        """
        Sample field using Cholesky factorization of the covariance and of 
        the precision. 
        """
        #
        # Initialize Gaussian Random Field
        # 
        n = 201  # size
        H = 0.5  # Hurst parameter in [0.5,1]
        
        # Form covariance and precision matrices
        x = np.arange(1,n)
        X,Y = np.meshgrid(x,x)
        K = fbm_cov(X,Y,H)
        
        # Compute the precision matrix
        I = np.identity(n-1)
        Q = linalg.solve(K,I)

        # Define mean
        mean = np.random.rand(n-1,1)
        
        # Define Gaussian field
        u_cov = GaussianField(n-1, mean=mean, K=K, mode='covariance')  
        u_prc = GaussianField(n-1, mean=mean, K=Q, mode='precision')
        
        # Define generating white noise
        z = u_cov.iid_gauss(n_samples=10)
        
        
        u_chol_prec = u_prc.sample(z=z, mode='precision', decomposition='chol')
        u_chol_cov = u_cov.sample(z=z, mode='covariance', decomposition='eig')
        u_chol_can = u_prc.sample(z=z, mode='canonical', decomposition='chol')
         
        fig, ax = plt.subplots(1,3, figsize=(7,3))
        ax[0].plot(u_chol_prec, linewidth=0.5); 
        ax[0].set_title('Precision'); 
        ax[0].axis('tight')
        
        ax[1].plot(u_chol_cov, linewidth=0.5); 
        ax[1].set_title('Covariance'); 
        ax[1].axis('tight')
        
        ax[2].plot(u_chol_can, linewidth=0.5); 
        ax[2].set_title('Canonical'); 
        ax[2].axis('tight')
        fig.suptitle('Samples')
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        fig.savefig('gaussian_field_chol_samples.eps')
        
    
    def test_eig_sample(self):
        #
        # Initialize Gaussian Random Field
        # 
        n = 201  # size
        H = 0.5  # Hurst parameter in [0.5,1]
        
        # Form covariance and precision matrices
        x = np.arange(1,n)
        X,Y = np.meshgrid(x,x)
        K = fbm_cov(X,Y,H)
        
        # Compute the precision matrix
        I = np.identity(n-1)
        Q = linalg.solve(K,I)
        
        
        # Define mean
        mean = np.random.rand(n-1,1)
        
        # Define Gaussian field
        u_cov = GaussianField(n-1, mean=mean, K=K, mode='covariance')  
        u_prc = GaussianField(n-1, mean=mean, K=Q, mode='precision')
        
        # Define generating white noise
        z = u_cov.iid_gauss(n_samples=10)
        
        u_eig_prec = u_prc.sample(z=z, mode='precision', decomposition='eig')
        u_eig_cov = u_cov.sample(z=z, mode='covariance',decomposition='eig')
        u_eig_can = u_prc.sample(z=z, mode='canonical',decomposition='eig')
         
        fig, ax = plt.subplots(1,3, figsize=(7,3))
        ax[0].plot(u_eig_prec, linewidth=0.5); 
        ax[0].set_title('Precision'); 
        ax[0].axis('tight')
        
        ax[1].plot(u_eig_cov, linewidth=0.5); 
        ax[1].set_title('Covariance'); 
        ax[1].axis('tight')
        
        ax[2].plot(u_eig_can, linewidth=0.5); 
        ax[2].set_title('Canonical'); 
        ax[2].axis('tight')
        fig.suptitle('Samples')
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        fig.savefig('gaussian_field_eig_samples.eps')
    
    
    def test_degenerate_sample(self):
        """
        Test support and reduced covariance 
        """
        oort = 1/np.sqrt(2)
        V = np.array([[0.5, oort, 0, 0.5], 
                      [0.5, 0, -oort, -0.5],
                      [0.5, -oort, 0, 0.5],
                      [0.5, 0, oort, -0.5]])
        
        # Eigenvalues
        d = np.array([4,3,2,0], dtype=float)
        Lmd = np.diag(d)
        
        # Covariance matrix
        K = V.dot(Lmd.dot(V.T))
        
        # Zero mean Gaussian field
        u_ex = GaussianField(4, K=K, mode='covariance', support=V[:,0:3])
        u_im = GaussianField(4, K=K, mode='covariance')
        u_im.update_support()
        
        # Check reduced covariances
        self.assertTrue(np.allclose(u_ex.covariance().get_matrix(),
                                    u_im.covariance().get_matrix().toarray()))
        
        # Check supports 
        V_ex = u_ex.support()
        V_im = u_im.support()
        
        # Ensure they have the same sign    
        for i in range(V_ex.shape[1]):
            if V_ex[0,i] < 0:
                V_ex[:,i] = -V_ex[:,i]
                
            if V_im[0,i] < 0:
                V_im[:,i] = -V_im[:,i]
                
        self.assertTrue(np.allclose(V_ex,V_im))
        u_ex.set_support(V_ex)
        u_im.set_support(V_im)

        # Compare samples                
        z = u_ex.iid_gauss(n_samples=1)
        u_ex_smp = u_ex.sample(z=z, decomposition='chol')
        u_im_smp = u_im.sample(z=z, decomposition='chol')
        self.assertTrue(np.allclose(u_ex_smp,u_im_smp))
       
       
    def test_specify_support_twice(self):
        # TODO: Haven't done it for precision yet
        oort = 1/np.sqrt(2)
        V = np.array([[0.5, oort, 0, 0.5], 
                      [0.5, 0, -oort, -0.5],
                      [0.5, -oort, 0, 0.5],
                      [0.5, 0, oort, -0.5]])
        
        # Eigenvalues
        d = np.array([4,0,2,0], dtype=float)
        Lmd = np.diag(d)
        
        # Covariance matrix
        K = V.dot(Lmd.dot(V.T))
        
        #
        # Restrict subspace in two steps
        # 
        # Field with predefined subspace
        u = GaussianField(4, K=K, mode='covariance', support=V[:,0:3])
        
        # Further restrict subspace (automatically)
        u.update_support()
        
        # 
        # Reduce subspace at once 
        # 
        v = GaussianField(4,K=K, mode='covariance', support=V[:,[0,2]])
        
        # Check that the supports are the same
        U = u.support()
        V = v.support()
        
        self.assertTrue(np.allclose(U - V.dot(V.T.dot(U)), np.zeros(U.shape)))
        self.assertTrue(np.allclose(V - U.dot(U.T.dot(V)), np.zeros(V.shape)))
        
        
    def test_eig_condition(self):
        """
        Conditioning using Eigen-decomposition
        """
        oort = 1/np.sqrt(2)
        V = np.array([[0.5, oort, 0, 0.5], 
                      [0.5, 0, -oort, -0.5],
                      [0.5, -oort, 0, 0.5],
                      [0.5, 0, oort, -0.5]])
        
        # Eigenvalues
        d = np.array([4,3,2,0], dtype=float)
        Lmd = np.diag(d)
        
        # Covariance matrix
        K = V.dot(Lmd.dot(V.T))
        
        
        KK = SPDMatrix(K)
        
        # Mean
        mu = np.linspace(0,3,4)
        
        # Transformation
        A = np.array([[1,2,3,4],
                      [2,4,6,8]], dtype=float)

        N = V[:,np.abs(d)<1e-13]
        for i in range(A.shape[0]):
            ai = A[i].T
            vi = ai - N.dot(N.T.dot(ai))
            if linalg.norm(vi)>1e-7:
                vi = vi/linalg.norm(vi)
                N = np.append(N, vi[:,None], axis=1)
            
            #print(vi)
        #print(N) 

        
        A = np.array([[1,0,0,-1]])
        
                
        # Check compatibility
        P_A = A.T.dot(linalg.solve(A.dot(A.T),A))
        #print((1-P_A).dot(0))
        #print(A.T - N.dot(N.T.dot(A.T)))
           
        X = GaussianField(4, mean=mu, K=K)
        
        z = X.iid_gauss(n_samples=100)
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.plot(X.eig_sample(n_samples=100), linewidth=0.5)
        #plt.show()
        
        #x_cnd = X.chol_condition(A, np.array([[0]]), mode='covariance', n_samples=20)
        #print(A.dot(x_cnd))
        #plt.plot(x_cnd)
        #plt.show()