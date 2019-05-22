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
        u = GaussianField(mean=mean, covariance=KK, precision=QQ)  
        
        # Check vitals
        self.assertEqual(u.covariance(), KK)
        self.assertEqual(u.size(),n-1)
        self.assertEqual(u.precision(), QQ)
        self.assertTrue(np.allclose(u.mean(),mean))
        self.assertEqual(u.mean(n_copies=2).shape[1],2)
        self.assertTrue(np.allclose(u.b(), linalg.solve(Q,u.mean())))
        
    
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
        
        # Define SPD matrices
        KK = SPDMatrix(K)
        QQ = SPDMatrix(Q)
        
        # Define mean
        mean = np.random.rand(n-1,1)
        
        # Define Gaussian field
        u = GaussianField(mean=mean, covariance=KK, precision=QQ)  
        
        # Define generating white noise
        z = u.iid_gauss(n_samples=10)
        
        u_chol_prec = u.chol_sample(z=z, mode='precision')
        u_chol_cov = u.chol_sample(z=z, mode='covariance')
        u_chol_can = u.chol_sample(z=z, mode='canonical')
         
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
        
        # Define SPD matrices
        KK = SPDMatrix(K)
        QQ = SPDMatrix(Q)
        
        # Define mean
        mean = np.random.rand(n-1,1)
        
        # Define Gaussian field
        u = GaussianField(mean=mean, covariance=KK, precision=QQ)  
        
        # Define generating white noise
        z = u.iid_gauss(n_samples=10)
        
        u_chol_prec = u.eig_sample(z=z, mode='precision')
        u_chol_cov = u.eig_sample(z=z, mode='covariance')
        u_chol_can = u.eig_sample(z=z, mode='canonical')
         
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
        fig.savefig('gaussian_field_eig_samples.eps')
    
    
    def test_chol_condition(self):
        """
        Conditioning using Cholesky 
        """
        pass
    
    
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
            
            print(vi)
        print(N) 

        
        A = np.array([[1,0,0,-1]])
        
                
        # Check compatibility
        P_A = A.T.dot(linalg.solve(A.dot(A.T),A))
        print((1-P_A).dot(0))
        #print(A.T - N.dot(N.T.dot(A.T)))
           
        X = GaussianField(mean=mu, covariance=KK)
        
        z = X.iid_gauss(n_samples=100)
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.plot(X.eig_sample(n_samples=100), linewidth=0.5)
        #plt.show()
        
        x_cnd = X.chol_condition(A, np.array([[0]]), mode='covariance', n_samples=20)
        print(A.dot(x_cnd))
        plt.plot(x_cnd)
        plt.show()