import unittest

from gmrf import GaussianField
from gmrf import SPDMatrix

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from scipy import sparse as sp


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
        #
        # TODO: Haven't done it for precision yet
        #
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
        
    
    def test_condition_with_nullspace(self):
        """
        Test conditioning with an existing nullspace
        """
        
        #
        # Define Gaussian Field with degenerate support 
        # 
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
        
        mu = np.array([1,2,3,4])[:,None]
        
        # Zero mean Gaussian field
        u_ex = GaussianField(4, mean=mu, K=K, mode='covariance', 
                             support=V[:,0:3])
        
        #
        # Conditioned random field (hard constraint)
        # 
        # Condition on Ax=e (A full rank)
        A = np.array([[1,2,3,2],[2,4,6,4]])
        e = np.array([[1],[5]])
        
        # Matrix A is not full rank -> error
        with self.assertRaises(np.linalg.LinAlgError):
            u_ex.condition(A,e)
            
        # Full rank matrix
        A = np.array([[1,2,3,2],[3,9,8,7]])
        
        # Compute conditioned field
        u_cnd = u_ex.condition(A,e,output='field')
        X_cnd = u_cnd.sample(n_samples=100)
        
        # Sample by Kriging
        X_kriged = u_cnd.sample(n_samples=100)
        
        # Check that both samples satisfy constraint
        self.assertTrue(np.allclose(A.dot(X_cnd)-e,0))
        self.assertTrue(np.allclose(A.dot(X_kriged)-e,0))
        
        # Check that the support of the conditioned field is contained in
        # that of the unconditioned one 
        self.assertTrue(np.allclose(u_ex.project(u_cnd.support(),'nullspace'),0))
        
        plt.close('all')
        fig, ax = plt.subplots(1,3, figsize=(7,3))
        ax[0].plot(u_ex.sample(n_samples=100), 'k', linewidth=0.1)
        ax[0].set_title('Unconditioned Field')
        
        ax[1].plot(X_kriged,'k',linewidth=0.1)
        ax[1].plot(u_cnd.mean())
        ax[1].set_title('Kriged Sample')
    
        ax[2].plot(X_cnd,'k',linewidth=0.1)
        ax[2].set_title('Sample of conditioned field')
        
        fig.suptitle('Samples')
        fig.tight_layout()
        fig.subplots_adjust(top=0.8)
        fig.savefig('degenerate_gf_conditioned_samples.eps')

    def test_condition_ptwise(self):
        #
        # Initialize Gaussian Random Field
        # 
        # Resolution
        l_max = 9
        n = 2**l_max+1  # size
        
        # Hurst parameter
        H = 0.5  # Hurst parameter in [0.5,1]
        
        # Form covariance and precision matrices
        x = np.arange(1,n+1)
        X,Y = np.meshgrid(x,x)
        K = fbm_cov(X,Y,H)
        
        # Compute the precision matrix
        I = np.identity(n)
        Q = linalg.solve(K,I)
        
        # Plot meshes
        fig, ax = plt.subplots(1,1)
        n = 2**l_max + 1
        for l in range(l_max):
            nl = 2**l + 1
            i_spp = [i*2**(l_max-l) for i in range(nl)]
            ax.plot(x[i_spp],l*np.ones(nl),'.')
        #ax.plot(x,'.', markersize=0.1)
        #plt.show()
        
        
        # Plot conditioned field
        fig, ax = plt.subplots(3,3)
        
        # Define original field
        u = []
        n = 2**(l_max)+1
        for l in range(l_max):
            nl = 2**l + 1
            i_spp = [i*2**(l_max-l) for i in range(nl)]
            V_spp = I[:,i_spp]
            if l==0:
                u_fne = GaussianField(n, K=K, mode='covariance',\
                                      support=V_spp)
                u_obs = u_fne.sample()
                i_obs = np.array(i_spp)
            else:
                u_fne = GaussianField(n, K=K, mode='covariance',\
                                      support=V_spp)
                u_cnd = u_fne.condition(i_obs,u_obs[i_obs], output='field')
                u_obs = u_cnd.sample()
                i_obs = np.array(i_spp)
            u.append(u_obs)
            
            # Plot 
            for ll in range(l,l_max):
                i,j = np.unravel_index(ll,(3,3))
                if ll == l:
                    ax[i,j].plot(x[i_spp],5*np.exp(0.01*u_obs[i_spp]),linewidth=0.5)
                else:
                    ax[i,j].plot(x[i_spp],5*np.exp(0.01*u_obs[i_spp]),'g', linewidth=0.1,alpha=0.1)
            fig.savefig('successive_conditioning.pdf')
            
    def test_condition_pointswise(self):
        """
        Generate samples and random field  by conditioning on pointwise data
        """
        #
        # Initialize Gaussian Random Field
        # 
        # Resolution
        max_res = 10
        n = 2**max_res+1  # size
        
        # Hurst parameter
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
        
        u_obs = u_cov.sample(z=z)
        
        # Index of measured observations
        A = np.arange(0,n-1,2)
        
        # observed quantities        
        e = u_obs[A,0][:,None]
        #print('e shape', e.shape)
        
        # change A into matrix
        k = len(A)
        rows = np.arange(k)
        cols = A
        vals = np.ones(k)
        AA = sp.coo_matrix((vals, (rows,cols)),shape=(k,n-1)).toarray()
        
        
        AKAt = AA.dot(K.dot(AA.T))
        KAt  = K.dot(AA.T)
        
        U,S,Vt = linalg.svd(AA)
        #print(U)
        #print(S)
        #print(Vt)
        
        #print(AA.dot(u_obs)-e)
        
        k = e.shape[0]
        Ko = 0.01*np.identity(k)
        
        # Debug
        K = u_cov.covariance()
        #U_spp = u_cov.support()
        #A_cmp = A.dot(U_spp)
    
        u_cond = u_cov.condition(A,e,Ko=Ko,n_samples=100)
        """
        plt.close('all')
        plt.plot(A,e,'--')
        plt.plot(u_obs[:,0])
        plt.plot(u_cond,'k',linewidth=0.1, alpha=0.5)
        plt.show()
        """
    
    
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