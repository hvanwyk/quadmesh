from gmrf import Covariance
from gmrf import GaussianField

import Tasmanian

import numpy as np
import matplotlib.pyplot as plt
import unittest
from scipy.stats import norm

class TestGaussHermite(unittest.TestCase):
    def test_standard_normal(self):
        """
        Test modes of a standard normal density 
        """
        # Initialize sparse grid
        dim = 1
        level = 3
        moments = [1,0,1,0,3]
        
        # Define Gauss-Hermite physicist's rule exp(-x**2)
        grid = Tasmanian.makeGlobalGrid(dim, 1, level, "level", "gauss-hermite")
        
        # 
        # Explicit
        #
        for i in range(len(moments)): 
            z = grid.getPoints()             # quadrature nodes
            w = grid.getQuadratureWeights()  # quadrature weights
            y = np.sqrt(2)*z                 # transform to N(0,1)
            c_norm = np.sqrt(np.pi)**dim     # normalization constant
            mom_a = np.sum(w*(y[:,0]**i))/c_norm
            mom_e = moments[i]
            self.assertAlmostEqual(mom_a, mom_e)
            
        #
        # Using integrate
        #
        for i in range(len(moments)):
            z = grid.getPoints()             # quadrature nodes
            y = np.sqrt(2)*z                 # transform to N(0,1)
            c_norm = np.sqrt(np.pi)**dim     # normalization constant
            
            grid.loadNeededPoints(y**i)
            mom_a = grid.integrate()/c_norm
            mom_e = moments[i]
            
            self.assertAlmostEqual(mom_a[0], mom_e)
            
    def test_gaussian_random_field(self):
        """
        Reproduce statistics of Gaussian random field
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
        d = np.array([4,3,2,1], dtype=float)
        Lmd = np.diag(d)
        
        # Covariance matrix
        K = V.dot(Lmd.dot(V.T))
        
        mu = np.array([1,2,3,4])[:,None]
                
        # Zero mean Gaussian field
        dim = 4
        eta = GaussianField(dim, mean=mu, K=K, mode='covariance')
        n_vars = eta.covariance().size()
        level = 1
        
        # Define Gauss-Hermite physicist's rule exp(-x**2)    
        grid = Tasmanian.makeGlobalGrid(n_vars, 4, level, "level", "gauss-hermite-odd")
        
        
        # Evaluate the Gaussian random field at the Gauss points
        z = grid.getPoints()
        y = np.sqrt(2)*z

        const_norm = np.sqrt(np.pi)**n_vars
        
        # Evaluate the random field at the Gauss points
        w = grid.getQuadratureWeights()
        etay = eta.sample(z=y.T)
        n = grid.getNumPoints()
        I = np.zeros(4)
        II = np.zeros((4,4))
        for i in range(n):
            II += w[i]*np.outer(etay[:,i]-mu.ravel(),etay[:,i]-mu.ravel())
            I += w[i]*etay[:,i]
        I /= const_norm    
        II /= const_norm
        
        self.assertTrue(np.allclose(II,K))
        self.assertTrue(np.allclose(I,mu.ravel()))
        
        
    def test_interpolant(self):
        dim = 1
        level = 3
        grid = Tasmanian.makeGlobalGrid(dim,1,level,'level','gauss-hermite')
        #f = lambda x: np.exp(-np.abs(x))
        f = lambda x: np.sum(x**3,axis=1)[:,None]
        
        
        # Evaluate function at abscissae
        z = grid.getPoints()
        fz = f(z)
        
        # Store points in grid
        grid.loadNeededPoints(fz)
        
        # Evaluate on a finer grid
        x = np.linspace(-1,1,100)[:,None]
        y = grid.evaluateBatch(x)
        
        # Check accuracy
        self.assertTrue(np.allclose(y,f(x)))
        
    def test_surrogate(self):
        #
        # Use sparse grid interpolant to sample
        # 
        dim = 1
        level = 3
        grid = Tasmanian.makeGlobalGrid(dim,1,level,'level','gauss-hermite')
 
        # Convert from physicist's to probabilist's variable       
        z = np.sqrt(2)*grid.getPoints()
        
        # Evaluate function at given points and store
        f = lambda x: x**2
        fz = f(z)
        grid.loadNeededPoints(fz)
        
        # Generate random sample of standard normal variables
        x = np.random.normal(size=(10000,1))
        
        # Convert to physicist's domain and evaluate batch
        x2 = grid.evaluateBatch(x/np.sqrt(2))
        self.assertTrue(np.allclose(x2,x**2))
        
        I = grid.integrate()/np.sqrt(np.pi)
        self.assertAlmostEqual(I[0],1)
        
        
    def test_transform(self):
        """
        Approximate moments of a Gaussian random vector 
        
            X ~ N([3,4], [[2,1],[1,3]])
            
        by a sparse grid method based on the interval [-1,1]^2 
        """
        #
        # Define Sparse Grid on [-1,1]^2
        # 
        dim = 2
        level = 40
        grid = Tasmanian.makeGlobalGrid(dim,1,level,'level','gauss-legendre')
        n_points = grid.getNumPoints()
        y = grid.getPoints()
        
        #
        # Transform Points to Z~N(0,I)
        # 
        z = norm.ppf(0.5*y+0.5)
        dz = 0.5**dim
        
        #
        # Define Gaussian Field
        # 
        K = np.array([[2,1],[1,3]])
        m = np.array([3,4])
        
        # Eigendecomposition
        lmd, V = np.linalg.eigh(K)
        lmd = lmd[::-1]
        V = V[:,::-1]
        sqrtD = np.diag(np.sqrt(lmd))

        X = V.dot(sqrtD.dot(z.T)) 
        Y = X + np.tile(m[:,None],(1,n_points))
        
        #
        # Recompute mean and covariance matrix
        # 
        w = grid.getQuadratureWeights()*dz
        ma = np.zeros(2)
        Ka = 0
        for i in range(n_points):
            ma += Y[:,i]*w[i]
            Ka += X[1,i]*X[0,i]*w[i]

