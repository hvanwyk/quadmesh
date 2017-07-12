"""
Compute covariance matrices from covariance functions

Source: C. E. Rasmussen & C. K. I. Williams, 
        Gaussian Processes for Machine Learning, 
        the MIT Press, 2006
"""
import numpy as np

def constant_cov(x,y,c):
    """
    Constant covariance kernel
    
        C(x,y) = c
    
    """
    return c*np.ones(x.shape[0])
    

def linear_cov(x,y, sgm=1):
    """
    Linear covariance
    
        C(x,y) = <x,y>  (Euclidean inner product)
     
    """
    return sgm + np.sum(x*y, axis=1)

    
def sqr_exponential_cov(x,y,l):
    """
    Squared exponential covariance function
    
        C(x,y) = exp(-|x-y|^2/(2l^2))
    
    """
    d = distance(x,y)
    return np.exp(-d**2/(2*l**2))

    
def ornstein_uhlenbeck_cov(x,y,l):
    """
    Ornstein-Uhlenbeck covariance function
    
        C(x,y) = exp(-|x-y|/l)
        
    """
    d = distance(x,y)
    return np.exp(-d/l)

    
def matern_cov(x,y,sgm,nu,l):
    """
    Matern covariance function
    """
    d = distance(x,y)
    K = sgm**2*2**(1-nu)/gamma(nu)*(np.sqrt(2*nu)*d/l)**nu*\
        kv(nu,np.sqrt(2*nu)*d/l)
    #
    # Modified Bessel function undefined at d=0, covariance should be 1
    #
    K[np.isnan(K)] = 1
    return K
    
    
def rational_cov(x,y,a):
    """
    Rational covariance
    
        C(x,y) = 1/(1 + |x-y|^2)^a
         
    """
    d = distance(x,y)
    return (1/(1+d**2))**a


def distance(x,y):
    """
    Compute the Euclidean distance vector between rows in x and rows in y
    """
    #
    # Check wether x and y have the same dimensions
    # 
    assert x.shape == y.shape, 'Vectors x and y have incompatible shapes.'
    return np.sqrt(np.sum((x-y)**2,axis=1)) 


if __name__ == '__main__':
    cov_fns = {'constant': constant_cov,
               'linear': linear_cov, 
               'squared_exponential': sqr_exponential_cov, 
               'ornstein_uhlenbeck': ornstein_uhlenbeck_cov,
               'matern': matern_cov}
    
    # =========================================================================
    # One dimensional
    # =========================================================================
    
    #
    # Covariance matrices by finite differences
    #
    x = np.linspace()
    
    #
    # Covariance by finite elements
    # 