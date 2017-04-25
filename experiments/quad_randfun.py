"""
Compute the spatial integral of a random function f(x,w)

We want to investigate how conditional sampling can be used to simplify the
integrand.

Steps:

1) For a given deterministic integrand, compute the integral via the 
    trapezoidal rule.
    
2) Generate sample paths of a stochastic integrand. 
    
3) Generate conditional sample paths of the integrand


"""
# -----------------------------------------------------------------------------
# Local
# -----------------------------------------------------------------------------
from gmrf import sqr_exponential_cov

# -----------------------------------------------------------------------------
# External
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def trapezoidal(f,a,b,n):
    """
    Compute the numerical integral I_[a,b] f(x) dx using the trapezoidal rule
    with n sub-intervals
    
    Inputs:
    
        f: function, function to be integrated
    
        a,b: double, interval endpoints
        
        n: int, number of subintervals
        
    Outputs:
    
        I = (b-a)/2n * [f(x0) + 2f(x1) + 2f(x2) + ... + 2f(xn-1) + f(xn)]
        
    
    Note: May replace a,b,n with mesh (adaptive?)  
    """ 
    x = np.linspace(a,b,n+1)
    fx = f(x)
    w = 0.5*(b-a)/n*np.array([1] + [2]*(n-1) + [1])
    return np.sum(w*fx)

  
if __name__ == '__main__':
    f = lambda x: np.sin(0.5*np.pi*x)
    a, b = 0, 1
    l = .08
    Ie = 2/np.pi
    E_trap = []
    n_quad_max = 100
    for n in np.arange(1,n_quad_max):
        x = np.linspace(a,b,n+1)
        X,Y = np.meshgrid(x,x)
        S = sqr_exponential_cov(X.ravel()[:,np.newaxis], Y.ravel()[:,np.newaxis], l).reshape(n+1,n+1)
        L = la.cholesky(S+np.eye(n+1, n+1), lower=True)
        Ia = trapezoidal(f, a, b, n)
        E_trap.append(np.abs(Ia-Ie))
    fig, ax = plt.subplots(2,2)
    n_range = np.arange(1,n_quad_max)
    ax[0,0].spy(L)
    ax[0,1].imshow(S)
    ax[1,0].imshow(np.abs(S-L.dot(L.T)))
    ax[1,1].loglog(n_range, np.array(E_trap),n_range,1/(n_range**2))
    plt.show()
        
        
    