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
import scipy.stats as stats

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

 
def sample_field(mu, Cov, n_sample=None, z=None):
    m = len(mu)
    if n_sample is None and z is None:
        raise Exception('Inputs n_sample and z cannot both be None.')
    elif n_sample is not None and z is not None:
        raise Exception('One of the inputs n_sample and z must be None.')
    if z is None:
        if n_sample == 1:
            z = np.random.standard_normal(size=(m,))
        else:
            z = np.random.standard_normal(size=(m,n_sample))
    if n_sample is None:
        if len(z.shape) == 1:
            n_sample = 1
        else:
            n_sample = z.shape[1]
        
    L = la.cholesky(Cov, lower=True)
    if n_sample == 1:
        return L.dot(z) + mu
    else:
        return L.dot(z) + np.tile(mu,(n_sample,1)).transpose() 
            
     
if __name__ == '__main__':
    
    mu_fn = lambda x: np.sin(np.pi*x)
    cov_fn = lambda x,y: 0.1*(1+10*x**2)*np.exp(-np.abs(x-y))
    Ie = 2/np.pi
    a, b = 0, 1
    k_max = 10
    n_sample = np.int(1e5)
    fig, ax = plt.subplots(2,2)
    for k in np.arange(1,k_max):
        m = 2**k
        xm = np.linspace(a,b,m+1)
        mu_m = mu_fn(xm)
        [X,Y] = np.meshgrid(xm,xm)
        Cov = cov_fn(X.ravel(),Y.ravel()).reshape(m+1,m+1)
        fX = sample_field(mu_m, Cov, n_sample=n_sample)
        
        ax[0,0].plot(xm, np.mean(fX,axis=1))
        ax[0,1].plot(xm, np.var(fX,axis=1))
        ax[1,1].plot(xm, fX[:,0])
        #
        # Compute integral
        #
        # Integration weights 
        w = 0.5*(b-a)/m*np.array([[1]+[2]*(m-1)+[1]])
        
        # Pathwise Trapezoidal Integral
        IfX = np.sum(fX * np.tile(w,(n_sample,1)).transpose(), axis = 0)
        varIfX = np.var(IfX)
        
        # Trapezoidal Integral of the mean
        IEfX = np.sum(w*np.mean(fX,axis=1))
        
        # Trapezoidal integral of expectation
        Imu = np.sum(mu_m * w)  
        
        ax[1,0].loglog(m, np.abs(Imu-Ie)**2, '+k',
                       m, np.abs(varIfX)/n_sample, '+k',
                       m, varIfX/n_sample + (Imu-Ie)**2,'.b')
        #               m, np.abs(np.mean(IfX)-Ie)**2, '.r')        
        #ax[1,0].loglog(m,np.abs(np.mean(IfX)-Ie)**2,'.k',\
        #               m,np.abs(IEfX - Ie),'+r',\
        #               m,((b-a)/m)**2,'.b')
                       
    """    
    
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
    """
    plt.show()
        
        
    