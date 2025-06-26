"""
Series of experiments to compare hierachical sampling schemes for bivariate functions with random inputs.
In particular, consider functions f(Y1, Y2), where Y1 and Y2 are random variables with a joint distribution.
"""
from matplotlib import axes
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

def f1(x,y):
    """Example function f1(x,y) = x^2 + y^2"""
    return x**2 + y**2

def f2(x,y):
    """Example function f2(x,y) = sin(x) + cos(y)"""
    return np.sin(x) + np.cos(y)

def f3(x,y):
    """Example function f3(x,y) = exp(x^2)"""
    return np.exp(-y**2) + 0.05 * x

def integrate_function():
    pass

if __name__ == "__main__":
   

    # Generate random samples from a bivariate normal distribution
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    
    # Comput the eigendecomposition of the covariance matrix
    eigvals, eigvecs = np.linalg.eig(cov)
    print("Eigenvalues:", eigvals)
    print("Eigenvectors:\n", eigvecs)
 
    samples = multivariate_normal.rvs(mean=mean, cov=cov, size=10000)
    x_samples, y_samples = samples[:, 0], samples[:, 1]
    x_min, x_max = np.min(x_samples), np.max(x_samples)
    y_min, y_max = np.min(y_samples), np.max(y_samples)
    x, y =  np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max,100)
    xx, yy = np.meshgrid(x,y)

    # Create a joint plot with marginal distributions
    sns.jointplot(x=x_samples, y=y_samples, height=5, kind='kde', cmap='Blues', fill=True, thresh=0,alpha=0.5)
    plt.contour(xx, yy, f3(xx, yy), levels=30, colors='gray')
    l = lambda x: np.ones(x.shape)*mean[0]
    #l = lambda x: mean[1] + eigvecs[1,0] * (x - mean[0]) / eigvecs[0,0]
    plt.plot(x, l(x), color='red', linestyle='--', label='Subspace')   
    plt.plot([],[], color='gray', label=r'$f(y_1,y_2)$')
    plt.legend()
    plt.xlabel(r'$Y_1$')  
    plt.ylabel(r'$Y_2$')
    plt.grid()
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.tight_layout()
    filename = '/home/hans-werner/Dropbox/work/research/projects/spatially_indexed_noise/'+\
               'phd_notes/fig/ex0c_jointplot.png'
    plt.savefig(filename, dpi=300)
    #plt.show()



    # Evaluate the function at the sampled points
    fig, ax = plt.subplots(figsize=(5, 5))
    Z_samples1 = f3(x_samples, y_samples)
    Z_samples2 = f3(x_samples, l(x_samples))
    
    sns.kdeplot(Z_samples1, label=r'$f(y_1,y_2)$', color='blue', fill=True, alpha=0.5, ax=ax)
    sns.kdeplot(Z_samples2, label=r'$\hat{f}(y_1)$', color='orange', fill=True, alpha=0.5, ax=ax)
    plt.title(r'Density of $Q$')
    plt.xlabel(r'$Q$')
    plt.ylabel(r'$\pi_Q$')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename.replace('jointplot.png', 'qoi.png'), dpi=300)
    plt.show()