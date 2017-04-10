'''
Created on Feb 8, 2017

@author: hans-werner
'''
import scipy.sparse as sp
import pymetis
from scikits.sparse.cholmod import cholesky
import numpy as np
from sklearn import datasets  
from scipy.linalg import decomp_cholesky


class Gmrf(object):
    '''
    Gaussian Markov Random Field
    '''


    def __init__(self, mu=None, precision=None, covariance=None, mesh=None, element=None):
        """
        Constructor
        
        Inputs:
        
            
        """
        #
        # Precision and Covariance
        #
        if precision is not None:
            self.__Q = precision
            if sp.isspmatrix_csr(precision):
                self.__sparse_precision = True
                self.__L = cholesky(precision)
            else:
                self.__sparse_precision = False
                self.__L = np.linalg.cholesky(precision)
        
        if covariance is not None:
            self.__Sigma = covariance
            self.__Lcov  = np.linalg.cholesky(covariance)    
        
        if covariance is not None and precision is not None:
            #
            # Both covariance and precision specified - check compatibility
            #
            n_pre = precision.shape[0]
            n_cov = covariance.shape[0]
            assert n_pre == n_cov, \
                'Incompatibly shaped precision and covariance.'
                
            assert np.allclose(np.dot(covariance, precision),\
                               np.eye(n_pre),rtol=1e-10),\
               'Covariance and precision are not inverses.' 
            
            
        self.__mu = mu
        self.mesh = mesh
        self.element = element
       
    
    
    def Q(self):
        """
        Return the precision matrix
        """
        return self.__Q
    
    
    def Sigma(self):
        """
        Return the covariance matrix
        """
        pass
        
        
    def mu(self):
        """
        Return the mean of the random vector
        """
        return self.__mu
    
    
    def n(self):
        """
        Return the dimension of the random vector 
        """
        return self.__n
    
    
    def kl_expansion(self, k=None):
        """
        Inputs:
        
        Outputs:
        
        """
        pass
    
    
    def sample(self, ):
        """
        
        """
        pass
    
    
    def condition(self, constraint=None, constraint_type='pointwise'):
        """
        
        Inputs:
        
            constraint: tuple, parameters specifying the constraint, determined
                by the constraint type:
                
                'pointwise': 
                
                'hard':
                
                'soft':
        
            constraint_type: str, 'pointwise' (default), 'hard', 'soft'.
        """
        pass
    
    
    def __condition_hard_constraint(self):
        """
        
        """
        pass
        
        
    def __condition_soft_constraint(self):
        """
        
        """
        pass
    
    