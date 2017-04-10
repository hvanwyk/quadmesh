'''
Created on Feb 8, 2017

@author: hans-werner
'''
from finite_element import System, QuadFE
from mesh import Mesh
import scipy.sparse as sp
#import pymetis
from scikits.sparse.cholmod import cholesky
import numpy as np
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
    
    
    def matern_precision(self, mesh, element, alpha, kappa):
        """
        Return the precision matrix for the Matern random field defined on the 
        spatial mesh. The field X satisfies
        
            (k^2 - Delta)^{a/2} X = W
        
        Inputs: 
        
            mesh: Mesh, finite element mesh on which the field is defined
            
            element: QuadFE, finite element space of piecewise polynomials
            
            alpha: int, positive integer (doubles not yet implemented).
            
            kappa: double, positive regularization parameter.
            
            
        Outputs:
        
            Q: sparse matrix, in CSC format
        """
        system = System(mesh, element)
        
        #
        # Assemble (kappa * I + K)
        # 
        G = system.assemble(bilinear_forms=[(kappa,'u','v'),(1,'ux','vx'),(1,'uy','vy')])
        G = G.tocsr()
        
        #
        # Lumped mass matrix
        # 
        M = system.assemble(bilinear_forms=[(1,'u','v')]).tocsr()
        M_lumped_inv = sp.diags(1/np.array(M.sum(axis=1)).squeeze())
        
        
        #Ml = sp.diags(Ml)
        if np.mod(alpha,2) == 0:
            #
            # Even power alpha
            # 
            Q = G
            count = 1
        else:
            #
            # Odd power alpha
            # 
            Q = G.dot(M_lumped_inv.dot(G))
            count = 2
        while count < alpha:
            Q = G.dot(M_lumped_inv.dot(Q.dot(M_lumped_inv.dot(G))))
            count += 2
        
        return Q