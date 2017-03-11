'''
Created on Feb 8, 2017

@author: hans-werner
'''
import scipy.sparse as sparse
from scikits.sparse.cholmod import *
from sklearn import datasets  
class Gmrf(object):
    '''
    Gaussian Markov Random Field
    '''


    def __init__(self, mu, Q):
        """
        Constructor
        """
        self.__Q = Q
        self.__mu = mu
        
    def sample(self):
        """
        
        """
        pass
    
    
    def condition_hard_constraint(self):
        """
        
        """
        pass
        
        
    def condition_soft_constraint(self):
        """
        
        """
        pass
    
    