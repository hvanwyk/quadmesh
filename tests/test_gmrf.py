'''
Created on Mar 11, 2017

@author: hans-werner
'''
from __future__ import division, absolute_import
import unittest
import numpy as np
from gmrf import Gmrf
import pymetis
from six.moves import range
import matplotlib.pyplot as plt

class TestGmrf(unittest.TestCase):


    def test_constructor(self):
        Q = np.array([[6,-1,0,-1],[-1,6,-1,0],[0,-1,6,-1],[-1,0,-1,6]])
        S = np.linalg.inv(Q)
        mu = np.zeros(4)
        X = Gmrf(mu=mu, precision=Q, covariance=S)
        X.Q()

    def test_sample(self):
        import scipy.sparse

        #F = scipy.sparse.rand(100, 100, density=0.02)
        #M = F.transpose() * F
        #plt.spy(M, markersize=0.5)
        #adjacency_list = [M.getrow(i).indices for i in range(M.shape[0])]
        #node_nd = pymetis.nested_dissection(adjacency=adjacency_list)
        #perm, iperm = np.array(node_nd[0]), np.array(node_nd[1])
        #plt.show()
        #assert np.all(perm[iperm] == np.array(range(perm.size)))
    
    
        
    
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()