from gmrf import CovarianceKernel
from gmrf import Covariance
from mesh import QuadMesh
from mesh import Mesh1D
from fem import QuadFE
from fem import DofHandler
from function import Nodal
from plot import Plot
import matplotlib.pyplot as plt
from gmrf import modchol_ldlt
from scipy import linalg

import numpy as np
from scipy import linalg as la

import unittest

class TestCovariance(unittest.TestCase):
    """
    Test class for covariance
    """        
    def test_constructor(self):
        pass
    
    
    
    
    
    def test_assembly(self):
        for mesh in [Mesh1D(resolution=(10,)), QuadMesh(resolution=(30,30))]:
            
            dim = mesh.dim()
            
            element = QuadFE(dim, 'Q1')
            
            dofhandler = DofHandler(mesh, element)
            cov_kernel = CovarianceKernel('gaussian', {'sgm': 2, 'l': 0.2, 'M': None}, dim)
            
            covariance = Covariance(cov_kernel, dofhandler, method='projection')
            KK = covariance.get_matrix()
            print(KK)
            K = covariance.assembler().af[0]['bilinear'].get_matrix().toarray()
            
            #self.assertTrue(np.allclose(KK,K))
            covariance.eig_decomp()
            d, V = covariance.get_eig_decomp()
            
            U = Nodal(data=covariance.sample(n_samples=1), dofhandler=dofhandler, dim=dim)
            plot = Plot()
            if dim==1:
                plt.imshow(KK)
                #plt.plot(V[:,:10])
                plt.show()
                plot.line(U)
            elif dim==2:
                #pass
                plot.contour(U)
            
            # Solve the generalized svd
            U, s, Vh = la.svd(K)
            
            
            n = U.shape[0]
            n_sample = 1
            Z = np.random.normal(size=(n,n_sample))
            
            Y = U.dot(np.diag(np.sqrt(s)).dot(Z))
            
            y = Nodal(data=Y, dofhandler=dofhandler, dim=dim)
            
            print(Y.shape)
            
            plot = Plot()
            
            if dim==1:
                #pass
                plot.line(y)
            elif dim==2:
                #pass
                plot.contour(y)
            