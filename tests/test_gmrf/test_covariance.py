from gmrf import CovKernel
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
    Test class for covariance assembly
    """
    def test_modchol_ldlt(self):
        A = np.array([[1, 1, 0, 1],
                      [1, 1, 1, 0],
                      [0, 1, 1, 1],
                      [1, 0, 1, 1]])
        
        # A = randn(4); A = A + A';  % Or try random symemtrix A.
        
        A_eigs,dummy = linalg.eigh(A) # Check definitness of A.
        print('Eigenvalues of A', A_eigs)
        
        L, D, P, D0 = modchol_ldlt(A) 
        print('L',L)
        print('D',D)
        print('P',P)
        print('D0',D0)
        
        print('L*D0*L.t', L.dot(D0.dot(L.T)))
        print(P.dot(A.dot(P.T)) - L.dot(D0.dot(L.T)))
        residual = linalg.norm(P.dot(A.dot(P.T)) - L.dot(D0.dot(L.T)),1)/linalg.norm(A,1) # Should be order rho*eps.
        
        print('Residual', residual)
        A_pert = P.T.dot(L.dot(D.dot(L.T.dot(P))))     # Modified matrix: symmetric pos def.
        A_pert_eigs, dummy = linalg.eigh(A_pert)  # Should all be >= sqrt(eps)*norm(A,'fro').
        print('perturbed eigenvalues', A_pert_eigs)
        
        #L1, D1, P1, D01, rho = modchol_ldlt_m(A);
        #rel_diffs = [
        #norm(L-L1,1)/norm(L,1)
        #norm(D-D1,1)/norm(D,1)
        #norm(P-P1,1)/norm(P,1)
        #norm(D0-D01,1)/norm(D0,1)];
        
        #fprintf('Max relative difference between matrices computed by\n')
        #fprintf('modchol_ldlt and modchol_ldlt_m is %g\n', max(rel_diffs))
        
        A = np.array([[1,2,1],[2,0,1],[1,1,1]])
        L,P,D,D0 = modchol_ldlt(A) 
        #print(L,P,D,D0)
        
    def test_constructor(self):
        pass
    
    
    def test_assembly(self):
        for mesh in [Mesh1D(resolution=(100,)), QuadMesh(resolution=(40,40))]:
            
            dim = mesh.dim()
            
            element = QuadFE(dim, 'Q1')
            
            dofhandler = DofHandler(mesh, element)
            cov_kernel = CovKernel('gaussian', {'sgm': 2, 'l': 0.2, 'M': None}, dim)
            
            covariance = Covariance(cov_kernel, dofhandler, method='interpolation')
            
            
            K = covariance.assembler().af[0]['bilinear'].get_matrix().toarray()
            
            covariance.compute_svd()
         
            U = Nodal(data=covariance.sample(n_samples=1), dofhandler=dofhandler, dim=dim)
            plot = Plot()
            if dim==1:
                plot.line(U)
            elif dim==2:
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
                plot.line(y)
            elif dim==2:
                plot.contour(y)
            