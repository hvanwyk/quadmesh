from gmrf import CovKernel
from mesh import QuadMesh
from mesh import Mesh1D
from fem import QuadFE
from fem import DofHandler
from function import Function
from plot import Plot
import matplotlib.pyplot as plt

import numpy as np

import unittest

class TestCovKernel(unittest.TestCase):

    def test_eval(self):    
        #
        # Out of the box covariance kernels
        #
        fig, ax = plt.subplots(6,4, figsize=(5,7))
        
        cov_names = ['constant', 'linear', 'gaussian', 
                     'exponential', 'matern', 'rational']
        
        anisotropies = {1: [None, 2], 2: [None, np.diag([2,1])]}
        m_count = 0
        for mesh in [Mesh1D(resolution=(10,)), QuadMesh(resolution=(10,10))]:
            # Dimension
            dim = mesh.dim()

            #
            # Construct computational mesh
            # 
            # Piecewise constant elements
            element = QuadFE(dim,'DQ0')
            
            # Define dofhandler -> get vertices
            dofhandler = DofHandler(mesh, element)
            dofhandler.distribute_dofs()
            dofhandler.set_dof_vertices()
            v = dofhandler.get_dof_vertices()
            
            # Define meshgrid for 1 and 2 dimensions
            n_dofs = dofhandler.n_dofs()
            M1,M2 = np.mgrid[0:n_dofs,0:n_dofs]
            if dim == 1:
                X = v[:,0][M1].ravel()
                Y = v[:,0][M2].ravel()
            elif dim == 2:
                X = np.array([v[:,0][M1].ravel(), v[:,1][M1].ravel()]).T
                Y = np.array([v[:,0][M2].ravel(), v[:,1][M2].ravel()]).T
            
            a_count = 0
            isotropic_label = ['isotropic','anisotropic']
            for M in anisotropies[dim]:
                # Cycle through anisotropies
                
                # Define covariance parameters
                cov_pars = {'constant': {'sgm':1},   
                            'linear': {'sgm': 1, 'M': M}, 
                            'gaussian': {'sgm': 1, 'l': 0.1, 'M': M}, 
                            'exponential': {'l': 0.1, 'M': M}, 
                            'matern': {'sgm': 1, 'nu': 2, 'l': 0.5, 'M': M}, 
                            'rational': {'a': 3, 'M': M}}
                
                
               
        
                c_count = 0
                for cov_name in cov_names:
                    
                    C = CovKernel(cov_name, cov_pars[cov_name])
                    Z = C.eval(X,Y).reshape(M1.shape)
                    
                    col = int(m_count*2**1 + a_count*2**0)
                    row = c_count
                    ax[row, col].imshow(Z)
                    if col==0:
                        ax[row,col].set_ylabel(cov_name)
                    
                    if row==0:
                        ax[row,col].set_title('%dD mesh\n %s'%(dim, isotropic_label[a_count]))
                        
                    ax[row, col].set_xticks([],[])
                    ax[row, col].set_yticks([],[])
                                        
                    c_count += 1
                a_count += 1
            m_count += 1
        fig.savefig('test_covkernel_eval.eps')
    
    
    
    def test_slice(self):
        """
        Test the slice method
        """
        fig, ax = plt.subplots(6,4, figsize=(5,7))
        plot = Plot(quickview=False)
        
        cov_names = ['constant', 'linear', 'gaussian', 
                     'exponential', 'matern', 'rational']
        
        anisotropies = {1: [None, 2], 2: [None, np.diag([2,1])]}
        m_count = 0
        for mesh in [Mesh1D(resolution=(20,)), QuadMesh(resolution=(20,20))]:
            
            # Dimension
            dim = mesh.dim()

            element = QuadFE(dim, 'Q2')

            # Get midpoint
            if dim==1:
                # 1D
                x0 = 0.5
            elif dim==2:
                # 2D
                x0 = np.array([0.5,0.5])  
            
            a_count = 0
            isotropic_label = ['isotropic','anisotropic']
            for M in anisotropies[dim]:
                # Cycle through anisotropies
                
                # Define covariance parameters
                cov_pars = {'constant': {'sgm':1},   
                            'linear': {'sgm': 1, 'M': M}, 
                            'gaussian': {'sgm': 1, 'l': 0.1, 'M': M}, 
                            'exponential': {'l': 0.1, 'M': M}, 
                            'matern': {'sgm': 1, 'nu': 2, 'l': 0.5, 'M': M}, 
                            'rational': {'a': 3, 'M': M}}   
        
                c_count = 0
                for cov_name in cov_names:
                    
                    C = CovKernel(cov_name, cov_pars[cov_name], dim=dim)
                    Cs = C.slice(x0, pos=0)
                    
                    f = Function(Cs, 'explicit', mesh=mesh, element=element)
                
                    col = int(m_count*2**1 + a_count*2**0)
                    row = c_count
                    
                    
                    if col==0:
                        ax[row,col].set_ylabel(cov_name)
                    
                    if row==0:
                        ax[row,col].set_title('%dD mesh\n %s'%(dim, isotropic_label[a_count]))
                    if dim==1:
                        ax[row, col] = plot.line(f, axis=ax[row, col])
                    elif dim==2:
                        
                        ax[row, col] = plot.contour(f, axis=ax[row, col], colorbar=False) 
                           
                    ax[row, col].set_xticks([],[])
                    ax[row, col].set_yticks([],[])
                    
                        
                    c_count += 1
                a_count += 1
            m_count += 1
        fig.savefig('test_covkernel_slice.eps')
        
       