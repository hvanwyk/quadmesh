from mesh import QuadMesh
from mesh import Mesh1D

from fem import DofHandler
from fem import QuadFE
from fem import Basis

from function import Nodal
from function import Constant 
from function import Explicit

from gmrf import EllipticField

from plot import Plot
import unittest
import numpy as np
import matplotlib.pyplot as plt

class TestEllipticField(unittest.TestCase):
    def test_constructor(self):
        #
        # Define mesh, element, and dofhandler 
        #
        mesh = QuadMesh(box=[0,20,0,20], resolution=(20,20), periodic={0,1})
        dim = mesh.dim()
        element = QuadFE(dim, 'Q3')
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        basis = Basis(dofhandler,'u')
        
        alph = 2
        kppa = 1
        
        # Symmetric tensor gma T + bta* vv^T
        gma = 0.1
        bta = 25
        
        p = lambda x: 10/np.pi*(0.75*np.sin(np.pi*x[:,0]/10)+\
                                0.25*np.sin(np.pi*x[:,1]/10))
        f = Nodal(f=p, basis=basis)
        fx = f.differentiate((1,0))
        fy = f.differentiate((1,1))
        
        #plot.contour(f)
        
        x = np.linspace(0,20,12)
        X,Y = np.meshgrid(x,x)
        xy = np.array([X.ravel(),Y.ravel()]).T
        U = fx.eval(xy).reshape(X.shape)
        V = fy.eval(xy).reshape(X.shape)
            
        v1 = lambda x: -0.25*np.cos(np.pi*x[:,1]/10)
        v2 = lambda x:  0.75*np.cos(np.pi*x[:,0]/10)
        
        U = v1(xy).reshape(X.shape)
        V = v2(xy).reshape(X.shape)
        
        #plt.quiver(X,Y, U, V)
        #plt.show()
        
        h11 = Explicit(lambda x: gma + bta*v1(x)*v1(x), dim=2)
        h12 = Explicit(lambda x: bta*v1(x)*v2(x), dim=2)
        h22 = Explicit(lambda x: gma + bta*v2(x)*v2(x), dim=2)
        
        tau = (h11, h12, h22)
        
        #tau = (Constant(2), Constant(1), Constant(1))
        #
        # Define default elliptic field 
        # 
        u = EllipticField(dofhandler, kappa=1, tau=tau, gamma=2)
        Q = u.precision()
        v = Nodal(data=u.sample(mode='precision', decomposition='chol'), basis=basis)
        
        
        plot = Plot(20)
        plot.contour(v)
        
        
    def test_powers(self):
        pass

        
    def test_submesh(self):
        """
        Generate Elliptic field on a submesh
        """
        pass
    
    