

from mesh import QuadMesh
from mesh import Mesh1D
from fem import QuadFE
from fem import DofHandler
from fem import Basis

from function import Nodal
from plot import Plot
import matplotlib.pyplot as plt
from gmrf import GaussianField, modchol_ldlt, Covariance
from scipy import linalg

import numpy as np
from scipy import linalg as la

import unittest

import plot

class TestCovariance(unittest.TestCase):
    """
    Test class for covariance
    """        
    def test_constructor(self):
        mesh = Mesh1D(box=[0, 1], resolution=(1,))
        mesh.cells.record(0)
        for i in range(4):
            mesh.cells.refine(new_label=i+1)
        

        DQ0 = QuadFE(1, 'DQ0')
        dh = DofHandler(mesh,DQ0)
        dh.distribute_dofs()
        phi = Basis(dh)

        # Test the constructor 
        cov = Covariance(dh,name='gaussian',parameters={'l':0.1})
        n = cov.get_size()
        eta = GaussianField(n,covariance=cov)
        print('mean:', eta.get_mean())
        xx = eta.sample()

        V = cov.get_eigenvectors()
        

    def test_assembly(self):
        pass           


if __name__ == '__main__':
    unittest.main()