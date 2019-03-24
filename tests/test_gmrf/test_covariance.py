from gmrf import CovKernel
from gmrf import Covariance
from mesh import QuadMesh
from mesh import Mesh1D
from fem import QuadFE
from fem import DofHandler
from function import Function
from plot import Plot
import matplotlib.pyplot as plt

import numpy as np

import unittest

class TestCovariance(unittest.TestCase):
    """
    Test class for covariance assembly
    """
    def test_constructor(self):
        pass
    
    def test_assembly(self):
        for mesh in [Mesh1D(resolution=(100,)), QuadMesh(resolution=(10,10))]:
            
            dim = mesh.dim()
            
            element = QuadFE(dim, 'DQ0')
            
            dofhandler = DofHandler(mesh, element)
            cov_kernel = CovKernel('gaussian', {'sgm': 1, 'l': 0.1, 'M': None}, dim)
            
            covariance = Covariance(cov_kernel, dofhandler)