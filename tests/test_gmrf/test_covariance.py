from gmrf import CovKernel
from mesh import QuadMesh
from mesh import Mesh1D
from fem import QuadFE
from fem import DofHandler
from function import Nodal
from plot import Plot
import matplotlib.pyplot as plt
from gmrf import modchol_ldlt, Covariance
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
        pass           