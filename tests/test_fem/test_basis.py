from fem import QuadFE
from fem import DofHandler
from fem import Basis

from mesh import QuadMesh
from mesh import Mesh1D

import unittest

class TestBasis(unittest.TestCase):
    def test_set(self):
        mesh = QuadMesh(resolution=(1,1))
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        px = Basis(dofhandler, 'ux')
        p  = Basis(dofhandler, 'ux')
        
        p_set = set([px, p])
