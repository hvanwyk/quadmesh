from fem import QuadFE
from fem import DofHandler
from fem import Basis

from mesh import QuadMesh

import unittest

class TestBasis(unittest.TestCase):
    def test_set(self):
        mesh = QuadMesh(resolution=(1,1))
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        px = Basis(dofhandler, 'ux')
        p  = Basis(dofhandler, 'ux')
        
        self.assertNotEqual(px, p)
        
    def test_same_dofs(self):
        #
        # Construct nested mesh
        # 
        mesh = QuadMesh()
        mesh.record(0)
        
        for dummy in range(2):
            mesh.cells.refine()
        #
        # Define dofhandler
        #
        element = QuadFE(mesh.dim(),'Q1')
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
    
        #
        # Define basis functions
        #
        phi0 = Basis(dofhandler, 'u', subforest_flag=0)
        phi0_x = Basis(dofhandler, 'ux', subforest_flag=0)
        phi1 = Basis(dofhandler, 'u')
        
        self.assertTrue(phi0.same_mesh(phi0_x))
        self.assertFalse(phi0.same_mesh(phi1))