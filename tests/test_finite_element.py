"""
Created 11/22/2016
@author: hans-werner
"""
import unittest
from finite_element import FiniteElement, QuadFE, TriFE, DofHandler, GaussRule
from mesh import Mesh

class TestFiniteElement(unittest.TestCase):
    """
    Test FiniteElement class
    """
    pass

class TestQuadFE(unittest.TestCase):
    """
    Test QuadFE class
    """
    pass

class TestTriFE(unittest.TestCase):
    """
    Test TriFE classe
    
    """
    pass

class TestDofHandler(unittest.TestCase):
    """
    Test DofHandler class
    """
    def test_constructor(self):
        pass
    
    def test_distribute_dofs(self):
        #
        # Construct Complicated Mesh
        # 
        mesh = Mesh.newmesh()
        mesh.root_node().mark()
        mesh.refine()
        
        mesh.root_node().children['SE'].mark()
        mesh.refine()
        
        mesh.root_node().children['SE'].children['SW'] = None
    
        
    def test_fill_in_dofs(self):
        pass
    
    def test_positions_along_edge(self):
        mesh = Mesh.newmesh()
        element_type = 'Q3'
        V = QuadFE(2,element_type)
        dofhandler = DofHandler(mesh,V)
        direction = 'N'
        positions = dofhandler.positions_along_edge(direction)
        self.assertEqual(positions[1],('N',0),'Position should be (N,0).')
                 
        
    def test_assign_dofs(self):
        pass
       
    def test_get_dofs(self):
        pass
    
    def test_make_hanging_node_constraints(self):
        pass
        
        

class GaussRule(unittest.TestCase):
    """
    Test GaussRule class
    """
    pass
