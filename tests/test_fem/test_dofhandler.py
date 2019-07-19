import unittest
from fem import DofHandler, QuadFE
from mesh import Mesh1D, Mesh2D, QuadMesh
import numpy as np
from plot import Plot
from diagnostics import Verbose
    
class TestDofHandler(unittest.TestCase):
    """
    Test dofhandler
    """
    def test_clear_dofs(self):
        # Define new dofhandler
        mesh = Mesh2D(resolution=(1,1))
        cell = mesh.cells.get_child(0)
        element = QuadFE(2, 'Q2')
        dofhandler = DofHandler(mesh, element)
        # Fill dofs
        dofhandler.fill_dofs(cell)
        
        # Clear dofs 
        dofhandler.clear_dofs()
        
        # Check that there are no dofs
        self.assertIsNone(dofhandler.get_cell_dofs(cell))
    
    
    def test_assign_get_global_dofs(self):
        # 
        # Assigning Dofs to entities within cell
        #  
        # =====================================================================
        # 1D
        # =====================================================================
        mesh = Mesh1D(resolution=(1,))
        cell = mesh.cells.get_child(0)
        etypes = ['DQ0', 'DQ1', 'DQ2', 'DQ3']
        #
        # Expected Dofs
        # 
        edofs = dict.fromkeys(etypes)
        edofs['DQ0'] = {'all': [0], 'pos': ([0], [2]), 'vertex': None, 'edge': None}
        edofs['DQ1'] = {'all': [0,1], 'pos': ([1], [3]), 'vertex': (1, [1]), 'edge': None}
        edofs['DQ2'] = {'all': [0,1,2], 'pos': ([2], [6]), 'vertex': (1, [1]), 'edge': [3]}
        edofs['DQ3'] = {'all': [0,1,2,3], 'pos': ([3], [6]), 'vertex': (1, [1]), 'edge': [3,4]}
        for etype in etypes:
            #
            # New Dofhandler
            # 
            element = QuadFE(1,etype)
            dh = DofHandler(mesh, element)
            
            for key in edofs[etype].keys():
                #
                # Assign all dofs
                # 
                if key=='all':
                    dh.assign_dofs(edofs[etype][key], cell)
                    self.assertEqual(dh.get_cell_dofs(cell), edofs[etype][key])
                #
                # Assign dofs at given positions
                # 
                elif key=='pos':
                    pos, dofs = edofs[etype]['pos']
                    dh.assign_dofs(dofs, cell, pos=pos)
                    self.assertEqual(dh.get_cell_dofs(cell)[pos[0]], dofs[0])
                #
                # Assign dofs to given vertex
                #  
                elif key=='vertex':
                    if edofs[etype]['vertex'] is not None:
                        v_num, dofs = edofs[etype]['vertex']
                        vertex = cell.get_vertex(v_num)
                        dh.assign_dofs(dofs, cell, vertex)
                        self.assertEqual(dh.get_cell_dofs(cell, vertex), dofs)
                #
                # Assign dofs to interval
                # 
                elif key=='edge':
                    if edofs[etype][key] is not None:
                        dofs = edofs[etype][key]
                        dh.assign_dofs(dofs, cell, cell)
                        self.assertEqual(dh.get_cell_dofs(cell, cell, interior=True), dofs)
                #
                # Clear dofs to reuse DofHandler
                # 
                dh.clear_dofs()
            
        # =====================================================================    
        # 2D
        # =====================================================================
        # New mesh
        mesh = Mesh2D(resolution=(1,1))
        cell = mesh.cells.get_child(0)
        etypes = ['DQ0', 'Q1', 'Q2', 'Q3']
        #
        # List of dofs to assign and check
        # 
        edofs = dict.fromkeys(etypes)
        edofs['DQ0'] = {'all': [0], 'pos': ([0],[2]) , 
                        'vertex': None, 'edge': None, 'cell': [0] }
        edofs['Q1'] = {'all': [0,1,2,3], 'pos': ([1],[2]), 
                       'vertex': (1,[2]), 'edge': None, 'cell': None }
        edofs['Q2'] = {'all': [0,1,2,3,4,5,6,7,8], 'pos': ([5],[7]) , 
                       'vertex': (3,[4]), 'edge': (2, [1]), 'cell': [1] }
        edofs['Q3'] = {'all': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 'pos': ([11],[111]), 
                       'vertex': (3,[1]), 'edge': (3,[2,3]), 'cell': [1,2,3,4]}
        
        for etype in etypes:
            # New element adn Dofhandler
            element = QuadFE(2, etype)
            dh = DofHandler(mesh, element)
            for key in edofs[etype].keys():
                #
                # Assign all dofs
                # 
                if key=='all':
                    dh.assign_dofs(edofs[etype][key], cell)
                    self.assertEqual(dh.get_cell_dofs(cell), edofs[etype][key])
                #
                # Assign dofs to specific position
                # 
                elif key=='pos':
                    pos, dofs = edofs[etype][key]
                    dh.assign_dofs(dofs, cell, pos=pos)
                    self.assertEqual(dh.get_cell_dofs(cell)[pos[0]],dofs[0])
                #
                # Assign dof to specific vertex
                #
                elif key=='vertex':
                    if edofs[etype][key] is not None:
                        v_num, dofs = edofs[etype][key]
                        vertex = cell.get_vertex(v_num)
                        dh.assign_dofs(dofs, cell, vertex)
                        self.assertEqual(dh.get_cell_dofs(cell, vertex), dofs)
                #
                # Assign dofs to specific edge
                # 
                elif key=='edge':
                    if edofs[etype][key] is not None:
                        e_num, dofs = edofs[etype][key]
                        he = cell.get_half_edge(e_num)
                        dh.assign_dofs(dofs, cell, he)
                        self.assertEqual(dh.get_cell_dofs(cell, he, interior=True),dofs)
                #
                # Assign dofs to cell interior
                # 
                elif key=='cell':
                    if edofs[etype][key] is not None:
                        dofs = edofs[etype][key]
                        dh.assign_dofs(dofs, cell, cell)
                        self.assertEqual(dh.get_cell_dofs(cell, cell, interior=True), dofs)
                
                #
                # Clear dofs to re-use DofHandler
                # 
                dh.clear_dofs()
            

      
    def test_get_local_dofs(self):
        """
        Extract local dofs from a corner vertex, halfEdge or cell
        """      
        local_dofs = {1: {'DQ0': [[0], [], [0]], 
                          'DQ1': [[0,1], [1], []], 
                          'DQ2': [[0,1,2], [1], [2]],
                          'DQ3': [[0,1,2,3], [1], [2,3]]
                          }, 
                      2: {'DQ0': [[0], [], [], [0]],
                          'DQ1': [[0,1,2,3], [1], [], []],
                          'DQ2': [[0,1,2,3,4,5,6,7,8], [1], [6], [8]],
                          'DQ3': [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [1], [8,9], [12,13,14,15]]
                          }
                      } 
        etypes = ['DQ' + i for i in '0123']        
        for dim in range(1,3):
            if dim==1:
                mesh = Mesh1D(box=[2,4], resolution=(1,))
                cell = mesh.cells.get_child(0)
                vertex  = cell.get_vertex(1)
                entities = [None, vertex, cell]
            elif dim==2:
                mesh = QuadMesh(box = [0,2,0,2], resolution=(2,2))
                cell = mesh.cells.get_child(1)
                vertex = cell.get_vertex(1)
                half_edge = cell.get_half_edge(2)
                entities = [None, vertex, half_edge, cell]
                
            for etype in etypes:
                element = QuadFE(dim, etype)
                dofhandler = DofHandler(mesh, element)
                for i_entity in range(len(entities)):
                    entity = entities[i_entity]
                    dofs = dofhandler.get_cell_dofs(cell, entity=entity,
                                                    doftype='local', interior=True)
                    self.assertEqual(local_dofs[dim][etype][i_entity], dofs)
                
                
    def test_fill_dofs(self):
        etypes = ['DQ0', 'DQ1','DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']
        
        # =====================================================================
        # 1D 
        # =====================================================================
        #
        # Non-periodic
        # 
        mesh = Mesh1D(resolution=(1,))
        expected_dofs = {'DQ0': [0], 
                         'DQ1': [0,1], 
                         'DQ2': [0,1,2], 
                         'DQ3': [0,1,2,3],
                         'Q1': [0,1],
                         'Q2': [0,1,2],
                         'Q3': [0,1,2,3]}
        cell = mesh.cells.get_child(0)
        for etype in etypes:
            element = QuadFE(1, etype) 
            dofhandler = DofHandler(mesh, element)
            dofhandler.fill_dofs(cell)
            dofs = dofhandler.get_cell_dofs(cell)
            self.assertEqual(expected_dofs[etype], dofs)
            
        #
        # Periodic 
        # 
        mesh = Mesh1D(resolution=(1,), periodic=True)
        expected_dofs = {'DQ0': [0], 
                         'DQ1': [0,1], 
                         'DQ2': [0,1,2], 
                         'DQ3': [0,1,2,3],
                         'Q1': [0,0],
                         'Q2': [0,0,1],
                         'Q3': [0,0,1,2]}
        cell = mesh.cells.get_child(0)
        for etype in etypes:
            element = QuadFE(1, etype) 
            dofhandler = DofHandler(mesh, element)
            dofhandler.fill_dofs(cell)
            dofs = dofhandler.get_cell_dofs(cell)
            self.assertEqual(dofs, expected_dofs[etype])
        
        # =====================================================================
        # 2D
        # =====================================================================
        #
        # Non-periodic
        # 
        mesh = Mesh2D(resolution=(1,1))
        cell = mesh.cells.get_child(0)
        expected_dofs = {'DQ0': [0], 
                         'DQ1': [0,1,2,3], 
                         'DQ2': [0,1,2,3,4,5,6,7,8], 
                         'DQ3': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
                         'Q0': [0], 
                         'Q1': [0,1,2,3], 
                         'Q2': [0,1,2,3,4,5,6,7,8], 
                         'Q3': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
        for etype in etypes:
            element = QuadFE(2, etype)
            dofhandler = DofHandler(mesh, element)
            dofhandler.fill_dofs(cell)
            dofs = dofhandler.get_cell_dofs(cell)
            self.assertEqual(dofs, expected_dofs[etype])
            
        #
        # Periodic x direction
        # 
        mesh = Mesh2D(resolution=(1,1), periodic={0})
        cell = mesh.cells.get_child(0)
        expected_dofs = {'DQ0': [0], 
                         'DQ1': [0,1,2,3], 
                         'DQ2': [0,1,2,3,4,5,6,7,8], 
                         'DQ3': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
                         'Q0': [0], 
                         'Q1': [0,0,1,1], 
                         'Q2': [0,0,1,1,2,3,4,3,5], 
                         'Q3': [0,0,1,1,2,3,4,5,6,7,5,4,8,9,10,11]}
        for etype in etypes:
            element = QuadFE(2, etype)
            dofhandler = DofHandler(mesh, element)
            dofhandler.fill_dofs(cell)
            dofs = dofhandler.get_cell_dofs(cell)
            self.assertEqual(dofs, expected_dofs[etype])
            
        #
        # Periodic in both directions
        # 
        mesh = Mesh2D(resolution=(1,1), periodic={0,1})
        cell = mesh.cells.get_child(0)
        expected_dofs = {'DQ0': [0], 
                         'DQ1': [0,1,2,3], 
                         'DQ2': [0,1,2,3,4,5,6,7,8], 
                         'DQ3': [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
                         'Q0': [0], 
                         'Q1': [0,0,0,0], 
                         'Q2': [0,0,0,0,1,2,1,2,3], 
                         'Q3': [0,0,0,0,1,2,3,4,2,1,4,3,5,6,7,8]}
        for etype in etypes:
            element = QuadFE(2, etype)
            dofhandler = DofHandler(mesh, element)
            dofhandler.fill_dofs(cell)
            dofs = dofhandler.get_cell_dofs(cell)
            self.assertEqual(dofs, expected_dofs[etype])
            
    
    def test_share_dofs_with_neighbors(self):
        # =====================================================================
        # 1D 
        # =====================================================================
        #
        # Non-Periodic
        # 
        
        # Define new mesh
        mesh = Mesh1D(resolution=(2,))
        lcell = mesh.cells.get_child(0)
        rcell = mesh.cells.get_child(1)
        etypes = ['DQ0','DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']
        
        # Specify expected dofs
        edofs = dict.fromkeys(etypes)
        edofs['DQ0'] = None
        edofs['DQ1'] = None
        edofs['DQ2'] = None
        edofs['DQ3'] = None
        edofs['Q1'] = [1, None]
        edofs['Q2'] = [1, None, None]
        edofs['Q3'] = [1, None, None, None]
        
        for etype in etypes:
            # New Dofhandler
            element = QuadFE(1,etype)
            dh = DofHandler(mesh, element)
            
            # Fill left dofs and share with right neighbor
            dh.fill_dofs(lcell)
            dh.share_dofs_with_neighbors(lcell, lcell.get_vertex(1))
            self.assertEqual(dh.get_cell_dofs(rcell), edofs[etype])
        
            dh.clear_dofs()
            
            # Fill left dofs and share with all neighbors
            dh.fill_dofs(lcell)
            dh.share_dofs_with_neighbors(lcell)
            self.assertEqual(dh.get_cell_dofs(rcell), edofs[etype])
        
        #
        # Periodic
        #
        
        # Define new mesh
        mesh = Mesh1D(resolution=(2,), periodic=True)
        lcell = mesh.cells.get_child(0)
        rcell = mesh.cells.get_child(1)
        
        # Specify expected dofs
        edofs = dict.fromkeys(etypes)
        edofs['DQ0'] = {'right': None, 'all': None}
        edofs['DQ1'] = {'right': None, 'all': None}
        edofs['DQ2'] = {'right': None, 'all': None}
        edofs['DQ3'] = {'right': None, 'all': None}
        edofs['Q1'] = {'right': [1, None], 'all': [1, 0]}
        edofs['Q2'] = {'right': [1, None, None], 'all': [1, 0, None]}
        edofs['Q3'] = {'right': [1, None, None, None], 
                       'all': [1, 0, None, None]}
        
        for etype in etypes:
            # New Dofhandler
            element = QuadFE(1,etype)
            dh = DofHandler(mesh, element)
            
            # Fill left dofs and share with right neighbor
            dh.fill_dofs(lcell)
            dh.share_dofs_with_neighbors(lcell, lcell.get_vertex(1))
            self.assertEqual(dh.get_cell_dofs(rcell), edofs[etype]['right'])
        
            dh.clear_dofs()
            
            # Fill left dofs and share with all neighbors
            dh.fill_dofs(lcell)
            dh.share_dofs_with_neighbors(lcell)
            self.assertEqual(dh.get_cell_dofs(rcell), edofs[etype]['all'])
        
        # =====================================================================
        # 2D
        # =====================================================================
        #
        # Non-periodic
        # 
        mesh = Mesh2D(resolution=(2,2))
        c00 = mesh.cells.get_child(0)
        c10 = mesh.cells.get_child(1)
        c11 = mesh.cells.get_child(3)
            
        edofs = dict.fromkeys(etypes)
        edofs['DQ0'] = {'vertex': {c10: None, c11: None}, 
                        'edge': {c10: None, c11: None}, 
                        'all': {c10: None, c11: None}}
        edofs['DQ1'] = {'vertex': {c10: None, c11: None}, 
                        'edge': {c10: None, c11: None}, 
                        'all': {c10: None, c11: None}}
        edofs['DQ2'] = {'vertex': {c10: None, c11: None}, 
                        'edge': {c10: None, c11: None}, 
                        'all': {c10: None, c11: None}}
        edofs['DQ3'] = {'vertex': {c10: None, c11: None}, 
                        'edge': {c10: None, c11: None}, 
                        'all': {c10: None, c11: None}}
        edofs['Q1'] = {'vertex': {c10: [2], c11: [2]}, 
                        'edge': {c10: None, c11: None}, 
                        'all': {c10: [1,None, None, 2], 
                                c11: [2, None, None, None]}}
        edofs['Q2'] = {'vertex': {c10: [2], c11: [2]}, 
                        'edge': {c10: [5], c11: None}, 
                        'all': {c10: [1, None, None, 2, None, None, None, 5, None], 
                                c11: [2] + [None]*8}}
        edofs['Q3'] = {'vertex': {c10: [2], c11: [2]}, 
                        'edge': {c10: [7,6], c11: None}, 
                        'all': {c10: [1, None, None, 2, None, None, None, None, 
                                      None, None, 7, 6, None, None, None, None], 
                                c11: [2] + [None]*15}}
        
        for etype in etypes:
            # New element 
            element = QuadFE(2, etype)
            
            # New dofhandler
            dh = DofHandler(mesh, element)
            # ***************************            
            # Share dofs accross vertex 2
            # ***************************
            # Fill 
            dh.fill_dofs(c00)
            
            # Share 
            vertex = c00.get_vertex(2)
            dh.share_dofs_with_neighbors(c00, vertex)
            
            # Check
            self.assertEqual(dh.get_cell_dofs(c10, vertex), edofs[etype]['vertex'][c10])
            self.assertEqual(dh.get_cell_dofs(c11, vertex), edofs[etype]['vertex'][c11])
            
            # Clear
            dh.clear_dofs()

            # *****************************
            # Share dofs across half_edge 1
            # *****************************
            
            # Fill
            dh.fill_dofs(c00)            
            
            # Share
            edge = c00.get_half_edge(1)
            twin = edge.twin()
            dh.share_dofs_with_neighbors(c00, edge)
            
            # Check
            self.assertEqual(dh.get_cell_dofs(c10, twin, interior=True), edofs[etype]['edge'][c10])
            self.assertEqual(dh.get_cell_dofs(c11, twin, interior=True), edofs[etype]['edge'][c11])
            
            # Clear
            dh.clear_dofs()
            
            # *****************************
            # Share dofs with all neighbors
            # *****************************
            
            # Fill
            dh.fill_dofs(c00)
            
            # Share
            dh.share_dofs_with_neighbors(c00)
            
            # Check
            self.assertEqual(dh.get_cell_dofs(c10), edofs[etype]['all'][c10])
            self.assertEqual(dh.get_cell_dofs(c11), edofs[etype]['all'][c11])
            
        #
        # Periodic in both directions   
        # 
        mesh = Mesh2D(resolution=(2,2), periodic={0,1})
        c00 = mesh.cells.get_child(0)
        c10 = mesh.cells.get_child(1)
        c11 = mesh.cells.get_child(3)
        
    
        edofs = dict.fromkeys(etypes)
        edofs['DQ0'] = {'vertex': {c10: None, c11: None}, 
                        'edge': {c10: None, c11: None}, 
                        'all': {c10: None, c11: None}}
        edofs['DQ1'] = {'vertex': {c10: None, c11: None}, 
                        'edge': {c10: None, c11: None}, 
                        'all': {c10: None, c11: None}}
        edofs['DQ2'] = {'vertex': {c10: None, c11: None}, 
                        'edge': {c10: None, c11: None}, 
                        'all': {c10: None, c11: None}}
        edofs['DQ3'] = {'vertex': {c10: None, c11: None}, 
                        'edge': {c10: None, c11: None}, 
                        'all': {c10: None, c11: None}}
        edofs['Q1'] = {'vertex': {c10: [0], c11: [0]}, 
                        'edge': {c10: None, c11: None}, 
                        'all': {c10: [1,0, 3, 2], 
                                c11: [2, 3, 0, 1]}}
        edofs['Q2'] = {'vertex': {c10: [0], c11: [0]}, 
                        'edge': {c10: [7], c11: None}, 
                        'all': {c10: [1, 0, 3, 2, None, 7, None, 5, None], 
                                c11: [2,3,0,1] + [None]*5}}
        edofs['Q3'] = {'vertex': {c10: [0], c11: [0]}, 
                        'edge': {c10: [11,10], c11: None}, 
                        'all': {c10: [1, 0, 3, 2, None, None, 11, 10, 
                                      None, None, 7, 6, None, None, None, None], 
                                c11: [2,3,0,1] + [None]*12}}
        
        for etype in etypes:
            # New element 
            element = QuadFE(2, etype)
            
            # New dofhandler
            dh = DofHandler(mesh, element)
            # ***************************            
            # Share dofs accross vertex 0
            # ***************************
            # Fill 
            dh.fill_dofs(c00)
            
            # Share 
            vertex = c00.get_vertex(0)   
            self.assertTrue(vertex.is_periodic())
            dh.share_dofs_with_neighbors(c00, vertex)
            
            # Check
            c10_vertex = vertex.get_periodic_pair(c10)[0]
            self.assertEqual(dh.get_cell_dofs(c10, c10_vertex), edofs[etype]['vertex'][c10])
            
            c11_vertex = vertex.get_periodic_pair(c11)[0]
            self.assertEqual(dh.get_cell_dofs(c11, c11_vertex), edofs[etype]['vertex'][c11])
            
            # Clear
            dh.clear_dofs()

            # *****************************
            # Share dofs across half_edge 3
            # *****************************
            
            # Fill
            dh.fill_dofs(c00)            
            
            # Share
            edge = c00.get_half_edge(3)
            twin = edge.twin()
            dh.share_dofs_with_neighbors(c00, edge)
            
            # Check
            self.assertEqual(dh.get_cell_dofs(c10, twin, interior=True), edofs[etype]['edge'][c10])
            self.assertEqual(dh.get_cell_dofs(c11, twin, interior=True), edofs[etype]['edge'][c11])
            
            # Clear
            dh.clear_dofs()
            
            # *****************************
            # Share dofs with all neighbors
            # *****************************
            
            # Fill
            dh.fill_dofs(c00)
            
            # Share            
            dh.share_dofs_with_neighbors(c00)
            
            # Check
            self.assertEqual(dh.get_cell_dofs(c10), edofs[etype]['all'][c10])
            self.assertEqual(dh.get_cell_dofs(c11), edofs[etype]['all'][c11])
            
        
    def test_share_dofs_with_children(self):
        #
        # 1D Test
        # 
        mesh = Mesh1D(resolution=(1,))
        mesh.cells.refine()
        for etype in ['DQ0','DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']:
            # New DofHandler
            element = QuadFE(1, etype)
            dh = DofHandler(mesh, element)
            
            # Fill in parent dofs and share with children
            cell = mesh.cells.get_child(0)
            dh.fill_dofs(cell)
            dh.share_dofs_with_children(cell)
            
            # Expected dofs for children
            left_child_dofs = {'DQ0': [None], 
                               'DQ1': [0, None], 
                               'DQ2': [0, 2, None], 
                               'DQ3': [0, None, None, 2], 
                               'Q1': [0, None], 
                               'Q2': [0, 2, None], 
                               'Q3': [0, None, None, 2]}
            right_child_dofs = {'DQ0': [None],
                                'DQ1': [None, 1],
                                'DQ2': [None, 1, None],
                                'DQ3': [None, 1, 3, None], 
                                'Q1':[None, 1], 
                                'Q2': [2, 1, None],
                                'Q3': [None, 1, 3, None]}
            left_child = cell.get_child(0)
            right_child = cell.get_child(1)
            #
            # Check whether shared dofs are as expected.
            # 
            self.assertEqual(dh.get_cell_dofs(left_child), left_child_dofs[etype])
            self.assertEqual(dh.get_cell_dofs(right_child), right_child_dofs[etype])
        #
        # 2D Test
        # 
        mesh = QuadMesh(resolution=(1,1))
        mesh.cells.refine()
        for etype in ['DQ0','DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']:
            
            # New dofhandler
            element = QuadFE(2, etype)
            dh = DofHandler(mesh, element)
            
            # Fill in parent dofs and share with children
            cell = mesh.cells.get_child(0)
            dh.fill_dofs(cell)
            dh.share_dofs_with_children(cell)
            
            # Expected dofs for children
            child_dofs = {0: {'DQ0': [None], 
                               'DQ1': [0, None, None, None], 
                               'DQ2': [0, 4, 8, 7, None, None, None, None, None], 
                               'DQ3': [0, None, None, None, None, 4, None, None, 
                                       None, None, 11, None, None, None, None, 12], 
                               'Q1': [0, None, None, None], 
                               'Q2': [0, 4, 8, 7, None, None, None, None, None], 
                               'Q3': [0, None, None, None, None, 4, None, None, 
                                      None, None, 11, None, None, None, None, 12]}, 
                          1: {'DQ0': [None], 
                               'DQ1': [None, 1, None, None], 
                               'DQ2': [None, 1, 5, None, None, None, None, None, None], 
                               'DQ3': [None, 1, None, None, 5, None, None, 6, 
                                       None, None, None, None, None, None, 13, None], 
                               'Q1': [None, 1, None, None], 
                               'Q2': [4, 1, 5, 8, None, None, None, None, None], 
                               'Q3': [None, 1, None, None, 5, None, None, 6, 
                                      None, None, None, None, None, None, 13, None]},
                          2: {'DQ0': [None], 
                               'DQ1': [None, None, 2, None], 
                               'DQ2': [None, None, 2, 6, None, None, None, None, None], 
                               'DQ3': [None, None, 2, None, None, None, 7, None, 
                                       None, 8, None, None, 15, None, None, None], 
                               'Q1': [None, None, 2, None], 
                               'Q2': [8, 5, 2, 6, None, None, None, None, None], 
                               'Q3': [None, None, 2, None, None, None, 7, None, 
                                       None, 8, None, None, 15, None, None, None]},
                          3: {'DQ0': [None], 
                               'DQ1': [None, None, None, 3], 
                               'DQ2': [None, None, None, 3, None, None, None, None, None], 
                               'DQ3': [None, None, None, 3, None, None, None, 
                                       None, 9, None, None, 10, None, 14, None, None], 
                               'Q1': [None, None, None, 3], 
                               'Q2': [7, 8, 6, 3, None, None, None, None, None], 
                               'Q3': [None, None, None, 3, None, None, None, 
                                       None, 9, None, None, 10, None, 14, None, None]}}
            for i in range(4):
                child = cell.get_child(i)
                #
                # Check whether shared dofs are as expected
                # 
                self.assertEqual(dh.get_cell_dofs(child), child_dofs[i][etype])
       
           
    def test_distribute_dofs(self):
        show_plots = False
        if show_plots:
            plot = Plot()
           
            #
            # Define QuadMesh with hanging node
            # 
            mesh = QuadMesh(resolution=(1,1), periodic={0,1})        
            mesh.cells.refine()
            mesh.cells.get_child(0).get_child(0).mark(flag=0)                
            mesh.cells.refine(refinement_flag=0) 
            etypes = ['DQ0','DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3'] 
            for etype in etypes:
                # Define new element
                element = QuadFE(2,etype)
        
                # Distribute dofs
                dofhandler = DofHandler(mesh, element)
                dofhandler.distribute_dofs()
    
                plot.mesh(mesh, dofhandler=dofhandler, dofs=True)
            
    
    def test_n_dofs(self):
        """
        Check that the total number of dofs is correct
        
        NOTE: A mesh with multiple levels has dofs on coarser levels that may not appear in leaves
        """    
        etypes = ['DQ0', 'DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']
        #
        # Single cell
        #
        n_dofs = dict.fromkeys([0,1])
        n_dofs[0] = {'DQ0': 1, 'DQ1': 2, 'DQ2': 3, 'DQ3': 4, 'Q1': 2, 'Q2': 3, 'Q3': 4}
        n_dofs[1] = {'DQ0': 1, 'DQ1': 4, 'DQ2': 9, 'DQ3': 16, 'Q1': 4, 'Q2': 9, 'Q3':16}
        for dim in range(2):
            if dim==0:
                mesh = Mesh1D()
            elif dim==1:
                mesh = QuadMesh()
        
            for etype in etypes:
                element = QuadFE(dim+1, etype)
                dofhandler = DofHandler(mesh, element)
                dofhandler.distribute_dofs()
                self.assertEqual(n_dofs[dim][etype], dofhandler.n_dofs())
            
        #
        # Mesh with multiple cells
        #     
        n_dofs = dict.fromkeys([0,1])
        n_dofs[0] = {'DQ0': 2, 'DQ1': 4, 'DQ2': 6, 'DQ3': 8, 'Q1': 3, 'Q2': 5, 'Q3': 7}
        n_dofs[1] = {'DQ0': 4, 'DQ1': 16, 'DQ2': 36, 'DQ3': 64, 'Q1': 9, 'Q2': 25, 'Q3': 49}
        for dim in range(2):
            if dim==0:
                mesh = Mesh1D(resolution=(2,))
            elif dim==1:
                mesh = QuadMesh(resolution=(2,2))
            
            for etype in etypes:
                element = QuadFE(dim+1, etype)
                dofhandler = DofHandler(mesh, element)                
                dofhandler.distribute_dofs()
                self.assertEqual(n_dofs[dim][etype], dofhandler.n_dofs())

        
    def test_set_hanging_nodes(self):
        """
        Check that functions in the finite element space can be interpolated
        by linear combinations of shape functions at supporting nodes.
        
        TODO: Move this test to tests for system
        """        
        #
        # Define QuadMesh with hanging node
        # 
        mesh = QuadMesh(resolution=(1,1))        
        mesh.cells.refine()
        mesh.cells.get_child(0).get_child(0).mark(flag=0)                
        mesh.cells.refine(refinement_flag=0)
        
        c_00 = mesh.cells.get_child(0).get_child(0)
        #
        # Define test functions to interpolate
        # 
        test_functions = {'Q1': lambda x,y: x + y, \
                          'Q2': lambda x,y: x**2 + y**2,\
                          'Q3': lambda x,y: x**3*y + y**2*x**2}
        etypes = ['Q1', 'Q2', 'Q3']
        plot = Plot()
        for etype in etypes:
            #print(etype)
            # Define new element
            element = QuadFE(2,etype)
        
            # Distribute dofs and set vertices
            dofhandler = DofHandler(mesh, element)
            dofhandler.distribute_dofs()
            dofhandler.set_dof_vertices()
            
            # Determine hanging nodes
            dofhandler.set_hanging_nodes()
            hanging_nodes = dofhandler.get_hanging_nodes()
            for dof, support in hanging_nodes.items():
                # Get hanging node vertex
                x_hgnd = dofhandler.get_dof_vertices(dof)

                # Extract support indices and weights
                js, ws = support 
                
                # Extract
                dofs_glb = dofhandler.get_cell_dofs(c_00)
                
                #print(dof, js)
                
                # Local dof numbers for supporting nodes
                dofs_loc_supp = [i for i in range(element.n_dofs()) if dofs_glb[i] in js]
                #x_dofs = c_00.reference_map(element.reference_nodes())
                
                phi_supp = element.shape(x_hgnd, cell=c_00, local_dofs=dofs_loc_supp)
                #print(phi_supp, js)
                
                # Evaluate test function at hanging node 
                #f_hgnd = test_functions[etype](x_hgnd[0],x_hgnd[1])

                #print('Weighted sum of support function', np.dot(phi_supp,ws))
                
                #print(f_hgnd - np.dot(phi_supp, ws))
                #phi_hgnd = element.shape(x_dofs, cell=c_00, local_dofs=dofs_loc_hgnd)
                
                #print(phi_supp)
                #print(phi_hgnd)
            #plot.mesh(mesh, dofhandler=dofhandler, dofs=True)
            
            # Evaluate 
            c_01 = mesh.cells.get_child(0).get_child(1)
            c_022 = mesh.cells.get_child(0).get_child(2).get_child(2)
            #print(dofhandler.get_global_dofs(c_022))
            x_ref = element.reference_nodes()
            #print(dofhandler.get_global_dofs(c_01))
            #print(dofhandler.get_hanging_nodes())
            x = c_01.reference_map(x_ref)
            
            #plot.mesh(mesh, dofhandler=dofhandler, dofs=True)

    def test_get_region_dofs(self):
        """
        Test the function for returning the dofs associated with a region.
        """   
        # 
        # 2D
        #
        mesh = QuadMesh()
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2, etype)
            dofhandler = DofHandler(mesh, element)
            dofhandler.distribute_dofs()
            
            # 
            # Mark half-edges 
            #  
            bnd_right = lambda x,dummy: np.abs(x-1)<1e-9
            mesh.mark_region('right', bnd_right, \
                             entity_type='half_edge', \
                             on_boundary=True)
            
            # Check that mesh.mark_region is doing the right thing. 
            cell = mesh.cells.get_child(0)
            marked_edge = False
            for he in cell.get_half_edges():
                if he.is_marked('right'):
                    #
                    # All vertices should be on the boundary
                    # 
                    marked_edge = True
                    for v in he.get_vertices():
                        x,y = v.coordinates()
                    self.assertTrue(bnd_right(x,y))
                else:
                    #
                    # Not all vertices on should be on the boundary
                    # 
                    on_right = True
                    for v in he.get_vertices():
                        x,y = v.coordinates()
                        if not bnd_right(x,y):
                            on_right = False
                    self.assertFalse(on_right)
            #
            # Some half-edge should be marked
            # 
            self.assertTrue(marked_edge)
            
            #
            # Check that we get the right number of dofs
            #
            n_dofs = {True: {'Q1': 0, 'Q2': 1, 'Q3': 2}, 
                      False: {'Q1': 2, 'Q2': 3, 'Q3': 4}}
            for interior in [True, False]:
                dofs = dofhandler.get_region_dofs(entity_type='half_edge', \
                                                  entity_flag='right', \
                                                  interior=interior, \
                                                  on_boundary=True)
                #
                # Check that we get the right number of dofs
                # 
                self.assertEqual(len(dofs), n_dofs[interior][etype])
                
    
    def test_timings(self):
        """
        """
        comment = Verbose()
        mesh = QuadMesh()
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh, element)
        for dummy in range(9):
            mesh.cells.refine()
            comment.tic()
            dofhandler.distribute_dofs()
            comment.toc()
            print(dofhandler.n_dofs())
            
