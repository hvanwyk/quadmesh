import unittest
from fem import DofHandler, QuadFE
from mesh import Mesh, Mesh1D, Mesh2D, QuadMesh, QuadCell, DCEL, Vertex, HalfEdge
import numpy as np
from plot import Plot
from mesh import convert_to_array
    
class TestDofHandler(unittest.TestCase):
    """
    Test dofhandler
    """
    def clear_dofs(self):
        pass
    
    
    def test_assign_and_get_global_dofs(self):
        #
        # Assigning Dofs to entities within cell
        # 
        mesh = Mesh2D(resolution=(1,1))
        cell = mesh.cells.get_child(0)
        element_types = ['DQ0', 'Q1', 'Q2', 'Q3']
        cell_dofs = [[0], [], [8], [12,13, 14,15]]
        vertex_dofs = [[], [0,1,2,3], [0,1,2,3], [0,1,2,3]]
        he_dofs = [[], [], [5], [6,7] ]
        i_etype = 0
        for etype in element_types: 
            # =================================================================
            # Assign dofs
            # ================================================================= 
            element = QuadFE(2,etype)
            dofhandler = DofHandler(mesh, element)
            dofs = [i for i in range(element.n_dofs())]
            dofhandler.assign_dofs(dofs, cell)
            # =================================================================
            # Read dofs 
            # ================================================================= 
            #
            # Cell dofs
            #
            cell_dofs_read = dofhandler.get_global_dofs(cell, entity=cell)
            self.assertEqual(cell_dofs_read, cell_dofs[i_etype])
            #
            # Vertex Dofs
            #
            for i_vertex in range(4):
                vertex_dofs_read = \
                    dofhandler.get_global_dofs(cell, \
                    entity=cell.get_vertex(i_vertex))
                if i_etype == 0:
                    self.assertEqual(vertex_dofs_read, [])
                else:
                    self.assertEqual(vertex_dofs_read[0], vertex_dofs[i_etype][i_vertex])
            #
            # HalfEdge Dofs
            #
            i_he = 1
            he = cell.get_half_edge(i_he)
            he_read = dofhandler.get_global_dofs(cell, entity=he)
            if i_etype==0:
                self.assertEqual(he_read, [])
            else:
                self.assertEqual(he_read, he_dofs[i_etype])

            # =================================================================
            # Assign dofs to specific entities
            # =================================================================
            #
            # Vertex 
            #
            if i_etype!=0:
                dofhandler.clear_dofs()
                vertex = cell.get_vertex(0)
                dofhandler.assign_dofs([10], cell, vertex)
                self.assertEqual(dofhandler.get_global_dofs(cell, vertex),[10])
                
                # try to assign another dof
                dofhandler.assign_dofs([20], cell, vertex)
                self.assertEqual(dofhandler.get_global_dofs(cell, vertex),[10])
            #    
            # HalfEdge
            #
            dofhandler.clear_dofs()
            he = cell.get_half_edge(2)
            dofs_he_2 = [[],[],[6], [8,9]]
            dofs_he_dummy = [[],[],[70], [90, 80]]
            dofhandler.assign_dofs(dofs_he_2[i_etype], cell, he)
            he_dofs_read = dofhandler.get_global_dofs(cell, he)
            self.assertEqual(dofs_he_2[i_etype],he_dofs_read)  
            
            # try to assign another set of dofs
            dofhandler.assign_dofs(dofs_he_dummy[i_etype], cell, he)
            he_dofs_read = dofhandler.get_global_dofs(cell, he)
            self.assertEqual(dofs_he_2[i_etype],he_dofs_read)
                      
            #
            # Cell Interior
            # 
            dofhandler.clear_dofs()
            dofhandler.assign_dofs(cell_dofs[i_etype], cell, cell)
            cell_dofs_read = dofhandler.get_global_dofs(cell, cell)
            self.assertEqual(cell_dofs[i_etype], cell_dofs_read)
            
            # try to assign another set of dofs
            cell_dofs_dummy = [[12], [], [10], [122,123, 124,125]]
            dofhandler.assign_dofs(cell_dofs_dummy[i_etype], cell, cell)
            cell_dofs_read = dofhandler.get_global_dofs(cell, cell)
            self.assertEqual(cell_dofs[i_etype], cell_dofs_read)
            
            i_etype += 1
            
    
    def test_fill_dofs(self):
        mesh = Mesh2D(resolution=(2,2))
        cell = mesh.cells.get_child(0)
        element_types = ['DQ0', 'Q1', 'Q2', 'Q3']
        for etype in element_types:
            element = QuadFE(2, etype)
            dofhandler = DofHandler(mesh, element)
            if etype!='DQ0':
                dofhandler.assign_dofs([10], cell, cell.get_vertex(0))
            
            dofhandler.fill_dofs(cell)
            if etype!='DQ0':
                self.assertEqual(dofhandler.n_dofs(),element.n_dofs()-1)
        
    
    def test_share_dofs_with_neighbors(self):
        #
        # Non-periodic
        # 
        for dim in np.arange(1,3):
            if dim==1:
                mesh = Mesh1D(resolution=(1,))
            elif dim==2:
                mesh = QuadMesh(resolution=(1,1))
            mesh.cells.refine()
            cell = mesh.cells.get_child(0).get_child(1)
            for etype in ['DQ0','DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']:
                element = QuadFE(dim, etype)
                d = DofHandler(mesh, element)
                d.fill_dofs(cell)
                d.share_dofs_with_neighbors(cell)
                #
                # None of the torn elements get dofs
                # 
                if d.element.torn_element():
                    if dim==1:
                        child = mesh.cells.get_child(0).get_child(0)
                        self.assertIsNone(d.get_global_dofs(child))
                    elif dim==2:
                        for half_edge in cell.get_half_edges():
                            nb = cell.get_neighbors(half_edge)
                            if nb is not None:
                                nb_dofs = d.get_global_dofs(nb)
                                self.assertIsNone(nb_dofs)
                #
                # 1D
                # 
                if dim==1:  
                    if d.element.element_type() in ['Q1', 'Q2', 'Q3']:
                        child = mesh.cells.get_child(0).get_child(0)
                        dofs = d.get_global_dofs(child)
                        self.assertEqual(dofs[1], 0)
                        self.assertIsNone(dofs[0])
                #
                # 2D
                #
                elif dim==2:
                    #
                    # Q1
                    # 
                    if d.element.element_type()=='Q1':
                        for child in mesh.cells.get_child(0).get_children():
                            if child.get_node_position()==0:
                                dofs = d.get_global_dofs(child)
                                self.assertEqual(dofs[1], 0)
                                self.assertEqual(dofs[2], 3)
                                
                            if child.get_node_position()==2:
                                dofs = d.get_global_dofs(child)
                                self.assertEqual(dofs[0],3)
                                self.assertEqual(dofs[1],2)
                                
                            if child.get_node_position()==3:
                                dofs = d.get_global_dofs(child)
                                self.assertEqual(dofs[1], 3)
                                for i in [0,2,3]:
                                    self.assertIsNone(dofs[i])
                    #
                    # Q2
                    # 
                    if d.element.element_type()=='Q2':
                        for child in mesh.cells.get_child(0).get_children():
                            if child.get_node_position()==0:
                                dofs = d.get_global_dofs(child)
                                self.assertEqual(dofs[1],0)
                                self.assertEqual(dofs[2],3)
                                self.assertEqual(dofs[5],7)                        
                    #
                    # Q3
                    # 
                    if d.element.element_type()=='Q3':
                        #plot = Plot()
                        #plot.mesh(mesh, dofhandler=d, dofs=True)
                        for child in mesh.cells.get_child(0).get_children():
                            if child.get_node_position()==0:
                                dofs = d.get_global_dofs(child)
                                self.assertEqual(dofs[1],0)
                                self.assertEqual(dofs[2],3)
                                self.assertEqual(dofs[6],11)
                                self.assertEqual(dofs[7],10)
                     
        #
        # Periodic in the x-direction
        # 
        
        mesh = QuadMesh(resolution=(2,2), periodic={0})
        cell = mesh.cells.get_child(1)    
        for etype in ['DQ0','DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']:
            element = QuadFE(2, etype)
            d = DofHandler(mesh, element)
            d.fill_dofs(cell)
            d.share_dofs_with_neighbors(cell)
            #
            # Q1
            # 
            if d.element.element_type()=='Q1':
                for child in mesh.cells.get_children():
                    if child.get_node_position()==0:
                        dofs = d.get_global_dofs(child)    
                        self.assertEqual(dofs[0],1)
                        self.assertEqual(dofs[3],2)
                        self.assertEqual(dofs[1],0)
                        self.assertEqual(dofs[2],3)
                        
                    if child.get_node_position()==3:
                        dofs = d.get_global_dofs(child)
                        self.assertEqual(dofs[0],3)
                        self.assertEqual(dofs[1],2)
        
                    if child.get_node_position()==2:
                        dofs = d.get_global_dofs(child)
                        self.assertEqual(dofs[1], 3)
                        for i in [2,3]:
                            self.assertIsNone(dofs[i])
            #
            # Q2
            # 
            if d.element.element_type()=='Q2':
                for child in mesh.cells.get_children():
                    if child.get_node_position()==0:
                        dofs = d.get_global_dofs(child)
                        self.assertEqual(dofs[1],0)
                        self.assertEqual(dofs[2],3)
                        self.assertEqual(dofs[5],7)                        
            #
            # Q3
            # 
            if d.element.element_type()=='Q3':
                for child in mesh.cells.get_children():
                    if child.get_node_position()==0:
                        dofs = d.get_global_dofs(child)
                        self.assertEqual(dofs[1],0)
                        self.assertEqual(dofs[2],3)
                        self.assertEqual(dofs[6],11)
                        self.assertEqual(dofs[7],10)
       
        
    def test_share_dofs_with_children(self):
        mesh = Mesh1D(resolution=(1,))
        mesh.cells.refine()
        for etype in ['DQ0','DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']:
            element = QuadFE(1, etype)
            dh = DofHandler(mesh, element)
            cell = mesh.cells.get_child(0)
            dh.fill_dofs(cell)
            if etype=='Q3':
                assert cell.has_children(), 'Cell should have children'
                cell_dofs = dh.get_global_dofs(cell)
                for child in cell.get_children():
                    dofs = []
                    pos = []
                    i_child = child.get_node_position()
                    for vertex in dh.element.reference_cell().get_dof_vertices(0):
                        if vertex.get_pos(1, i_child) is not None:
                            pos.append(vertex.get_pos(1, i_child))
                            dofs.append(cell_dofs[vertex.get_pos(0)])
                    dh.assign_dofs(dofs, child, pos=pos)
        
        
    def test_distribute_dofs(self):
        mesh = QuadMesh(resolution=(1,1))
        mesh.cells.refine()
        for etype in ['DQ0','DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']:
            element = QuadFE(2, etype)
            dh = DofHandler(mesh, element)
            dh.distribute_dofs()
    
    def test_get_dof_vertices(self):
        pass
    
    def test_set_hanging_nodes(self):
        mesh = QuadMesh(resolution=(1,1))
        element = QuadFE(2,'Q1')
        mesh.cells.refine()
        mesh.cells.get_child(0).get_child(0).mark(0)
        mesh.cells.refine(refinement_flag=0)
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        for cell in mesh.cells.get_leaves():
            print(dofhandler.get_global_dofs(cell))
        plot = Plot()
        plot.mesh(mesh, dofhandler=dofhandler, dofs=True)
        dofhandler.set_hanging_nodes()
        print(dofhandler.get_hanging_nodes())
    
    
    
    '''
    def test_distribute_dofs(self):
        #
        # Mesh
        # 
        mesh = Mesh()
        mesh.refine()
        mesh.root_node().children['SE'].mark(1)
        mesh.refine(1)
        
        etype = 'Q1'
        element = QuadFE(2,etype)
        dofhandler = DofHandler(mesh,element)
        dofhandler.distribute_dofs()
        
        exact_dofs = [[0,1,2,3],[1,4,5,6],[4,7,6,8],[5,6,3,9],
                      [6,8,9,10],[2,3,11,12],[3,10,12,13]]
        
        count = 0
        for leaf in mesh.root_node().get_leaves():
            cell_dofs = dofhandler.get_global_dofs(leaf)
            self.assertEqual(cell_dofs, exact_dofs[count],\
                             'Cell %d dofs do not match given dofs.'%(count))
            count += 1
       
        #
        # Nested version
        # 
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh,element) 
        dofhandler.distribute_dofs()
        dofcount = dofhandler.n_dofs() 
        dofhandler.clear_dofs()
        dofhandler.distribute_dofs(nested=True)
        
        # Can we access the root node's dofs? 
        root_ndofs = len(dofhandler.get_global_dofs(mesh.root_node()))  
        self.assertEqual(root_ndofs, element.n_dofs(),\
                         'Root node dofs incorrect.')
        
        # Do we have the same total number of dofs? 
        self.assertEqual(dofcount, dofhandler.n_dofs(),\
                         'Discrepancy in number of dofs.')
        
        
    def test_share_dofs_with_children(self):
        mesh = Mesh()
        mesh.refine()
        # Expected dofs 
        sw_child_dofs = {'Q1': [0,None,None,None],\
                         'Q2': [0,6,4,8]+[None]*5,\
                         'Q3': [0,None,None,None,\
                                None,4,None,None,None,8,None,\
                                None,None,None,None,12] }
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            dofhandler = DofHandler(mesh,element)
            #
            # Fill in Dofs for parental node
            # 
            node = mesh.root_node()
            dofhandler.fill_dofs(node)
            #
            # Share dofs with children
            #
            dofhandler.share_dofs_with_children(node)
            child = node.children['SW']
            child_dofs = dofhandler.get_global_dofs(child)
            dof_err = 'Dof inheritance incorrect for space %s'%(etype)
            self.assertEqual(child_dofs, sw_child_dofs[etype], dof_err)
            
        # Discontinuous elements
        sw_child_dofs = {'DQ0': None,\
                         'DQ1': [0,None,None,None],\
                         'DQ2': [0,6,4,8]+[None]*5,\
                         'DQ3': [0,None,None,None,\
                                 None,4,None,None,None,8,None,None,\
                                 None,None,None,12] }
        se_child_dofs = {'DQ0': None,\
                         'DQ1': [None,1,None,None],\
                         'DQ2': [None,1,None,5]+[None]*5,\
                         'DQ3': [None,1,None,None,\
                                 None,None,None,6,9,None,None,None,\
                                 None,None,13,None] }
        for etype in ['DQ'+str(i) for i in range(4)]:
            element = QuadFE(2,etype)
            dofhandler = DofHandler(mesh,element)
            #
            # Fill in Dofs for parental node
            # 
            dofhandler.fill_dofs(node)
            #
            # Share dofs with children 
            #
            dofhandler.share_dofs_with_children(node)
            child = node.children['SW']
            child_dofs = dofhandler.get_global_dofs(child)
            dof_err = 'Dof inheritance of SW child incorrect'+\
                      ' for space %s'%(etype)
            self.assertEqual(child_dofs, sw_child_dofs[etype], dof_err)
            
            child = node.children['SE']
            child_dofs = dofhandler.get_global_dofs(child)
            dof_err = 'Dof inheritance of SE child incorrect'+\
                      ' for space %s'%(etype)
            self.assertEqual(child_dofs, se_child_dofs[etype], dof_err)
            
            
    def test_share_dofs_with_neighbors(self):
        #
        # Mesh
        # 
        mesh = Mesh()
        mesh.refine()
        mesh.root_node().children['SE'].mark(1)
        mesh.refine(1) 
        
        #
        # Nodes
        #
        node = mesh.root_node().children['SW']
        n_nbr = node.get_neighbor('N')
        ne_nbr = node.get_neighbor('NE')
        e_nw_nbr = node.get_neighbor('E').children['NW']
        
        dofs_to_check = {'Q1': {'N': [2,3,None,None], 
                                'NE':[3,None,None,None], 
                                'E-NW':[None,None,3,None]},
                         'Q2': {'N': [2,3,None,None,None,None,7,None,None],
                                'NE':[3]+[None]*8,
                                'E-NW':[5,None,3]+[None]*6}, 
                         'Q3': {'N': [2,3]+[None]*6+[10,11]+[None]*6,
                                'NE':[3]+[None]*15,
                                'E-NW':[None,None,3,None,7]+[None]*11}}
        neighbors = {'N': n_nbr, 'NE': ne_nbr, 'E-NW': e_nw_nbr}
        
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            dofhandler = DofHandler(mesh, element)
        
            #
            # Fill in Dofs
            # 
            dofhandler.fill_dofs(node)
            
            #
            # Share dofs with neighbors
            #
            dofhandler.share_dofs_with_neighbors(node)
            for direction in ['N','NE','E-NW']:
                nbr_dofs = dofhandler.get_global_dofs(neighbors[direction]) 
                self.assertEqual(nbr_dofs, dofs_to_check[etype][direction],\
                             'Dofs shared incorrectly %s:'%(direction))
            
        mesh = Mesh(grid=DCEL(resolution=(2,2)))
        mesh.refine()
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh,element)
        root = mesh.root_node()
        dofhandler.distribute_dofs(nested=True)
        test_dofs = [[0,1,2,3],[1,4,3,5],[2,3,6,7],[3,5,7,8]]
        count = 0
        for child in root.get_children():
            self.assertEqual(dofhandler.get_global_dofs(child),\
                             test_dofs[count],\
                             'Dofs incorrectly distributed in grid.')
            count += 1
            
            
    def test_fill_dofs(self):
        mesh = Mesh()
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            dpe = element.n_dofs()
            dofhandler = DofHandler(mesh,element)
            #
            # Fill dofs for root node
            # 
            node = mesh.root_node()
            dofhandler.fill_dofs(node)
            node_dofs = dofhandler.get_global_dofs(node)
            self.assertEqual(node_dofs, list(range(dpe)),\
                             'Dofs not filled in correctly.')
            dof_count_error = 'The total number of dofs should be %d'%(dpe)
            self.assertEqual(dofhandler.n_dofs(), dpe, dof_count_error)
            
            #
            # Refine mesh and fill in dofs for a child
            #
            mesh.refine()
            child = node.children['SW']
            dofhandler.fill_dofs(child)
            child_dofs = dofhandler.get_global_dofs(child)
            self.assertEqual(child_dofs, list(np.arange(dpe,2*dpe)),\
                             'Child dofs not filled correctly.')
            dof_count_error = 'The total number of dofs should be %d'%(2*dpe)
            self.assertEqual(dofhandler.n_dofs(), 2*dpe, dof_count_error)
            
        #
        # Check dof count 
        # 
        mesh = Mesh(grid=DCEL(resolution=(2,2)))
        mesh.refine()
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh,element)
        count = 0
        for leaf in mesh.root_node().get_leaves():
            dofhandler.fill_dofs(leaf)
            self.assertEqual(dofhandler.n_dofs(),count+element.n_dofs(),\
                             'Dof count is adjusted incorrectly.')
            count += 4
    
    def test_assign_dofs(self):
        mesh = Mesh()
        element = QuadFE(2,'Q2')
        dofhandler = DofHandler(mesh,element)
        count_1 = dofhandler.n_dofs()
        node = mesh.root_node()
        #
        # Check Errors
        #
        self.assertRaises(IndexError, dofhandler.assign_dofs, node, 9, 12)
        self.assertRaises(Exception, dofhandler.assign_dofs, node, 'SSW', 12)
        self.assertRaises(Exception, dofhandler.assign_dofs, node, 2, -12)
        
        #
        # Check output
        # 
        dofhandler.assign_dofs(node,'NW',50)
        cell_dofs = dofhandler.get_global_dofs(node)
        self.assertEqual(cell_dofs[2], 50, \
                         'Dof in norhtwest corner should be 50')
        
        dofhandler.assign_dofs(node,[0,'NW'],[11,50])
        cell_dofs = dofhandler.get_global_dofs(node)
        self.assertEqual(cell_dofs[2], 50, \
                         'Dof in norhtwest corner should be 50')
        self.assertEqual(cell_dofs[0], 11, \
                         'First Dof should be 11')
        
        mesh.refine()
        child = node.children['SW']
        dofhandler.assign_dofs(child,[0,'NW'],[20,22])
        cell_dofs = dofhandler.get_global_dofs(child)
        self.assertEqual(cell_dofs[2], 22, \
                         'Dof in norhtwest corner should be 22')
        
        
        #
        # Make sure no extra dofs were counted during assignment
        #
        count_2 = dofhandler.n_dofs()
        self.assertEqual(count_1, count_2, 'The dof count should remain the same')


    def test_pos_to_int(self):
        # TODO: test
        pass
    
    def test_pos_to_dof(self):
        # TODO: test
        pass
    
    def test_get_global_dofs(self):
        # TODO: finish
        mesh = Mesh()
        for i in range(2):
            mesh.refine()
            mesh.record(i)
        element = QuadFE(2,'DQ0')
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs(nested=True)
        
        dofs_1 = dofhandler.get_global_dofs(flag=1)
        for leaf in mesh.root_node().get_leaves(flag=0):
            leaf_dofs = dofhandler.get_global_dofs(node=leaf)
        
        
        
        
    
    def test_n_dofs(self):
        # TODO: test
        pass
    
    def test_dof_vertices(self):
        # TODO: test
        pass
    '''