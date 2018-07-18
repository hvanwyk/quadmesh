import unittest
from fem import QuadFE
from fem import GaussRule
from mesh import Vertex, HalfEdge, QuadCell, convert_to_array
import numpy as np

class TestQuadFE(unittest.TestCase):
    """
    Test QuadFE class
    """   
    def test_reference_quadcell(self):
        """
        Test Constructor
        
        TODO:
        
            - test 1D: 
            - test all reference cells (?)
        """    
        # =====================================================================
        # Reference Interval/Cell
        # =====================================================================
        #
        #  1D   
        # 
        #
        # Check if reference interval contain the correct vertices
        # 
        etypes = ['DQ0', 'DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']
        dv = dict.fromkeys(etypes)
        dv['DQ0'] = {0: [0.5], 1: {0: [0.25], 1: [0.75]}}
        dv['DQ1'] = {0: [0, 1], 1: {0: [0,0.5], 1:[0.5, 1]}}
        dv['DQ2'] = {0: [0, 1, 0.5], 1: {0: [0,0.5,0.25], 1:[0.5,1,0.75]}}
        dv['DQ3'] = {0: [0, 1, 1/3, 2/3], 1: {0: [0,0.5,1/6,1/3], 1:[0.5,1,2/3, 5/6]}}
        dv['Q1'] =  {0: [0, 1], 1: {0: [0,0.5], 1:[0.5,1]}}
        dv['Q2'] =  {0: [0, 1, 0.5], 1: {0: [0,0.5, 0.25], 1:[0.5, 1, 0.75]}}
        dv['Q3'] =  {0: [0, 1, 1/3, 2/3], 1: {0: [0,0.5,1/6,1/3], 1:[0.5,1,2/3,5/6]}}
        for etype in etypes:
            element = QuadFE(1, etype)
            cell = element.reference_cell()
            for level in range(2):
                if level==0:
                    v = convert_to_array(cell.get_dof_vertices(level))
                    self.assertTrue(np.allclose(np.array(dv[etype][level]), v[:,0]))
                elif level==1:
                    for child in range(2):
                        v = convert_to_array(cell.get_dof_vertices(level, child))
                        self.assertTrue(np.allclose(np.array(dv[etype][level][child]), v[:,0]))
        #
        # Check if the vertices on the fine level are correctly linked with those on the coarse level
        # 
        # TODO:
        
        #
        # Test Positions
        # 
        for etype in ['DQ0', 'DQ1', 'DQ2', 'DQ3', 'Q1', 'Q2', 'Q3']:
            element = QuadFE(2, etype)
            cell = element.reference_cell()
            if etype=='Q1' or etype=='DQ1':
                # Level 1, child 0, vertex 0 = Level 0, vertex 0
                vertex = cell.get_dof_vertices(1, 0, 0)
                self.assertEqual(vertex.get_pos(1,0), vertex.get_pos(0))
                
                # Level 1, child 2, vertex 2 = Level 0, vertex 1
                vertex = cell.get_dof_vertices(1,2,2)
                self.assertEqual(vertex.get_pos(1,2), vertex.get_pos(0))
                
                # Level 1, child 2, vertex 0 has no inherited vertex
                vertex = cell.get_dof_vertices(1,2,0)
                self.assertIsNone(vertex.get_pos(0))
                
            elif etype=='Q2':
                # All children share middle vertex
                for i in range(4):
                    vertex = cell.get_dof_vertices(1, i, (i+2)%4)
                    self.assertEqual(vertex.get_pos(0), 8)
            elif etype=='DQ2':
                # Only first child shares middle vertex
                for i in range(4):
                    vertex = cell.get_dof_vertices(1, i, (i+2)%4)
                    if i==0:
                        self.assertEqual(vertex.get_pos(0), 8)
                    else:
                        self.assertIsNone(vertex.get_pos(0))
                        
                # Level 1, child 1, vertex 0 has no inherited vertex        
                vertex = cell.get_dof_vertices(1, 1, 0)
                self.assertIsNone(vertex.get_pos(0))
                
                # Level 1, child 1, vertex 5 has no inherited vertex
                vertex = cell.get_dof_vertices(1, 1, 5)
                self.assertIsNone(vertex.get_pos(0))
            elif etype=='Q3' or etype=='DQ3':
                # Level 1, child 2, vertex 2 = Level 0, vertex 2
                vertex = cell.get_dof_vertices(1, 2, 2)
                self.assertEqual(vertex.get_pos(1,2), vertex.get_pos(0))   
            
                # Level 1, child 2, vertex 6 = Level 0, vertex 7
                vertex = cell.get_dof_vertices(1, 2, 6)
                self.assertEqual(vertex.get_pos(0), 7)
                
                # Level 1, child 2, vertex 12 = Level 0, vertex 15
                vertex = cell.get_dof_vertices(1, 2, 12)
                self.assertEqual(vertex.get_pos(0), 15)
      
            
                                         
    def test_element_type(self):
        for etype in ['DQ0', 'Q1', 'Q2', 'Q3', 'DQ1', 'DQ2', 'DQ3']:
            element = QuadFE(2, etype)
            self.assertEqual(element.element_type(),etype,\
                             'Element type not correct.')
    
    
    def test_polynomial_degree(self):
        count = 1
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            n = element.polynomial_degree()
            self.assertEqual(n, count,\
                 'Incorrect polynomial degree %d for element %s'%(n,etype) )
            count += 1
            
    
    def test_n_dofs(self):
        element = QuadFE(2,'Q1')
        n_dofs = element.n_dofs()
        self.assertEqual(n_dofs, 4, 'Number of dofs is 4.')
            
       
    def test_reference_nodes(self):
        pass   
    
    
    def test_constraint_coefficients(self):
        """
        UNFINISHED
        """
        '''
        test_function = {'Q1': lambda x,y: 2*x*(1-y)+2, 
                         'Q2': lambda x,y: (x-2)*(y+1),
                         'Q3': lambda x,y: x**3*(1-y)*2-x**2*(1-y)+3}
        # Dofs on coarse 
        coarse_nodes = {'Q1': [0,1], 'Q2': [0,6,1], 'Q3': [0,8,9,1]}
        hanging_nodes = {'Q1': [[1],[0]], 'Q2': [[6],[6]],'Q3':[[8,1],[0,9]]}
        
        # Interpolation points in the left cell
        xy_left = np.zeros((5,2))
        xy_left[:,0] = np.linspace(0,0.5,5)
        
        # Interpolation points on the right cell
        xy_right = np.zeros((5,2))
        xy_right[:,0] = np.linspace(0.5,1,5)
        
        xy_ref = np.zeros((5,2))
        xy_ref[:,0] = np.linspace(0,1,5) 
        
        # Combined Interpolation points
        xy = np.concatenate((xy_left,xy_right),axis=0)
        
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            f = test_function[etype]
            dpe = element.n_dofs('edge')+2
            # Interpolate over coarse edge
            x_ref = element.reference_nodes()
            f_interp = np.zeros(xy.shape[0])
            for n in coarse_nodes[etype]:
                fi = f(x_ref[n,0],x_ref[n,1])
                f_interp += fi*element.phi(n, xy)
            
            cc = element.constraint_coefficients()
            
            # Reference vertices for left cell
            x_left_ref = np.zeros(x_ref.shape)
            x_left_ref[:,0] = 0.5*x_ref[:,0]
            f_left_interp = np.zeros((5,)) 
            for l in range(dpe):
                # dofs in left cell
                il = coarse_nodes[etype][l]
                
                # Evaluate function at reference node
                fi = f(x_left_ref[il,0],x_left_ref[il,1])
                
                # Shape functions
                phi = np.zeros((5,))
                if il in hanging_nodes[etype][0]:
                    # Hanging node
                    for i in range(dpe):
                        ic = coarse_nodes[etype][i]
                        phi += cc[0][l][i]*element.phi(ic, xy_left)
                else:
                    # Not a hanging node
                    phi = element.phi(il,xy_ref)
                f_left_interp += fi*phi
                
            f_left_vals = f(xy_left[:,0],xy_left[:,1])
            #print(f_left_vals)
            #print(f_left_interp)
            x_right_ref = np.zeros(x_ref.shape)
            x_right_ref[:,0] = 0.5 + 0.5*x_ref[:,0]
            f_right_interp = f(x_right_ref[:,0],x_right_ref[:,1]) 
            
            
            # Exact function at interpolation points
            f_vals = f(xy[:,0],xy[:,1])
            self.assertTrue(np.allclose(f_vals,f_interp),\
                            'Coarse cell interpolation incorrect.')
            '''
                      
                        
    def test_phi(self):
        for etype in ['DQ0','Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            n_dofs = element.n_dofs()
            I = np.eye(n_dofs)
            x = element.reference_nodes()
            for n in range(n_dofs):
                self.assertTrue(np.allclose(element.phi(n,x),I[:,n]),\
                                'Shape function evaluation incorrect')
                
                
    def test_dphi(self):
        """
        Define piecewise linear, quadratic, and cubic polynomials on the
        reference element, compute their first partial derivatives and 
        compare with that of the nodal interpolant. 
        """
        pass
 
    
    def test_d2phi(self):
        """
        Define piecewise linear, quadratic, and cubic polynomials on the
        reference triangle, compute their first partial derivatives and 
        compare with that of the nodal interpolant.
        """
        pass
    
    
    def test_shape(self):
        """
        Test shape functions
        """
        test_functions = {'Q1': (lambda x,y: (x+1)*(y-1), lambda x,y: y-1, \
                                 lambda x,y: x+1), 
                          'Q2': (lambda x,y: x**2 -1, lambda x,y: 2*x, \
                                 lambda x,y: 0*x),
                          'Q3': (lambda x,y: x**3 - y**3, lambda x,y: 3*x**2, \
                                 lambda x,y: -3*y**2)}
        #
        # Over reference cell
        # 
        cell_integrals = {'Q1': [-0.75,-0.5,1.5], 
                          'Q2': [-2/3.,1.0,0.0],
                          'Q3': [0.,1.0,-1.0]}
        
        derivatives = [(0,),(1,0),(1,1)] 
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            n_dofs = element.n_dofs()
            x_ref = element.reference_nodes()
            #
            # Sanity check
            # 
            I = np.eye(n_dofs)
            self.assertTrue(np.allclose(element.shape(x_ref),I),\
                            'Shape functions incorrect at reference nodes.')
            y = np.random.rand(5,2)
            rule2d = GaussRule(9, element=element)
            weights = rule2d.weights()
            x_gauss = rule2d.nodes()
            f_nodes = test_functions[etype][0](x_ref[:,0],x_ref[:,1])
            for i in range(3):
                phi = element.shape(y, derivatives=derivatives[i])
                f = test_functions[etype][i]
                #
                # Interpolation
                #
                fvals = f(y[:,0],y[:,1])            
                self.assertTrue(np.allclose(np.dot(phi,f_nodes),fvals),\
                                'Shape function interpolation failed.')
                #
                # Integration
                #
                phi = element.shape(x_gauss, derivatives=derivatives[i])  
                self.assertAlmostEqual(np.dot(weights,np.dot(phi,f_nodes)),\
                                 cell_integrals[etype][i],places=8,\
                                 msg='Incorrect integral.')
        #
        # Non-rectangular quadcell
        #
        test_functions['Q1'] = (lambda x,y: (x+1) + (y-1), 
                                lambda x,y: np.ones(x.shape), 
                                lambda x,y: np.ones(y.shape))
        
        # Vertices
        v0 = Vertex((0,0))
        v1 = Vertex((0.5,0.5))
        v2 = Vertex((0,2))
        v3 = Vertex((-0.5, 0.5))
        
        # half_edges
        h01 = HalfEdge(v0,v1)
        h12 = HalfEdge(v1,v2)
        h23 = HalfEdge(v2,v3)
        h30 = HalfEdge(v3,v0)
        
        # quadcell
        cell = QuadCell([h01, h12, h23, h30])
        
        for etype in ['Q1', 'Q2', 'Q3']:
            element = QuadFE(2, etype)
            n_dofs = element.n_dofs()
            x_ref = element.reference_nodes()
            x = cell.reference_map(x_ref)
            #
            # Sanity check
            # 
            I = np.eye(n_dofs)
            self.assertTrue(np.allclose(element.shape(x, cell),I),\
                            'Shape functions incorrect at reference nodes.')
            y = cell.reference_map(np.random.rand(5,2))
            self.assertTrue(all(cell.contains_points(y)),\
                            'Cell should contain all mapped points')
            f_nodes = test_functions[etype][0](x[:,0],x[:,1])
            
            for i in range(3):
                phi = element.shape(y, cell=cell, derivatives=derivatives[i])
                f = test_functions[etype][i]
                #
                # Interpolation
                #
                fvals = f(y[:,0],y[:,1])  
                self.assertTrue(np.allclose(np.dot(phi,f_nodes),fvals),\
                                'Shape function interpolation failed.')
                
        #
        # 1D
        # 