"""
Created 11/22/2016
@author: hans-werner
"""
import unittest
from finite_element import QuadFE, DofHandler, GaussRule, System
from mesh import Mesh, Edge, Vertex
from numpy import sqrt, sum, dot, sin, pi, array, abs, empty, zeros, max, \
                  allclose, eye, random


class TestFiniteElement(unittest.TestCase):
    """
    Test FiniteElement class
    """
    pass

class TestQuadFE(unittest.TestCase):
    """
    Test QuadFE class
    """
    def test_n_dofs(self):
        element = QuadFE(2,'Q1')
        n_dofs = element.n_dofs()
        self.assertEqual(n_dofs, 4, 'Number of dofs is 4.')
        
    def test_shape_functions(self):
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            n_dofs = element.n_dofs()
            I = eye(n_dofs)
            x = element.reference_nodes()
            for n in range(n_dofs):
                self.assertTrue(allclose(element.phi(n,x),I[:,n]),\
                                'Shape function evaluation incorrect')
                
                
    def test_get_local_edge_dofs(self):
        edge_dofs_exact = {'Q1': {'W':[0,2],'E':[1,3],'S':[0,1],'N':[2,3]},
                           'Q2': {'W':[0,2,4],'E':[1,3,5],\
                                  'S':[0,1,6],'N':[2,3,7]},
                           'Q3': {'W':[0,2,4,5],'E':[1,3,6,7],\
                                  'S':[0,1,8,9],'N':[2,3,10,11]}} 
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            for direction in ['W','E','S','N']:
                edge_dofs = element.get_local_edge_dofs(direction)
                self.assertEqual(edge_dofs, edge_dofs_exact[etype][direction],\
                                 'Edge dofs incorrect')

class TestTriFE(unittest.TestCase):
    """
    Test TriFE classe
    
    """

    
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
        
        

class TestGaussRule(unittest.TestCase):
    """
    Test GaussRule class
    """
    pass

class TestSystem(unittest.TestCase):
    """
    Test System class
    """
    
    def test_assembly(self):
        
        # ---------------------------------------------------------------------
        # One square 
        # ---------------------------------------------------------------------
        #
        # Mass Matrix
        # 
        mesh = Mesh.newmesh()
        V = QuadFE(2,'Q1')
        s = System(mesh,V, n_gauss=(3,9))
        lf = [(1,'v')]
        bf = [(1,'u','v')]
        A,b = s.assemble(bf,lf)
        AA = 1/36.0*array([[4,2,2,1],[2,4,1,2],[2,1,4,2],[1,2,2,4]])
        self.assertTrue(allclose(A.toarray(),AA),'Incorrect mass matrix')
        b_check = 0.25*array([1,1,1,1])
        self.assertTrue(allclose(b,b_check),'Right hand side incorrect')
        #
        # Stiffness Ax
        # 
        bf = [(1,'ux','vx')]
        Ax = s.assemble(bilinear_forms=bf)
        AAx = 1/6.0*array([[2,-2,1,-1],[-2,2,-1,1],[1,-1,2,-2],[-1,1,-2,2]])
        self.assertTrue(allclose(Ax.toarray(),AAx),
                               'Incorrect stiffness matrix')
        #
        # Stiffness Ay
        # 
        bf = [(1,'uy','vy')]
        A = s.assemble(bilinear_forms=bf)
        AAy = 1/6.0*array([[2,1,-2,-1],[1,2,-1,-2],[-2,-1,2,1],[-1,-2,1,2]])
        self.assertTrue(allclose(A.toarray(),AAy), 'Ay incorrect')


        #
        # Use matrices to integrate
        #
        q = lambda x,y: x*(1-x)*y*(1-y)
        bilinear_forms = [(q,'u','v')]
        linear_forms = [(1,'v')]
        A,_ = s.assemble(bilinear_forms, linear_forms) 
        v = array([1.,1.,1.,1.])
        self.assertAlmostEqual(dot(v,A.tocsr().dot(v))-1.0/36.0, 0,8,\
                               'Should integrate to 4/pi^2.')
        
        # ---------------------------------------------------------------------
        # Elaborate tests with Boundary conditions
        # ---------------------------------------------------------------------
        def m_dirichlet(x,y):
            """
            Dirichlet Node Marker: x = 0
            """
            return (abs(x)<1e-10)
        
        def m_neumann(edge):
            """
            Neumann Edge Marker: x = 1
            """
            x = array(edge.vertex_coordinates())
            return (abs(x[:,0]-1)<1e-9).all()
                
        def m_robin_1(edge):
            """
            Robin Edge Marker: y = 0 
            """
            x = array(edge.vertex_coordinates())
            return (abs(x[:,1]-0)<1e-9).all()  
        
        def m_robin_2(edge):
            """
            Robin Edge Marker: y = 1
            """
            x = array(edge.vertex_coordinates())
            return (abs(x[:,1]-1)<1e-9).all() 
            
        def g_dirichlet(x,y):
            """
            Dirichlet function
            """
            return zeros(shape=x.shape)
        
        def g_neumann(x,y):
            """
            Neumann function
            """        
            return -y*(1-y)
        
        def g_robin_1(x,y):
            """
            Robin boundary conditions for y = 0
            """
            return -x*(1-x)      
        
        def g_robin_2(x,y):
            """
            Robin boundary conditions for y = 1
            """
            return -0.5*x*(1-x)
        cell = mesh.root_quadcell()
        edge_east = cell.get_edges('E')
    
        gamma_1 = 1.0
        gamma_2 = 2.0
        
        u = lambda x,y: x*(1-x)*y*(1-y)  # exact solution
        f = lambda x,y: 2.0*(x*(1-x)+y*(1-y))  # forcing term
        bf = [(1,'ux','vx'),(1,'uy','vy'),(1,'u','v')]  # bilinear forms
        lf = [(f,'v')]  # linear forms
        s = System(mesh, V, n_gauss=(3,9))
        A,b = s.assemble(bilinear_forms=bf,linear_forms=lf,\
                         boundary_conditions=None)
        # check system matrix
        self.assertTrue(allclose(A.toarray(),AA+AAx+AAy),
                        'System matrix incorrect')
        # check right hand side
        bb = 1/6.0*array([1,1,1,1])
        self.assertTrue(allclose(b,bb), 'Right hand side incorrect')
        
        # 
        # Add Neumann boundary conditions
        # 
        bc_1 = {'dirichlet': None, 'neumann': [(m_neumann,g_neumann)],
                'robin': None}
        A,b = s.assemble(bilinear_forms=bf, linear_forms=lf, \
                         boundary_conditions=bc_1)
        # check system matrix
        self.assertTrue(allclose(A.toarray(),AA+AAx+AAy),
                               'System matrix incorrect')
        # check right hand side
        bneu = -1/12.0*array([0,1,0,1])

        self.assertTrue(allclose(b,bb+bneu),\
                        'Right hand side with Neumann incorrect')
        
        #
        # Add Robin boundary conditions
        #
        bc_2 = {'dirichlet': None,\
                'neumann': [(m_neumann,g_neumann)], \
                'robin': [(m_robin_1, (gamma_1,g_robin_1)),\
                          (m_robin_2,(gamma_2,g_robin_2))],\
                'periodic': None}
        A,b = s.assemble(bilinear_forms=bf, linear_forms=lf, \
                         boundary_conditions=bc_2)
        # Check system matrix
        R1 = 1/6.0*array([[2,1,0,0],[1,2,0,0],[0,0,0,0],[0,0,0,0]])
        R2 = 1/3.0*array([[0,0,0,0],[0,0,0,0],[0,0,2,1],[0,0,1,2]])
        self.assertTrue(allclose(A.toarray(),AA+AAx+AAy+R1+R2),'System matrix incorrect')
        
        # check right hand side
        bR1 = -1/12.0*array([1,1,0,0])
        bR2 = -1/12.0*array([0,0,1,1])
        self.assertTrue(allclose(b,bb+bneu+bR1+bR2),\
                        'Right hand side incorrect.')
        #
        # Add Dirichlet boundary conditions 
        # 
        bc_3 = {'dirichlet': [(m_dirichlet,g_dirichlet)],\
                'neumann': [(m_neumann,g_neumann)], \
                'robin': [(m_robin_1, (gamma_1,g_robin_1)),\
                          (m_robin_2,(gamma_2,g_robin_2))],\
                'periodic': None}
        
        A,b = s.assemble(bilinear_forms=bf, linear_forms=lf, \
                   boundary_conditions=bc_3)
        # Check system matrix
        AAdir = array([[1,0,0,0],[0,10.0/9.0,0,-1.0/9.0],
                       [0,0,1.0,0],[0,-1./9.,0,13./9.]]) 
        self.assertTrue(allclose(A.toarray(),AAdir),'System matrix incorrect')
        bbdir = 1/6.0*zeros((4,))
        self.assertTrue(allclose(b,bbdir),'Right hand side incorrect') 
        
        #
        # Test Q2
        #
        mesh = Mesh.newmesh()
        element = QuadFE(2,'Q2')
        s = System(mesh,element,n_gauss=(3,9))
        bf = [(1,'u','v'),(1,'ux','vx'),(1,'uy','vy')]
        lf = [(f,'v')]
        A, b = s.assemble(bilinear_forms=bf, linear_forms=lf)
        rule2d = GaussRule(9,shape='quadrilateral')
        r = rule2d.nodes()
        w = rule2d.weights()
        n_dofs = element.n_dofs()
        AA = zeros((n_dofs,n_dofs))
        AAx = zeros((n_dofs,n_dofs))
        AAy = zeros((n_dofs,n_dofs))
        for i in range(n_dofs):
            for j in range(n_dofs):
                phii = element.phi(i, r)
                phij = element.phi(j, r)
                phiix = element.dphi(i, r, var=0)
                phijx = element.dphi(j, r, var=0)
                phiiy = element.dphi(i, r, var=1)
                phijy = element.dphi(j, r, var=1)
                AA[i,j] = sum(w*phii*phij)
                AAx[i,j] = sum(w*phiix*phijx)
                AAy[i,j] = sum(w*phiiy*phijy)
        self.assertTrue(allclose(AA+AAx+AAy,A.toarray()),'Mass matrix not correct.')
        
        #
        # Test Neumann condition
        # 
        rule_1d = GaussRule(3,shape='edge')
        e_neu = mesh.root_quadcell().get_edges('W')
        r_ref_1d = rule_1d.nodes()
        r_phys_1d = rule_1d.map(e_neu, r_ref_1d)
        w_ref_1d = rule_1d.weights()
        element_1d = QuadFE(1,element.element_type())
        phi0 = element_1d.phi(0,r_ref_1d)
        phi2 = element_1d.phi(2,r_ref_1d)
        phi1 = element_1d.phi(1,r_ref_1d)
        
        w_neu = w_ref_1d*rule_1d.jacobian(e_neu)
        g_neu = g_neumann(r_phys_1d[:,0],r_phys_1d[:,1])
        
        bb_neu = zeros((n_dofs,))
        bb_neu[1] = sum(w_neu*phi0*g_neu)
        bb_neu[5] = sum(w_neu*phi2*g_neu)
        bb_neu[3] = sum(w_neu*phi1*g_neu)
        self.assertTrue(allclose(bb_neu,array([0,-1/60.,0,-1/60.,0,-2./15.,0,0,0])),\
                        'Integration over Neumann boundary incorrect')
        
        A_neu,b_neu = s.assemble(bilinear_forms=bf, linear_forms=lf, \
                             boundary_conditions={'dirichlet': None, 
                                                  'neumann': [(m_neumann,g_neumann)],
                                                  'robin': None})
        self.assertTrue(allclose(b_neu,bb_neu+b),'Right hand side incorrect.')
        self.assertTrue(allclose(A_neu.toarray(),AA+AAx+AAy))
        
        #
        # Test Robin 1 conditions       
        #
        e_r1 = mesh.root_quadcell().get_edges('S') 
        
        r_phys_1d = rule_1d.map(e_r1,r_ref_1d)
        w_r1 = w_ref_1d*rule_1d.jacobian(e_r1)
        g_r1 = g_robin_1(r_phys_1d[:,0], r_phys_1d[:,1])
        bb_r1 = zeros((n_dofs,))
        bb_r1[0] = sum(w_r1*phi0*g_r1)
        bb_r1[1] = sum(w_r1*phi1*g_r1)
        bb_r1[6] = sum(w_r1*phi2*g_r1)
        
        R1 = zeros((n_dofs,n_dofs))
        R1[0,0] = sum(w_r1*phi0*phi0)
        R1[0,1] = sum(w_r1*phi0*phi1)
        R1[1,1] = sum(w_r1*phi1*phi1)
        R1[1,0] = R1[0,1]
        R1[0,6] = sum(w_r1*phi0*phi2)
        R1[6,0] = R1[0,6]
        R1[6,6] = sum(w_r1*phi2*phi2)
        R1[1,6] = sum(w_r1*phi1*phi2)
        R1[6,1] = sum(w_r1*phi1*phi2)
        
        A_r1, b_r1 = \
            s.assemble(bilinear_forms=bf,linear_forms=lf,\
                       boundary_conditions={'dirichlet': None,
                                            'neumann':[(m_neumann,g_neumann)],
                                            'robin': [(m_robin_1,(gamma_1,g_robin_1))]})
        self.assertTrue(allclose(AA+AAx+AAy+R1,A_r1.toarray()),'Robin condition 1, system incorrect.')
        self.assertTrue(allclose(b+bb_neu+bb_r1,b_r1),'Robin conditions 1, rhs incorrect.')
        
        #
        # Test Robin 2 conditions
        # 
        e_r2 = mesh.root_quadcell().get_edges('N') 
        
        r_phys_1d = rule_1d.map(e_r2,r_ref_1d)
        w_r1 = w_ref_1d*rule_1d.jacobian(e_r2)
        g_r1 = g_robin_1(r_phys_1d[:,0], r_phys_1d[:,1])
        bb_r1 = zeros((n_dofs,))
        bb_r1[0] = sum(w_r1*phi0*g_r1)
        bb_r1[1] = sum(w_r1*phi1*g_r1)
        bb_r1[6] = sum(w_r1*phi2*g_r1)
        
        R1 = zeros((n_dofs,n_dofs))
        R1[0,0] = sum(w_r1*phi0*phi0)
        R1[0,1] = sum(w_r1*phi0*phi1)
        R1[1,1] = sum(w_r1*phi1*phi1)
        R1[1,0] = R1[0,1]
        R1[0,6] = sum(w_r1*phi0*phi2)
        R1[6,0] = R1[0,6]
        R1[6,6] = sum(w_r1*phi2*phi2)
        R1[1,6] = sum(w_r1*phi1*phi2)
        R1[6,1] = sum(w_r1*phi1*phi2)
        
        A_r1, b_r1 = \
            s.assemble(bilinear_forms=bf,linear_forms=lf,\
                       boundary_conditions={'dirichlet': None,
                                            'neumann':[(m_neumann,g_neumann)],
                                            'robin': [(m_robin_1,(gamma_1,g_robin_1))]})
        self.assertTrue(allclose(AA+AAx+AAy+R1,A_r1.toarray()),'Robin condition 1, system incorrect.')
        self.assertTrue(allclose(b+bb_neu+bb_r1,b_r1),'Robin conditions 1, rhs incorrect.')
         
        dofhandler = DofHandler(mesh,element)
        dofhandler.distribute_dofs()
        x = dofhandler.mesh_nodes()
        ui = u(x[:,0],x[:,1])
      
        #self.assertTrue(allclose(A.dot(ui),b), 
        #                'Nodal valued function does not solve system')
      
        #
        # Two squares
        #
        mesh = Mesh.newmesh(box=[2.0,2.5,1.0,3.0], grid_size=(2,1))
        mesh.refine()
        s = System(mesh,V,n_gauss=(6,9))
        bilinear_forms = [(1,'u','v')]
        linear_forms = [(1,'v')]
        A,_ = s.assemble(bilinear_forms=bilinear_forms, 
                       linear_forms=linear_forms)
      
        A_check = 1/36.0*array([[4,2,2,1,0,0],[2,8,1,4,2,1],[2,1,4,2,0,0],
                               [1,4,2,8,1,2],[0,2,0,1,4,2],[0,1,0,2,2,4]])
        self.assertAlmostEqual((A.toarray()-A_check).all(), 0, 12,\
                               'Incorrect mass matrix')
        
        
        
        #
        # Test a fine mesh (multiple elements ;))
        # 
        mesh = Mesh.newmesh(grid_size=(20,20))
        s = System(mesh,V)
        
        # Test hanging nodes
        
        
    def test_line_integral(self):
        # Define quadrature rule
        rule = GaussRule(2, shape='edge')
        w = rule.weights()
        
        # function f to be integrated over edge e
        f = lambda x,y: x**2*y
        e = Edge(Vertex((0,0)),Vertex((1,1)))
        
        # Map rule to physical entity
        x_ref = rule.map(e)
        jac = rule.jacobian(e)
        fvec = f(x_ref[:,0],x_ref[:,1])
        
        self.assertAlmostEqual(sum(dot(fvec,w))*jac,1/sqrt(2)/2,places=10,\
                               msg='Failed to integrate x^2y.')
        self.assertAlmostEqual(sum(w)*jac, sqrt(2), places=10,\
                               msg='Failed to integrate 1.')
        
    def test_flux_integral(self):
        """
        """
    
    def test_f_eval_loc(self):
        mesh = Mesh.newmesh()
        cell = mesh.root_quadcell()
        node = mesh.root_node()
        test_functions = {'Q1': lambda x,y: (2+x)*(y-3),
                          'Q2': lambda x,y: (2+x**2+x)*(y-2)**2,
                          'Q3': lambda x,y: (2*x**3-3*x)*(y**2-2*y)} 
        x_test = random.rand(5,2)
    
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            system = System(mesh,element)
            f = test_functions[etype]             
            f_test = f(x_test[:,0],x_test[:,1])
            # -----------------------------------------------------------------
            # Interpolation
            # -----------------------------------------------------------------
            #
            # f in functional form                   
            # 
            f_loc = system.f_eval_loc(f, entity=cell,x=x_test)
            self.assertTrue(allclose(f_loc,f_test),\
                                   'Function not correctly interpolated.')
        
            #
            # f in nodal form
            #
            local_dofs = system.get_node_dofs(node)
            x = system.mesh_nodes()
            x_loc = x[local_dofs,:]
            f_nodes = f(x_loc[:,0],x_loc[:,1])
            f_loc = system.f_eval_loc(f_nodes,entity=cell,x=x_test)
            self.assertTrue(allclose(f_loc,f_test), \
                                   'Function not correctly interpolated.')
        
            # -----------------------------------------------------------------
            # Quadrature
            # -----------------------------------------------------------------
             
    
    def test_shape_eval(self):
        test_functions = {'Q1': (lambda x,y: (x+1)*(y-1), lambda x,y: y-1, \
                                 lambda x,y: x+1), 
                          'Q2': (lambda x,y: x**2 -1, lambda x,y: 2*x, \
                                 lambda x,y: 0*x),
                          'Q3': (lambda x,y: x**3 - y**3, lambda x,y: 3*x**2, \
                                 lambda x,y: -3*y**2)}
        
        cell_integrals = {'Q1': [-0.75,-0.5,1.5], 
                          'Q2': [-2/3.,1.0,0.0],
                          'Q3': [0.,1.0,-1.0]}
        edge_integrals_west = {'Q1': [-0.5,-0.5,1.0],
                               'Q2': [-1.0,0.0,0.0],
                               'Q3': [-0.25,0.0,-1.0]} 
        derivatives = [(0,),(1,0),(1,1)]
        mesh = Mesh.newmesh()
        cell = mesh.root_quadcell() 
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            system = System(mesh,element)
            n_dofs = element.n_dofs()
            x = element.reference_nodes()
            #
            # Sanity check
            # 
            I = eye(n_dofs)
            self.assertTrue(allclose(system.shape_eval(x=x),I),\
                            'Shape functions incorrect at reference nodes.')
            y = random.rand(5,2)
            weights = system.cell_rule().weights()
            f_nodes = test_functions[etype][0](x[:,0],x[:,1])
            for i in range(3):
                phi = system.shape_eval(derivatives=derivatives[i], x=y)
                f = test_functions[etype][i]
                #
                # Interpolation
                #
                fvals = f(y[:,0],y[:,1])            
                self.assertTrue(allclose(dot(phi,f_nodes),fvals),\
                                'Shape function interpolation failed.')
                #
                # Integration
                #
                phi = system.shape_eval(derivatives=derivatives[i])  
                self.assertAlmostEqual(dot(weights,dot(phi,f_nodes)),\
                                 cell_integrals[etype][i],places=8,\
                                 msg='Incorrect integral.')
            #
            # On Edges   
            # 
            y = random.rand(5)
            
            for direction in ['W','E','S','N']:
                edge = cell.get_edges(direction)
                weights = system.edge_rule().weights()*\
                    system.edge_rule().jacobian(edge)
                #
                # Sanity check
                # 
                edge_dofs = element.get_local_edge_dofs(direction)
                x_ref = x[edge_dofs,:]
                entity = system.make_generic((edge,direction))
                phi = system.shape_eval(entity=entity,x=x_ref)
                self.assertTrue(allclose(phi, I[edge_dofs,:]), \
                                'Shape function incorrect at edge ref nodes.')
                y_phys = system.edge_rule().map(edge, y)
                f_nodes = test_functions[etype][0](x[:,0],x[:,1])
                for i in range(3):
                    #
                    # Interpolation
                    # 
                    phi = system.shape_eval(derivatives=derivatives[i],\
                                            entity=entity,x=y_phys)
                    f = test_functions[etype][i]
                    fvals = f(y_phys[:,0],y_phys[:,1])
                    self.assertTrue(allclose(dot(phi,f_nodes),fvals),\
                                'Shape function interpolation failed.') 
                    #
                    # Quadrature
                    # 
                    if direction == 'W':
                        phi = system.shape_eval(derivatives=derivatives[i],\
                                                entity=entity)
                        self.assertAlmostEqual(dot(weights,dot(phi,f_nodes)),\
                                 edge_integrals_west[etype][i],places=8,\
                                 msg='Incorrect integral.')
    
    
       
        
                        
    def test_make_generic(self):
        mesh = Mesh.newmesh()
        element = QuadFE(2,'Q1')
        system = System(mesh, element)
        cell = mesh.root_quadcell()
        self.assertEqual(system.make_generic(cell), 'cell', \
                         'Cannot convert cell to "cell"')
        for direction in ['W','E','S','N']:
            edge = cell.get_edges(direction)
            self.assertEqual(system.make_generic((edge,direction)),\
                             ('edge',direction),\
                             'Cannot convert edge to generic edge')
            
    def test_parse_derivative_info(self):
        mesh = Mesh.newmesh()
        element = QuadFE(2,'Q2')
        system = System(mesh,element)
        self.assertEqual(system.parse_derivative_info('u'), (0,),\
                         'Zeroth derivative incorrectly parsed')
        self.assertEqual(system.parse_derivative_info('ux'), (1,0),\
                         'Zeroth derivative incorrectly parsed')
        self.assertEqual(system.parse_derivative_info('vy'), (1,1),\
                         'Zeroth derivative incorrectly parsed')
        