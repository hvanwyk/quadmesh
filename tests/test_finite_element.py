"""
Created 11/22/2016
@author: hans-werner
"""
import unittest
from finite_element import QuadFE, DofHandler, GaussRule, System
from mesh import Mesh, Edge, Vertex
from numpy import sqrt, sum, dot, sin, pi, array, abs, empty, zeros

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
        
        

class TestGaussRule(unittest.TestCase):
    """
    Test GaussRule class
    """
    
    def test_assembly(self):
        
        #
        # One square 
        #
        mesh = Mesh.newmesh()
        V = QuadFE(2,'Q1')
        s = System(mesh,V, n_gauss=(3,9))
        # bilinear_forms = [(1,'u','v'),(1,'ux','vx'),(1,'uy','vy')];
        lf = [(1,'v')]
        bf = [(1,'u','v')]
        A,b = s.assemble(bf,lf)
        
        A_check = 1/36.0*array([[4,2,2,1],[2,4,1,2],[2,1,4,2],[1,2,2,4]])
        self.assertAlmostEqual((A.toarray()- A_check).all(),0, 12,\
                                'Incorrect mass matrix')
    
        b_check = 0.25*array([1,1,1,1])
        self.assertAlmostEqual((b-b_check).all(), 0, 12,\
                              'Righ hand side incorrect')
        bf = [(1,'ux','vx')]
        Ax = s.assemble(bilinear_forms=bf)
        Ax_check = 1/6.0*array([[2,-2,1,-1],[-2,2,-1,1],[1,-1,2,-2],[-1,1,-2,2]])
        self.assertAlmostEqual((Ax.toarray()-Ax_check).all(), 0, 12, \
                               'Incorrect stiffness matrix')
        #
        # Use matrices to integrate
        #
        q = lambda x,y: x*(1-x)*y*(1-y)
        bilinear_forms = [(q,'u','v')]
        linear_forms = [(1,'v')]
        A,_ = s.assemble(bilinear_forms, linear_forms) 
        v = array([1.,1.,1.,1.])
        AA = A.tocsr()
        print(AA.dot(v))
        self.assertAlmostEqual(dot(v,AA.dot(v))-1.0/36.0, 0,8,\
                               'Should integrate to 4/pi^2.')
        
        #
        # Elaborate test with Boundary conditions
        # 
        def m_dirichlet(x,y):
            """
            Dirichlet Node Marker: x = 0
            """
            return (abs(x)<1e-10)
        
        def m_neumann(edge):
            """
            Neumann Edge Marker: x = 1
            """
            x = edge.vertex_coordinates()
            return (abs(x[:,0]-1)<1e-9).all()
                
        def m_robin_1(edge):
            """
            Robin Edge Marker: y = 0 
            """
            x = edge.vertex_coordinates()
            return (abs(x[:,1]-0)<1e-9).all()  
        
        def m_robin_2(edge):
            """
            Robin Edge Marker: y = 1
            """
            x = edge.vertex_coordinates()
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
            return y*(1-y)
        
        def g_robin_1(x,y):
            """
            Robin boundary conditions for y = 0
            """
            return -(x*(1-x)) 
        
        def g_robin_2(x,y):
            """
            Robin boundary conditions for y = 1
            """
            return -0.5*x*(1-x)
            
        gamma_1 = 1.0
        gamma_2 = 2.0
        bnd_conditions = {'dirichlet': [(m_dirichlet,g_dirichlet)],\
                          'neumann': [(m_neumann,g_neumann)], \
                          'robin': [(m_robin_1, (gamma_1,g_robin_1)),\
                                    (m_robin_2,(gamma_2,g_robin_2))],\
                          'periodic': None}
        u = lambda x,y: x*(1-x)*y*(1-y)
        f = lambda x,y: 2.0*(x*(1-x)+y*(1-y))
        bf = [(-1,'ux','vx'),(1,'uy','vy'),(1,'u','v')]
        lf = [(f,'v')]
        s = System(mesh, V, n_gauss=(3,9))
        A,b = s.assemble(bilinear_forms=bf, linear_forms=lf, \
                   boundary_conditions=bnd_conditions)
        
        
        cell = mesh.root_quadcell()
        dofhandler = DofHandler(mesh,V)
        dofhandler.distribute_dofs()
        for direction in ['W','E','S','N']:
            edge = cell.get_edges(direction)
            if m_neumann(edge):
                print('%s-Edge is Neumann'%(direction))
            elif m_robin_1(edge):
                print('%s-Edge is Robin'%(direction))
            x = edge.vertex_coordinates()
            is_dirichlet = m_dirichlet(x[:,0],x[1,:])
            if is_dirichlet.any():
                for y in x[is_dirichlet,:]:
                    print('Node (%.2f,%.2f) is Dirichlet'%(y[0],y[1]))
        x = dofhandler.mesh_nodes()
        ui = u(x[:,0],x[:,1])
        fi = f(x[:,0],x[:,1])
        A = A.toarray()
        print(A)
        print(b)
        self.assertAlmostEqual((A.dot(ui)-b).all(),0, 10, 'Ax-b should be zero')
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
