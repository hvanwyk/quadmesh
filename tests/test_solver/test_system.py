import unittest
from assembler import Assembler
from assembler import GaussRule
from fem import QuadFE
from fem import DofHandler
from mesh import Mesh, DCEL, QuadMesh, Mesh2D
import numpy as np
import numpy.linalg as la

'''             
class TestSystem(unittest.TestCase):
    """
    Test Assembler class
    
    Purpose:
    
        - To help assemble system, matrices, residuals, jacobians
        - To incorporate hanging nodes
        - To assemble samples of systems
        
    Scope:
    
        - Dimensions: 1D and 2D
        - Boundary conditions: Neumann, Dirichlet, Robin
        - Should account for periodicity  
        - Linear/Noninear Assembly
        
    Features:
        - Assembly of multiple systems (sampling)
        - Return multiple constituent system matrices
        
    Methods: 
    
        - assemble
        - extract hanging nodes
        - resolve hanging nodes
        
        
    Tests: 
    
        Single Interval, QuadCell
            - Integration (entries)
            - Solution
            - Boundary Conditions
        
        2 Intervals, 2x2 QuadMesh
             - periodicity
    """
    

    def test_assemble(self):
        
        # =====================================================================
        # One Cell
        # =====================================================================
        mesh = Mesh2D()
        
        # ---------------------------------------------------------------------
        # Piecewise Linear
        # ---------------------------------------------------------------------
        V = QuadFE(2,'Q1')
        s = Assembler(mesh,V, n_gauss=(3,9))
        #
        # Mass Matrix
        # 
        lf = [(1,'v')]
        bf = [(1,'u','v')]
        A,b = s.assemble(bf,lf)
        AA = 1/36.0*np.array([[4,2,2,1],[2,4,1,2],[2,1,4,2],[1,2,2,4]])
        self.assertTrue(np.allclose(A.toarray(),AA),'Incorrect mass matrix')
        b_check = 0.25*np.array([1,1,1,1])
        self.assertTrue(np.allclose(b,b_check),'Right hand side incorrect')
        #
        # Stiffness Ax
        # 
        bf = [(1,'ux','vx')]
        Ax = s.assemble(bilinear_forms=bf)
        AAx = 1/6.0*np.array([[2,-2,1,-1],[-2,2,-1,1],[1,-1,2,-2],[-1,1,-2,2]])
        self.assertTrue(np.allclose(Ax.toarray(),AAx),
                               'Incorrect stiffness matrix')
        #
        # Stiffness Ay
        # 
        bf = [(1,'uy','vy')]
        A = s.assemble(bilinear_forms=bf)
        AAy = 1/6.0*np.array([[2,1,-2,-1],[1,2,-1,-2],[-2,-1,2,1],[-1,-2,1,2]])
        self.assertTrue(np.allclose(A.toarray(),AAy), 'Ay incorrect')
        #
        # Use matrices to integrate
        #
        q = lambda x,y: x*(1-x)*y*(1-y)
        bilinear_forms = [(q,'u','v')]
        linear_forms = [(1,'v')]
        A,dummy = s.assemble(bilinear_forms, linear_forms) 
        v = np.array([1.,1.,1.,1.])
        self.assertAlmostEqual(np.dot(v,A.tocsr().dot(v))-1.0/36.0, 0,8,\
                               'Should integrate to 4/pi^2.')
        
        # ---------------------------------------------------------------------
        # Higher order: Test with boundary conditions
        # ---------------------------------------------------------------------        
        def m_neumann(edge):
            """
            Neumann Edge Marker: x = 1
            """
            x = np.array(edge.vertex_coordinates())
            return (np.abs(x[:,0]-1)<1e-9).all()
        
        def g_neumann(x,y):
            """
            Neumann function
            """        
            return -y*(1-y)
                
        def m_robin_1(edge):
            """
            Robin Edge Marker: y = 0 
            """
            x = np.array(edge.vertex_coordinates())
            return (np.abs(x[:,1]-0)<1e-9).all()  
        
        def g_robin_1(x,y):
            """
            Robin boundary conditions for y = 0
            """
            return -x*(1-x)
        
        def m_robin_2(edge):
            """
            Robin Edge Marker: y = 1
            """
            x = np.array(edge.vertex_coordinates())
            return (np.abs(x[:,1]-1)<1e-9).all() 
        
        def g_robin_2(x,y):
            """
            Robin boundary conditions for y = 1
            """
            return -0.5*x*(1-x)
        
        def m_dirichlet(x,y):
            """
            Dirichlet Node Marker: x = 0
            """
            return (np.abs(x)<1e-10)
            
        def g_dirichlet(x,y):
            """
            Dirichlet function
            """
            return np.zeros(shape=x.shape)
               
        gamma_1 = 1.0
        gamma_2 = 2.0
        u = lambda x,y: x*(1-x)*y*(1-y)  # exact solution
        f = lambda x,y: 2.0*(x*(1-x)+y*(1-y))+u(x,y)  # forcing term
        bf = [(1,'ux','vx'),(1,'uy','vy'),(1,'u','v')]  # bilinear forms
        lf = [(f,'v')]  # linear forms
        cell = mesh.root_node().cell()
        node = mesh.root_node()
        for etype in ['Q2','Q3']:
            element = QuadFE(2,etype)
            s = Assembler(mesh,element)
            x = s.dof_vertices()
            ui = u(x[:,0],x[:,1])
            n_dofs = s.n_dofs()
            #
            # Assemble without boundary conditions
            # 
            A, b = s.assemble(bilinear_forms=bf, linear_forms=lf)
            AA = s.form_eval((1,'u','v'), node)
            AAx = s.form_eval((1,'ux','vx'),node)
            AAy = s.form_eval((1,'uy','vy'),node)
            bb  = s.form_eval((f,'v'),node)
            self.assertTrue(np.allclose(AA+AAx+AAy,A.toarray()), 
                            'System matrix not correct')
            self.assertTrue(np.allclose(bb,b),'Forcing term incorrect.') 
    
        
            #
            # Add Neumann
            #
            bc_1 = {'dirichlet': None, 'neumann': [(m_neumann,g_neumann)],
                    'robin': None}
            A,b = s.assemble(bilinear_forms=bf, linear_forms=lf,\
                             boundary_conditions=bc_1)
            bb_neu = s.form_eval((g_neumann,'v'), node, edge_loc='E')
            self.assertTrue(np.allclose(bb+bb_neu,b),'Forcing term incorrect.')
            
            #
            # Add Robin
            # 
            bc_2 = {'dirichlet': None,\
                    'neumann':   [(m_neumann,g_neumann)], \
                    'robin':     [(m_robin_1, (gamma_1,g_robin_1)),\
                                  (m_robin_2,(gamma_2,g_robin_2))],\
                    'periodic':  None}
            A,b = s.assemble(bilinear_forms=bf, linear_forms=lf, \
                             boundary_conditions=bc_2)
            
            bb_R1 = gamma_1*s.form_eval((g_robin_1,'v'), node, edge_loc='S')
            AA_R1 = gamma_1*s.form_eval((g_robin_1,'u','v'),node,edge_loc='S')
        
            bb_R2 = gamma_2*s.form_eval((g_robin_2,'v'),node,edge_loc='N')
            AA_R2 = gamma_2*s.form_eval((g_robin_2,'u','v'),node, edge_loc='N')
            
            
            self.assertTrue(np.allclose(AA+AAx+AAy+AA_R1+AA_R2,A.toarray()), 
                            'System matrix not correct')
            self.assertTrue(np.allclose(bb+bb_neu+bb_R1+bb_R2,b),\
                            'Forcing term incorrect.')
            
            #
            # Add Dirichlet
            #
            bc_3 = {'dirichlet': [(m_dirichlet,g_dirichlet)],\
                    'neumann': [(m_neumann,g_neumann)], \
                    'robin': [(m_robin_1, (gamma_1,g_robin_1)),\
                              (m_robin_2,(gamma_2,g_robin_2))],\
                    'periodic': None}
            A,b = s.assemble(bilinear_forms=bf,linear_forms=lf,\
                             boundary_conditions=bc_3)
            AAA = AA+AAx+AAy+AA_R1+AA_R2
            bbb = bb+bb_neu+bb_R1+bb_R2
            #
            # Explicitly enforce boundary conditions
            # 
            i_dir = s.get_edge_dofs(node, 'W')    
            for i in range(n_dofs):
                if i in i_dir:
                    bbb[i] = ui[i]
                for j in range(n_dofs):
                    if i in i_dir:
                        if j==i:
                            AAA[i,j] = 1
                        else:
                            AAA[i,j] = 0
                    else:
                        if j in i_dir:
                            bb[i] -= AAA[i,j]*ui[j]
                            AAA[i,j] = 0
            self.assertTrue(np.allclose(AAA,A.toarray()),\
                            'System matrix incorrect')
            self.assertTrue(np.allclose(bbb,b),\
                            'Right hand side incorrect')
            
            #
            # Check solution
            # 
            ua = la.solve(A.toarray(),b)
            self.assertTrue(np.allclose(ui,ua),\
                            'Solution incorrect.')
      
            # Dirichlet all round
            def m_bnd_nodes(x,y):
                tol = 1e-8
                return ((np.abs(x) < tol) | (np.abs(x-1) < tol) \
                    | (np.abs(y) < tol) | (np.abs(y-1) < tol))  
                
            
            g_dir = lambda x,y: np.zeros(shape=x.shape)
                    
            bc = {'dirichlet': [(m_bnd_nodes,g_dir)],
                  'neumann': None,
                  'robin': None,
                  'periodic': None}
            A,b = s.assemble(bilinear_forms=bf, linear_forms=lf, \
                             boundary_conditions=bc)
            ua = la.solve(A.toarray(),b)
            self.assertTrue(np.allclose(ua,ui), 'Solution incorrect')
            
        # =====================================================================
        # Multiple Cells
        # =====================================================================
        #
        # Test by integration
        # 
        mesh = QuadMesh(resolution=(2,2))
        
        trial_functions = {'Q1': lambda x,y: (x-1),
                           'Q2': lambda x,y: x*y**2,
                           'Q3': lambda x,y: x**3*y}
        test_functions = {'Q1': lambda x,y: x*y, 
                          'Q2': lambda x,y: x**2*y, 
                          'Q3': lambda x,y: x**3*y**2}
        integrals = {'Q1': [-1/12,1/2,0], 
                     'Q2': [1/16, 1/4, 1/4],
                     'Q3': [1/28,9/20,1/7]}
        bf_list = [(1,'u','v'),(1,'ux','vx'),(1,'uy','vy')]
        
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            s = Assembler(mesh,element)
            x = s.dof_vertices()
            u = trial_functions[etype]
            v = test_functions[etype]
            ui = u(x[:,0],x[:,1])
            vi = v(x[:,0],x[:,1])
            n_nodes = s.n_dofs()
            for i in range(3):
                A = s.assemble(bilinear_forms=[bf_list[i]])
                AA = np.zeros((n_nodes,n_nodes))
                for node in mesh.root_node().get_leaves():
                    cell_dofs = s.get_global_dofs(node)
                    AA_loc = s.form_eval(bf_list[i], node)
                    block = np.ix_(cell_dofs,cell_dofs)
                    AA[block] = AA[block] + AA_loc 
                
                self.assertAlmostEqual(vi.dot(AA.dot(ui)),\
                                       integrals[etype][i], 8,\
                                       'Manual assembly incorrect.')
                self.assertTrue(np.allclose(A.toarray(),AA),\
                                'System matrix different')
                self.assertAlmostEqual(vi.dot(A.toarray().dot(ui)),\
                                       integrals[etype][i], 8,\
                                       'Assembly incorrect.')
          
        #
        # 10x10 grid     
        # 
        mesh = Mesh(grid=DCEL(resolution=(10,10)))
        mesh.refine()
        u = lambda x,y: x*(1-x)*y*(1-y)  # exact solution
        f = lambda x,y: 2.0*(x*(1-x)+y*(1-y))+u(x,y)  # forcing term 
        for etype in ['Q2','Q3']:
            element = QuadFE(2,etype)
            system = Assembler(mesh,element)
            A,b = system.assemble(bilinear_forms=bf, linear_forms=lf,\
                                  boundary_conditions=bc)
            ua = la.solve(A.toarray(), b)
            x = system.dof_vertices()
            ue = u(x[:,0], x[:,1])
            self.assertTrue(np.allclose(ua, ue), 'Solution incorrect')
        
            A,b = system.assemble(bilinear_forms=bf, linear_forms=lf,\
                                  boundary_conditions=bc_3)
            ua = la.solve(A.toarray(),b)
            self.assertTrue(np.allclose(ua,ue), 'Solution incorrect')
        
        # =====================================================================
        # Test hanging nodes
        # =====================================================================
        mesh = Mesh()
        mesh.root_node().mark(1)
        mesh.refine(1)
        mesh.root_node().children['SW'].mark(2)
        mesh.refine(2)
        element = QuadFE(2,'Q2')
        
                
        system = Assembler(mesh,element)
        A,b = system.assemble(bilinear_forms=bf, linear_forms=lf,\
                              boundary_conditions=bc)
        
        # TODO: FIXME
        #print(A.shape)
        #dh = DofHandler(mesh,element)
        #hn = dh.get_hanging_nodes()
        #print('======================')
        #for h in hn.keys():
        #    print(h)
        #print('======================')
        A,b = system.extract_hanging_nodes(A,b,compress=False)
        
        #print(A.shape)
        ua = la.solve(A.toarray(),b)
        #ua = system.resolve_hanging_nodes(uua)
        x = system.dof_vertices()
        ue = u(x[:,0],x[:,1])
        self.assertTrue(np.allclose(ua,ue), 'Solutions not close.')
        #plot.function(ax, ua, mesh, element=element)
        #plt.show()
        
        
    def test_extract_hanging_nodes(self):
        """
        A = np.array([[1,1,1,1],[0,-2,1,0],[1,3,1,0],[0,0,-1,3]])
        b = np.array([2,4,0,-2])
        xe = np.array([1,-1,2,0])
        hanging_nodes = {2:([0,1],[1,-1])}
        
        A1 = sp.lil_matrix(A)
        A1,b1 = self.extract_hanging_nodes(A1,b)
        
        A2 = sp.lil_matrix(A)
        A2,b2 = self.incporate_hanging_nodes(A2,b)
        """
    
    def test_resolve_hanging_nodes(self):
        pass
    
        
    def test_get_n_nodes(self):
        pass
    
    
    def test_get_global_dofs(self):
        pass
    
    
    def test_get_edge_dofs(self):
        pass
    
    
    def test_dof_vertices(self):
        pass
    
    
    def test_x_loc(self):
        pass
    
    
    def test_bilinear_loc(self):
        pass
     
    
    def test_linear_loc(self):
        pass
    
    
    def test_shape_eval(self):
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
        edge_integrals_west = {'Q1': [-0.5,-0.5,1.0],
                               'Q2': [-1.0,0.0,0.0],
                               'Q3': [-0.25,0.0,-1.0]} 
        derivatives = [(0,),(1,0),(1,1)]
        mesh = QuadMesh()
        cell = mesh.root_node().cell() 
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            system = Assembler(mesh,element)
            n_dofs = element.n_dofs()
            x_ref = element.reference_nodes()
            #
            # Sanity check
            # 
            I = np.eye(n_dofs)
            self.assertTrue(np.allclose(system.shape_eval(x_ref=x_ref),I),\
                            'Shape functions incorrect at reference nodes.')
            y = np.random.rand(5,2)
            weights = system.cell_rule().weights()
            f_nodes = test_functions[etype][0](x_ref[:,0],x_ref[:,1])
            for i in range(3):
                phi = system.shape_eval(derivatives=derivatives[i], x_ref=y)
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
                phi = system.shape_eval(derivatives=derivatives[i])  
                self.assertAlmostEqual(np.dot(weights,np.dot(phi,f_nodes)),\
                                 cell_integrals[etype][i],places=8,\
                                 msg='Incorrect integral.')
            #
            # On Edges   
            # 
            y = np.random.rand(5)
            for direction in ['W','E','S','N']:
                edge = cell.get_edges(direction)
                weights = system.edge_rule().weights()*\
                    system.edge_rule().jacobian(edge)
                #
                # Sanity check
                # 
                edge_dofs = element.loc_dofs_on_edge(direction)
                x_ref_edge = x_ref[edge_dofs,:]  # element nodes on edge
                phi = system.shape_eval(x_ref=x_ref_edge)
                self.assertTrue(np.allclose(phi, I[edge_dofs,:]), \
                                'Shape function incorrect at edge ref nodes.')
                y_phys = system.edge_rule().map(edge, y)
                f_nodes = test_functions[etype][0](x_ref[:,0],x_ref[:,1])
                for i in range(3):
                    #
                    # Interpolation
                    # 
                    phi = system.shape_eval(derivatives=derivatives[i],\
                                            x_ref=y_phys)
                    f = test_functions[etype][i]
                    fvals = f(y_phys[:,0],y_phys[:,1])
                    self.assertTrue(np.allclose(np.dot(phi,f_nodes),fvals),\
                                'Shape function interpolation failed.') 
                    #
                    # Quadrature
                    # 
                    if direction == 'W':
                        phi = system.shape_eval(derivatives=derivatives[i],\
                                                edge_loc=direction)
                        self.assertAlmostEqual(np.dot(weights,np.dot(phi,f_nodes)),\
                                 edge_integrals_west[etype][i],places=8,\
                                 msg='Incorrect integral.')
        #
        # Over arbitrary cell
        #
        mesh = Mesh(grid=DCEL(box=[1,4,1,3]))
        cell = mesh.root_node().cell()
        y = np.random.rand(5,2)
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            system = Assembler(mesh,element)
            y_phys = system.cell_rule().map(cell, x=y)
            x_ref = system.dof_vertices()
            f_nodes = test_functions[etype][0](x_ref[:,0],x_ref[:,1]) 
            for i in range(3):
                #
                # Interpolation
                #  
                phi = system.shape_eval(derivatives=derivatives[i],\
                                        cell=cell, x=y_phys)
                f_vals = test_functions[etype][i](y_phys[:,0],y_phys[:,1])
                self.assertTrue(np.allclose(np.dot(phi,f_nodes),f_vals),\
                                'Shape function interpolation failed.')
        mesh = Mesh(grid=DCEL(box=[0,0.5,0,0.5]))
        cell = mesh.root_node().cell()
        node = mesh.root_node()
        u = lambda x,y: x*y**2
        v = lambda x,y: x**2*y
        element = QuadFE(2,'Q2')
        system = Assembler(mesh,element)
        x = system.dof_vertices()
        ui = u(x[:,0],x[:,1])
        vi = v(x[:,0],x[:,1])
        
        phi = system.shape_eval(cell=cell)
        uhat = phi.dot(ui)
        vhat = phi.dot(vi)
        weights = system.cell_rule().weights()*\
                  system.cell_rule().jacobian(cell)
        A = system.form_eval((1,'u','v'), node)
        self.assertAlmostEqual(vi.dot(A.dot(ui)), 0.000244141,8,\
                               'Local bilinear form integral incorrect.')
        self.assertAlmostEqual(vi.dot(A.dot(ui)), np.sum(uhat*vhat*weights),8,\
                               'Local bilinear form integral does not match quad.')
        
    def test_f_eval(self):
        """
        TODO: Finish, mesh function and derivatives
        TODO: Superfluous with Function class
        """
        test_functions = {'Q1': lambda x,y: (2+x)*(y-3),
                          'Q2': lambda x,y: (2+x**2+x)*(y-2)**2,
                          'Q3': lambda x,y: (2*x**3-3*x)*(y**2-2*y)}
        mesh = Mesh(grid=DCEL(resolution=(5,5)))
        mesh.refine()
        x_test = np.random.rand(10,2)
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            system = Assembler(mesh,element)
            f = test_functions[etype]
            f_test = f(x_test[:,0],x_test[:,1])
            # 
            # Function given explicitly
            # 
            self.assertTrue(np.allclose(system.f_eval(f, x_test),f_test,1e-10),\
                            'Explicit function not correctly interpolated')
            #
            # Nodal function
            #
            x = system.dof_vertices()
            fn = f(x[:,0],x[:,1])
            self.assertTrue(np.allclose(system.f_eval(fn, x_test),f_test,1e-10),\
                            'Explicit function not correctly interpolated')
            
            #
            # Mesh function
            # 
            
    def test_f_eval_loc(self):
        mesh = Mesh()
        cell = mesh.root_node().cell()
        node = mesh.root_node()
        test_functions = {'Q1': lambda x,y: (2+x)*(y-3),
                          'Q2': lambda x,y: (2+x**2+x)*(y-2)**2,
                          'Q3': lambda x,y: (2*x**3-3*x)*(y**2-2*y)} 
        cell_integrals = {'Q1': -25/4, 'Q2': 119/18, 'Q3': 2/3}
        n_edge_integrals = {'Q1': -5, 'Q2': 17/6 ,'Q3': 1}
        x_test = np.random.rand(5,2)
    
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            system = Assembler(mesh,element)
            f = test_functions[etype]             
            f_test = f(x_test[:,0],x_test[:,1])
            # -----------------------------------------------------------------
            # Interpolation
            # -----------------------------------------------------------------
            #
            # f in functional form                   
            # 
            f_loc = system.f_eval_loc(f, node=node,x=x_test)
            self.assertTrue(np.allclose(f_loc,f_test),\
                                   'Function not correctly interpolated.')
        
            #
            # f in nodal form
            #
            local_dofs = system.get_global_dofs(node)
            x = system.dof_vertices()
            x_loc = x[local_dofs,:]
            f_nodes = f(x_loc[:,0],x_loc[:,1])
            f_loc = system.f_eval_loc(f_nodes,node=node,x=x_test)
            self.assertTrue(np.allclose(f_loc,f_test), \
                                   'Function not correctly interpolated.')
        
            # -----------------------------------------------------------------
            # Quadrature
            # -----------------------------------------------------------------
            cell_rule = system.cell_rule() 
            wg = cell_rule.weights()*cell_rule.jacobian(cell)
            fg = system.f_eval_loc(f,node=node)
            self.assertAlmostEqual(np.sum(fg*wg), cell_integrals[etype],\
                                   8, 'Cell integral incorrect')
            
            n_edge = cell.get_edges('N')
            edge_rule = system.edge_rule()
            wg = edge_rule.weights()*edge_rule.jacobian(n_edge)
            fg = system.f_eval_loc(f,node=node, edge_loc='N')
            self.assertAlmostEqual(np.sum(fg*wg), n_edge_integrals[etype],\
                                   8, 'Cell integral incorrect')

            
    def test_form_eval(self):
        mesh = Mesh(grid=DCEL(box=[1,2,1,2]))
        trial_functions = {'Q1': lambda x,y: (x-1),
                           'Q2': lambda x,y: x*y**2,
                           'Q3': lambda x,y: x**3*y}
        test_functions = {'Q1': lambda x,y: x*y, 
                          'Q2': lambda x,y: x**2*y, 
                          'Q3': lambda x,y: x**3*y**2}
        #
        # Integrals over current cell
        # 
        cell_integrals = {'Q1': [5/4,3/4], 
                          'Q2': [225/16,35/2], 
                          'Q3': [1905/28,945/8]}
        edge_integrals = {'Q1': [3,3/2], 
                          'Q2': [30,30], 
                          'Q3': [240,360]}
        cell = mesh.root_node().cell() 
        node = mesh.root_node()
        f = lambda x,y: (x-1)*(y-1)**2
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            system = Assembler(mesh,element)
            x_loc = system.x_loc(cell)
            u = trial_functions[etype](x_loc[:,0],x_loc[:,1])
            v = test_functions[etype](x_loc[:,0],x_loc[:,1])
            
            #
            # Bilinear form
            #
            b_uv = system.form_eval((1,'u','v'), node)
            self.assertAlmostEqual(v.dot(b_uv.dot(u)),
                                   cell_integrals[etype][0],8, 
                                   '{0}: Bilinear form (1,u,v) incorrect.'\
                                   .format(etype))
            
            
            b_uvx = system.form_eval((1,'u','vx'), node)
            self.assertAlmostEqual(v.dot(b_uvx.dot(u)),
                                   cell_integrals[etype][1],8, 
                                   '{0}: Bilinear form (1,u,vx) incorrect.'\
                                   .format(etype))
            
            
            #
            # Edges
            #
            be_uv = system.form_eval((1,'u','v'), node, edge_loc='E')
            self.assertAlmostEqual(v.dot(be_uv.dot(u)),
                                   edge_integrals[etype][0],8, 
                                   '{0}: Bilinear form (1,u,v) incorrect.'\
                                   .format(etype))
            
            be_uvx = system.form_eval((1,'u','vx'), node, edge_loc='E')
            self.assertAlmostEqual(v.dot(be_uvx.dot(u)),
                                   edge_integrals[etype][1],8, 
                                   '{0}: Bilinear form (1,u,vx) incorrect.'\
                                   .format(etype))
            #
            # Linear form
            #
            
            # cell
            f = trial_functions[etype] 
            f_v = system.form_eval((f,'v'), node)
            self.assertAlmostEqual(f_v.dot(v), cell_integrals[etype][0],8, 
                                   '{0}: Linear form (f,v) incorrect.'\
                                   .format(etype))
            
            
            f_vx = system.form_eval((f,'vx'), node)
            self.assertAlmostEqual(f_vx.dot(v), cell_integrals[etype][1],8, 
                                   '{0}: Linear form (f,vx) incorrect.'\
                                   .format(etype))
            
            # edges
            fe_v = system.form_eval((f,'v'), node, edge_loc='E')
            self.assertAlmostEqual(fe_v.dot(v), edge_integrals[etype][0],8, 
                                   '{0}: Linear form (f,v) incorrect.'\
                                   .format(etype))
            
            fe_vx = system.form_eval((f,'vx'), node, edge_loc='E')
            self.assertAlmostEqual(fe_vx.dot(v), edge_integrals[etype][1],8, 
                                   '{0}: Linear form (f,vx) incorrect.'\
                                   .format(etype))
            
        #
        # A general cell        
        # 
        mesh = Mesh(grid=DCEL(box=[1,4,1,3]))
        element = QuadFE(2,'Q1')
        system = Assembler(mesh,element)
        cell = mesh.root_node().cell()
        node = mesh.root_node()
        A = system.form_eval((1,'ux','vx'),node)
        #
        # Use form to integrate
        # 
        u = trial_functions['Q1']
        v = test_functions['Q1']
        x = system.dof_vertices()
        ui = u(x[:,0],x[:,1])
        vi = v(x[:,0],x[:,1])
        self.assertAlmostEqual(vi.dot(A.dot(ui)), 12, 8, 'Integral incorrect.')
    
    
    def test_cell_rule(self):
        pass
    
    
    def test_edge_rule(self):
        pass
    
    """          
    def test_make_generic(self):
        mesh = Mesh.newmesh()
        element = QuadFE(2,'Q1')
        system = Assembler(mesh, element)
        cell = mesh.root_node().cell()
        self.assertEqual(system.make_generic(cell), 'cell', \
                         'Cannot convert cell to "cell"')
        for direction in ['W','E','S','N']:
            edge = cell.get_edges(direction)
            self.assertEqual(system.make_generic((edge,direction)),\
                             ('edge',direction),\
                             'Cannot convert edge to generic edge')
    """
    
            
    def test_parse_derivative_info(self):
        mesh = Mesh()
        element = QuadFE(2,'Q2')
        system = Assembler(mesh,element)
        self.assertEqual(system.parse_derivative_info('u'), (0,),\
                         'Zeroth derivative incorrectly parsed')
        self.assertEqual(system.parse_derivative_info('ux'), (1,0),\
                         'Zeroth derivative incorrectly parsed')
        self.assertEqual(system.parse_derivative_info('vy'), (1,1),\
                         'Zeroth derivative incorrectly parsed')
    
        
    def test_interpolate(self):
        mesh = Mesh()
        mesh.refine()
        mesh.record()  # label 0
        
        mesh.refine()
        mesh.root_node().children['SW'].children['NE'].mark('r')
        mesh.refine('r')
        mesh.record()  # label 1
        
        functions = {'Q1': lambda x,y: 2*x - 3*y, 
                     'Q2': lambda x,y: 2*x**2*y - 3*y**2 + 2,\
                     'Q3': lambda x,y: x**3 + y**3 - 2*x*y**2}
        
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)
            system = Assembler(mesh, element, nested=True)
            x_coarse = system.dofhandler().dof_vertices(flag=0)
            x_fine = system.dofhandler().dof_vertices(flag=1)
            ufn = functions[etype]
            u_coarse = ufn(x_coarse[:,0],x_coarse[:,1])
            u_fine = ufn(x_fine[:,0],x_fine[:,1])
            uh_fine = system.interpolate(0,1, u_coarse)
            
            #
            # Direct interpolation
            # 
            self.assertTrue(np.allclose(u_fine,uh_fine,1e-9),\
                            'Interpolant should match fine function')
            # 
            # Interpolation matrix
            # 
            I = system.interpolate(0,1)
            I = I.tocsc()
            n_dofs_coarse = system.dofhandler().n_dofs(0)
            n_dofs_fine = system.dofhandler().n_dofs(1)
            self.assertEqual(I.shape,(n_dofs_fine,n_dofs_coarse))
            self.assertAlmostEqual(n_dofs_coarse, x_coarse.shape[0], 10,\
                                   'Number of coarse nodes not equal n_dofs.')
            self.assertTrue(np.allclose(u_fine,I.dot(u_coarse),1e-9),\
                            'Interpolation matrix incorrect.')
            #
            # Restrict to coarse dofs
            # 
            R = system.restrict(0, 1)
            self.assertTrue(np.allclose(np.dot(R,u_fine),u_coarse,1e-9))
'''
   