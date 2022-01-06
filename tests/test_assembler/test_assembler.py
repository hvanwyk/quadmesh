from assembler import Assembler, Form, Kernel, IIForm, IPForm, GaussRule
from fem import QuadFE, DofHandler, Basis
from function import Constant, Nodal, Explicit
from mesh import QuadMesh, Mesh1D, QuadCell, HalfEdge, Vertex, convert_to_array

import unittest
import numpy as np


class TestAssembler(unittest.TestCase):
    """
    Test Assembler Class
    """
    def test_constructor(self):
        pass
        
    def test_shape_info(self):
        """
        Test whether system extracts the correct information
        """
        #
        # 1D 
        # 
        mesh = Mesh1D(resolution=(1,))
        Q1 = QuadFE(1,'Q1')
        dhQ1 = DofHandler(mesh, Q1)
        dhQ1.distribute_dofs()
        
        ux = Basis(dhQ1, 'ux')
        v =  Basis(dhQ1, 'v')
        system = Assembler(Form(trial=ux, test=v))
        
        cell = mesh.cells.get_child(0)
        
        si = system.shape_info(cell)
        self.assertTrue(cell in si)
        self.assertTrue(ux in si[cell])
        self.assertTrue(v in si[cell])
        
        
        #
        # 2D
        # 
        mesh = QuadMesh(resolution=(1,1))
        Q1 = QuadFE(2, 'Q1')
        Q2 = QuadFE(2, 'Q2')
        
        dhQ1 = DofHandler(mesh, Q1)
        dhQ1.distribute_dofs()
        
        dhQ2 = DofHandler(mesh, Q2)
        dhQ2.distribute_dofs()
        
        phiQ1 = Basis(dhQ1)
        phiQ2 = Basis(dhQ2)
        
        # Kernel
        f = Nodal(lambda x:x[:,0]*x[:,1], basis=phiQ2)
        g = Nodal(lambda x:x[:,0]+x[:,1], basis=phiQ1)
        F = lambda f,g: f-g
        kernel = Kernel([f,g], derivatives=['fx','g'], F=F)
        
        # Form defined over HalfEdge with flag 1
        u = Basis(dhQ1, 'u')
        vx = Basis(dhQ2, 'vx')
        v = Basis(dhQ2, 'v') 
        
        form_1 = Form(kernel, trial=u, test=vx, dmu='ds', flag=1)
        form_2 = Form(trial=u, test=v, dmu='dx', flag=2)
                      
        problem = [form_1, form_2]
                      
        # Assembler
        system = Assembler(problem, mesh)
        
        # Shape info on cell 
        cell = mesh.cells.get_child(0)
        si = system.shape_info(cell)
        
        # Empty info dictionary (Cell hasn't been marked)
        self.assertEqual(si, {})
        
        # Mark HalfEdge and cell and do it again
        he = cell.get_half_edge(0)
        he.mark(1)
        cell.mark(2)
        si = system.shape_info(cell)
        
        # Check that shape info contains the right stuff
        for region in [he, cell]:
            self.assertTrue(region in si)
        
        for basis in [u,vx]:    
            self.assertTrue(basis in si[he])
            
        for basis in [u,v]:
            self.assertTrue(basis in si[cell])
        
    def test_shape_eval(self):
        """
        Routine evaluates all shape functions on given cell 
        """
        #
        # Diamond-shaped region
        #
        
        # Vertices 
        A = Vertex((0,-1))
        B = Vertex((1,0))
        C = Vertex((0,1))
        D = Vertex((-1,0))
        
        # Half-edges
        AB = HalfEdge(A,B)
        BC = HalfEdge(B,C)
        CD = HalfEdge(C,D)
        DA = HalfEdge(D,A)
        
        # Cell
        cell = QuadCell([AB,BC,CD,DA])
        
        # 1D Quadrature Rule
        rule = GaussRule(4,shape='interval')
        xx = rule.nodes()
        ww = rule.weights()
        
        #
        # Map rule to physical domain (CD)
        #
        xg, mg = CD.reference_map(xx, jac_r2p=True)
        xg = convert_to_array(xg)
        
        # Modify weights
        jac = mg['jac_r2p']
        wg = ww*np.array(np.linalg.norm(jac[0]))
        
        # Check length of edge is sqrt(2)
        self.assertTrue(np.allclose(np.sum(wg),np.sqrt(2)))
        
        # Check Int x^2 y on CD
        f = lambda x,y: x**2*y
        fx = f(xg[:,0],xg[:,1])
        
        self.assertTrue(np.allclose(np.sum(fx*wg), np.sqrt(2)*(1/12)))
        
        #
        # Use shape functions to evaluate line integral
        #
        
        # 
        # Map 1D rule onto reference cell 
        # 
        Q = QuadFE(2,'Q1')
        rcell = Q.reference_cell()
        
        # Determine equivalent Half-edge on reference element 
        i_he = cell.get_half_edges().index(CD)
        ref_he = rcell.get_half_edge(i_he)
        
        # Get 2D reference nodes
        b,h = convert_to_array(ref_he.get_vertices())
        x_ref = np.array([b[i]+xx*(h[i]-b[i]) for i in range(2)]).T
        
        # Map 2D reference point to phyisical cell
        xxg, mg = cell.reference_map(x_ref, jac_r2p=False, 
                                     jac_p2r=True, hess_p2r=True)
        
        self.assertTrue(np.allclose(xxg,xg))
        
        
        # Evaluate the shape functions 
        phi = Q.shape(x_ref,cell,[(0,),(1,0),(1,1)],mg['jac_p2r'])
        
        # x = phi1 - phi3
        self.assertTrue(np.allclose(phi[0][:,1]-phi[0][:,3],xg[:,0]))
        
        # y = -phi0 + phi2
        self.assertTrue(np.allclose(-phi[0][:,0]+phi[0][:,2],xg[:,1]))
        
        # g(x,y) = x - y = phi*[1,1,-1-1]
        c = np.array([1,1,-1,-1])
        g = phi[0].dot(c)
        
        # Int_CD x-y ds
        self.assertTrue(np.allclose(np.sum(wg*g),-np.sqrt(2)))
        
        # Integrals involving phi_x, phi_y
        gx = phi[1].dot(c)
        gy = phi[2].dot(c)
        
        n = CD.unit_normal()
        
        self.assertTrue(np.allclose(np.sum(wg*(gx*n[0]+gy*n[1])),-2))
         
           
    def test_integrals_1d(self):
        """
        Test system assembly
        """ 
        #
        # Constant form
        #
        
        # Mesh 
        mesh = Mesh1D(box=[1,2], resolution=(1,))
        
        # Kernel
        kernel = Kernel(Explicit(f=lambda x:x[:,0], dim=1))
        
        problem = Form(kernel)
        assembler = Assembler(problem, mesh=mesh)
        assembler.assemble()
        self.assertAlmostEqual(assembler.get_scalar(),3/2)
        
        
        #
        # Linear forms (x,x) and (x,x') over [1,2] = 7/3, 3/2
        # 
        
        # Elements
        Q1 = QuadFE(mesh.dim(),'Q1')
        Q2 = QuadFE(mesh.dim(),'Q2')
        Q3 = QuadFE(mesh.dim(),'Q3')
        
        # Dofhandlers
        dQ1 = DofHandler(mesh, Q1)
        dQ2 = DofHandler(mesh, Q2)
        dQ3 = DofHandler(mesh, Q3)
        
        # Distribute dofs
        [d.distribute_dofs() for d in [dQ1, dQ2, dQ3]]
        
        for dQ in [dQ1, dQ2, dQ3]:
            # Basis
            phi = Basis(dQ, 'u')
            phi_x = Basis(dQ, 'ux')
            
            # Kernel function
            xfn = Nodal(f=lambda x: x[:,0], basis=phi)
            
            # Kernel 
            kernel = Kernel(xfn)
            
            # Form
            problem = [[Form(kernel, test=phi)],
                       [Form(kernel, test=phi_x)]]
            
            # Assembly
            assembler = Assembler(problem)
            assembler.assemble()
            
            # Check b^Tx = (x,x)
            b0 = assembler.get_vector(0)
            self.assertAlmostEqual(np.sum(b0*xfn.data()[:,0]),7/3)
            
            b1 = assembler.get_vector(1)
            self.assertAlmostEqual(np.sum(b1*xfn.data()[:,0]),3/2)
            
        #
        # Bilinear forms
        #
        # Compute (1,x,x) = 7/3, or (x^2, 1, 1) = 7/3 
        
        for dQ in [dQ1, dQ2, dQ3]:
            # Basis
            phi = Basis(dQ,'u')
            phi_x = Basis(dQ, 'ux')
            
            # Kernel function 
            x2fn = Explicit(f=lambda x: x[:,0]**2, dim=1)
            xfn = Nodal(f=lambda x: x[:,0], basis=phi)
            
            # Form 
            problems = [[Form(1, test=phi, trial=phi)], 
                        [Form(Kernel(xfn), test=phi, trial=phi_x)],
                        [Form(Kernel(x2fn), test=phi_x, trial=phi_x)]]
            
            # Assemble
            assembler = Assembler(problems)
            assembler.assemble()
            
            x = xfn.data()[:,0]
            for i_problem in range(3):
                A = assembler.get_matrix(i_problem)
                self.assertAlmostEqual(x.T.dot(A.dot(x)), 7/3)
            
            
        # ======================================================================
        # Test 1: Assemble simple bilinear form (u,v) on Mesh1D
        # ======================================================================
        # Mesh 
        mesh = Mesh1D(resolution=(1,))
        Q1 = QuadFE(mesh.dim(),'Q1')
        Q2 = QuadFE(mesh.dim(),'Q2')
        Q3 = QuadFE(mesh.dim(),'Q3')
        
        
        
        # Test and trial functions
        dhQ1 = DofHandler(mesh, QuadFE(1, 'Q1'))
        dhQ1.distribute_dofs()
        
        u = Basis(dhQ1, 'u')
        v = Basis(dhQ1, 'v')
        
        # Form
        form = Form(trial=u, test=v)

        # Define system
        system = Assembler(form, mesh)
        
        # Get local information
        cell = mesh.cells.get_child(0)
        si = system.shape_info(cell)
        
        # Compute local Gauss nodes
        xg,wg,phi,dofs = system.shape_eval(si, cell)
        self.assertTrue(cell in xg)
        self.assertTrue(cell in wg)
        
        # Compute local shape functions
        self.assertTrue(cell in phi)
        self.assertTrue(u in phi[cell])
        self.assertTrue(v in phi[cell])
        self.assertTrue(u in dofs[cell])
        
        # Assemble system                
        system.assemble()
        
        # Extract system bilinear form
        A = system.get_matrix()

        # Use bilinear form to integrate x^2 over [0,1]
        f = Nodal(lambda x: x, basis=u)
        fv = f.data()[:,0]
        self.assertAlmostEqual(np.sum(fv*A.dot(fv)),1/3)
        
      
        
        
        # ======================================================================
        # Test 3: Constant form (x^2,.,.) over 1D mesh
        # ======================================================================
        
        # Mesh
        mesh = Mesh1D(resolution=(10,))
        
        # Nodal kernel function
        Q2 = QuadFE(1, 'Q2')
        dhQ2 = DofHandler(mesh,Q2)
        dhQ2.distribute_dofs()
        phiQ2 = Basis(dhQ2)
        
        f = Nodal(lambda x:x**2, basis=phiQ2)
        kernel = Kernel(f=f)
        
        # Form
        form = Form(kernel = kernel)
        
        # Generate and assemble the system
        system = Assembler(form, mesh)
        system.assemble()
        
        # Check 
        self.assertAlmostEqual(system.get_scalar(),1/3)
        
        # =====================================================================
        # Test 4: Periodic Mesh
        # =====================================================================
        #
        # TODO: NO checks yet
        # 
        
        mesh = Mesh1D(resolution=(2,), periodic=True)
        
        # 
        Q1 = QuadFE(1,'Q1')
        dhQ1 = DofHandler(mesh, Q1)
        dhQ1.distribute_dofs()
        u = Basis(dhQ1, 'u')
        
        form = Form(trial=u, test=u)
        
        system = Assembler(form, mesh)
        system.assemble()
        
        
        
        # =====================================================================
        # Test 5: Assemble simple sampled form
        # ======================================================================
        mesh = Mesh1D(resolution=(3,))
        
        Q1 = QuadFE(1,'Q1')
        dofhandler = DofHandler(mesh, Q1)
        dofhandler.distribute_dofs()
        phi = Basis(dofhandler)
        
        xv = dofhandler.get_dof_vertices()
        n_points = dofhandler.n_dofs()
        
        n_samples = 6
        a = np.arange(n_samples)
        
        f = lambda x, a: a*x
        
        fdata = np.zeros((n_points,n_samples))
        for i in range(n_samples):
            fdata[:,i] = f(xv,a[i]).ravel()
            
        
        # Define sampled function
        fn = Nodal(data=fdata, basis=phi)
        kernel = Kernel(fn)
        #
        # Integrate I[0,1] ax^2 dx by using the linear form (ax,x) 
        # 
        v = Basis(dofhandler, 'v')
        form = Form(kernel=kernel, test=v)
        system = Assembler(form, mesh)
        system.assemble()
        
        one = np.ones(n_points)
        for i in range(n_samples):
            b = system.get_vector(i_sample=i)
            self.assertAlmostEqual(one.dot(b),0.5*a[i])
        
        
        #
        # Integrate I[0,1] ax^4 dx using bilinear form (ax, x^2, x)
        #
        Q2 = QuadFE(1,'Q2')
        dhQ2 = DofHandler(mesh,Q2)
        dhQ2.distribute_dofs()
        u = Basis(dhQ2, 'u')
        
        # Define form
        form = Form(kernel=kernel, test=v, trial=u)
        
        # Define and assemble system
        system = Assembler(form, mesh)
        system.assemble()
        
        # Express x^2 in terms of trial function basis
        dhQ2.distribute_dofs() 
        xvQ2 = dhQ2.get_dof_vertices()
        xv_squared = xvQ2**2
        
        for i in range(n_samples):
            #
            # Iterate over samples 
            # 
            
            # Form sparse matrix
            A = system.get_matrix(i_sample=i)
            
            # Evaluate the integral
            I = np.sum(xv*A.dot(xv_squared))
            
            # Compare with expected result
            self.assertAlmostEqual(I, 0.2*a[i])
              
                  
        # =====================================================================
        # Test 6: Working with submeshes
        # =====================================================================
        mesh = Mesh1D(resolution=(2,))
        
    
    
    def test_assemble_ipform(self):
        # =====================================================================
        # Test 7: Assemble Kernel
        # =====================================================================
        mesh = Mesh1D(resolution=(10,))
        
        Q1 = QuadFE(1,'DQ1')
        dofhandler = DofHandler(mesh, Q1)
        dofhandler.distribute_dofs()
        
        phi = Basis(dofhandler,'u')
        
        k = Explicit(lambda x,y:x*y, n_variables=2, dim=1)
        kernel = Kernel(k)
        form = IPForm(kernel, test=phi, trial=phi)
        
        assembler = Assembler(form, mesh)
        assembler.assemble()
        
        #af = assembler.af[0]['bilinear']
        M = assembler.get_matrix().toarray()
        
        u = Nodal(lambda x: x, basis=phi)
        v = Nodal(lambda x: 1-x, basis=phi)
        
        u_vec = u.data()
        v_vec = v.data()
        
        I = v_vec.T.dot(M.dot(u_vec))
        self.assertAlmostEqual(I[0,0], 1/18)
        
    
    def test_assemble_iiform(self):
        
        mesh = Mesh1D(resolution=(1,))
        
        Q1 = QuadFE(1,'DQ1')
        dofhandler = DofHandler(mesh, Q1)
        dofhandler.distribute_dofs()
        
        phi = Basis(dofhandler,'u')
        
        k = Explicit(lambda x,y:x*y, n_variables=2, dim=1)
        kernel = Kernel(k)
        
        form = IIForm(kernel, test=phi, trial=phi)
                     
        assembler = Assembler(form, mesh)
        assembler.assemble()
        Ku = Nodal(lambda x: 1/3*x, basis=phi)
         
         
        #af = assembler.af[0]['bilinear']
        M = assembler.get_matrix().toarray()
        
        u = Nodal(lambda x: x, basis=phi)
        u_vec = u.data()
        self.assertTrue(np.allclose(M.dot(u_vec), Ku.data()))
        
                
    def test_integrals_2d(self):
        """
        Test Assembly of some 2D systems
        """
        mesh = QuadMesh(box=[1,2,1,2], resolution=(2,2))
        mesh.cells.get_leaves()[0].mark(0)
        mesh.cells.refine(refinement_flag=0)
        
        # Kernel
        kernel = Kernel(Explicit(f=lambda x:x[:,0]*x[:,1], dim=2))
        
        problem = Form(kernel)
        assembler = Assembler(problem, mesh=mesh)
        assembler.assemble()
        self.assertAlmostEqual(assembler.get_scalar(),9/4)
        
        #
        # Linear forms (x,x) and (x,x') over [1,2]^2 = 7/3, 3/2
        # 
        
        # Elements
        Q1 = QuadFE(mesh.dim(),'Q1')
        Q2 = QuadFE(mesh.dim(),'Q2')
        Q3 = QuadFE(mesh.dim(),'Q3')
        
        # Dofhandlers
        dQ1 = DofHandler(mesh, Q1)
        dQ2 = DofHandler(mesh, Q2)
        dQ3 = DofHandler(mesh, Q3)
        
        # Distribute dofs
        [d.distribute_dofs() for d in [dQ1, dQ2, dQ3]]
        
        for dQ in [dQ1, dQ2, dQ3]:
            # Basis
            phi = Basis(dQ, 'u')
            phi_x = Basis(dQ, 'ux')
            
            # Kernel function
            xfn = Nodal(f=lambda x: x[:,0], basis=phi)
            yfn = Nodal(f=lambda x: x[:,1], basis=phi)
            
            # Kernel 
            kernel = Kernel(xfn)
            
            # Form
            problem = [[Form(kernel, test=phi)],
                       [Form(kernel, test=phi_x)]]
            
            # Assembly
            assembler = Assembler(problem)
            assembler.assemble()
            
            # Check b^Tx = (x,y)
            b0 = assembler.get_vector(0)
            self.assertAlmostEqual(np.sum(b0*yfn.data()[:,0]),9/4)
            
            b1 = assembler.get_vector(1)
            self.assertAlmostEqual(np.sum(b1*xfn.data()[:,0]),3/2)
            self.assertAlmostEqual(np.sum(b1*yfn.data()[:,0]),0)
              
        #
        # Bilinear forms
        #
        # Compute (1,x,y) = 9/4, or (xy, 1, 1) = 9/4 
        
        for dQ in [dQ1, dQ2, dQ3]:
            # Basis
            phi = Basis(dQ,'u')
            phi_x = Basis(dQ, 'ux')
            phi_y = Basis(dQ, 'uy')
            
            # Kernel function 
            xyfn = Explicit(f=lambda x: x[:,0]*x[:,1], dim=2)
            xfn = Nodal(f=lambda x: x[:,0], basis=phi)
            yfn = Nodal(f=lambda x: x[:,1], basis=phi)
            
            # Form 
            problems = [[Form(1, test=phi, trial=phi)], 
                        [Form(Kernel(xfn), test=phi, trial=phi_x)],
                        [Form(Kernel(xyfn), test=phi_y, trial=phi_x)]]
            
            # Assemble
            assembler = Assembler(problems)
            assembler.assemble()
            
            x = xfn.data()[:,0]
            y = yfn.data()[:,0]
            for i_problem in range(3):
                A = assembler.get_matrix(i_problem)
                self.assertAlmostEqual(y.T.dot(A.dot(x)), 9/4)
        
    
    def test_edge_integrals(self):
        """
        Test computing
        """
        mesh = QuadMesh(resolution=(1,1))
        Q = QuadFE(2,'Q1')
        dQ = DofHandler(mesh, Q)
        dQ.distribute_dofs()
        
        phi = Basis(dQ,'u')
        f = Nodal(data=np.ones((phi.n_dofs(),1)),basis=phi)
        kernel = Kernel(f)
        form = Form(kernel, dmu='ds')
        assembler = Assembler(form, mesh)
        
        cell = mesh.cells.get_leaves()[0]
        shape_info = assembler.shape_info(cell)
        xg, wg, phi, dofs = assembler.shape_eval(shape_info, cell)
        

            
    def test01_solve_1d(self):
        """
        Test solving 1D systems 
        """
        mesh = Mesh1D(resolution=(20,))
        mesh.mark_region('left', lambda x: np.abs(x)<1e-9)
        mesh.mark_region('right', lambda x: np.abs(x-1)<1e-9)
        
        Q1 = QuadFE(1,'Q1')
        dQ1 = DofHandler(mesh, Q1)
        dQ1.distribute_dofs()
        
        phi = Basis(dQ1, 'u')
        phi_x = Basis(dQ1, 'ux')
        
        problem = [Form(1,test=phi_x, trial=phi_x), Form(0, test=phi)]
        assembler = Assembler(problem)
        assembler.add_dirichlet('left', dir_fn=0)
        assembler.add_dirichlet('right', dir_fn=1)
        assembler.assemble()
        
        # Get matrix dirichlet correction and right hand side
        A = assembler.get_matrix().toarray()
        x0 = assembler.assembled_bnd()
        b = assembler.get_vector()
        
                    
        ua = np.zeros((phi.n_dofs(),1))
        int_dofs = assembler.get_dofs('interior')
        ua[int_dofs,0] = np.linalg.solve(A,b-x0)
        
        dir_bc = assembler.get_dirichlet()
        dir_vals = np.array([dir_bc[dof] for dof in dir_bc])
        dir_dofs = [dof for dof in dir_bc]
        ua[dir_dofs] = dir_vals 
        
        ue_fn = Nodal(f=lambda x: x[:,0], basis=phi)
        ue = ue_fn.data()
        self.assertTrue(np.allclose(ue,ua))
        self.assertTrue(np.allclose(x0+A.dot(ua[int_dofs,0]),b))
        
    
    def test01_solve_2d(self):
        """
        Solve a simple 2D problem with no hanging nodes
        """
        mesh = QuadMesh(resolution=(5,5))
        
        # Mark dirichlet boundaries
        mesh.mark_region('left', lambda x,dummy: np.abs(x)<1e-9, 
                         entity_type='half_edge')
        
        mesh.mark_region('right', lambda x,dummy: np.abs(x-1)<1e-9,
                         entity_type='half_edge')
        
        Q1 = QuadFE(mesh.dim(),'Q1')
        dQ1 = DofHandler(mesh, Q1)
        dQ1.distribute_dofs()
        
        phi = Basis(dQ1, 'u')
        phi_x = Basis(dQ1, 'ux')
        phi_y = Basis(dQ1, 'uy')
        
        problem = [Form(1, test=phi_x, trial=phi_x),
                   Form(1, test=phi_y, trial=phi_y), 
                   Form(0, test=phi)]
        
        assembler = Assembler(problem)
        assembler.add_dirichlet('left', dir_fn=0)
        assembler.add_dirichlet('right', dir_fn=1)
        assembler.assemble()
        
        # Get matrix dirichlet correction and right hand side
        A = assembler.get_matrix().toarray()
        x0 = assembler.assembled_bnd()
        b = assembler.get_vector()
              
        
        ua = np.zeros((phi.n_dofs(),1))
        int_dofs = assembler.get_dofs('interior')
        ua[int_dofs,0] = np.linalg.solve(A,b-x0)
        
        dir_bc = assembler.get_dirichlet()
        dir_vals = np.array([dir_bc[dof] for dof in dir_bc])
        dir_dofs = [dof for dof in dir_bc]
        ua[dir_dofs] = dir_vals 
        
        ue_fn = Nodal(f=lambda x: x[:,0], basis=phi)
        ue = ue_fn.data()
        self.assertTrue(np.allclose(ue,ua))
        self.assertTrue(np.allclose(x0+A.dot(ua[int_dofs,0]),b))

     
    def test02_solve_2d(self):
        """
        Solve 2D problem with hanging nodes
        """   
        # Mesh
        mesh = QuadMesh(resolution=(2,2))
        mesh.cells.get_leaves()[0].mark(0)
        mesh.cells.refine(refinement_flag=0)
            
        mesh.mark_region('left', lambda x,y: abs(x)<1e-9, entity_type='half_edge')
        mesh.mark_region('right', lambda x,y: abs(x-1)<1e-9, entity_type='half_edge')
        
        # Element
        Q1 = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh,Q1)
        dofhandler.distribute_dofs()
        dofhandler.set_hanging_nodes()
            
        
        # Basis functions
        phi = Basis(dofhandler, 'u')
        phi_x = Basis(dofhandler, 'ux')
        phi_y = Basis(dofhandler, 'uy')
        
        # 
        # Define problem
        #
        problem = [Form(1, trial=phi_x, test=phi_x), 
                   Form(1, trial=phi_y, test=phi_y),
                   Form(0, test=phi)]
        
        ue = Nodal(f=lambda x: x[:,0], basis = phi)
        xe = ue.data().ravel()
        
            
        # 
        # Assemble without Dirichlet and without Hanging Nodes
        # 
        assembler = Assembler(problem)
        assembler.add_dirichlet('left',dir_fn=0)
        assembler.add_dirichlet('right', dir_fn=1)
        assembler.add_hanging_nodes()
        assembler.assemble()
        
        # Get dofs for different regions 
        int_dofs = assembler.get_dofs('interior')
            
        # Get matrix and vector
        A = assembler.get_matrix().toarray()
        b = assembler.get_vector()
        x0 = assembler.assembled_bnd()
        
        # Solve linear system
        xa = np.zeros(phi.n_dofs())
        xa[int_dofs] = np.linalg.solve(A,b-x0)
        
        # Resolve Dirichlet conditions
        dir_dofs, dir_vals = assembler.get_dirichlet(asdict=False)
        xa[dir_dofs] = dir_vals[:,0]
        
        # Resolve hanging nodes
        C = assembler.hanging_node_matrix()
        xa += C.dot(xa)
        
        self.assertTrue(np.allclose(xa, xe))
        
    
    def test03_solve_2d(self):
        """
        Test problem with Neumann conditions
        """
        #
        # Define Mesh
        # 
        mesh = QuadMesh(resolution=(2,1))
        mesh.cells.get_child(1).mark(1)
        mesh.cells.refine(refinement_flag=1)
        
        # Mark left and right boundaries
        bm_left = lambda x,dummy: np.abs(x)<1e-9
        bm_right = lambda x, dummy: np.abs(1-x)<1e-9
        mesh.mark_region('left', bm_left, entity_type='half_edge')
        mesh.mark_region('right', bm_right, entity_type='half_edge')
        
        for etype in ['Q1','Q2','Q3']:
            #
            # Define element and basis type
            #
            element = QuadFE(2,etype)
            dofhandler = DofHandler(mesh, element)
            dofhandler.distribute_dofs()
            
            u = Basis(dofhandler, 'u')
            ux = Basis(dofhandler, 'ux')
            uy = Basis(dofhandler, 'uy')
            
            #
            # Exact solution
            # 
            ue = Nodal(f=lambda x: x[:,0], basis=u)
            
            #
            # Set up forms
            # 
            one = Constant(1)
            ax = Form(kernel=Kernel(one), trial=ux, test=ux)
            ay = Form(kernel=Kernel(one), trial=uy, test=uy)
            L = Form(kernel=Kernel(Constant(0)), test=u)
            Ln = Form(kernel=Kernel(one), test=u, dmu='ds', flag='right')
            
            problem = [ax, ay, L, Ln]
            
            assembler = Assembler(problem, mesh)
            assembler.add_dirichlet('left',dir_fn=0)
            assembler.add_hanging_nodes()
            assembler.assemble()
            
            #
            # Automatic solve
            #
            ya = assembler.solve()
            self.assertTrue(np.allclose(ue.data()[:,0], ya))

            #
            # Explicit solve
            #
            
            # System Matrices 
            A = assembler.get_matrix().toarray()
            b = assembler.get_vector()
            x0 = assembler.assembled_bnd()
            
            # Solve linear system
            xa = np.zeros(u.n_dofs())
            int_dofs = assembler.get_dofs('interior')
            xa[int_dofs] = np.linalg.solve(A,b-x0)
            
            # Resolve Dirichlet conditions
            dir_dofs, dir_vals = assembler.get_dirichlet(asdict=False)
            xa[dir_dofs] = dir_vals[:,0]
            
            # Resolve hanging nodes
            C = assembler.hanging_node_matrix()
            xa += C.dot(xa)
            
            self.assertTrue(np.allclose(ue.data()[:,0], xa))
            
            
if __name__ == '__main__':
    unittest.main()