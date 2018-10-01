import unittest
from mesh import QuadMesh, Mesh1D, convert_to_array
from fem import QuadFE, Assembler, Function, DofHandler, Form, Basis, Kernel
import numpy as np
import scipy.sparse as sp


class TestAssembler(unittest.TestCase):
    """
    Test Assembler Class
    """
    def test_constructor(self):
        """
        Initialization involves:
        
            parsing input "problems"
            initializing dofhandlers (for different element types)
            initializing the assembled_forms
        """
        # =====================================================================
        # Test parsing "problems" input
        # ===================================================================== 
        mesh = QuadMesh(resolution=(1,1))
        Q1 = QuadFE(2, 'Q1')
        Q2 = QuadFE(2, 'Q2')
        u = Basis(Q1, 'u')
        v = Basis(Q2, 'v')
        
        form = Form(trial=u, test=v)
        # 
        # Single problem consisting of a single form
        # 
        problems = form
        system = Assembler(problems, mesh)
        self.assertTrue(system.single_form)
        self.assertTrue(system.single_problem)
        
        #
        # Single problem 
        # 
        problems = [form, form]
        system = Assembler(problems, mesh)
        self.assertTrue(system.single_problem)
        self.assertFalse(system.single_form)
        
        #
        # Multiple problems
        # 
        problems = [form, [form, form]]
        system = Assembler(problems, mesh)
        self.assertFalse(system.single_problem)
        self.assertFalse(system.single_form)
        
        
        # =====================================================================
        # Test "initialize assembled forms"
        # ===================================================================== 
        #
        # Initialize the system
        # 
        mesh = Mesh1D(resolution=(1,))
        
        #
        # Define Problem
        # 
        Q1 = QuadFE(1,'Q1')
        Q2 = QuadFE(1,'Q2')
        vx = Basis(Q1,'vx')
        v = Basis(Q1, 'v')
        w = Basis(Q2, 'wx')
        
        one = Kernel(Function(1,'constant'))    
        # Multiple forms, same element 
        problem_1 = [Form(one, trial=v, test=vx), Form(one, test=v), Form(one)]
        
        # Single bilinear form, multiple elements
        problem_2 = [Form(one, trial=v, test=w)]
        
        # Multiple linear forms, inconsistent elements
        problem_3 = [Form(one, test=w),Form(one, test=vx)]
        
        # Initialize assembled system for correct problems 
        problems = [problem_1, problem_2]
        system = Assembler(problems, mesh)
        
        
        # Check problem_1
        print(system.af[0].keys())
        self.assertEqual(system.af[0]['bilinear']['trial_etype'],'Q1')
        self.assertEqual(system.af[0]['bilinear']['test_etype'],'Q1')
        self.assertTrue('linear' in system.af[0])
        self.assertTrue('bilinear' in system.af[0])
        self.assertTrue('constant' in system.af[0])
        
        # Check problem_2 
        self.assertEqual(system.af[1]['bilinear']['trial_etype'],'Q1')
        self.assertEqual(system.af[1]['bilinear']['test_etype'],'Q2')
        self.assertFalse('linear' in system.af[1])
        self.assertFalse('constant' in system.af[1])
        self.assertTrue('bilinear' in system.af[1])
        
        # Check: error initializing assembled forms for problem 3 
        problems = [problem_3]
        self.assertRaises(Exception, Assembler.__init__, *(problems, mesh))
        
        # =====================================================================
        # Initialize DofHandlers
        # =====================================================================
        mesh = Mesh1D(resolution=(1,))
        Q1, Q2, Q3 = QuadFE(1,'Q1'), QuadFE(1,'Q2'), QuadFE(1,'Q3')
        f = Function(lambda x: x**2, 'nodal', mesh=mesh, element=Q1)
        kernel = Kernel(f)
        
        u = Basis(Q2, 'u')
        v = Basis(Q3, 'v')
        
        form = Form(kernel, trial=u, test=v)
        system = Assembler(form, mesh)
        
        etypes = ['Q1','Q2','Q3']
        for etype in etypes:
            self.assertTrue(etype in system.dofhandlers)
            self.assertEqual(system.dofhandlers['Q1'], f.dofhandler)
        
    def test_shape_info(self):
        """
        Test whether system extracts the correct information
        """
        #
        # 1D 
        # 
        mesh = Mesh1D(resolution=(1,))
        Q1 = QuadFE(1,'Q1')
        ux = Basis(Q1, 'ux')
        v =  Basis(Q1, 'v')
        system = Assembler(Form(trial=ux, test=v), mesh)
        
        cell = mesh.cells.get_child(0)
        
        si = system.shape_info(cell)
        self.assertTrue(cell in si)
        self.assertTrue((1,0) in si[cell]['Q1']['derivatives'])
        self.assertTrue((0,) in si[cell]['Q1']['derivatives'])
        self.assertEqual(Q1, si[cell]['Q1']['element'])
        
        #
        # 2D
        # 
        mesh = QuadMesh(resolution=(1,1))
        Q1 = QuadFE(2, 'Q1')
        Q2 = QuadFE(2, 'Q2')
        
        # Kernel
        f = Function(lambda x,y:x*y, 'nodal', mesh=mesh, element=Q2)
        g = Function(lambda x,y:x+y, 'nodal', mesh=mesh, element=Q1)
        F = lambda f,g: f-g
        kernel = Kernel([f,g], dfdx=['fx','g'], F=F)
        
        # Form defined over HalfEdge with flag 1 
        form_1 = Form(kernel, \
                      trial=Basis(Q1,'u'), \
                      test=Basis(Q2,'vx'), \
                      dmu='ds', flag=1)
        
        form_2 = Form(trial=Basis(Q1,'u'),\
                      test=Basis(Q2,'v'),\
                      dmu='dx', flag=2)
                      
        problem = [form_1, form_2]
                      
        # Assembler
        system = Assembler(problem, mesh)
        
        # Shape info on cell 
        cell = mesh.cells.get_child(0)
        si = system.shape_info(cell)
        
        # Empty info dictionary (HalfEdge hasn't been marked)
        self.assertEqual(si, {})
        
        # Mark HalfEdge and cell and do it again
        he = cell.get_half_edge(0)
        he.mark(1)
        cell.mark(2)
        si = system.shape_info(cell)
        
        # Check that shape info contains the right stuff
        for region in [he, cell]:
            self.assertTrue(region in si)
        
        for etype in ['Q1','Q2']:    
            self.assertTrue(etype in si[he])
            self.assertTrue(etype in si[cell])
        
        # Check if info contains the correct derivatives
        self.assertTrue((0,) in si[cell]['Q1']['derivatives'])
        self.assertTrue((0,) in si[he]['Q1']['derivatives'])
        self.assertFalse((1,0) in si[he]['Q1']['derivatives'])
        self.assertFalse((1,0) in si[cell]['Q1']['derivatives'])
        self.assertTrue((1,0) in si[he]['Q2']['derivatives'])
        
        # Check if info contains correct elements
        self.assertEqual(Q1, si[cell]['Q1']['element'])
        self.assertEqual(Q2, si[cell]['Q2']['element'])
        
    def test_assemble(self):
        """
        Test system assembly
        """ 
        # =====================================================================
        # 1D 
        # =====================================================================
        
        # ======================================================================
        # Test 1: Assemble simple bilinear form (u,v) on Mesh1D
        # ======================================================================
        # Mesh 
        mesh = Mesh1D(resolution=(2,))
        
        # Test and trial functions
        Q1   = QuadFE(1, 'Q1')
        u = Basis(Q1, 'u')
        v = Basis(Q1, 'v')
        
        # Form
        form = Form(trial=u, test=v)

        # Define system
        system = Assembler(form, mesh)
        
        # Get local information
        cell = mesh.cells.get_child(0)
        si = system.shape_info(cell)
        
        # Compute local Gauss nodes
        xg,wg = system.gauss_rules(si)
        self.assertTrue(cell in xg)
        self.assertTrue(cell in wg)
        
        # Compute local shape functions
        phi = system.shape_eval(si, xg, cell)
        self.assertTrue(cell in phi)
        self.assertTrue('Q1' in phi[cell])
        self.assertTrue((0,) in phi[cell]['Q1'])
        
        # Assemble system                
        system.assemble()
        
        # Extract system bilinear form
        rows = np.array(system.af[0]['bilinear']['rows'])
        cols = np.array(system.af[0]['bilinear']['cols'])
        vals = np.array(system.af[0]['bilinear']['vals'])
        
        A = sp.coo_matrix((vals,(rows, cols)))
        
        # Use bilinear form to integrate x^2 over [0,1]
        f = Function(lambda x: x, 'nodal', mesh=mesh, element=Q1)
        fv = f.fn()
        self.assertAlmostEqual(np.sum(fv*A.dot(fv)),1/3)
        
        # ======================================================================
        # Test 2: Linear form (x, test)
        # ======================================================================
        
        # Mesh
        mesh = Mesh1D(resolution=(2,))
        
        # Explicit function
        f = Function(lambda x: x, 'explicit')
        kernel = Kernel(f=f)
        
        # Test function
        Q1 = QuadFE(1,'Q1')
        v = Basis(Q1, 'v')
        
        # Define form and system
        form = Form(kernel=kernel, test=v)
        system = Assembler(form, mesh)
        
        # Assemble system
        system.assemble()
        
        # Evaluate function f(x)=x at dof vertices 
        xv = system.dofhandlers['Q1'].get_dof_vertices()
        fv = f.eval(xv)
        
        # Assemble linear form into
        bb = system.af[0]['linear']['vals']
        rows = system.af[0]['linear']['row_dofs']
        b = np.zeros(3)
        for row, bv in zip(rows, bb):
            b[row] += bv
        
        self.assertAlmostEqual(b.dot(fv), 1/3)
        
        
        # ======================================================================
        # Test 3: Constant form (x^2,.,.) over 1D mesh
        # ======================================================================
        
        # Mesh
        mesh = Mesh1D(resolution=(10,))
        
        # Nodal kernel function
        Q2 = QuadFE(1, 'Q2')
        f = Function(lambda x:x**2, 'nodal', mesh=mesh, element=Q2)
        kernel = Kernel(f=f)
        
        # Form
        form = Form(kernel = kernel)
        
        # Generate and assemble the system
        system = Assembler(form, mesh)
        system.assemble()
        
        # Check 
        self.assertAlmostEqual(system.af[0]['constant'],1/3)
        
        
        
        # =====================================================================
        # Test 4: Periodic Mesh
        # =====================================================================
        #
        # TODO: NO checks yet
        # 
        
        mesh = Mesh1D(resolution=(2,), periodic=True)
        
        # 
        Q1 = QuadFE(1,'Q1')
        u = Basis(Q1, 'u')
        
        form = Form(trial=u, test=u)
        
        system = Assembler(form, mesh)
        system.assemble()
        
        #print(system.af[0]['bilinear'])
        
        #print(system.dofhandlers['Q1'].get_global_dofs())
        
        
        # =====================================================================
        # Test 5: Assemble simple sampled form
        # ======================================================================
        mesh = Mesh1D(resolution=(3,))
        
        Q1 = QuadFE(1,'Q1')
        dofhandler = DofHandler(mesh, Q1)
        dofhandler.distribute_dofs()
        
        xv = dofhandler.get_dof_vertices()
        n_points = dofhandler.n_dofs()
        
        n_samples = 6
        a = np.arange(n_samples)
        
        f = lambda x, a: a*x
        
        fdata = np.zeros((n_points,n_samples))
        for i in range(n_samples):
            fdata[:,i] = f(xv,a[i]).ravel()
            
        
        # Define sampled function
        fn = Function(fdata, 'nodal', dofhandler=dofhandler)
        kernel = Kernel(fn)
        
        #
        # Integrate I[0,1] ax^2 dx by using the linear form (ax,x) 
        # 
        v = Basis(Q1, 'v')
        form = Form(kernel=kernel, test=v)
        system = Assembler(form, mesh)
        system.assemble()
        
        one = np.ones(n_points)
        b = system.af[0]['linear']['vals']
        integrals = one.dot(b)
        for i in range(n_samples):
            self.assertAlmostEqual(integrals[i],0.5*a[i])
        
        
        #
        # Integrate I[0,1] ax^4 dx using bilinear form (ax, x^2, x)
        #
        Q2 = QuadFE(1,'Q2')
        u = Basis(Q2, 'u')
        
        # Define form
        form = Form(kernel=kernel, test=v, trial=u)
        
        # Define and assemble system
        system = Assembler(form, mesh)
        system.assemble()
        
        # Extract data from bilinear form
        vals = system.af[0]['bilinear']['vals'] 
        rows = system.af[0]['bilinear']['rows']
        cols = system.af[0]['bilinear']['cols']
        
        # Express x^2 in terms of trial function basis
        dofhandlerQ2 = DofHandler(mesh, Q2)
        dofhandlerQ2.distribute_dofs() 
        xvQ2 = dofhandlerQ2.get_dof_vertices()
        xv_squared = xvQ2**2
        
        for i in range(n_samples):
            #
            # Iterate over samples 
            # 
            
            # Form sparse matrix
            A = sp.coo_matrix((vals[:,i],(rows,cols)))
            
            # Evaluate the integral
            I = np.sum(xv*A.dot(xv_squared))
            
            # Compare with expected result
            self.assertAlmostEqual(I, 0.2*a[i])
              
                  
        # =====================================================================
        # Test 6: Working with submeshes
        # =====================================================================
        mesh = Mesh1D(resolution=(2,))
        mesh.cells.record((0,1))
    
        
    
    '''
    def test_assemble(self):
        """
        Assembly
        """
        
        mesh = Mesh1D(resolution=(1,))
        element = QuadFE(1, 'Q1')
        system = Assembler(mesh, element)
        bf = [(1,'u','v')]
        #A = system.assemble(bf)
    '''
        
    '''    
    def test_form_eval(self):
        """
        Local (bi)linear forms
        
        1. Vary forms:
            a. Linear forms
            b. Bilinear forms
            
        2. Vary entity:
            a. Interval
            b. HalfEdge
            c. QuadCell (rectangle)
            d. QuadCell (quadrilateral)
            
        3. Vary kernel function
            a. explicit
            b. nodal
            c. multisample
            d. constant
            e. different mesh
            
        4. Vary test/trial functions
            a. Derivatives
            
        Check: 
        
            Integrals or values
        """
        # 
        # Test Local (Bi)linear Forms Explicitly
        # 
    
        #
        # Intervals
        # 
        mesh = Mesh1D(box=[2,5], resolution=(1,))
        I = mesh.cells.get_child(0)
        etypes = ['DQ0', 'Q1','Q2','Q3']
        for etype in etypes:
            element = QuadFE(1, etype)
            system = Assembler(mesh, element)
            one = Function(1,'constant')
            form = Form(one, 'u', 'v')
            A = form.eval(I)
            # Integrate constant function
            A = system.form_eval((Function(1, 'constant'),'u','v'), I)
            
            # Check whether the entries in A add up to the length of the interval
            self.assertAlmostEqual(A.sum(), 3)
        
        #
        # Regular QuadCell
        #
        mesh = QuadMesh()
        cell = mesh.cells.get_child(0)
        element = QuadFE(2,'Q1')
        system = Assembler(mesh, element)
        one = Function(1, 'constant')
        lf = (one,'v')
        bf = (one,'u','v')
        A = system.form_eval(bf, cell)
        AA = 1/36.0*np.array([[4,2,1,2],[2,4,2,1],[1,2,4,2],[2,1,2,4]])

        self.assertTrue(np.allclose(A,AA),'Incorrect mass matrix')
        
        b = system.form_eval(lf, cell)
        b_check = 0.25*np.array([1,1,1,1])
        self.assertTrue(np.allclose(b,b_check),'Right hand side incorrect')
        
        #
        # Integration tests
        #
        """
        Evaluate (f,u,v), or (f,ux,v) or whatever for a specific u,v, and f
        """
        # =====================================================================
        # 2D Mesh
        # =====================================================================
        
        # Define mesh and cell
        mesh = QuadMesh(box=[1,2,1,2])
        cell = mesh.cells.get_child(0)
        edge = cell.get_half_edge(1)
        
        # Define test and trial functions
        trial_functions = {'Q1': lambda x,dummy: (x-1),
                           'Q2': lambda x,y: x*y**2,
                           'Q3': lambda x,y: x**3*y}
        test_functions = {'Q1': lambda x,y: x*y, 
                          'Q2': lambda x,y: x**2*y, 
                          'Q3': lambda x,y: x**3*y**2}
    
        # Integrals over current cell: (1,u,v) and (1,u,vx) 
        cell_integrals = {'Q1': [5/4,3/4], 
                          'Q2': [225/16,35/2], 
                          'Q3': [1905/28,945/8]}
        edge_integrals = {'Q1': [3,3/2], 
                          'Q2': [30,30], 
                          'Q3': [240,360]}
        
        # Function for linear form
        f = lambda x,y: (x-1)*(y-1)**2
        
        # Function for bilinear form
        one = Function(1, 'constant')
        
        for etype in ['Q1','Q2','Q3']:
            #
            # Iterate over Element Types
            # 
            
            # Define element and system
            element = QuadFE(2,etype)
            system = Assembler(mesh,element)
            
            # Extract local reference vertices
            x_loc = system.x_loc(cell)
            
            # Evaluate local trial functions at element dof vertices
            u = trial_functions[etype](x_loc[:,0], x_loc[:,1])
            v = test_functions[etype](x_loc[:,0], x_loc[:,1])
            
            #
            # Bilinear forms on cell
            #
            # Compute (1,u,v) on cell
            b_uv = system.form_eval((one,'u','v'), cell)
            
            
            self.assertAlmostEqual(v.dot(b_uv.dot(u)),
                                   cell_integrals[etype][0],8, 
                                   '{0}: Bilinear form (1,u,v) incorrect.'\
                                   .format(etype))
            
            # Compute (1,u,vx) on cell
            b_uvx = system.form_eval((one,'u','vx'), cell)
            self.assertAlmostEqual(v.dot(b_uvx.dot(u)),
                                   cell_integrals[etype][1],8, 
                                   '{0}: Bilinear form (1,u,vx) incorrect.'\
                                   .format(etype))
            
            
            #
            # Compute bilinear form on edge
            #
            
            # Compute (1,u,v) on edge
            be_uv = system.form_eval((one,'u','v'), edge)
            self.assertAlmostEqual(v.dot(be_uv.dot(u)),
                                   edge_integrals[etype][0],8, 
                                   '{0}: Bilinear form (1,u,v) incorrect.'\
                                   .format(etype))
            
            # Compute (1,u,vx) on edge
            be_uvx = system.form_eval((one,'u','vx'), edge)
            self.assertAlmostEqual(v.dot(be_uvx.dot(u)),
                                   edge_integrals[etype][1],8, 
                                   '{0}: Bilinear form (1,u,vx) incorrect.'\
                                   .format(etype))
        
            #
            # Linear form
            #
            # On cell
            f = Function(trial_functions[etype], 'explicit') 
            f_v = system.form_eval((f,'v'), cell)
            self.assertAlmostEqual(f_v.dot(v), cell_integrals[etype][0],8, 
                                   '{0}: Linear form (f,v) incorrect.'\
                                   .format(etype))
            
            
            f_vx = system.form_eval((f,'vx'), cell)
            self.assertAlmostEqual(f_vx.dot(v), cell_integrals[etype][1],8, 
                                   '{0}: Linear form (f,vx) incorrect.'\
                                   .format(etype))
            
            # edges
            fe_v = system.form_eval((f,'v'), edge)
            self.assertAlmostEqual(fe_v.dot(v), edge_integrals[etype][0],8, 
                                   '{0}: Linear form (f,v) incorrect.'\
                                   .format(etype))
            
            fe_vx = system.form_eval((f,'vx'), edge)
            self.assertAlmostEqual(fe_vx.dot(v), edge_integrals[etype][1],8, 
                                   '{0}: Linear form (f,vx) incorrect.'\
                                   .format(etype))
            
        #
        # A rectacngular cell        
        # 
        mesh = QuadMesh(box=[1,4,1,3])
        element = QuadFE(2,'Q1')
        system = Assembler(mesh,element)
        cell = mesh.cells.get_child(0)
        A = system.form_eval((one,'ux','vx'),cell)
        #
        # Use form to integrate
        # 
        u = trial_functions['Q1']
        v = test_functions['Q1']
        x = system.x_loc(cell)
        ui = u(x[:,0],x[:,1])
        vi = v(x[:,0],x[:,1])
        self.assertAlmostEqual(vi.dot(A.dot(ui)), 12, 8, 'Integral incorrect.')
    '''