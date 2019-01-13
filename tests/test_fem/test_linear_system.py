import unittest
import numpy as np
from mesh import Mesh1D
from mesh import QuadMesh
from fem import DofHandler
from fem import Function
from fem import Basis
from fem import Form
from fem import Kernel
from fem import Assembler
from fem import QuadFE
from fem import LinearSystem
#from plot import Plot


class TestLinearSystem(unittest.TestCase):
    """
    Test Linear System class.
    """
    
    
    def test01_1d_dirichlet_linear(self):
        """
        Solve one dimensional boundary value problem with dirichlet 
        conditions on left and right
        """
        print('test01')
        #
        # Define mesh
        # 
        mesh = Mesh1D(resolution=(10,))
        
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(1,etype)
            
            #
            # Exact solution 
            # 
            ue = Function(lambda x: x, 'nodal', mesh=mesh, element=element) 
            
            #
            # Define Basis functions 
            #  
            u = Basis(element, 'u')
            ux = Basis(element, 'ux')
            
            #
            # Define bilinear form
            #
            one = Function(1, 'constant')
            zero = Function(0, 'constant')
            a = Form(kernel=Kernel(one), trial=ux, test=ux)
            L = Form(kernel=Kernel(zero), test=u)
            problem = [a,L]
            
            #
            # Assemble 
            # 
            assembler = Assembler(problem, mesh)
            assembler.assemble()
            
            #
            # Form linear system
            # 
            system = LinearSystem(assembler, 0)
            
            #
            # Dirichlet conditions 
            # 
            
            # Boundary functions 
            bm_left = lambda x: np.abs(x)<1e-9
            bm_rght = lambda x: np.abs(x-1)<1e-9
            
            # Mark boundary regions
            mesh.mark_region('left', bm_left, on_boundary=True)
            mesh.mark_region('right',bm_rght, on_boundary=True)
            
            # Add Dirichlet constraints
            system.add_dirichlet_constraint('left', ue)
            system.add_dirichlet_constraint('right', ue)
    
            
            #
            # Solve system
            # 
            system.solve_system()
            
            
            #
            # Get solution
            # 
            ua = system.get_solution(as_function=True)
            
            # Compare with exact solution
            self.assertTrue(np.allclose(ua.fn(), ue.fn()))
        
    def test02_1d_dirichlet_higher_order(self):
        print('test02')
        mesh = Mesh1D()
        for etype in ['Q2','Q3']:
            element = QuadFE(1,etype)
            
            # Exact solution
            ue = Function(lambda x:x*(1-x), 'nodal', mesh=mesh, element=element)
            
            # Basis functions 
            ux = Basis(element, 'ux')
            u = Basis(element, 'u')
            
            # Define coefficient functions
            one = Function(1,'constant')
            two = Function(2,'constant')
            
            # Define forms
            a = Form(kernel=Kernel(one), trial=ux, test=ux)
            L = Form(kernel=Kernel(two), test=u)
            problem = [a,L]
            
            # Assemble problem
            assembler = Assembler(problem, mesh)
            assembler.assemble()
            
            # Set up linear system
            system = LinearSystem(assembler, 0)
            
            # Boundary functions
            bnd_left = lambda x: np.abs(x)<1e-9 
            bnd_right = lambda x: np.abs(1-x)<1e-9
            
            # Mark mesh
            mesh.mark_region('left', bnd_left, entity_type='vertex')
            mesh.mark_region('right', bnd_right, entity_type='vertex')
            
            # Add Dirichlet constraints to system
            system.add_dirichlet_constraint('left',0)
            system.add_dirichlet_constraint('right',0)
            
            # Solve system
            system.solve_system()
            system.resolve_constraints()
            
            # Compare solution with the exact solution
            ua = system.get_solution(as_function=True)
            self.assertTrue(np.allclose(ua.fn(), ue.fn()))
    
    
    def test03_1d_mixed(self):
        print('test03')
        mesh = Mesh1D()
        element = QuadFE(1,'Q3')
        
        # Exact solution
        ue = Function(lambda x: x*(1-x), 'nodal', mesh=mesh, element=element)
        
        # Mark mesh regions
        bnd_right = lambda x: np.abs(x-1)<1e-9
        mesh.mark_region('right',bnd_right, entity_type='vertex', on_boundary='True')
        
        bnd_left = lambda x: np.abs(x)<1e-9
        mesh.mark_region('left',bnd_left, entity_type='vertex', on_boundary='True')
        
        # 
        # Forms
        #
        
        # Basis functions 
        u = Basis(element, 'u')
        ux = Basis(element, 'ux')
        
        # Linear form
        L  = Form(kernel=Kernel(Function(2, 'constant')), test=u)
        
        # Neumann form
        Ln = Form(kernel=Kernel(ue, dfdx='fx'), test=u, dmu='dv', flag='right')
        
        # Bilinear form
        a  = Form(kernel=Kernel(Function(1,'constant')), trial=ux, test=ux)
        
        # 
        # Assembly
        # 
        problem = [a, L, Ln]
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        
        #
        # Linear System
        #
        system = LinearSystem(assembler, 0)
        
        # Add Dirichlet constraints
        system.add_dirichlet_constraint('left',0)
        system.set_constraint_relation()
        
        #
        # Solve
        # 
        system.solve_system()

        
        # Compare solution with exact solution
        ua = system.get_solution(as_function=True)
        
        self.assertTrue(np.allclose(ua.fn(), ue.fn()))
    
    
    def test04_1d_periodic(self):
        print('test04')
        #
        # Dirichlet Problem on a Periodic Mesh
        # 
        
        # Define mesh, element
        mesh = Mesh1D(resolution=(100,), periodic=True)
        element = QuadFE(1,'Q3')
                  
        # Exact solution
        ue = Function(lambda x: np.sin(2*np.pi*x), 'nodal', mesh=mesh, element=element)
        
        #
        # Mark dirichlet regions
        #
        bnd_left = lambda x: np.abs(x)<1e-9
        mesh.mark_region('left', bnd_left, entity_type='vertex')
        
        #
        # Set up forms
        #
        
        # Basis functions
        u = Basis(element, 'u')
        ux = Basis(element, 'ux')
        
        # Bilinear form
        a = Form(kernel=Kernel(Function(1,'constant')), trial=ux, test=ux)
        
        # Linear form
        f = Function(lambda x: 4*np.pi**2*np.sin(2*np.pi*x), 'explicit')
        L = Form(kernel=Kernel(f), test=u)
        
        #
        # Assemble
        #
        problem = [a,L]
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        
        #
        # Linear System
        #
        system = LinearSystem(assembler,0)
        
        # Add dirichlet constraint
        system.add_dirichlet_constraint('left',0, on_boundary=False)
        
        # Assemble constraints
        #system.set_constraint_relation()
        #system.incorporate_constraints()
        system.solve_system()
        #system.resolve_constraints()
        
        # Compare with interpolant of exact solution
        ua = system.get_solution(as_function=True)
        
        #plot = Plot(2)
        #plot.line(ua)
        #plot.line(ue)
        self.assertTrue(np.allclose(ua.fn(), ue.fn()))
        # TODO: Problems
        
    def test05_2d_dirichlet(self):        
        """
        Two dimensional Dirichlet problem with hanging nodes
        """
        print('test05')
        #
        # Define mesh
        #
        mesh = QuadMesh(resolution=(1,2))
        mesh.cells.get_child(1).mark(1)
        mesh.cells.refine(refinement_flag=1)
        mesh.cells.refine()
        
        #
        # Mark left and right boundaries
        #
        bm_left = lambda x,dummy: np.abs(x)<1e-9
        bm_right = lambda x, dummy: np.abs(1-x)<1e-9
        mesh.mark_region('left', bm_left, entity_type='half_edge')
        mesh.mark_region('right', bm_right, entity_type='half_edge')
         
        for etype in ['Q1','Q2','Q3']:
            # 
            # Element
            # 
            element = QuadFE(2,etype)            
            
            #
            # Basis 
            #
            u = Basis(element, 'u')
            ux = Basis(element, 'ux')
            uy = Basis(element, 'uy')
            
            #
            # Construct forms
            # 
            ue = Function(lambda x,dummy: x, 'nodal', mesh=mesh, element=element)
            ax = Form(kernel=Kernel(Function(1,'constant')), trial=ux, test=ux)
            ay = Form(kernel=Kernel(Function(1,'constant')), trial=uy, test=uy)
            L = Form(kernel=Kernel(Function(0,'constant')), test=u)
            problem = [ax, ay, L]
            
            #
            # Assemble
            # 
            assembler = Assembler(problem, mesh)
            assembler.assemble()
            
            #
            # Linear System
            # 
            system = LinearSystem(assembler, 0)
            
            #
            # Constraints
            # 
            # Add dirichlet conditions
            system.add_dirichlet_constraint('left',ue)
            system.add_dirichlet_constraint('right',ue)
            
            
            #
            # Solve
            # 
            system.solve_system()
            #system.resolve_constraints()
            
            #
            # Check solution
            # 
            ua = system.get_solution(as_function=True)            
            self.assertTrue(np.allclose(ua.fn(),ue.fn()))
            
     
    def test06_2d_mixed(self):
        """
        Dirichlet problem with Neumann data on right and Dirichlet data on left
        """
        print('test06')
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
            u = Basis(element, 'u')
            ux = Basis(element, 'ux')
            uy = Basis(element, 'uy')
            
            #
            # Exact solution
            # 
            ue = Function(lambda x,dummy: x, 'nodal', mesh=mesh, element=element)
            
            #
            # Set up forms
            # 
            one = Function(1,'constant')
            ax = Form(kernel=Kernel(one), trial=ux, test=ux)
            ay = Form(kernel=Kernel(one), trial=uy, test=uy)
            L = Form(kernel=Kernel(Function(0,'constant')), test=u)
            Ln = Form(kernel=Kernel(one), test=u, dmu='ds', flag='right')
            
            problem = [ax, ay, L, Ln]
            
            assembler = Assembler(problem, mesh)
            assembler.assemble()
           
            system = LinearSystem(assembler,0)
            
            #
            # Add constraints
            # 
            system.add_dirichlet_constraint('left',0)
            #system.set_constraint_relation()
            #system.incorporate_constraints()
            
            #
            # Solve system
            # 
            system.solve_system()
            #system.resolve_constraints()
            
            #
            # Check solution
            # 
            ua = system.get_solution(as_function=True)
            self.assertTrue(np.allclose(ue.fn(), ua.fn()))
    
    
    def test07_1d_mesh_refinement(self):
        """
        Define the input parameters and solution on different resolution meshes
        """
        print('test07')
        #
        # Define mesh at two different resolutoins
        # 
        mesh = Mesh1D(resolution=(1,))
        mesh.cells.record(0)
        
        for dummy in range(3):
            mesh.cells.refine()
        
        #
        # Elements
        # 
        DQ0 = QuadFE(1, 'DQ0')
        Q1  = QuadFE(1, 'Q1')
        
        #
        # Diffusion parameter on coarse mesh
        # 
        q = Function(np.array([1]), 'nodal', mesh=mesh, element=DQ0, subforest_flag=0)
        f = Function(0, 'constant')
        
        #
        # Basis functions
        # 
        u = Basis(Q1, 'u')
        ux = Basis(Q1, 'ux')
        
        #
        # Forms
        # 
        a = Form(kernel=Kernel(q), trial=ux, test=ux)
        L = Form(kernel=Kernel(f), test=u)
        problem = [a,L]
        
        #
        # Assemble
        # 
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        
        #
        # System
        # 
        system = LinearSystem(assembler, 0)
        
        #
        # Dirichlet Constraints
        # 
        left = lambda x: np.abs(x)<1e-9
        right = lambda x: np.abs(1-x)<1e-9
        mesh.mark_region('left', left)
        mesh.mark_region('right', right)
        system.add_dirichlet_constraint('left',0)
        system.add_dirichlet_constraint('right',1)
        
        
        #
        # Solve 
        # 
        system.set_constraint_relation()
        #system.incorporate_constraints()
        system.solve_system()
        #system.resolve_constraints()
        
        #
        # Compare solution with the exact solution
        # 
        ua = system.get_solution(as_function=True)
        ue = Function(lambda x: x, 'nodal', mesh, Q1)        
        self.assertTrue(np.allclose(ua.fn(), ue.fn()))
    
    
    def test08_1d_sampled_rhs(self):
        print('test08')
        #
        # Mesh
        # 
        mesh = Mesh1D(resolution=(1,))
        mesh.mark_region('left', lambda x: np.abs(x)<1e-9, on_boundary=True)
        mesh.mark_region('right', lambda x: np.abs(1-x)<1e-9, on_boundary=True)
        
        #
        # Elements
        # 
        Q3 = QuadFE(1,'Q3')
        dofhandler = DofHandler(mesh, Q3)
        dofhandler.distribute_dofs()
        
        #
        # Define sampled right hand side and exact solution
        # 
        xv = dofhandler.get_dof_vertices()
        n_points = dofhandler.n_dofs()
        
        n_samples = 6
        a = np.arange(n_samples)
        
        f = lambda x, a: a*x
        u = lambda x,a: a/6*(x-x**3)+x
        fdata = np.zeros((n_points,n_samples))
        udata = np.zeros((n_points,n_samples))
        for i in range(n_samples):
            fdata[:,i] = f(xv,a[i]).ravel()
            udata[:,i] = u(xv,a[i]).ravel()
            
            
        # Define sampled function
        fn = Function(fdata, 'nodal', dofhandler=dofhandler)
        ue = Function(udata, 'nodal', dofhandler=dofhandler)
        
        #
        # Basis
        # 
        u = Basis(Q3, 'u')
        ux = Basis(Q3, 'ux')
        
        #
        # Forms
        # 
        one = Function(1,'constant') 
        a = Form(Kernel(one), test=ux, trial=ux)
        L = Form(Kernel(fn), test=u)
        problem = [a,L]
        
        #
        # Assembler
        # 
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        
        #
        # Linear System
        # 
        system = LinearSystem(assembler, 0)
        
        # Set constraints
        system.add_dirichlet_constraint('left',0)
        system.add_dirichlet_constraint('right',1)
        #system.set_constraint_relation()
        #system.incorporate_constraints()
        
        # Solve and resolve constraints
        system.solve_system()
        #system.resolve_constraints()
        
        # Extract finite element solution
        ua = system.get_solution(as_function=True)
        
        # Check that the solution is close
        self.assertTrue(np.allclose(ue.fn(), ua.fn()))
        
        
        
    
    def test09_1d_inverse(self):
        """
        Compute the inverse of a matrix and apply it to a vector/matrix.
        """
        print('test09')
        #
        # Mesh
        # 
        mesh = Mesh1D(resolution=(1,))
        mesh.mark_region('left', lambda x: np.abs(x)<1e-9, on_boundary=True)
        mesh.mark_region('right', lambda x: np.abs(1-x)<1e-9, on_boundary=True)
        
        #
        # Elements
        # 
        Q3 = QuadFE(1,'Q3')
        dofhandler = DofHandler(mesh, Q3)
        dofhandler.distribute_dofs()
        
        #
        # Define sampled right hand side and exact solution
        # 
        xv = dofhandler.get_dof_vertices()
        n_points = dofhandler.n_dofs()
        
        n_samples = 6
        a = np.arange(n_samples)
        
        f = lambda x, a: a*x
        u = lambda x,a: a/6*(x-x**3)+x
        fdata = np.zeros((n_points,n_samples))
        udata = np.zeros((n_points,n_samples))
        for i in range(n_samples):
            fdata[:,i] = f(xv,a[i]).ravel()
            udata[:,i] = u(xv,a[i]).ravel()
            
            
        # Define sampled function
        fn = Function(fdata, 'nodal', dofhandler=dofhandler)
        ue = Function(udata, 'nodal', dofhandler=dofhandler)
        
        #
        # Basis
        # 
        u = Basis(Q3, 'u')
        ux = Basis(Q3, 'ux')
        
        #
        # Forms
        # 
        one = Function(1,'constant') 
        a = Form(Kernel(one), test=ux, trial=ux)
        L = Form(Kernel(fn), test=u)
        problem = [[a],[L]]
        
        #
        # Assembler
        # 
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        
        #
        # Linear System
        # 
        system = LinearSystem(assembler, 0)
        
        # Set constraints
        system.add_dirichlet_constraint('left',0)
        system.add_dirichlet_constraint('right',1)
        #system.set_constraint_relation()
        #system.incorporate_constraints()
        
        
        # Solve and resolve constraints
        #b = assembler.af[1]['linear'].get_matrix()
        
        #system.factor()
        #system.modify_rhs(b)
        system.solve_system(assembler.af[1]['linear'])
        
        #system.resolve_constraints()
        
        # Extract finite element solution
        ua = system.get_solution(as_function=True)
        
        system2 = LinearSystem(assembler, \
                               bilinear_form=assembler.af[0]['bilinear'], 
                               linear_form=assembler.af[1]['linear'])
        # Set constraints
        system2.add_dirichlet_constraint('left',0)
        system2.add_dirichlet_constraint('right',1)
        #system2.set_constraint_relation()
        #system2.incorporate_constraints()
        
        system2.solve_system()
        #system2.resolve_constraints()
        u2 = system2.get_solution(as_function=True)
        
        #plot = Plot()
        #plot.line(ua)
        #plot.line(u2)
        #plot.line(ue)
    
        # Check that the solution is close
        self.assertTrue(np.allclose(ue.fn(), ua.fn()))
        self.assertTrue(np.allclose(ue.fn(), u2.fn()))
