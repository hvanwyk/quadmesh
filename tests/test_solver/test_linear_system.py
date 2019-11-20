import unittest
import numpy as np

from assembler import Form
from assembler import Kernel
from assembler import Assembler

from mesh import Mesh1D
from mesh import QuadMesh

from fem import DofHandler
from fem import Basis
from fem import QuadFE

from function import Nodal
from function import Explicit
from function import Constant

from solver import LinearSystem
from solver import LinearSystem
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
        #
        # Define mesh
        # 
        mesh = Mesh1D(resolution=(10,))
        
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(1,etype)
            dofhandler = DofHandler(mesh, element)
            #
            # Exact solution 
            # 
            ue = Nodal(f=lambda x: x, dofhandler=dofhandler) 
            
            #
            # Define Basis functions 
            #  
            u = Basis(dofhandler, 'u')
            ux = Basis(dofhandler, 'ux')
            
            #
            # Define bilinear form
            #
            one = Constant(1)
            zero = Constant(0)
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
            A = assembler.af[0]['bilinear'].get_matrix()
            b = assembler.af[0]['linear'].get_matrix()
          
            system = LinearSystem(u, A=A, b=b)
            
            
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
            #system.solve_system()
            system.solve_system()
            
            #
            # Get solution
            # 
            #ua = system.get_solution(as_function=True)
            uaa = system.get_solution(as_function=True)
            #uaa = uaa.data().ravel()
            
            
            # Compare with exact solution
            #self.assertTrue(np.allclose(ua.data(), ue.data()))
            self.assertTrue(np.allclose(uaa.data(), ue.data()))
            
            
    def test02_1d_dirichlet_higher_order(self):
        mesh = Mesh1D()
        for etype in ['Q2','Q3']:
            element = QuadFE(1,etype)
            dofhandler = DofHandler(mesh, element)
            
            # Exact solution
            ue = Nodal(f=lambda x:x*(1-x), dofhandler=dofhandler)
            
            # Basis functions 
            ux = Basis(dofhandler, 'ux')
            u = Basis(dofhandler, 'u')
            
            # Define coefficient functions
            one = Constant(1)
            two = Constant(2)
            
            # Define forms
            a = Form(kernel=Kernel(one), trial=ux, test=ux)
            L = Form(kernel=Kernel(two), test=u)
            problem = [a,L]
            
            # Assemble problem
            assembler = Assembler(problem, mesh)
            assembler.assemble()
            
            A = assembler.af[0]['bilinear'].get_matrix()
            b = assembler.af[0]['linear'].get_matrix()
            
            # Set up linear system
            #system = LinearSystem(assembler, 0)
            system = LinearSystem(u, A=A, b=b)
            
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
            self.assertTrue(np.allclose(ua.data(), ue.data()))
    
    
    def test03_1d_mixed(self):
        mesh = Mesh1D()
        element = QuadFE(1,'Q3')
        dofhandler = DofHandler(mesh, element)
        
        # Exact solution
        ue = Nodal(f=lambda x: x*(1-x), dofhandler=dofhandler)
        
        # Mark mesh regions
        bnd_right = lambda x: np.abs(x-1)<1e-9
        mesh.mark_region('right',bnd_right, entity_type='vertex', on_boundary='True')
        
        bnd_left = lambda x: np.abs(x)<1e-9
        mesh.mark_region('left',bnd_left, entity_type='vertex', on_boundary='True')
        
        # 
        # Forms
        #
        
        # Basis functions 
        u = Basis(dofhandler, 'u')
        ux = Basis(dofhandler, 'ux')
        
        # Linear form
        L  = Form(kernel=Kernel(Constant(2)), test=u)
        
        # Neumann form
        Ln = Form(kernel=Kernel([ue], derivatives=['fx']), test=u, dmu='dv', flag='right')
        
        # Bilinear form
        a  = Form(kernel=Kernel(Constant(1)), trial=ux, test=ux)
        
        # 
        # Assembly
        # 
        problem = [a, L, Ln]
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        A = assembler.af[0]['bilinear'].get_matrix()
        b = assembler.af[0]['linear'].get_matrix() 
        
        #
        # Linear System
        #
        system = LinearSystem(u, A=A, b=b)
        
        # Add Dirichlet constraints
        system.add_dirichlet_constraint('left',0)
        system.set_constraint_relation()
        
        
        #
        # Solve
        # 
        system.solve_system()
        
        # Compare solution with exact solution
        ua = system.get_solution(as_function=True)
        
        self.assertTrue(np.allclose(ua.data(), ue.data()))
    
    
    def test04_1d_periodic(self):
        #
        # Dirichlet Problem on a Periodic Mesh
        # 
        
        # Define mesh, element
        mesh = Mesh1D(resolution=(100,), periodic=True)
        element = QuadFE(1,'Q3')
        dofhandler = DofHandler(mesh, element)
                  
        # Exact solution
        ue = Nodal(f=lambda x: np.sin(2*np.pi*x), dofhandler=dofhandler)
        
        #
        # Mark dirichlet regions
        #
        bnd_left = lambda x: np.abs(x)<1e-9
        mesh.mark_region('left', bnd_left, entity_type='vertex')
        
        #
        # Set up forms
        #
        
        # Basis functions
        u = Basis(dofhandler, 'u')
        ux = Basis(dofhandler, 'ux')
        
        # Bilinear form
        a = Form(kernel=Kernel(Constant(1)), trial=ux, test=ux)
        
        # Linear form
        f = Explicit(lambda x: 4*np.pi**2*np.sin(2*np.pi*x), dim=1)
        L = Form(kernel=Kernel(f), test=u)
        
        #
        # Assemble
        #
        problem = [a,L]
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        A = assembler.af[0]['bilinear'].get_matrix()
        b = assembler.af[0]['linear'].get_matrix()
        
        #
        # Linear System
        #
        system = LinearSystem(u, A=A, b=b)
        
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
        self.assertTrue(np.allclose(ua.data(), ue.data()))
        # TODO: Problems
        
    def test05_2d_dirichlet(self):        
        """
        Two dimensional Dirichlet problem with hanging nodes
        """
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
            dofhandler = DofHandler(mesh, element)
            
            #
            # Basis 
            #
            u = Basis(dofhandler, 'u')
            ux = Basis(dofhandler, 'ux')
            uy = Basis(dofhandler, 'uy')
            
            #
            # Construct forms
            # 
            ue = Nodal(f=lambda x: x[:,0], dofhandler=dofhandler)
            ax = Form(kernel=Kernel(Constant(1)), trial=ux, test=ux)
            ay = Form(kernel=Kernel(Constant(1)), trial=uy, test=uy)
            L = Form(kernel=Kernel(Constant(0)), test=u)
            problem = [ax, ay, L]
            
            #
            # Assemble
            # 
            assembler = Assembler(problem, mesh)
            assembler.assemble()
            
            #
            # Get system matrices
            # 
            A = assembler.af[0]['bilinear'].get_matrix()
            b = assembler.af[0]['linear'].get_matrix()
            
            #
            # Linear System
            # 
            system = LinearSystem(u, A=A, b=b)
            
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
            self.assertTrue(np.allclose(ua.data(),ue.data()))
            
     
    def test06_2d_mixed(self):
        """
        Dirichlet problem with Neumann data on right and Dirichlet data on left
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
            u = Basis(dofhandler, 'u')
            ux = Basis(dofhandler, 'ux')
            uy = Basis(dofhandler, 'uy')
            
            #
            # Exact solution
            # 
            ue = Nodal(f=lambda x: x[:,0], dofhandler=dofhandler)
            xv = dofhandler.get_dof_vertices()
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
            assembler.assemble()
            
            #
            # System Matrices
            # 
            A = assembler.af[0]['bilinear'].get_matrix()
            b = assembler.af[0]['linear'].get_matrix()
            
            system = LinearSystem(u, A=A, b=b)
            
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
            self.assertTrue(np.allclose(ue.data(), ua.data()))
    
    
    def test07_1d_mesh_refinement(self):
        """
        Define the input parameters and solution on different resolution meshes
        """
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
        #
        #
        dhDQ0 = DofHandler(mesh, DQ0)
        dhQ1 = DofHandler(mesh, Q1)
        
        #
        # Diffusion parameter on coarse mesh
        # 
        q = Nodal(data=np.array([1]), dofhandler=dhDQ0, subforest_flag=0)
        f = Constant(0)
        
        #
        # Basis functions
        # 
        u = Basis(dhQ1, 'u')
        ux = Basis(dhQ1, 'ux')
        
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
        # System Matrices
        # 
        A = assembler.af[0]['bilinear'].get_matrix()
        b = assembler.af[0]['linear'].get_matrix()
        
        #
        # System
        # 
        system = LinearSystem(u, A=A, b=b)
        
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
        ue = Nodal(f=lambda x: x, dofhandler=dhQ1)        
        self.assertTrue(np.allclose(ua.data(), ue.data()))
    
    
    def test08_1d_sampled_rhs(self):
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
        fn = Nodal(data=fdata, dofhandler=dofhandler)
        ue = Nodal(data=udata, dofhandler=dofhandler)
        
        #
        # Basis
        # 
        u = Basis(dofhandler, 'u')
        ux = Basis(dofhandler, 'ux')
        
        #
        # Forms
        # 
        one = Constant(1) 
        a = Form(Kernel(one), test=ux, trial=ux)
        L = Form(Kernel(fn), test=u)
        problem = [a,L]
        
        #
        # Assembler
        # 
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        A = assembler.af[0]['bilinear'].get_matrix()
        b = assembler.af[0]['linear'].get_matrix()
        #
        # Linear System
        # 
        system = LinearSystem(u, A=A, b=b)
        
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
        self.assertTrue(np.allclose(ue.data(), ua.data()))
        
        
        
    
    def test09_1d_inverse(self):
        """
        Compute the inverse of a matrix and apply it to a vector/matrix.
        """
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
        fn = Nodal(data=fdata, dofhandler=dofhandler)
        ue = Nodal(data=udata, dofhandler=dofhandler)
        
        #
        # Basis
        # 
        u = Basis(dofhandler, 'u')
        ux = Basis(dofhandler, 'ux')
        
        #
        # Forms
        # 
        one = Constant(1) 
        a = Form(Kernel(one), test=ux, trial=ux)
        L = Form(Kernel(fn), test=u)
        problem = [[a],[L]]
        
        #
        # Assembler
        # 
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        
        A = assembler.af[0]['bilinear'].get_matrix()
        b = assembler.af[1]['linear'].get_matrix()
        
        #
        # Linear System
        # 
        system = LinearSystem(u, A=A)
        
        # Set constraints
        system.add_dirichlet_constraint('left',0)
        system.add_dirichlet_constraint('right',1)
        system.solve_system(b[:,0])
        
        
        # Extract finite element solution
        ua = system.get_solution(as_function=True)
        
        system2 = LinearSystem(u, A=A, b=b)
        
        # Set constraints
        system2.add_dirichlet_constraint('left',0)
        system2.add_dirichlet_constraint('right',1)
        system2.solve_system()
        u2 = system2.get_solution(as_function=True)
        
        
        # Check that the solution is close
        self.assertTrue(np.allclose(ue.data()[:,0], ua.data()[:,0]))
        self.assertTrue(np.allclose(ue.data(), u2.data()))
