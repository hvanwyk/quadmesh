import unittest
import numpy as np
from mesh import Mesh1D
from mesh import QuadMesh
from fem import Function
from fem import Basis
from fem import Form
from fem import Kernel
from fem import Assembler
from fem import QuadFE
from fem import LinearSystem

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
            system = LinearSystem(assembler)
            
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
            # Summarize all constraints in system x = Cx+d
            #
            system.set_constraint_matrix()
            
            #
            # Eliminate constraints from system
            # 
            system.incorporate_constraints()
            
            #
            # Solve system
            # 
            system.solve()
            
            #
            # Enforce Dirichlet constraints
            # 
            system.resolve_constraints()
            
            #
            # Get solution
            # 
            ua = system.sol(as_function=True)
            
            # Compare with exact solution
            self.assertTrue(np.allclose(ua.fn(), ue.fn()))
        
    def test02_1d_dirichlet_higher_order(self):
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
            system = LinearSystem(assembler)
            
            # Boundary functions
            bnd_left = lambda x: np.abs(x)<1e-9 
            bnd_right = lambda x: np.abs(1-x)<1e-9
            
            # Mark mesh
            mesh.mark_region('left', bnd_left, entity_type='vertex')
            mesh.mark_region('right', bnd_right, entity_type='vertex')
            
            # Add Dirichlet constraints to system
            system.add_dirichlet_constraint('left',0)
            system.add_dirichlet_constraint('right',0)
            
            # Incorporate all constraints
            system.set_constraint_matrix()
            system.incorporate_constraints()
            
            # Solve system
            system.solve()
            system.resolve_constraints()
            
            # Compare solution with the exact solution
            ua = system.sol(as_function=True)
            self.assertTrue(np.allclose(ua.fn(), ue.fn()))
    
    
    def test03_1d_mixed(self):
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
        system = LinearSystem(assembler)
        
        # Add Dirichlet constraints
        system.add_dirichlet_constraint('left',0)
        system.set_constraint_matrix()
        
        #
        # Solve
        # 
        system.incorporate_constraints()
        system.solve()
        system.resolve_constraints()
        
        # Compare solution with exact solution
        ua = system.sol(as_function=True)
        
        self.assertTrue(np.allclose(ua.fn(), ue.fn()))
    
    
    def test04_1d_periodic(self):
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
        system = LinearSystem(assembler)
        
        # Add dirichlet constraint
        system.add_dirichlet_constraint('left',0, on_boundary=False)
        
        # Assemble constraints
        system.set_constraint_matrix()
        system.incorporate_constraints()
        system.solve()
        system.resolve_constraints()
        
        # Compare with interpolant of exact solution
        ua = system.sol(as_function=True)
        self.assertTrue(np.allclose(ua.fn(), ue.fn()))
        
        
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
            system = LinearSystem(assembler)
            
            #
            # Constraints
            # 
            # Add dirichlet conditions
            system.add_dirichlet_constraint('left',ue)
            system.add_dirichlet_constraint('right',ue)
            system.set_constraint_matrix()
            system.incorporate_constraints()
            
            #
            # Solve
            # 
            system.solve()
            system.resolve_constraints()
            
            #
            # Check solution
            # 
            ua = system.sol(as_function=True)            
            self.assertTrue(np.allclose(ua.fn(),ue.fn()))
            
     
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
           
            system = LinearSystem(assembler)
            
            #
            # Add constraints
            # 
            system.add_dirichlet_constraint('left',0)
            system.set_constraint_matrix()
            system.incorporate_constraints()
            
            #
            # Solve system
            # 
            system.solve()
            system.resolve_constraints()
            
            #
            # Check solution
            # 
            ua = system.sol(as_function=True)
            self.assertTrue(np.allclose(ue.fn(), ua.fn()))
    