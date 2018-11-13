import unittest
import numpy as np
import scipy.sparse as sp
from mesh import Mesh1D
from mesh import QuadMesh
from mesh import convert_to_array
from fem import Function
from fem import Basis
from fem import Form
from fem import Kernel
from fem import Assembler
from fem import DofHandler
from fem import QuadFE
from fem import LinearSystem
from plot import Plot
import matplotlib.pyplot as plt

class TestLinearSystem(unittest.TestCase):
    """
    Test Linear System class.
    """
    
    
    def test01_1d_dirichlet_bvp(self):
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
        
    def test02_1d_dirichlet(self):
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
        # Define mesh
        mesh = QuadMesh(resolution=(1,2))
        mesh.cells.get_child(1).mark(1)
        mesh.cells.refine(refinement_flag=1)
        mesh.cells.refine()
        
        # Mark left and right boundaries
        bm_left = lambda x,dummy: np.abs(x)<1e-9
        bm_right = lambda x, dummy: np.abs(1-x)<1e-9
        mesh.mark_region('left', bm_left, entity_type='half_edge')
        mesh.mark_region('right', bm_right, entity_type='half_edge')
         
        for etype in ['Q1','Q2','Q3']:
            element = QuadFE(2,etype)            
            
            
            u = Basis(element, 'u')
            ux = Basis(element, 'ux')
            uy = Basis(element, 'uy')
            
            ue = Function(lambda x,dummy: x, 'nodal', mesh=mesh, element=element)
            ax = Form(kernel=Kernel(Function(1,'constant')), trial=ux, test=ux)
            ay = Form(kernel=Kernel(Function(1,'constant')), trial=uy, test=uy)
            L = Form(kernel=Kernel(Function(0,'constant')), test=u)
            problem = [ax, ay, L]
            assembler = Assembler(problem, mesh)
            assembler.assemble()
            system = LinearSystem(assembler)
            system.add_dirichlet_constraint('left',ue)
            system.add_dirichlet_constraint('right',ue)
            system.set_constraint_matrix()
            system.incorporate_constraints()
            system.solve()
            system.resolve_constraints()
            ua = system.sol(as_function=True)            
            self.assertTrue(np.allclose(ua.fn(),ue.fn()))
            
     
    def test06_2d_mixed(self):
        """
        Dirichlet problem with Neumann data on right and Dirichlet data on left
        """
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
            
            # kn = Kernel([ue,ue],['fx','fy'], lambda)
            Ln = Form(kernel=Kernel(one), test=u, dmu='ds', flag='right')
            
    '''
    def test_apply_dirichlet_bc(self):
        """
        
        """
        #
        # 1D Linear 
        # 
        
        # Define mesh
        mesh = Mesh1D(resolution=(100,))
        
        # Define elements and basis functions 
        Q1 = QuadFE(1,'Q1')
        u  = Basis(Q1, 'u')
        ux = Basis(Q1, 'ux')
        
        # Define Kernel functions and right hand sides
        one = Function(1,'constant')
        zero = Function(0,'constant')
        
        # Define Forms
        a = Form(kernel=Kernel(one), trial=ux, test=ux)
        L = Form(kernel=Kernel(zero), test=u)
        
        # Combine forms into problem 
        problem = [a,L]
        
        # Assemble problem over elements
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        
        # Define linear system for problem
        system = LinearSystem(assembler)
        
        # Define Dirichlet conditions on left boundary
        f_left = lambda x: np.abs(x)<1e-9
        mesh.mark_region('left',f_left)
        
        
        # Adjust system to account for Dirichlet boundary conditions
        system.extract_dirichlet_nodes('left')
        
        # Solve system 
        system.solve()
        
        # Return solutin
        u = system.sol(as_function=True)
        
        plot = Plot()
        plot.line(u)
     '''
    '''           
    def test_2D_problems(self):
        """
        Without hanging nodes
        """
        #mesh = QuadMesh(resolution=(2,2))
        
        
        """
        With hanging nodes 
        """
        # Define mesh
        mesh = QuadMesh(resolution=(2,2))
        mesh.cells.get_child(2).mark(1)
        mesh.cells.refine(refinement_flag=1)
        
        
        # Define elements and basis functions
        Q1= QuadFE(2,'Q1')
        u = Basis(Q1, 'u')
        ux = Basis(Q1, 'ux')
        
        # Define functions and kernels
        one = Function(1, 'constant')
        f   = Function(0, 'constant')
        
        # Define forms 
        a = Form(kernel=Kernel(one), trial=ux, test=ux)
        L = Form(kernel=Kernel(f), test=u)
        
        # Combine forms into problen
        problem = [a, L]
        
        # Assemble problem over elements
        assembler = Assembler(problem, mesh)
        assembler.assemble()
        
        dofhandler = assembler.dofhandlers['Q1']
        
        # Check that exact solution satisfies system
        ue = Function(lambda x,dummy: x, 'nodal', dofhandler=dofhandler)
 
        # 
        A = assembler.get_assembled_form('bilinear', 0) 
        b = assembler.get_assembled_form('linear', 0)
        
        print(A.dot(ue.fn())-b)
        
        
        
        # Define linear system
        system = LinearSystem(assembler)
        
        # Mark Dirichlet Regions
        f_left = lambda x,dummy: np.abs(x)<1e-9
        f_right = lambda x,dummy: np.abs(x-1)<1e-9
        
        mesh.mark_region('left', f_left, on_boundary=True)
        mesh.mark_region('right', f_right, on_boundary=True)
        

        
        # Extract Dirichlet conditions 
        system.add_dirichlet_constraint('left', 0)
        system.add_dirichlet_constraint('right',1)
        
        print(system.b())
        print(system.A().todense())
        
        # Solve system 
        system.solve()
        
        # 
        u = system.sol(as_function=True)
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        plot = Plot(quickview=False)
        ax = plot.wire(u, axis=ax)
        plt.show()
    '''
        
    def scratch(self):
        A = np.array([[2,1,0,1],[0,3,5,7],[1,2,2,2],[3,2,1,-1]])
        A = sp.coo_matrix(A)
        b = np.arange(1,5)
        #print(b)
        dir_dofs = [1]
        dir_vals = [10]
        
        A = A.tolil()
        
        r_count = 0
        for row in A.rows:
            if r_count in dir_dofs:
                #print('Row %d is a Dirichlet node'%(r_count))
                #
                # Dirichlet row:
                # 
                i_dbc = dir_dofs.index(r_count)
                val = dir_vals[i_dbc]
                A.rows[r_count] = [r_count]
                A.data[r_count] = [1]
                
                b[r_count] = val         
            else:
                #
                # Look for Dirichlet Columns
                # 
                for dof, val in zip(dir_dofs, dir_vals):
                    if dof in row:
                        #
                        # Nontrivial column associated with Dirichlet node
                        # 
                        i_dof = row.index(dof)
                        b[r_count] -= A.data[r_count][i_dof]*val
                        del A.data[r_count][i_dof]
                        del row[i_dof]
                        
            r_count += 1
            
        #print(A.toarray())
        #print(b)
    
        mesh = Mesh1D(resolution=(3,))
        element = QuadFE(1,'Q1')
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        dofhandler.set_dof_vertices()
        fn = lambda x: (np.abs(x-1)<1e-9) | (np.abs(x)<1e-9)
        for v in mesh.get_boundary_vertices():
            if fn(convert_to_array(v)):
                v.mark('bnd')
        
        dirichlet_dofs = []
        for cell in mesh.cells.get_leaves():        
            for v in cell.get_vertices():
                if v.is_marked('bnd'):
                    dofs = dofhandler.get_global_dofs(cell=cell, entity=v)
                    dirichlet_dofs.extend(dofs)
        
        u = Function(lambda x:x**2+2, 'nodal', dofhandler=dofhandler)
        #print(u.fn()[dirichlet_dofs])

    
    
    
    def test_1d_problems(self):
        """
        Solve a few 1d problems
        """   
        mesh = Mesh1D(resolution=(1,))
        Q2 = QuadFE(1,'Q2')
        f = Function(2, 'constant')
        ux = Basis(Q2,'ux')
        u = Basis(Q2, 'u')
        one = Function(1,'constant')
    
        a = Form(kernel=Kernel(one), trial=ux, test=ux)
        L = Form(kernel=Kernel(f), test=u)
        
        problem = [a,L]
        system = Assembler(problem, mesh)
        system.assemble()
        
        
        for v in mesh.get_boundary_vertices():
            v.mark('D1')
        
        dh = system.dofhandlers['Q2']
        dirichlet_dofs = []
        for cell in mesh.cells.get_leaves():
            for v in cell.get_vertices():
                if v.is_marked('D1'):
                    dofs = dh.get_global_dofs(cell=cell, entity=v)
                    dirichlet_dofs.extend(dofs)
        
        #print('Before applying Dirichlet boundary conditions')            
        A = system.get_assembled_form('bilinear')
        A = A.tolil()
        #print(A.data)
        
        
        #print(A.rows)
        
        #print('-'*20)

        A.rows = np.delete(A.rows, 0, 0)
        A.data = np.delete(A.data, 0, 0)
        
        #print(A.rows)
        #print(A.data)
        
        #print('\n\n\n')
        
        b = system.get_assembled_form('linear')
        #print(b)
        
        for vtx in mesh.get_boundary_vertices():
            vtx.mark(1)
        
        #system.extract_dirichlet_nodes('Doi1') 
        
        
        #print('After applying Dirichlet boundary conditions')
        A = system.get_assembled_form('bilinear')
        #print(A.todense())
        
        #A = system.af[0]['bilinear']
        b = system.get_assembled_form('linear')
        #print(A)
        #print(b)
        
    def test_problem_derivatives(self):
        pass
