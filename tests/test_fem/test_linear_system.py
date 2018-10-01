import unittest
import numpy as np
import scipy.sparse as sp
from mesh import Mesh1D, convert_to_array
from fem import Function, Basis, Form, Kernel, Assembler, DofHandler, QuadFE 

class TestLinearSystem(unittest.TestCase):
    """
    Test Linear System class.
    """
    def test_apply_dirichlet_bc(self):
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
        
        print(type(system.af[0]['bilinear']['rows']))
        
        for v in mesh.get_boundary_vertices():
            v.mark('D1')
        
        dh = system.dofhandlers['Q2']
        dirichlet_dofs = []
        for cell in mesh.cells.get_leaves():
            for v in cell.get_vertices():
                if v.is_marked('D1'):
                    dofs = dh.get_global_dofs(cell=cell, entity=v)
                    dirichlet_dofs.extend(dofs)
        
        print('Before applying Dirichlet boundary conditions')            
        A = system.get_assembled_form('bilinear')
        A = A.tolil()
        print(A.data)
        
        
        print(A.rows)
        
        print('-'*20)

        A.rows = np.delete(A.rows, 0, 0)
        A.data = np.delete(A.data, 0, 0)
        
        print(A.rows)
        print(A.data)
        
        print('\n\n\n')
        
        b = system.get_assembled_form('linear')
        print(b)
        
        for vtx in mesh.get_boundary_vertices():
            vtx.mark(1)
        
        system.extract_dirichlet_nodes('Doi1') 
        
        print(system.af[0]['dirichlet_cols'])
        
        print('After applying Dirichlet boundary conditions')
        A = system.get_assembled_form('bilinear')
        print(A.todense())
        
        #A = system.af[0]['bilinear']
        b = system.get_assembled_form('linear')
        print(A)
        print(b)
        
    def test_problem_derivatives(self):
        pass
