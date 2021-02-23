import unittest

from assembler import Assembler
from assembler import IIForm
from assembler import IPForm
from assembler import  Kernel

from fem import QuadFE
from fem import DofHandler
from fem import Basis

from function import Explicit
from function import Nodal

from mesh import Mesh1D

import numpy as np

class TestForm(unittest.TestCase):
    """
    Test Form 
    """
    def test_constructor(self):
        pass
    
    def test_shape_info(self):
        pass
    
    def test_integration_regions(self):
        pass
    
    def test_eval(self):
        pass
    
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
    
    
class TestIPForm(unittest.TestCase):
    def test_constructor(self):
        pass
    
    def test_eval(self):
        pass
    
    
  
class TestIIForm(unittest.TestCase):
    """
    Test Integral Form
    """ 
    def test_constructor(self):
        pass
    
    def test_eval_interpolation(self):
        #
        # 1D 
        # 
        
        # Mesh 
        mesh = Mesh1D(resolution=(3,))
        
        # Finite element space
        etype = 'Q1'
        element = QuadFE(1,etype)
        
        # Dofhandler
        dofhandler = DofHandler(mesh,element)
        dofhandler.distribute_dofs()
        
        # Basis functions
        phi = Basis(dofhandler, 'u')
        
        # Symmetric kernel function
        kfns = {'symmetric': lambda x,y: x*y,
                'non_symmetric': lambda x,y: x-y}
        
        vals = {'symmetric': np.array([0, 1/9*(1-8/27)]),
                'non_symmetric': np.array([1/3*(-1+8/27),1/6*5/9-1/3*19/27])}
        
        for ktype in ['symmetric', 'non_symmetric']:
            
            # Get kernel function
            kfn = kfns[ktype]
            
            # Define integral kernel
            kernel = Kernel(Explicit(kfn, dim=1, n_variables=2))
            
            # Define Bilinear Form
            form = IIForm(kernel, trial=phi, test=phi)
            
            # 
            # Compute inputs required for evaluating form_loc
            # 
            
            # Assembler 
            assembler = Assembler(form, mesh)
            
            # Cells
            cj = mesh.cells.get_child(2)
            ci = mesh.cells.get_child(0)
    
    
            # Shape function info on cells
            cj_sinfo = assembler.shape_info(cj)
            
            # Gauss nodes and weights on cell 
            xj_g, wj_g, phij, dofsj = assembler.shape_eval(cj_sinfo, cj)
                        
            #
            # Evaluate form
            # 
            form_loc = form.eval(cj, xj_g, wj_g, phij, dofsj)
            #
            # Define functions 
            # 
            u = Nodal(lambda x: x, basis=phi)
            
            #
            # Get local node values
            #
            
            # Degrees of freedom
            cj_dofs = phi.dofs(cj)
            ci_dofs = phi.dofs(ci)
             
            uj = u.data()[np.array(cj_dofs)]
                    
            
            # Evaluate Ici Icj k(x,y) y dy (1-x)dx
            fa = form_loc[ci_dofs].dot(uj)
            fe = vals[ktype][:,None] 
            
            self.assertTrue(np.allclose(fa, fe))

     
    def test_eval_projection(self):
        """
        Test validity of the local projection-based kernel. 
        
        Choose 
        
            u, v in V 
            k(x,y) in VxV (symmetric/non-symm)
            cell01, cell02
            
        Compare
        
            v^T*Kloc*u ?= Icell01 Icell02 k(x,y)u(y)dy dx
        """   
        #
        # 1D
        #
        
        # Mesh 
        mesh = Mesh1D(resolution=(3,))
        
        # Finite element space
        etype = 'Q1'
        element = QuadFE(1,etype)
        
        # Dofhandler
        dofhandler = DofHandler(mesh,element)
        dofhandler.distribute_dofs()
        
        # Basis functions
        phi = Basis(dofhandler, 'u')
        
        # Symmetric kernel function
        kfns = {'symmetric': lambda x,y: x*y,
                'non_symmetric': lambda x,y: x-y}
        
        vals = {'symmetric': (1/2*1/9-1/3*1/27)*(1/3-1/3*8/27),
                'non_symmetric': (1/18-1/3*1/27)*(1/2-2/9)+(1/18-1/3)*(1/3-1/3*8/27)}
        
        for ktype in ['symmetric', 'non_symmetric']:
            
            # Get kernel function
            kfn = kfns[ktype]
            
            # Define integral kernel
            kernel = Kernel(Explicit(kfn, dim=1, n_variables=2))
            
            # Define Bilinear Form
            form = IPForm(kernel, trial=phi, test=phi)
            
            # 
            # Compute inputs required for evaluating form_loc
            # 
            
            # Assembler 
            assembler = Assembler(form, mesh)
            
            # Cells
            ci = mesh.cells.get_child(0)
            cj = mesh.cells.get_child(2)
               
            # Shape function info on cells
            ci_sinfo = assembler.shape_info(ci)
            cj_sinfo = assembler.shape_info(cj)
            
            # Gauss nodes and weights on cell 
            xi_g, wi_g, phii, dofsi = assembler.shape_eval(ci_sinfo,ci)
            xj_g, wj_g, phij, dofsj = assembler.shape_eval(cj_sinfo,cj)
                        
            #
            # Evaluate form
            # 
            form_loc = form.eval((ci,cj), (xi_g,xj_g), \
                                 (wi_g,wj_g), (phii,phij), (dofsi,dofsj))
            #
            # Define functions 
            # 
            u = Nodal(f=lambda x: x, basis=phi)
            v = Nodal(f=lambda x: 1-x, basis=phi)
            
            #
            # Get local node values
            # 
            # Degrees of freedom
            ci_dofs = phi.dofs(ci)
            cj_dofs = phi.dofs(cj)
            uj = u.data()[np.array(cj_dofs)]
            vi = v.data()[np.array(ci_dofs)]
            
            if ktype == 'symmetric':   
                # Local form by hand
                c10 = 1/54 
                c11 = 1/27
                c20 = 3/2-1-3/2*4/9+8/27
                c21 = 4/27
                fl = np.array([[c10*c20, c10*c21],[c11*c20, c11*c21]])
                
                # Compare computed and explicit local forms
                self.assertTrue(np.allclose(fl, form_loc))
            
            
            # Evaluate Ici Icj k(x,y) y dy (1-x)dx
            fa = np.dot(vi.T, form_loc.dot(uj))
            fe = vals[ktype] 
            self.assertAlmostEqual(fa, fe)
