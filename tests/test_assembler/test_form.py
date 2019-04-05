import unittest

from assembler import Assembler
from assembler import IIForm
from assembler import IPForm
from assembler import  Kernel

from fem import QuadFE
from fem import DofHandler
from fem import Basis

from function import Function
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
    
    
class TestIForm(unittest.TestCase):
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
            xj_g, wj_g = assembler.gauss_rules(cj_sinfo)
            
            
            # Shape functions on cell 
            phij = assembler.shape_eval(cj_sinfo, xj_g, cj)
            
            #
            # Evaluate form
            # 
            form_loc = form.eval(cj, xj_g, wj_g, phij)
            #
            # Define functions 
            # 
            u = Nodal(lambda x: x, dofhandler=dofhandler, n_variables=1)
            
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
            xi_g, wi_g = assembler.gauss_rules(ci_sinfo)
            xj_g, wj_g = assembler.gauss_rules(cj_sinfo)
            
            
            # Shape functions on cell 
            phii = assembler.shape_eval(ci_sinfo, xi_g, ci)
            phij = assembler.shape_eval(cj_sinfo, xj_g, cj)
            
            #
            # Evaluate form
            # 
            form_loc = form.eval((ci,cj), (xi_g,xj_g), \
                                 (wi_g,wj_g), (phii,phij))
            #
            # Define functions 
            # 
            u = Function(lambda x: x, 'nodal', dofhandler=dofhandler)
            v = Function(lambda x: 1-x, 'nodal', dofhandler=dofhandler)
            
            #
            # Get local node values
            # 
            # Degrees of freedom
            ci_dofs = phi.dofs(ci)
            cj_dofs = phi.dofs(cj)
            uj = u.fn()[np.array(cj_dofs)]
            vi = v.fn()[np.array(ci_dofs)]
            
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
