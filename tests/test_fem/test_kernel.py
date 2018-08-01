import unittest
from mesh import QuadMesh, Mesh1D, convert_to_array, Tree
from fem import QuadFE, System, Function, GaussRule, Form, Kernel, DofHandler
import numpy as np
import matplotlib.pyplot as plt

class TestKernel(unittest.TestCase):
    """
    Test Kernel Object
    
    Features: 
    
        - Evaluation: 
            + Constant, Nodal, or Explicit
            + Function values, Derivatives
            + Using submeshes (Function?)
        - Sampling:
            + Sample compatibility
            + Evaluation samples
    """
    def test_constructor(self):
        # =====================================================================
        # Test 1D 
        # ===================================================================== 
        
        #
        # Kernel consists of a single explicit Function: 
        # 
        f1 = lambda x: x+2
        f = Function(f1, 'explicit')
        k = Kernel(f)
        x = np.linspace(0,1,100)
        n_points = len(x)
        
        # Check that it evaluates correctly.
        self.assertTrue(np.allclose(f1(x), k.eval(x).ravel()))
        
        # Check shape of kernel
        self.assertEqual(k.eval(x).shape, (n_points,))
        
        #
        # Kernel consists of a combination of two explicit functions
        # 
        f1 = Function(lambda x: x+2, 'explicit')
        f2 = Function(lambda x: x**2 + 1, 'explicit')
        F = lambda f1, f2: f1**2 + f2
        f_t = lambda x: (x+2)**2 + x**2 + 1
        k = Kernel([f1,f2], F=F)
        
        # Check evaluation
        self.assertTrue(np.allclose(f_t(x), k.eval(x).ravel()))
        
        # Check shape 
        self.assertEqual(k.eval(x).shape, (n_points,))
        

        #
        # Same thing as above, but with nodal functions
        # 
        mesh = Mesh1D(resolution=(1,))
        Q1 = QuadFE(1,'Q1')
        Q2 = QuadFE(1,'Q2')
        f1 = Function(lambda x: x+2, 'nodal', mesh=mesh, element=Q1)
        f2 = Function(lambda x: x**2 + 1, 'nodal', mesh=mesh, element=Q2)
        k = Kernel([f1,f2], F=F)
        
        # Check evaluation
        self.assertTrue(np.allclose(f_t(x), k.eval(x).ravel()))
        
        #
        # Replace f2 above with its derivative
        # 
        k = Kernel([f1,f2], dfdx=['f', 'fx'], F=F)
        f_t = lambda x: (x+2)**2 + 2*x
                
        # Check derivative evaluation F = F(f1, df2_dx)
        self.assertTrue(np.allclose(f_t(x), k.eval(x).ravel()))
        
        
        # 
        # Sampling 
        #
        one = Function(1, 'constant')
        f1 = Function(lambda x: x**2 + 1, 'explicit')
        
        # Sampled function
        a = np.linspace(0,1,11)
        n_samples = len(a)
        
        # Define Dofhandler
        dh = DofHandler(mesh, Q2)
        dh.distribute_dofs()
        dh.set_dof_vertices()
        xv = dh.get_dof_vertices()
        n_dofs = dh.n_dofs()
        
        # Evaluate parameterized function at mesh dof vertices
        f2_m  = np.empty((n_dofs, n_samples))
        for i in range(n_samples):
            f2_m[:,i] = xv.ravel() + a[i]*xv.ravel()**2
        f2 = Function(f2_m, 'nodal', dofhandler=dh)
        
        # Define kernel
        F = lambda f1, f2, one: f1 + f2 + one
        k = Kernel([f1,f2,one], F=F)
        
        # Evaluate on a fine mesh
        x = np.linspace(0,1,100)
        n_points = len(x)
        self.assertEqual(k.eval(x).shape, (n_points, n_samples))    
        for i in range(n_samples):
            # Check evaluation
            self.assertTrue(np.allclose(k.eval(x)[:,i], f1.eval(x) + x + a[i]*x**2+ 1))
            
        
        #
        # Sample multiple constant functions
        # 
        f1 = Function(a, 'constant')
        f2 = Function(lambda x: 1 + x**2, 'explicit')
        f3 = Function(f2_m[:,-1], 'nodal', dofhandler=dh)
        
        F = lambda f1, f2, f3: f1 + f2 + f3
        k = Kernel([f1,f2,f3], F=F)
        
        x = np.linspace(0,1,100)
        for i in range(n_samples):
            self.assertTrue(np.allclose(k.eval(x)[:,i], \
                                        a[i] + f2.eval(x) + f3.eval(x)))
        
        #
        # Submeshes
        # 
        mesh = Mesh1D(resolution=(1,))
        mesh_labels = Tree(regular=False)
        
        mesh = Mesh1D(resolution=(1,))
        Q1 = QuadFE(1,'Q1')
        Q2 = QuadFE(1,'Q2')
        
        f1 = Function(lambda x: x, 'nodal', mesh=mesh, element=Q1)
        f2 = Function(lambda x: -2+2*x**2, 'nodal', mesh=mesh, element=Q2)
        one = Function([1,2], 'constant', mesh=mesh)
    
        F = lambda f1, f2, one: 2*f1**2 + f2 + one
        
        I = mesh.cells.get_child(0)
        
        kernel = Kernel([f1,f2, one], F=F)
        
        rule1D = GaussRule(5,shape='interval')
        x = I.reference_map(rule1D.nodes())
        
        # TODO: 2D meshes 
        # TODO: Submeshes
         
        
    def test_n_samples(self):
        pass
    
    def test_eval(self):
        pass
    
    