import unittest
from mesh import QuadMesh, Mesh1D, convert_to_array, Tree
from assembler import Assembler
from assembler import GaussRule
from assembler import Form
from assembler import Kernel
from fem import QuadFE, DofHandler, Basis
from function import Explicit, Nodal, Constant
import numpy as np

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
        f = Explicit(f1, dim=1)
        k = Kernel(f)
        x = np.linspace(0,1,100)
        n_points = len(x)
        
        # Check that it evaluates correctly.
        self.assertTrue(np.allclose(f1(x), k.eval(x).ravel()))
        
        # Check shape of kernel
        self.assertEqual(k.eval(x).shape, (n_points,1))
        
        #
        # Kernel consists of a combination of two explicit functions
        # 
        f1 = Explicit(lambda x: x+2, dim=1)
        f2 = Explicit(lambda x: x**2 + 1, dim=1)
        F = lambda f1, f2: f1**2 + f2
        f_t = lambda x: (x+2)**2 + x**2 + 1
        k = Kernel([f1,f2], F=F)
        
        # Check evaluation
        self.assertTrue(np.allclose(f_t(x), k.eval(x).ravel()))
        
        # Check shape 
        self.assertEqual(k.eval(x).shape, (n_points,1))
        

        #
        # Same thing as above, but with nodal functions
        # 
        mesh = Mesh1D(resolution=(1,))
        Q1 = QuadFE(1,'Q1')
        Q2 = QuadFE(1,'Q2')
        
        dQ1 = DofHandler(mesh,Q1)
        dQ2 = DofHandler(mesh,Q2)
        
        # Distribute dofs
        [dQ.distribute_dofs() for dQ in [dQ1,dQ2]]
        
        # Basis functions
        phi1 = Basis(dQ1,'u')
        phi2 = Basis(dQ2,'u')
        
        f1 = Nodal(lambda x: x+2, basis=phi1)
        f2 = Nodal(lambda x: x**2 + 1, basis=phi2)
        k = Kernel([f1,f2], F=F)
        
        # Check evaluation
        self.assertTrue(np.allclose(f_t(x), k.eval(x).ravel()))
        
        #
        # Replace f2 above with its derivative
        # 
        k = Kernel([f1,f2], derivatives=['f', 'fx'], F=F)
        f_t = lambda x: (x+2)**2 + 2*x
                
        # Check derivative evaluation F = F(f1, df2_dx)
        self.assertTrue(np.allclose(f_t(x), k.eval(x).ravel()))
        
        
        # 
        # Sampling 
        #
        one = Constant(1)
        f1 = Explicit(lambda x: x**2 + 1, dim=1)
        
        # Sampled function
        a = np.linspace(0,1,11)
        n_samples = len(a)
        
        # Define Dofhandler
        dh = DofHandler(mesh, Q2)
        dh.distribute_dofs()
        dh.set_dof_vertices()
        xv = dh.get_dof_vertices()
        n_dofs = dh.n_dofs()
        
        phi = Basis(dh, 'u')
        
        # Evaluate parameterized function at mesh dof vertices
        f2_m  = np.empty((n_dofs, n_samples))
        for i in range(n_samples):
            f2_m[:,i] = xv.ravel() + a[i]*xv.ravel()**2
        f2 = Nodal(data=f2_m, basis=phi)
        
        # Define kernel
        F = lambda f1, f2, one: f1 + f2 + one
        k = Kernel([f1,f2,one], F=F)
        
        # Evaluate on a fine mesh
        x = np.linspace(0,1,100)
        n_points = len(x)
        self.assertEqual(k.eval(x).shape, (n_points, n_samples))    
        for i in range(n_samples):
            # Check evaluation            
            self.assertTrue(np.allclose(k.eval(x)[:,i], f1.eval(x)[:,i] + x + a[i]*x**2+ 1))
            
        
        #
        # Sample multiple constant functions
        # 
        f1 = Constant(data=a)
        f2 = Explicit(lambda x: 1 + x**2, dim=1)
        f3 = Nodal(data=f2_m[:,-1], basis=phi)
        
        F = lambda f1, f2, f3: f1 + f2 + f3
        k = Kernel([f1,f2,f3], F=F)
        
        x = np.linspace(0,1,100)
        for i in range(n_samples):
            self.assertTrue(np.allclose(k.eval(x)[:,i], \
                                        a[i] + f2.eval(x)[:,i] + f3.eval(x)[:,i]))
        
        #
        # Submeshes
        # 
        mesh = Mesh1D(resolution=(1,))
        mesh_labels = Tree(regular=False)
        
        mesh = Mesh1D(resolution=(1,))
        Q1 = QuadFE(1,'Q1')
        Q2 = QuadFE(1,'Q2')
        
        dQ1 = DofHandler(mesh,Q1)
        dQ2 = DofHandler(mesh,Q2)
        
        # Distribute dofs
        [dQ.distribute_dofs() for dQ in [dQ1,dQ2]]
        
        # Basis
        p1 = Basis(dQ1)
        p2 = Basis(dQ2) 
        
        
        f1 = Nodal(lambda x: x, basis=p1)
        f2 = Nodal(lambda x: -2+2*x**2, basis=p2)
        one = Constant(np.array([1,2]))
    
        F = lambda f1, f2, one: 2*f1**2 + f2 + one
        
        I = mesh.cells.get_child(0)
        
        kernel = Kernel([f1,f2, one], F=F)
        
        rule1D = GaussRule(5,shape='interval')
        x = I.reference_map(rule1D.nodes())
        
        # TODO: 2D meshes 
        # TODO: Submeshes
         
        
    def test_n_samples(self):
        # 
        # 1D
        #
        
        # Define mesh
        mesh = Mesh1D(resolution=(10,))
        
        # Define function  
        f = Explicit([lambda x: x, lambda x: -2+2*x**2], dim=1)
        n_samples = f.n_samples()
        
        k = Kernel(f)
        
        n_points = 101
        x0, x1 = mesh.bounding_box()
        x = np.linspace(x0,x1,n_points)
        
        self.assertEqual(k.eval(x).shape, (n_points, n_samples))
        self.assertTrue(np.allclose(k.eval(x)[:,0],x))
        self.assertTrue(np.allclose(k.eval(x)[:,1], -2+2*x**2))
                         
    def test_eval(self):
        pass
    
    
class TestIKernel(unittest.TestCase):
    """
    Test Integral kernel
    """
    def test_eval(self):
        #
        # 1D 
        # 
        
        # Define Kernel function
        def k_fn(x,y,c = 2):
            return x*y + c
        
        k = Explicit(f=lambda x,y,c: x*y+c, parameters={'c':1}, 
                     n_variables=2, dim=1)
        
        # Construct kernel, specifying parameter c
        kernel = Kernel(k)
        
        # Evaluation points
        x = np.ones((11,1))
        y = np.linspace(0,1,11)[:,None]
        
        # Check accuracy
        self.assertTrue(np.allclose(kernel.eval((x,y)), x*y+1))
        
        # Define kernel with default parameters
        k.set_parameters({'c':2})
        
        kernel = Kernel(k)
        
        # Check accuracy
        self.assertTrue(np.allclose(kernel.eval((x,y)), x*y+2))
        
    
    
    
    