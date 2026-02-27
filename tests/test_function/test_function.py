import unittest
from mesh import convert_to_array
from mesh import Mesh1D
from mesh import Mesh2D
from mesh import QuadMesh
from mesh import DCEL
from mesh import Forest

from fem import QuadFE
from fem import DofHandler
from fem import Basis

from function import Map
from function import Constant
from function import Explicit
from function import Nodal

import numbers 

from plot import Plot

import numpy as np
import matplotlib.pyplot as plt


class TestMap(unittest.TestCase):
    """
    Test the parent class
    """
    def test_constructor(self):
        """
        Test initialization
        """
        # Initialize empty map
        f = Map()
        self.assertTrue(isinstance(f, Map))
        self.assertIsNone(f.dim())
        
        #
        # Exceptions
        # 

        mesh = Mesh1D()
        
        # Mesh dimension incompatible with element
        element = QuadFE(2, 'DQ1')
        #self.assertRaises(Exception, Map, **{'mesh':mesh, 'element':element})

        # Dofhandler incompatibility
        element = QuadFE(1, 'Q1')
        dofhandler = DofHandler(mesh, element)
        #self.assertRaises(Exception, Map, **{'dofhandler': dofhandler, 
        #                                     'dim':2})
        
        # function returns the same mesh
        f1 = Map(dofhandler=dofhandler)
        f2 = Map(mesh=mesh)
        self.assertEqual(f1.mesh(),f2.mesh())
        
    
    def interpolate(self):
        pass
    
    def test_subsample_deterministic(self):
        """
        When evaluating a deterministic function while specifying a subsample,
        n_subsample copies of the function output should be returned.        
        """
        #
        # Deterministic functions 
        #
            
        # Functions  
        fns = {1: {1: lambda x: x[:,0]**2, 2: lambda x,y: x[:,0] + y[:,0]}, 
               2: {1: lambda x: x[:,0]**2 + x[:,1]**2, 
                   2: lambda x,y: x[:,0]*y[:,0] + x[:,1]*y[:,1]}}
        
        # Singletons
        x  = {1: {1: 2,     2: (3,4)},
              2: {1: (1,2), 2: ((1,2),(3,4))}}
        
        xv = {1: {1: [(2,),(2,)],     
                  2: ([(3,),(3,)],[(4,),(4,)])},
              2: {1: [(1,2),(1,2)], 
                  2: ([(1,2),(1,2)],[(3,4),(3,4)])}}
        
        vals = {1: {1: 4, 2: 7}, 2: {1: 5, 2: 11}} 
        subsample = np.array([2,3], dtype=int)
        
        for dim in [1,2]:
            #
            # Iterate over dimension
            # 
            
            # DofHandler
            if dim==1:
                mesh = Mesh1D(box=[0,5], resolution=(1,))
            elif dim==2:
                mesh = QuadMesh(box=[0,5,0,5])
            element = QuadFE(dim, 'Q2')
            dofhandler = DofHandler(mesh, element)
            dofhandler.distribute_dofs()
            basis = Basis(dofhandler)
            for n_variables in [1,2]:
                #
                # Iterate over number of variables
                # 
                
                #
                # Explicit
                # 
                f = fns[dim][n_variables]
                
                # Explicit
                fe = Explicit(f, n_variables=n_variables, dim=dim, \
                             subsample=subsample)
                
                # Nodal        
                fn = Nodal(f, n_variables=n_variables, basis=basis, dim=dim, \
                           dofhandler=dofhandler, subsample=subsample)
                
                # Constant
                fc = Constant(1, n_variables=n_variables, \
                              subsample=subsample)
                
                
                # Singleton input
                xn = x[dim][n_variables]
                
                # Explicit
                self.assertEqual(fe.eval(xn).shape[1],len(subsample))
                self.assertEqual(fe.eval(xn)[0,0],vals[dim][n_variables])
                self.assertEqual(fe.eval(xn)[0,1],vals[dim][n_variables])
                
                
                # Nodal
                self.assertEqual(fn.eval(xn).shape[1],len(subsample))
                self.assertAlmostEqual(fn.eval(xn)[0,0],vals[dim][n_variables])
                self.assertAlmostEqual(fn.eval(xn)[0,1],vals[dim][n_variables])
                
                
                # Constant
                self.assertEqual(fc.eval(xn).shape[1],len(subsample))
                self.assertAlmostEqual(fc.eval(xn)[0,0],1)
                self.assertAlmostEqual(fc.eval(xn)[0,1],1)
                
                # Vector input
                xn = xv[dim][n_variables]
                n_points = 2
                
                # Explicit                
                self.assertEqual(fe.eval(xn).shape, (2,2))
                for i in range(fe.n_subsample()):
                    for j in range(n_points):
                        self.assertEqual(fe.eval(xn)[i][j],vals[dim][n_variables])
            
                # Nodal
                self.assertEqual(fn.eval(xn).shape, (2,2))
                for i in range(fe.n_subsample()):
                    for j in range(n_points):
                        self.assertAlmostEqual(fn.eval(xn)[i][j],vals[dim][n_variables])
                
                # Constant 
                self.assertEqual(fc.eval(xn).shape, (2,2))
                for i in range(fe.n_subsample()):
                    for j in range(n_points):
                        self.assertEqual(fc.eval(xn)[i][j],1)
                
        
    def test_subsample_stochastic(self):
        """
        
        
        #
        # Evaluate sampled functions
        # 
        fns = {1: {1: lambda x,a: a*x**2, 
                   2: lambda x,y,a: a*(x + y)}, 
               2: {1: lambda x,a: a*(x[:,0]**2 + x[:,1]**2), 
                   2: lambda x,y,a: a*(x[:,0]*y[:,0] + x[:,1]*y[:,1])}}
        
        bad_subsample = np.array([2,3], dtype=np.int)
        subsample = np.array([0], dtype=np.int)
        
        pars = [{'a': 1}, {'a':2}]   
        for dim in [1,2]:
            #
            # Iterate over dimension
            # 
            for n_variables in [1,2]:
                #
                # Iterate over number of variables
                # 
                fn = fns[dim][n_variables]
                self.assertRaises(Exception, Explicit, *(fn,), 
                                  **{'parameters':pars, 
                                     'n_variables':n_variables, 
                                     'dim':dim, 'subsample':bad_subsample})
        
        #
        # 2 points
        # 
        n_points = 2
        for dim in [1,2]:
            #
            # Iterate over dimension
            # 
            for n_variables in [1,2]:
                #
                # Iterate over number of variables
                # 
                fn = fns[dim][n_variables]
                f = Explicit(fn, parameters={}, 
                             n_variables=n_variables, dim=dim, 
                             subsample=subsample)
                
                xn = x[dim][n_variables]
                
                
                #self.assertEqual(f.eval(xn).shape[0],n_points)
                #self.assertEqual(f.eval(xn).shape[1],f.n_samples())
                
                #for i in range(f.n_samples()):
                #        val = pars[i]['a']*vals[dim][n_variables]
                #        self.assertEqual(f.eval(xn)[j,i], val)
        """
         
    
    
class TestExplicit(unittest.TestCase):
    """
    Test explicit functions
    """
    def test_constructor(self):
        """
        Test initialization
        """
        f = Explicit(lambda x, a: a*x**2, parameters={'a':1}, 
                     dim=1, n_variables=1)
        f.eval(1)
        self.assertAlmostEqual(f.eval(1),1)
        
        #
        # Errors
        #
        self.assertRaises(Exception, Explicit)
        
        
         
    def test_set_rules(self):
        
        #
        # Errors
        # 
        
        # Number of parameter dicts differs from number of rules  
        flist = [lambda x,a: a*x**2, lambda x: x]
        plist = [{}, {}, {}]
        self.assertRaises(Exception, Explicit, *(flist,), 
                          **{'parameters':plist, 'dim':1})
    
        # Specify that function is symmetric, but n_variables=1
        f = lambda x: x
        self.assertRaises(Exception, Explicit, *(f,), **{'symmetric':True})
        
        
        #
        # Expected 
        # 
        
        # Only one set of parameters multiple functions
        f = Explicit(flist, parameters={}, dim=1)
        self.assertEqual(len(f.parameters()),2)
        
        # Only one function multiple parameters
        rule = lambda x,a: a*x**2
        parameters = [{'a':1}, {'a': 2}]
        f = Explicit(rule, parameters=parameters, dim=1)
        self.assertEqual(len(f.rules()),2)
        
    
    def test_add_rules(self):
        # Define explicit function
        rule = lambda x,a: a*x**2
        parameters = [{'a':1}, {'a': 2}]
        f = Explicit(rule, parameters=parameters, dim=1)
        
        
        # Raises an error attempt to add multiple functions 
        self.assertRaises(Exception, f.add_rule, *(lambda x: x,), 
                          **{'parameters':[{},{}]})
        
        # Add a single rule and evaluate at a point
        f.add_rule(lambda x: x)
        x = 2
        vals = [4, 8, 2]
        
        for i in range(3):
            
            #(*(x,),**{'a':1}))
            self.assertEqual(f.eval(x)[0,i],vals[i])
        
        
    def test_set_parameters(self):
        # Define explicit function 
        rule = lambda x, a: a*x[:,0]-x[:,1]
        parms = [{'a':1}, {'a':2}]
        f = Explicit(rule, parameters=parms, dim=2)
        
        x = (1,2)
        self.assertTrue( np.allclose(f.eval(x), np.array([-1,0])) )
    
        # Modify parameter
        f.set_parameters({'a':2}, pos=0)
        self.assertTrue( np.allclose(f.eval(x), np.array([0,0])) )
    
    
    
    def test_eval(self):
        
        #
        # Evaluate single univariate/bivariate functions in 1 or 2 dimensions
        # 
        fns = {1: {1: lambda x: x**2, 2: lambda x,y: x + y}, 
               2: {1: lambda x: x[:,0]**2 + x[:,1]**2, 
                   2: lambda x,y: x[:,0]*y[:,0] + x[:,1]*y[:,1]}}
        
        # Singletons
        x  = {1: {1: 2,     2: (3,4)},
              2: {1: (1,2), 2: ((1,2),(3,4))}}
        
        vals = {1: {1: 4, 2: 7}, 2: {1: 5, 2: 11}} 
        
        for dim in [1,2]:
            #
            # Iterate over dimension
            # 
            for n_variables in [1,2]:
                #
                # Iterate over number of variables
                # 
                fn = fns[dim][n_variables]
                f = Explicit(fn, n_variables=n_variables, dim=dim)
                
                xn = x[dim][n_variables]
                self.assertEqual(f.eval(xn),vals[dim][n_variables])
        
        #
        # Evaluate sampled functions
        # 
        fns = {1: {1: lambda x,a: a*x**2, 
                   2: lambda x,y,a: a*(x + y)}, 
               2: {1: lambda x,a: a*(x[:,0]**2 + x[:,1]**2), 
                   2: lambda x,y,a: a*(x[:,0]*y[:,0] + x[:,1]*y[:,1])}}
        
        pars = [{'a': 1}, {'a':2}]
        
        #
        # Singletons
        #
        x  = {1: {1: 2,     2: (3,4)},
              2: {1: (1,2), 2: ((1,2),(3,4))}}
        
        vals = {1: {1: 4, 2: 7}, 2: {1: 5, 2: 11}} 
        
        for dim in [1,2]:
            #
            # Iterate over dimension
            # 
            for n_variables in [1,2]:
                #
                # Iterate over number of variables
                # 
                fn = fns[dim][n_variables]
                f = Explicit(fn, parameters=pars, 
                             n_variables=n_variables, dim=dim)
                
                xn = x[dim][n_variables]
                self.assertEqual(f.eval(xn)[0][0],vals[dim][n_variables])
                self.assertEqual(f.eval(xn)[0][1],2*vals[dim][n_variables])
        
        #
        # 2 points
        # 
        n_points = 2
        x  = {1: {1: [(2,),(2,)],     
                  2: ([(3,),(3,)],[(4,),(4,)])},
              2: {1: [(1,2),(1,2)], 
                  2: ([(1,2),(1,2)],[(3,4),(3,4)])}}
        for dim in [1,2]:
            #
            # Iterate over dimension
            # 
            for n_variables in [1,2]:
                #
                # Iterate over number of variables
                # 
                fn = fns[dim][n_variables]
                f = Explicit(fn, parameters=pars, 
                             n_variables=n_variables, dim=dim)
                
                xn = x[dim][n_variables]
                self.assertEqual(f.eval(xn).shape[0],n_points)
                self.assertEqual(f.eval(xn).shape[1],f.n_samples())
                
                for i in range(f.n_samples()):
                    for j in range(2):
                        val = pars[i]['a']*vals[dim][n_variables]
                        self.assertEqual(f.eval(xn)[j,i], val)
        

class TestNodal(unittest.TestCase):
    """
    Test Nodal test functions
    """ 
    def test_constructor(self):
        # 
        # Errors
        #
        
        # Nothing specified
        self.assertRaises(Exception, Nodal)
        
        # Nominal case
        mesh = QuadMesh()
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh,element)
        dofhandler.distribute_dofs()
        basis = Basis(dofhandler)
        
        data = np.arange(0,4)
        
        f = Nodal(data=data, basis=basis, mesh=mesh, element=element)
        self.assertEqual(f.dim(),2)
        self.assertTrue(np.allclose(f.data().ravel(),data))
        
        # Now change the data -> Error
        false_data = np.arange(0,6)
        self.assertRaises(Exception,Nodal, 
                          **{'data':false_data, 'mesh':mesh, 'element':element})
    
        # Now omit mesh or element
        kwargs = {'data': data, 'mesh': mesh}
        self.assertRaises(Exception, Nodal, **kwargs)
        
        kwargs = {'data': data, 'element': element}
        self.assertRaises(Exception, Nodal, **kwargs)
    
        
    def test_set_data(self): 
        meshes = {1: Mesh1D(), 2: QuadMesh()}
        elements = {1: QuadFE(1, 'Q2'), 2: QuadFE(2, 'Q2')}
        
        #
        # Use function to set data
        #
        fns = {1: {1: lambda x: 2*x[:,0]**2, 
                   2: lambda x,y: 2*x[:,0] + 2*y[:,0]}, 
               2: {1: lambda x: x[:,0]**2 + x[:,1], 
                   2: lambda x,y: x[:,0]*y[:,0]+x[:,1]*y[:,1]}}
        
        parms = {1: {1: [{},{}], 
                     2: [{},{}]}, 
                 2: {1: [{},{}], 
                     2: [{},{}]}}
        
        for dim in [1,2]:
            # Get mesh and element
            mesh = meshes[dim]
            element = elements[dim]
            
            # Set dofhandler
            dofhandler = DofHandler(mesh, element)
            dofhandler.distribute_dofs()
            n_dofs = dofhandler.n_dofs()
            
            # Set basis
            basis = Basis(dofhandler)
            
            # Determine the shapes of the data
            det_shapes = {1: (n_dofs, 1),  2: (n_dofs,n_dofs, 1)}
            smp_shapes = {1: (n_dofs, 2), 2: (n_dofs, n_dofs, 2)}
            
            # Get a vertex
            i = np.random.randint(n_dofs)
            j = np.random.randint(n_dofs)
            dofhandler.set_dof_vertices()
            x = dofhandler.get_dof_vertices()
            
            x1 = np.array([x[i,:]])
            x2 = np.array([x[j,:]])
                        
            for n_variables in [1,2]:
                fn = fns[dim][n_variables]
                parm = parms[dim][n_variables]
                #
                # Deterministic
                # 
                f = Nodal(f=fn, basis=basis, n_variables=n_variables)
                
                # Check shape
                self.assertEqual(f.data().shape, det_shapes[n_variables])
                
                # Check value
                if n_variables==1:
                    val = fn(x1)[0]
                    self.assertEqual(val, f.data()[i])
                else:
                    val = fn(x1, x2)
                    self.assertEqual(val[0], f.data()[i,j])
                
                #
                # Sampled 
                # 
                f = Nodal(f=fn, parameters=parm, basis=basis, n_variables=n_variables)
                
                # Check shape
                self.assertEqual(f.data().shape, smp_shapes[n_variables])
                                
                # Check that samples are stored in the right place
                if n_variables==1:
                    self.assertTrue(np.allclose(f.data()[:,0],f.data()[:,1]))
                elif n_variables==2:
                    self.assertTrue(np.allclose(f.data()[:,:,0], f.data()[:,:,1]))
                
        #
        # Use arrays to set data        
        # 
        
                        
    def test_add_data(self):
        pass
    
    
    def test_n_samples(self):
        #
        # Sampled Case
        # 
        meshes = {1: Mesh1D(), 2: QuadMesh()}
        elements = {1: QuadFE(1, 'Q2'), 2: QuadFE(2, 'Q2')}
            
        # Use function to set data
        fns = {1: {1: lambda x: 2*x[:,0]**2, 
                   2: lambda x,y: 2*x[:,0] + 2*y[:,0]}, 
               2: {1: lambda x: x[:,0]**2 + x[:,1], 
                   2: lambda x,y: x[:,0]*y[:,0]+x[:,1]*y[:,1]}}
        
        # n_samples = 2
        parms = {1: {1: [{},{}], 
                     2: [{},{}]}, 
                 2: {1: [{},{}], 
                     2: [{},{}]}}
        
        for dim in [1,2]:
            mesh = meshes[dim]
            element = elements[dim]
            dofhandler = DofHandler(mesh,element)
            dofhandler.distribute_dofs()
            basis = Basis(dofhandler)
            for n_variables in [1,2]:
                fn = fns[dim][n_variables]
                parm = parms[dim][n_variables]
                #
                # Deterministic
                # 
                f = Nodal(f=fn,  
                          mesh=mesh, basis=basis, element=element, 
                          dim=dim, n_variables=n_variables)
                self.assertEqual(f.n_samples(),1)
                
                #
                # Sampled
                # 
                f = Nodal(f=fn, parameters=parm, basis=basis,
                          mesh=mesh, element=element, 
                          dim=dim, n_variables=n_variables)
                self.assertEqual(f.n_samples(),2)
        
    
    def test_derivative(self):
        """
        Compute the derivatives of a Nodal Map
        """
        # Define meshes for each dimension
        meshes = {1: Mesh1D(resolution=(2,)), 
                  2: QuadMesh(resolution=(2,2))}
        
        # Define elements for each dimension
        elements = {1: QuadFE(1, 'Q2'), 
                    2: QuadFE(2, 'Q2')}
        
        
        # Use function to set data
        fns = {1: lambda x: 2*x[:,0]**2, 
               2: lambda x: x[:,0]**2 + x[:,1]} 
        
        derivatives = {1: [(1,0),(2,0)], 
                       2: [(1,0), (1,1), (2,0,0)]}
        
        
        dfdx_exact = {1: [lambda x: 4*x[:,0][:,None], 
                          lambda x: 4*np.ones(x.shape)],
                      2: [lambda x: 2*x[:,0][:,None], 
                          lambda x: np.ones(x.shape),
                          lambda x: 2*np.ones(x.shape)]}
                
        # n_samples = 2
        parms = {1: [{},{}], 
                 2: [{},{}]} 
        
        for dim in [1,2]:
            mesh = meshes[dim]
            
            # Random points in domain
            n_points = 5
            if dim==1:
                x_min, x_max = mesh.bounding_box()
                x = x_min + 0.5*(x_max-x_min)*np.random.rand(n_points)
                x = x[:,np.newaxis]
            elif dim==2:
                x_min, x_max, y_min, y_max = mesh.bounding_box()
                x = np.zeros((n_points,2))
                x[:,0] = x_min + (x_max-x_min)*np.random.rand(n_points)
                x[:,1] = y_min + (y_max-y_min)*np.random.rand(n_points)
                
            element = elements[dim]
            fn = fns[dim]
            dofhandler = DofHandler(mesh, element)
            dofhandler.distribute_dofs()
            basis = Basis(dofhandler)
            #
            # Deterministic
            # 
            f = Nodal(f=fn, basis=basis, mesh=mesh, element=element, 
                      dim=dim, n_variables=1)
            
            count = 0
            for derivative in derivatives[dim]:
                # Evaluate the derivative
                dfdx = f.differentiate(derivative)
                self.assertTrue(np.allclose(dfdx.eval(x=x), 
                                            dfdx_exact[dim][count](x)))
                count += 1
            #
            # Sampled
            #
            parm = parms[dim] 
            f = Nodal(f=fn, parameters=parm, basis=basis, mesh=mesh, element=element, dim=dim)
            count = 0
            for derivative in derivatives[dim]:
                # Evaluate the derivative
                dfdx = f.differentiate(derivative)
                
                self.assertTrue(np.allclose(dfdx.eval(x=x)[:,0], 
                                            dfdx_exact[dim][count](x)[:,0]))
                self.assertTrue(np.allclose(dfdx.eval(x=x)[:,1], 
                                            dfdx_exact[dim][count](x)[:,0]))
                count += 1
    
    
    def test_eval_phi(self):
        """
        Check that the shapes are correct
        """
        #
        # Mesh
        # 
        mesh = QuadMesh()
        dim = mesh.dim()
        
        # 
        # element information
        #
        element = QuadFE(dim,'Q2')
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        
        # Basis
        basis = Basis(dofhandler)

        # Define phi
        n_points = 5
        phi = np.random.rand(n_points,2)
        dofs = [0,1]
        
        # 
        # Deterministic Data
        #
        n_dofs = dofhandler.n_dofs()
        data = np.random.rand(n_dofs)
        
        # Define deterministic function
        f = Nodal(data=data, basis=basis)
        
        # Evaluate and check dimensions
        fx = f.eval(phi=phi, dofs=dofs) 
        self.assertEqual(fx.shape, (n_points,1))
        
        #
        # Sampled Data
        #
        n_samples = 4
        data = np.random.rand(n_dofs,n_samples)
        
        # Define stochastic function
        f = Nodal(data=data, basis=basis, dofhandler=dofhandler)
        
        # Evaluate and check dimensions
        fx = f.eval(phi=phi, dofs=dofs)
        self.assertEqual(fx.shape,(n_points,n_samples))
        
        #
        # Bivariate deterministic
        # 
        data = np.random.rand(n_dofs, n_dofs, 1)
        
        # Define deterministic function
        f = Nodal(data=data, basis=basis, dofhandler=dofhandler, n_variables=2)
        fx = f.eval(phi=(phi,phi), dofs=(dofs,dofs))
        self.assertEqual(fx.shape, (n_points,1))
        
        #
        # Bivariate sampled
        # 
        data = np.random.rand(n_dofs, n_dofs, n_samples)
        
        # Define stochastic function
        f = Nodal(data=data, basis=basis, dofhandler=dofhandler, n_variables=2)
        fx = f.eval(phi=(phi,phi), dofs=(dofs,dofs))
        self.assertEqual(fx.shape, (n_points, n_samples))
        
        #
        # Trivariate deterministic
        # 
        data = np.random.rand(n_dofs, n_dofs, n_dofs,1)
        f = Nodal(data=data, basis=basis, dofhandler=dofhandler, n_variables=3)
        
    
    def test_eval_x(self):
        #
        # Evaluate Nodal function at a given set of x-values
        # 
        
        # Meshes and elements
        meshes = {1: Mesh1D(resolution=(2,)), 2: QuadMesh(resolution=(2,1))}
        elements = {1: QuadFE(1, 'Q2'), 2: QuadFE(2, 'Q2')}
        
        
        # Use function to set data
        fns = {1: {1: lambda x: 2*x[:,0]**2, 
                   2: lambda x,y: 2*x[:,0] + 2*y[:,0]}, 
               2: {1: lambda x: x[:,0]**2 + x[:,1], 
                   2: lambda x,y: x[:,0]*y[:,0]+x[:,1]*y[:,1]}}
        
        # n_samples = 2
        parms = {1: {1: [{},{}], 
                     2: [{},{}]}, 
                 2: {1: [{},{}], 
                     2: [{},{}]}}
        
        n_points = 1
        for dim in [1,2]:
            mesh = meshes[dim]
            element = elements[dim]
            dofhandler = DofHandler(mesh, element)
            dofhandler.distribute_dofs()
            #dofhandler.get_region_dofs()
            basis = Basis(dofhandler)
            #
            # Define random points in domain
            #
            if dim==1:
                x_min, x_max = mesh.bounding_box()
                
                x = x_min + 0.5*(x_max-x_min)*np.random.rand(n_points)
                x = x[:,np.newaxis]
                
                y = x_min + (x_max-x_min)*np.random.rand(n_points)
                y = y[:,np.newaxis]
            elif dim==2:
                x_min,x_max,y_min,y_max = mesh.bounding_box()
                
                x = np.zeros((n_points,2))
                x[:,0] = x_min + (x_max-x_min)*np.random.rand(n_points)
                x[:,1] = y_min + (y_max-y_min)*np.random.rand(n_points)
                
                y = np.zeros((n_points,2))
                y[:,0] = x_min + (x_max-x_min)*np.random.rand(n_points)
                y[:,1] = y_min + (y_max-y_min)*np.random.rand(n_points)
                

            for n_variables in [1,2]:
                fn = fns[dim][n_variables]
                parm = parms[dim][n_variables]
                #
                # Deterministic
                # 
                f = Nodal(f=fn,basis=basis,
                          mesh=mesh, element=element, 
                          dim=dim, n_variables=n_variables)
                
                if n_variables==1:
                    xx = x
                    fe = fn(x)
                elif n_variables==2:
                    xx = (x,y)
                    fe = fn(*xx)
                fx = f.eval(x=xx)
                self.assertTrue(np.allclose(fx,fe))
                                
                #
                # Sampled
                # 
                f = Nodal(f=fn, parameters=parm, basis=basis,
                          mesh=mesh, element=element, 
                          dim=dim, n_variables=n_variables)
                self.assertEqual(f.n_samples(),2)
                
                
                fx = f.eval(x=xx)
                self.assertTrue(np.allclose(fx[:,0],fe))
                self.assertTrue(np.allclose(fx[:,1],fe))
                
    
    def test_project(self):
        """
        Test projection operator
        """
        mesh = QuadMesh()
        mesh.record(0)
        
        # Refine mesh a couple of times
        for i in range(4):   
            mesh.cells.refine(new_label=i+1)
        
        element = QuadFE(2,'DQ0')
        
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        phi = Basis(dofhandler)
        
        plot = Plot()
        plot.mesh(mesh, subforest_flag=4)
                
        f = Nodal(lambda x: np.cos(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1]), basis=phi)
        plot.contour(f)    
        
        dof_forest = Forest()
        
         
    
class TestConstant(unittest.TestCase):
    """
    Test Constant function
    """
    def test_constructor(self):
        pass
    
    
    def test_set_data(self):
        pass
    
    
    def test_eval(self):
        pass
    
        
'''        
class TestFunction(unittest.TestCase):
    """
    Test Function class
    
    
    a. Define function on a submesh, refine it
    """
    def test_constructor_and_eval(self):
        """
        Test Constructor
        """     
        mesh = QuadMesh()
        mesh.cells.record(0)
        mesh.cells.refine(new_label=1)
        mesh.cells.refine(new_label=2)
        
              
        f = lambda x,y: np.sin(np.pi*x)*np.cos(2*np.pi*y) + np.cos(np.pi*y)
        
        fig = plt.figure()
        
        #
        # Mesh points at which to plot function
        #
        vtcs = mesh.cells.get_child(0).get_vertices() 
        vtcs = convert_to_array(vtcs)
        x0, y0 = vtcs[0,:]
        x1, y1 = vtcs[2,:] 
        nx, ny = 30,30
        x,y = np.linspace(x0,x1,nx), np.linspace(y0,y1,ny)
        xx, yy = np.meshgrid(x,y)
        xy = np.array([xx.ravel(),yy.ravel()]).transpose()
        
        #
        # Test interpolant and eval
        #
        fn = Function(f,'explicit', dim=2)
        
        
        ax = fig.add_subplot(3,4,1, projection='3d')
        #for node in mesh.root_node().get_leaves():
        #    cell = node.cell()
        #    print(cell.contains_point(xy).shape)
        zz = fn.eval(xy)  
        ax.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                        linewidth=0, antialiased=True)
        ax.set_title('f')
        #
        # Interpolate function: continous elements
        # 
        continuous_etype = ['Q1','Q2','Q3']
        for i in range(3):
            etype = continuous_etype[i]
            element = QuadFE(2,etype)
            fn_interp = fn.interpolant(mesh, element)
            
            ax = fig.add_subplot(3,4,2+i, projection='3d')
            zz = fn_interp.eval(xy)
            ax.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                            linewidth=0, antialiased=True)
            ax.set_title(etype)
        #
        # Interpolate function: discontinuous elements
        #
        """
        Note: The plots in row 2 should be the same as those in row 1, with
            the exception of DQ0.
        """
        discontinuous_etype = ['DQ0','DQ1','DQ2','DQ3']
        for i in range(4):
            etype = discontinuous_etype[i]
            element = QuadFE(2,etype)
            
            dh = DofHandler(mesh, element)
            dh.distribute_dofs()
            dh.set_dof_vertices()
            x = dh.get_dof_vertices()
        
            fn_interp = fn.interpolant(mesh, element)
            
            ax = fig.add_subplot(3,4,5+i, projection='3d')
            zz = fn_interp.eval(xy)
            ax.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                            linewidth=0, antialiased=True)
            ax.set_title(etype)
        
        
        #
        # Differentiate the function with respect to y
        # 
        etype_list = ['DQ0','Q1','Q2','Q3']
        for i in range(4):
            etype = etype_list[i]
            element = QuadFE(2,etype)
            fn_interp = fn.interpolant(mesh, element)
            
            ax = fig.add_subplot(3,4,9+i, projection='3d')
            df_dx = fn_interp.derivative((1,1))
            zz = df_dx.eval(xy)
            ax.plot_surface(xx,yy,zz.reshape(xx.shape),cmap='viridis', \
                            linewidth=0, antialiased=True)
            ax.set_title(etype)
        
        plt.tight_layout(pad=1, w_pad=2, h_pad=2)
        #plt.show()
        
        #
        # Function defined on meshes of various resolution, evaluated on 
        # the same cell.
        # 
        mesh = Mesh1D(resolution=(1,))
        mesh.cells.record(1)
        mesh.cells.refine(subforest_flag=1, new_label=2)
        
        element = QuadFE(1,'Q1')
        dofhandler = DofHandler(mesh, element)
        
        # Function on mesh coarser than cell
        f1 = Function(lambda x:x**2, 'nodal', \
                      dofhandler=dofhandler, \
                      subforest_flag=1)
        
        # Function on mesh containing cell
        f2 = Function(lambda x:x**2, 'nodal',\
                      dofhandler=dofhandler,\
                      subforest_flag=2)
        
        mesh.cells.refine(subforest_flag=2, new_label=3)
        
        # Function on mesh finer than cell
        f3 = Function(lambda x:x**2, 'nodal',\
                      dofhandler=dofhandler,\
                      subforest_flag=3)
        
        # Evaluate on [0,0.5] at x=1/3 
        cell = mesh.cells.get_child(0).get_child(0)
        x = [(1/3,)]
        
        # Check answers
        self.assertEqual(f1.eval(x=x, cell=cell), np.array([1/3]))
        self.assertEqual(f2.eval(x=x, cell=cell), np.array([1/6]))
        self.assertTrue(np.allclose(f3.eval(x=x, cell=cell), np.array([1/8])))
        
        
        """
        fig = plt.figure()
        plot = Plot()
        element = QuadFE(2,'DQ0')
        fn.interpolant(mesh, element)
        ax = fig.add_subplot(1,3,1)
        ax = plot.mesh(mesh, ax, mesh_flag=0, color_marked=[0])
        ax = fig.add_subplot(1,3,2)
        ax = plot.mesh(mesh, ax, mesh_flag=1, color_marked=[1])
        ax = fig.add_subplot(1,3,3)
        ax = plot.mesh(mesh, ax, mesh_flag=2, color_marked=[2])
        plt.show()
        
        fig, ax = plt.subplots(2,4)
        
        count = 0
        for etype in etype_list:
            print(etype)            
            element = QuadFE(2,etype)
            dh = DofHandler(mesh,element)
            dh.distribute_dofs(nested=True)
            x = dh.dof_vertices(flag=1)
            fv = f(x[:,0],x[:,1])
                 
            fn = Function(fv, mesh=mesh, element=element, flag=1)
            
            # Plot 
            i,j = count//4, count%4
            ax[i,j] = plot.contour(ax[i,j],fig,fn,mesh,element)
            count += 1
        plt.show()
        
        
        n_flags = 3
        n_element_types = 3
        for j in range(n_flags):
            for i in range(n_element_types):
                ax[i,j] = plot.mesh(ax[i,j], mesh, cell_numbers=True, node_flag=j)
                etype = etype_list[i]
                element = QuadFE(2,etype)
                dh = DofHandler(mesh, element)
                dh.distribute_dofs(nested=True)
                x = dh.dof_vertices(flag=j)
                ax[i,j].plot(x[:,0],x[:,1],'.b')
        plt.show()
        """
                
        # Explicit function
        
        # Nodal continuous function
        
        # Nodal discontinuous function
    
    
    def test_nodal_on_mesh(self):
        """
        Test whether a function is correctly identified as a nodal and 
        compatible with given (sub)mesh
        """
        # Define mesh
        mesh = QuadMesh(resolution=(1,1))
        element = QuadFE(2,'Q1')
        dofhandler = DofHandler(mesh, element)
        
        # Nodal Function compatible with mesh
        f1 = Function(lambda x,y: x*y, 'nodal', dofhandler=dofhandler) 
        self.assertTrue(f1.mesh_compatible(mesh))
        
        # Explicit function: incompatible with mesh
        f2 = Function(lambda x,y: x*y, 'explicit', dofhandler=dofhandler)
        self.assertFalse(f2.mesh_compatible(mesh))
        
        # Hierarchical mesh
        mesh.cells.record(flag=1)
        mesh.cells.refine(new_label=2)
        dofhandler.distribute_dofs()
        
        # Function defined over coarse mesh
        f3 = Function(lambda x,y:x*y, 'nodal', dofhandler=dofhandler, \
                      subforest_flag=1)
        # Function compatible with coarse mesh, but not with fine one
        self.assertTrue(f3.mesh_compatible(mesh, subforest_flag=1))
        self.assertFalse(f3.mesh_compatible(mesh, subforest_flag=2))
        
        # Interpolate function on finer mesh now compatible
        element_2 = QuadFE(2,'Q2')
        f4 = f3.interpolant(mesh, element_2, subforest_flag=2)
        self.assertTrue(f4.mesh_compatible(mesh, subforest_flag=2))
        
        # Define another mesh -> incompatible
        mesh2 = QuadMesh(resolution=(1,1))
        self.assertFalse(f1.mesh_compatible(mesh2))
        
        
    def test_assign(self):
        """
        Test the assign method
        """
        #
        # New nodal function
        #  
        # Define Mesh and elements
        mesh = QuadMesh(resolution=(2,2))
        element = QuadFE(2,'DQ0')
        
        # Initialize
        vf = np.empty(4,)
        dh = DofHandler(mesh, element)
        dh.distribute_dofs()
        dofs = dh.get_region_dofs()
        x_mpt = convert_to_array(dh.get_dof_vertices(dofs),2)
        f = Function(vf, 'nodal', mesh=mesh, element=element)
        
        #
        # Assign new value to function (vector)
        #
        f_det = np.arange(1,5)
        f.assign(f_det)
        
        self.assertTrue(np.allclose(f.eval(x_mpt),f_det))
        
        #
        # Assign random sample to function
        #
        f_rand = np.random.rand(4,10)
        f.assign(f_rand) 
        
        self.assertTrue(np.allclose(f.eval(x_mpt),f_rand),\
                        'Function value assignment incorrect.')
        
        
        #
        # Now assign in specific position
        #
        n_samples = f.n_samples()
        f.assign(np.arange(1,5), pos=0)
        
        f_eval_0 = f.eval(x_mpt, samples=np.array([0]))
        f_eval_1ton = f.eval(x_mpt, samples=np.arange(1,n_samples))
        self.assertTrue(np.allclose(f_eval_0.ravel(), f_det.T), 
                        'Function value assignment incorrect.')
        self.assertTrue(np.allclose(f_eval_1ton, f_rand[:,1:]), \
                        'Function value assignment incorrect.')
                                    

        f.assign(np.arange(1,5),pos=0)
        self.assertTrue(np.allclose(f.eval(x_mpt)[:,1:],f_rand[:,1:]),\
                        'Function value assignment incorrect.')
        self.assertTrue(np.allclose(f.eval(x_mpt, samples=0).ravel(),\
                                    np.arange(1,5)),\
                        'Function value assignment incorrect.')
        
    
    def test_flag(self):
        mesh = QuadMesh()
        mesh.cells.refine()
        node = mesh.cells.get_child(0)
        node.mark('1')
        for pos in [0,1]:
            node.get_child(pos).mark('1')
        element = QuadFE(2,'Q1')
        fn = lambda x,y: x+y
        f = Function(fn, 'nodal', mesh=mesh, element=element, subforest_flag='1')
        self.assertEqual(f.subforest_flag(),'1', 'Flag incorrect.')
    
    
    def test_input_dim(self):
        # Explicit
        f = Function(lambda x:x**2, 'explicit', dim=1)
        self.assertEqual(f.dim(), 1, \
                         'The function should have one input.')
        
        f = Function(lambda x,y: x+y, 'explicit', dim=2)
        self.assertEqual(f.dim(), 2, \
                         'The function should have two inputs.')
        
        # Nodal 
        #2D 
        mesh = QuadMesh()
        mesh.cells.refine()
        element = QuadFE(2,'Q1')
        vf = np.empty(9,)
        f = Function(vf, 'nodal', mesh=mesh, element=element)
        self.assertEqual(f.dim(),2,\
                         'Function should have two inputs.')
        
    
    def test_n_samples(self): 
        mesh = QuadMesh()
        mesh.cells.refine()
        element = QuadFE(2, 'Q1')
        # Assign 1d vector to function values -> sample size should be None
        vf = np.empty(9,)
        f = Function(vf, 'nodal', mesh=mesh, element=element)
        self.assertEqual(f.n_samples(),None, \
                         'Number of samples should be None.')
        # Assign column vector to function values -> sample size should be 1
        f = Function(vf[:,np.newaxis], 'nodal', mesh=mesh, element=element)
        self.assertEqual(f.n_samples(),1, \
                         'Number of samples should be 1.')
        
        # Assign (9,10) array to function values -> sample size should be 10 
        f = Function(np.empty((9,10)), 'nodal', mesh=mesh, element=element)
        self.assertEqual(f.n_samples(), 10, \
                         'Number of samples should be 10.')
        
        
    def test_interpolate(self):
        #
        # Interpolate a coarse function on a fine mesh
        # 
        mesh = Mesh1D(resolution=(2,))
        mesh.cells.record(0)
        mesh.cells.get_child(0).mark()
        mesh.cells.refine(refinement_flag=True, new_label=1)
        element = QuadFE(1,'Q1')
        dofhandler = DofHandler(mesh, element)
        
        fn = lambda x: x**2
        f = Function(fn, 'nodal', dofhandler=dofhandler, subforest_flag=1)
        x = np.linspace(0,1,5)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = np.linspace(0,1,1000)
        plt.plot(x,f.eval(x),'k--')
        
        element = QuadFE(1, 'DQ0')
        fi = f.interpolant(mesh, element, subforest_flag=0)
        plt.plot(x,fi.eval(x),'b--')
        
        
    def test_derivative(self):
        pass
    
    def test_times(self):
        pass
'''        
