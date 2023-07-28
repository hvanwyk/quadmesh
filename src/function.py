from fem import DofHandler, QuadFE, Basis
from fem import parse_derivative_info
from mesh import convert_to_array, Vertex, Mesh, Interval, Cell
import numbers
import numpy as np
import time
from diagnostics import Verbose

# TODO: Default sample size is 1 (not None)

class Map(object):
    """
    Function Class 
    """
    def __init__(self, basis=None, mesh=None, element=None, dofhandler=None, \
                 subforest_flag=None, subregion_flag=None, \
                 dim=None, n_variables=1, symmetric=False, \
                 subsample=None):
        """
        Constructor:
        
        
        Inputs:
                 
            *mesh [None]: Mesh, on which the function will be defined
            
            *element [None]: Element, on whch the function will be defined
            
            *dofhandler [None]: DofHandler, specifying the mesh and element on
                which the function is defined.
            
            *subforest_flag [None]: str/int, marker specifying submesh
              
            *subregion_flag [None]: str/int, marker specifying sub-region
                    
            *dim [None]: int, dimension of underlying domain
            
            *n_variables [1], int, number of input variables
        
            *symmetric: bool, (if n_variables==2), is f(x,y) = f(y,x)? 
            
            
        Note: We allow for the option of specifying multiple realizations 
        
            - If the function is not stochastic, the number of samples is None
        
            - If the function has multiple realizations, its function values 
                are stored in an (n_dofs, n_samples) array. 
        """
        #
        # Store basis function
        # 
        self.__basis = basis
        
        # Subsample
        self.__subsample = subsample
        self.__subregion_flag = subregion_flag
        
        #
        # Store sub-mesh and sub-region flags
        # 
        if basis is None:
            subforest_flag = None
        else:
            subforest_flag = basis.subforest_flag()
        self.__subforest_flag = subforest_flag    
            
        #self.__subforest_flag = subforest_flag
        #self.__subregion_flag = subregion_flag
        
        # =====================================================================
        # Parse DofHandler, Mesh, Element
        # =====================================================================
        #
        # Check mesh
        # 
        if mesh is not None:
            assert isinstance(mesh, Mesh), 'Input mesh should be "Mesh" class.'
            
        
        #
        # Define DofHandler
        # 
        if dofhandler is not None:
            #
            # Dofhandler passed explicitly
            # 
            assert isinstance(dofhandler, DofHandler), \
            'Input "dofhandler" should be of type DofHandler.'
            
        elif mesh is not None and element is not None:
            #
            # DofHandler given in terms of mesh and element
            # 
            dofhandler = DofHandler(mesh, element)
        
        #
        # Store dofhandler
        # 
        self.__dofhandler = dofhandler
        
        #
        # Distribute dofs and dof-vertices
        #
        if self.dofhandler() is not None:
            self.dofhandler().distribute_dofs()
            self.dofhandler().set_dof_vertices()
        else:
            # 
            # Store mesh
            # 
            self.__mesh = mesh
            #
            # Store element
            # 
            self.__element = element
        
        
        # 
        # Parse Dimensions
        # 
        if basis is not None:
            #
            # Get dimension from basis
            # 
            dim = basis.dofhandler().mesh.dim()
        if dim is not None:
            #
            # Check format
            #
            assert type(dim) is int, 'Input "dim" should be an integer.'
            assert 0<dim and dim <= 2, 'Input "dim" should be in {1,2}.'
           
        # Store dimension
        self.__dim = dim
        
           
        #
        # Parse number of variables
        # 
        # Check format
        assert type(n_variables) is int, \
            'Input "n_variables" should be an integer.'
            
        self.__n_variables = n_variables
        
        #
        # Parse symmetry
        #
        # If function is symmetric, it should have 2 variables
        if symmetric:
            assert self.n_variables()<=2, \
            'Symmetric functions should be at most bivariate'
        self.__symmetric = symmetric
        
        
    def n_variables(self):
        """
        Return the number of input variables for the function
        """
        return self.__n_variables
    
        
    def n_samples(self):
        """
        Return the number of samples 
        """
        pass
    
    
    def set_subsample(self, i=None):
        """
        Set subset of samples to be evaluated
        
        Input:
        
            i: int, numpy array of sample indices
            
        Notes: 
        
            We allow subsamples to be determined for deterministic functions.
            In this case, copies of the deterministic function values will be
            returned for each subsample.
            
        TODO: DELETE!
        """
        if i is not None:
            #
            # Non-trivial subsample
            # 
            assert type(i) is np.ndarray, \
            'subsample index set should be an array'
            
            assert len(i.shape)==1, \
            'Subsample index is a 1-dimensional integer array.'
            
            assert all([isinstance(ii, numbers.Integral) for ii in i]), \
            'Subsample should be an integer array.'
        
        self.__subsample = i
    
    
    def subsample(self):
        """
        Returns the index set representing the subsample or else a list of
        integers from 0 to n_samples-1.
        
        TODO: DELETE!
        """
        if self.__subsample is None and self.n_samples() is not None:
            #
            # Stochastic function with no subsample specified
            # 
            return np.arange(self.n_samples())
        else:
            #
            # Return subsample.
            # 
            return self.__subsample
    
    
    def n_subsample(self):
        """
        Returns size of the subsample
        
        TODO: DELETE
        """
        if self.__subsample is None:
            return self.n_samples()
        else:
            return len(self.__subsample)
        
    
    def dim(self):
        """
        Return the dimension of the underlying domain
        """
        return self.__dim
    

    def is_symmetric(self):
        """
        Returns true if the function is symmetric
        """
        return self.__symmetric
    
    
    def basis(self):
        """
        Returns the function's basis function
        """
        return self.__basis
    

    def mesh(self):
        """
        Returns the function's mesh
        """
        if self.__basis is not None:
            return self.__basis.dofhandler().mesh
    
    
    def dofhandler(self):
        """
        Returns the function's dofhandler 
        """
        
        return self.__dofhandler
    
    
    def subforest_flag(self):
        """
        Returns the submesh flag
        """    
        return self.__subforest_flag
    
    
    def subregion_flag(self):
        """
        """
        return self.__subregion_flag
    
    
    def parse_x(self, x):
        """
        Parse input variable x 
        
        Input:
            
            x: double, [list of] (dim, ) tuples, or dim Vertex objects, 
                or an (n_points, -) array. If n_variables>1, can also be a
                tuple of [lists of]....
                
        Output:
        
            xx: n_variables list of (n_points, dim) arrays (one for each 
                variable).  
        """
        n_variables = self.n_variables()
        if n_variables > 1:
            #
            # More than one variable -> should be passed as tuple
            # 
            assert type(x) is tuple, \
            'Input "x" for multivariable functions should be a tuple'
        
            assert len(x)==n_variables, \
            'Input "x" incompatible with number of variables.'
        else:    
            #
            # One variable
            # 
            x = (x,)
            
        # Convert to usable format
        xx = []
        for i in range(n_variables):    
            xx.append(convert_to_array(x[i], dim=self.dim()))
        
        n_points = xx[0].shape[0]
        assert all([x.shape[0]==n_points for x in xx]), \
            'Each variable should have the same number of points.'
        return xx
        
    
    def parse_fx(self, fx):
        """
        Returns output appropriate for singleton points
        
        Input:
        
            fx: double, (n_points, n_samples) array of points
        
        
        Output:
        
            fx: double, (n_points, n_samples) function output
        """
        #
        # Parse fx (n_points, n_samples)   
        # 
        n_samples = self.n_samples()
        if n_samples==1 and self.subsample() is not None:
            #
            # Copy output n_subsample times
            # 
            n_subsample = self.n_subsample()
            
            #
            # Vector input
            # 
            if len(fx.shape)==1:
                #
                # (n_points,) vector
                #
                return np.tile(fx[:,np.newaxis], (1,n_subsample))
            elif len(fx.shape)==2:
                #
                # (n_points,1) vector
                #
                if fx.shape[1]!=1:
                    print(fx.shape)
                assert fx.shape[1]==1, \
                'Number of columns should be 1.'
                return np.tile(fx, (1, n_subsample))
        else:    
            return fx
        
        
    def eval(self):
        """
        Container function for subclasses
        """
        pass
        
            
    def interpolant(self, dofhandler, subforest_flag=None):
        """
        Return the interpolant of the function on a (new) dofhandler 
        
        Inputs:
            
            dofhandler: DofHandler, determines the mesh and elements on which 
                to interpolate. 
            
            subforest_flag [None]: str/int, optional mesh marker
            
            
        Output:
        
            Function, of nodal type that interpolates the given function at
                the dof vertices defined by the pair (mesh, element).
                
        Note: 
        
            Currently, only univariate functions are supported.
        """
        assert self.n_variables() == 1, 'Only functions with 1 input variable '+\
            'can currently be interpolated.'
        
        dofhandler.distribute_dofs(subforest_flag=subforest_flag)
        dofs = dofhandler.get_region_dofs(subforest_flag=subforest_flag)
        x = dofhandler.get_dof_vertices(dofs)
        basis = Basis(dofhandler,subforest_flag=subforest_flag)       
        #
        # Evaluate function at dof vertices
        #
        fv = self.eval(x)
        #
        # Define new function
        #
        return Nodal(data=fv, basis=basis) 
    
'''    
class Function(object):
    """
    Function class for finite element objects.
    
    Attributes:
    
        mesh [None]: Mesh, computational mesh
        
        element [None]: Element, element
        
        dofhandler [None]: DofHandler object, defined via mesh and element 
        
        __dim: int, number of spatial dimensions 
        
        __flag: str/int, marker for submesh on which function is defined
        
        __f: function/vector, used to compute function values 
        
        __n_samples: int, number of samples stored 
        
        __type: str, 'explicit' for explicit function, 'nodal' for nodal 
            finite element function.
    
    Methods:
    
        global_dofs: Returns list of of global dofs associated with fn nodes
        
        eval: Evaluate the function at given set of points
        
        interpolant: Interpolate a Function on a (different) mesh, element
        
        project: Project a Function onto a finite element space defined by
            a mesh, element pair.
        
        derivative: Returns the derivative of given Function as a Function.
        
        times: Returns the product of given function with another
    """
    def __init__(self, f, fn_type, mesh=None, element=None, dofhandler=None,\
                 subforest_flag=None, subregion_flag=None, dim=None):
        """
        Constructor:
        
        
        Inputs:
    
            f: function or vector whose length is consistent with the dofs 
                required by the mesh/element/subforest or dofhandler/subforest.
                f can also be passed as an (n_dofs, n_samples) array.  
                 
            fn_type: str, function type ('explicit', 'nodal', or 'constant')
            
            *mesh [None]: Mesh, on which the function will be defined
            
            *element [None]: Element, on whch the function will be defined
            
            *dofhandler [None]: DofHandler, specifying the mesh and element on
                which the function is defined.
            
            *submesh_flag [None]: str/int, marker specifying submesh
              
            *subregion_flag [None]: str/int, marker specifying sub-region
                    
        Note: We allow for the option of specifying multiple realizations 
            - If the function is not stochastic, the number of samples is None
            - If the function has multiple realizations, its function values 
                are stored in an (n_dofs, n_samples) array. 
        """ 
        #
        # Store sub-mesh and sub-region flags
        # 
        self.__subforest_flag = subforest_flag
        self.__subregion_flag = subregion_flag
        
        
        #
        # Store function type
        #
        assert fn_type in ['explicit', 'nodal','constant'], \
            'Input "fn_type" should be "explicit", "nodal", or "constant".'      
        self.__type = fn_type
        
        
        # =====================================================================
        # Parse DofHandler, Mesh, Element
        # =====================================================================
        #
        # Define DofHandler
        # 
        if dofhandler is not None:
            #
            # Dofhandler passed explicitly
            # 
            assert isinstance(dofhandler, DofHandler), \
            'Input "dofhandler" should be of type DofHandler.'
            
        elif mesh is not None and element is not None:
            #
            # DofHandler given in terms of mesh and element
            # 
            dofhandler = DofHandler(mesh, element)
        
        #
        # Store dofhandler
        # 
        self.__dofhandler = dofhandler
        
        #
        # Distribute dofs and dof-vertices
        #
        if self.dofhandler() is not None:
            self.dofhandler().distribute_dofs()
            self.dofhandler().set_dof_vertices()
        else:
            # 
            # Store mesh
            # 
            self.__mesh = mesh
            #
            # Store element
            # 
            self.__element = element
        
        #
        # Check that function has the minimum number of characterizing pars
        # 
        if self.fn_type() == 'nodal':
            #
            # Nodal functions require DofHandler
            #
            assert self.dofhandler() is not None,\
            'Functions of type "nodal" require a DofHandler.'
            
        elif self.fn_type() == 'explicit':
            #
            # Explicit functions require a mesh
            # 
            assert self.mesh() is not None or dim is not None, \
            'Functions of type "explicit" require a mesh or dim.'
    
        
        #
        # Set dimension
        # 
        if dim is None:
            #
            # Dimension not explicitly passed -> get it from mesh
            # 
            if mesh is not None:
                dim = mesh.dim()
            elif dofhandler is not None:
                dim = dofhandler.mesh.dim()
        else:
            #
            # Dimension given -> check consistency
            # 
            if mesh is not None:
                assert dim==mesh.dim(), \
                'Mesh dimension incompatible with input "dim".'
            if dofhandler is not None:
                assert dim==dofhandler.mesh.dim(),\
                'Mesh dimension incompatible with dofhandler dim.'
                
        #
        # Dimension necessary for explicit or nodal functions 
        # 
        if self.fn_type()=='explicit' or self.fn_type()=='nodal':
            assert dim is not None, \
            'Explicit functions should have dimension specified.'

        self.__dim = dim
                
  
        # =====================================================================
        # Store functions
        # =====================================================================
        self.__f = None
        self.set_rules(f)
        
        
    
    def dofhandler(self):
        """
        Returns the function's dofhandler 
        """
        return self.__dofhandler
        
        
    def mesh(self):
        """
        Returns the function's mesh
        """
        if self.__dofhandler is not None:
            #
            # Mesh attached to the dofhandler
            # 
            return self.dofhandler().mesh
        else:
            #
            # Return mesh
            #
            return self.__mesh 
       
    
    def dim(self):
        """
        Returns the dimension of the function's domain
        """
        return self.__dim
    
    
    def n_samples(self):
        """
        Returns the number of realizations stored by the function, or 
        None if not sampled.
        """
        if self.fn_type()=='explicit':
            #
            # Explicit functions (samples stored as lists)
            # 
            if type(self.__f) is list:
                n_samples = len(self.__f)
            else: 
                n_samples = None
        elif self.fn_type()=='nodal':
            #
            # Nodal functions (array dimensions reflect sample size)
            # 
            if len(self.__f.shape)==1:
                #
                # 1D array
                # 
                n_samples = None
            else:
                #
                # 2D array
                # 
                n_samples = self.__f.shape[1]
        elif self.fn_type()=='constant':
            #
            # Constant functions (sample size=length of vector)
            # 
            if isinstance(self.__f, np.ndarray):
                n_samples = len(self.__f)
            else:
                n_samples = None
        
        return n_samples
    
        
    def fn_type(self):
        """
        Returns function type
        """
        return self.__type

    
    def fn(self):
        """
        Return the function 'kernel'
        """
        return self.__f
    
    
    def set_rules(self, f):
        """
        'Modifies existing function (replacing entries)
        
        Inputs:
        
            f: function, (see self.add for admissible input)
            
            pos: int, position(s) at which to modify function
            
            
        Modifies: 
        
            self.__f
                  
        """
        fn_type = self.fn_type()
        if fn_type == 'explicit':
            # 
            # Explicit function
            # 
            if type(f) is list:
                #
                # List of functions
                #
                assert(all([callable(fi) for fi in f])), \
                'For "explicit" type, input "f" should be callable.'
            else:
                #
                # Single function
                # 
                assert callable(f), 'Input "f" should be callable.'    
        elif fn_type == 'nodal':
            # 
            # Nodal (finite element) function
            # 
            assert self.dofhandler is not None, \
            'If function_type is "nodal", dofhandler '\
            '(or mesh and element required).' 
            
            #
            # Get dof-vertices
            # 
            sf = self.subforest_flag()
            x = self.dofhandler().get_dof_vertices(subforest_flag=sf)
            n_dofs = self.dofhandler().n_dofs(subforest_flag=sf)
            
            if callable(f):
                #
                # Function passed explicitly
                #
                if self.dim() == 1:
                    #
                    # 1D function
                    # 
                    f = f(x[:,0]) 
                elif self.dim() == 2:
                    #
                    # 2D function
                    # 
                    f = f(x[:,0],x[:,1])
                    
            elif type(f) is list:
                #
                # Functions passed as list       
                # 
                if self.dim() == 1:
                    #
                    # 1D functions 
                    # 
                    f = np.array([fi(x[:,0]) for fi in f]) 
                elif self.dim()==2:
                    #
                    # 
                    # 
                    f = np.array([fi(x[:,0],x[:,1]) for fi in f])
            
            elif type(f) is np.ndarray:
                # 
                # Function passed as an array
                #
                assert n_dofs == f.shape[0], 'The number of rows %d ' + \
                'in f does not match the number of dofs %d'%(f.shape[0],n_dofs)
                  
                
        elif fn_type == 'constant':
            # 
            # Constant function
            # 
            if type(f) is np.ndarray:
                #
                # Array of numbers 
                # 
                assert len(f.shape)==1, 'Constant functions are passed '+\
                'as scalars or vectors.'
            elif type(f) is list:
                #
                # Function passed as list of numbers
                # 
                assert all([isinstance(fi, numbers.Real) for fi in f]),\
                'For "constant" functions, input list must contain scalars.'
                
                f = np.array(f)
            else:
                assert isinstance(f, numbers.Real)
                 
        else:
            #
            # Type not recognized
            # 
            raise Exception('Variable function_type should be: '+\
                            ' "explicit", "nodal", or "constant".')            
    
        #
        # Store function
        # 
        self.__f = f
    
    
    def add_rules(self):
        """
        Add a function to the sample.
        """ 
        pass
    
    def assign(self, v, pos=None):
        """
        Assign function values to the function in the specified sample position
        
        Inputs: 
        
            v: double, array 
            
            pos: int, array or constant (indicating position)
        
        TODO: Replace   
        """
        assert self.fn_type() != 'explicit', \
        'Only nodal or constant Functions can be assigned function values'
        
        if pos is not None:
            # Check if position is compatible with sample size
            if isinstance(pos, numbers.Real):
                assert pos < self.n_samples(),\
                'Position "pos" exceeds the sample size.'
            elif type(pos) is np.ndarray:
                assert pos.max() < self.n_samples(),\
                'Maximum position in "pos" exceeds sample size.'
        if self.fn_type() == 'nodal':
            #
            # Nodal function
            #
            if not isinstance(v, numbers.Real):
                assert v.shape[0] == self.fn().shape[0],\
                'Assigned vector incompatible length with function.'
            
            if pos is not None:
                self.__f[:,pos] = v
            else:
                # No position specified: overwrite values
                self.__f = v   
                
                # Update sample size
                if len(v.shape) == 1:
                    # 1d vector -> no samples
                    n_samples = None
                elif len(v.shape) == 2:
                    n_samples = v.shape[1]
                    self.__n_samples = n_samples
                    
        elif self.fn_type() == 'constant':
            #
            # Constant function
            #
            if pos is not None: 
                self.__f[pos] = v
            else:
                self.__f = v
                                          
    
    def mesh_compatible(self, mesh, subforest_flag=None):                          
        """
        Determine whether the function is a nodal function defined on the 
        specified (sub)mesh. If this is the case, the functions can be 
        evaluated using shape functions , which may be computed once during 
        local assembly. 
        """
        if self.fn_type()=='nodal' \
        and self.mesh()==mesh \
        and self.subforest_flag()==subforest_flag:
            return True
        else:
            return False
        
    
    def subforest_flag(self):
        """
        Returns the submesh flag
        """    
        return self.__subforest_flag
    
    
    def subregion_flag(self):
        """
        Returns the subregion flag
        """
        return self.__subregion_flag
    
    
    
    
    
    def eval(self, x=None, cell=None, phi=None, dofs=None, \
             derivative=(0,), samples='all'):
        """
        Evaluate function at an array of points x
        
        The function can be evaluated by:
        
            1. Specifying the points x (in a compatible format).
                - Search through mesh to find cells containing x's
                - Get function values at local dofs  
                - Evaluate shape functions for each cell
                - fv = phi*f_loc
        
            2. Specifying points and cell
                - Check that x in cell
                - Get function values at local dofs  
                - Evaluate shape functions on cell
                - f = phi*f_loc
                
            3. Specifying phi and dofs (x, derivatives, cell not checked)
                - Get function values at local dofs
                - Compute f = phi*f_loc 
            
            
        Inputs:
        
            *x: double, function input in the form of an (n_points, dim) array,
                or a list of vertices or a list of tuples.
            
            *cell: Cell, on which f is evaluated. If included, all points in x
                should be contained in it. 
            
            *phi: shape functions (if function is nodal). 
            
            *dofs: list/np.ndarray listing the degrees of freedom associated
                 with columns of the shape functions. 
                
            *derivative: int, tuple, (order,i,j) where order specifies the order
                of the derivative, and i,j specify the variable wrt which we 
                differentiate, e.g. (2,0,0) computes d^2p/dx^2 = p_xx,
                (2,1,0) computes d^2p/dxdy = p_yx
            
            *samples: int, (r, ) integer array specifying the samples to evaluate
                or use 'all' to denote all samples
        
        Output:
        
            f(x): If function is deterministic (i.e. n_samples is None), then 
                f(x) is an (n_points, ) numpy array. Otherwise, f(x) is an 
                (n_points, n_samples) numpy array of outputs
            
        """
        flag = self.subforest_flag()
        dim = self.dim()
        
        # =====================================================================
        # Parse x
        # =====================================================================
        if x is not None:
            # Deal with singletons 
            if type(x) is tuple \
            or isinstance(x, Vertex) \
            or isinstance(x, numbers.Real):
                is_singleton = True
                x = [x]
            else:
                is_singleton = False
            
            #
            # Convert input to array
            # 
            x = convert_to_array(x)
            if dim is not None:
                #
                # Function defined for specific number of variables (not constant)
                #
                assert x.shape[1]==dim, \
                'Input dimension incompatible with dimension of function.'
            
            n_points = x.shape[0]
                
        # =====================================================================
        # Parse sample size
        # =====================================================================
        n_samples = self.n_samples()
        if n_samples is not None:
            #
            # Only stochastic functions can be sampled
            # 
            if samples is 'all':
                samples = np.array(range(n_samples))
            if samples is not 'all':
                if type(samples) is int:
                    samples = np.array([samples])
                else:
                    assert type(samples) is np.ndarray, \
                    'vector specifying samples should be an array'
                    
                    assert len(samples.shape) == 1, \
                    'sample indexing vector should have dimension 1'
                    
                    assert self.n_samples() > samples.max(), \
                    'Sample paths not stored in function.'
            
            # Subsample size       
            sample_size = len(samples)  
        
        # =====================================================================
        # Parse function type
        # =====================================================================
        if self.fn_type() == 'explicit':
            #
            # Explicit function
            # 
            assert derivative==(0,), \
                'Unable to take derivatives of function directly.'+\
                'Interpolate/Project onto finite element space first.'              
            
            if self.n_samples() is None:
                #
                # Deterministic function
                # 
                if dim == 1:
                    #
                    # 1D
                    # 
                    f_vec = self.__f(x[:,0])
                elif dim == 2:
                    #
                    # 2D
                    # 
                    f_vec = self.__f(x[:,0], x[:,1])
                
            else:
                #
                # Stochastic function
                # 
                f_vec = np.empty((n_points, sample_size))
                for i in samples:
                    if dim == 1:
                        #
                        # 1D
                        # 
                        f_vec[:,i] = self.__f[i](x[:,0])
                    elif dim == 2:
                        #
                        # 2D 
                        # 
                        f_vec[:,i] = self.__f[i](x[:,0],x[:,1])
                        
        
        elif self.fn_type() == 'nodal':
            #
            # Nodal functions 
            # 
            
            # Get dof information
            sf, rf = self.subforest_flag(), self.subregion_flag()
            dh = self.dofhandler()
            mesh = dh.mesh
            gdofs = dh.get_region_dofs(subforest_flag=sf, entity_flag=rf)
            
            if phi is not None:
                # =============================================================
                # Shape function specified
                # =============================================================
                #
                # Checks
                # 
                assert dofs is not None, \
                    'When shape function provided, require input "dofs".'
                
                
                assert all([dof in gdofs for dof in dofs]),\
                    'Nodal function not defined at given dofs.' 
                
                assert len(dofs)==phi.shape[1], \
                    'Number of columns in phi should equal the number of dofs'
                 
                #
                # Evaluate function at local dofs 
                # 
                idx_cell = [gdofs.index(i) for i in dofs]
    
                if self.n_samples() is None:
                    f_loc = self.__f[idx_cell]
                else:
                    f_loc = self.__f[np.ix_(idx_cell, samples)]
                
                #
                # Combine local 
                # 
                f_vec = np.dot(phi, f_loc)
                return f_vec
            else:
                # =============================================================
                # Must compute shape functions (need x)
                # =============================================================
                assert x is not None, \
                    'Need input "x" to evaluate shape functions.'
                
                #
                # Initialize output array
                #
                if n_samples is None:
                    f_vec = np.empty(n_points)
                else:
                    f_vec = np.empty((n_points,sample_size))
                
                #
                # Determine tree cells to traverse
                # 
                if cell is None:
                    #
                    # Cell not specified
                    # 
                    cell_list = mesh.cells.get_leaves(subforest_flag=sf)
                else:
                    #
                    # Cell given
                    # 
                    assert all(cell.contains_points(x)), \
                    'Cell specified, but not all points contained in cell.'
                    
                    if sf is not None:
                        #
                        # Function is defined on a flagged submesh
                        # 
                        if not cell.is_marked(sf):
                            #
                            # Function defined on a coarser mesh
                            #
                            while not cell.is_marked(sf):
                                # 
                                # Get smallest cell in function mesh that contains cell 
                                # 
                                cell = cell.get_parent()
                            cell_list = [cell]
                        elif cell.has_children(flag=sf):
                            #
                            # Function defined on a finer mesh
                            #
                            cell_list = cell.get_leaves(subtree_flag=flag)
                        else:
                            #
                            # Cell is marked with flag and has no flagged children
                            # 
                            cell_list = [cell]
                    else:
                        cell_list = [cell]
                            
                #
                # Evaluate function within each cell
                #
                for cell in cell_list:
                    #
                    # Evaluate function at local dofs 
                    # 
                    idx_cell = [gdofs.index(i) for i in \
                                self.dofhandler().get_cell_dofs(cell)]  
                    if self.n_samples() is None:
                        f_loc = self.__f[idx_cell]
                    else:
                        f_loc = self.__f[np.ix_(idx_cell, samples)]
        
                    #
                    # Evaluate shape function at x-values
                    #    
                    in_cell = cell.contains_points(x)
                    x_loc = x[in_cell,:]
                    phi = self.dofhandler().element.shape(x_loc, cell=cell, \
                                                          derivatives=derivative)
                    #
                    # Update output vector
                    # 
                    if n_samples is None:
                        f_vec[in_cell] = np.dot(phi, f_loc)
                    else:
                        f_vec[in_cell,:] = np.dot(phi, f_loc)
                                                            
        elif self.fn_type() == 'constant':
            
            if n_samples is None:
                f_vec = self.fn()*np.ones((n_points))
            else:
                one = np.ones((n_points, sample_size))
                f_vec = one*self.fn()[samples]
                            
        else:
            raise Exception('Function type must be "explicit", "nodal", '+\
                            ' or "constant".')
                                
        if is_singleton:
            #
            # Singleton input
            # 
            if n_samples is None:
                #
                # Deterministic function
                # 
                return f_vec[0]
            else:
                #
                # Sampled function
                # 
                return f_vec[0,:]
        else:
            #
            # Vector input
            # 
            return f_vec
        
        
        
    def interpolant(self, mesh=None, element=None, dofhandler=None, \
                    subforest_flag=None):
        """
        Return the interpolant of the function on a (new) mesh/element pair 
        
        Inputs:
            
            mesh: Mesh, Physical mesh on which to interpolate
            
            element: QuadFE, element that determines the interpolant
            
            subforest_flag [None]: str/int, optional mesh marker
            
        Output:
        
            Function, of nodal type that interpolates the given function at
                the dof vertices defined by the pair (mesh, element).
        """
        if dofhandler is None:
            if mesh is None:
                mesh = self.dofhandler.mesh
                
            if element is None:
                element = self.dofhandler.element
        #
        # Determine dof vertices
        # 
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs(subforest_flag=subforest_flag)
        dofs = dofhandler.get_region_dofs(subforest_flag=subforest_flag)
        x = dofhandler.get_dof_vertices(dofs)       
        #
        # Evaluate function at dof vertices
        #
        fv = self.eval(x)
        #
        # Define new function
        #
        return Function(fv, fn_type='nodal', dofhandler=dofhandler, \
                        subforest_flag=subforest_flag) 
    
    
    def derivative(self, derivative):
        """
        Returns the derivative of the function f (stored as a Function). 
        
        Input
        
            derivative: int, tuple, (order,i,j) where order specifies the order
                of the derivative, and i,j specify the variable wrt which we 
                differentiate, e.g. (2,0,0) computes d^2f/dx^2 = f_xx,
                (2,1,0) computes d^2f/dxdy = f_yx
                
                
        Output
        
            df^p/dx^qdy^{p-q}: Function, derivative of current function on the
                same mesh/element.
        """
        sf = self.subforest_flag()
        dim = self.dim()  
        mesh, element = self.mesh(), self.dofhandler().element
        
        #
        # Tear element if necessary 
        # 
        etype = element.element_type()
        if etype[0] == 'Q':
            etype = 'D' + etype
        element = QuadFE(dim, etype)
        
        #
        # Define new dofhandler
        # 
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        #
        # Determine dof vertices
        #
        dofs = dofhandler.get_region_dofs(subforest_flag=sf)
        x = dofhandler.get_dof_vertices(dofs)       
        #
        # Evaluate function at dof vertices
        #
        fv = self.eval(x, derivative=derivative)
        #
        # Define new function
        #
        return Function(fv, 'nodal', dofhandler=dofhandler, \
                        subforest_flag=sf) 
    
    
    def times(self, g):
        """
        Form the product of function with another function g. When possible, 
        the product will have the same properties as self. 
        
        
        Inputs: 
        
            g: Function, to be multiplied by
            
            
        Output:
        
            fg: Function, product of self and g.
            
            
        Note: The product's function type is determined by the following table 
        
         g  \  f   : 'explicit' | 'nodal(f)'   | 'constant'
        ---------------------------------------------------  
        'explicit' : 'explicit' | 'nodal(f)'   | 'explicit'
        'nodal(g)' : 'nodal(g)' | 'nodal(f/g)' | 'nodal(g)' 
        'constant' : 'explicit' | 'nodal(f)'   | 'constant'
        
        
        TODO: This is messy, get rid of it
         
        In the case of 'nodal(f/g)', we determine the finite element space 
        as follows:
            - If element(f) = (D)Qi and element(g) = (D)Qj, 
                then element(fg)=(D)Q_{max(i,j)}
            - If element(f) = DQi and element(g) = Qj (or vice versa), 
                then element(fg) = DQi. 
        """
        assert isinstance(g, Function)
        dim = self.input_dim()
        assert dim == g.input_dim() or dim is None or g.input_dim() is None,\
            'Function domains have incompatible dimensions'
            
        # Determine product's function type
        ftype = self.fn_type()
        gtype = g.fn_type() 
        
        if ftype == 'explicit':
            if gtype == 'explicit':
                #
                # fg is explicit
                #
                if dim == 1:
                    fg_fn = lambda x: self.fn()(x)*g.fn()(x)
                elif dim == 2:
                    fg_fn = lambda x,y: self.fn()(x,y)*g.fn()(x,y)
                fg = Function(fg_fn, 'explicit')
            elif gtype == 'nodal':
                #
                # fg nodal
                #
                x = g.dofhandler.get_dof_vertices()
                fg_fn = self.eval(x)*g.fn()
                fg = Function(fg_fn, 'nodal', dofhandler=g.dofhandler)
            elif gtype == 'constant':
                #
                # fg explicit
                #
                if dim == 1:
                    fg_fn = lambda x: g.fn()*self.eval(x)
                elif dim == 2:
                    fg_fn = lambda x,y: g.fn()*self.eval(x,y)
                    
        elif ftype == 'nodal':
            if gtype == 'explicit':
                #
                # fg nodal
                #
                x = self.dofhandler.dof_vertices()
                fg_fn = self.fn()*g.eval(x)
                fg = Function(fg_fn, 'nodal', dofhandler=self.dofhandler)
            elif gtype == 'nodal':
                #
                # fg nodal 
                #
                pass 
            elif gtype == 'constant':
                #
                # fg nodal
                # 
                fg_fn = g.fn()*self.fn()
                fg = Function(fg_fn, 'nodal', dofhandler=self.dofhandler)
        elif ftype == 'constant':
            if gtype == 'explicit':
                #
                # fg explicit
                #
                if g.input_dim() == 1:
                    fg_fn = lambda x: self.fn()*g.fn()(x)
                elif g.input_dim() == 2:
                    fg_fn = lambda x, y: self.fn()*g.fn()(x,y)
                fg = Function(fg_fn, 'explicit')
            elif gtype == 'nodal':
                #
                # fg nodal
                #
                fg_fn = self.fn()*g.fn()
                fg = Function(fg_fn, 'nodal', dofhandler=g.dofhandler)
            elif gtype == 'constant':
                #
                # fg constant
                #
                fg_fn = self.fn()*g.fn()
                fg = Function(fg_fn, 'constant')
        return fg
'''

class Explicit(Map):
    """
    Explicit function
    """
    def __init__(self, f, parameters={}, mesh=None, element=None, \
                 dofhandler=None, subforest_flag=None, subregion_flag=None, \
                 dim=None, n_variables=1, subsample=None, symmetric=False):
        """
        Constructor
        
        Inputs:
        
            *f: (list of) function(s) of the form 
                fi = f(x1,..,xn_vars, **parameters) for i=1,...,n_samples 
                Each variable xi will be an (n_points, dim) array.
            
            *parameters: (list of) dictionary(/ies) of keyword parameters
            
            For other input parameters, see Map class
            
        TODO: subregion flag doesn't do anything (should play a role in eval).
        """
        Map.__init__(self, mesh=mesh, element=element, dofhandler=dofhandler, \
                     subforest_flag=subforest_flag, subregion_flag=subregion_flag, \
                     dim=dim, n_variables=n_variables, subsample=subsample,\
                     symmetric=symmetric)
        
        # Define rules
        self.set_rules(f, parameters)
        
        #
        # Checks
        # 
        
        # Dimension should be known
        assert self.dim() is not None, \
            'The dimension of the domain should be specified.'
        
        # Number of inputs should be 1 or 2
        assert self.n_variables() in [1,2], \
            'The number of inputs should be 1 or 2.'
       
       
    def n_samples(self):
        """
        Determine the number of samples
        """
        return len(self.__f)
    
    
    def parameters(self):
        """
        Return function parameters 
        """
        return self.__parameters
    
    
    def set_parameters(self, parameters, pos=None):
        """
        Modify function's parameters 
        
        
        """
        assert type(parameters) is dict, 'Input parameters must be a dict.'
        
        if pos is None:
            #
            # No position specified, modify all parameters
            # 
            self.__parameters = [parameters for dummy in self.parameters()]
        else:
            #
            # Position specified
            # 
            assert self.n_samples()>pos, 'Input "pos" out of bounds.'
            self.__parameters[pos] = parameters
    
    
    def rules(self):
        """
        Return functions
        """
        return self.__f
    
    
    def set_rules(self, f, parameters={}, pos=None):
        """
        Set functions
        
        Inputs:
        
            f: (list of) functions, fi = f(x1,...,xnvars, **parameters)
            
            parameters: (list of) dictionaries of keyword arguments
            
            *pos [None]: int, position at which to set the function 
        """            
        #
        # Check whether to insert a single function/parameters pair
        # 
        if pos is not None:
            assert callable(f), 'Input "f" should be callable.'
            assert type(parameters) is dict, \
            'Input "parameters" should be a dict.'
            assert self.n_sample()>pos, \
            'Input "pos" incompatible with sample size.'
            self.__f[pos] = f
            self.__parameters[pos] = parameters
        else:
            #
            # Check function inputs
            # 
            if type(f) is list:
                assert all([callable(fi) for fi in f]), \
                'Input "f" should be a (list of) functions.'
                
                # Check no position specified 
                assert pos is None, \
                'Can only add individual functions at specific positions.'
            else:
                assert callable(f), 'Input "f" should be callable.'
                f = [f]
                
            # 
            # Check parameters input
            # 
            if type(parameters) is list:
                # Check that all parameters are dictionaries
                assert all([type(p) is dict for p in parameters]), \
                'Input "parameters" should be a list of dictionaries.'
            else:
                # Check that parameters are a dictionary
                assert type(parameters) is dict, \
                'Input "parameters" should be a dictionary.' 
                parameters = [parameters]
                
            #
            # Parse f - parameter compatibility   
            # 
            if len(f)>1 and len(parameters)>1:
                #
                # More than one function, more than one parameter set
                #
                assert len(f)==len(parameters), \
                'Inputs "f" and "parameters" should have the same length'
            elif len(f)>1:
                #
                # More than one function, single set of parameters
                #
                
                # Extend parameters
                p0 = parameters[0]
                parameters = [p0 for dummy in f]
                
            elif len(parameters)>1:
                #
                # One function, more than one set of parameters
                # 
                
                # Extend f
                f0 = f[0]
                f = [f0 for dummy in parameters]
            # 
            # Store functions
            #
            self.__f = f
            self.__parameters = parameters
                            
            
    def add_rule(self, f, parameters={}):
        """
        Add sample functions 
        
        Inputs: 
        
            f: function
            
            parameters: dictionary parameter
            
        Note: 
        
            Can only add one rule at a time 
        """ 
        assert callable(f), 'Input "f" should be callable.'
        assert type(parameters) is dict, \
            'Input "parameters" should be a dictionary.'
        #
        # Append to list
        # 
        self.__f.append(f)
        self.__parameters.append(parameters)
    
    
    def eval(self, x):
        """
        Evaluate function at point x
        
        Inputs:
        
            x: (list of) tuple(s), Vertex, or number(s) or numpy array of input
                variables.
                            
            subsample: int (k,) array of subsample indices
    
            
        Outputs:
             
            f(x): If function is deterministic (i.e. n_samples is None), then 
                f(x) is an (n_points, 1) numpy array. Otherwise, f(x) is an 
                (n_points, n_samples) numpy array of outputs   
        """ 
        x = self.parse_x(x)
        n_points = x[0].shape[0]
        n_samples = self.n_samples()
        
        # =====================================================================
        # Evaluate function(s)
        # =====================================================================
        if n_samples>1:
            #
            # Multiple samples 
            # 
            fx = np.empty((n_points, n_samples))
            for i in range(self.n_samples()):
                fi, pi = self.__f[i], self.__parameters[i]   
                fx[:,i] = fi(*x, **pi).ravel()
        else:
            #
            # Sample size is 1
            # 
            fx = np.empty((n_points, 1))
            f, p = self.__f[0], self.__parameters[0]
            fx[:,0] = f(*x, **p).ravel()
            
        # Returns function value   
        return self.parse_fx(fx)
            
            
class Nodal(Map):
    """
    Nodal functions
    """    
    def __init__(self, f=None, parameters={}, data=None, basis=None, \
                 mesh=None, element=None, dofhandler=None, subforest_flag=None, \
                 subregion_flag=None, dim=None, n_variables=1, \
                 subsample=None, symmetric=False):
        """
        Constructor
        
        Inputs:
          
            *f: (list of) function(s) of the form 
                fi = f(x1,..,xn_vars, **parameters) for i=1,...,n_samples 
                Each variable xi will be an (n_points, dim) array.
            
            *parameters: (list of) dictionary(/ies) of keyword parameters
                        
            *data [None]: array of function values at finite element dof 
                vertices. Size is consistent with the dofs 
                required by the mesh/element/subforest or dofhandler/subforest.
            
            For other inputs, see Map constructor
        """
        Map.__init__(self, mesh=mesh, element=element, dofhandler=dofhandler, basis=basis,\
                     subforest_flag=subforest_flag, subregion_flag=subregion_flag, \
                     dim=dim, n_variables=n_variables, subsample=subsample,\
                     symmetric=symmetric)
        # 
        # Checks
        # 
        
        # Dimension should be given
        assert self.dim() is not None, \
            'Dimension required for nodal functions.'
            
        # Dimension should be 1 or 2
        assert self.dim() in [1,2], \
            'Dimension should be 1 or 2 for nodal functions.'
            
        # Basis should be given
        assert self.basis() is not None, \
            'Basis required for nodal functions.'
        
        #
        # Store nodal values
        # 
        self.set_data(data=data, f=f, parameters=parameters)
        
        
    def n_samples(self):
        """
        Returns the number of samples 
        """
        return self.__data.shape[-1]
    
    
    def data(self):
        """
        Returns the Nodal function's data array
        """
        return self.__data
        
    
    def set_data(self, data=None, f=None, parameters={}):
        """
        Set the function's nodal values.
        
        Inputs:
        
            values: (n_dofs, n_samples) array of function values
            
            f: (list of) lambda function(s) 
            
        NOTE: Can currently only define univariate and bivariate functions 
            using list of lambda functions.
        """
        if data is not None:
            #
            # Function determined by values
            # 
            # Check consistency with dofhandler
            n_dofs = self.basis().n_dofs()
            assert data.shape[0]==n_dofs, \
            'Shape of input "values" inconsistent with dofhandler.'
            if len(data.shape)<2:
                # Data passed as a 1D array, convert to matrix 
                data = data[:,None]
                    
        elif f is not None:
            #
            # Function (or list of functions) given
            # 
            
            # Define an explicit function with the properties 
            fn = Explicit(f, parameters=parameters, dim=self.dim(), 
                          n_variables=self.n_variables())
            
            #
            # Get Dof-vertices
            # 
            dofs = self.basis().dofs()
            n_dofs = len(dofs)
            x = self.basis().dofhandler().get_dof_vertices(dofs) 
            
            n_variables = self.n_variables()
            if n_variables==1:
                #
                # Univariate function
                # 
                data = fn.eval(x)
            elif n_variables==2:
                #
                # Bivariate function
                # 
                cols, rows = np.mgrid[0:n_dofs,0:n_dofs]
                x1,x2 = x[rows.ravel(),:], x[cols.ravel(),:]
                
                n_samples = fn.n_samples()                
                #
                # Sampled function
                # 
                data = fn.eval((x1,x2)).reshape((n_dofs,n_dofs,n_samples))
        
        #
        # Store data
        #     
        self.__data = data    
        
        
        if self.data() is not None:
            # Check that dimensions are consistent
            assert self.n_variables()+1==len(data.shape), \
            'Sampled function data dimension incorrect' 
        
        
    
    
    def modify_data(self, data, i_sample=0, dofs=None):
        """
        Modify nodal data
        
        Inputs:
        
            data: double, (n_dofs, 1) array
            
            i_sample: int, sample index
            
            dofs: int, list of indices 
        """
        if len(data.shape)==2:
            assert data.shape[1]==1, \
            'Data should be of size (n_dofs,) or (n_dofs,1)'
            
            # Turn into 1d array
            data = data.ravel()
            
        idx = self.basis().d2i(dofs)
        self.__data[idx,i_sample] = data


    def add_samples(self, data):
        """
        Add new sample paths
        
        Inputs:
        
            data: (n_dofs, n_samples), array 
            
            i_sample: int, sample index
            
            dofs: int, list of dof indices
            append
        """
        if self.data() is None:
            #
            # Store new data           
            # 
            self.set_data(data)
        else:
            #
            # Add to old data
            #
            n_dofs = data.shape[0]
            sf = self.subforest_flag()
            assert n_dofs==self.basis().dofhandler().n_dofs(subforest_flag=sf),\
                'Data size is not consistent'
            
            # Convert 1D data array to to 2D
            if len(data.shape)==1:
                data = data[:,None]
            
            new_data = np.append(self.data(),data,axis=1)
            self.set_data(new_data)
            
    
    def eval(self, x=None, cell=None, phi=None, dofs=None, \
             derivative=None):
        """
        Evaluate function at an array of points x
        
        The function can be evaluated by:
        
            1. Specifying the points x in a compatible format 
                (see "convert_to_array").
                - Search through mesh to find cells containing x's
                - Get function values at local dofs  
                - Evaluate shape functions for each cell
                - fv = phi*f_loc
        
            2. Specifying points and cell
                - Check that x in cell
                - Get function values at local dofs  
                - Evaluate shape functions on cell
                - f = phi*f_loc
                
            3. Specifying phi and dofs (x, derivatives, cell not checked)
                - Get function values at local dofs
                - Compute f = phi*f_loc 
            
            
        Inputs:
        
            NOTE: If n_variables=2, use tuples/list for x, cell, phi, dofs, and 
                derivative
            
            *x: double, function input in the form of an (n_points, dim) array,
                or a list of vertices or a list of tuples.
            
            *cell: Cell, on which f is evaluated. If included, all points in x
                should be contained in it. 
            
            *phi: shape functions (if function is nodal). 
            
            *dofs: list/np.ndarray listing the degrees of freedom associated
                 with columns of the shape functions. 
                
            *derivative: int, tuple, (order,i,j) where order specifies the order
                of the derivative, and i,j specify the variable wrt which we 
                differentiate, e.g. (2,0,0) computes d^2p/dx^2 = p_xx,
                (2,1,0) computes d^2p/dxdy = p_yx
        
            *is_singleton: False, specifies whether the input was a singleton.
            
        
        Output:
        
            If the function is deterministic, return an (n_points, ) vector of
                function values
                
            If the function is stochastic, return an (n_points, n_samples)
                array.
                    
        """
        #
        # Initialization
        # 
        n_samples = self.n_samples()
        n_variables = self.n_variables()
        sf, rf = self.subforest_flag(), self.subregion_flag()
        
        #
        # Get dof information
        #
        dh = self.basis().dofhandler()
        mesh = dh.mesh
        element = dh.element
        
        # =====================================================================
        # Shape functions provided
        # =====================================================================
        if phi is not None:
            #
            # Parse shape functions
            # 
            if type(phi) is np.ndarray:
                phi = [phi]
            phi = list(phi)
                
            assert len(phi)==n_variables, \
            'Number of shape functions incompatible with number of variables.'
            
            #
            # Parse dofs
            #
            if dofs is None:
                #
                # Get local dofs if not specified
                #
                dofs = [b.dofs(cell) for b in phi] 
            
            
            if type(dofs) is tuple:
                # Convert tuple to list
                dofs = list(dofs)
            
            assert type(dofs) is list, \
            'Dofs should be passed as list'
            
            if all([type(dof) is np.int for dof in dofs]):
                #
                # list of numbers -> 1 variable
                # 
                dofs = [dofs]
                
            assert len(dofs)==n_variables, \
            'Number of dof-lists incompatible with nunmber of variables.'
            
            #
            # Get local nodes 
            # 
            i_f = []
            for i in range(n_variables):
                #
                # Convert dofs to array indices
                # 
                i_f.append(self.basis().d2i(dofs[i]))
            
            
            #
            # Add sub-sample information   
            #                
            if n_samples>1:
                i_f.append(self.subsample())
            #
            # Get local array (n1,...,nk,n_samples)  
            #  
            f_loc = self.data()[np.ix_(*i_f)]
            
            #
            # Convert shape functions into tensors
            #
            P = 1
            for i in range(n_variables):
                #
                # For each variable
                # 
                for dummy in range(n_variables-1):
                    #
                    # For each of the other variables, add a trivial dimension
                    # 
                    phi[i] = phi[i][:,:,None]
                #
                # Move dof axis to the right position
                # 
                phi[i] = np.swapaxes(phi[i], 1+i, 1)

                #
                # Multiply tensors, using python's broadcasting
                # 
                P = P*phi[i]
        
            #
            # Compute f
            # 
            
            # Determine axes over which to sum
            p_axes = [i+1 for i in range(n_variables)]
            f_axes = [i for i in range(n_variables)]
            
            # Contract the tensor
            fx = np.tensordot(P, f_loc, axes=(p_axes, f_axes))
            
            #
            # Parse fx (n_points, n_samples)   
            # 
            return self.parse_fx(fx)   
        else:        
            # =====================================================================
            # First Evaluate Shape functions
            # =====================================================================
            #
            # Parse derivative
            #  
            # Convert into lists
            if derivative is None:
                derivative = [(0,)]*n_variables
            elif type(derivative) is tuple:
                derivative = [derivative]
            elif type(derivative) is str:
                derivative = [derivative]
            
            # Ensure that the derivative is a tuple
            derivative = [parse_derivative_info(dfdx) for dfdx in derivative]
            
            # Compatibility with n_varibles
            if len(derivative)==1:
                derivative = derivative*n_variables
            else:
                assert len(derivative)==n_variables, \
                'Input "derivative" length incompatible with number of variables.'
                                 
            #
            # Parse x input
            # 
            if x is not None:
                xx = self.parse_x(x)
                n_points = xx[0].shape[0]
            else:
                assert phi is not None, \
                'If input "x" is None, input "phi" should be given.'
             
            #
            # Bin points
            #
            if cell is None:
                #
                # Cell not specified, bin from mesh
                # 
                bins = []
                for i in range(n_variables):
                    bins.append(mesh.bin_points(xx[i], subforest_flag=sf))
            else:
                #
                # Cell(s) given - one for each variable
                # 
                if type(cell) is tuple:
                    # Convert tuple to list
                    cells = list(cell)
                else:
                    assert isinstance(cell, Cell) or isinstance(cell, Interval),\
                    'Input "cell" should be a Cell object.'
                    cells = [cell]  # Convert to list
                
                if len(cells)==1:
                    # Same cell for each variable, enlarge list
                    cells = cells*n_variables
                else:
                    assert len(cells)==n_variables, \
                    'Number of cells incompatible with number of variables'
                    
                #
                # Bin from cell
                # 
                bins = []
                for i, cell in zip(range(n_variables),cells):
                    bins.append(cell.bin_points(xx[i], subforest_flag=sf))
           
            
            #
            # Evaluate local shape functions and get dofs
            # 
            phis = []   # phi
            dofs = []  # dofs 
            pidx = []  # point index
            udofs = []  # unique dofs
            for i in range(n_variables):
                #
                # For each variable
                # 
                phis.append([])
                dofs.append([])
                pidx.append([])
                udofs.append(set())
                for cell, i_points in bins[i]:
                    #
                    # For each cell
                    #
                    
                    # Map physical point to reference element
                    y = xx[i][i_points]
                    x_ref, mg = cell.reference_map(y, mapsto='reference',
                                                   jac_p2r=True, hess_p2r=True)
                    
                    # Record point indices
                    pidx[i].append(i_points)
                    
                    # Locate the dofs
                    dofi = dh.get_cell_dofs(cell)
                    udofs[i] = udofs[i].union(set(dofi))
                    dofs[i].append(dofi)
                    
                    # Compute the shape functions 
                    
                    phii = element.shape(x_ref, cell=cell, 
                                         derivatives=derivative[i],
                                         jac_p2r=mg['jac_p2r'],
                                         hess_p2r=mg['hess_p2r'])
                    phis[i].append(phii)
                    
                udofs[i] = list(udofs[i])
            
            #
            # Get unique dofs
            # 
            Phi = []
            for i in range(n_variables):
                n_basis = len(udofs[i])
                Phi.append(np.zeros((n_points, n_basis)))
                for phi, i_pt, dof in zip(phis[i], pidx[i], dofs[i]):
                    i_col = [udofs[i].index(d) for d in dof]
                    Phi[i][np.ix_(i_pt, i_col)] = phi
            
                assert Phi[i].shape[1]==n_basis
            #
            # Compute f(x) using the shape functions and dofs 
            #  
            fx = self.eval(phi=Phi, dofs=udofs)
            return fx
        """
               
        if phi is not None:
            # =============================================================
            # Shape function specified
            # =============================================================
            assert dofs is not None, \
            'When shape function provided, require input "dofs".'
            if n_variables==1:
                #
                # Single input variable
                # 
                
                #
                # Checks
                #             
                assert all([dof in gdofs for dof in dofs]),\
                    'Nodal function not defined at given dofs.' 
                
                assert len(dofs)==phi.shape[1], \
                    'Number of columns in phi should equal the number of dofs'
                 
                #
                # Evaluate function at local dofs 
                # 
                idx_cell = [gdofs.index(i) for i in dofs]
        
                if n_samples is None:
                    #
                    # Deterministic function
                    # 
                    f_loc = self.data()[idx_cell]
                else:
                    #
                    # Stochastic function
                    # 
                    f_loc = self.data()[np.ix_(idx_cell, self.subsample())]
                
                #
                # Combine local 
                # 
                f_vec = np.dot(phi, f_loc)
                return f_vec
            
            elif n_variables==2:
                #
                # Two inputs
                #
                phir, phic = phi
                rdofs, cdofs = dofs
                
                # 
                # Checks
                # 
                assert all([dof in gdofs for dof in rdofs]),\
                    'Nodal function not defined at given dofs'
                    
                assert all([dof in gdofs for dof in cdofs]),\
                    'Nodal function not defined at given dofs'
                    
                assert len(rdofs==phir.shape[1]), \
                    'Number of columns in phi_rows should equal the number of dofs'

                assert len(cdofs==phic.shape[1]), \
                    'Number of columns in phi_cols should equal the number of dofs'

                #
                # Evaluate functions at local dofs 
                # 
                idx_rcell = [gdofs.index(i) for i in rdofs]
                idx_ccell = [gdofs.index(i) for i in cdofs]
                
                if n_samples is None:
                    #
                    # Deterministic function
                    # 
                    f_loc = self.data()[np.ix_(idx_rcell,idx_ccell)]
                    f_vec = np.sum(phir*(phic.T.dot(f_loc)),axis=1)
                else:
                    #
                    # Stochastic function
                    #
                    f_loc = self.data()[np.ix_(idx_rcell,idx_ccell, self.subsample())]
                    Aphic = np.tensordot(phic, f_loc, axes=(1,1))
                    rAc = phir*Aphic.transpose((2,0,1))
                    rAc = rAc.transpose((1,2,0))
                    f_vec = np.sum(rAc, axis=1) 
      
        else:
            # =============================================================
            # Must compute shape functions (need x)
            # =============================================================
            if n_variables==1:
                #
                # Single variable
                # 
                assert x is not None, \
                    'Need input "x" to evaluate shape functions.'
            
                
                
                    
                #
                # Determine tree cells to traverse
                # 
                if cell is None:
                    #
                    # Cell not specified
                    # 
                    bins = mesh.bin_points(x, subforest_flag=sf)
                else:
                    #
                    # Cell given
                    # 
                    bins = cell.bin_points(x, subforest_flag=sf)
                
                
                for cell, i_points in bins:
                    #
                    # Evaluate shape functions
                    # 
                    dofs = dh.get_cell_dofs(cell)
                    x_loc = x[i_points]
                    
                    phi = element.shape(x_loc, cell=cell, derivatives=derivative)
                    
                    #
                    # Compute nodal function at given points
                    # 
                    f_vec[i_points] = self.eval(phi, dofs)
                    
            elif n_variables==2:
                #
                # Bivariate function
                # 
                
                #
                # Parse x
                #  
                    
                if cell is not None:
                    #
                    # Cell tuple given
                    #
                    cell_1, cell_2 = cell
                    
                    # Bin points within each cell
                    bins_1 = cell_1.bin_points(x1, subforest_flag=sf)
                    bins_2 = cell_2.bin_points(x2, subforest_flag=sf)
                
                else:
                    #
                    # No cells, use mesh
                    # 
                    bins_1 = mesh.bin_points(x1, subforest_flag=sf)
                    bins_2 = mesh.bin_points(x2, subforest_flag=sf)
                    
                n_dofs = element.n_dofs()  
                rphi = np.empty((n_points, n_dofs))
                rdofs = np.empty((n_points, n_dofs))
                for c1, i1 in bins_1:
                    #
                    # Evaluate phir 
                    # 
                    rdofs[i1] = dh.get_cell_dofs(c1)
                    rphi[i1] = element.shape(x1[i1], cell=c1, derivatives=der1)
                     
                cphi = np.empty((n_points, n_dofs))
                cdofs = np.empty((n_points, n_dofs))
                for c2, i2 in bins_2:
                    #
                    # Evaluate column shape functions 
                    # 
                    cdofs[i2] = dh.get_cell_dofs(c2)
                    cphi[i2] = element.shape(x2[i2], cell=c2, derivatives=der2)
                
                if n_samples is None:
                    # Deterministic function
                    pass
                else:
                    # Stochastic function
                    pass
                    
                    
                
                    
            #
            # Evaluate function within each cell
            #
            for cell in cell_list:
                #
                # Evaluate function at local dofs 
                # 
                idx_cell = [gdofs.index(i) for i in \
                            self.dofhandler().get_cell_dofs(cell)]  
                if self.n_samples() is None:
                    f_loc = self.__f[idx_cell]
                else:
                    f_loc = self.__f[np.ix_(idx_cell, self.subsamples())]
    
                #
                # Evaluate shape function at x-values
                #    
                in_cell = cell.contains_points(x)
                x_loc = x[in_cell,:]
                phi = self.dofhandler().element.shape(x_loc, cell=cell, \
                                                      derivatives=derivative)
                #
                # Update output vector
                # 
                if n_samples is None:
                    f_vec[in_cell] = np.dot(phi, f_loc)
                else:
                    f_vec[in_cell,:] = np.dot(phi, f_loc)
                """

    def project(self, basis):
        """
        Project the current Nodal function onto a given basis set
        
        Inputs: 
            
            basis: 
        """
        if basis.element_type() != 'DQ0':
            raise Exception('Projection currently only implemented for piecewise constant basis')
        pass    
    
    
    def lift(self):
        """
        Lift the current Nodal function onto a finer basis set
        
        Inputs: 
        
            basis: Basis, 
        """
        pass
        
        
    def differentiate(self, derivative):
        """
        Returns the derivative of the function f (stored as a Nodal Map). 
        
        Input
        
            derivative: int, tuple, (order,i,j) where order specifies the order
                of the derivative, and i,j specify the variable wrt which we 
                differentiate, e.g. (2,0,0) computes d^2f/dx^2 = f_xx,
                (2,1,0) computes d^2f/dxdy = f_yx
                
                
        Output
        
            df^p/dx^qdy^{p-q}: Function, derivative of current function on the
                same mesh/element.
        """
        sf = self.subforest_flag()
        dim = self.dim()  
        basis = self.basis() 
        mesh, element = basis.dofhandler().mesh, basis.dofhandler().element
        
        #
        # Tear element if necessary 
        # 
        etype = element.element_type()
        if etype[0] == 'Q':
            etype = 'D' + etype
        element = QuadFE(dim, etype)
        
        #
        # Define new dofhandler
        # 
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        basis = Basis(dofhandler)
        
        #
        # Determine dof vertices
        #
        dofs = dofhandler.get_region_dofs(subforest_flag=sf)
        x = dofhandler.get_dof_vertices(dofs)       
        #
        # Evaluate function at dof vertices
        #
        fv = self.eval(x, derivative=derivative)
        #
        # Define new function
        #
        return Nodal(data=fv, basis=basis, subforest_flag=sf) 
    
    
class Constant(Map):
    """
    Constant functions
    """
    def __init__(self, data=None, n_variables=1, subsample=None):
        """
        Constructor
        
        Inputs:
          
            *f: (list of) function(s) of the form 
                fi = f(x1,..,xn_vars, **parameters) for i=1,...,n_samples 
                Each variable xi will be an (n_points, dim) array.
            
            *parameters: (list of) dictionary(/ies) of keyword parameters
                        
            *data [None]: array of function values at finite element dof 
                vertices. Size is consistent with the dofs 
                required by the mesh/element/subforest or dofhandler/subforest.
            
            For other inputs, see Map constructor
        """
        Map.__init__(self, dim=None, n_variables=n_variables, subsample=subsample,
                     symmetric=True)
        self.set_data(data)
        
        
    def set_data(self, data):
        """
        Parse and store data for constant function
        
        Inputs:
        
            data: double, 
        """
        is_number = isinstance(data, numbers.Real)
        is_array = type(data) is np.ndarray
        assert is_number or is_array,  \
        'Input "data" should be a number of a one dimensional array'     
        
        
        if isinstance(data, numbers.Real):
            #
            # Data is a single number
            # 
            data = np.array([data])
        elif type(data) is np.ndarray: 
            assert len(data.shape)==1, \
            '"data" array should be one dimensional.'
        self.__data = data
        
        
    def data(self):
        """
        Return function's values
        """
        return self.__data
    
    
    def n_samples(self):
        """
        Returns the sample size 
        """
        return len(self.data())
    
    
    def eval(self, x):
        """
        Evaluate constant function
        
        Input:
        
            x: double, (n_points,n_samples) array of points 
                or tuple
        """
        # Parse input
        x = self.parse_x(x)
        n_points = x[0].shape[0]
        n_sample = self.n_samples()        
        if n_sample==1:
            #
            # Deterministic function, copied 
            #
            fx = np.ones((n_points,n_sample))*self.data()
        else:
            #
            # Stochastic function 
            # 
            fx = np.outer(np.ones(n_points), self.data()[self.subsample()])
                
        return self.parse_fx(fx)
