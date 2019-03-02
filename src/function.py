from fem import DofHandler, QuadFE
from mesh import convert_to_array, Vertex
import numbers
import numpy as np



class Map(object):
    """
    Function Class 
    
    TODO: Rename to Function later
    """
    def __init__(self, f=None, parameters=None, data=None, mesh=None, \
                 element=None, dofhandler=None, subforest_flag=None, \
                 subregion_flag=None, dim=None, n_variables=1):
        """
        Constructor:
        
        
        Inputs:
    
            *f: function or vector whose length is consistent with the dofs 
                required by the mesh/element/subforest or dofhandler/subforest.
                f can also be passed as an (n_dofs, n_samples) array.  
            
            *parameters: (list of) dictionary(/ies) of keyword parameters
            
            *data [None]: array of function values at finite element dof 
                vertices. 
                 
            *mesh [None]: Mesh, on which the function will be defined
            
            *element [None]: Element, on whch the function will be defined
            
            *dofhandler [None]: DofHandler, specifying the mesh and element on
                which the function is defined.
            
            *subforest_flag [None]: str/int, marker specifying submesh
              
            *subregion_flag [None]: str/int, marker specifying sub-region
                    
            *dim [None]: int, dimension of underlying domain
            
            *n_variables [1], int, number of input variables
        
            
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
        
        
        # =====================================================================
        # Parse Dimensions
        # =====================================================================
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
            
            # Check format
            assert type(dim) is int, 'Input "dim" should be an integer.'
            assert 0<dim and dim <= 2, 'Input "dim" should be in {1,2}.'
            
            # Check mesh compatibility
            if mesh is not None:
                assert dim==mesh.dim(), \
                'Mesh dimension incompatible with input "dim".'
                
            # Check dofhandler compatibility
            if dofhandler is not None:
                assert dim==dofhandler.mesh.dim(),\
                'Mesh dimension incompatible with dofhandler dim.'
                
        # Store dimension
        self.__dim = dim
    
        # =====================================================================
        # Parse rules and data
        # =====================================================================            
        # 
        # Store functions
        # 
        self.__f = None
        self.set_rules(f, parameters)
        
        #
        # Store nodal function values
        #
        self.set_data(data)
        
           
        #
        # Parse number of variables
        # 
        # Check format
        assert type(n_variables) is int, \
            'Input "n_variables" should be an integer.'
        self.__n_variables = n_variables
    
    
    def n_variables(self):
        """
        Return the number of input variables for the function
        """
        return self.__n_variables
    
        
    def n_samples(self):
        """
        Return the number of samples 
        """
        return self.__n_samples
    
    
    def dim(self):
        """
        Return the dimension of the underlying domain
        """
        return self.__dim
    

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
        Returns the subregion flag
        """
        return self.__subregion_flag
    
    
    def rules(self):
        """
        Returns the rule 
        """
        return self.__f

    
    
    def data(self):
        """
        Return function (usu. interpolation) data array
        """
        return self.__data
    
    
    def set_data(self, data):
        """
        Store function data 
        """
        pass
        
    
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
                
        Note: 
        
            Currently, only univariate functions are supported.
        """
        assert self.n_variables == 1, 'Only functions with 1 input variable '+\
            'can currently be interpolated.'
        
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
        return Nodal(data=fv, dofhandler=dofhandler, \
                      subforest_flag=subforest_flag) 
    
    
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


class Explicit(Map):
    """
    Explicit function
    """
    def __init__(self, f=None, parameters=None, mesh=None, element=None, \
                 dofhandler=None, subforest_flag=None, subregion_flag=None, \
                 dim=None, n_variables=1):
        """
        Constructor
        
        Inputs:
        
            See "Function" class
            
        """
        Map.__init__(self, f=f, parameters=parameters, data=None, mesh=mesh, \
                     element=element, dofhandler=dofhandler, \
                     subforest_flag=subforest_flag, subregion_flag=subregion_flag, \
                     dim=dim, n_variables=n_variables)
        
        #
        # Checks
        # 
        
        # Dimension should be known
        assert self.dim() is not None, \
            'The dimension of the domain should be specified.'
        
        # Number of inputs should be 1 or 2
        assert self.n_inputs() in [1,2], \
            'The number of inputs should be 1 or 2.'
        

    def n_samples(self):
        """
        Determine the number of samples
        """
        pass
    
    
    def set_rules(self, f, parameters, pos=None):
        """
        Set functions
        """
        if type(f) is list: 
            #
            # Function passed as list
            #
            
            # Check no position specified 
            assert pos is None, \
            'Can only add individual functions at specific positions.'
            
            # Check parameter format
            assert type(parameters) is list, \
            'Input "parameters" should be a list.' 
                
            assert all([type(p) is dict for p in parameters]), \
            'Input "parameters" should be a list of dictionaries.'
    
            # Store functions and parameters
            self.__f = f 
            self.__parameters = parameters
        else:
            #
            # Single function
            # 
            assert callable(f), 'Input "f" should be callable.'
            assert type(parameters) is dict, \
            'Input "parameters" should be a dictionary.'
            
            # Store values
            if pos is not None:
                #
                # Insert function at pos in list
                # 
                n_samples = self.n_samples()
                assert n_samples is not None and n_samples>pos, \
                'Input "pos" out of bounds.'
                
                # Store at specific position
                self.__f[pos] = f
                self.__parameters[pos] = parameters
            else:
                # Store single function
                self.__f = f 
                self.__parameters = parameters             
            
            
    def add_rules(self, f, parameters={}):
        """
        Add sample functions 
        
        Inputs: 
        
            f: list of functions
            
            parameters: list of dictionary parameters
        """
            
        self.__f = [self.__f, f]
    
    
    def set_parameters(self, parameters, pos=None):
        """
        Modify function parameters
        
        Inputs:
        
            parameters: dict, dictionary of parameters
            
            pos: int, position of parameters to modify
        """
        # Check parameters are a dictionary
        assert type(parameters) is dict, \
        'Input "paramters" should be a dictionary.'
        
        n_samples = self.n_samples()
        if pos is None:
            assert n_samples is None,\
            'No samples. Input "pos" should be "None".'
            
            # Store parameters
            self.__parameters = parameters
        else:
            assert n_samples > pos, \
            'Input "pos" out of bounds.'
            
            # Store parameters 
            self.__parameters[pos] = parameters
    
    
    def eval(self, x, y=None, subsample=None):
        """
        Evaluate function at point x
        
        Inputs:
        
            x: (list of) tuple(s), Vertex, or number(s) or numpy array of input
                variables.
                
            y: Second variable (if function takes two variables)
            
            subsample: int (k,) array of subsample indices
    
            
        Outputs:
             
            f(x): If function is deterministic (i.e. n_samples is None), then 
                f(x) is an (n_points, ) numpy array. Otherwise, f(x) is an 
                (n_points, n_samples) numpy array of outputs   
        """ 
        n_variables = self.n_variables()
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
            if subsample is None:
                subsample = np.array(range(n_samples))
            else:
                if type(subsample) is int:
                    subsample = np.array([subsample])
                else:
                    assert type(subsample) is np.ndarray, \
                    'vector specifying samples should be an array'
                    
                    assert len(subsample.shape) == 1, \
                    'sample indexing vector should have dimension 1'
                    
                    assert self.n_samples() > subsample.max(), \
                    'Subsample index out of bounds.'
            
            # Subsample size       
            n_subsample = len(subsample)
        else:
            n_subsample = None
            
            
        if n_samples is None:
            #
            # Deterministic function
            # 
            if n_variables==1:
                #
                # 1 Variable (any dimension)
                #   
                f_vec = self.__f(x)
            elif n_variables==2:
                #
                # 2 Variables (any dimension)
                # 
                f_vec = self.__f(x[:,0], x[:,1])
                
        else:
            #
            # Stochastic function
            # 
            f_vec = np.empty((n_points, n_subsample))
            for i in subsample:
                if n_variables==1:
                    #
                    # 1 Variable (any dimension)
                    # 
                    f_vec[:,i] = self.__f[i](x)
                elif dim == 2:
                    #
                    # 2D 
                    # 
                    f_vec[:,i] = self.__f[i](x[:,0],x[:,1])
                        


    
class Nodal(Function):
    """
    Nodal functions
    """    
    def __init__(self, f=None, values=None, mesh=None, element=None, \
                 dofhandler=None, subforest_flag=None, subregion_flag=None, \
                 dim=None, n_variables=1):
        """
        Constructor
        
        Inputs:
        
            See Map constructor
        """
        Map.__init__(self, f=f, values=values, mesh=mesh, element=element,\
                     dofhandler=dofhandler, subforest_flag=subforest_flag, \
                     subregion_flag=subregion_flag, dim=dim, n_variables=n_variables)
        # 
        # Checks
        # 
        
        # Dimension should be given
        assert self.dim() is not None, \
            'Dimension required for nodal functions.'
            
        # Dimension should be 1 or 2
        assert self.dim() in [1,2], \
            'Dimension should be 1 or 2 for nodal functions.'
            
        # Dofhandler should be given
        assert self.dofhandler() is not None, \
            'DofHandler required for nodal functions.'
        
        #
        # Store nodal values
        # 
        self.set_values(values=values, f=f)
        
    
    def set_values(self, values=None, f=None):
        """
        Set the function's nodal values.
        
        Inputs:
        
            values: (n_dofs, n_samples) array of function values
            
            f: (list of) lambda function(s) 
        """
        if values is not None:
            #
            # Function determined by values
            # 
            # Check consistency with dofhandler
            sf = self.subforest_flag()
            n_dofs = self.dofhandler().n_dofs(subforest_flag=sf)
            assert values.shape[0]==n_dofs, \
            'Shape of input "values" inconsistent with dofhandler.'
            
            # Store values
            self.__values = values
        
    
    def eval(self):
        """
        Evaluate Function
        """ 
        pass
    
    
    def mesh_compatible(self, mesh, subforest_flag):
        """
        Determine whether the NodalFn is compatible with the current
        mesh and subforest flag
        """
        if self.mesh()==mesh and self.subforest_flag()==subforest_flag:
            #
            # Same (sub)mesh
            # 
            return True
        else:
            #
            # Different (sub)mesh
            # 
            return False
        
        
    def derivative(self):
        """
        Returns the function's derivative as a NodalFn. 
        """
        pass
    
    
class Constant(Function):
    """
    Constant functions
    """
    def __init__(self):
        """
        Constructor
        """
        pass
    
    
    def values(self):
        """
        Return function's values
        """
        return self.__values 
    
    
    def eval(self):
        """
        Evaluate function
        """
        pass


