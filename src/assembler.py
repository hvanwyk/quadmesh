import numpy as np 
import numbers  
from scipy import sparse, linalg 
from mesh import Vertex, Interval, HalfEdge, QuadCell, convert_to_array
from function import Function, Map, Nodal, Explicit, Constant
from fem import DofHandler, parse_derivative_info, Basis
 
 
class GaussRule(object):
    """
    Gaussian Quadrature weights and nodes on reference cell
    """
    def __init__(self, order, element=None, shape=None):
        """
        Constructor 
        
        Inputs: 
                    
            order: int, order of quadrature rule
                1D rule: order in {1,2,3,4,5,6}
                2D rule: order in {1,4,9,16,25,36} for quadrilaterals
                                  {1,3,7,13} for triangles 
            
            element: Element object
            
                OR 
            
            shape: str, 'interval', 'triangle', or 'quadrilateral'
             
        """
        #
        # Determine shape of cells
        # 
        if element is None:
            # Shape specified directly
            assert shape is not None, 'Must specify either element or cell shape.'                    
        else:
            # Element given  
            shape = element.cell_type()
        
        # Check if shape is supported
        assert shape in ['interval','triangle','quadrilateral'], \
            "Use 'interval', 'triangle', or 'quadrilateral'."
            
        # Get dimension
        dim = 1 if shape=='interval' else 2
        
        
        #
        # Tensorize 1D rules if cell is quadrilateral
        # 
        use_tensor_product_rules = \
            ( dim == 1 or shape == 'quadrilateral' ) 
        if use_tensor_product_rules:
            #
            # Determine the order of constituent 1D rules
            # 
            if dim == 1:
                assert order in [1,2,3,4,5,6], 'Gauss rules in 1D: 1,2,3,4,5,6.'
                order_1d = order
            elif dim == 2:
                assert order in [1,4,9,16,25,36], 'Gauss rules over quads in 2D: 1,4,16,25'
                order_1d = int(np.sqrt(order))
                
            r = [0]*order_1d  # initialize as list of zeros
            w = [0]*order_1d
            #
            # One Dimensional Rules
            #         
            if order_1d == 1:
                r[0] = 0.0
                w[0] = 2.0
            elif order_1d == 2:
                # Nodes
                r[0] = -1.0 /np.sqrt(3.0)
                r[1] = -r[0]
                # Weights
                w[0] = 1.0
                w[1] = 1.0
            elif order_1d == 3:
                # Nodes
                r[0] =-np.sqrt(3.0/5.0)
                r[1] = 0.0
                r[2] =-r[0]
                # weights
                w[0] = 5.0/9.0
                w[1] = 8.0/9.0
                w[2] = w[0]
            elif order_1d == 4:
                # Nodes
                r[0] =-np.sqrt((3.0+2.0*np.sqrt(6.0/5.0))/7.0)
                r[1] =-np.sqrt((3.0-2.0*np.sqrt(6.0/5.0))/7.0)
                r[2] =-r[1]
                r[3] =-r[0]
                # Weights
                w[0] = 0.5 - 1.0 / ( 6.0 * np.sqrt(6.0/5.0) )
                w[1] = 0.5 + 1.0 / ( 6.0 * np.sqrt(6.0/5.0) )
                w[2] = w[1]
                w[3] = w[0]
            elif order_1d == 5:
                # Nodes
                r[0] =-np.sqrt(5.0+4.0*np.sqrt(5.0/14.0)) / 3.0
                r[1] =-np.sqrt(5.0-4.0*np.sqrt(5.0/14.0)) / 3.0
                r[2] = 0.0
                r[3] =-r[1]
                r[4] =-r[0]
                # Weights
                w[0] = 161.0/450.0-13.0/(180.0*np.sqrt(5.0/14.0))
                w[1] = 161.0/450.0+13.0/(180.0*np.sqrt(5.0/14.0))
                w[2] = 128.0/225.0
                w[3] = w[1]
                w[4] = w[0]
            elif order_1d == 6:
                # Nodes
                r[0] = -0.2386191861
                r[1] = -0.6612093865
                r[2] = -0.9324695142
                r[3] = - r[0]
                r[4] = - r[1]
                r[5] = - r[2]
                # Weights
                w[0] = .4679139346
                w[1] = .3607615730
                w[2] = .1713244924
                w[3] = w[0]
                w[4] = w[1]
                w[5] = w[2]
            
            #
            # Transform from [-1,1] to [0,1]
            #     
            r = [0.5+0.5*ri for ri in r]
            w = [0.5*wi for wi in w]
            
            if dim == 1:
                self.__nodes = np.array(r)
                self.__weights = np.array(w)
            elif dim == 2:
                #
                # Combine 1d rules into tensor product rules
                #  
                nodes = []
                weights = []
                for i in range(len(r)):
                    for j in range(len(r)):
                        nodes.append((r[i],r[j]))
                        weights.append(w[i]*w[j])
                self.__nodes = np.array(nodes)
                self.__weights = np.array(weights)
                
        elif element.cell_type == 'triangle':
            #
            # Two dimensional rules over triangles
            #
            assert order in [1,3,7,13], 'Gauss rules on triangles in 2D: 1, 3, 7 or 13.'
            if order == 1:
                # 
                # One point rule
                #
                r = [(2.0/3.0,1.0/3.0)]
                w = [0.5]
            elif order == 3:
                # 
                # 3 point rule
                #
                r = [0]*order
                
                r[0] = (2.0/3.0, 1.0/6.0)
                r[1] = (1.0/6.0, 2.0/3.0)
                r[2] = (1.0/6.0, 1.0/6.0)
        
                w = [0]*order
                w[0] = 1.0/6.0
                w[1] = w[0]
                w[2] = w[0]
                               
            elif order == 7:
                # The following points correspond to a 7 point rule,
                # see Dunavant, IJNME, v. 21, pp. 1129-1148, 1995.
                # or Braess, p. 95.
                #
                # Nodes
                # 
                t1 = 1.0/3.0
                t2 = (6.0 + np.sqrt(15.0))/21.0
                t3 = 4.0/7.0 - t2
               
                r    = [0]*order 
                r[0] = (t1,t1)
                r[1] = (t2,t2)
                r[2] = (1.0-2.0*t2, t2)
                r[3] = (t2,1.0-2.0*t2)
                r[4] = (t3,t3)
                r[5] = (1.0-2.0*t3,t3)
                r[6] = (t3,1.0-2.0*t3);
                
                #
                # Weights
                #
                t1 = 9.0/80.0
                t2 = ( 155.0 + np.sqrt(15.0))/2400.0
                t3 = 31.0/240.0 - t2
                 
                w     = [0]*order
                w[0]  = t1
                w[1]  = t2
                w[2]  = t2
                w[3]  = t2
                w[4]  = t3
                w[5]  = t3
                w[6]  = t3
            
            elif order == 13:
                r     = [0]*order
                r1    = 0.0651301029022
                r2    = 0.8697397941956
                r4    = 0.3128654960049
                r5    = 0.6384441885698
                r6    = 0.0486903154253
                r10   = 0.2603459660790
                r11   = 0.4793080678419
                r13   = 0.3333333333333
                r[0]  = (r1,r1)
                r[1]  = (r2,r1)
                r[2]  = (r1,r2)
                r[3]  = (r4,r6)
                r[4]  = (r5,r4)
                r[5]  = (r6,r5) 
                r[6]  = (r5,r6) 
                r[7]  = (r4,r5) 
                r[8]  = (r6,r4) 
                r[9]  = (r10,r10) 
                r[10] = (r11,r10) 
                r[11] = (r10,r11) 
                r[12] = (r13,r13) 
            
                w     = [0]*order
                w1    = 0.0533472356088
                w4    = 0.0771137608903
                w10   = 0.1756152574332
                w13   = -0.1495700444677
                w[0]  = w1
                w[1]  = w1
                w[2]  = w1
                w[3]  = w4
                w[4]  = w4
                w[5]  = w4
                w[6]  = w4
                w[7]  = w4
                w[8]  = w4
                w[9] = w10
                w[10] = w10
                w[11] = w10
                w[12] = w13
                
                w = [0.5*wi for wi in w]
                
            self.__nodes = np.array(r)
            self.__weights = np.array(w)  
        self.__cell_type = shape
        self.__dim = dim
        
        
    def nodes(self):
        """
        Return quadrature nodes 
        """
        return self.__nodes
       
        
    def weights(self):
        """
        Return quadrature weights
        """
        return self.__weights
       
    
    def n_nodes(self):
        """
        Return the size of the rule
        """
        return len(self.__weights)
    
    
    def dim(self):
        """
        Return the dimension of the rule
        """
        return self.__dim
    
    
    def mapped_rule(self, region):
        """
        Return the rule associated with a specific Cell, Interval, or HalfEdge
        """
        #
        # Map quadrature rule to entity (cell/halfedge)
        # 
        if isinstance(region, Interval):
            #
            # Interval
            #
            # Check compatiblity
            assert self.dim()==1, 'Interval requires a 1D rule.'
            
            # Get reference nodes and weights 
            x_ref = self.nodes()
            w_ref = self.weights()
            
            # Map reference quadrature nodes to cell 
            xg, jac = region.reference_map(x_ref, jacobian=True)
            
            # Modify the quadrature weights
            wg = w_ref*np.array(jac)
            
        elif isinstance(region, HalfEdge):
            #
            # Edge
            # 
            # Check compatibility
            assert self.dim()==1, 'Half Edge requires a 1D rule.'
            
            # Get reference quadrature nodes and weights 
            x_ref = self.nodes()
            w_ref = self.weights()
            
            # Map reference nodes to halfedge
            xg, jac = region.reference_map(x_ref, jacobian=True)
            
            # Modify the quadrature weights
            wg = w_ref*np.array(np.linalg.norm(jac[0]))
            
        elif isinstance(region, QuadCell):
            #
            # Quadrilateral
            #
            # Check compatibility
            assert self.dim()==2, 'QuadCell requires 2D rule.'
             
            x_ref = self.nodes()
            w_ref = self.weights()
            
            # Map reference quaddrature nodes to quadcell
            xg, jac = region.reference_map(x_ref, jacobian=True)
            
            # Modify quadrature weights
            wg = w_ref*np.array([np.linalg.det(j) for j in jac])
            
        else:
            raise Exception('Only Intervals, HalfEdges, & QuadCells supported')
        #
        # Return Gauss nodes and weights
        # 
        return xg, wg
    
    
    
class Kernel(object):
    """
    Kernel (combination of Functions) to be used in Forms
    """
    def __init__(self, f, F=None, subsample=None):
        """
        Constructor
        
        Inputs:
        
            f: single Function, or list of Functions
            
            *f_kwargs: dict, (list of) keyword arguments to be passed to the f's 
            
            *F: function, lambda function describing how the f's are combined 
                and modified to form the kernel
               
            *subsample: int, numpy array of subsample indices
        """
        # 
        # Store input function(s)
        #
        if type(f) is not list:
            #
            # Single function
            # 
            assert isinstance(f, Map), 'Input "f" should be a "Map" object.'
            f = [f]
        self.__f = f
        n_functions = len(self.__f)
        
        #
        # Store meta function F
        # 
        # Check that F takes the right number of inputs
        if F is not None:
            assert F.__code__.co_argcount==n_functions
        else:
            # Store metafunction F
            assert n_functions == 1, \
                'If input "F" not specified, only one function allowed.'
            F = lambda f: f
        self.__F = F
        
        #
        # Check that samples are compatible with functions
        #
        if subsample is not None:
            for f in self.__f:
                f.set_subsample(subsample)
                
        #
        # Store samples
        # 
        self.__subsample = subsample
        self.__n_samples = len(subsample)    
    
    def set_subsample(self, subsample):
        """
        Set kernel's subsample
        
        Input:
        
            subsample: int, numpy array specifying subsample indices
        """
        for f in self.__f:
            f.set_subsample(subsample)
        self.__subsample = subsample
    
    
    def subsample(self):
        """
        Returns the subsample of functions used
        """
        return self.__subsample 
    
    
    def eval(self, x=None):
        """
        Evaluate the kernel at the points stored in x 
        
        Inputs:
        
            *x: (n_points, dim) array of points at which to evaluate the kernel
            
            
        Output:
        
            Kernel function evaluated at point x.
        """
        #
        # Evaluate constituent functions 
        # 
        f_vals = []
        for f in self.__f:
            fv = f.eval(x=x)
            f_vals.append(fv)
               
        #
        # Combine functions using metafunction F 
        # 
        return self.__F(*f_vals)

    
class Form(object):
    """
    Constant, Linear, or Bilinear forms (integrals)
    """
    def __init__(self, kernel=None, trial=None, test=None,\
                 dmu='dx', flag=None, dim=None):
        """
        Constructor
        
        Inputs:
        
            *kernel: Kernel, specifying the form's kernel  
            
            *trial: Basis, basis function representing the trial space
            
            *test: Basis, basis function representing the test space  
            
            *dmu: str, area of integration
                'dx' - integrate over a cell
                'ds' - integrate over an edge
                'dv' - integrate over a vertex    
                
            *flag: str/int/tuple cell/half_edge/vertex marker
                    
            *dim: int, dimension of the domain.
        """        
        #
        # Parse test function
        # 
        if test is not None:
            dim = test.element.dim()
            
            assert isinstance(test, Basis), \
            'Input "test" must be of type "Basis".'
        self.test = test

        #
        # Parse trial function
        # 
        if trial is not None:
            # Check that trial is a Basis
            assert isinstance(trial, Basis), \
            'Input "trial" must be of type "Basis".'
            
            # Check that dimensions are compatible
            assert dim==trial.element.dim(), \
            'Test and trial functions should be defined over the same '+\
            ' dimensional domain.'  
        self.trial = trial
        
        #
        # Parse measure
        # 
        assert dmu in ['dx', 'ds', 'dv'], \
        'Input "dmu" should be "dx", "ds", or "dv".'
        
        #
        # Check: ds can only be used in 2D
        # 
        if dmu=='ds' and test is not None:
            assert dim==2, 'Measure "ds" can only be defined over 2D region.'
         
        self.dmu = dmu
        
        #
        # Parse kernel
        # 
        if kernel is not None:
            #
            # Check that kernel is the right type
            # 
            if isinstance(kernel, Map):
                #
                # Kernel entered as Map
                #  
                kernel = Kernel(kernel)
            elif isinstance(kernel, numbers.Real):
                #
                # Kernel entered as real number
                # 
                kernel = Kernel(Constant(kernel))
            else:
                #
                # Otherwise, kernel must be of type Kernel
                # 
                assert isinstance(kernel, Kernel), \
                'Input "kernel" must be of class "Kernel".'
        else:
            #
            # Default Kernel
            # 
            kernel = Kernel(Constant(1))
        self.kernel = kernel
        
        self.flag = flag

        #
        # Determine Form type
        #
        if self.test is None:
            #
            # Constant form
            #  
            form_type = 'constant'
        elif self.trial is None:
            #
            # Linear form
            # 
            form_type = 'linear'
        else:
            #
            # Bilinear form
            # 
            form_type = 'bilinear'
        self.type = form_type

        

    def shape_info(self, compatible_functions=set()):
        """
        Determine all shape functions that must be evaluated (f, trial, and test)
        
        Inputs:
        
            compatible_functions: set, of Functions that are defined on the 
                current mesh cell.
            
        Output: 
        
            info: dictionary, containing information about the form's shape fns
                with the following keys:
                
                    etype: str, finite element type ('Q1'-'Q3', or 'DQ0'-'DQ3') 
                        
                    element: QuadFE, element associated with etype
                    
                    derivatives: set, of all derivatives to be computed for
                        given etype (in tuple form).
                            
        """
        info = {}
        if self.trial is not None:
            #
            # Add derivatives associated with trial function
            # 
            etype = self.trial.element.element_type()
            if etype not in info:
                #
                # Add element type of trial function
                # 
                info[etype] = {'element': self.trial.element,
                               'derivatives': set()}
            D = self.trial.derivative
            info[etype]['derivatives'].add(D)
        if self.test is not None:
            #
            # Add derivatives associated with the test function
            # 
            etype = self.test.element.element_type()
            if etype not in info:
                #
                # Add element type of test function 
                #
                info[etype] = {'element': self.test.element,
                               'derivatives': set()}
            D = self.test.derivative
            info[etype]['derivatives'].add(D)
        if self.kernel is not None:
            #
            # Add derivatives associated with the kernel function
            #     
            for (f, dfdx) in zip(self.kernel.f(), self.kernel.derivatives()):
                #
                # Iterate over constituent functions
                # 
                if f in compatible_functions:
                    # 
                    # function is compatible with mesh
                    # 
                    etype = f.dofhandler().element.element_type()
                    if etype not in info:
                        #
                        # Add element type of kernel function
                        # 
                        info[etype] = {'element': f.dofhandler().element,
                                       'derivatives': set()}
                    info[etype]['derivatives'].add(dfdx)
        return info
        
    
    def integration_regions(self, cell):
        """
        Determine the regions over which the form is integrated, using
        information from dmu and markers
        """
        regions = []
        dmu = self.dmu
        if dmu=='dx':
            #
            # Integration region is a cell
            #
            if self.flag is None or cell.is_marked(self.flag):
                #
                # Valid Cell
                # 
                regions.append(cell)
        elif dmu=='ds':
            #
            # Integration region is a half-edge
            # 
            for half_edge in cell.get_half_edges():
                #
                # Iterate over half edges
                # 
                if self.flag is None or half_edge.is_marked(self.flag):
                    #
                    # Valid HalfEdge
                    # 
                    regions.append(half_edge)
        elif dmu=='dv':
            #
            # Integration region is a vertex
            # 
            for vertex in cell.get_vertices():
                #
                # Iterate over cell vertices
                # 
                if self.flag is None or vertex.is_marked(self.flag):
                    #
                    # Valid vertex
                    # 
                    regions.append(vertex)
        return regions
        
    
    def eval(self, cell, xg, wg, phi=None, dofs=None, compatible_functions=set()):
        """
        Evaluates the local kernel, test, (and trial) functions of a (bi)linear
        form on a given entity.
        
        Inputs:
            
            cell: Cell containing subregions over which Form is defined
            
            xg: dict, Gaussian quadrature points, indexed by regions.
            
            wg: dict, Gaussian quadrature weights, indexed by regions.
            
            phi: shape functions evaluated at the Gauss quadrature points
            
            *dofs: int, global degrees of freedom associated with cell 
                (and elements), used to possibly evaluate Kernel 
                
            *compatible_functions: set, of functions compatible with the 
                underlying mesh and dofhandlers
        
        Outputs:
        
            Constant-, linear-, or bilinear forms and their associated local
            degrees of freedom.
        
        
        TODO: Explain what the output looks like! 
        Note: This method should be run in conjunction with the Assembler class                  
        """
        # Determine regions over which form is defined
        regions = self.integration_regions(cell)
        
        # Number of samples (if any)
        n_samples = self.kernel.n_samples
        
        f_loc = None
        for region in regions: 
            #
            # Compute kernel, weight by quadrature weights    
            #
            kernel = self.kernel
            Ker = kernel.eval(x=xg[region])
            wKer = (wg[region]*Ker.T).T        
    
            if self.type=='constant':
                #
                # Constant form
                # 
                
                # Initialize form if necessary
                if f_loc is None:
                    if n_samples is None:
                        f_loc = 0
                    else:
                        f_loc = np.zeros(n_samples)
                #
                # Update form
                # 
                f_loc += np.sum(wKer, axis=0)
                
            elif self.type=='linear':
                #
                # Linear form
                #
                # Check that phi is given
                assert phi is not None,\
                'Evaluating (bi)linear form. Require shape functions'
                
                # Define test function
                test_der = self.test.derivative
                test_etype = self.test.element.element_type()
                n_dofs_test = self.test.element.n_dofs()
                test = phi[region][test_etype][test_der]
                
                # Initialize forms if necessary
                if f_loc is None:
                    if n_samples is None:
                        f_loc = np.zeros(n_dofs_test)
                    else:
                        f_loc = np.zeros((n_dofs_test,n_samples))
                        
                # Update form
                f_loc += np.dot(test.T, wKer)
                
            elif self.type=='bilinear':
                #
                # Bilinear form
                # 
                
                # Check that phi is given
                assert phi is not None, \
                'Evaluating (bi)linear form. Require shape functions'
          
                # Define the test function
                test_der = self.test.derivative
                test_etype = self.test.element.element_type()
                n_dofs_test = self.test.element.n_dofs()
                test = phi[region][test_etype][test_der]
                
                # Define the trial function
                trial_der = self.trial.derivative
                trial_etype = self.trial.element.element_type()
                n_dofs_trial = self.trial.element.n_dofs()
                trial = phi[region][trial_etype][trial_der]
                
                #
                # Initialize local matrix if necessary
                # 
                if f_loc is None:
                    #
                    # Initialize form
                    # 
                    if n_samples is None:
                        f_loc = np.zeros((n_dofs_test,n_dofs_trial))
                    else:
                        f_loc = np.zeros((n_dofs_test,n_dofs_trial,n_samples))
                
                #
                # Update form
                # 
                if n_samples is None:
                    #
                    # Deterministic kernel
                    # 
                    '''
                    f_loc_det = np.dot(test.T, np.dot(np.diag(wg[region]*Ker),trial))
                    f_loc += f_loc_det.reshape((n_dofs_test*n_dofs_trial,), order='F')
                    ''' 
                    f_loc += np.dot(test.T, np.dot(np.diag(wg[region]*Ker),trial))
                else:
                    #
                    # Sampled kernel
                    # 
                    '''
                    f_loc_smp = []
                    for i in range(n_dofs_trial):
                        f_loc_smp.append(np.dot(test.T, (trial[:,i]*wKer.T).T))
                    f_loc += np.concatenate(f_loc_smp, axis=0)
                    '''
                    for i in range(n_dofs_trial):
                        f_loc[:,i,:] += np.dot(test.T, (trial[:,i]*wKer.T).T)
        
        
        #
        # Initialize zero local matrix if necessary 
        # 
        if f_loc is None:
            if self.type == 'constant':
                #
                # Constant form
                # 
                if n_samples is None:
                    #
                    # Deterministic form
                    #
                    f_loc = 0
                else:
                    #
                    # Sampled form
                    # 
                    f_loc = np.zeros(n_samples)
            elif self.type=='linear':
                #
                # Linear form
                #
                n_dofs_test = self.test.element.n_dofs()
                if n_samples is None:
                    #
                    # Deterministic form
                    # 
                    f_loc = np.zeros(n_dofs_test)
                else:
                    #
                    # Sampled form
                    # 
                    f_loc = np.zeros((n_dofs_test, n_samples))
                
            elif self.type=='bilinear':
                #
                # Bilinear form
                # 
                n_dofs_test = self.test.element.n_dofs()
                n_dofs_trial = self.trial.element.n_dofs()
                if n_samples is None:
                    #
                    # Deterministic form
                    #
                    f_loc = np.zeros((n_dofs_test, n_dofs_trial))
                else:
                    #
                    # Sampled form
                    #
                    f_loc = np.zeros((n_dofs_test, n_dofs_trial, n_samples))   
        #
        # Return f_loc
        # 
        return f_loc
        """                
        for region in regions:
                    n_samples = kernel.n_samples
            if self.test is not None:
                
                #
                # Need test function               
                # 
                drv = parse_derivative_info(self.test.derivative)
                test_etype = self.test.element.element_type()
                test = phi[region][test_etype][drv] 
                n_dofs_test = test.shape[1]
                if self.trial is not None:
                    #
                    # Need trial function
                    # 
                    drv = parse_derivative_info(self.trial.derivative)
                    trial_etype = self.trial.element.element_type()
                    trial = phi[region][trial_etype][drv]
                    n_dofs_trial = trial.shape[1]
                    #
                    #  Bilinear form               
                    #
                    if n_samples is None:
                        #
                        # Deterministic Kernel
                        # 
                        f_loc = np.dot(test.T, np.dot(np.diag(wg[region]*Ker),trial))
                        f_loc.reshape((n_dofs_test*n_dofs_trial,1), order='F') 
                    else:
                        #
                        # Sampled kernel
                        # 
                        f_loc = np.dot(test.T, np.reshape(np.kron(trial, wKer),(n_gauss,-1), order='F'))
                        f_loc.reshape((n_dofs_test*n_dofs_trial, n_samples), order='F')
                    #
                    # Extract local dofs
                    # 
                    rows, cols = np.meshgrid(np.arange(n_dofs_test), 
                                             np.arange(n_dofs_trial), 
                                             indexing='ij')
                    rows = rows.ravel() 
                    cols = cols.ravel()
                    
                    return f_loc, rows, cols
                else:
                    #
                    # Linear Form
                    #
    
                    rows = np.arange(n_dofs_test)
                    
                    return f_loc, rows     
            else:
                #
                # Simple integral
                # 
                f_loc =            
                return f_loc
        """
  
  
class FormII(Form):
    """
    Bilinear form arising from the interpolatory approximation of an integral
    operator.
    
        Cu(x) = I_D k(x,y) u(y) dy
        
        Ku(x)_i =  I_D k(xi,y) u(y) dy, i=1,...,n_dofs
    """        
    def __init__(self, kernel=None, trial=None, test=None, dmu='dx', flag=None):
        """        
        Constructor
        
        Inputs:
        
            *kernel: Kernel, specifying the form's kernel  
            
            *trial: Basis, basis function representing the trial space
            
            *test: Basis, basis function representing the test space  
            
            *dmu: str, area of integration
                'dx' - integrate over a cell
                'ds' - integrate over an edge
                'dv' - integrate over a vertex    
                
            *flag: str/int/tuple cell/half_edge/vertex marker
                    
        """
        #
        # Initialize form
        # 
        Form.__init__(self, kernel=kernel, trial=trial, test=test, 
                      dmu=dmu, flag=flag)
        
        #
        # Checks
        # 
        assert trial is not None and test is not None,\
        'Both trial and test functions should be specified.'
        
        #assert kernel.n_inputs()==2, 'Integral kernel must be bivariate.'
        
    
    def eval(self, cell, x, xg, wg, phi):
        """
        Evaluate the local bilinear form 
        
        I_{Ej} k(xi, y) phij(y)dy phii(x) for all dof-vertices xi 
        
        where Ej is a mesh cell
        
        Inputs:
        
            cell: Cell, containing subregions over which Form is defined
            
            x: (n, dim) array of interpolation points over mesh
            
            xg: Gaussian quadrature points
            
            wg: Gaussian quadrature weights
            
            phi: shape functions evaluated at quadrature points
        """
        n = x.shape[0] 
         
        # Determine the integration regions
        reg = self.integration_regions(cell)
        
        # =====================================================================
        # Specify the test and trial functions 
        # =====================================================================
        # Derivatives of trial functions
        der = self.trial.derivative
        
        # Number of dofs 
        n_dofs = self.trial.element.n_dofs()
         
        # Element types 
        trl_etype = self.trial.element.element_type()
            
        f_loc = None
        for reg in self.integration_regions(cell):
            # Get trial functions evaluated at Gauss nodes
            trial = phi[reg][trl_etype][der]
            
            x_g = xg[reg]
            w_g = wg[reg]
            
            #
            # Initialize local matrix if necessary
            # 
            if f_loc is None:
                #
                # Initialize form
                # 
                f_loc = np.zeros((n,n_dofs))
    
                #
                # Evaluate covariance function at the local Gauss points
                # 
                n_gauss = x_g.shape[0]
                ii,jj = np.meshgrid(np.arange(n),np.arange(n_gauss), indexing='ij')
                if self.dim == 1:
                    x1, x2 = x[ii.ravel()], x_g[jj.ravel()]
                elif self.dim == 2:
                    x1, x2 = x[ii.ravel(),:],x_g[jj.ravel(),:]
        
                C_loc = self.kernel.eval(x1,x2)
                C_loc = C_loc.reshape(n,n_gauss)
        
                #
                # Compute local integral                   
                # 
                # Weight shape functions 
                Wphi = np.diag(w_g).dot(trial)
                
                # Combine
                f_loc += C_loc.dot(Wphi)
        return f_loc
        
    
    
class FormIP(Form):
    """
    Bilinear form arising from the projection based approximation of an integral
    operator.
    
        Cu(x) = I_D k(x,y) u(y) dy
        
        Kij = I_D I_D k(x,y) phij(y)dy phii(x) dx, 
        
    Note: The approximation of Cu(x) is given by 
    
        Cu(x) ~= M^{-1} K u
    """
    def __init__(self, kernel=None, trial=None, test=None, dmu='dx', flag=None):
        """        
        Constructor
        
        Inputs:
        
            *kernel: Kernel, specifying the form's kernel  
            
            *trial: Basis, basis function representing the trial space
            
            *test: Basis, basis function representing the test space  
            
            *dmu: str, area of integration
                'dx' - integrate over a cell
                'ds' - integrate over an edge
                'dv' - integrate over a vertex    
                
            *flag: str/int/tuple cell/half_edge/vertex marker
                    
        """
        #
        # Initialize form
        # 
        Form.__init__(kernel=kernel, trial=trial, test=test, dmu=dmu, flag=flag)
        
        #
        # Checks
        # 
        assert trial is not None and test is not None,\
        'Integral forms have both test and trial functions'
        
        assert kernel.n_inputs()==2, \
        'Integral kernel must be bivariate.'
        
    
    
    def eval(self, cells, xg, wg, phi):
        """
        Evaluates the local bilinear form 
        
         I_{Ei} I_{Ej} k(x,y) phij(y) dy phii(x) dx,
        
        where Ei, Ej are mesh cells
        
        Inputs:
        
            cells: Cells (2,) pair, containing subregions over which Form is defined
            
            xg: dict, (2,) pair of Gaussian quadrature points
            
            wg: dict, (2,) pair of Gaussian quadrature weights
            
            phi: (2,) pair of shape functions evaluated at quadrature points 
        """
        # Cells
        ci, cj = cells
        
        # Determine integration regions
        regi = self.integration_regions(ci)
        regj = self.integration_regions(cj)
        
        # =====================================================================
        # Specify the test and trial functions 
        # =====================================================================
        # Derivatives of test functions
        deri, derj = self.test.derivative, self.trial.derivative
        
        # Element types 
        etypei = self.test.element.element_type()
        etypej = self.trial.element.element_type()
         
        # Degrees of freedom
        n_dofsi = self.test.element.n_dofs()
        n_dofsj = self.trial.element.n_dofs()
        
        # Sample size
        n_samples = self.kernel.n_samples 
        
        f_loc = None
        for regi in self.integration_regions(ci):
            for regj in self.integration_regions(cj):
                # Access test(i) and trial(j) functions
                phii = phi[0][regi][etypei][deri]
                phij = phi[1][regj][etypej][derj]
                
                # Get quadrature nodes
                xi_g = xg[0][regi] 
                xj_g = xg[1][regj]
                
                # Get quadrature weights
                wi_g = wg[0][regi]
                wj_g = wg[1][regj]
            
                #
                # Initialize local matrix if necessary
                # 
                if f_loc is None:
                    #
                    # Initialize form
                    # 
                    if n_samples is None:
                        f_loc = np.zeros((n_dofsi,n_dofsj))
                    else:
                        f_loc = np.zeros((n_dofsi,n_dofsj,n_samples))
                #
                # Evaluate kernel function at the local Gauss points
                # 
                n_gauss = xi_g.shape[0]
                ig = np.arange(n_gauss)
                ii,jj = np.meshgrid(ig,ig,indexing='ij')
                
                if self.dim() == 1:
                    x1, x2 = xi_g[ii.ravel()], xj_g[jj.ravel()]
                elif self.dim() == 2:
                    x1, x2 = xi_g[ii.ravel(),:],xj_g[jj.ravel(),:]
        
                C_loc = self.kernel.eval(x1,x2)
                C_loc = C_loc.reshape(n_gauss,n_gauss)
        
                #
                # Compute local integral                   
                # 
                # Weight shape functions 
                Wphii = np.diag(wi_g).dot(phii)
                Wphij = np.diag(wj_g).dot(phij)
                
                # Combine
                f_loc += np.dot(Wphii.T, C_loc.dot(Wphij))
                
        # Return local form
        return f_loc
     
'''
class IForm(Form):
    """
    Bilinear form for an integral operator 
    
        Cu(x) = I_D k(x,y) u(y) dy
        
    TODO: Replace with FormII and FormIP
    """
    def __init__(self, kernel, trial=None, test=None, dmu='dx', flag=None,
                 form_type='projection'):
        """
        Constructor
        
        Inputs:
        
            *kernel: Kernel, specifying the form's kernel  
            
            *trial: Basis, basis function representing the trial space
            
            *test: Basis, basis function representing the test space  
            
            *dmu: str, area of integration
                'dx' - integrate over a cell
                'ds' - integrate over an edge
                    
            *flag: str/int/tuple cell/half_edge/vertex marker
        
            *approximation_type: str ('projection' or 'interpolation').
        """
        
        self.type = 'bilinear'
        self.flag = flag
        
        #
        # Trial space
        #  
        assert isinstance(trial, Basis),\
        'Input "trial" should be of type "Basis".'
        self.trial = trial 
        
        # Dimension
        self.__dim = self.trial.element.dim()
        
        #
        # Test space
        # 
        assert isinstance(test, Basis),\
        'Input "test" should be of type "Basis".'
        self.test = test
        
        #
        # Check kernel
        # 
        self.kernel = kernel
        
        #
        # Check measure
        # 
        self.dmu = dmu
        
        #
        # Record form type
        #
        assert form_type in ['projection', 'interpolation'], \
        'Input "approximation_type" is either "projection" or "interpolation".'
        self.__approximation_type = form_type
      
    
    def dim(self):
        """
        Returns the dimension of the underlying domain
        """
        return self.__dim  
      
        
    def assembly_type(self):
        """
        Specify whether the operator is approximated via projection or 
        interpolation.
        """
        return self.__approximation_type
    
    
    def eval(self, cells, xg, wg, phi):
        """
        Evaluate the integral form between two cells
        """
        if self.assembly_type()=='projection':
            #
            # Projection mode
            # 
            return self.eval_projection(cells, xg, wg, phi)
        elif self.assembly_type()=='interpolation':
            #
            # Interpolation
            #
            return self.eval_interpolation(cells, xg, wg, phi)
            
        
    def eval_interpolation(self, cell, x, xg, wg, phi):
        """
        Evaluate the local bilinear form 
        
        I_{Ej} k(xi, y) phij(y)dy phii(x) for xi in Ei
        
        where Ei, Ej are mesh cells
        
        Inputs:
        
            cell: Cell, containing subregions over which Form is defined
            
            x: (n, dim) array of interpolation points over mesh
            
            xg: Gaussian quadrature points
            
            wg: Gaussian quadrature weights
            
            phi: shape functions evaluated at quadrature points
        """
        n = x.shape[0] 
         
        # Determine the integration regions
        reg = self.integration_regions(cell)
        
        # =====================================================================
        # Specify the test and trial functions 
        # =====================================================================
        # Derivatives of trial functions
        der = self.trial.derivative
        
        # Number of dofs 
        n_dofs = self.trial.element.n_dofs()
         
        # Element types 
        etype = self.trial.element.element_type()
            
        f_loc = None
        for reg in self.integration_regions(cell):
            # Get trial functions evaluated at Gauss nodes
            trial = phi[reg][etype][der]
            
            x_g = xg[reg]
            w_g = wg[reg]
            
            #
            # Initialize local matrix if necessary
            # 
            if f_loc is None:
                #
                # Initialize form
                # 
                f_loc = np.zeros((n,n_dofs))
    
                #
                # Evaluate covariance function at the local Gauss points
                # 
                n_gauss = x_g.shape[0]
                ii,jj = np.meshgrid(np.arange(n),np.arange(n_gauss), indexing='ij')
                if self.dim() == 1:
                    x1, x2 = x[ii.ravel()], x_g[jj.ravel()]
                elif self.dim() == 2:
                    x1, x2 = x[ii.ravel(),:],x_g[jj.ravel(),:]
        
                C_loc = self.kernel.eval(x1,x2)
                C_loc = C_loc.reshape(n,n_gauss)
        
                #
                # Compute local integral                   
                # 
                # Weight shape functions 
                Wphi = np.diag(w_g).dot(trial)
                
                # Combine
                f_loc += C_loc.dot(Wphi)
        return f_loc
        
    
    def eval_projection(self, cells, xg, wg, phi):
        """
        Evaluates the local bilinear form 
        
         I_{Ei} I_{Ej} k(x,y) phij(y) dy phii(x) dx,
        
        where Ei, Ej are mesh cells
        
        Inputs:
        
            cells: Cells (2,) pair, containing subregions over which Form is defined
            
            xg: dict, (2,) pair of Gaussian quadrature points
            
            wg: dict, (2,) pair of Gaussian quadrature weights
            
            phi: (2,) pair of shape functions evaluated at quadrature points 
        """
        # Cells
        ci, cj = cells
        
        # Determine integration regions
        regi = self.integration_regions(ci)
        regj = self.integration_regions(cj)
        
        # =====================================================================
        # Specify the test and trial functions 
        # =====================================================================
        # Derivatives of test functions
        deri, derj = self.test.derivative, self.trial.derivative
        
        # Element types 
        etypei = self.test.element.element_type()
        etypej = self.trial.element.element_type()
         
        # Degrees of freedom
        n_dofsi = self.test.element.n_dofs()
        n_dofsj = self.trial.element.n_dofs()
        
        # Sample size
        n_samples = self.kernel.n_samples 
        
        f_loc = None
        for regi in self.integration_regions(ci):
            for regj in self.integration_regions(cj):
                # Access test(i) and trial(j) functions
                phii = phi[0][regi][etypei][deri]
                phij = phi[1][regj][etypej][derj]
                
                # Get quadrature nodes
                xi_g = xg[0][regi] 
                xj_g = xg[1][regj]
                
                # Get quadrature weights
                wi_g = wg[0][regi]
                wj_g = wg[1][regj]
            
                #
                # Initialize local matrix if necessary
                # 
                if f_loc is None:
                    #
                    # Initialize form
                    # 
                    if n_samples is None:
                        f_loc = np.zeros((n_dofsi,n_dofsj))
                    else:
                        f_loc = np.zeros((n_dofsi,n_dofsj,n_samples))
                #
                # Evaluate kernel function at the local Gauss points
                # 
                n_gauss = xi_g.shape[0]
                ig = np.arange(n_gauss)
                ii,jj = np.meshgrid(ig,ig,indexing='ij')
                
                if self.dim() == 1:
                    x1, x2 = xi_g[ii.ravel()], xj_g[jj.ravel()]
                elif self.dim() == 2:
                    x1, x2 = xi_g[ii.ravel(),:],xj_g[jj.ravel(),:]
        
                C_loc = self.kernel.eval(x1,x2)
                C_loc = C_loc.reshape(n_gauss,n_gauss)
        
                #
                # Compute local integral                   
                # 
                # Weight shape functions 
                Wphii = np.diag(wi_g).dot(phii)
                Wphij = np.diag(wj_g).dot(phij)
                
                # Combine
                f_loc += np.dot(Wphii.T, C_loc.dot(Wphij))
                
        # Return local form
        return f_loc
'''
    
class Assembler(object):
    """
    Representation of sums of bilinear/linear/constant forms as 
    matrices/vectors/numbers.  
    """
    def __init__(self, problems, mesh, subforest_flag=None, n_gauss=(4,16)):
        """
        Constructor
        
        - Define the quadrature rules that will be used for assembly
        
        - Collect information from all forms to construct the dofhandlers
            necessary for evaluating kernels and shape functions and for 
            storing assembled forms in arrays.
            
        - Initialize AssembledForm's, objects for storing the assembled
            matrices, vectors, or constants. 
        
        
        Inputs:
        
            problems: list of bilinear, linear, or constant Forms
        
            mesh: Mesh, finite element mesh
            
            subforest_flag: submesh marker over which to assemble forms
                        
            n_gauss: int tuple, number of quadrature nodes in 1d and 2d respectively
        
        
        TODO: We make a big production of recording whether there is a single problem
            and or form, but never use it.
        """
        # =====================================================================
        # Store mesh
        # =====================================================================
        self.mesh = mesh
        self.subforest_flag = subforest_flag
        
        # =====================================================================
        # Initialize Gauss Quadrature Rule
        # =====================================================================
        self.n_gauss_2d = n_gauss[1]
        self.n_gauss_1d = n_gauss[0]
        dim = self.mesh.dim()
        if dim==1:
            #
            # 1D rule over intervals
            # 
            self.cell_rule = GaussRule(self.n_gauss_1d,shape='interval')            
        elif dim==2:
            #
            # 2D rule over rectangles 
            # 
            self.edge_rule = GaussRule(self.n_gauss_1d,shape='interval')
            self.cell_rule = GaussRule(self.n_gauss_2d,shape='quadrilateral')
            
        # =====================================================================
        # Parse "problems" input
        # =====================================================================
        single_problem = False
        single_form = False
        problem_error = 'Input "problems" should be (i) a Form, (ii) a list '+\
                        'of Forms, or (iii) a list of a list of Forms.'
        if type(problems) is list:
            #
            # Multiple forms (and/or problems)
            # 
            single_form = False
            if all([isinstance(problem, Form) for problem in problems]):
                #
                # Single problem consisting of multiple forms
                #  
                single_problem = True
                problems = [problems]
            else:
                #
                # Multiple problems
                #
                single_problem = False
                for problem in problems:
                    if type(problem) is not list:
                        #
                        # Found problem not in list form
                        # 
                        assert isinstance(problem, Form), problem_error
                        #
                        # Convert form to problem
                        #
                        problems[problems.index(problem)] = [problem]  
        else:
            #
            # Single form
            #
            assert isinstance(problems, Form), problem_error
            single_problem = True
            single_form = True
            problems = [[problems]]
        
        # Store info    
        self.single_problem = single_problem
        self.single_form = single_form
        self.problems = problems
            
        # =====================================================================
        # Initialize Dofhandlers   
        # =====================================================================
        self.initialize_dofhandlers()
        
        """
        #
        # Compute local-to-global mappings
        # 
        for dofhandler in self.dofhandlers.values():
            dofhandler.set_l2g_map(subforest_flag=self.subforest_flag)
        """
        
        #
        # Initialize dictionaries for storing assembled forms
        #
        self.initialize_assembled_forms()
    
    
    def initialize_assembled_forms(self):
        """
        Initialize list of AssembledForm's encoding the assembled forms associated
        with each problem
        """
        af = []
        for i_problem in range(len(self.problems)):
            problem = self.problems[i_problem]
            af_problem = {}
            for form in problem:
                n_samples = self.n_samples(i_problem, form.type)
                if form.type=='constant':
                    #
                    # Constant type forms
                    # 
                    if 'constant' not in af_problem:
                        #
                        # New assembled form
                        # 
                        af_problem['constant'] = \
                            AssembledForm(form, n_samples=n_samples)
                    else:
                        #
                        # Check compatibility with existing assembled form
                        # 
                        af_problem['constant'].check_compatibility(form)
                        
                elif form.type=='linear':  
                    #
                    # Linear form
                    # 
                    if 'linear' not in af_problem:
                        #
                        # New assembled form
                        # 
                        af_problem['linear'] = \
                            AssembledForm(form, n_samples=n_samples)
                    else:
                        #
                        # Check compatibility with existing assembled form
                        # 
                        af_problem['linear'].check_compatibility(form)
                    
                elif form.type=='bilinear':
                    #
                    #  Bilinear form     
                    # 
                    if 'bilinear' not in af_problem:
                        #
                        # New assembled form
                        #
                        af_problem['bilinear'] = \
                            AssembledForm(form, n_samples=n_samples)
                    else:
                        #
                        # Check compatibility with existing assembled form
                        # 
                        af_problem['bilinear'].check_compatibility(form)
            #
            # Update list of problems
            # 
            af.append(af_problem)
        #
        # Store assembled forms
        # 
        self.af = af 
        
 
    def initialize_dofhandlers(self):
        """
        Identify and initialize dofhandlers necessary to assemble problem
        
        Inputs:
        
            problems: list of Forms, describing the problem to be assembled.
            
            subforest_flag: str/int/tuple, submesh on which problem defined.
        
        
        Modified:
        
            dofhandlers: dictionary, dofhandlers[etype] containing the 
                dofhandler for the type of finite element.
                
            compatible_functions: set, of functions that are compatible with
                the current mesh and elements. These can be readily evaluated
                over each mesh cell through the use of shape functions.
        """
        #
        # Extract all elements
        # 
        elements = set()
        dofhandlers = {}
        compatible_functions = set()
        for problem in self.problems:
            for form in problem:
                
                if form.trial is not None:
                    #
                    # Add elements associated with trial functions
                    #  
                    elements.add(form.trial.element)
                    
                if form.test is not None:
                    #
                    # Add elements associated with test functions
                    # 
                    elements.add(form.test.element)
                
                #
                # Add elements associated with kernels
                #
                if not isinstance(form, IForm):
                    #
                    # For now, mesh compatibility only valid for traditional
                    # kernels. 
                    # 
                    for f in form.kernel.f():
                        if f.mesh_compatible(self.mesh, \
                                             subforest_flag=self.subforest_flag):
                            #
                            # Function is nodal and defined over submesh
                            # 
                            compatible_functions.add(f)
                            element = f.dofhandler().element 
                            elements.add(element)
                            etype = element.element_type()
                            if etype not in dofhandlers:
                                #
                                # Add existing dofhandler to list of dofhandlers
                                # 
                                dofhandlers[etype] = f.dofhandler()
        #
        # Store the set of functions that are compatible with the mesh/subforest
        #  
        self.compatible_functions = compatible_functions        
        
        #             
        # Define new DofHandlers not already used by functions
        # 
        for element in elements:
            etype = element.element_type()
            if etype not in dofhandlers:
                #
                # Require a dofhandler for this element
                #                                 
                dofhandlers[etype] = DofHandler(self.mesh, element)
                dofhandlers[etype].distribute_dofs()
        self.dofhandlers = dofhandlers
            
            
            
    def assemble(self, clear_cell_data=True):
        """
        Assembles constant, linear, and bilinear forms over computational mesh,

        
        Input:
        
            problems: A list of finite element problems. Each problem is a list
                of constant, linear, and bilinear forms. 
                 
               
        Output:
        
            assembled_forms: list of dictionaries (one for each problem), each of 
                which contains:
                
            A: double coo_matrix, system matrix determined by bilinear forms and 
                boundary conditions.
                
            b: double, right hand side vector determined by linear forms and 
                boundary conditions.
        
        
        """                 
        #
        # Assemble forms over mesh cells
        #      
        sf = self.subforest_flag 
        cells = self.mesh.cells.get_leaves(subforest_flag=sf)
        n_cells = len(cells) 
        for i in range(n_cells):
            ci = cells[i]
            #
            # Get global cell dofs for each element type  
            #
            ci_dofs = self.cell_dofs(ci)
            ci_addr = ci.get_node_address()
            
            #
            # Determine what shape functions and Gauss rules to 
            # compute on current cells
            # 
            ci_sinfo = self.shape_info(ci)
            
            # 
            # Compute Gauss nodes and weights on cell
            # 
            xi_g, wi_g = self.gauss_rules(ci_sinfo)
            
            #
            # Compute shape functions on cell
            #  
            phii = self.shape_eval(ci_sinfo, xi_g, ci)
            
            #
            # Assemble local forms and assign to global dofs
            #
            for problem in self.problems:
                #
                # Loop over problems
                # 
                i_problem = self.problems.index(problem)
                for form in problem:
                    #
                    # Evaluate form
                    #
                    if isinstance(form, FormII):
                        # =====================================================
                        # Integral form by interpolation
                        # =====================================================
                        # DOF vertices 
                        dh = form.kernel.dofhandler()
                        x_dofs = dh.get_dof_vertices(subforest_flag=sf)
                        
                        # DOFs
                        dofs_tst = dh.get_region_dofs(subforest_flag=sf)
                        
                        # Get local trial dofs
                        etype_trl = form.trial.element.element_type()
                        dofs_trl = ci_dofs[etype_trl]
                        
                        # Evaluate local 
                        form_loc = form.eval_interpolation(ci, x_dofs, xi_g, wi_g, phii)
                        
                        # Store dofs and values in assembled form
                        af = self.af[i_problem]['bilinear']  
                        af.update(ci_addr, form_loc, 
                                  row_dofs=dofs_tst, col_dofs=dofs_trl)
                        
                    elif isinstance(form, FormIP):
                        # =====================================================
                        # Integral form by projection
                        # =====================================================
                        #
                        # Determine the number of cells in inner loop
                        #
                        if form.kernel.is_symmetric():
                            # Symmetric kernel
                            n_ocells = i
                        else:
                            # Non-symmetric kernel 
                            n_ocells = n_cells
                        
                        for j in range(n_ocells+1):
                            #
                            # Other cell
                            #
                            cj = cells[j]
                            
                            #
                            # Get other cell's dofs and address
                            # 
                            cj_dofs = self.cell_dofs(cj)
                            cj_addr = cj.get_node_address()
                            
                            #
                            # Shape function info on ocell
                            # 
                            cj_sinfo = self.shape_info(cj)
                            
                            # 
                            # Compute Gauss nodes and weights on cell
                            # 
                            xj_g, wj_g = self.gauss_rules(cj_sinfo)
                            
                            #
                            # Compute shape functions on cell
                            #  
                            phij = self.shape_eval(cj_sinfo, xj_g, cj)
                           
                            #
                            # Evaluate integral form
                            # 
                            form_loc = form.eval((ci,cj), (xi_g,xj_g), \
                                                 (wi_g,wj_g), (phii,phij))
                                                        
                            # 
                            # Store in Assembled form
                            # 
                            # Test dof indices
                            etype_tst = form.test.element.element_type()
                            etype_trl = form.trial.element.element_type()
                            
                            # Trial dof indices
                            dofs_tst = ci_dofs[etype_tst]
                            dofs_trl = cj_dofs[etype_trl]    
                            
                            af = self.af[i_problem]['bilinear']
                            af.update((ci_addr, cj_addr), form_loc,  
                                      row_dofs = dofs_tst, col_dofs = dofs_trl)
                            
                            if form.kernel.is_symmetric() and ci!=cj:
                                #
                                # Symmetric kernel, store the transpose
                                # 
                                af.update((cj_addr, ci_addr), form_loc.T, 
                                          row_dofs = dofs_trl, col_dofs = dofs_tst)
                            
                    else:  
                        # =====================================================
                        # Not integral form
                        # =====================================================
                        #
                        # Evaluate form 
                        # 
                        form_loc = form.eval(ci, xi_g, wi_g, phii, ci_dofs, \
                                             self.compatible_functions)                   
                        
                        if form.type=='constant':
                            #
                            # Constant form
                            #
                             
                            # Increment value
                            af = self.af[i_problem]['constant'] 
                            af.update(ci_addr, form_loc) 
                            
                            
                        elif form.type=='linear':
                            # 
                            # Linear form
                            # 
                            
                            # Extract test dof indices
                            etype_tst = form.test.element.element_type()
                            dofs_tst  = ci_dofs[etype_tst]
                                    
                            # Store dofs and values in assembled_form
                            af = self.af[i_problem]['linear'] 
                            af.update(ci_addr, form_loc, row_dofs=dofs_tst)
                            
                            
                        elif form.type=='bilinear':
                            #
                            # Bilinear Form
                            # 
                                                    
                            # Test dof indices
                            etype_tst = form.test.element.element_type()
                            etype_trl = form.trial.element.element_type()
                            
                            # Trial dof indices
                            dofs_tst = ci_dofs[etype_tst]
                            dofs_trl = ci_dofs[etype_trl]    
                            
                            # Store dofs and values in assembled form
                            af = self.af[i_problem]['bilinear']  
                            af.update(ci_addr, form_loc, 
                                      row_dofs=dofs_tst, col_dofs=dofs_trl)
                                                                
        #
        # Post-process assembled forms
        # 
        for i_problem in range(len(self.problems)):
            for form_type in self.af[i_problem].keys():
                #
                # Iterate over assembled forms
                # 
                af = self.af[i_problem][form_type]
                
                #
                # Consolidate assembly
                ## 
                af.consolidate(clear_cell_data=clear_cell_data)
                
        
        '''   
        #
        # Assemble forms over boundary edges 
        #  
        if isinstance(self.mesh, Mesh2D):
            
            #
            # Determine flags used to mark boundary edges
            #
            boundary_segments = \
                self.mesh.get_boundary_segments(subforest_flag=subforest_flag)
            
            for problem in problems:
                for nc in problem['bc']['neumann']:
                    bnd_segs = self.mesh.get_boundary_segments(subforest_flag=subforest_flag, flag=nc['marker'])     
        '''  
            
        '''
            if boundary_conditions is not None:
                #
                # Unpack boundary data
                # 
                if 'dirichlet' in boundary_conditions:
                    bc_dirichlet = boundary_conditions['dirichlet']
                else:
                    bc_dirichlet = None
                
                if 'neumann' in boundary_conditions:
                    bc_neumann = boundary_conditions['neumann']
                else:
                    bc_neumann = None
                    
                if 'robin' in boundary_conditions:
                    bc_robin = boundary_conditions['robin']
                else:
                    bc_robin = None
            
        rows = []
        cols = []
        dir_dofs_encountered = set()
        for node in self.mesh.root_node().get_leaves():
            node_dofs = self.dofhandler.get_global_dofs(node)
            cell = node.cell()            
            #
            # Assemble local system matrices/vectors
            # 
            if bilinear_forms is not None:
                bf_loc = np.zeros((n_dofs,n_dofs))
                for bf in bilinear_forms:
                    bf_loc += self.form_eval(bf, node)
                    
            if linear_forms is not None:
                lf_loc = np.zeros((n_dofs,))
                for lf in linear_forms:
                    lf_loc += self.form_eval(lf, node)
                    
            if boundary_conditions:
                #
                # Boundary conditions
                # 
                for direction in ['W','E','S','N']:
                    edge = cell.get_edges(direction)
                    #
                    # Check for Neumann conditions
                    # 
                    neumann_edge = False
                    if bc_neumann is not None:
                        for bc_neu in bc_neumann:
                            m_neu,g_neu = bc_neu 
                            if m_neu(edge):
                                # --------------------------------------------- 
                                # Neumann edge
                                # ---------------------------------------------
                                neumann_edge = True
                                #
                                # Update local linear form
                                #
                                lf_loc += self.form_eval((g_neu,'v'),node, \
                                                         edge_loc=direction)
                                break
                        #
                        # Else Check Robin Edge
                        #
                        if not neumann_edge and bc_robin is not None:                    
                            for bc_rob in bc_robin:
                                m_rob, data_rob = bc_rob
                                if m_rob(edge):
                                    # ---------------------------------------------
                                    # Robin edge
                                    # ---------------------------------------------
                                    gamma_rob, g_rob = data_rob
                                    #
                                    # Update local bilinear form
                                    #
                                    bf_loc += \
                                        gamma_rob*self.form_eval((1,'u','v'),\
                                                                 node,\
                                                                 edge_loc=direction)
                                    #
                                    # Update local linear form
                                    # 
                                    lf_loc += \
                                        gamma_rob*self.form_eval((g_rob,'v'),\
                                                                 node,\
                                                                 edge_loc=direction)
                                    break                           
                #
                #  Check for Dirichlet Nodes
                #
                x_ref = self.element.reference_nodes()
                x_cell = self.rule_2d.map(cell,x=x_ref) 
                cell_dofs = np.arange(n_dofs)
                if bc_dirichlet is not None:
                    list_dir_dofs_loc = []
                    for bc_dir in bc_dirichlet:
                        m_dir,g_dir = bc_dir
                        is_dirichlet = m_dir(x_cell[:,0],x_cell[:,1])
                        if is_dirichlet.any():
                            dir_nodes_loc = x_cell[is_dirichlet,:]
                            dir_dofs_loc = cell_dofs[is_dirichlet]
                            list_dir_dofs_loc.extend(dir_dofs_loc)
                            for j,x_dir in zip(dir_dofs_loc,dir_nodes_loc):
                                #
                                # Modify jth row 
                                #
                                notj = np.arange(n_dofs)!=j
                                uj = g_dir(x_dir[0],x_dir[1])
                                if node_dofs[j] not in dir_dofs_encountered: 
                                    bf_loc[j,j] = 1.0
                                    bf_loc[j,notj]=0.0
                                    lf_loc[j] = uj
                                else:
                                    bf_loc[j,:] = 0.0  # make entire row 0
                                    lf_loc[j] = 0.0
                                #
                                # Modify jth column and right hand side
                                #
                                lf_loc[notj] -= bf_loc[notj,j]*uj 
                                bf_loc[notj,j] = 0.0
                    
                    for dof in list_dir_dofs_loc:
                        dir_dofs_encountered.add(dof)            
        
            #
            # Local to global mapping
            #
            for i in range(n_dofs):
                #
                # Update right hand side
                #
                if linear_forms is not None:
                    linvec[node_dofs[i]] += lf_loc[i]
                #
                # Update system matrix
                # 
                if bilinear_forms is not None:
                    for j in range(n_dofs):
                        rows.append(node_dofs[i]) 
                        cols.append(node_dofs[j]) 
                        bivals.append(bf_loc[i,j])                                 
        #            
        # Save results as a sparse matrix 
        #
        out = []
        if bilinear_forms is not None:
            A = sparse.coo_matrix((bivals,(rows,cols)))
            out.append(A) 
        if linear_forms is not None:
            out.append(linvec) 
        if len(out) == 1:
            return out[0]
        elif len(out) == 2:
            return tuple(out)
        '''
    
    '''
    def map_to_global(self, form_loc, form, cell):
        """
        Maps local form on a cell (in terms of local shape functions) onto the 
        global form (in terms of global basis functions). Global basis functions
        are the same as shape functions, except in cells adjoining hanging nodes.
        There, global basis fns are extended to ensure continuity over hanging n.
        
        Input:
        
            loc_form: double, np.array representing the local form returned
                by method 'Form.eval'
            
            
            form: Form, class used to extract element types
                    
            cell: Cell, mesh cell over which assembly is occurring.
            
            
        Output: 
        
            form_glb: double, array evaluated form in terms of global basis
                functions. 
                
                'constant': (1,) or (n_smpl, ) array
                'linear': (n_tst_glb, 1) or (n_tst_glb, n_smpl) array
                'bilinear: (n_tst_glb, n_trl_glb, n_smpl) array
        
        TODO: Not necessary.
        """
        subforest_flag = self.subforest_flag
        if form.type=='constant':
            #
            # Constant form
            # 
            return form_loc
        elif form.type=='linear':
            #
            # Linear form
            # 
            
            # Get element types for test functions
            etype_tst = form.test.element.element_type()
            
            # Extract dofhandler
            dh_tst = self.dofhandlers[etype_tst]
            
            # Retrieve local to global mapping
            l2g_tst = dh_tst.get_l2g_map(cell, subforest_flag=subforest_flag)
            
            # Get global dofs
            dofs_tst = list(l2g_tst.keys())
            
            # Convert l2g map to matrix
            l2g_tst = np.array(list(l2g_tst.values()))

            # Compute linear form in terms of global basis
            L = l2g_tst.dot(form_loc)
            
            # Return global linear form and global test dofs
            return L, dofs_tst
        
        elif form.type=='bilinear':
            #
            # Bilinear form
            # 
            
            # Get element types for test and trial functions
            etype_tst = form.test.element.element_type()
            etype_trl = form.trial.element.element_type()
            
            # Extract dofhandlers for both element types
            dh_tst = self.dofhandlers[etype_tst]
            dh_trl = self.dofhandlers[etype_trl]
            
            # Retrieve the local to global mapping for each dh over the cell
            l2g_tst = dh_tst.get_l2g_map(cell, subforest_flag=subforest_flag)
            l2g_trl = dh_trl.get_l2g_map(cell, subforest_flag=subforest_flag)
            
            # Get global dofs 
            dofs_tst = list(l2g_tst.keys())
            dofs_trl = list(l2g_trl.keys())
            
            # Convert l2g maps to matrix form
            l2g_tst = np.array(list(l2g_tst.values()))
            l2g_trl = np.array(list(l2g_trl.values()))
            
            # Compute bilinear form in terms of global basis
            dim = len(form_loc.shape)
            if dim==3:
                #
                # Sampled bilinear form (n_tst, n_trl, n_smpl)
                # 
                
                # Change to (n_smpl, n_tst, n_trl)
                form_loc = form_loc.transpose([2,0,1])
                
                # Multiply each slice by Test*(..)*Trial^T  
                B = l2g_tst.dot(form_loc.dot(l2g_trl.T))
                
                # Change dimensions to (n_glb_tst, n_glb_trl, n_smpl)
                B = B.transpose([0,2,1])
            elif dim==2:
                #
                # Deterministic bilinear form (n_tst, n_trl)
                # 
                B = l2g_tst.dot(form_loc).dot(l2g_trl.T)
            return B, dofs_tst, dofs_trl
    '''   
    
    '''                             
    def consolidate_assembly(self):
        """
        Postprocess assembled forms to make them amenable to linear algebra 
        operations. This includes renumbering equations that involve only a 
        subset of the degreees of freedom.
        
        Bilinear Form:
        
            row_dofs: (n_row_dofs, ) ordered numpy array of mesh dofs 
                corresponding to test functions.
                
            col_dofs: (n_col_dofs, ) ordered numpy array of unique mesh dofs
                corresponding to trial space
            
            rows: (n_nonzero,) row indices (renumbered)
            
            cols: (n_nonzero,) column indices (renumbered). 
            
            vals: (n_nonzero, n_samples) numpy array of matrix values
                corresponding to each row-column pair
            
            
        Linear Form:
        
            row_dofs: (n_row_dofs,) order array of mesh dofs corresponding
                to row dofs
            
            vals: (n_row_dofs, n_samples) array of vector values for each dof.  
        
        
        Constant form:
        
            vals: (n_samples, ) array of integral values.
            
        TODO: Delete!
        """
        
        for i_problem in range(len(self.problems)):
            for form_type in self.af[i_problem].keys():
                form = self.af[i_problem][form_type]
                n_samples = self.n_samples(i_problem, form_type)
                if form_type=='bilinear':
                    # =========================================================
                    # Bilinear Form
                    # =========================================================
                    #
                    # Parse row and column dofs
                    # 
                    # Flatten
                    rows = []
                    cols = []
                    vals = []
                    rcv = (form['row_dofs'], form['col_dofs'], form['vals'])
                    for rdof, cdof, val in zip(*rcv):
                        #
                        # Store global dofs in vectors
                        #
                        R,C = np.meshgrid(rdof,cdof)
                        rows.append(R.ravel())
                        cols.append(C.ravel())
                        
                        #
                        # Store values
                        # 
                        n_entries = len(rdof)*len(cdof) 
                        if n_samples is None:
                            #
                            # Deterministic form
                            # 
                            vals.append(val.reshape(n_entries, order='F'))
                        else:
                            #
                            # Sampled form
                            # 
                            v = val.reshape((n_entries,n_samples), order='F')
                            vals.append(v)
                    
                    # 
                    rows = np.concatenate(rows, axis=0)
                    cols = np.concatenate(cols, axis=0)
                    vals = np.concatenate(vals, axis=0)
                       
                    #
                    # Renumber dofs from 0 ... n_dofs
                    # 
                    # Extract sorted list of unique dofs for rows and columns
                    unique_rdofs = list(set(list(rows)))
                    unique_cdofs = list(set(list(cols)))
                    
                    # Dof to index mapping for rows
                    map_rows = np.zeros(unique_rdofs[-1]+1, dtype=np.int)
                    map_rows[unique_rdofs] = np.arange(len(unique_rdofs))
                    
                    # Dof-to-index mapping for cols
                    map_cols = np.zeros(unique_cdofs[-1]+1, dtype=np.int)
                    map_cols[unique_cdofs] = np.arange(len(unique_cdofs))
                    
                    # Transform from dofs to indices
                    rows = map_rows[rows]
                    cols = map_cols[cols]
                    
                    # Store row and column information
                    form['row_dofs'] = np.array(unique_rdofs)
                    form['col_dofs'] = np.array(unique_cdofs)
                    form['rows'] = rows
                    form['cols'] = cols
                    form['vals'] = vals
                    
                elif form_type=='linear':
                    # =========================================================
                    # Linear Form 
                    # =========================================================
                    #
                    # Parse row dofs
                    # 
                    # Flatten list of lists
                    rows = [item for sublist in form['row_dofs'] for item in sublist]
                    
                    # Extract sorted list of unique dofs for rows and columns
                    unique_rdofs = list(set(rows))
                    n_dofs = len(unique_rdofs)
                    
                    # Convert rows into numpy array
                    rows = np.array(rows) 
                    
                    # Dof-to-index mapping for rows
                    map_rows = np.zeros(unique_rdofs[-1]+1, dtype=np.int)
                    map_rows[unique_rdofs] = np.arange(n_dofs)
                    
                    # Transform from dofs to indices
                    rows = map_rows[rows]
                    
                    # Concatenate all function values in a vector
                    vals = np.concatenate(form['vals'])
                    if n_samples is None:
                        # 
                        # Deterministic problem
                        # 
                        b = np.zeros(n_dofs)
                        for i in range(n_dofs):
                            b[i] = vals[rows==unique_rdofs[i]].sum()
                    else:
                        #
                        # Sampled linear form
                        # 
                        b = np.zeros((n_dofs,n_samples))
                        for i in range(n_dofs):
                            b[i,:] = vals[rows==unique_rdofs[i],:].sum(axis=0)

                    # Store arrays
                    form['row_dofs'] = np.array(unique_rdofs)        
                    form['vals'] = b
                elif form_type=='constant':
                    #
                    # Constant form
                    # 
                    pass
    '''        
        
    
    def get_assembled_form(self, form_type, i_problem=0, i_sample=None):
        """
        Return the sparse matrix, vector, or constant representations of 
        bilinear, linear, or constant form respectively for each problem 
        and for each system realization.
        
        Inputs: 

            form_type: str ('bilinear','linear','constant')
        
            i_problem: int, problem index
            
            i_sample: int, sample index 
            
            
        Outputs:
        
            forms: tuple or single output depending on form_type and 
                content of assembled forms
        """
        # 
        # Check inputs
        # 
        assert i_problem < len(self.problems), \
        'Problem index exceeds number of problems assembled.'
        
        n_samples = self.n_samples(i_problem, form_type)
        if n_samples is not None:
            assert i_sample < n_samples, \
            'Sample index exceeds number of samples for problem.'
        
        assert form_type in ['constant', 'linear', 'bilinear'],\
        'Input "form_type" should be "constant", "linear", or "bilinear".'
        
        assert form_type in self.af[i_problem],\
        'Specified "form_type" not assembled for problem %d'%(i_problem)
        
        #
        # Determine form's sample size and check compatibility
        # 
        n_samples = self.n_samples(i_problem, form_type)
        if n_samples is None:
            #
            # Deterministic Form
            #
            assert i_sample is None or i_sample==0,\
            'Input "i_sample" should be "0" or "None" for deterministic form'
        else:
            #
            # Sampled Form       
            # 
            assert i_sample is not None, 'No sample index specified.'
            assert  i_sample < n_samples, 'Sample index exceeds sample size'
        
        if form_type=='bilinear':
            #
            # Bilinear Form
            # 
            rows = self.af[i_problem]['bilinear']['rows']
            cols = self.af[i_problem]['bilinear']['cols']
            vals = self.af[i_problem]['bilinear']['vals']
                
            if n_samples is None:
                #
                # Deterministic bilinear form
                #      
                A = sparse.coo_matrix((vals,(rows,cols)))        
            else:
                #
                # Sampled bilinear form 
                # 
                A = sparse.coo_matrix((vals[:,i_sample], (rows,cols)))
            return A
            
        elif form_type=='linear':
            #
            # Linear Form
            # 
            b = self.af[i_problem]['linear']['vals']
            if n_samples is not None:
                #
                # Sampled linear form
                # 
                b = b[:,i_sample]
            return b
        
        elif form_type=='constant':
            #
            # Constant form
            # 
            c = self.af[i_problem]['constant']['vals']
            if n_samples is not None:
                c = c[i_sample]
            return c
                
    
            
    '''    
    def get_boundary_dofs(self, etype, bnd_marker):
        """
        Determine the Dofs associated with the boundary vertices marked by
        the bnd_marker flag.
        
        Inputs: 
        
            etype: str, element type used to identify dofhandler
            
            bnd_marker: flag, used to identify boundary segment.
            
            i_problem: int, problem index
            
        TODO: Test
        TODO: Make this more general and move it to DofHandler class
        TODO: Delete!!
        """    
        # Check that dofhandler     
        assert etype in self.dofhandlers, \
        'Element type not recognized. Use different value for input "etype".'
        
        dofhandler = self.dofhandlers[etype]
        subforest_flag = self.subforest_flag
        mesh = dofhandler.mesh
        dim = mesh.dim()
        
        dofs = []
        if dim == 1:
            # =================================================================
            # 1D Mesh
            # =================================================================
            #
            # Obtain cells and vertices on the boundary
            # 
            cell_left, cell_right = mesh.get_boundary_cells(subforest_flag)        
            v_left, v_right = mesh.get_boundary_vertices()
            
            #
            # Get associated dofs and store in list
            # 
            dofs.extend(dofhandler.get_global_dofs(cell=cell_left, entity=v_left))
            dofs.extend(dofhandler.get_global_dofs(cell=cell_right, entity=v_right))

        elif dim == 2:
            # =================================================================
            # 2D Mesh
            # =================================================================
            #
            # Extract boundary segments
            # 
            bnd_segments = mesh.get_boundary_segments(subforest_flag, bnd_marker)
            for segment in bnd_segments:
                for edge in segment:
                    #
                    # Compute dofs associated with boundary edge
                    # 
                    edge_dofs = dofhandler.get_global_dofs(cell=edge.cell(), \
                                                           entity=edge)
                    dofs.extend(edge_dofs)
        return dofs
    '''
        
        
    def n_samples(self, i_problem, form_type):
        """
        Returns the number of realizations of problem i_problem
        
        Inputs:
        
            i_problem: int, 0<i_problem<len(self.problems) problem index
            
            form_type: str 'constant', 'linear', or 'bilinear'.
            
        Output:
        
            n_samples: int (or None), number of samples associated with the given
                form type of the given problem
        """ 
        n_samples = None   
        for form in self.problems[i_problem]:
            if form.type==form_type:
                #
                # Consider only forms of given type
                # 
                if form.kernel.n_samples is not None:
                    #
                    # Sampling in effect
                    # 
                    if n_samples is None:
                        #
                        # New trivial sample size
                        # 
                        n_samples = form.kernel.n_samples
                    else: 
                        #
                        # There exists a nontrivial sample size
                        # 
                        if form.kernel.n_samples > 1:
                            #
                            # Kernel contains more than one sample. 
                            # Check for consistency
                            # 
                            assert n_samples == form.kernel.n_samples,\
                        '    Inconsistent sample sizes in kernels'
        return n_samples    
                
        
        
    def shape_info(self, cell):
        """
        Determine what shape functions must be computed and over what region
        within a particular cell.
        
        Inputs:
        
            problems: list of problems (described in 'assemble' method)
            
            cell: cell over which to assemble
            
            subforest_flag: str/int/tuple marks submesh over which to assemble
            
            
        Output:
        
            info: (nested) dictionary, whose entries info[region][element] 
                consist of sets of tuples representing derivatives of the
                shape functions to be computed
        """
        info = {}
        for problem in self.problems:
            for form in problem:
                if form.dmu == 'dx':
                    # 
                    # Integral over cell
                    # 
                    if form.flag is None or cell.is_marked(form.flag):
                        #
                        # Cell marked by flag specified by form
                        # 
                        if cell not in info:
                            # Initialize cell key if necessary
                            info[cell] = {}
                        
                        #
                        # Get shape information from the form
                        # 
                        form_info = form.shape_info(self.compatible_functions)
                        #
                        # Update shape function information for cell
                        # 
                        for etype in form_info.keys():
                            if etype not in info[cell]:
                                # Initialize etype key if necessary
                                info[cell][etype] = \
                                    {'element': form_info[etype]['element'],
                                     'derivatives': set()}
                            D = form_info[etype]['derivatives']
                            info[cell][etype]['derivatives'].update(D)
                        
                elif form.dmu == 'ds':
                    #
                    # Integral over half-edge
                    # 
                    for half_edge in cell.get_half_edges():
                        if form.flag is None or half_edge.is_marked(form.flag):
                            #
                            # HalfEdge marked by flag specified by Form
                            # 
                            # Initialize ith key if necessary
                            if half_edge not in info:
                                info[half_edge] = {}
                            
                            #
                            # Get shape information from form
                            # 
                            form_info = form.shape_info(self.compatible_functions)
                            
                            #
                            # Update shape function information on edge
                            # 
                            for etype in form_info.keys():
                                if etype not in info[half_edge]:
                                    #
                                    # Initialize etype key if necessary
                                    #
                                    info[half_edge][etype] = \
                                        {'element': form_info[etype]['element'],
                                         'derivatives': set()}
                                #
                                # Update derivatives 
                                # 
                                D = form_info[etype]['derivatives']
                                info[half_edge][etype]['derivatives'].update(D)
                elif form.dmu == 'dv':
                    #
                    # Evaluate integrand at vertex
                    # 
                    for vertex in cell.get_vertices():
                        if form.flag is None or vertex.is_marked(form.flag):
                            #
                            # Vertex marked by flag specified by Form
                            # 
                            # Initialize ith key if necessary
                            if vertex not in info:
                                info[vertex] = {}
                            
                            #
                            # Get shape information from form
                            # 
                            form_info = form.shape_info(self.compatible_functions)
                            
                            #
                            # Update shape function information on vertex
                            # 
                            for etype in form_info.keys():
                                if etype not in info[vertex]:
                                    #
                                    # Initialize etype key if necessary
                                    # 
                                    info[vertex][etype] = \
                                        {'element': form_info[etype]['element'],
                                         'derivatives': set()}
                                #
                                # Update derivatives
                                # 
                                D = form_info[etype]['derivatives']
                                info[vertex][etype]['derivatives'].update(D)
                    
                            
        for region in info.keys():
            for etype in info[region].keys():
                #
                # Turn derivative set into list (consistent ordering).
                # 
                info[region][etype]['derivatives'] = \
                    list(info[region][etype]['derivatives'])
        return info
                
    
    def cell_dofs(self, cell):
        """
        Returns the global degrees of freedom assciated with cell for all 
        elements considered
        
        Input:
        
            cell: Cell, indexing a dictionary with stored dofs 
        
        Output:
        
            cell_dofs: dict, consisting of the dofs associated with a cell.
        """
        subforest_flag = self.subforest_flag
        cell_dofs = {}
        for etype in self.dofhandlers.keys():
            
            # Get Dofhandler for etype
            dh = self.dofhandlers[etype]
            
            # Get cell dofs 
            cd = dh.get_cell_dofs(cell, subforest_flag=subforest_flag)
            
            # Turn into an array
            cell_dofs[etype] = np.array(cd)
            
        return cell_dofs

    
    def gauss_rules(self, shape_info):
        """
        Compute the Gauss nodes and weights over all regions specified by the 
        shape_info dictionary. 
        
        Inputs:
        
            shape_info: dict, generated for each cell by means of 
                self.shape_info(cell).
                
        Outputs:
        
            xg: dict, of Gauss nodes on cell, indexed by cell's subregions
            
            wg: dict, of Gauss weights on cell, indexed by cell's subregionss
        
        """
        xg, wg = {}, {}
        for region in shape_info.keys():
            #
            # Map quadrature rule to entity (cell/halfedge)
            # 
            if isinstance(region, Interval):
                #
                # Interval
                # 
                xg[region], wg[region] = self.cell_rule.mapped_rule(region)
            elif isinstance(region, HalfEdge):
                #
                # Edge
                #
                xg[region], wg[region] = self.edge_rule.mapped_rule(region)
            elif isinstance(region, QuadCell):
                #
                # Quadrilateral
                #
                xg[region], wg[region] = self.cell_rule.mapped_rule(region)
            elif isinstance(region, Vertex):
                #
                # Vertex
                # 
                xg[region], wg[region] = region.coordinates(), 1
            else:
                raise Exception('Only Intervals, HalfEdges, Vertices, & QuadCells supported')  
        return xg, wg
        
        
                                        
    def shape_eval(self, shape_info, xg, cell):
        """
        Evaluate the element shape functions (and their derivatives) at the 
        Gauss quadrature points in each region specified by "shape_info". 
        
        
        Inputs:
        
            shape_info: dictionary, regions: list of regions (QuadCell, Interval, or HalfEdge) over 
                which gaussian quadrature rules and shape functions are sought. 
            
            derivatives: dictionary, containing a list of shape function 
                derivatives sought over each region. 
             
            etypes: 
        
        Output:
        
            phi: dictionary, of the form phi[region][derivative]
            
            xg: dictionary, of the form xg[region]
            
            wg: dictionary, of the form wg[region]
        """
        phi = {}
        for region in shape_info.keys():
            #
            # Evaluate shape functions at quadrature points
            #
            phi[region] = {}
            for etype in shape_info[region].keys():
                #
                # Add etype key to phi if necessary
                # 
                if etype not in phi[region]:
                    phi[region][etype] = {}
                
                #
                # Compute shape functions
                # 
                element = shape_info[region][etype]['element']
                D = shape_info[region][etype]['derivatives']
                p = element.shape(xg[region], derivatives=D, cell=cell)
                
                # Convert list to dictionary
                count = 0
                for drv in shape_info[region][etype]['derivatives']:
                    phi[region][etype][drv] = p[count]
                    count += 1
                
        return phi        
            
    
        
        
            
    
    
        
          
    def interpolate(self, marker_coarse, marker_fine, u_coarse=None):
        """
        Interpolate a coarse grid function at fine grid points.
        
        Inputs:
        
            marker_coarse: str/int, tree node marker denoting the cells of the
                coarse grid.
            
            marker_fine: str/int, tree node marker labeling the cells of the
                fine grid.
                
            u_coarse: double, nodal vector defined on the coarse grid.
            
            
        Outputs: 
        
            if u_coarse is not None:
                
                u_interp: double, nodal vector of interpolant
                
            elif u_coarse is None:
            
                I: double, sparse interplation matrix, u_fine = I*u_coarse
            
        """
        #
        # Initialize
        # 
        n_coarse =  self.dofhandler.n_dofs(marker_coarse)
        n_fine = self.dofhandler().n_dofs(marker_fine)
        if u_coarse is None:
            #
            # Initialize sparse matrix
            # 
            rows = []
            cols = []
            vals = []
        else:
            #
            # Interpolated nodes
            #
            u_interp = np.empty(n_fine)
        
        #    
        # Construct
        # 
        for node in self.mesh.root_node().get_leaves(marker_fine):
            if node.has_parent(marker_coarse):
                parent = node.get_parent(marker_coarse)
                node_dofs = self.dofhandler.get_global_dofs(node)
                parent_dofs = self.dofhandler.get_global_dofs(parent)
                x = self.dofhandler.dof_vertices(node)
                phi = self.shape_eval(cell=parent.cell(), x=x)
                if u_coarse is not None:
                    #
                    # Update nodal vector
                    # 
                    u_interp[node_dofs] = \
                        np.dot(phi,u_coarse[parent_dofs])
                else:
                    #
                    # Update interpolation matrix
                    # 
                    for i in range(len(node_dofs)):
                        fine_dof = node_dofs[i]
                        if fine_dof not in rows:
                            #
                            # New fine dof
                            # 
                            for j in range(len(parent_dofs)):
                                coarse_dof = parent_dofs[j]
                                phi_val = phi[i,j] 
                                if abs(phi_val) > 1e-9:
                                    rows.append(fine_dof)
                                    cols.append(coarse_dof)
                                    vals.append(phi_val)
        #
        # Return 
        # 
        if u_coarse is not None:
            return u_interp
        else:
            I = sparse.coo_matrix((vals,(rows,cols)),\
                                  shape=(n_fine,n_coarse))
            return I
    
    
    def restrict(self, marker_coarse, marker_fine, u_fine=None):
        """
        Restrict a fine grid function to a coarse mesh.
        
        Inputs:
        
            marker_coarse: str/int, tree node marker denoting the cells of the
                coarse grid.
            
            marker_fine: str/int, tree node marker labeling the cells of the
                fine grid.
                
            u_fine: nodal vector defined on the fine grid. 
            
        
        Outputs:
        
            if u_fine is not None:
            
                u_restrict: double, nodal vector defined on coarse grid
                
            if u_fine is None:
            
                R: double, sparse restriction matrix, u_restrict = R*u_fine
                
        TODO: The "correct" restriction operator is the transpose of the interpolation operator.
        """
        I = self.interpolate(marker_coarse, marker_fine)
        I = I.toarray()
        Q,R = linalg.qr(I, mode='economic')
        R = linalg.solve(R, Q.T)
        if u_fine is None:
            return R
        else:
            return R.dot(u_fine)

class AssembledForm(object):
    """
    bilinear, linear, or constant form
    """   
    def __init__(self, form, n_samples=None):
        """
        Constructor
        
        Inputs: 
        
            form: Form object (constant, linear,
                or bilinear) to be assembled.
                
            n_samples: int, 
        """ 
        #
        # Check that input is a Form
        # 
        assert isinstance(form, Form), 'Input "form" should be a "Form" object'
            
        
        # Record form type    
        self.type = form.type
        
        if self.type == 'constant':
            #
            # Constant form
            # 
            # Cell address 
            self.cell_address = []
                        
            # Values
            self.cell_vals = []
        
        elif self.type == 'linear':
            #
            # Linear form
            # 
            # Cell address 
            self.cell_address = []
            
            # Degrees of freedom
            self.cell_row_dofs = []
            
            # Values
            self.cell_vals = []
            
            # Element type (test function)
            self.test_etype = form.test.element.element_type()
            
        elif self.type == 'bilinear':
            #
            # Bilinear form
            # 
            # Cell address 
            self.cell_address = []
            
            # Degrees of freedom
            self.cell_row_dofs = []
            self.cell_col_dofs = []
            
            # Values
            self.cell_vals = []
            
            # Element types (test and trial)
            self.test_etype = form.test.element.element_type()
            self.trial_etype = form.trial.element.element_type()
            
        #
        # Record the sample size for the assembled form, ensuring that 
        # it is compatible with that of the form   
        # 
        fn_samples = form.kernel.n_samples
        if n_samples is None:
            #
            # No sample size provided, use sample that of form
            # 
            self.n_samples = fn_samples
        elif n_samples==1:
            #
            # Update sample size only if form's sample size > 1
            # 
            if fn_samples is not None and fn_samples > 1:
                self.n_samples = fn_samples
            else:
                self.n_samples = n_samples
        elif n_samples > 1:
            #
            # Check that number of samples is compatible
            # 
            if fn_samples is not None:
                if fn_samples > 1:
                    assert n_samples == fn_samples, 'Input "n_samples" is '+\
                        'incompatible with sample size of form.'
            self.n_samples = n_samples
        
        #
        # Initialize subsample       
        # 
        self.sub_sample = None
        
            
    def check_compatibility(self, form):
        """
        Determine whether the input "form" is consistent with the assembled 
        form. In particular, they should be of the same type ('constant',
        'bilinear', or 'linear') and their test/trial functions should have
        the same element type.
        
        Input:
        
            form: Form, to be checked.
            
            update_sample_size: bool, if assembled form has sample size 1 or 
                None and form has a sample size greater than 
        """
        assert isinstance(form, Form), 'Input "form" must be a "Form" object.'
        assert self.type == form.type, 'Input "form" should have the same '+\
            'type as AssembledForm.'
        if self.type == 'linear':
            assert self.test_etype == form.test.element.element_type(),\
                'Test element types not compatible.'
        elif self.type == 'bilinear':
            assert self.test_etype == form.test.element.element_type(),\
                'Test element types not compatible.'
            assert self.trial_etype == form.trial.element.element_type(),\
                'Trial element types not compatible.'
        
        #
        # Check sample size
        # 
        sample_error = 'Sample size incompatible with that of form.'
        fn_samples = form.kernel.n_samples
        n_samples = self.n_samples
        if n_samples is None:
            assert fn_samples is None, sample_error
        elif n_samples == 1:
            assert fn_samples is None or fn_samples == 1, sample_error
        elif n_samples > 1:
            assert fn_samples is None or fn_samples == n_samples, sample_error
            

    def update(self, address, vals, row_dofs=None, col_dofs=None):
        """
        Update assembled form
        
        Inputs:
        
            address: int tuple, representing the cell's address within the mesh 
            
            vals: double, (array of) local form values 
            
            row_dofs: int, array of row dofs associated with current values
            
            col_dofs: int, array of column dofs associated with current values
        """
        #
        # Update cell address
        # 
        self.cell_address.append(address)
        
        #
        # Update form values
        # 
        self.cell_vals.append(vals)
        
        if self.type=='linear':
            #
            # Update row dofs
            # 
            assert row_dofs is not None, 'Linear forms should contain row dofs'
            self.cell_row_dofs.append(row_dofs)
        elif self.type == 'bilinear':
            #
            # Update row dofs 
            # 
            assert row_dofs is not None, 'Bilinear forms should contain'+\
                'row dofs.'
            self.cell_row_dofs.append(row_dofs)
            
            #
            # Update column dofs
            # 
            assert col_dofs is not None, 'Bilinear forms should containt'+\
                'column dofs.'
            self.cell_col_dofs.append(col_dofs)
    
    
    
    def consolidate(self, clear_cell_data=True):
        """
        Postprocess assembled form to make it amenable to linear algebra 
        operations. This includes renumbering equations that involve only a 
        subset of the degreees of freedom.
        
        Input:
        
            clear_cell_data: bool, specify whether to delete cellwise specific
                data (such as dofs and vals).
        
        Bilinear Form:
        
            row_dofs: (n_row_dofs, ) ordered numpy array of mesh dofs 
                corresponding to test functions.
                
            col_dofs: (n_col_dofs, ) ordered numpy array of unique mesh dofs
                corresponding to trial space
            
            rows: (n_nonzero,) row indices (renumbered)
            
            cols: (n_nonzero,) column indices (renumbered). 
            
            vals: (n_nonzero, n_samples) numpy array of matrix values
                corresponding to each row-column pair
            
            
        Linear Form:
        
            row_dofs: (n_row_dofs,) order array of mesh dofs corresponding
                to row dofs
            
            vals: (n_row_dofs, n_samples) array of vector values for each dof.  
        
        
        Constant form:
        
            vals: (n_samples, ) array of integral values.
        
        """ 
        n_samples = self.n_samples
        if self.type=='bilinear':
            # =========================================================
            # Bilinear Form
            # =========================================================
            #
            # Parse row and column dofs
            # 
            # Flatten
            rows = []
            cols = []
            vals = []
            rcv = (self.cell_row_dofs, self.cell_col_dofs, self.cell_vals)
            for rdof, cdof, val in zip(*rcv):
                #
                # Store global dofs in vectors
                #
                R,C = np.meshgrid(rdof,cdof)
                rows.append(R.ravel())
                cols.append(C.ravel())
                
                #
                # Store values
                # 
                n_entries = len(rdof)*len(cdof) 
                if n_samples is None:
                    #
                    # Deterministic form
                    # 
                    vals.append(val.reshape(n_entries, order='F'))
                else:
                    #
                    # Sampled form
                    # 
                    v = val.reshape((n_entries,n_samples), order='F')
                    vals.append(v)
            
            # Store data in numpy arrays
            rows = np.concatenate(rows, axis=0)
            cols = np.concatenate(cols, axis=0)
            vals = np.concatenate(vals, axis=0)
               
            #
            # Renumber dofs from 0 ... n_dofs
            # 
            # Extract sorted list of unique dofs for rows and columns
            unique_rdofs = list(set(list(rows)))
            unique_cdofs = list(set(list(cols)))
            
            # Dof to index mapping for rows
            map_rows = np.zeros(unique_rdofs[-1]+1, dtype=np.int)
            map_rows[unique_rdofs] = np.arange(len(unique_rdofs))
            
            # Dof-to-index mapping for cols
            map_cols = np.zeros(unique_cdofs[-1]+1, dtype=np.int)
            map_cols[unique_cdofs] = np.arange(len(unique_cdofs))
            
            # Transform from dofs to indices
            rows = map_rows[rows]
            cols = map_cols[cols]
            
            # Store row and column information
            self.row_dofs = np.array(unique_rdofs)
            self.col_dofs = np.array(unique_cdofs)
            self.rows = rows
            self.cols = cols
            self.vals = vals
            
            if clear_cell_data:
                #
                # Delete local data
                #  
                self.cell_vals = []
                self.cell_row_dofs = []
                self.cell_col_dofs = []
                self.cell_address = []
            
        elif self.type=='linear':
            # =========================================================
            # Linear Form 
            # =========================================================
            #
            # Parse row dofs
            # 
            # Flatten list of lists
            rows = [item for sublist in self.cell_row_dofs for item in sublist]
            
            # Extract sorted list of unique dofs for rows and columns
            unique_rdofs = list(set(rows))
            n_dofs = len(unique_rdofs)
            
            # Convert rows into numpy array
            rows = np.array(rows) 
            
            # Dof-to-index mapping for rows
            map_rows = np.zeros(unique_rdofs[-1]+1, dtype=np.int)
            map_rows[unique_rdofs] = np.arange(n_dofs)
            
            # Transform from dofs to indices
            rows = map_rows[rows]
            
            # Concatenate all function values in a vector
            vals = np.concatenate(self.cell_vals)
            if n_samples is None:
                # 
                # Deterministic problem
                # 
                b = np.zeros(n_dofs)
                for i in range(n_dofs):
                    b[i] = vals[rows==unique_rdofs[i]].sum()
            else:
                #
                # Sampled linear form
                # 
                b = np.zeros((n_dofs,n_samples))
                for i in range(n_dofs):
                    b[i,:] = vals[rows==unique_rdofs[i],:].sum(axis=0)

            # Store arrays
            self.row_dofs = np.array(unique_rdofs)        
            self.vals = b
            
            #
            # Clear cellwise data
            # 
            if clear_cell_data:
                self.cell_row_dofs = []
                self.cell_vals = []
                self.cell_address = []
                
        elif self.type=='constant':
            # =================================================================
            # Constant form
            # =================================================================
            if n_samples is None:
                #
                # Deterministic constant form - a number  
                # 
                self.vals = 0
            else:
                #
                # Sampled constant form - a vector
                # 
                self.vals = np.zeros(n_samples)
            #
            # Aggregate cellwise values
            # 
            for cell_val in self.cell_vals:
                self.vals += cell_val
                
            #
            # Clear cell
            # 
            if clear_cell_data:
                self.cell_vals = []
                self.cell_address = []
    
                
    def set_subsample(self, subsample):
        """
        Set subset of realizations. This information is used
        
        Input: 
        
            samples: int, numpy array containing integers below self.n_sample
        """
        #
        # Check that input "samples" is compatible
        # 
        n_samples = self.n_samples
        assert n_samples is not None, 'AssembledForm is not sampled.'
    
        assert all([s < n_samples for s in subsample]),\
            'Some subsamples entries exceed total number of samples.'
    
        #
        # Record subsample
        #
        self.sub_sample = subsample
    
    
    def clear_subsample(self):
        """
        Remove restriction on samples
        """   
        self.subsample = None
        
    
    def subsample_size(self):
        """
        Returns the size of the subsample (or n_samples) 
        """
        if self.subsample is None:
            #
            # No subsample specified -> return sample size
            # 
            return self.n_samples
        else:
            #
            # Return number of entries in subsample vector
            # 
            return len(self.subsample)
     
                
    def get_matrix(self):
        """
        Returns the linear algebra object associated with the AssembledForm. In
        particular: 
        
        Bilinear: An (n_sample, ) list of sparse matri(ces) 
        
        Linear: An (n_dofs, n_samples) matrix whose columns correspond to the 
            realizations of the linear form.
        
        Constant: An (n_samples, ) array whose entries coresponds to the
            realizations of the constant form.
    
        """
        if self.type == 'bilinear':
            #
            # Bilinear Form
            # 
            if self.n_samples is None:
                #
                # Single sample -> deterministic form
                # 
                A = sparse.coo_matrix((self.vals, (self.rows, self.cols)))
                return A.tocsr()
            else:
                #
                # Multiple samples -> list of deterministic forms
                # 
                if self.sub_sample is None:
                    #
                    # Entire sample
                    #
                    matrix_list = []
                    for w in range(self.n_samples):
                        A = sparse.coo_matrix((self.vals[:,w], \
                                               (self.rows, self.cols)))
                        matrix_list.append(A.tocsr())
                    return matrix_list
                else: 
                    #
                    # Sub-sample
                    # 
                    matrix_list = []
                    for w in self.sub_sample:
                        A = sparse.coo_matrix((self.vals[:,w], \
                                              (self.rows, self.cols)))
                        matrix_list.append(A.tocsr())
                    return matrix_list
        elif self.type == 'linear':
            #
            # Linear Form
            #
            if self.n_samples is None:
                #
                # Single sample, deterministic form
                # 
                return sparse.csr_matrix(self.vals).T
                
            else:
                if self.sub_sample is None:
                    #
                    # Entire sample as matrix
                    # 
                    return sparse.csc_matrix(self.vals)
                else:
                    #
                    # Sub-sample
                    # 
                    return sparse.csc_matrix(self.vals[:,self.sub_sample])
        elif self.type == 'constant':
            #
            # Constant form
            #
            if self.n_samples is None:
                #
                # Single sample, a number
                #  
                return self.vals
            else:
                if self.subsample is None:
                    #
                    # Return entire sample
                    #
                    return self.vals
                else:
                    #
                    # Return sub-sample
                    # 
                    return self.vals[self.subsample]
