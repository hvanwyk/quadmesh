import numpy as np
import numbers
from scipy import sparse
from scipy import linalg
import scipy.sparse.linalg as spla

from mesh import Vertex, Interval, HalfEdge, QuadCell, convert_to_array
from function import Map, Nodal, Constant
from fem import parse_derivative_info, Basis
from inspect import signature
import time

class GaussRule():
    """
    Description:
    ------------

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

            shape: str, 'interval', 'triangle', or 'quadrilateral'.

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
                r[6] = (t3,1.0-2.0*t3)

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


    def shape(self):
        """
        Return the geometric shape of the reference region 
        """
        return self.__cell_type
        
        
    def scale_rule(self, position, scale, nodes=None, weights=None):
        """
        Description
        -----------
        Scale the quadrature rule defined over the reference region to an 
        equivalent rule over a sub-region. When mapped to a physical region, 
        this will result in a quadrature over an appropriate sub-region thereof.  
        
        This is useful when evaluating integrals over cells in which the 
        integrands have different levels of resolution.
         
        
        Parameters
        ----------
        position : double,
            Position of sub-region within larger region. If the region is an
            interval, the position is a scalar, if it is a quadrilateral, the 
            position is a pair.
                        
        scale : double >0,
            Shrinkage factor of the sub-region size (length) relative to that 
            of the region.  
            
        nodes: double, 
            Nodes on the reference cell. If none are specified, the stored 
            quadrature nodes are used.  
            
        weights: double, 
            Quadrature weights on the reference cell. 
        
        
        Returns
        -------
        x_scaled : double, 
            Vector of quadrature nodes defined over the sub-region. 
        
        w_scaled : double, 
            Vector of quadrature weights over sub-region.
        
            
        Notes
        -----
        1. Use region.reference_map to generate quadrature rules over physical
            regions.
            
        2. Use region.subcell_position to determine the relative position and 
            size of a region's sub-region (Intervals, HalfEdges, QuadCells).
            
        TODO: Test   
        """
        if nodes is None:
            # Quadrature Rule
            quadrature = True
            
            # Get reference quadrature nodes and weights
            nodes, weights = self.nodes(), self.weights()
        else:
            # Evaluation
            quadrature = False
            
        
        # Scale and shift nodes as specified    
        x_scaled = position + scale*nodes
        
        if not quadrature:
            #
            # Return only the scaled and shifted nodes
            #
            return x_scaled
        else:
            #
            # Adjust the quadrature weights
            #  
            shape = self.shape()
            if shape == 'interval':
                #
                # On the unit interval
                #
                
                # Scale reference weights by Jacobian
                w_scaled = scale*weights
                
            elif shape == 'quadrilateral':
                #
                # On the unit square
                # 
                assert isinstance(position,np.ndarray), \
                    'Input "position" must be a numpy array.'
                    
                assert position.shape == (1,2), \
                    'Position must have dimensions (1,2).'
                    
                assert np.dim(scale)==0, \
                    'Input "scalar" should have dimension 0.'
                
                # Scale the weights 
                w_scaled = scale**2*weights 
                
            else:
                raise Exception('Only shapes of type "interval" or '+\
                                '"quadrilateral" supported.')
            
            return x_scaled, w_scaled
        
    
    def map_rule(self, region, nodes=None, basis=None):
        """
        Description
        -----------
        Maps a set of reference nodes to a physical region, adjusts the 
        associated quadrature weights and evaluates the shape functions.  
        
        Parameters
        ----------
        region : {QuadCell, Interval, HalfEdge}, 
            Region to which the rule (or just nodes) is mapped.
        
        nodes : double, 
            Nodes on the reference cell. If none are specified, the stored 
            quadrature nodes are used.  
                            
        basis : (list of) Basis,
            Basis functions to be evaluated on the given region. 
        
        
        Returns
        -------
        xg : double, 
            Quadrature (or evaluation) nodes on physical region.
            
        wg : double, 
            Quadrature weights associated with the physical region.
            
        shapes : double,
            Basis-indexed dictionary of arrays corresponding to the shape 
            functions evaluated at the given nodes. 
        
        
        Notes
        -----
        TODO: This method replaces "mapped_rule", which can be deleted once this 
            is done and tested.
            
        TODO: Test this method. 
        
        """
        if nodes is None:
            # Quadrature rule (default) 
            quadrature = True
            
            # Reference nodes
            nodes, weights = self.nodes(), self.weights()
        else:
            # Evaluation at given nodes
            quadrature = False
            
        if basis is None:
            #
            # No basis specified -> No need for shape functions
            # 
            if quadrature:
                #
                # Quadrature nodes (modify the weights)
                # 
                
                # Map to physical region 
                xg, mg = region.reference_map(nodes, jac_r2p=True)
                
                #
                # Update the weights using the Jacobian
                # 
                jac = mg['jac_r2p']
                if isinstance(region, Interval):
                    # Interval
                    dxdr = np.array(jac)
                elif isinstance(region, HalfEdge):
                    # HalfEdge
                    dxdr = np.array(np.linalg.norm(jac[0]))
                elif isinstance(region, QuadCell):
                    # QuadCell
                    dxdr = np.array([np.linalg.det(j) for j in jac])
                else:
                    raise Exception('Only regions of type "Interval",' + \
                                    '"HalfEdge", or "QuadCell" supported.')
                
                # Modify the reference weights
                wg = weights*dxdr
                
                # Return the nodes and weights
                return xg, wg 
            else:
                #
                # Nodes specified: No quadrature weights
                # 
                xg = region.reference_map(nodes)
                
                # Return only the mapped nodes
                return xg
            
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Below here, basis is not None!! 
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Initialize dictionary of shapes 
        shapes = dict.fromkeys(basis,0)
        
        #
        # Evaluating basis functions on HalfEdges -> map onto reference cell
        # 
        if isinstance(region, HalfEdge):
            #
            # Half-Edge
            # 
            
            # Get physical cell from HalfEdge
            cell = region.cell()
               
            # Get reference cell from single basis function
            ref_cell = basis[0].dofhandler().element.reference_cell()
    
            # Determine equivalent half-edge on reference element
            i_he = cell.get_half_edges().index(region)
            ref_he = ref_cell.get_half_edge(i_he)
            b,h = convert_to_array(ref_he.get_vertices())
            
            # Map 1D nodes onto reference HalfEdge
            nodes = np.array([b[i]+nodes*(h[i]-b[i]) for i in range(2)]).T
        
        
        #
        # Group the basis according to scales
        # 
        grouped_basis = {}
        for b in basis:
            # Determine mesh-flag associated with basis function
            basis_meshflag = b.subforest_flag()
            
            # Add basis to list under meshflag 
            if basis_meshflag not in grouped_basis:
                grouped_basis[basis_meshflag] = [b]
            else:
                grouped_basis[basis_meshflag].append(b)
        
        
        #
        # Scale nodes and weights if necessary!
        # 
        
        # Group nodes and weights according to meshflag
        grouped_nodes_weights = {}
        for meshflag, basis in enumerate(grouped_basis):
            #
            # Determine position of region's cell relative to basis cell 
            # 
            if isinstance(region, HalfEdge):
                cell = region.cell()
            else:
                cell = region
            
            # Get cell on which basis is defined
            coarse_cell = cell.nearest_ancestor(meshflag)
            
            #
            # Determine scaling
            # 
            if cell != coarse_cell:
                # Cell is strictly contained in coarse_cell
                position, scale = coarse_cell.subcell_position(cell)
                
                #
                # Scaled nodes and weights
                #
                if quadrature:
                    # Quadrature nodes
                    grouped_nodes_weights[meshflag] = \
                        self.scale_rule(position, scale, nodes, weights)
                else:
                    # Evaluation nodes
                    grouped_nodes_weights[meshflag] = \
                        self.scale_rule(position, scale, nodes)
            else:
                #
                # Unscaled nodes and weights
                # 
                if quadrature: 
                    # Quadrature nodes
                    grouped_nodes_weights[meshflag] = (nodes, weights)
                else:
                    # Evaluation nodes
                    grouped_nodes_weights[meshflag] = nodes
            
            
            #
            # Parse Basis for required derivatives
            #
            
            # Check whether we need jacobians and/or Hessians  
            jac_p2r = any([b.derivative()[0]>=1 for b in basis])
            hess_p2r = any([b.derivative()[0]==2 for b in basis])
            
            #
            # Map points to physical region
            #
            if quadrature:
                #
                # Quadrature rule
                #
                x_ref, w_ref = grouped_nodes_weights[meshflag]
                xg, mg = region.reference_map(x_ref, jac_r2p=True, 
                                              jac_p2r=jac_p2r, 
                                              hess_p2r=hess_p2r)
                #
                # Update the weights using the Jacobian
                # 
                jac = mg['jac_r2p']
                if isinstance(region, Interval):
                    # Interval
                    dxdr = np.array(jac)
                elif isinstance(region, HalfEdge):
                    # HalfEdge
                    dxdr = np.array(np.linalg.norm(jac[0]))
                elif isinstance(region, QuadCell):
                    # QuadCell
                    dxdr = np.array([np.linalg.det(j) for j in jac])
                else:
                    raise Exception('Only regions of type "Interval",' + \
                                    '"HalfEdge", or "QuadCell" supported.')
                
                # Modify the reference weights
                wg = w_ref*dxdr
                
            else:
                #
                # Evaluation
                #
                x_ref = grouped_nodes_weights[meshflag]
                xg, mg = region.reference_map(x_ref, 
                                              jac_p2r=jac_p2r, 
                                              hess_p2r=hess_p2r)
                
            for b in basis:
                #
                # Evaluate the basis functions at the (scaled) reference points
                #
                element = b.dofhandler().element
                D = b.derivative()
                jac_p2r = mg['jac_p2r'] if D[0] in [1,2] else None
                hess_p2r = mg['hess_p2r'] if D[0]==2 else None

                shapes[b] = \
                    element.shape(x_ref=x_ref, derivatives=D,
                                  jac_p2r=jac_p2r, hess_p2r=hess_p2r)
        if quadrature:
            # Quadrature 
            return xg, wg, shapes
        else:
            # Evaluation
            return xg, shapes
        
        
    
    def mapped_rule(self, region, basis=[], jac_p2r=False, hess_p2r=False):
        """
        Return the rule associated with a specific Cell, Interval, or HalfEdge
        as well as the inverse jacobians and hessians associated with the
        transformation.

        Parameters:
        -----------

            region : object, {Interval, HalfEdge, or Cell}
                Region to which rule is mapped.

            basis : list,
                List of basis functions defined on the region

            jac_p2r, hess_p2r: bool, indicate whether the jacobian and hessian
                of the inverse mapping should be returned. These are useful
                when evaluating the gradients and second derivatives of shape
                functions.

        TODO: Move assembler.shape_eval part to here.
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
            xg, mg = region.reference_map(x_ref, jac_r2p=True,
                                          jac_p2r=jac_p2r,
                                          hess_p2r=hess_p2r)

            # Get Jacobian of forward mapping
            jac = mg['jac_r2p']

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
            xg, mg = region.reference_map(x_ref, jac_r2p=True,
                                           jac_p2r=jac_p2r,
                                           hess_p2r=hess_p2r)
            # Get jaobian of forward mapping
            jac = mg['jac_r2p']

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

            # Map reference quadrature nodes to quadcell
            xg, mg = region.reference_map(x_ref, jac_r2p=True,
                                          jac_p2r=jac_p2r,
                                          hess_p2r=hess_p2r)

            # Get Jacobian of forward mapping
            jac = mg['jac_r2p']

            # Modify quadrature weights
            wg = w_ref*np.array([np.linalg.det(j) for j in jac])

        else:
            raise Exception('Only Intervals, HalfEdges, & QuadCells supported')
        #
        # Return Gauss nodes and weights, and Jacobian/Hessian of inverse map
        #
        if any([jac_p2r,hess_p2r]):
            return xg, wg, mg
        else:
            return xg, wg



class Kernel(object):
    """
    Kernel (combination of Functions) to be used in Forms
    """
    def __init__(self, f, derivatives=None, F=None):
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
        # Parse function derivatives
        #
        dfdx = []
        if derivatives is None:
            #
            # No derivatives specified
            #
            dfdx = [None for dummy in self.__f]
        elif type(derivatives) is list:
            #
            # derivatives specified in list
            #
            assert len(derivatives)==n_functions, \
            'The size of input "derivatives" incompatible with '+\
            'that of input "f".'
            dfdx = derivatives
        else:
            #
            # Single derivative applies to all functions
            #
            dfdx = parse_derivative_info(derivatives)
            dfdx = [dfdx for dummy in self.__f]
        self.__dfdx = dfdx

        #
        # Store meta function F
        #
        # Check that F takes the right number of inputs
        if F is None:
            # Store metafunction F
            assert n_functions == 1, \
                'If input "F" not specified, only one function allowed.'
            F = lambda f: f
        self.__F = F

        # Store function signature of F
        sigF = signature(F)

        # Figure out which of the
        cell_args = {}
        for arg in ['cell', 'region', 'phi', 'dofs']:
            if arg in sigF.parameters:
                cell_args[arg] = None
        bound = sigF.bind_partial(**cell_args)

        self.__bound = bound
        self.__signature = sigF


        # Store subsample
        self.set_sample_size()


    def basis(self):
        """
        Determine the basis functions used in the Kernel
        """
        basis = []
        for f in self.__f:
            if isinstance(f, Nodal):
                basis.append(f.basis())
        return basis


    def set_sample_size(self):
        """
        Set kernel's sample size
        
        Notes
        =====
        The kernel's sample size is determined as the sample size of the 
        constituent functions, provided they all have same sample size or a 
        sample size of 1. 
        """
        n_samples = 1
        for f in self.f():
            fn_samples = f.n_samples()
            if fn_samples>1:
                #
                # f has multiple realizations
                # 
                if n_samples==1:
                    #
                    # Update Kernel sample size
                    # 
                    n_samples = fn_samples
                else:
                    #
                    # Check that sample sizes match
                    # 
                    assert n_samples==fn_samples, 'Kernel sample size ' + \
                    'incompatible with that of constituent functions' 
        
        # Store sample size
        self.__n_samples = n_samples


    def n_samples(self):
        """
        Returns the number of Kernel realizations
        """
        return self.__n_samples
    

    def f(self):
        """
        Returns the list of functions
        """
        return self.__f


    def F(self):
        """
        Returns the metafunction
        """
        return self.__F


    def is_symmetric(self):
        """
        Returns True if all functions in the kernel are symmetric.
        """
        return all([f.is_symmetric() for f in self.f()])



    def eval(self, x, phi=None, cell=None, region=None, dofs=None):
        """
        Evaluate the kernel at the points stored in x

        Inputs:

            x: (n_points, dim) array of points at which to evaluate the kernel

            phi: basis-indexed dictionary of shape functions

            region: Geometric region (Cell, Interval, HalfEdge, Vertex)
                Included for modified kernels

            cell: Interval or QuadCell on which kernel is to be evaluated

            phi: (basis-indexed) shape functions over region


        Output:

            Kernel function evaluated at point x.

        TODO: FIX KERNEL! Interaction with assembler
            - Different mesh sizes
            - Derivatives vs. Basis functions.
        """
        #
        # Evaluate constituent functions
        #
        f_vals = []
        n_samples = self.n_samples()
        for f, dfdx in zip(self.__f, self.__dfdx):
            if isinstance(f, Nodal):
                phi_f = phi if phi is None else phi[f.basis()]
                dof_f = None if dofs is None else dofs[f.basis()]
                if dof_f is None or phi_f is None:
                    fv = f.eval(x=x, derivative=dfdx, cell=cell)
                else:
                    fv = f.eval(x=x, derivative=dfdx, cell=cell, phi=phi_f,
                                dofs=dof_f)
            else:
                fv = f.eval(x=x)
            
            #
            # Copy function values if n_samples > 1 and function is deterministic
            # 
            if n_samples > 1 and fv.shape[1]==1:
                #
                # Copy the function values
                # 
                fv = np.tile(fv, (1,n_samples))
            
            #
            # Update list of function values
            # 
            f_vals.append(fv)

        #
        # Combine functions using meta-function F
        #

        # Figure out which of the keyword parameters F can take
        signature = self.__signature
        bound = self.__bound
        cell_args = {'phi': phi, 'cell': cell, 'region':region, 'dofs':dofs}
        for arg, val in cell_args.items():
            if arg in signature.parameters:
                bound.arguments[arg] = val

        # Evaluate F
        return self.__F(*f_vals, **bound.kwargs)



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
                'ds' - integrate over a half-edge
                'dv' - integrate over a vertex

            *flag: str/int/tuple cell/half_edge/vertex marker

            *dim: int, dimension of the domain.
        """
        #
        # Parse test function
        #
        if test is not None:
            dim = test.dofhandler().element.dim()

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
            assert dim==trial.dofhandler().element.dim(), \
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



    def basis(self):
        """
        Returns
        =======

        basis: list of Basis objects,

            Returns a list of all the form's basis functions (trial, test, and
            those used to define the Kernel).

        """
        basis = []
        if self.test is not None:
            #
            # Add test basis
            #
            basis.append(self.test)
        if self.trial is not None:
            #
            # Add trial basis
            #
            basis.append(self.trial)

        #
        # Add basis functions from the kernel
        #
        basis.extend(self.kernel.basis())

        #
        # Return basis list
        #
        return basis



    def dim(self):
        """
        Return the dimension of the form

        0 = constant
        1 = linear
        2 = bilinear

        """
        if self.test is None:
            #
            # Constant
            #
            return 0
        elif self.trial is None:
            #
            # Linear
            #
            return 1
        else:
            #
            # Bilinear
            #
            return 2


    def regions(self, cell):
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


    def eval(self, cell, xg, wg, phi, dofs):
        """
        Evaluates the local kernel, test, (and trial) functions of a (bi)linear
        form on a given entity.

        Inputs:

            cell: Cell containing subregions over which Form is defined

            xg: dict, Gaussian quadrature points, indexed by regions.

            wg: dict, Gaussian quadrature weights, indexed by regions.

            phi: dict, shape functions, indexed by regions -> basis

            dofs: dict, global degrees of freedom associated with region,
                indexed by region -> basis

        Outputs:

            Constant-, linear-, or bilinear forms and their associated local
            degrees of freedom.


        TODO: Explain what the output looks like!
        Note: This method should be run in conjunction with the Assembler class
        """
        # Determine regions over which form is defined
        regions = self.regions(cell)

        # Number of samples
        n_samples = self.kernel.n_samples()

        f_loc = None
        for region in regions:
            # Get Gauss points in region
            x = xg[region]

            #
            # Compute kernel, weight by quadrature weights
            #
            kernel = self.kernel
            Ker = kernel.eval(x=x, region=region, cell=cell,
                              phi=phi[region], dofs=dofs[region])

            # Weight kernel using quadrature weights
            wKer = (wg[region]*Ker.T).T
            if self.type=='constant':
                #
                # Constant form
                #

                # Initialize form if necessary
                if f_loc is None:
                    f_loc = np.zeros((1,n_samples))
                #
                # Update form
                #
                f_loc += np.sum(wKer, axis=0)

            elif self.type=='linear':
                #
                # Linear form
                #

                # Test functions evaluated at Gauss nodes
                n_dofs_test = self.test.dofhandler().element.n_dofs()
                test = phi[region][self.test]

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

                # Test functions evaluated at Gauss nodes
                n_dofs_test = self.test.dofhandler().element.n_dofs()
                test = phi[region][self.test]

                # Trial functions evaluated at Gauss nodes
                n_dofs_trial = self.trial.dofhandler().element.n_dofs()
                trial = phi[region][self.trial]

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
                n_dofs_test = self.test.dofhandler().element.n_dofs()
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
                n_dofs_test = self.test.dofhandler().element.n_dofs()
                n_dofs_trial = self.trial.dofhandler().element.n_dofs()
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


class IIForm(Form):
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


    def eval(self, cell, xg, wg, phi, dofs):
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
        # =====================================================================
        # Interpolate in the test function component
        # =====================================================================
        test = self.test
        x = test.dofhandler().get_dof_vertices(test.subforest_flag())
        n = x.shape[0]

        # =====================================================================
        # Specify trial function
        # =====================================================================
        trial = self.trial

        # Number of dofs
        n_dofs = trial.dofhandler().element.n_dofs()

        f_loc = None
        for reg in self.regions(cell):
            # Get trial functions evaluated at Gauss nodes
            phi_g = phi[reg][trial]
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
                x = (x[ii.ravel(),:],x_g[jj.ravel(),:])
                """
                if self.dim == 1:
                    x1, x2 = x[ii.ravel()], x_g[jj.ravel()]
                elif self.dim == 2:

                """

                C_loc = self.kernel.eval(x, region=reg, cell=cell,
                                         phi=phi[reg], dofs=dofs[reg])
                C_loc = C_loc.reshape(n,n_gauss)

                #
                # Compute local integral
                #
                # Weight shape functions
                Wphi = np.diag(w_g).dot(phi_g)

                # Combine
                f_loc += C_loc.dot(Wphi)
        return f_loc



class IPForm(Form):
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
        Form.__init__(self, kernel=kernel, trial=trial, test=test, dmu=dmu, flag=flag)

        #
        # Checks
        #
        assert trial is not None and test is not None,\
        'Integral forms have both test and trial functions'

        for f in kernel.f():
            assert f.n_variables()==2, 'Integral kernel must be bivariate.'



    def eval(self, cells, xg, wg, phi, dofs):
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
        regi = self.regions(ci)
        regj = self.regions(cj)

        # =====================================================================
        # Specify the test and trial functions
        # =====================================================================
        test = self.test
        trial = self.trial

        # Degrees of freedom
        n_dofsi = self.test.dofhandler().element.n_dofs()
        n_dofsj = self.trial.dofhandler().element.n_dofs()

        # Sample size
        n_samples = self.kernel.n_samples()

        f_loc = None
        for regi in self.regions(ci):
            for regj in self.regions(cj):
                # Access test(i) and trial(j) functions
                phii = phi[0][regi][test]
                phij = phi[1][regj][trial]

                # Get quadrature nodes
                xi_g = xg[0][regi]
                xj_g = xg[1][regj]

                # Get quadrature weights
                wi_g = wg[0][regi]
                wj_g = wg[1][regj]

                # Get dofs
                dofi = dofs[0][regi]
                dofj = dofs[1][regj]

                #
                # Initialize local matrix if necessary
                #
                if f_loc is None:
                    #
                    # Initialize form
                    #
                    if n_samples==1:
                        f_loc = np.zeros((n_dofsi,n_dofsj))
                    else:
                        f_loc = np.zeros((n_dofsi,n_dofsj,n_samples))
                #
                # Evaluate kernel function at the local Gauss points
                #
                n_gauss = xi_g.shape[0]
                ig = np.arange(n_gauss)
                ii,jj = np.meshgrid(ig,ig,indexing='ij')

                x = (xi_g[ii.ravel(),:],xj_g[jj.ravel(),:])
                """
                if self.dim() == 1:
                    x1, x2 = xi_g[ii.ravel()], xj_g[jj.ravel()]
                elif self.dim() == 2:
                    x1, x2 = xi_g[ii.ravel(),:],xj_g[jj.ravel(),:]
                """
                #x, phi=None, cell=None, region=None, dofs=None)
                C_loc = self.kernel.eval(x, cell=(ci,cj), region=(regi, regj),
                                         phi=(phi[0][regi],phi[1][regj]),
                                         dofs=(dofi,dofj))
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

    TODO: Replace with IIForm and IPForm
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
    Description
    -----------
    Representation of sums of bilinear/linear/constant forms as
    matrices/vectors/numbers.

    Attributes
    ----------
    
    Methods
    -------
    

    Parameters
    ----------
    problems : list of Forms
        List of bilinear, linear, or constant Forms

    mesh : Mesh
        Finite element mesh on which to assemble the forms

    submesh_flag : str/double, default=None
        Sub-mesh marker specifying the sub-mesh over which to assemble forms

    n_gauss: (int,int), default=(4,16)
        Number of Gauss nodes used for 1D and 2D quadrature rules in assembly.

    
    Notes
    -----
    During the initialization:
        1. The variational 'problems' are stored in a list. Each problem is
           itself a list of constant, linear, and/or bilinear Forms.
        2. The problems are checked for mutual consistency and compatibility 
           with the mesh. 
        3. The cell and edge quadrature rules are stored as GaussRule objects.
        4. The list for storing various AssembledForms is initialized for every
           problem and every type of form. 
        5. The Dirichlet boundary conditions for each problem are stored a dict
        6. The hanging nodes for each problem are stored in a dict. 
        
        
    See Also
    --------
    AssembledForms, Forms, and GaussRule in assembler
    """
    def __init__(self, problems, mesh, subforest_flag=None, n_gauss=(4,16)):
        """
        Constructor
        """
        #
        # Store mesh and sub-mesh marker
        #
        self.__mesh = mesh
        self.__subforest_flag = subforest_flag

        #
        # Parse "problems" Input
        #
        problem_error = 'Input "problems" should be (i) a Form, (ii) a list '+\
                        'of Forms, or (iii) a list of a list of Forms.'
        if type(problems) is list:
            #
            # Multiple forms (and/or problems)
            #
            if all([isinstance(problem, Form) for problem in problems]):
                #
                # Single problem consisting of multiple forms
                #
                problems = [problems]
            else:
                #
                # Multiple problems
                #
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
            problems = [[problems]]


        # Store info
        self.problems = problems


        #
        # Check whether the problem is consistent and compatible with mesh
        # 
        for problem in problems:
            self.check_problem(problem)
            
        #
        # Initialize Gauss Quadrature Rule
        #
        self.n_gauss_2d = n_gauss[1]
        self.n_gauss_1d = n_gauss[0]
        dim = self.mesh().dim()
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

        #
        # Initialize list for storing assembled forms [iproblem][dim]
        #
        af = []
        for problem in self.problems:
            p_af = [None]*3
            for form in problem:
                dim = form.dim()
                if p_af[dim] is None:
                    # Initialize new assembled form
                    p_af[dim] = AssembledForm(dim)

                # Incorporate form
                p_af[dim].add_form(form)
            af.append(p_af)
        self.__af = af


        #
        # Initialize dictionaries to store Dirichlet boundary conditions and
        # hanging node conditions.
        #
        dirichlet_bc = []
        hanging_nodes = []
        for dummy in problems:
            dirichlet_bc.append({})
            hanging_nodes.append({})

        # Store result
        self.__dirichlet_bc = dirichlet_bc
        self.__hanging_nodes = hanging_nodes


    def check_problem(self, problem):
        """
        Description
        -----------
        Determine whether a problem is (i) compatible with the mesh, i.e. 
        whether the basis functions are defined on a sub-mesh of assembler 
        mesh, and (ii) whether the problem is consistent, i.e. all linear
        (resp. bilinear) forms appearing in list have the same test (resp. 
        trial and test) basis functions. 
        
        The method raises an error when either of these conditions do not hold.
        
        
        Parameters
        ----------
        problem : list of Forms, 
            A problem consists of a list of constant, linear, and/or bilinear
            Forms. 
            
        See Also
        --------
        Forest.is_contained_in

        NOTE: Strictly speaking, bilinear forms 
        """
        mesh = self.mesh()
        flag = self.subforest_flag()
        
        linear_test = None
        bilinear_trial, bilinear_test = None, None
        for form in problem:
            #
            # Check whether forms are compatible with each other  
            # 
            if form.dim() == 1:
                #
                # Linear form
                # 
                if linear_test is None:
                    # 
                    # Add reference test function
                    # 
                    linear_test = form.test
                else:
                    #
                    # Compare against reference test
                    # 
                    assert linear_test.dofhandler() == form.test.dofhandler(),\
                        'The test functions of every linear form in the '+\
                        'problem should have the same dofhandler.'
                        
                    assert linear_test.subforest_flag()==form.test.subforest_flag(),\
                        'The test functions of every linear form in the '+\
                        'problem should be defined on the same sub-mesh.'
                        
            elif form.dim()==2:
                #
                # Bilinear form
                # 
                if bilinear_test is None:
                    #
                    # Add reference test function
                    # 
                    bilinear_test = form.test
                else:
                    #
                    # Compare against reference test 
                    # 
                    assert bilinear_test.dofhandler() == form.test.dofhandler(),\
                        'The test functions of every bilinear form in the ' +\
                        'problem should have the same dofhandler.'
                        
                    assert bilinear_test.subforest_flag()==form.test.subforest_flag(),\
                        'The test functions of every bilinear form in the '+\
                        'problem should be defined on the same sub-mesh.'
                        
                if bilinear_trial is None:
                    #
                    # Add reference trial function
                    # 
                    bilinear_trial = form.trial
                else:
                    #
                    # Compare against reference trial basis
                    #
                    assert bilinear_trial.dofhandler() == form.trial.dofhandler(),\
                        'The trial functions of every bilinear form in the '+\
                        'problem shoule have the same dofhandler.'
                        
                    assert bilinear_trial.subforest_flag()==form.trial.subforest_flag(),\
                        'The trial functions of every bilinear form in the '+\
                        'problem should be defined on the same sub-mesh.'
                    
            #
            # Check whether form is compatible with the assembler's mesh
            # 
            for basis in form.basis():
                #
                # Check that the mesh is the same
                #
                basis_mesh = basis.mesh()
                assert basis.mesh()==self.mesh(), \
                    'The basis and assembler should be defined on the ' +\
                    'same mesh.'

                #
                # Check that the assembler mesh is a refinement of the basis
                # mesh.
                #
                basis_flag = basis.subforest_flag()
                assembler_flag = self.subforest_flag()
                
                assert mesh.cells.is_contained_in(assembler_flag, basis_flag), \
                    'The assembler mesh should be a refinement of the '+\
                    'basis mesh.'
    
    
    def mesh(self):
        """
        Returns
        -------
        Mesh
            Returns the mesh over which the forms are assembled.

        """
        return self.__mesh


    def subforest_flag(self):
        """
        Returns
        -------
        string, int, or double
            Returns a sub-mesh flag over which the form is assembled.

        """
        return self.__subforest_flag


    def assembled_forms(self, i_problem=0):
        return self.__af[i_problem]



    def assemble(self, keep_cellwise_data=False, region_flag=None):
        """
        Description
        -----------
        Assembles constant, linear, and bilinear forms over computational mesh.


        Parameters
        ----------

            problems : list of (list of) forms,
                A list of finite element problems. Each problem is a list of
                constant, linear, and bilinear forms.

            keep_cellwise_data : bool, whether to store the cell-wise default=False

            region_flag : str/int/tuple, default=None, region marker


        Output:

            assembled_forms: list of dictionaries (one for each problem), each of
                which contains:

            A: double coo_matrix, system matrix determined by bilinear forms and
                boundary conditions.

            b: double, right hand side vector determined by linear forms and
                boundary conditions.

        Note: If problems contain one integral form (IPFORM), then the assembly
            uses a double loop of cells. This is inefficient if problems are mixed.
        """
        t_shape_info = 0
        #t_gauss_rules = 0
        t_shape_eval = 0
        t_form_eval = 0
        #t_get_node_address = 0
        #t_af_update = 0
        #t_af_consolidate = 0
        #t_reference_map = 0


        #
        # Assemble forms over mesh cells
        #
        sf = self.subforest_flag()
        cells = self.mesh().cells.get_leaves(subforest_flag=sf, flag=region_flag)
        for ci in cells:
            #
            # Compute shape functions on cell
            #
            tic = time.time()
            xi_g, wi_g, phii, dofsi = self.shape_eval(ci)
            t_shape_eval += time.time()-tic

            #
            # Assemble local forms and assign to global dofs
            #
            for i_problem, problem in enumerate(self.problems):
                #
                # Loop over problems
                #
                for form in problem:
                    #
                    # Loop over forms
                    #

                    # Get form dimension (0-constant, 1-linear, 2-bilinear)
                    dim = form.dim()

                    # Get assembled form
                    aform = self.assembled_forms(i_problem)[dim]
                    #
                    # Evaluate form
                    #
                    if not isinstance(form, IPForm):
                        #
                        # Not an integral form
                        #

                        # Evaluate local form
                        tic = time.time()
                        form_loc = form.eval(ci, xi_g, wi_g, phii, dofsi)
                        t_form_eval += time.time()-tic

                        # Uppdate assembled form cellwise
                        if dim == 0:
                            #
                            # Constant form
                            #
                            aform.update_cellwise(ci, form_loc)

                        elif dim == 1:
                            #
                            # Linear form
                            #
                            dofs = [form.test.dofs(ci)]
                            aform.update_cellwise(ci, form_loc, dofs=dofs)

                        elif dim == 2:
                            #
                            # Bilinear form
                            #

                            # Trial dofs
                            dofs_trl = form.trial.dofs(ci)

                            # Test dofs
                            if isinstance(form, IIForm):
                                # Interpolatory Integral forms use all dofs
                                dofs_tst = form.test.dofs(None)
                            else:
                                dofs_tst = form.test.dofs(ci)

                            # Update assembled form
                            dofs = [dofs_tst, dofs_trl]
                            aform.update_cellwise(ci, form_loc, dofs=dofs)


                    if isinstance(form, IPForm):
                        #
                        # Form is Double Integral
                        #
                        for cj in cells:
                            #
                            # Compute shape function on cell
                            #
                            xj_g, wj_g, phij, dofsj = self.shape_eval(cj)

                            #
                            # Evaluate integral form
                            #
                            form_loc = form.eval((ci,cj), (xi_g,xj_g), \
                                                 (wi_g,wj_g), (phii,phij),\
                                                 (dofsi,dofsj))

                            # Test and trial dofs
                            dofs_tst = form.test.dofs(ci)
                            dofs_trl = form.trial.dofs(cj)

                            #
                            # Update Assembled Form
                            #
                            aform.update_cellwise(ci, form_loc,
                                                  dofs = [dofs_tst, dofs_trl])

                            #
                            # Special efficiency when kernel is symmetric
                            #
                            if form.kernel.is_symmetric():
                                if ci!=cj:
                                    #
                                    # Symmetric kernel, store the transpose
                                    #
                                    aform.update_cellwise(ci, form_loc.T,
                                                          dofs = [dofs_trl, dofs_tst])
                                else:
                                    #
                                    # Symmetric forms assembled over subtriangular block
                                    #
                                    break

            #
            # Aggregate cellwise information
            #
            for i_problem in range(self.n_problems()):
                # Get Dirichlet BC's
                dir_bc = self.get_dirichlet(i_problem)

                # Get hanging nodes
                hng = self.get_hanging_nodes(i_problem)

                for dim in range(3):
                    aform = self.assembled_forms(i_problem)[dim]
                    if aform is not None:
                        #
                        # Update aggregate
                        #
                        aform.distribute(ci, dir_bc=dir_bc, hng=hng)
                        #
                        # Delete cellwise information
                        #
                        if not keep_cellwise_data:
                            aform.clear_cellwise_data(ci)

        #
        # Consolidate arrays
        #
        for i_problem in range(self.n_problems()):
            for dim in range(3):
                aform = self.assembled_forms(i_problem)[dim]
                if aform is not None:
                    aform.consolidate()
        """
        for i_problem in range(len(self.problems)):
            for form_type in self.af()[i_problem].keys():
                #
                # Iterate over assembled forms
                #
                af = self.af[i_problem][form_type]

                #
                # Consolidate assembly
                #
                tic = time.time()
                af.consolidate(clear_cell_data=clear_cell_data)
                t_af_consolidate += time.time()-tic
                print('t_consolidate', t_af_consolidate)
        print('Timings')
        print('Shape infor',t_shape_info)
        print('Shape Eval', t_shape_eval)
        print('Form Eval', t_form_eval)
        print('Get node address', t_get_node_address)
        print('AF update', t_af_update)
        print('AF consolidate', t_af_consolidate)
        """
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

    def get_matrix(self, i_problem=0, i_sample=0):
        """
        Return the sparse matrix representation of the bilinear form of
        specified sample of specified problem.


        Inputs:

            i_problem: int [0], problem index

            i_sample: int, [0], sample index


        Output:

            A: double, sparse array representing bilinear form
        """
        aform = self.__af[i_problem][2]
        return aform.aggregate_data()['array'][i_sample]


    def get_vector(self, i_problem=0, i_sample=0):
        """
        Return the vector representation of the linear form of the specified
        sample of specified problem.


        Inputs:

            i_problem: int [0], problem index

            i_sample: int [0], sample index


        Output:

            b: double, vector representing linear form
        """
        assembled_form = self.__af[i_problem][1]
        data = assembled_form.aggregate_data()['array']
        if type(i_sample) is int:
            #
            # Single sample
            #
            return data[i_sample]
        else:
            #
            # Multiple samples
            #
            assert type(i_sample) is list, \
            'Input "i_sample" should be a (list of) integer(s).'

            return np.array([data[i] for i in i_sample])


    def get_scalar(self, i_problem=0, i_sample=0):
        """
        Return the scalar representation of the constant form of the specified
        sample of the specified problem


        Inputs:

            i_problem: int [0], problem index

            i_sample: int [0], sample index, 

        Output:

            c: double, scalar representing constant form
        """
        aform = self.__af[i_problem][0]
        return aform.aggregate_data()['array'][i_sample]


    def get_dofs(self, dof_type, i_problem=0):
        """
        Get dofs for problem, divided into 'interior', 'dirichlet', and
        'hanging_nodes'.

        Inputs:

            i_problem: int, problem index
        """
        if dof_type == 'dirichlet':
            #
            # DOFs of Dirichlet Boundaries
            #
            dir_bc = self.get_dirichlet(i_problem=i_problem)
            dir_dofs = list(dir_bc.keys())
            return dir_dofs

        elif dof_type == 'hanging_nodes':
            #
            # DOFs of hanging nodes
            #
            hng = self.get_hanging_nodes(i_problem=i_problem)
            hng_dofs = list(hng.keys())
            return hng_dofs

        elif dof_type == 'interior':
            #
            # Interior dofs
            #
            bform = self.assembled_forms(i_problem=i_problem)[2]
            int_dofs = bform.aggregate_data()['udofs'][0]
            return int_dofs


    def assembled_bnd(self, i_problem=0, i_sample=0):
        """
        Returns
        """
        aform = self.__af[i_problem][2]
        if aform is not None:
            return aform.dirichlet_correction()['array'][i_sample]


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
                n_kernel_sample = form.kernel.n_subsample()
                #
                # Consider only forms of given type
                #
                if n_kernel_sample is not None:
                    #
                    # Sampling in effect
                    #
                    if n_samples is None:
                        #
                        # New trivial sample size
                        #
                        n_samples = n_kernel_sample
                    else:
                        #
                        # There exists a nontrivial sample size
                        #
                        if n_kernel_sample > 1:
                            #
                            # Kernel contains more than one sample.
                            # Check for consistency
                            #
                            assert n_samples == n_kernel_sample,\
                        '    Inconsistent sample sizes in kernels'
        return n_samples


    def n_problems(self):
        """
        Returns the number of problems
        """
        return len(self.problems)


    def shape_info(self, cell):
        """
        Determine what shape functions must be computed and over what region
        within a particular cell.

        Inputs:

            cell: cell over which to assemble


        Output:

            info: (nested) dictionary, whose entries info[region][element]
                consist of Basis functions representing the shape functions
                associated with the cell.
        """
        info = {}
        for problem in self.problems:
            for form in problem:
                #
                # Form basis functions on same mesh/flag
                #
                basis = form.basis()
                for region in form.regions(cell):
                    #
                    # Record form's regions
                    #
                    if not region in info.keys():
                        info[region] = set()
                    info[region].update(basis)
        return info


    def quad_rules(self, shape_info, cell):
        """
        Description
        -----------
        Compute the quadrature nodes and weights on the reference region for 
        all cases specified by the shape_info dictionary. 


        Parameters
        ----------
        shape_info: dict, 
            Sets of basis functions required to be evaluated over each 
            sub-region of the cell (see Assembler.shape_info) 
            
        cell : Cell, 
            Cell over which the problems are assembled.


        Returns
        -------
        x: dict, 
            Dictionary of Gauss nodes on reference cell, indexed by cell's 
            subregions and the basis' subforest_flag (magnification)

        w: dict, 
            Dictionary of Gauss weights on reference cell, indexed by cell's 
            subregions and the basis' subforest_flag (magnification)
            
            
        Notes
        -----
        
        """
        x, w = {}, {}
        for region in shape_info.keys():
            #
            # Initialize dictionary for the given region
            #  
            x[region] = {}
            w[region] = {}
            
            if isinstance(region, Interval) or \
               isinstance(region, QuadCell):
                #
                # Integral over Cell
                # 
                x_ref = self.cell_rule.nodes()
                w_ref = self.cell_rule.weights()
                
            elif isinstance(region, HalfEdge):
                #
                # Integral over edge 
                # 
                x_ref = self.edge_rule.nodes()
                w_ref = self.edge_rule.weights() 
            
                # Initialize indicator for  
                map_cell = False
                
                if len(shape_info[region])>0:
                    #
                    # Need shape evaluations on half-edge: map to reference cell
                    # 
                    
                    # Get reference cell from single basis function
                    basis = list(shape_info[region])[0]
                    ref_cell = basis.dofhandler().element.reference_cell()

                    # Determine equivalent half-edge on reference element
                    i_he = cell.get_half_edges().index(region)
                    ref_he = ref_cell.get_half_edge(i_he)

                    # Get 2D reference nodes
                    b,h = convert_to_array(ref_he.get_vertices())
                    x_ref = np.array([b[i]+x_ref*(h[i]-b[i]) for i in range(2)]).T
                    
                    # Indicate to map entire cell instead of  
                    
            for basis in shape_info[region]:
                sf = basis.subforest_flag()
                if not sf in x[region]:
                    pass
                                    
        xg, wg, mg = {}, {}, {}
        for region in shape_info.keys():

            #
            # Determine whether shape derivatives will be needed
            #
            if any([basis.derivative()[0]==1 for basis in shape_info[region]]):
                #
                # Need Jacobian of Inverse Mapping
                #
                jac_p2r = True
            else:
                jac_p2r = False

            if any([basis.derivative()[0]==2 for basis in shape_info[region]]):
                #
                # Need Hessian of inverse mapping
                #
                hess_p2r = True
            else:
                hess_p2r = False

            #
            # Map quadrature rule to entity (cell/halfedge)
            #
            if isinstance(region, Interval):
                #
                # Interval
                #
                xg[region], wg[region], mg[region] = \
                    self.cell_rule.mapped_rule(region, jac_p2r=jac_p2r,
                                               hess_p2r=hess_p2r)

            elif isinstance(region, HalfEdge):
                #
                # HalfEdge
                #
                xg[region], wg[region], mg[region] = \
                    self.edge_rule.mapped_rule(region, jac_p2r=jac_p2r,
                                               hess_p2r=hess_p2r)

            elif isinstance(region, QuadCell):
                #
                # Quadrilateral
                #
                xg[region], wg[region], mg[region] = \
                    self.cell_rule.mapped_rule(region, jac_p2r=jac_p2r,
                                               hess_p2r=hess_p2r)

            elif isinstance(region, Vertex):
                #
                # Vertex
                #
                xg[region], wg[region] = convert_to_array(region.coordinates()), 1
            else:
                raise Exception('Only Intervals, HalfEdges, Vertices, & '+\
                                'QuadCells supported.')
        #
        # Return results
        #
        if any([hess_p2r,jac_p2r]):
            return xg, wg, mg
        else:
            return xg, wg


    def shape_eval(self, cell):
        """
        Description
        -----------
        (i) Map reference quadrature rule and (ii) evaluate the element shape
        functions (and their derivatives) at the mapped quadrature points in
        each region specified by "shape_info".


        Inputs:

            shape_info: dictionary, whose keys are the integration regions
                (QuadCell, Interval, or HalfEdge) over which to integrate and
                whose values are the basis functions to be integrated.

            cell: cell over which to integrate


        Output:

            xg: dictionary (indexed by regions), of mapped quadrature nodes.

            wg: dictionary (indexed by regions), of mapped quadrature weights.

            phi: dictionary phi[region][basis] of shape functions evaluated at
                the quadrature nodes.

        TODO: A big chunk can be moved to GaussRule, map_rule
        """
        # Initialize
        xg, wg, phi, dofs = {}, {}, {}, {}

        # Get information for computing
        shape_info = self.shape_info(cell)
        
        # Iterate over integration regions
        for region in shape_info.keys():
            #
            # Get global dof numbers for the region
            #
            # Initialize degrees of freedom
            dofs[region] = {}
            for basis in shape_info[region]:
                #
                # Get region dofs for each basis
                #
                dofs[region][basis] = basis.dofs(cell)

            #
            # Determine whether shape derivatives will be needed for region
            #
            if any([basis.derivative()[0] in [1,2] for basis in shape_info[region]]):
                #
                # Need Jacobian of inverse mapping
                #
                jac_p2r = True
            else:
                jac_p2r = False

            if any([basis.derivative()[0]==2 for basis in shape_info[region]]):
                #
                # Need Hessian of inverse mapping
                #
                hess_p2r = True
            else:
                hess_p2r = False

            #
            # Map reference quadrature nodes to physical ones
            #
            if isinstance(region, Interval):
                #
                # Interval
                #
                # Check compatiblity
                assert self.mesh().dim()==1, 'Interval requires a 1D rule.'

                # Get reference nodes and weights
                x_ref = self.cell_rule.nodes()
                w_ref = self.cell_rule.weights()

                # Map to physical region
                xg[region], mg = \
                    region.reference_map(x_ref, jac_r2p=True,
                                         jac_p2r=jac_p2r,
                                         hess_p2r=hess_p2r)

                # Get jacobian of forward mapping
                jac = mg['jac_r2p']

                # Modify the quadrature weights
                wg[region] = w_ref*np.array(jac)

            elif isinstance(region, HalfEdge):
                #
                # Edge
                #

                # Reference nodes and weights
                r = self.edge_rule.nodes()
                w_ref = self.edge_rule.weights()

                # Map from interval to physical region
                xg[region], mg = region.reference_map(r, jac_r2p=True,
                                         jac_p2r=jac_p2r, hess_p2r=hess_p2r)

                # Get jaobian of forward mapping
                jac = mg['jac_r2p']

                # Modify the quadrature weights
                wg[region] = w_ref*np.array(np.linalg.norm(jac[0]))

                # To evaluate phi (and derivatives), map 1D reference nodes
                # to 2D ones and record jacobians/hessians
                if len(shape_info[region])>0:
                    # There are shape functions associated with region

                    # Get reference cell from single basis function
                    basis = list(shape_info[region])[0]
                    ref_cell = basis.dofhandler().element.reference_cell()

                    # Determine equivalent Half-edge on reference element
                    i_he = cell.get_half_edges().index(region)
                    ref_he = ref_cell.get_half_edge(i_he)

                    # Get 2D reference nodes
                    b,h = convert_to_array(ref_he.get_vertices())
                    x_ref = np.array([b[i]+r*(h[i]-b[i]) for i in range(2)]).T

                    # Map 2D reference point to physical cell
                    xg[region], mg = cell.reference_map(x_ref, jac_r2p=False,
                                                        jac_p2r=True,
                                                        hess_p2r=True)

            elif isinstance(region, QuadCell):
                #
                # Quadrilateral
                #
                # Check compatibility
                assert self.mesh().dim()==2, 'QuadCell requires 2D rule.'

                # Get reference nodes and weights
                x_ref = self.cell_rule.nodes()
                w_ref = self.cell_rule.weights()

                # Map to physical region
                xg[region], mg = \
                    region.reference_map(x_ref, jac_r2p=True,
                                         jac_p2r=jac_p2r,
                                         hess_p2r=hess_p2r)

                # Get Jacobian of forward mapping
                jac = mg['jac_r2p']

                # Modify quadrature weights
                wg[region] = w_ref*np.array([np.linalg.det(j) for j in jac])

            elif isinstance(region, Vertex):
                #
                # Vertex (special case)
                #
                xg[region], wg[region] = convert_to_array(region.coordinates()), 1

                #
                # Determine reference vertex corresponding to Vertex region
                #
                basis = list(shape_info[region])[0]
                ref_cell = basis.dofhandler().element.reference_cell()

                # Determine equivalent Vertex on reference element
                i_v = cell.get_vertices().index(region)
                v = ref_cell.get_vertex(i_v)
                x_ref = convert_to_array(v, dim=v.dim())

                # Map to physical
                xg[region], mg = cell.reference_map(x_ref, jac_r2p=True,
                                                    jac_p2r=jac_p2r,
                                                    hess_p2r=hess_p2r)
                wg[region] = 1
            else:
                raise Exception('Only Intervals, HalfEdges, Vertices & '+\
                                'QuadCells supported')

            #
            # Evaluate (derivatives of) basis functions at the quadrature nodes
            #
            phi[region] = {}
            for basis in shape_info[region]:
                #
                # Iterate over basis functions
                #
                if basis not in phi[region]:
                    #
                    # Evaluate basis functions over regions
                    #
                    element = basis.dofhandler().element
                    D = basis.derivative()
                    jac_p2r = mg['jac_p2r'] if D[0] in [1,2] else None
                    hess_p2r = mg['hess_p2r'] if D[0]==2 else None

                    p = element.shape(x_ref=x_ref, derivatives=D, cell=cell,
                                      jac_p2r=jac_p2r, hess_p2r=hess_p2r)
                    phi[region][basis] = p

        # Return mapped quadrature nodes, weights, and shape functions
        return xg, wg, phi, dofs


    def add_dirichlet(self, dir_marker, dir_fn=0, on_bnd=True, i_problem=0):
        """
        Add a Dirichlet condition to a problem, i.e. a set of dofs and vals
        corresponding to dirichlet conditions.

        Inputs:


            dir_marker: str/int flag to identify dirichlet halfedges

            i_problem: int, problem index

            dir_fn: Map/scalar, defining the Dirichlet boundary conditions.

            on_bnd: bool, True if function values are prescribed on boundary.


        Outputs:

            None


        Notes:

        To maintain the dimensions of the matrix, the trial and test function
        spaces must be the same, i.e. it must be a Galerkin approximation.

        Specifying the Dirichlet conditions this way is necessary if there
        are hanging nodes, since a Dirichlet node may be a supporting node for
        one of the hanging nodes.


        Modified Attributes:

            __dirichlet_bc: (i_problem indexed) list of dictionaries, containing
                dofs and values of corresponding to dirichlet nodes.
        """
        #
        # Extract dofhandler information from trial function
        #
        bilinear = self.assembled_forms(i_problem)[2]
        trial = bilinear.basis()[1]
        dh = trial.dofhandler()
        sf = trial.subforest_flag()

        #
        # Get Dofs Associated with Dirichlet boundary
        #
        if dh.mesh.dim()==1:
            #
            # One dimensional mesh
            #
            dir_dofs = dh.get_region_dofs(entity_type='vertex', \
                                          entity_flag=dir_marker,\
                                          interior=False, \
                                          on_boundary=on_bnd,\
                                          subforest_flag=sf)
        elif dh.mesh.dim()==2:
            #
            # Two dimensional mesh
            #
            dir_dofs = dh.get_region_dofs(entity_type='half_edge',
                                          entity_flag=dir_marker,
                                          interior=False,
                                          on_boundary=on_bnd, \
                                          subforest_flag=sf)
            #print('The Dirichlet Dofs are', dir_dofs)
        # Number of dirichlet dofs
        n_dirichlet = len(dir_dofs)

        #
        # Evaluate dirichlet function at vertices associated with dirichlet dofs
        #
        if isinstance(dir_fn, numbers.Number):
            #
            # Dirichlet function is constant
            #
            dir_vals = dir_fn*np.ones((n_dirichlet,1))

        elif isinstance(dir_fn, Nodal) and dir_fn.basis().same_dofs(trial):
            #
            # Nodal function whose dofs coincide with problem dofs
            #
            idx = dir_fn.dof2idx(dir_dofs)
            dir_vals = dir_fn.data()[idx,:]
        else:
            #
            # Evaluate the function explicitly at the dirichlet vertices
            #
            dir_verts = dh.get_dof_vertices(dir_dofs)
            x_dir = convert_to_array(dir_verts)
            dir_vals = dir_fn.eval(x_dir)

        #
        # Store Dirichlet Dofs and Values
        #
        dir_bc = self.__dirichlet_bc[i_problem]
        for dof,vals in zip(dir_dofs,dir_vals):
            dir_bc[dof] = vals


        # Store result
        self.__dirichlet_bc[i_problem] = dir_bc


    def get_dirichlet(self, i_problem=0, asdict=True):
        """
        Return dirichlet boundary conditions as the dict {dofs:vals}
        """
        if asdict:
            return self.__dirichlet_bc[i_problem]
        else:
            dir_dofs = list(self.__dirichlet_bc[i_problem])
            dir_vals = np.array(list(self.__dirichlet_bc[i_problem].values()))
            return dir_dofs, dir_vals


    def add_hanging_nodes(self, i_problem=0):
        """
        Add hanging nodes to a problem, as computed by the dofhandler.
        """
        # Get bilinear form
        biform = self.assembled_forms(i_problem)[2]
        assert biform is not None, "Problem has no bilinear form."

        # Get trial function of bilinear form
        phi = biform.basis()[1]
        assert phi.same_dofs(biform.basis()[0]), \
            "Trial and test functions should have the same dofs."

        # Set hanging nodes
        self.__hanging_nodes[i_problem] \
            = phi.dofhandler().get_hanging_nodes(subforest_flag=self.subforest_flag())


    def get_hanging_nodes(self, i_problem=0):
        """
        Return hanging nodes
        """
        return self.__hanging_nodes[i_problem]


    def hanging_node_matrix(self, i_problem=0):
        """
        Return matrix used to reconstruct hanging node values from support.
        """
        # Get dof information
        biform = self.assembled_forms(i_problem)[2]  # bilinear form
        test = biform.basis()[0]  # test function basis (same as trial)
        n_dofs = test.n_dofs()

        # Get hanging nodes
        hng = self.__hanging_nodes[i_problem]

        # Initialize r,c,v triplets
        rows = []
        cols = []
        vals = []
        for h_dof, supp in hng.items():
            for s_dof, s_coef in zip(*supp):
                # Update rows (hanging nodes)
                rows.append(test.d2i(h_dof))

                # Update cols (supporting nodes)
                cols.append(test.d2i(s_dof))

                # Update vals (supporting coefficients)
                vals.append(s_coef)

        C = sparse.coo_matrix((vals,(rows,cols)),shape=(n_dofs,n_dofs))
        return C


    def solve(self, i_problem=0, i_matrix=0, i_vector=0):
        """
        Solve the assembled problem

        Inputs:

            i_problem: int, problem index

            i_matrix: int, matrix (sample) index

            i_vector: int, vector (sample) index (can be a list)


        Output:

            u: double, (n_dofs, n_samples) solution array.


        TODO: Sort out the sampling - multiple right hand sides
        """
        #
        # Determine problem's number of dofs
        #

        # From linear form
        phi_lin = self.assembled_forms(i_problem)[1].basis()[0]
        assert phi_lin is not None, 'Missing assembled linear form.'
        n_dofs = phi_lin.n_dofs()

        for phi_bil in self.assembled_forms(i_problem)[2].basis():
            assert phi_bil.same_dofs(phi_lin), \
            'Linear and bilinear forms should have the same basis.'


        # Get system matrix
        A = self.get_matrix(i_problem=i_problem, i_sample=i_matrix).tocsc()

        # System vectors
        b = self.get_vector(i_problem=i_problem, i_sample=i_vector)

        # Assembled Dirichlet BC
        x0 = self.assembled_bnd(i_problem=i_problem, i_sample=i_matrix)

        #
        # Initialize solution array
        #
        u = np.zeros(n_dofs)

        #
        # Solve linear system on interior dofs
        #
        int_dofs = self.get_dofs('interior', i_problem=i_problem)
        u[int_dofs] = spla.spsolve(A,b-x0)

        #
        # Resolve Dirichlet conditions
        #
        dir_dofs, dir_vals = self.get_dirichlet(i_problem=i_problem, asdict=False)
        u[dir_dofs] = dir_vals[:,0]

        # Resolve hanging nodes
        C = self.hanging_node_matrix(i_problem=i_problem)
        u += C.dot(u)

        return u     



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
    def __init__(self, dim):
        """
        Constructor

        Inputs:

            dim: int, dimension of the forms to be included (0,1,2)
        """
        #
        # Initialize dimensions, basis, and sample size
        #
        self.__dim = dim
        self.__basis = None
        self.__n_samples = 1

        #
        # Initialize cellwise- and aggregate data
        #
        self.__cellwise_data = {}
        self.__aggregate_data = {'dofs':[[] for dummy in range(dim)],
                                 'udofs': [],
                                 'vals': [],
                                 'array': None}
        if dim == 2:
            #
            # Bilinear form
            #
            self.__bnd = {'dofs': [], 'vals': []}
            self.__hng = {'dofs': [[],[]], 'vals': []}



    def dim(self):
        """
        Return dimension
        """
        return self.__dim


    def basis(self):
        """
        Return basis vectors
        """
        return self.__basis


    def n_dofs(self):
        """
        Return number of dofs for problem
        """
        return [basis.n_dofs() for basis in self.basis()]


    def n_samples(self):
        """
        Return sample size
        """
        return self.__n_samples


    def cellwise_data(self):
        """
        Return cellwise data
        """
        return self.__cellwise_data


    def aggregate_data(self):
        """
        Return aggregate data
        """
        return self.__aggregate_data


    def dirichlet_correction(self):
        """
        Return the dirichlet correction term
        """
        return self.__bnd


    def add_form(self, form):
        """
        Add form to aggregate form.

        Inputs:

            form: Form, to be added
        """
        # Check that input is a Form
        assert isinstance(form, Form), 'Input "form" should be a "Form" object'

        # Check dimension
        assert form.dim()==self.dim(), 'Input "form" has incompatible dim.'

        #
        # Get basis from form
        #
        if form.type == 'bilinear':
            form_basis = [form.test, form.trial]
        elif form.type == 'linear':
            form_basis = [form.test]
        elif form.type == 'constant':
            form_basis = None

        #
        # Update/compare aggregate's basis
        #
        if self.basis() is None:
            #
            # Store basis
            #
            self.__basis = form_basis
        else:
            #
            # Basis functions should have the same dofs
            #
            for basis, fbasis in zip(self.basis(), form_basis):
                assert basis.same_dofs(fbasis), \
                'Basis functions have incompatible dofs'

        #
        # Update/compare sample size
        #
        n_smpl = self.n_samples()
        n_smpl_form = form.kernel.n_samples()
        
 
        if n_smpl_form > 1:
            if n_smpl==1:
                #
                # Update sample size of aggregate form
                #
                self.__n_samples = n_smpl_form
            else:
                #
                # Check that sample size is the same
                #
                assert n_smpl==n_smpl_form, 'Sample sizes incompatible.'



    def update_cellwise(self, cell, vals, dofs=None):
        """
        Update cellwise assembled form
        """
        n_samples = self.n_samples()
        dim = self.dim()
        data = self.__cellwise_data
        if cell not in data:
            #
            # Initialize cellwise data
            #
            data[cell] = {'dofs': [[] for i in range(dim)],
                          'vals': []}
        #
        # Postprocess dofs and vals for multilinear forms
        #
        if dim > 1:
            #
            # Postprocess dofs-vals for bilinear forms
            #

            # Form grid of dofs
            R, C = np.meshgrid(*dofs)
            dofs[0] = list(R.ravel())
            dofs[1] = list(C.ravel())

            n_entries = len(dofs[0])
            
            #if vals.shape[1]==1:
            #    print(vals)
            
            if n_samples > 1:
                if vals.shape[-1]==1:
                    #
                    # Single sample: Copy it 
                    # 
                    vals = np.tile(vals, (1,n_samples))
                
            vals = vals.reshape((n_entries,n_samples), order='F')
            
            """
            if n_samples > 1:
                if len(vals.shape) == 1:
                    #
                    # Deterministic form -> copy it
                    # 
                    vals = np.tile(vals[:,np.newaxis], (1,n_samples))
                else:
                    #
                    # Sampled form
                    # 
            """       


        # Update dofs
        for i in range(dim):
            # Update ith set of dofs
            data[cell]['dofs'][i].extend(dofs[i])

        # Update vals
        data[cell]['vals'].extend(vals)

        self.__cellwise_data = data


    def distribute(self, cell, dir_bc={}, hng={}, clear_cellwise_data=False):
        """
        Update the aggregate assembled form data, incorporating constraints
        arising from hanging nodes and Dirichlet boundary conditions.

        For a concrete example, consider the following system

            a11 a12 a13 a14   u1     b1
            a21 a22 a23 a24   u2  =  b2
            a31 a32 a33 a34   u3     b3
            a41 a42 a43 a44   u4     b4

        1. Suppose Dirichlet conditions u2=g2 and u4=g4 are prescribed.
        The system is converted to

            a11 a13  u1  =  b1 - a12*g2 - a14*g4
            a31 a33  u3     b3 - a32*g2 - a34*g4

        The solution [u1,u3]^T of this system can then be enlarged with the
        dirichlet boundary values g2 and g4.

        2. Hanging nodes arise as a result of local mesh refinement. In
        particular when a dof-vertex of one cell is not a dof-vertex of
        its neighboring cell. We resolve hanging nodes by enforcing
        continuity accross edges, i.e. requiring that the node value of a
        function at a hanging node can be computed by evaluating a linear
        combination of basis functions centered at a set of supporting nodes
        in the coarse neighbor element.

        Use DofHandler.set_hanging_nodes() and DofHandler.get_hanging_nodes
        to determine the supporting dofs and coefficients for a given
        mesh-element pair.

        To incorporate the hanging nodes constraint into a system, we need to
        replace both the test and trial functions at the hanging node by linear
        combinations of its supporting basis. We therefore have to

        (i) Distribute the hanging node columns of the system matrix A amongst
            its supporting columns (trial)

                                        AND/OR

        (ii) Distribute the equation associated with the hanging node to the
            equations associated with supporting dofs (test).

        (iii) In each case, if a supporting dof is a dirichlet dof, should be
            dealt with accordingly (either ignored - row - or moved to the rhs
            - col -).

        Suppose u2 in the above is a hanging node supported by u1 and u4, with
        coefficients c1 and c4, i.e.

            u2 = c1*u1 + c4*u4

        Then the coefficients aij i,j in {1,2} are modified to


            aaij = aij + ci*ai2 + cj*a2j + ci*cj*a22

        If, in addition, uj=gj is a dirichlet node, then the rhs is modified
        by subtracting aaij*gj


        Inputs:

            cell: Cell, on which constraints are incorporated

            dir_bc: dict, dirichlet-dof-indexed dictionary whose entries
                are the function values at the dirichlet nodes.

            hng: dict, hanging-node-dof-indexed dictionary with entries
                consisting of supporting dofs and coefficients


            clear_cellwise_data: bool, whether or not to delete dof-value data
                stored separately for current cell.
        """
        #
        # Get cellwise data
        #
        if clear_cellwise_data:
            cell_data = self.__cellwise_data[cell]
        else:
            cell_data = self.__cellwise_data[cell].copy()

        # Aggregate data
        data = self.__aggregate_data
        vals = cell_data['vals']

        dim = self.dim()
        if dim == 0:
            #
            # Constant form
            #
            data['vals'].extend(vals)

        elif dim == 1:
            #
            # Linear form
            #
            rows, = cell_data['dofs']

            while len(rows)>0:
                r,v = rows.pop(), vals.pop()

                if r in hng:
                    #
                    # Hanging Node -> distribute to supporting rows
                    #
                    for supp, coef in zip(*hng[r]):
                        #
                        # Add supporting dofs and modified vals to list
                        #
                        rows.append(supp)
                        vals.append(coef*v)
                elif r not in dir_bc:
                    #
                    # Interior Node
                    #
                    data['dofs'][0].append(r)
                    data['vals'].append(v)

        elif dim == 2:
            #
            # Bilinear form
            #
            rows, cols = cell_data['dofs']

            while len(rows)>0:
                r, c, v = rows.pop(), cols.pop(), vals.pop()

                if r in hng:
                    #
                    # Hanging node -> distribute to supporting rows
                    #
                    for supp, coef in zip(*hng[r]):
                        #
                        # Add supporting dofs and modified vals to list
                        #
                        rows.append(supp)
                        cols.append(c)
                        vals.append(v*coef)

                elif r not in dir_bc:
                    #
                    # Interior row
                    #
                    if c in hng:
                        #
                        # Column hanging node -> distribute to supporting cols
                        #
                        for supp, coef in zip(*hng[c]):
                            #
                            # Add supporting dofs and modified vals to todo list
                            #
                            rows.append(r)
                            cols.append(supp)
                            vals.append(v*coef)

                    elif c in dir_bc:
                        #
                        # Dirichlet column -> update bnd function
                        #
                        self.__bnd['dofs'].append(r)
                        self.__bnd['vals'].append(v*dir_bc[c])

                    else:
                        #
                        # Interior column
                        #
                        data['dofs'][0].append(r)
                        data['dofs'][1].append(c)
                        data['vals'].append(v)

        if clear_cellwise_data:
            self.clear_cellwise_data(cell)


    def clear_cellwise_data(self, cell):
        """
        Remove dof-val data stored at cellwise level
        """
        if cell in self.__cellwise_data:
            self.__cellwise_data.pop(cell)


    def consolidate(self):
        """
        Postprocess assembled form to make it amenable to linear algebra
        operations. This includes renumbering equations that involve only a
        subset of the degrees of freedom.

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
        dim = self.dim()
        dofs = self.__aggregate_data['dofs']

        #
        # Determine unique list of dofs associated with asmbld_form
        #
        udofs = []  # unique dofs
        n_idx = []  # number
        dof2idx = []  # dof-to-index mapping
        for i in range(dim):
            #
            # Get unique dofs from data
            #
            udofs.append(list(set(dofs[i])))

            #
            # Number of index vectors
            #
            n_idx.append(len(udofs[i]))

            #
            # Dof-to-index mapping for interior dofs
            #
            dof2idx.append(np.zeros(udofs[i][-1]+1, dtype=int))
            dof2idx[i][udofs[i]] = np.arange(n_idx[i])

        #
        # Store values in (n, n_samples) array
        #
        vals = self.__aggregate_data['vals']
        vals = np.array(vals)

        n_samples = self.n_samples()
        if dim == 0:
            #
            # Constant (scalar)
            #

            # Sum up all entries
            c = np.sum(vals, axis=0)

            # Store result
            self.__aggregate_data['array'] = c

        elif dim == 1:
            #
            # Linear (vector)
            #
            rows = dof2idx[0][dofs[0]]
            n_rows = n_idx[0]

            b = []
            for i in range(n_samples):
                b.append(np.bincount(rows,vals[:,i],n_rows))
            self.__aggregate_data['array'] = b
            self.__aggregate_data['udofs'] = udofs

        elif dim == 2:
            #
            # Bilinear (matrix)
            #

            # Dirichlet term
            x0_dofs = dof2idx[0][self.__bnd['dofs']]
            x0_vals = np.array(self.__bnd['vals'])

            # Matrix rows & cols
            rows = dof2idx[0][dofs[0]]
            cols = dof2idx[1][dofs[1]]

            # Dimensions
            n_rows, n_cols = n_idx

            A = []
            x0 = []
            for i in range(n_samples):
                # Form sparse assembled matrix
                Ai = sparse.coo_matrix((vals[:,i],(rows, cols)),
                                       shape=(n_rows,n_cols))
                A.append(Ai)

                # Form
                if len(x0_dofs)>0:
                    x0.append(np.bincount(x0_dofs, x0_vals[:,i],n_rows))

            # Store result
            self.__aggregate_data['array'] = A
            self.__aggregate_data['udofs'] = udofs
            self.__bnd['array'] = x0
