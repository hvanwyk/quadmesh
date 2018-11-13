import numpy as np
from scipy import sparse, linalg
import scipy.sparse.linalg as spla
import numbers
from mesh import Vertex, HalfEdge, Mesh2D
from mesh import Cell, QuadCell, Interval
from mesh import RHalfEdge, RInterval, RQuadCell
from mesh import convert_to_array
from bisect import bisect_left, bisect_right
from itertools import count


def parse_derivative_info(dstring):
        """
        Input:
        
            string: string of the form *,*x,*y,*xx, *xy, *yx, *yy, where * 
                stands for any letter.
            
        Output: 
        
            tuple, encoding derivative information  
        """
        s = list(dstring)
        if len(s) == 1:
            #
            # No derivatives
            # 
            return (0,)
        elif len(s) == 2:
            #
            # First order derivative
            # 
            if s[1] == 'x':
                # wrt x
                return (1,0)
            elif s[1] == 'y':
                # wrt y
                return (1,1)
            else:
                raise Exception('Only two variables allowed.')
        elif len(s) == 3:
            #
            # Second derivative
            # 
            if s[1]=='x' and s[2]=='x':
                # f_xx
                return (2,0,0)
            elif s[1]=='x' and s[2]=='y':
                # f_xy
                return (2,0,1)
            elif s[1]=='y' and s[2]=='x':
                # f_yx
                return (2,1,0)
            elif s[1]=='y' and s[2]=='y':
                # f_yy
                return (2,1,1)
            else:
                raise Exception('Use uxx,uxy,uyx, or uyy or v*.')
        else:
            raise Exception('Higher order derivatives not supported.')
    


"""
Finite Element Classes
"""

class FiniteElement(object):
    """
    Parent Class: Finite Elements
    """
    def __init__(self, dim, element_type, cell_type):   
        self._element_type = element_type
        self._dim = dim    
        self._cell_type = cell_type
        
    def dim(self):
        """
        Returns the spatial dimension
        """
        return self._dim
     
     
    def cell_type(self):
        """
        Returns 'quadrilateral', 'triangle' or None
        """
        return self._cell_type
        
        
class QuadFE(FiniteElement):
    """
    Galerkin finite elements on quadrilateral cells 
    """
    def __init__(self, dim, element_type):
        if dim==1:
            cell_type = 'interval'
        elif dim==2:
            cell_type = 'quadrilateral'
            
        FiniteElement.__init__(self, dim, element_type, cell_type)
                
        if element_type == 'DQ0':
            """
            -------------------------------------------------------------------
            Constant Elements
            -------------------------------------------------------------------
            
            -----     
            | 0 |      --0--
            -----                     
            """
            p = [lambda x: np.ones(shape=x.shape)]
            px = [lambda x: np.zeros(shape=x.shape)]
            pxx = [lambda x: np.zeros(shape=x.shape)]
            if dim==1:
                dofs_per_vertex = 0
                dofs_per_edge = 1
                dofs_per_cell = 0
                basis_index = [0]
            elif dim==2:
                dofs_per_vertex = 0
                dofs_per_edge = 0
                dofs_per_cell = 1
                basis_index = [(0,0)]
            else:
                raise Exception('Only 1D and 2D supported.')
                
        elif element_type in ['Q1','DQ1']:
            """
            -------------------------------------------------------------------
            Linear Elements
            -------------------------------------------------------------------
        
            3---2
            |   |      0---1
            0---1
            """
            
            p  = [lambda x: 1-x, lambda x: x]
            px = [lambda x:-1*np.ones(x.shape), lambda x: 1*np.ones(x.shape)]
            pxx = [lambda x: 0*np.ones(x.shape), lambda x: 0*np.ones(x.shape)]
            if dim == 1:
                #
                # One Dimensional
                # 
                dofs_per_vertex = 1
                dofs_per_edge = 0
                dofs_per_cell = 0
                basis_index  = [0,1]
            elif dim == 2:
                #
                # Two Dimensional
                #
                dofs_per_vertex = 1
                dofs_per_edge = 0
                dofs_per_cell = 0
                basis_index = [(0,0),(1,0),(1,1),(0,1)]
        
        elif element_type in ['Q2','DQ2']:
            """
            -------------------------------------------------------------------
            Quadratic Elements
            -------------------------------------------------------------------
        
            3---6---2
            |       |
            7   8   5      0---2---1 
            |       |
            0---4---1
        
            """
            
            p =  [ lambda x: 2*x*x-3*x + 1.0, 
                   lambda x: 2*x*x-x, 
                   lambda x: 4.0*x-4*x*x ]
                
            px = [ lambda x: 4*x -3, 
                   lambda x: 4*x-1,
                   lambda x: 4.0-8*x ]
            
            pxx = [ lambda x: 4*np.ones(x.shape), \
                    lambda x: 4*np.ones(x.shape), \
                    lambda x: -8*np.ones(x.shape)]
            
            if dim == 1:
                #
                # One Dimensional
                # 
                dofs_per_vertex = 1
                dofs_per_edge = 1
                dofs_per_cell = 0
                basis_index = [0,1,2]
            elif dim == 2:
                #
                # Two Dimensional
                #
                dofs_per_vertex = 1 
                dofs_per_edge = 1
                dofs_per_cell = 1
                basis_index = [(0,0),(1,0),(1,1),(0,1),
                               (2,0),(1,2),(2,1),(0,2),(2,2)]
            else:
                raise Exception('Only 1D and 2D currently supported.')
             
        elif element_type in ['Q3','DQ3']:
            """
            -------------------------------------------------------------------
            Cubic Elements
            -------------------------------------------------------------------
            
            3----9---8----2
            |             |
            10  14   15   7
            |             |    0---2---3---1
            11  12   13   6
            |             |
            0----4---5----1 
            """
            
            p = [lambda x: -4.5*(x-1./3.)*(x-2./3.)*(x-1.),
                 lambda x:  4.5*x*(x-1./3.)*(x-2./3.),
                 lambda x: 13.5*x*(x-2./3.)*(x-1.),
                 lambda x: -13.5*x*(x-1./3.)*(x-1.) ]
                
            px = [lambda x: -13.5*x*x + 18*x - 5.5,
                  lambda x: 13.5*x*x - 9*x + 1.0,
                  lambda x: 40.5*x*x - 45*x + 9.0,
                  lambda x: -40.5*x*x + 36*x -4.5]
            
            pxx = [lambda x: -27*x + 18, \
                   lambda x: 27*x - 9, \
                   lambda x: 81*x - 45, \
                   lambda x: -81*x + 36]
            
            if dim == 1:
                #
                # One Dimensional
                #
                dofs_per_vertex = 1
                dofs_per_edge = 2
                dofs_per_cell = 0
                basis_index = [0,1,2,3]
                #ref_nodes = np.array([0.0,1.0,1/3.0,2/3.0])
            elif dim == 2:
                #
                # Two Dimensional
                #
                dofs_per_vertex = 1 
                dofs_per_edge = 2
                dofs_per_cell = 4
                basis_index = [(0,0),(1,0),(1,1),(0,1),
                               (2,0),(3,0),(1,2),(1,3),(3,1),(2,1),(0,3),(0,2),
                               (2,2),(3,2),(2,3),(3,3)]
                #ref_nodes = np.array([[0.0,0.0],[1.0,0.0],
                #                      [0.0,1.0],[1.0,1.0],
                #                      [0.0,1./3.],[0.0,2./3.],
                #                      [1.0,1./3.],[1.0,2./3.], 
                #                      [1./3.,0.0],[2./3.,0.0], 
                #                      [1./3.,1.0],[2./3.,1.0],
                #                      [1./3.,1./3.],[2./3.,1./3.],
                #                      [1./3.,2./3.],[2./3.,2./3.]])
                #pattern = ['SW','SE','NW','NE','W','W','E','E','S','S','N','N',
                #           'I','I','I','I']
        else:
            raise Exception('Element type {0} not recognized'.format(element_type))
        self.__dofs = {'vertex':dofs_per_vertex, 'edge':dofs_per_edge,'cell':dofs_per_cell}               
        self.__basis_index = basis_index
        self.__p = p
        self.__px = px
        self.__pxx = pxx
        self.__element_type = element_type
        self.__torn_element = True if element_type[:2]=='DQ' else False
        if self.dim()==1:
            self.__reference_cell = RInterval(self)
        elif self.dim()==2:
            self.__reference_cell = RQuadCell(self)
        else: 
            raise Exception('Can only construct 1D and 2D Reference Cells')
        #if dim == 2:
        #    self.pattern = pattern
    
    '''
    def local_dof_matrix(self):
        # TODO: Delete
        if hasattr(self, '__local_dof_matrix'):
            #
            # Return matrix if already computed
            #
            return self.__local_dof_matrix
        else:
            #
            # Construct matrix
            # 
            poly_degree = self.polynomial_degree()
            n = poly_degree + 1
            local_dof_matrix = np.zeros((n,n))
            count = 0
            #
            # Vertices upside down Z
            #
            dpv = self.n_dofs('vertex')
            if dpv > 0:
                local_dof_matrix[[0,0,-1,-1],[0,-1,0,-1]] = np.arange(0,4*dpv)
                count += 4*dpv
            
            #
            # Edges
            # 
            dpe = self.n_dofs('edge')
            
            # East
            local_dof_matrix[1:-1,0] = np.arange(count,count+dpe) 
            count += dpe 
            
            # West
            local_dof_matrix[1:-1,-1] = np.arange(count,count+dpe) 
            count += dpe  
            
            # South
            local_dof_matrix[0,1:-1] = np.arange(count,count+dpe) 
            count += dpe 
            
            # North
            local_dof_matrix[-1,1:-1] = np.arange(count,count+dpe) 
            count += dpe 
            
            #
            # Interior
            # 
            dpi = int(np.sqrt(self.n_dofs('cell')))
            i_dofs = np.arange(count,count+dpi*dpi).reshape((dpi,dpi))
            local_dof_matrix[1:-1,1:-1] = i_dofs
            
            self.__local_dof_matrix = local_dof_matrix
            return local_dof_matrix
    '''
            
    def basis_index(self):
        return self.__basis_index
        
    
    def element_type(self):
        """
        Return the finite element type (Q0, Q1, Q2, or Q3)
        """ 
        return self.__element_type
    
    
    def torn_element(self):
        """
        Return whether the element is torn (discontinuous)
        """
        return self.__torn_element
    
        
    def polynomial_degree(self):
        """
        Return the finite element's polynomial degree 
        """
        return int(list(self.__element_type)[-1])
    
        
    def n_dofs(self,key=None):
        """
        Return the number of dofs per elementary entity
        """
        # Total Number of dofs
        if key is None:
            dim = self.dim()
            if dim==1:
                return 2*self.__dofs['vertex'] + self.__dofs['edge']
            elif dim==2:
                return 4*self.__dofs['vertex'] + 4*self.__dofs['edge'] + \
                        self.__dofs['cell']
        else:
            assert key in self.__dofs.keys(), 'Use "vertex","edge", "cell" for key'
            return self.__dofs[key]
    
    
    def reference_nodes(self):
        """
        Returns all dof vertices of reference cell
        """
        x = self.reference_cell().get_dof_vertices()
        return convert_to_array(x, self.dim())
        
      
    def reference_cell(self):
        """
        Returns the referernce cell
        """
        return self.__reference_cell
    
     
        
    def phi(self, n, x):
        """
        Evaluate the nth basis function at the point x
        
        Inputs: 
        
            n: int, basis function number
            
            x: double, point at which function is to be evaluated
               (double if dim=1, or tuple if dim=2) 
        """
        x = convert_to_array(x)
        assert n < self.n_dofs(), 'Basis function index exceeds n_dof'
        #
        # 1D 
        # 
        if self.dim() == 1:
            i = self.__basis_index[n]
            return self.__p[i](x)
        #
        # 2D
        # 
        elif self.dim() == 2:
            i1,i2 = self.__basis_index[n]
            return self.__p[i1](x[:,0])*self.__p[i2](x[:,1])
            
        else:
            raise Exception('Only 1D and 2D elements supported.')


    def dphi(self,n,x,var=0):
        """
        Evaluate the partial derivative nth basis function
        
        Inputs:
        
            n: int, basis function number
            
            x: double, point at which we evaluate the derivative
            
            var: int, variable w.r.t. which we differentiate (0 or 1)
            
        Output:
        
          dphi_dx or dphi_dy  
        """
        assert n < self.n_dofs(), 'Basis index exceeds n_dofs.'
        assert var < 2, 'Use 0 or 1 for var.'
        #
        # 1D
        # 
        if self.dim() == 1: 
            i = self.__basis_index[n]
            return self.__px[i](x)
        #
        # 2D
        # 
        elif self.dim() == 2:
            i1,i2 = self.__basis_index[n]
            if var == 0:
                #
                # dphi_dx
                #
                return self.__px[i1](x[:,0])*self.__p[i2](x[:,1])
            elif var == 1:
                #
                # dphi_dy
                # 
                return self.__p[i1](x[:,0])*self.__px[i2](x[:,1])
   
   
    def d2phi(self, n, x, var):
        """
        Evaluate the second partial derivative of the nth basis function
        
        Inputs: 
        
            n: int, basis function number 
            
            x: double, (n_points, dim) array of points at which to evaluate
                the basis function.
                
            var: int, dim-tuple of variables (0 or 1) wrt which we differentiate  
                e.g. var=(0,1) computes phi_xy(x)
                
        Output:
        
            d2phi
        
        """
        assert n < self.n_dofs(), 'Basis index exceeds n_dofs.'
        assert all(var) < 2, 'Use tuple of 0s for x or 1s for y.'
        
        #
        # 1D 
        #
        if self.dim() == 1:
            i = self.__basis_index[n]
            return self.__pxx[i](x)
        #
        # 2D
        #          
        elif self.dim() == 2:
            i1,i2 = self.__basis_index[n]
            j1,j2 = var 
            if j1==0 and j2==0:
                # 
                # p_xx
                #
                return self.__pxx[i1](x[:,0])*self.__p[i2](x[:,1])
            elif (j1==0 and j2==1) or (j1==1 and j2==0):
                #
                # p_xy or p_yx
                #
                return self.__px[i1](x[:,0])*self.__px[i2](x[:,1])
            elif j1==1 and j2==1:
                #
                # p_yy
                #
                return self.__p[i1](x[:,0])*self.__pxx[i2](x[:,1])
         
    
    def shape(self, x, cell=None, derivatives=(0,), local_dofs='all'):
        """
        Evaluate all shape functions at a given points
        
        Inputs: 
        
            x: double, points at which shape functions are to be evaluated. 
                forms
                
                1. (n_points, n_dim) array, or
                
                2. list of n_points n_dim-tuples,
                
                3. list of n_points vertices  
            
            
            cell [None]: QuadCell, optionally specify cell in which case:
                
                1. the input x is first mapped to the reference, 
                
                2. when computing derivatives, shape functions are modified to
                   account for the coordinate mapping.
             
             
            derivatives: list of tuples, (order,i,j) where 
                
                1. order specifies the order of the derivative,
                
                2. i,j specify the variable wrt which we differentiate
                    e.g. (2,0,0) computes d^2p/dx^2 = p_xx,
                         (2,1,0) computes d^2p/dxdy = p_yx
                      
                        
            local_dofs: int, list of local dof indices whose entries in
                range(self.n_dofs).
                
                
        Output: 
        
            phi: double, list of (n_points, len(local_dofs)) arrays of 
                 (derivatives of ) shape functions, evaluated at the given 
                 points.
                 
                 

        """
        #
        # Convert x to array
        #
        x = convert_to_array(x, self.dim())      
        n_points = x.shape[0]
        
        #
        # Only one derivative specified
        # 
        if not type(derivatives) is list:
            derivatives = [derivatives]
            is_singleton = True
        else:
            is_singleton = False
        #
        # Check whether points are valid.
        # 
        if cell is None:
            #
            # Points should lie in the reference domain 
            #
            assert all(x.ravel() >= 0), 'All entries should be nonnegative.'
            assert all(x.ravel() <= 1), 'All entries should be at most 1.'
            x_ref = x
            
        elif cell is not None:
            #
            # Points should lie in the physical domain
            # 
            assert np.all(cell.contains_points(x)), \
                'Not all points contained in the cell'
                
            #
            # Map points to reference cell
            # 
            
            # Analyze derivative to determine whether to include 
            # the hessian and/or jacobian.
            jacobian = any(der[0]==1 or der[0]==2 for der in derivatives)
            hessian  = any(der[0]==2 for der in derivatives)
            if hessian:
                #
                # Return jacobian and hessian
                # 
                x_ref, J, H = \
                    cell.reference_map(x, jacobian=jacobian, hessian=hessian,\
                                       mapsto='reference')
            elif jacobian:
                #
                # Return only jacobian
                # 
                x_ref, J = \
                    cell.reference_map(x, jacobian=jacobian,\
                                       mapsto='reference')
            else: 
                #
                # Return only point
                # 
                x_ref = cell.reference_map(x, mapsto='reference')
        #
        # Check local_dof numbers
        # 
        n_dofs = self.n_dofs()
        if local_dofs=='all':
            #
            # No local dofs given: use them all
            # 
            local_dofs = [i for i in range(n_dofs)]
        else:
            #
            # Local dofs given: check if they're ok.
            # 
            all(type(i) is np.int for i in local_dofs),
            'Local dofs must be of type int.'
            
            all((i>=0) and (i<=n_dofs) for i in local_dofs),
            'Local dofs not in range.'
        n_dofs_loc = len(local_dofs)
        phi = []    
        for der in derivatives:
            p = np.zeros((n_points,n_dofs_loc))
            if der[0] == 0:
                #
                # No derivatives
                #
                for i in range(n_dofs_loc):
                    p[:,i] = self.phi(local_dofs[i], x_ref).ravel() 
            elif der[0] == 1:
                i_var = der[1]
                # 
                # First derivatives
                #
                if cell is not None:
                    if isinstance(cell, Interval):
                        #
                        # Interval
                        # 
                        ds_dx = np.array(J)
                        for i in range(n_dofs_loc):
                            p[:,i] = self.dphi(local_dofs[i], x_ref, var=i_var).ravel()
                            p[:,i] = ds_dx*p[:,i]
                    elif isinstance(cell, QuadCell):
                        if cell.is_rectangle():
                            #
                            # Rectangular cells are simpler
                            # 
                            dst_dxy = np.array([Ji[i_var,i_var] for Ji in J])
                            for i in range(n_dofs_loc):
                                p[:,i] = self.dphi(local_dofs[i], x_ref, var=i_var)
                                p[:,i] = dst_dxy*p[:,i]
                        else:
                            #
                            # Quadrilateral cells
                            # 
                            ds_dxy = np.array([Ji[0,i_var] for Ji in J])
                            dt_dxy = np.array([Ji[1,i_var] for Ji in J])
                            for i in range(n_dofs_loc):
                                dN_ds = self.dphi(local_dofs[i], x_ref, var=0)
                                dN_dt = self.dphi(local_dofs[i], x_ref, var=1)
                                p[:,i] = dN_ds*ds_dxy + dN_dt*dt_dxy                            
                else:
                    for i in range(n_dofs_loc):
                        p[:,i] = self.dphi(local_dofs[i], x_ref, var=i_var)
            elif der[0]==2:
                #
                # Second derivatives
                #
                if self.dim()==1:
                    i_var = der[1]
                elif self.dim()==2:
                    i_var, j_var = der[1:]
                if cell is not None:
                    if isinstance(cell, Interval):
                        #
                        # Interval
                        # 
                        ds_dx = np.array(J)
                        for i in range(n_dofs_loc):
                            p[:,i] = (ds_dx)**2*self.d2phi(local_dofs[i], x_ref, der[1:]).ravel()
                    elif isinstance(cell, QuadCell):
                        if cell.is_rectangle():
                            #
                            # Rectangular cell: mixed derivatives 0
                            # 
                            dri_dxi = np.array([Ji[i_var,i_var] for Ji in J])
                            drj_dxj = np.array([Ji[j_var,j_var] for Ji in J])
                            for i in range(n_dofs_loc):
                                p[:,i] = \
                                    dri_dxi*drj_dxj*self.d2phi(local_dofs[i], x_ref, der[1:])
                            
                        else:
                            #
                            # General quadrilateral
                            # 
                            # First partial dertivatives of (s,t) wrt xi, xj
                            s_xi = np.array([Ji[0,i_var] for Ji in J]) 
                            s_xj = np.array([Ji[0,j_var] for Ji in J])
                            t_xi = np.array([Ji[1,i_var] for Ji in J])
                            t_xj = np.array([Ji[1,j_var] for Ji in J])
                            
                            # Second mixed partial derivatives of (s,t) wrt xi, xj
                            s_xixj = np.array([Hi[i_var,j_var,0] for Hi in H])
                            t_xixj = np.array([Hi[i_var,j_var,1] for Hi in H])
                            
                            for i in range(n_dofs_loc):
                                #
                                # Reference partial derivatives
                                # 
                                N_s = self.dphi(local_dofs[i], x_ref, var=0)
                                N_t = self.dphi(local_dofs[i], x_ref, var=1)
                                N_ss = self.d2phi(local_dofs[i], x_ref, (0,0))
                                N_st = self.d2phi(local_dofs[i], x_ref, (0,1))
                                N_tt = self.d2phi(local_dofs[i], x_ref, (1,1))
                                #
                                # Mapped second derivative
                                # 
                                p[:,i] = N_ss*s_xj*s_xi + N_st*t_xj*s_xi +\
                                         N_s*s_xixj + \
                                         N_st*s_xj*t_xi + N_tt*t_xj*t_xi +\
                                         N_t*t_xixj
                else:
                    #
                    # No mapping
                    # 
                    for i in range(n_dofs_loc):
                        p[:,i] = self.d2phi(local_dofs[i], x_ref, der[1:])
            phi.append(p)
            
        if is_singleton:
            return phi[0]
        else:
            return phi
            
            
    def constraint_coefficients(self):
        """
        Returns the constraint coefficients of a typical bisected edge. 
        Vertices on the coarse edge are numbered in increasing order, 
        e.g. 0,1,2,3 for Q2,
        
        Output:
        
            constraint: double, dictionary whose keys are the fine node numbers  
            
        Notes: This works only for 2D quadcells
        """        
        dpe = self.n_dofs('edge')
        edge_shapefn_index = [0] + [i for i in range(2,dpe+2)] + [1]
        coarse_index = [2*r for r in range(dpe+2)]
        fine_index = range(2*dpe+3)
        constraint = [{},{}]
        for i in fine_index:
            if not i in coarse_index:
                c = []
                for j in edge_shapefn_index:
                    c.append(self.__p[j](0.5*float(i)/float(dpe+1)))
                if i < dpe+1:
                    constraint[0][i] = c
                elif i==dpe+1:
                    constraint[0][i] = c
                    constraint[1][i-(dpe+1)] = c
                else:
                    constraint[1][i-(dpe+1)] = c  
        return constraint
        

class TriFE(FiniteElement):
    """
    Continuous Galerkin finite elements on triangular cells

        Define a shape function on the reference triangle with vertices 
        (0,0), (1,0), and (0,1).

    """
    def __init__(self, dim, element_type):
        """
        Constructor
        
        Inputs:
        
            dim: int, physical dimension
            
            element_type: str, type of triangular element 
                ('P1','P2','P3',or 'Bubble')
        """
        FiniteElement.__init__(self,dim,element_type)

        #
        # One dimensional 
        #
        if dim == 1:
            if element_type == 'P1':
                n_dof = 2
                self.__phi = [lambda x: 1-x, lambda x: x]
                self.__phix = [lambda x: -1.0, lambda x: 1.0]  
                            
                 
            elif element_type == 'P2':
                n_dof = 3
                self.__phi = [lambda x: 2*x*x-3*x + 1.0, 
                              lambda x: 2*x*x-x, 
                              lambda x: 4.0*x-4*x*x]
                
                self.__phix = [lambda x: 4*x -3, 
                               lambda x: 4*x-1,
                               lambda x: 4.0-8*x]
                
            elif element_type == 'P3':
                n_dof = 4
                self.__phi = [lambda x: -4.5*(x-1./3.)*(x-2./3.)*(x-1.),
                              lambda x:  4.5*x*(x-1./3.)*(x-2./3.),
                              lambda x: 13.5*x*(x-2./3.)*(x-1.),
                              lambda x: -13.5*x*(x-1./3.)*(x-1.) ]
                
                self.__phix = [lambda x: -13.5*x*x + 18*x - 5.5,
                               lambda x: 13.5*x*x - 9*x + 1.0,
                               lambda x: 40.5*x*x - 45*x + 9.0,
                               lambda x: -40.5*x*x + 36*x -4.5]
            else: 
                raise Exception('Use P1, P2, or P3 for element_type.')
            
        elif dim == 2:
            #
            # Two dimensional
            # 
            if element_type == 'P1':
                #
                # Piecewise linear basis
                #
                n_dof = 3
                self.__phi = [lambda x,y: 1.0-x-y, lambda x,y: x, lambda x,y: y]

                self.__phix = [lambda x,y: -1.0, lambda x,y: 1.0, lambda x,y: 0.0]

                self.__phiy = [lambda x,y: -1.0, lambda x,y: 0.0, lambda x,y: 1.0]

            elif element_type == 'P2':
                #
                # Piecewise quadratic basis
                #
                n_dof = 6
                self.__phi = \
                    [lambda x,y: 1.0 - 3*x - 3*y + 2*x*x + 4*x*y + 2*y*y,
                     lambda x,y:     - 1*x       + 2*x*x,
                     lambda x,y:           - 1*y                 + 2*y*y,
                     lambda x,y:       4*x       - 4*x*x - 4*x*y,
                     lambda x,y:                           4*x*y,
                     lambda x,y:             4*y         - 4*x*y - 4*y*y]

                self.__phix = \
                    [lambda x,y:     - 3.0       + 4*x   + 4*y,
                     lambda x,y:     - 1.0       + 4*x,
                     lambda x,y: 0.0,
                     lambda x,y:       4.0       - 8*x   - 4*y,
                     lambda x,y:                           4*y,
                     lambda x,y:                         - 4*y]

                self.__phiy = \
                    [lambda x,y:           - 3.0         + 4*x   + 4*y,
                     lambda x,y: 0.0,
                     lambda x,y:           - 1.0                 + 4*y,
                     lambda x,y:                         - 4*x,
                     lambda x,y:                           4*x,
                     lambda x,y:             4.0         - 4*x   - 8*y]

            elif element_type == 'Bubble':
                #
                # Bubble elements
                #
                n_dof = 7
                self.__phi = \
                    [lambda x,y: (1.-x-y)*(2*(1.-x-y)-1.) +  3*(1.-x-y)*x*y,
                     lambda x,y: x*(2*x-1.)               +  3*(1.-x-y)*x*y,
                     lambda x,y: y*(2*y-1.)               +  3*(1.-x-y)*x*y,
                     lambda x,y: 4*(1.-x-y)*x             - 12*(1.-x-y)*x*y,
                     lambda x,y: 4*x*y                    - 12*(1.-x-y)*x*y,
                     lambda x,y: 4*y*(1.-x-y)             - 12*(1.-x-y)*x*y,
                     lambda x,y: 27*(1.-x-y)*x*y]

                self.__phix = \
                    [lambda x,y: -3.0 + 4*x +  7*y -  6*x*y -  3*(y**2),
                     lambda x,y: -1.0 + 4*x +  3*y -  6*x*y -  3*(y**2),
                     lambda x,y:               3*y -  6*x*y -  3*(y**2),
                     lambda x,y:  4.0 - 8*x - 16*y + 24*x*y + 12*(y**2),
                     lambda x,y:            -  8*y + 24*x*y + 12*(y**2),
                     lambda x,y:            - 16*y + 24*x*y + 12*(y**2),
                     lambda x,y:              27*y - 54*x*y - 27*(y**2)]

                self.__phiy = \
                    [lambda x,y: -3.0 +  7*x + 4*y -  6*x*y -  3*(x**2),
                     lambda x,y:         3*x       -  6*x*y -  3*(x**2),
                     lambda x,y: -1.0 +  3*x + 4*y -  6*x*y -  3*(x**2),
                     lambda x,y:       -16*x       + 24*x*y + 12*(x**2),
                     lambda x,y:       - 8*x       + 24*x*y + 12*(x**2),
                     lambda x,y:  4.0 - 16*x - 8*y + 24*x*y + 12*(x**2),
                     lambda x,y:        27*x       - 54*x*y - 27*(x**2)]
        self.__n_dof = n_dof
        self.__cell_type = 'triangle'
        
    
    def phi(self,n,x):
        """
        Evaluate the nth basis function at the point x
        
        Inputs:
        
            n: int, basis function index
            
            x: double, point at which to evaluate the basis function
        """
        assert n < self.n_dofs(), 'Basis function index exceeds n_dof'
        #
        # 1D 
        # 
        if self.dim() == 1:
            return self.__phi[n](x)
        #
        # 2D
        # 
        elif self.dim() == 2:
            return self.__phi[n](*x)
        
        
    def dphi(self,n,x,var=0):
        """
        Evaluate the partial derivative of the nth basis function
        """
        assert n < self.n_dofs(), 'Basis function index exceeds n_dof'
        #
        # 1D
        # 
        if self.dim() == 1:
            return self.__phix[n](x)
        #
        # 2D 
        #
        elif self.dim() == 2:
            if var == 0:
                return self.__phix[n](*x)
            elif var == 1:
                return self.__phiy[n](*x)
            else:
                raise Exception('Can only differentiate wrt variable 0 or 1.')


class Function(object):
    """
    Function class for finite element objects.
    
    Attributes:
    
        mesh [None]: Mesh, computational mesh
        
        element [None]: FiniteElement, element
        
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
    
    
    def __init__(self, f, fn_type, mesh=None, element=None, \
                 dofhandler=None, subforest_flag=None):
        """
        Constructor:
        
        
        Inputs:
    
            f: function or vector whose length is consistent with the dofs 
                required by the mesh/element/subforest or dofhandler/subforest.
                f can also be passed as an (n_dofs, n_samples) array.  
                
            fn_type: str, function type ('explicit', 'nodal', or 'constant')
            
            *mesh [None]: Mesh, on which the function will be defined
            
            *element [None]: FiniteElement, on whch the function will be defined
            
            *dofhandler [None]: DofHandler, specifying the mesh and element on
                which the function is defined.
            
            *subforest_flag [None]: str/int, marker specifying submesh
            
            
        Note: We allow for the option of specifying multiple realizations 
            - If the function is not stochastic, the number of samples is None
            - If the function has multiple realizations, its function values 
                are stored in an (n_dofs, n_samples) array. 
    
        """ 
        #
        # Construct DofHandler if possible
        # 
        if dofhandler is not None:
            assert isinstance(dofhandler, DofHandler), 'Input dofhandler ' +\
                'should be of type DofHandler.'
        elif mesh is not None and element is not None:
            dofhandler = DofHandler(mesh, element)
        self.dofhandler = dofhandler
        
        # Distribute Dofs and store them
        if self.dofhandler is not None:
            self.dofhandler.distribute_dofs()
            self.dofhandler.set_dof_vertices()
            self.__global_dofs = \
                self.dofhandler.get_region_dofs(subforest_flag=subforest_flag)
        #
        # Store function type
        #
        assert fn_type in ['explicit', 'nodal','constant'], \
            'Input "fn_type" should be "explicit", "nodal", or "constant".'      
        self.__type = fn_type

        if fn_type == 'explicit':
            # 
            # Explicit function
            # 
            dim = f.__code__.co_argcount
            if mesh is not None:
                assert dim == mesh.dim(), \
                'Number of inputs and mesh dimension do not match.'
            elif dofhandler is not None:
                assert dim == dofhandler.mesh.dim(), \
                'Number of inputs and mesh dimension do not match.'
                  
            n_samples = None               
            fn = f
            
        elif fn_type == 'nodal':
            # 
            # Nodal (finite element) function
            # 
            assert self.dofhandler is not None, \
            'If function_type is "nodal", dofhandlercell '\
            '(or mesh and element required).' 
            
            
            if callable(f):
                #
                # Function passed explicitly
                #
                dim = f.__code__.co_argcount
                assert dim == dofhandler.mesh.dim(), \
                'Number of inputs and mesh dimension do not match.'
                
                x = dofhandler.get_dof_vertices(dofs=self.global_dofs())
                nf = dofhandler.n_dofs(subforest_flag=subforest_flag)
                n_samples = None
                if dim == 1:
                    fn = f(x[:,0])
                elif dim == 2: 
                    fn = f(x[:,0],x[:,1])
                    
            elif type(f) is np.ndarray:
                # 
                # Function passed as an array
                # 
                # Determine number of samples
                if len(f.shape)==1:
                    n_samples = None
                    nf = f.shape[0]
                    fn = f
                else:
                    nf, n_samples = f.shape
                    fn = f
                n_dofs = self.dofhandler.n_dofs(subforest_flag=subforest_flag)
                
                assert nf == n_dofs,\
                    'The number of entries of f %d does not equal'+\
                    ' the number of dofs %d.'%(nf, n_dofs) 
                dim = self.dofhandler.mesh.dim() 
            
                
        elif fn_type == 'constant':
            # 
            # Constant function
            # 
            dim = None
            # Determine number of samples
            if type(f) is np.ndarray:
                assert len(f.shape)==1, 'Constant functions are passed '+\
                'as scalars or vectors.'
                n_samples = len(f)
                fn = f
            elif isinstance(f, numbers.Real):
                n_samples = None
                fn = f
            elif type(f) is list:
                n_samples = len(f)
                fn = np.array(f)
                 
        else:
            raise Exception('Variable function_type should be: '+\
                            ' "explicit", "nodal", or "constant".')        
        
        self.__dim = dim       
        self.__f = fn
        self.__flag = subforest_flag 
        self.__n_samples = n_samples
        
 
    def assign(self, v, pos=None):
        """
        Assign function values to the function in the specified sample position
        
        Inputs: 
        
            v: double, array 
            
            pos: int, array or constant (indicating position)
            
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
        and self.dofhandler.mesh==mesh \
        and self.flag()==subforest_flag:
            return True
        else:
            return False
 
         
    def global_dofs(self):
        """
        Returns the global dofs associated with the function values. 
        (Only appropriate for nodal type functions).
        """    
        if self.fn_type() == 'nodal':
            return self.__global_dofs
        else:
            raise Exception('Function must be of type "nodal".')
    
    
    def flag(self):
        """
        Returns the flag used to define the mesh restriction on which 
        the function is defined
        """    
        return self.__flag
    
        
    def input_dim(self):
        """
        Returns the dimension of the function's domain
        """
        return self.__dim
    
    
    def n_samples(self):
        """
        Returns the number of realizations stored by the function
        """
        return self.__n_samples
    
        
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
        flag = self.__flag
        dim = self.__dim
        
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
                
        # =====================================================================
        # Parse sample size
        # =====================================================================
        if samples is not 'all':
            if type(samples) is int:
                sample_size = 1
                samples = np.array([samples])
            else:
                assert type(samples) is np.ndarray, \
                'vector specifying samples should be an array'
                
                assert len(samples.shape) == 1, \
                'sample indexing vector should have dimension 1'
                
                assert self.__n_samples > samples.max(), \
                'Sample paths not stored in function.'
                
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
                      
            assert samples=='all', \
                'Use samples="all" for explicit functions.'
            
            if dim == 1:
                f_vec = self.__f(x[:,0])
            elif dim == 2:
                f_vec = self.__f(x[:,0], x[:,1])
            else:
                raise Exception('Only 1D and 2D inputs supported.')
    
        elif self.fn_type() == 'nodal':
            
            if phi is not None:
                # =============================================================
                # Shape function specified
                # =============================================================
                #
                # Checks
                # 
                assert dofs is not None, \
                    'When shape function provided, require input "dofs".'
                
                if not all([dof in self.__global_dofs for dof in dofs]):
                    print(dofs)
                    print(self.__global_dofs)
                assert all([dof in self.__global_dofs for dof in dofs]),\
                    'Nodal function not defined at given dofs.' 
                
                assert len(dofs)==phi.shape[1], \
                    'Number of columns in phi should equal the number of dofs'
                 
                #
                # Evaluate function at local dofs 
                # 
                idx_cell = [self.__global_dofs.index(i) for i in dofs]
    
                if self.n_samples() is None:
                    f_loc = self.__f[idx_cell]
                elif samples is 'all':
                    f_loc = self.__f[idx_cell,:]
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
                n_samples = self.n_samples()
                if n_samples is None:
                    f_vec = np.empty(x.shape[0])
                elif samples is 'all':
                    f_vec = np.empty((x.shape[0],n_samples))
                else:
                    f_vec = np.empty((x.shape[0],sample_size))    
                
                #
                # Determine tree cells to traverse
                # 
                if cell is None:
                    #
                    # Cell not specified
                    # 
                    cell_list = self.dofhandler.mesh.cells.get_leaves(subforest_flag=flag)
                else:
                    #
                    # Cell given
                    # 
                    assert all(cell.contains_points(x)), \
                    'Cell specified, but not all points contained in cell.'
                    
                    if flag is not None:
                        #
                        # Function is defined on a flagged submesh
                        # 
                        if not cell.is_marked(flag):
                            #
                            # Function defined on a coarser mesh
                            #
                            while not cell.is_marked(flag):
                                # 
                                # Get smallest cell in function mesh that contains cell 
                                # 
                                cell = cell.get_parent()
                            cell_list = [cell]
                        elif cell.has_children(flag=flag):
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
                    idx_cell = [self.__global_dofs.index(i) for i in \
                                self.dofhandler.get_cell_dofs(cell)]  
                    if self.n_samples() is None:
                        f_loc = self.__f[idx_cell]
                    elif samples is 'all':
                        f_loc = self.__f[idx_cell,:]
                    else:
                        f_loc = self.__f[np.ix_(idx_cell, samples)]
        
                    #
                    # Evaluate shape function at x-values
                    #    
                    in_cell = cell.contains_points(x)
                    x_loc = x[in_cell,:]
                    phi = self.dofhandler.element.shape(x_loc, cell=cell, \
                                                        derivatives=derivative)
                    #
                    # Update output vector
                    # 
                    if n_samples is None:
                        f_vec[in_cell] = np.dot(phi, f_loc)
                    else:
                        f_vec[in_cell,:] = np.dot(phi, f_loc)
                                                            
        elif self.fn_type() == 'constant':
            n_samples = self.n_samples()
            
            if n_samples is None:
                f_vec = self.fn()*np.ones((x.shape[0]))
            elif samples == 'all':
                one = np.ones((x.shape[0], n_samples))
                f_vec = one*self.fn()
            else:
                one = np.ones((x.shape[0], sample_size))
                f_vec = one*self.fn()[samples]
                            
        else:
            raise Exception('Function type must be "explicit", "nodal", '+\
                            ' or "constant".')
                                
        if is_singleton:
            return f_vec[0]
        else:
            return f_vec
        
        
    def interpolant(self, mesh=None, element=None, subforest_flag=None):
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
        flag = self.__flag
        dim = self.__dim  
        mesh, element = self.dofhandler.mesh, self.dofhandler.element
        
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
        dofs = dofhandler.get_region_dofs(subforest_flag=flag)
        x = dofhandler.get_dof_vertices(dofs)       
        #
        # Evaluate function at dof vertices
        #
        fv = self.eval(x, derivative=derivative)
        #
        # Define new function
        #
        return Function(fv, 'nodal', dofhandler=dofhandler, \
                        subforest_flag=flag) 
    
    
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
    
    
class DofHandler(object):
    """
    Degrees of freedom handler
    """
    def __init__(self, mesh, element):
        """
        Constructor
        """
        self.element = element
        self.mesh = mesh
        self.__global_dofs = {}
        self.constraints = {'constrained_dofs': [], 
                            'supporting_dofs': [], 
                            'coefficients': [],
                            'affine_terms': []}
        self.__hanging_nodes = {}  # TODO: delete
        self.__l2g = {}     # TODO: delete
        self.__dof_count = 0
        self.__dof_vertices = {}
    
    def clear_dofs(self):
        """
        Clear all dofs
        """
        self.__global_dofs = {}
        self.__dof_count = 0
        self.__dof_vertices = {}
        
                
    def distribute_dofs(self, subforest_flag=None):
        """
        Global enumeration of degrees of freedom        
        
        If the mesh is hierarchical, the dofs are assigned to all cells in the 
        mesh forest, i.e. at different coarseness levels.       
        """
        dim = self.element.dim()
        if dim==1:
            for interval in self.mesh.cells.traverse(mode='breadth-first',
                                                     flag=subforest_flag):
                #
                # Fill own dofs
                # 
                self.fill_dofs(interval)

                if not self.element.torn_element():
                    #
                    # Continuous element 
                    #
                    self.share_dofs_with_neighbors(interval)
                    
                if interval.has_children(flag=subforest_flag):
                    #
                    # Interval has children: share dofs with them
                    # 
                    self.share_dofs_with_children(interval)
        elif dim==2:
            #
            # Ensure the mesh is balanced
            # 
            assert self.mesh.is_balanced(), \
                'Mesh must be balanced before dofs can be distributed.'
            
            for cell in self.mesh.cells.traverse(mode='breadth-first', 
                                                 flag=subforest_flag):
                #
                # Fill own dofs
                # 
                self.fill_dofs(cell)
                
                if not self.element.torn_element():
                    #
                    # Continuous element: share dofs with neighbors 
                    # 
                    self.share_dofs_with_neighbors(cell)
                
                if cell.has_children(flag=subforest_flag):
                    #
                    # Cell has children: share dofs with them
                    # 
                    self.share_dofs_with_children(cell)
        self.set_dof_vertices(subforest_flag=subforest_flag)
    
    
    def share_dofs_with_children(self, cell):
        """
        Share cell's dofs with its children (only works for quadcells)
        """
        assert cell.has_children(), 'Cell should have children'
        cell_dofs = self.get_cell_dofs(cell)
        for child in cell.get_children():
            dofs = []
            pos = []
            i_child = child.get_node_position()
            for vertex in self.element.reference_cell().get_dof_vertices():
                if vertex.get_pos(1, i_child) is not None:
                    pos.append(vertex.get_pos(1, i_child))
                    dofs.append(cell_dofs[vertex.get_pos(0)])
            self.assign_dofs(dofs, child, pos=pos)

             
    def share_dofs_with_neighbors(self, cell, pivot=None, flag=None):                     
        """
        Assign shared degrees of freedom to neighboring cells
        
        Inputs:
        
            cell: Cell in Mesh
            
            *pivot: Vertex or HalfEdge wrt which we seek neighbors
            
            *flag: marker, restricting the neighboring cells 
        """
        if self.element.torn_element():
            #
            # Discontinuous Elements don't share dofs
            # 
            return 
        if pivot is None:
            #
            # No pivot specified: Share with all neighbors
            # 
            for vertex in cell.get_vertices():
                #
                # 1D and 2D: Share dofs with neighbors about vertices
                #
                self.share_dofs_with_neighbors(cell, vertex, flag=flag)
            if self.element.dim()==2:
                #
                # 2D
                # 
                for half_edge in cell.get_half_edges():
                    #
                    # Share dofs with neighbors about half_edges
                    #
                    self.share_dofs_with_neighbors(cell, half_edge, flag=flag)
        else:
            dofs = self.get_cell_dofs(cell, pivot, interior=True)
            if len(dofs)!=0:
                if isinstance(pivot, Vertex):
                    # 
                    # Vertex
                    #
                    if self.element.dim()==1:
                        #
                        # 1D
                        # 
                        # Get neighbor and check that it's not None
                        nbr = cell.get_neighbor(pivot)
                        if nbr is None:
                            return
                        if pivot.is_periodic():
                            for v_nbr in pivot.get_periodic_pair(nbr):
                                self.assign_dofs(dofs, nbr, v_nbr)
                        else:
                            self.assign_dofs(dofs, nbr, pivot)
                    elif self.element.dim()==2:
                        #
                        # 2D
                        # 
                        for nbr in cell.get_neighbors(pivot, flag=flag):
                            if pivot.is_periodic():
                                for v_nbr in pivot.get_periodic_pair(nbr):
                                    self.assign_dofs(dofs, nbr, v_nbr)
                            else:
                                self.assign_dofs(dofs, nbr, pivot)
                        
                elif isinstance(pivot, HalfEdge):
                    #
                    # HalfEdge
                    # 
                    nbr = cell.get_neighbors(pivot, flag=flag)
                    if nbr is not None:
                        dofs.reverse()
                        self.assign_dofs(dofs, nbr, pivot.twin())
                        
            
    def fill_dofs(self, cell):
        """
        Fill in cell's dofs 
        
        Inputs:
        
            node: cell, whose global degrees of freedom are to be augmented
    
        Modify: 
        
            __global_dofs[node]: updated global dofs
            
            __n_global_dofs: updated global dofs count
        """
        #
        # Preprocessing
        #
        if not cell in self.__global_dofs:
            #
            # Cell not in dictionary -> add it
            # 
            self.__global_dofs[cell] = [None]*self.element.n_dofs()
        cell_dofs = self.get_cell_dofs(cell)  
        
        if not None in cell_dofs:
            #
            # Dofs already filled, move on
            # 
            return
        
        count = self.__dof_count
        dim = self.element.dim()
                
        own_neighbor = False
        if self.mesh.is_periodic() and not self.element.torn_element():
            # =================================================================
            # Periodic Mesh 
            # =================================================================
            #
            # Check if cell/interval is its own neighbor
            #        
            own_neighbor = False
            if dim==1:
                for vertex in cell.get_vertices():
                    nb = cell.get_neighbor(vertex)
                    if nb == cell:
                        own_neighbor = True
                        break
            elif dim==2:
                for half_edge in cell.get_half_edges():
                    nb = cell.get_neighbors(half_edge)
                    if nb == cell:
                        own_neighbor = True
                        break
        #
        # If own neighbor, fill dofs differently
        #     
        if own_neighbor:
            #
            # Assign dofs to vertices
            # 
            dpv = self.element.n_dofs('vertex')
            if dpv != 0:
                dofs = []
                for vertex in cell.get_vertices():
                    # Refill dofs to be assigned
                    while len(dofs)<dpv:
                        dofs.append(count)
                        count += 1
                 
                    # Assign dofs to each vertex
                    self.assign_dofs(dofs, cell, vertex)
                    for c,v in vertex.get_periodic_pair():
                        self.assign_dofs(dofs, c, v)

                    # Throw out used dofs
                    vertex_dofs = self.get_cell_dofs(cell, vertex)
                    for dof in dofs:
                        if dof in vertex_dofs:
                            dofs.pop(dofs.index(dof))
            if dim==1:
                #
                # 1D Mesh: Fill in interior 
                # 
                dpe = self.element.n_dofs('edge')
                
                if dpe != 0:
                    # Refill dofs to be assigned
                    while len(dofs)<dpe:
                        dofs.append(count)
                        count += 1
                        
                    # Assign new dofs to interval
                    self.assign_dofs(dofs, cell, cell)
                
            elif dim==2:   
                #
                # Assign dofs to HalfEdges
                # 
                dpe = self.element.n_dofs('edge')
                if dpe != 0:
                    for half_edge in cell.get_half_edges():
                        # Refill dofs to be assigned
                        while len(dofs)<dpe:
                            dofs.append(count)
                            count += 1
    
                        # Assign dofs to half_edge 
                        self.assign_dofs(dofs, cell, half_edge)
                        
                        if half_edge.is_periodic():
                            twin = half_edge.twin()
                            twin_dofs = [i for i in reversed(dofs)]
                            self.assign_dofs(twin_dofs, twin.cell(), twin)
                        
                        # Throw out used dofs
                        he_dofs = self.get_cell_dofs(cell, half_edge, interior=True)
                        for dof in he_dofs:
                            if dof in dofs:
                                dofs.remove(dof)
                #
                # Assign dofs to cell interior
                # 
                dpc = self.element.n_dofs('cell')
                if dpc != 0:
                    # Refine cell dofs to be assigned
                    while len(dofs)<dpc:
                        dofs.append(count)
                        count += 1
                        
                    # Assign dofs to cell
                    self.assign_dofs(dofs, cell, cell)
                    
                    # Throw out used dofs
                    cell_dofs = self.get_cell_dofs(cell, cell, interior=True)
                    for dof in dofs:
                        if dof in cell_dofs:
                            dofs.pop(dofs.index(dof))
            
            # Subtract unused dofs from total dof count 
            count -= len(dofs)
                
        else:
            #
            # Augment existing list
            #
            dofs_per_cell = self.element.n_dofs()
            for k in range(dofs_per_cell):
                if cell_dofs[k] is None:
                    cell_dofs[k] = count
                    count += 1
            self.__global_dofs[cell] = cell_dofs
        self.__dof_count = count                 
                    
    
    def assign_dofs(self, dofs, cell=None, entity=None, pos=None):
        """
        Assign the degrees of freedom (dofs) to entities in cell/interval. 
        The result is stored in the DofHandler's "global_dofs" dictionary. 
        
        Inputs:
        
            dofs: int, list of dofs to be assigned
        
            cell: Cell/Interval, within which to assign dofs
            
            entity: Cell/Interval entity indicating the locations at which dofs
                are assigned.
                In 1D: Interval, Vertex
                In 2D: Cell, HalfEdge, Vertex
                
            pos: int, list (same length as dofs) of locations
                at which to assign dofs.         
        """
        dim = self.mesh.dim()
        #
        # Preprocessing
        #
        if not cell in self.__global_dofs:
            #
            # Cell not in dictionary -> add it
            # 
            self.__global_dofs[cell] = [None]*self.element.n_dofs()
        cell_dofs = self.get_cell_dofs(cell)  
        
                
        if pos is not None:
            # =================================================================
            # Simplest way: list of positions and dofs provided
            # =================================================================
            assert len(pos)==len(dofs), \
                'The number of dofs must equal the number of assigned positions'
            for i in range(len(pos)):
                if cell_dofs[pos[i]] is None:
                    cell_dofs[pos[i]] = dofs[i]
                     
        elif entity is None:
            # =================================================================
            # No entity specified, fill in all cell/interval's dofs
            # =================================================================
            # Check that there are enough dofs in the dofs vector
            assert len(dofs)==len(cell_dofs), \
                'Number of dofs should equal number of assigned positions'
            for i in range(len(dofs)):
                if cell_dofs[i] is None:
                    cell_dofs[i] = dofs[i]
        
        elif isinstance(entity, Vertex):
            # =================================================================
            # Fill in dof associated with Vertex (first couple)
            # =================================================================
            i = 0
            assert cell is not None, 'Cell/Interval must be specified.'
            dpv = self.element.n_dofs('vertex')
            assert len(dofs)==dpv, 'Number of dofs should be %d.'%(dpv)
            if dpv==0:
                return
            for vertex in cell.get_vertices():
                if vertex==entity:
                    # Check that spaces are empty
                    for j in range(dpv):
                        if cell_dofs[i+j] is None:
                            cell_dofs[i+j] = dofs[j]   
                    break
                i += dpv
                
        elif isinstance(entity, Interval):
            # =================================================================
            # Dofs associated with Interval
            # =================================================================
            dpv = self.element.n_dofs('vertex')
            dpe = self.element.n_dofs('edge')
            assert len(dofs)==dpe, 'Number of dofs %d incorrect. Should be %d'%(len(dofs),dpe)
            if dpe==0:
                # No dofs associated with HalfEdge
                return
            i = 2*dpv
            for j in range(dpe):
                if cell_dofs[i+j] is None:
                    cell_dofs[i+j] = dofs[j]    
                    
        elif isinstance(entity, HalfEdge) and dim==2:
            # =================================================================
            # Dofs associated with HalfEdge
            # =================================================================
            dpv = self.element.n_dofs('vertex')
            dpe = self.element.n_dofs('edge')
            assert len(dofs)==dpe, 'Number of dofs incorrect for HalfEdge.'
            if dpe==0:
                # No dofs associated with HalfEdge
                return
            cell = entity.cell()
            i = cell.n_vertices()*dpv
            for half_edge in cell.get_half_edges():
                if half_edge==entity:
                    # Check that spaces are empty
                    for j in range(dpe):
                        if cell_dofs[i+j] is None:
                            cell_dofs[i+j] = dofs[j]
                    break
                i += dpe   
                    
        elif entity==cell:
            # =================================================================
            # Dofs associated with cell
            # =================================================================
            dpv = self.element.n_dofs('vertex')
            dpe = self.element.n_dofs('edge')
            dpc = self.element.n_dofs('cell')
            assert len(dofs)==dpc, 'Number of dofs incorrect for Cell'
            if dpc==0:
                # No dofs associated with Cell
                return
            i = cell.n_vertices()*dpv + cell.n_half_edges()*dpe
            #assert all(dof is None for dof in cell_dofs[i:i+dpc]),\
            #    'Cannot overwrite exisiting dofs'
            for j in range(dpc):
                if cell_dofs[i+j] is None:
                    cell_dofs[i+j] = dofs[j]
        else:
            # =================================================================
            # Not recognized
            # =================================================================
            raise Exception('Entity not recognized.')
        self.__global_dofs[cell] = cell_dofs
        
        
    def get_local_dofs(self, cell, entities=None):
        """
        Return the local dofs associated with a given geometric entity 
        (Cell, Interval, HalfEdge, or corner Vertex) within a specified cell. 
        
        The order of counting is:
        
            - Corner Vertices
            - HalfEdges
            - Interior Cell Vertices
        
        Inputs:
        
            cell: Cell/Interval in which entity is contained
            
            entity: Vertex/HalfEdge or Cell/Interval whose local dofs we seek
            
        
        Outputs: 
        
            dofs: int, list of dofs associated with the given entity
            
        TODO: Delete
            
        """
        if entities is None:
            #
            # No entity specified -> return all local dofs
            # 
            return [i for i in range(self.element.n_dofs())]
        
        if not type(entities) is list:
            entities = [entities]
        
        dofs = []
        for entity in entities:
            if isinstance(entity, Vertex):
                #
                # Corner Vertex
                #
                dpv = self.element.n_dofs('vertex')
                cell_vertices = cell.get_vertices()
                assert entity in cell_vertices, 'Vertex not contained in cell.'
                dof_start = cell_vertices.index(entity)*dpv
                dofs += [i for i in range(dof_start, dof_start + dpv)]
            
            elif isinstance(entity, Interval):
                #
                # Interval (Interior Dofs)
                #
                assert entity==cell, \
                    'Entity is an Interval and should equal "cell".'
                dpv = self.element.n_dofs('vertex')
                dpe = self.element.n_dofs('edge')
                if dpe > 0:
                    dof_start = 2*dpv
                    dofs += [i for i in range(dof_start, dof_start+dpe)]
            
            elif isinstance(entity, HalfEdge):
                #
                # HalfEdge
                # 
                dpv = self.element.n_dofs('vertex')
                dpe = self.element.n_dofs('edge')
                if dpe > 0:
                    cell_edges = cell.get_half_edges()
                    assert entity in cell_edges, \
                        'HalfEdge not contained in cell.'
                    i_he = cell_edges.index(entity)
                    dof_start = cell.n_vertices()*dpv + i_he*dpe 
                    dofs += [i for i in range(dof_start, dof_start+dpe)]
                
            elif isinstance(entity, Cell):
                #
                # Cell
                # 
                dpv = self.element.n_dofs('vertex')
                dpe = self.element.n_dofs('edge')
                dpc = self.element.n_dofs('cell')
                if dpc > 0:
                    assert entity==cell, \
                        'Input "entity" is a Cell and should equal input "cell".'
                    dof_start = cell.n_vertices()*dpv + cell.n_half_edges()*dpe
                    dofs += [i for i in range(dof_start, dof_start + dpc)]
        dofs.sort()
        return dofs
       
            
    def get_global_dofs(self, cell=None, entity=None, subforest_flag=None, 
                        mode='breadth-first', all_dofs=False):
        """
        Return all global dofs corresponding to a given cell, or one of its Vertices/HalfEdges 
        edges.
        
        Inputs:
        
            *cell: Cell, whose dofs we seek. 
            
            *entity: Vertex/HalfEdge within cell, whose dofs we seek
            
            *subforest_flag: flag, specifying submesh.
            
            *mode: str, ['breadth-first'] or 'depth-first', mode of mesh traversal
            
            *all_dofs: bool, if True return all dofs in hierarchical mesh
                if False return only dofs associated with LEAF nodes.
            
            
        Outputs:
        
             global_dofs: list of global dofs 
             
        TODO: Delete
        """
        # =====================================================================
        # Get everything
        # =====================================================================
        if cell is None and entity is None: 
         
            mesh_dofs = set()
            if all_dofs:
                #
                # Return all dofs in the hierarchical mesh
                # 
                cells = self.mesh.cells
                for cell in cells.traverse(flag=subforest_flag, mode=mode):
                    mesh_dofs = mesh_dofs.union(self.__global_dofs[cell])
            else:
                #
                # Return only dofs corresponding to LEAF nodes
                # 
                cells = self.mesh.cells
                for cell in cells.get_leaves(subforest_flag=subforest_flag, mode=mode):
                    mesh_dofs = mesh_dofs.union(self.__global_dofs[cell])
            return list(mesh_dofs)
        
        # =====================================================================
        # Cell's Dofs
        # =====================================================================    
        if entity is None:
            if cell is None or cell not in self.__global_dofs:
                return None
            else:
                return self.__global_dofs[cell]
        
        if entity is not None:
            # =================================================================
            # HalfEdge or Vertix's Dof(s)
            # =================================================================
            if cell is None or cell not in self.__global_dofs:
                return None
            
            if isinstance(entity, Vertex):
                # =============================================================
                # Vertex (need a cell)
                # =============================================================
                assert cell is not None, 'Need cell.'
                cell_dofs = self.__global_dofs[cell]
                i = 0
                dpv = self.element.n_dofs('vertex')
                for vertex in cell.get_vertices():
                    if vertex==entity:
                        return cell_dofs[i:i+dpv]
                    i += dpv
            elif isinstance(entity, Interval):
                # =============================================================
                # Interval
                # =============================================================
                assert cell is not None, 'Need cell.'
                cell_dofs = self.__global_dofs[cell]
                dpv = self.element.n_dofs('vertex')
                dpe = self.element.n_dofs('edge')
                i = 2*dpv
                return cell_dofs[i:i+dpe]
            
            elif isinstance(entity, HalfEdge):
                # =============================================================
                # HalfEdge
                # =============================================================
                dpv = self.element.n_dofs('vertex')
                dpe = self.element.n_dofs('edge')
                cell = entity.cell()
                cell_dofs = self.__global_dofs[cell]
                i = cell.n_vertices()*dpv
                for half_edge in cell.get_half_edges():
                    if entity==half_edge:
                        return cell_dofs[i:i+dpe]
                    i += dpe
            elif isinstance(entity, Cell):
                # =============================================================
                # Cell
                # =============================================================
                dpv = self.element.n_dofs('vertex')
                dpe = self.element.n_dofs('edge')
                dpc = self.element.n_dofs('cell')
                cell_dofs = self.__global_dofs[cell]
                i = cell.n_vertices()*dpv + cell.n_half_edges()*dpe
                return cell_dofs[i:i+dpc]
        return None         
    ''' 
    def get_global_dofs(self, cell=None, entity=None, interior=False):
        """
        Return all global dofs corresponding to a given cell, or one of its Vertices/HalfEdges 
        edges.
        
        Inputs:
        
            *cell: Cell, whose dofs we seek. 
            
            *entity: Vertex/HalfEdge within cell, whose dofs we seek
            
            *subforest_flag: flag, specifying submesh.
            
            *mode: str, ['breadth-first'] or 'depth-first', mode of mesh traversal
            
            *all_dofs: bool, if True return all dofs in hierarchical mesh
                if False return only dofs associated with LEAF nodes.
            
            
        Outputs:
        
             global_dofs: list of global dofs 
        """
        # =====================================================================
        # No cell specified -> get all dofs within region
        # =====================================================================
        if cell is None:
            mesh_dofs = set()
            cells = self.mesh.cells
            if hierarchical_traversal:
                for cell in cells.traverse(flag=subforest_flag, mode=mode):
                    #
                    # Traverse all cells in hierarchichal mesh
                    # 
                    if entity is None:
                        #
                        # No entity is specified -> return all dofs in the cell
                        # 
                        mesh_dofs = mesh_dofs.union(self.__global_dofs[cell])
                    elif 
            else:
                for cell in cells.get_leaves(subforest_flag=subforest_flag, \
                                             mode=mode):
                    #
                    # Get only LEAF cells
                    # 
                    
                    
                
            if entity is None:
                
              
         
            mesh_dofs = set()
            if all_dofs:
                #
                # Return all dofs in the hierarchical mesh
                # 
                cells = self.mesh.cells
                for cell in cells.traverse(flag=subforest_flag, mode=mode):
                    mesh_dofs = mesh_dofs.union(self.__global_dofs[cell])
            else:
                #
                # Return only dofs corresponding to LEAF nodes
                # 
                cells = self.mesh.cells
                for cell in cells.get_leaves(subforest_flag=subforest_flag, mode=mode):
                    mesh_dofs = mesh_dofs.union(self.__global_dofs[cell])
            return list(mesh_dofs)
        
        # =====================================================================
        # Cell is specified
        # =====================================================================    
        if entity is None:
            if cell is None or cell not in self.__global_dofs:
                return None
            else:
                return self.__global_dofs[cell]
        
        if entity is not None:
            # =================================================================
            # HalfEdge or Vertix's Dof(s)
            # =================================================================
            if cell is None or cell not in self.__global_dofs:
                return None
            
            if isinstance(entity, Vertex):
                # =============================================================
                # Vertex (need a cell)
                # =============================================================
                assert cell is not None, 'Need cell.'
                cell_dofs = self.__global_dofs[cell]
                i = 0
                dpv = self.element.n_dofs('vertex')
                for vertex in cell.get_vertices():
                    if vertex==entity:
                        return cell_dofs[i:i+dpv]
                    i += dpv
            elif isinstance(entity, Interval):
                # =============================================================
                # Interval (interior)
                # =============================================================
                assert cell is not None, 'Need cell.'
                cell_dofs = self.__global_dofs[cell]
                dpv = self.element.n_dofs('vertex')
                dpe = self.element.n_dofs('edge')
                i = 2*dpv
                return cell_dofs[i:i+dpe]
            
            elif isinstance(entity, HalfEdge):
                # =============================================================
                # HalfEdge
                # =============================================================
                dpv = self.element.n_dofs('vertex')
                dpe = self.element.n_dofs('edge')
                cell = entity.cell()
                cell_dofs = self.__global_dofs[cell]
                i = cell.n_vertices()*dpv
                for half_edge in cell.get_half_edges():
                    if entity==half_edge:
                        return cell_dofs[i:i+dpe]
                    i += dpe
            elif isinstance(entity, Cell):
                # =============================================================
                # Cell (interior)
                # =============================================================
                dpv = self.element.n_dofs('vertex')
                dpe = self.element.n_dofs('edge')
                dpc = self.element.n_dofs('cell')
                cell_dofs = self.__global_dofs[cell]
                i = cell.n_vertices()*dpv + cell.n_half_edges()*dpe
                return cell_dofs[i:i+dpc]
        return None
    '''    
        
    def get_cell_dofs(self, cell, entity=None, interior=False, doftype='global', 
                      subforest_flag=None):
        """
        Returns all (global/local) dofs of a specific entity within a cell
        
        
        Inputs:
        
            cell: Cell, cell within which to seek global dofs 
            
            entity: Vertex/HalfEdge, sub-entity
            
            interior: bool, if True only return dofs associated with entity interior
            
                cell: No dofs associated with half-edges or vertices
                
                half_edge: No dofs associated with endpoint vertices
                
                interval: No dofs associated with endpoint vertices 

            subforest_flag: str/int/tuple, 
            
        Outputs:
        
            dofs: int, list of global dofs associated with entity within
                cell.
                
        Note: Replaces ```get_global_dofs``` method
        """
        if doftype=='global':
            #
            # Global dofs sought
            # 
            # Check that we have global dofs
            if cell is None or cell not in self.__global_dofs:
                return None
            
            #
            # Check whether we are on a submesh
            # 
            if subforest_flag is not None:
                cell_dofs = self.__global_dofs[subforest_flag][cell]
            else:
                cell_dofs = self.__global_dofs[cell]
        elif doftype=='local':
            #
            # Local dofs sought
            # 
            cell_dofs = [i for i in range(self.element.n_dofs())]
            if cell is None:
                return cell_dofs
             
        if entity is None:
            # =================================================================
            # No entity specified: Return all cell dofs
            # =================================================================
            dofs = cell_dofs
        
        elif isinstance(entity, Vertex):
            # =============================================================
            # Vertex
            # =============================================================
            i = 0
            dpv = self.element.n_dofs('vertex')
            for vertex in cell.get_vertices():
                if vertex==entity:
                    dofs = cell_dofs[i:i+dpv]
                    break
                i += dpv
                
        elif isinstance(entity, Interval):
            # =============================================================
            # Interval
            # =============================================================
            if not interior:
                #
                # All dofs within interval (equivalent to entity=None)
                # 
                dofs = cell_dofs
            else:
                #
                # Inly dofs interior to interval
                #  
                dpv = self.element.n_dofs('vertex')
                dpe = self.element.n_dofs('edge')
                i = 2*dpv
                dofs = cell_dofs[i:i+dpe]
        
        elif isinstance(entity, HalfEdge):
            # =============================================================
            # HalfEdge
            # =============================================================
            # Dofs per entity
            dpv = self.element.n_dofs('vertex')
            dpe = self.element.n_dofs('edge')
            
            dofs = []
            if not interior:
                #
                # Return all dofs within halfedge (including endpoints)   
                #
                end_points = entity.get_vertices()
                i = 0
                for v in cell.get_vertices():
                    #
                    # Add endpoint vertices
                    # 
                    if v in end_points:
                        dofs.append(cell_dofs[i])
                    i += dpv 
            else:
                #
                # Return only dofs interior to half-edge
                # 
                i = cell.n_vertices()*dpv
            #
            # Add dofs on the interior of the edge
            # 
            for half_edge in cell.get_half_edges():
                if entity==half_edge:
                    dofs.extend(cell_dofs[i:i+dpe])
                    break
                i += dpe
                    
        elif isinstance(entity, Cell):
            # =============================================================
            # Cell 
            # =============================================================
            # Dofs per entity
            dpv = self.element.n_dofs('vertex')
            dpe = self.element.n_dofs('edge')
            dpc = self.element.n_dofs('cell')
            
            if not interior:
                #
                # Return all cell dofs (same as entity=None)
                # 
                dofs = cell_dofs
            else:
                #
                # Return only dofs interior to cell
                #
                i = cell.n_vertices()*dpv + cell.n_half_edges()*dpe
                dofs = cell_dofs[i:i+dpc]
        #
        # Return list of dofs
        # 
        return dofs
    
    
    def get_region_dofs(self, entity_type='cell', entity_flag=None, 
                        interior=False, on_boundary=False, subforest_flag=None):
        """
        Returns all global dofs of a specific entity type within a mesh region.
        
        
        Inputs: 
        
            entity_type: str, specifying the type of entities whose dofs we seek.
                If None, then return all dofs within cell. Possible values:
                'cell', 'half_edge', 'interval', 'vertex'
            
            entity_flag: str/int/tuple, marker used to specify subset of entities
                
            interior: bool, if True only return dofs associated with entity interior
                (See "get_cell_global_dofs")
                
            on_boundary: bool, if True, seek only dofs on the boundary
            
            subforest_flag: str/int/tuple, submesh marker
                
                
        Output:
        
            dofs: list of all dofs associated with region
        """
        dofs = set()
        for entity, cell in self.mesh.get_region(entity_flag, entity_type, 
                                                 on_boundary=on_boundary,
                                                 subforest_flag=subforest_flag,
                                                 return_cells=True):
            #
            # Iterate over all entities within region
            # 
            new_dofs = self.get_cell_dofs(cell, entity, interior=interior)
            dofs = dofs.union(new_dofs)
        return list(dofs)
    
    
    def set_dof_vertices(self, subforest_flag=None):
        """
        Construct a list of all vertices in the mesh
        """
        # Check if dof_vertices already set
        if len(self.__dof_vertices) >= self.n_dofs(subforest_flag=subforest_flag):
            return
        
        # Reference nodes
        x_ref = self.element.reference_nodes()
        
        # Iterate over mesh
        for cell in self.mesh.cells.traverse(flag=subforest_flag):
                        
            # Retrieve cell's dofs
            dofs = self.get_cell_dofs(cell)
            
            if not all([dof in self.__dof_vertices for dof in dofs]):
                #
                # Not all dofs in cell have been a assigned a vertex
                # 
                
                # Compute cell vertices
                x = cell.reference_map(x_ref)
            
                # Add vertices to dictionary
                for i in range(len(dofs)):
                    dof = dofs[i]
                    if dof not in self.__dof_vertices:
                        self.__dof_vertices[dof] = {}
                        self.__dof_vertices[dof][cell] = x[i,:]
                    else:
                        if any([np.allclose(x[i,:], vertex) for vertex in \
                                self.__dof_vertices[dof].values()]):
                            #
                            # Periodic vertex
                            #
                            self.__dof_vertices[dof][cell] = x[i,:] 
                        
    
    def get_dof_vertices(self, dofs=None, cells=None, subforest_flag=None, \
                         as_array=True):
        """
        Returns a list of vertices corresponding to a given list of dofs
        
        Inputs:
        
            dofs: int, list of global dofs
            
            *cells [None]: dictionary of type {dof: Cell}, specifying the cell
            in which the dofs must occur. 
        """
        is_singleton = False
        if type(dofs) is np.int:
            dofs = [dofs]
            is_singleton = True
        
        if dofs is None:
            dofs = self.get_region_dofs(subforest_flag=subforest_flag)
        vertices = []
        for dof in dofs:
            if len(self.__dof_vertices[dof])==1:
                #
                # Only one cell for given dof 
                # 
                for vertex in self.__dof_vertices[dof].values():
                    vertices.append(vertex)
            else:
                #
                # Multiple cells contain the same dof 
                # 
                if cells is not None:
                    if dof in cells:
                        #
                        # If cell specified, return the `right' vertex
                        # 
                        vertices.append(self.__dof_vertices[dof][cells[dof]])
                else:
                    # 
                    # No cell specified, return the first vertex
                    #
                    for vertex in self.__dof_vertices[dof].values():
                        vertices.append(vertex)
                        break        
        if is_singleton:
            return vertices[0]
        else:
            if as_array:
                #
                # Return vertices as an (n_dof, dim) array
                # 
                return convert_to_array(vertices, self.element.dim())
            else:
                #  
                # Return vertices as an n_dof list of (1,dim) arrays
                #
                return vertices
        
        
    
    def n_dofs(self, all_dofs=False, subforest_flag=None):
        """
        Return the total number of degrees of freedom distributed so far
        """
        if all_dofs and subforest_flag is None:
            #
            # Total number of dofs known immediately
            # 
            return self.__dof_count
        #
        # Extract the dofs and compute the length
        #
        dofs = self.get_global_dofs(all_dofs=all_dofs, subforest_flag=subforest_flag)
        return len(dofs)
      
      
    def set_l2g_map(self, subforest_flag=None):
        """
        Set up the mapping expressing the global basis functions in terms of
        local ones. This is used during assembly to ensure that the solution
        is continuous across edges with hanging nodes
        
        
        Input:
        
            subforest_flag: str/int/tuple, marker specifiying submesh.
            
            
        Outputs:
        
            None
            
        
        Internal:
        
            self.__l2g[subforest_flag]: dictionary, indexed by cells, whose 
                entries l2g[cell] are themselves dictionaries, indexed by 
                global dofs and whose values are the numpy arrays of 
                coefficients of the associated global basis function in terms 
                of the local basis.
                
                
        Note: 
        
            Ideally, this should be done within the ```set_hanging_nodes```
            method. 
            
        TODO: I think this is not necessary
        
        """  
        mesh = self.mesh
        element = self.element
        l2g = dict.fromkeys(mesh.cells.get_leaves(subforest_flag=subforest_flag))
        for cell in mesh.cells.get_leaves(subforest_flag=subforest_flag):
            n_dofs = element.n_dofs()
            gdofs = self.get_cell_dofs(cell)
            #
            # Initialize local-to-global map
            # 
            if l2g[cell] is None:
                #
                # Dictionary keys are global dofs in cell
                # 
                l2g[cell] = dict.fromkeys(gdofs)
                #
                # Values are expansion coefficients ito local basis
                # 
                I = np.identity(n_dofs)
                for i in range(n_dofs):
                    l2g[cell][gdofs[i]] = I[i,:]
            
            if self.mesh.dim()==1:
                #
                # One dimensional mesh has no hanging nodes
                # 
                continue
            #
            # Search for hanging nodes
            #     
            for he in cell.get_half_edges():
                if he.twin() is not None and he.twin().has_children():
                    #
                    # Edge with hanging nodes
                    # 
                    
                    # Collect dofs on long edge
                    le_ldofs = self.get_cell_dofs(cell, entity=he, doftype='local')
                    le_gdofs = [gdofs[ldof] for ldof in le_ldofs]
                    
                    #
                    # Iterate over subtending cells
                    # 
                    twin = he.twin()
                    for che in twin.get_children():
                        subcell = che.cell()
                        #
                        # Initialize mapping for sub-cell
                        # 
                        if l2g[subcell] is None:
                            #
                            # Dictionary keys are global dofs in cell
                            # 
                            sc_gdofs = self.get_cell_dofs(subcell)
                            l2g[subcell] = dict.fromkeys(sc_gdofs)
                            #
                            # Values are expansion coefficients ito local basis
                            # 
                            I = np.identity(n_dofs)
                            for i in range(n_dofs):
                                l2g[subcell][sc_gdofs[i]] = I[i,:]
                                
                        # =============================================================
                        # Expansion coefficients of global basis function on sub-cell 
                        # =============================================================
                    
                        #    
                        # Local dofs on sub-edge
                        #    
                        se_ldofs = self.get_cell_dofs(subcell, entity=che, \
                                                      doftype='local')
                        
                        #
                        # Get vertices associated with these local dofs
                        # 
                        rsub = element.reference_nodes()[se_ldofs,:]
                        x = subcell.reference_map(rsub, mapsto='physical')
                        
                        #
                        # Evaluate coarse scale basis functions at fine scale vertices
                        # 
                        r = cell.reference_map(x, mapsto='reference')
                        
                        for le_ldof, le_gdof in zip(le_ldofs, le_gdofs):
                            #
                            # Evaluate global basis function at all sub-edge dof-verts
                            # 
                            vals = element.phi(le_ldof,r)
                            coefs = np.zeros(n_dofs)
                            coefs[se_ldofs] = vals
                            l2g[subcell][le_gdof] = coefs
                            
                        #
                        # Constrain hanging node dofs
                        #
                        # Global dofs on small edge
                        se_gdofs = [sc_gdofs[ldof] for ldof in se_ldofs]
                        for se_ldof, se_gdof in zip(se_ldofs, se_gdofs):
                            if se_gdof not in le_gdofs:
                                #
                                # Hanging Dof must be adjusted 
                                # 
                                l2g[subcell][se_gdof]
        #
        # Store local-to-global mapping in a dictionary                    
        # 
        if subforest_flag is None:
            self.__l2g = l2g
        else: 
            self.__l2g[subforest_flag] = l2g
            
            
    def get_l2g_map(self, cell, subforest_flag=None):
        """
        Return the local to global map for a particular cell
        """
        if subforest_flag is not None:
            #
            # Submesh
            # 
            # Check that l2g mapping is present
            assert subforest_flag in self.__l2g.keys(), \
                'First run "set_l2g_map" for this submesh.'
            
            return self.__l2g[subforest_flag][cell]
        else:
            #
            # On entire mesh
            # 
            # Check that l2g mapping is present
            assert len(self.__l2g) != 0, \
                'First run "set_l2g_map". ' 
            
            return self.__l2g[cell]
    
    
    def set_hanging_nodes(self, subforest_flag=None):
        """
        Set up the constraint matrix satisfied by the mesh's hanging nodes.
        
        Note: 
        
            - Hanging nodes can only be found once the mesh has been balanced.
        
            - Hanging nodes are never periodic
        """
        if self.mesh.dim()==1:
            #
            # One dimensional meshes have no hanging nodes
            # 
            return 
        
        #
        # Discontinuous elements don't have hanging nodes
        # 
        if self.element.torn_element(): 
            return 
        hanging_nodes = {}
        for cell in self.mesh.cells.get_leaves(subforest_flag=subforest_flag):
            for half_edge in cell.get_half_edges():
                #
                # Search in all directions
                # 
                nb = cell.get_neighbors(half_edge, flag=subforest_flag)
                if nb is not None:
                    if nb.has_children(flag=subforest_flag):
                        #
                        # Neighbor has children -> Look for hanging nodes
                        #
                        dofs = self.get_global_dofs(nb)
                        twin = half_edge.twin() 
        
                        #
                        # Look for equivalent half_edge in reference cell
                        #
                        i_twin = nb.get_half_edges().index(twin) 
                        reference_cell = self.element.reference_cell()
                        rhe = reference_cell.get_half_edge(i_twin)
                        
                        #
                        # Determine indices of the supporting dofs
                        # 
                        coarse_vertices = [rhe.base(), rhe.head()]
                        coarse_vertices.extend(rhe.get_edge_dof_vertices())
                        
                        i_supp_dofs = set()
                        for v in coarse_vertices:
                            i_supp_dofs.add(v.get_pos(0)) 
                        supporting_dofs = [dofs[i] for i in i_supp_dofs]
                        #
                        # Determine the hanging nodes
                        #
                        for i_rhe_ch in range(2):
                            # 
                            # Loop over sub-halfedges
                            #
                            rhe_ch = rhe.get_child(i_rhe_ch)
                            fine_vertices = [rhe_ch.base(), rhe_ch.head()]
                            fine_vertices.extend(rhe_ch.get_edge_dof_vertices())
                            for v in fine_vertices:
                                if v.get_pos(0) is None:
                                    # 
                                    # Hanging node
                                    #
                                    
                                    # Compute constraint coefficients  
                                    constraint_coefficients = []
                                    for i in i_supp_dofs:
                                        constraint_coefficients.append(self.element.phi(i, v)[0])
                                    #constraint_coefficients = np.array(constraint_coefficients)
                                    
                                    # Store in dictionary
                                    he_ch = twin.get_child(i_rhe_ch)
                                    hn_dof = self.get_global_dofs(he_ch.cell())
                                    i_child = rhe_ch.cell().get_node_position()
                                    constrained_dof = hn_dof[v.get_pos(1, i_child)]
                                    
                                    hanging_nodes[constrained_dof] = \
                                        (supporting_dofs, constraint_coefficients)
                                    
                                    if constrained_dof not in self.constraints['constrained_dofs']:
                                        #
                                        # Update Constraints
                                        # 
                                        self.constraints['constrained_dofs'].append(constrained_dof)
                                        self.constraints['supporting_dofs'].append(supporting_dofs)
                                        self.constraints['coefficients'].append(constraint_coefficients)
                                        self.constraints['affine_terms'].append(0)
        #
        # Store all hanging nodes in dictionary
        #  
        if subforest_flag is not None:
            self.__hanging_nodes[subforest_flag] = hanging_nodes
        else:
            self.__hanging_nodes = hanging_nodes                

      
    def get_hanging_nodes(self, subforest_flag=None):
        """
        Returns hanging nodes of current mesh
        """
        if hasattr(self,'__hanging_nodes'): 
            if subforest_flag is None:
                return self.__hanging_nodes
            else:
                return self.__hanging_nodes[subforest_flag]
        else:
            self.set_hanging_nodes()
            return self.__hanging_nodes
        
    
    def has_hanging_nodes(self, subforest_flag=None):
        """
        Determine whether there are hanging nodes
        """
        if self.mesh.dim()==1:
            #
            # One dimensional meshes don't have hanging nodes
            # 
            return False
            
        elif self.__hanging_nodes == {}:
            #
            # 2D Mesh with no hanging nodes
            # 
            return False
            
        else:
            #
            # 2D Mesh with hanging nodes
            # 
            if subforest_flag is not None:
                #
                # Hanging nodes for subforest flag? 
                # 
                return subforest_flag in self.__hanging_nodes
            else:
                #
                # Has hanging nodes, return True
                # 
                return True
            
        
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
            
            element: FiniteElement object
            
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
    def __init__(self, f, dfdx=None, F=None, samples='all'):
        """
        Constructor
        
        Inputs:
        
            f: single Function, or list of Functions
            
            dfdx: list of strings, derivatives of function to be evaluated
                In 1D: 'f', 'fx', 'fxx'
                In 2D: 'f', 'fx', 'fy', 'fxy', 'fyx', 'fxx', 'fyy'
                The number of strings in dfdx should equal the number of 
                functions in f. If no derivatives are to be computed, use
                dfdx = None. 
            
            *samples: 'all', or numpy integer array specifying what samples to
                take. 
            
            *F: function, lambda function describing how the f's are modified 
                to form the kernel
        """
        
        # 
        # Store input function(s)
        #
        if type(f) is not list:
            assert isinstance(f, Function), \
                'Input "f" should be a (tuple of) Function(s).'
            f = [f]
        self.f = f
        n_functions = len(f)
        
        #
        # Store derivatives
        #     
        if dfdx is None:
            #
            # No derivative specified -> simple function evaluation
            #
            dfdx = [parse_derivative_info('f')]*n_functions 
        elif type(dfdx) is str:
            #
            # Only one derivative specified -> apply to all functions
            #
            dfdx = [parse_derivative_info(dfdx)]*n_functions
        elif type(dfdx) is list:
            #
            # Derivatives passed in list -> check for compatibility
            #
            assert len(dfdx)==n_functions, \
                'Number of derivatives in "dfdx" should equal n_functions'
            dfdx = [parse_derivative_info(der) for der in dfdx]
        self.dfdx = dfdx
        
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
        self.F = F
        
        #
        # Check that samples are compatible with functions
        #
        n_samples = None
        if samples=='all':
            #
            # Compute all samples
            # 
            for f in self.f:
                #
                # Check that all function have the same number of samples
                # 
                if f.n_samples() is not None:
                    if n_samples is None:
                        #
                        # Initialize number of samples 
                        # 
                        n_samples = f.n_samples()
                    elif f.n_samples()>1:
                        #
                        # function has more than one sample
                        # 
                        assert f.n_samples()==n_samples,\
                            'Kernel function sample sizes incompatible.'
        elif type(samples) is np.ndarray:
            #
            # Subsample: Check that every sampled function contains subsample 
            #
            n_samples = len(samples)
            n_max = samples.max()
            for f in self.f:
                if f.n_samples() is not None:
                    #
                    # Maximum sample index may not exceed sample size
                    # 
                    assert n_max >= f.n_samples(), \
                        'Maximum sample index exceeds sample size.'
        #
        # Store samples
        # 
        self.samples = samples
        self.n_samples = n_samples
                        
    
    def eval(self, x=None, cell=None, phi=None, dofs=None, 
             compatible_functions=set()):
        """
        Evaluate the kernel at the points stored in x 
        
        Inputs:
        
            *x: (n_points, dim) array of points at which to evaluate the kernel
            
            *cell: Cell/Interval within which the points x reside
            
            *phi: dictionary of shape functions used to evaluate 
                nodal constituent functions f. Must also specify dofs
                 
            *dofs: list of (n_dofs,) degrees of freedom associated with shape
                 functions
                 
            *compatible_functions: set of functions determined (a priori) to
                be compatible with the current mesh. These functions can 
        """
        x = convert_to_array(x)
        #
        # Determine dimensions of output array
        #
        n_points = x.shape[0]
        n_samples = self.n_samples
        
        #
        # Evaluate constituent functions 
        # 
        f_vals = []
        for f, dfdx in zip(self.f, self.dfdx):
            if f in compatible_functions: 
                etype = f.dofhandler.element.element_type()
                #
                # f may be evaluated through shape functions
                # 
                fv = f.eval(phi=phi[etype][dfdx], dofs=dofs[etype], derivative=dfdx, \
                            samples=self.samples)
            else:
                #
                # f must be evaluated explicitly
                # 
                fv = f.eval(x=x, cell=cell, derivative=dfdx, \
                            samples=self.samples)
            if n_samples is not None:
                if f.n_samples() is None:
                    #
                    # Deterministic function
                    # 
                    fv = (np.ones((n_samples,n_points))*fv).T
            f_vals.append(fv)
        #
        # Combine functions using metafunction F 
        # 
        return self.F(*f_vals)

        
    
class Basis(object):
    """
    Finite element basis function
    """
    def __init__(self, element, derivative='v'):
        """
        Constructor
        
        Inputs:
        
            element: FiniteElement, element associated with basis function
            
            derivative: str, derivative of the basis function 
                'v', 'vx', 'vy', 'vxy', 'vyx', or 'vyy'
                (first letter is irrelevant)
        """
        self.element = element
        self.derivative = parse_derivative_info(derivative)
        
    
class Form(object):
    """
    Constant, Linear, or Bilinear forms (integrals)
    """
    def __init__(self, kernel=None, trial=None, test=None, dmu='dx',\
                 flag=None, samples=None):
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
            
            *samples: integer array o
            
        """
        # TODO: Check that the dimensions are consistent
        
        #
        # Parse trial function
        # 
        if trial is not None:
            assert isinstance(trial, Basis), \
            'Input "trial" must be of type "Basis".'  
        self.trial = trial
        
        #
        # Parse test function
        # 
        if test is not None:
            assert isinstance(test, Basis), \
            'Input "test" must be of type "Basis".'
        self.test = test
        
        #
        # Parse measure
        # 
        assert dmu in ['dx', 'ds', 'dv'], \
        'Input "dmu" should be "dx", "ds", or "dv".'
        
        # TODO: Add Check: ds can only be used in 2D.
         
        self.dmu = dmu
        
        #
        # Parse kernel
        # 
        if kernel is not None:
            #
            # Check that kernel is the right type
            # 
            assert isinstance(kernel, Kernel), \
            'Input "kernel" must be of class "Kernel".'
        else:
            #
            # Default Kernel
            # 
            kernel = Kernel(Function(1, 'constant'))
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
        
        
            compatible_functions: set, of Functions that are defined on 
            
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
            for (f, dfdx) in zip(self.kernel.f, self.kernel.dfdx):
                #
                # Iterate over constituent functions
                # 
                if f in compatible_functions:
                    # 
                    # function is compatible with mesh
                    # 
                    etype = f.dofhandler.element.element_type()
                    if etype not in info:
                        #
                        # Add element type of kernel function
                        # 
                        info[etype] = {'element': f.dofhandler.element,
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
            Ker = kernel.eval(x=xg[region], cell=cell, phi=phi[region], dofs=dofs,\
                              compatible_functions=compatible_functions)
            
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
                test = phi[region][test_etype][test_der]
                n_dofs_test = test.shape[1]
                
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
                test = phi[region][test_etype][test_der]
                n_dofs_test = test.shape[1]
                
                # Define the trial function
                trial_der = self.trial.derivative
                trial_etype = self.trial.element.element_type()
                trial = phi[region][trial_etype][trial_der]
                n_dofs_trial = trial.shape[1]
                
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
        
    def bilinear_loc(self,weight,kernel,trial,test):
        """
        Compute the local bilinear form over an element
        
        TODO: DELETE
        """
        if len(kernel.shape)==1:
            #
            # Deterministic kernel
            # 
            B_loc = np.dot(test.T, np.dot(np.diag(weight*kernel),trial))
        else:
            #
            # Sampled kernel
            # 
            B_loc = []
            n_sample = kernel.shape[1]
            for i in range(n_sample):
                B_loc.append(np.dot(test.T, np.dot(np.diag(weight*kernel[:,i]),trial)))
        return B_loc
    
    
    def linear_loc(self,weight,kernel,test):
        """
        Compute the local linear form over an element
        
        TODO: DELETE!
        """
        if len(kernel.shape)==1:
            #
            # Deterministic kernel
            # 
            L_loc = np.dot(test.T, weight*kernel)
        else:
            #
            # Sampled kernel
            # 
            L_loc = []
            n_sample = kernel.shape[1]
            for i in range(n_sample):
                L_loc.append(np.dot(test.T, weight*kernel[:,i]))
        return L_loc        
        
    
class Assembler(object):
    """
    Representation of sums of bilinear/linear forms as matrices/vectors  
    
    Attributes:
    
        problems
        
        single_form
        
        single_problem 
        
        mesh
        
        subforest_flag
        
        n_gauss1d
        
        n_gauss2d
        
        cell_rule
        
        edge_rule
        
        dofhandlers
        
        assembled_forms
    """
    def __init__(self, problems, mesh, subforest_flag=None, n_gauss=(4,16)):
        """
        Assemble a finite element system
        
        Inputs:
        
            problems: list of bilinear, linear, or constant Forms
        
            mesh: Mesh, finite element mesh
            
            subforest_flag: submesh marker over which to assemble
                        
            n_gauss: int tuple, number of quadrature nodes in 1d and 2d respectively
                        
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
        Initialize list of dictionaries encoding the assembled forms associated
        with each problem
        """
        af = []
        for problem in self.problems:
            af_problem = {}
            for form in problem:
                if form.type=='constant':
                    #
                    # Constant type forms
                    # 
                    if 'constant' not in af_problem:
                        af_problem['constant'] = 0
                elif form.type=='linear':  
                    #
                    # Linear form
                    # 
                    if 'linear' not in af_problem:
                        af_problem['linear'] = {'row_dofs': [], 
                                                'vals': [],
                                                'test_etype': None}
                        
                    if af_problem['linear']['test_etype'] is None:
                        #
                        # Test element type not yet specified
                        # 
                        af_problem['linear']['test_etype'] = \
                            form.test.element.element_type() 
                    else:
                        #
                        # Check for test element consistency
                        # 
                        assert af_problem['linear']['test_etype']\
                            ==form.test.element.element_type(), \
                            'All linear forms in problem should have the '+\
                            'same test function space.'
                elif form.type=='bilinear':
                    #
                    #  Bilinear form     
                    # 
                    if 'bilinear' not in af_problem:
                        #
                        # Initialize bilinear assembled form
                        #
                        af_problem['bilinear'] = \
                            {'row_dofs': [], 'col_dofs': [], 'vals': [],\
                             'test_etype': None, 'trial_etype': None}
                            
                    if af_problem['bilinear']['test_etype'] is None:
                        #
                        # Test element type not yet specified
                        # 
                        af_problem['bilinear']['test_etype'] = \
                            form.test.element.element_type()
                    else:
                        #
                        # Check for test element consistency
                        # 
                        assert af_problem['bilinear']['test_etype']\
                            ==form.test.element.element_type(),\
                            'All bilinear forms in problem should have '+\
                            'the same test function space.'
                        
                    if af_problem['bilinear']['trial_etype'] is None:
                        #
                        # Trial element type not yet specified
                        # 
                        af_problem['bilinear']['trial_etype'] = \
                            form.trial.element.element_type()
                    else:
                        #
                        # Check for trial element consistency
                        # 
                        assert af_problem['bilinear']['trial_etype']\
                            ==form.trial.element.element_type(),\
                            'All bilinear forms in problem should have '+\
                            'the same trial function space.'
                        
            af.append(af_problem)
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
                for f in form.kernel.f:
                    if f.mesh_compatible(self.mesh, \
                                         subforest_flag=self.subforest_flag):
                        #
                        # Function is nodal and defined over submesh
                        # 
                        compatible_functions.add(f)
                        element = f.dofhandler.element 
                        elements.add(element)
                        etype = element.element_type()
                        if etype not in dofhandlers:
                            #
                            # Add existing dofhandler to list of dofhandlers
                            # 
                            dofhandlers[etype] = f.dofhandler
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
            
            
            
    def assemble(self, consolidate=True):
        """
        Assembles constant, linear, and bilinear forms over computational mesh,

        
        Input:
        
            problems: A list of finite element problems. Each problem is a list
                of constant, linear, and bilinear forms. 
                 
               
        Output:
        
            assembled_forms: list of dictionaries (one for each problem), each of 
                which contains:
                
                'bf': dictionary summarizing assembled bilinear forms with fields
                    
                    'i': list of row entries 
                    
                    'j': list of column entries
                    
                    'val': list of matrix values 
                    
                    'dir_dofs': set, consisting of all dofs corresponding to 
                        Dirichlet vertices
            
                'lf': vector (or matrix), of assembled linear forms
                    
                
            A: double coo_matrix, system matrix determined by bilinear forms and 
                boundary conditions.
                
            b: double, right hand side vector determined by linear forms and 
                boundary conditions.
        
        
        """                 
        #
        # Assemble forms over mesh cells
        #            
        for cell in self.mesh.cells.get_leaves(subforest_flag=self.subforest_flag):
            #
            # Get global cell dofs for each element type  
            #
            cell_dofs = self.cell_dofs(cell)
            
            #
            # Determine what shape functions and Gauss rules to compute on current cells
            # 
            shape_info = self.shape_info(cell)
            
            # 
            # Compute Gauss nodes and weights on cell
            # 
            xg, wg = self.gauss_rules(shape_info)
            
            #
            # Compute shape functions on cell
            #  
            phi = self.shape_eval(shape_info, xg, cell)
            
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
                    form_loc = form.eval(cell, xg, wg, phi, cell_dofs, \
                                         self.compatible_functions)                   
                    
                    if form.type=='constant':
                        #
                        # Constant form
                        # 
                        
                        # Increment value
                        self.af[i_problem]['constant'] += form_loc 
                    elif form.type=='linear':
                        # 
                        # Linear form
                        # 
                        
                        # Extract test dof indices
                        etype_tst = form.test.element.element_type()
                        dofs_tst  = cell_dofs[etype_tst]
                                
                        # Store dofs and values in assembled_form
                        self.af[i_problem]['linear']['row_dofs'].append(dofs_tst)
                        self.af[i_problem]['linear']['vals'].append(form_loc)
                        
                    elif form.type=='bilinear':
                        #
                        # Bilinear Form
                        # 
                        
                        # Test dof indices
                        etype_tst = form.test.element.element_type()
                        etype_trl = form.trial.element.element_type()
                        
                        # Trial dof indices
                        dofs_tst = cell_dofs[etype_tst]
                        dofs_trl = cell_dofs[etype_trl]    
                        
                        # Store dofs and values in assembled form 
                        self.af[i_problem]['bilinear']['row_dofs'].append(dofs_tst)
                        self.af[i_problem]['bilinear']['col_dofs'].append(dofs_trl)
                        self.af[i_problem]['bilinear']['vals'].append(form_loc)
        #
        # Post-process assembled forms
        #  
        if consolidate:
            self.consolidate_assembly()
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
        
            n_samples: int, number of samples associated with the given
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
                        # New non-trivial sample size
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
        
        
        
class LinearSystem(object):
    """
    Linear system object consisting of a single coefficient matrix and right 
    hand side vector, together with the associated dof information.
    
    Attributes:
    
        __A
        
        assembler
        
        __b
        
        __compressed
        
        __dirichlet
        
        __dofs
        
        __etype
        
        __hanging_nodes: dict, {i_hn:[is_1,...,is_k], [cs_1,...,cs_k]}
            i_hn: hanging node index
            is_j: index of jth supporting node
            cs_j: coefficient of jth supporting basis function, i.e.
            
            phi_{i_hn} = cs_1*phi_{is_1} + ... + cs_k*phi_{is_k}
                 
        __i_problem
        
    
        
        
    Methods:
    
        dofs: returns system dofs
        
        A: returns system matrix 
        
        b: returns right hand side vector 
        
        has_hanging_nodes: reveals presence of hanging nodes 
        
        is_compressed: indicates whether dirichlet- and hanging
            node dofs are to be removed from system.
        
        extract_dirichlet_nodes: incorporates dirichlet conditions
        
        extract_hanging_nodes: incorporates hanging nodes into system 
        
        resolve_dirichlet_nodes: assigns dirichlet values to 
            compressed solution 
        
        resolve_hanging_nodes: applies hanging node interpolations to
            compressed solution
        
        restrict: 
        
        interpolate
     
    """
    def __init__(self, assembler, i_problem=0, i_sample=(0,0)):
        """
        Constructor
        
        
        Inputs:
        
            assembler: Assembler, 
            
            i_problem: int (default 0), problem index
            
            i_sample: int, tuple (i_bilinear_sample, i_linear_sample), 
                specifying the index of the assembled bilinear and 
                linear forms to be used to construct the system.
                
            compressed: bool (default=False), indicating whether unknowns
                which can be determined by Dirichlet boundary conditions
                or hanging node interpolation relations, should be removed
                from the linear system Ax=b.
                
        """
        self.assembler = assembler
        self.__i_problem = i_problem
        
        #
        # Extract forms
        # 
        problem = assembler.af[i_problem]
        bilinear_form = problem['bilinear']
        linear_form = problem['linear']
        
        #
        # Determine element type  
        # 
        etype = bilinear_form['trial_etype'] 

        #
        # Check that the element type is consistent 
        #  
        assert etype==bilinear_form['test_etype'], \
        'Trial and test spaces must have the same element type.'
        
        assert etype==linear_form['test_etype'],\
        'Test and trial spaces must have same element type.'

        self.__etype = etype
        
        
        #
        # Determine system dofs   
        # 
        dofs = bilinear_form['row_dofs']
        
        #
        # Check that dofs are consistent (should be if element type is)
        # 
        assert np.allclose(dofs, bilinear_form['col_dofs']), \
        'Test and trial dofs should be the same.'
        
        assert np.allclose(dofs, linear_form['row_dofs']), \
        'Test and trial dofs should be the same.'
        
        self.__dofs = dofs
        

        #
        # Form Dof-to-Equation Mapping
        # 
        n_dofs = len(dofs)
        dof2eqn = np.zeros(dofs[-1]+1, dtype=np.int)
        dof2eqn[dofs] = np.arange(n_dofs, dtype=np.int)
        
        # Store mapping 
        self.__dof2eqn = dof2eqn
        
        #
        # Form system matrix
        # 
        rows = bilinear_form['rows'] 
        cols = bilinear_form['cols']
        vals = bilinear_form['vals']
        
        #
        # Check sample index
        # 
        i_bilinear_sample = i_sample[0]
        if len(vals.shape)>1 and vals.shape[1]>1:
            #
            # Multiple bilinear forms, extract sample
            #  
            assert i_bilinear_sample < vals.shape[1],\
            'Sample index exceeds sample size.'
            
            vals = bilinear_form['vals'][i_bilinear_sample]
        #
        # Store as sparse matrix 
        # 
        A = sparse.coo_matrix((vals,(rows,cols)))
        self.__A = A.tocsr()
    
        #
        # Form right hand side
        #
        linear_form = problem['linear']
        vals = linear_form['vals']
        i_linear_sample = i_sample[1]
        if len(vals.shape)>1 and vals.shape[1]>1:
            #
            #  Multiple linear forms, extract sample
            # 
            assert i_linear_sample < vals.shape[1],\
            'Sample index %d exceeds sample size %d.'\
            %(i_linear_sample, vals.shape[1])
            
            vals = linear_form['vals'][i_linear_sample]

        self.__b = sparse.csr_matrix(vals).T
        
        #
        # Initialize solution vector
        #  
        self.__u = np.zeros(n_dofs)      
            
        #
        # List of Hanging nodes 
        # 
        subforest_flag = assembler.subforest_flag
        self.dofhandler().set_hanging_nodes(subforest_flag=subforest_flag)
        
        
    def dofhandler(self):
        """
        Return the system's dofhandler
        """   
        return self.assembler.dofhandlers[self.etype()]
    
    
    def dofs(self):
        """
        Return system dofs
        """
        return self.__dofs

    
    def dof2eqn(self, dofs):
        """
        Convert vector of dofs to equivalent equations
        """
        return self.__dof2eqn[dofs]
    
    
    def etype(self):
        """
        Return system element type
        """
        return self.__etype


    def A(self):
        """
        Return system matrix 
        """
        return self.__A 
    
    
    def b(self):
        """
        Return right hand side
        """
        return self.__b
        
    
    def C(self):
        """
        Return constraint matrix
        """ 
        return self.__C
    
    
    def d(self):
        """
        Return constraint affine term
        """
        return self.__d
     
        
    def sol(self, as_function=False):
        """
        Returns the solution of the linear system 
        """
        if not as_function:
            #
            # Return solution vector
            # 
            return self.__u
        else: 
            #
            # Return solution as nodal function
            # 
            u = Function(self.__u, 'nodal', mesh=self.assembler.mesh, \
                         dofhandler=self.dofhandler(), \
                         subforest_flag=self.assembler.subforest_flag)
            return u
       
    
    def add_dirichlet_constraint(self, bnd_marker, dirichlet_function=0, on_boundary=True):
        """
        Modify an assembled bilinear/linear pair to account for Dirichlet 
        boundary conditions. The system matrix is modified "in place", 
        i.e. 
    
            a11 a12 a13 a14   u1     b1
            a21 a22 a23 a24   u2  =  b2 
            a31 a32 a33 a34   u3     b3
            a41 a42 a43 a44   u4     b4
            
        Suppose Dirichlet conditions u2=g2 and u4=g4 are prescribed. 
        If compressed=False, the system is converted to
        
            a11  0  a13  0   u1     b1 - a12*g2 - a14*g4
             0   1   0   0   u2  =  g2   
            a31  0  a33  0   u3     b3 - a32*g2 - a34*g4
             0   0   0   1   u4     g4 
        
        If compressed=True, the system is converted to
        
            a11 a13  u1  = b1 - a12*g2 - a14*g4
            a31 a33  u3    b3 - a32*g2 - a34*g3
        
        The solution [u1,u3]^T of this system is then enlarged with the 
        dirichlet boundary values g2 and g4 by invoking 'resolve_dirichlet_nodes' 
        
    
        Inputs:
        
            bnd_marker: str/int flag to identify boundary
            
            i_problem: int, problem index (default = 0)
            
            dirichlet_function: Function, defining the Dirichlet boundary 
                conditions.
            
            
        Notes:
        
        To maintain the dimensions of the matrix, the trial and test function 
        spaces must be the same, i.e. it must be a Galerkin approximation. 
        
        Specifying the Dirichlet conditions this way is necessary if there
        are hanging nodes (uncompressed), since a Dirichlet node may be a
        supporting node for one of the hanging nodes.  
                
                
        Inputs:
        
            bnd_marker: flag, used to mark the Dirichlet boundary
                        
            dirichlet_fn: Function, specifying the function values on the  
                Dirichlet boundary. 
        
            
        Outputs:
        
            None 
            
            
        Modified Attributes:
        
            __A: modify Dirichlet rows and colums (shrink)
            
            __b: modify right hand side (shrink)
            
            dirichlet: add dictionary,  {mask: np.ndarray, vals: np.ndarray}
        """
        #
        # Get Dofs Associated with Dirichlet boundary
        #
        subforest_flag = self.assembler.subforest_flag
        dh = self.dofhandler()
        
        if dh.mesh.dim()==1:
            #
            # One dimensional mesh
            # 
            dirichlet_dofs = dh.get_region_dofs(entity_type='vertex', \
                                                entity_flag=bnd_marker,\
                                                interior=False, \
                                                on_boundary=on_boundary,\
                                                subforest_flag=subforest_flag)
        elif dh.mesh.dim()==2:
            #
            # Two dimensional mesh
            #
            dirichlet_dofs = dh.get_region_dofs(entity_type='half_edge', 
                                                entity_flag=bnd_marker, 
                                                interior=False, 
                                                on_boundary=on_boundary, \
                                                subforest_flag=subforest_flag) 
        
        
        #
        # Evaluate dirichlet function at vertices associated with dirichlet dofs
        # 
        dirichlet_vertices = dh.get_dof_vertices(dirichlet_dofs)
        if isinstance(dirichlet_function, numbers.Number):
            #
            # Dirichlet function is constant
            # 
            n_dirichlet = len(dirichlet_dofs)
            if dirichlet_function==0:
                #
                # Homogeneous boundary conditions
                # 
                dirichlet_vals = np.zeros(n_dirichlet)
            else:
                #
                # Non-homogeneous, constant boundary conditions
                # 
                dirichlet_vals = dirichlet_function*np.ones(n_dirichlet)
        else:
            #
            # Nonhomogeneous, nonconstant Dirichlet boundary conditions 
            #
            x_dir = convert_to_array(dirichlet_vertices)
            dirichlet_vals = dirichlet_function.eval(x_dir)
        
        constraints = dh.constraints
        for dof, val in zip(dirichlet_dofs, dirichlet_vals):
            constraints['constrained_dofs'].append(dof)
            constraints['supporting_dofs'].append([])
            constraints['coefficients'].append([])
            constraints['affine_terms'].append(val)
        
        """
        #
        # Mark Dirichlet Dofs
        #
        dofs = self.dofs()
        n_dofs = len(dofs)
        dirichlet_mask = np.zeros(n_dofs, dtype=np.bool)
        for dirichlet_dof in dirichlet_dofs:
            dirichlet_mask[dofs==dirichlet_dof] = True
        
        
        # =====================================================================
        # Modify matrix-vector pair
        # =====================================================================
        A = self.A()
        b = self.b()
        if not self.is_compressed():
            #
            # Not compressed: No need to keep track of indices
            # 
            # Convert to list of lists format
            A = A.tolil()
            
            for i_row in range(n_dofs):
                #
                # Iterate over rows
                # 
                print('Dirichlet Dof?', dofs[i_row])
                if dofs[i_row] in dirichlet_dofs:
                    #
                    # Dirichlet row
                    #  
                    
                    # Turn row into [0,...,0,1,0,...,0]
                    A.rows[i_row] = [i_row]
                    A.data[i_row] = [1] 
                    
                    # Assign Dirichlet value to b[i]
                    i_dirichlet = dirichlet_dofs.index(dofs[i_row])
                    b[i_row] = dirichlet_vals[i_dirichlet]
                    
                    print(dofs[i_row], 'dirichlet row')
                    print('assigning', dirichlet_vals[i_dirichlet], 'to entry', i_row)
                    print('b=', b)
                else:
                    #
                    # Check for Dirichlet columns 
                    # 
                    new_row = []
                    new_data = []
                    n_cols = len(A.rows[i_row])  # number of elements in row
                    for j_col, col in zip(range(n_cols), A.rows[i_row]):
                        #
                        # Iterate over columns
                        #
                        if dofs[col] in dirichlet_dofs:
                            #
                            # Dirichlet column: move it to the right
                            # 
                            j_dirichlet = dirichlet_dofs.index(dofs[col])
                            b[i_row] -= A.data[i_row][j_col]*dirichlet_vals[j_dirichlet]
                        else:
                            #
                            # Store unaffected columns in new list
                            # 
                            new_row.append(col)
                            new_data.append(A.data[i_row][j_col])
                    A.rows[i_row] = new_row
                    A.data[i_row] = new_data
        else:
            #
            # Compressed format
            #
            i_free = self.free_indices()
            
                    
            #
            # Convert A to sparse column format
            # 
            A = A.tocsc()
            
            n_rows, n_cols = A.shape
            assert n_rows==n_cols, \
            'Number of columns and rows should be equal.'
            
            assert n_rows == np.sum(i_free), \
            'Dimensions of matrix not compatible with cumulative mask.'+\
            '# rows: %d, # free indices: %d'%(n_rows, np.sum(i_free))
            
            assert n_rows == len(b), \
            'Matrix dimensions not compatible with right hand side.'
            
            #
            # Adjust the right hand side
            #
            reduced_dirichlet_mask = dirichlet_mask[i_free]
            g = np.zeros(n_rows)
            g[reduced_dirichlet_mask] = dirichlet_vals
            b -= A.dot(g)
            
            
            A = A[~reduced_dirichlet_mask,:][:,~reduced_dirichlet_mask]
            b = b[~reduced_dirichlet_mask]
            
            # Convert back to coo format
            A = A.tocoo()
        
        
        #
        # Store Dirichlet information
        # 
        self.dirichlet.append({'mask': dirichlet_mask, 'vals': dirichlet_vals})
        self.__A = A
        self.__b = b
        
        """
        """
        for row, i_row in zip(A.rows, range(n_rows)):
            if row_dofs[i_row] in dirichlet_test_dofs:
                #
                # Dirichlet row: Mark for deletion
                # 
                dirichlet_rows[i_row] = True
            
            for col, i_col in zip(row, range(n_rows)): 
                #
                # Iterate over columns
                # 
                if col_dofs[col] in dirichlet_trial_dofs:
                    #
                    # Column contains a Dirichlet dof
                    # 
                    dirichlet_cols[col] = True
                    
                    # Adjust right hand side
                    i_trial = dirichlet_trial_dofs.index(col_dofs[col])
                    b[i_row] -= A.rows[i_row][i_col]*dirichlet_vals[i_trial]
                    
                    # Zero out entry in system matrix
                    del row[i_col]
                    del A.data[i_row][i_col]
        
        # =====================================================================
        # Modify Matrix
        # =====================================================================
        if compressed: 
            #
            # Delete all rows corresponding to Dirichlet test functions
            # and all columns corresponding to Dirichlet trial functions
            # 
            A = A.tocsc()
            
        else:
            #
            #  Add rows of the identity to A 
            # 
            n_dirichlet_rows = sum(dirichlet_rows)
            n_dirichlet_cols = sum(dirichlet_cols)
            if test_etype != trial_etype:
                #
                # More rows than
                # 
                print('Not supported yet')
            elif n_dirichlet_rows < n_dirichlet_cols:
                print('Not supported')
            else:
                #
                # Simplest case: Replace Dirichlet Rows those of Identity Matrix
                # 
                A = A.csc()
                A[dirichlet_rows,:][:,dirichlet_cols] = 1
                b[dirichlet_rows] = dirichlet_vals
                pass
            pass
        """
   
    
      
    def set_constraint_matrix(self):
        """
        Define the constraint matrix C and affine term d so that 
        
            x = Cx + d,
            
        where the rows in C corresponding to unconstrained dofs are rows of the
        identity matrix.
        """
        dofs = self.dofs()
        n_dofs = len(dofs)
        
        #    
        # Define constraint matrix
        #
        constraints = self.dofhandler().constraints
        c_dofs = np.array(constraints['constrained_dofs'], dtype=np.int)
        c_rows = []
        c_cols = []
        c_vals = []  
        for dof, supp, coeffs, dummy in zip(*constraints.values()):
            #
            # Iterate over constrained dofs, supporting dofs, and coefficients
            # 
            for s_dof, ck in zip(supp, coeffs):
                #
                # Populate rows (constraints), columns (supports), and 
                # values (coeffs)
                # 
                c_rows.append(self.dof2eqn(dof))
                c_cols.append(self.dof2eqn(s_dof))
                c_vals.append(ck)
        C = sparse.coo_matrix((c_vals,(c_rows, c_cols)),(n_dofs,n_dofs))
        C = C.tocsr()
        
        #
        # Add diagonal terms for unconstrained dofs
        # 
        one = np.ones(n_dofs)
        one[self.dof2eqn(c_dofs)] = 0 
        I = sparse.dia_matrix((one, 0),shape=(n_dofs,n_dofs));        
        C += I
        
        # Store constraint matrix
        self.__C = C
                
        #
        # Define constraint vector
        # 
        d = np.zeros(n_dofs)
        d[c_dofs] = np.array(constraints['affine_terms'])
        
        # Store constraint vector
        self.__d = d
    
    
    def incorporate_constraints(self):
        """
        Incorporate constraints due to (i) boundary conditions, (ii) hanging 
        nodes, and/or other linear compatibility conditions. Constraints are 
        of the form 
        
            x = Cx + d
            
        where C is an (n,n) sparse matrix of constraints and d is an (n,) 
        vector. The constraints are incorporated in the following steps
        
        Step 1:  Replace constrained variable in each row with the 
            appropriate linear combinations of supporting variables and/or
            right hand sides
        
        Step 2: Zero out columns corresponding to constrained variables.
        
        Step 3: Distribute equation at constrained dof to equations at 
            supporting dofs. 
            
        Step 4: Replace constrained dof's equation with kth row with scaled
            trivial equation a*x_k = 0
         
        """
        dofs = self.dofs()
        n_dofs = len(dofs)
        
        A, b = self.A(), self.b()
        C, d = self.C(), self.d()
        
        constraints = self.dofhandler().constraints
        c_dofs = constraints['constrained_dofs']
        
        #
        # Eliminate constrained variables
        # 
        for c_dof in c_dofs:
            #
            # Equation number of constrained dof
            #
            k = self.dof2eqn(c_dof)
            
            #
            # Form outer product A[:,k]*C[k,:]
            #
            ck = C.getrow(k)
            ak = A.getcol(k)
            
            #
            # Modify A's columns
            #
            A += ak.dot(ck)
            
            #
            # Modify b's rows
            #
            b -= d[k]*ak
            
            #
            # Remove Column k
            #
            one = np.ones(n_dofs)
            one[k] = 0
            Imk = sparse.dia_matrix((one,0),shape=(n_dofs,n_dofs))
            A = A.dot(Imk)

            #
            # Distribute constrained equation among supporting rows
            # 
            
            #
            # Modify A's rows 
            # 
            ak = A.getrow(k)            
            A += ck.T.dot(ak)
            
            #
            # Modify b's rows
            # 
            bk = b[k].toarray()[0,0]
            b += bk*ck.T
                        
            #
            # Zero out row k 
            # 
            A = Imk.dot(A)
            
            #
            # Add diagonal row
            # 
            zero = np.zeros(n_dofs)
            zero[k] = 1
            Ik  = sparse.dia_matrix((zero,0), shape=(n_dofs,n_dofs))
            A += Ik
        
        #
        # Set diagonal entries of constrained nodes equal to mean(A[k,k]) 
        # 
        a_diag = A.diagonal()
        ave_vec = np.ones(n_dofs)
        n_cdofs = len(c_dofs)
        if n_dofs > n_cdofs:
            #
            # If there are unconstrained dofs, use average diagonal to scale 
            # 
            ave_vec[c_dofs] = (np.sum(a_diag)-n_cdofs)/(n_dofs-n_cdofs)
        I_ave = sparse.dia_matrix((ave_vec,0), shape=(n_dofs,n_dofs))
        A = A.dot(I_ave)
          
        #
        # Set b = 0 at constraints 
        # 
        zc = np.ones(n_dofs)
        zc[c_dofs] = 0
        Izc = sparse.dia_matrix((zc,0),shape=(n_dofs,n_dofs)) 
        b = Izc.dot(b)
        
        
        self.__A = A
        self.__b = b
        
   
    def solve(self):
        """
        Returns the solution (in vector form) of a problem
        
        Inputs:
        
            return_solution_function: bool, if true, return solution as nodal
                function expanded in terms of finite element basis. 
                
                
        Outputs: 
        
            u: double, (n_dofs,) vector representing the values of the
                solution at the node dofs.
                
                or 
                
                Function, representing the finite element solution
            
        """ 
        A = self.A()
        b = self.b()

        self.__u = sparse.linalg.spsolve(A,b)    
    
    
    def resolve_constraints(self, x=None):
        """
        Impose constraints on a vector x
        """  
        if x is None:
            u = self.__u
        else:
            u = x
            
        #
        # Get constraint system x = Cx + d
        # 
        C, d = self.C(), self.d()
        
        #
        # Modify dofs that don't depend on others
        # 
        n_dofs = self.dofhandler().n_dofs()
        ec_dofs = [i for i in range(n_dofs) if C.getrow(i).nnz==0]
        u[ec_dofs] = d[ec_dofs]
        
        #
        # Modify other dofs
        # 
        u = C.dot(u) + d
        
        #
        # Store or return result       
        # 
        if x is None:
            self.__u = u
        else:
            return u
        
    '''
    def extract_hanging_nodes(self):
        """
        Incorporate hanging nodes into linear system, by 
        
        1. Replacing equations in rows corresponding to hanging nodes with 
            interpolation formulae.
            
        2. Zeroing out hanging node columns, compensating by adding entries            
            in supporting node columns, which may require changing the sparsity
            structure of the matrix.  
        
        When compressing, the rows and columns corresponding to hanging nodes
        are removed. This is recorded in self.hanging_nodes_mask
        
        
        NOTE:
        
            - For simplicity of implementation, we assume that the system has 
                not already been compressed. This requires hanging nodes to 
                be extracted BEFORE extracting dirichlet nodes.
        
        """
        if not self.has_hanging_nodes():
            return 
        
        # Convert A to a lil matrix
        A = self.A().tolil() 
        b = self.b()
        
        dofs = self.dofs()
        n_rows = A.shape[0]  
        
        #
        # Check assumption, that the number of dofs equals the system size!
        # 
        assert n_rows == len(dofs), \
        'Number of dofs should equal system size.'
        
        #
        # Vector for converting dofs to matrix indices
        #
        dof2idx = np.zeros(np.max(dofs)+1, dtype=np.int)
        dof2idx[dofs] = np.arange(n_rows)
        
        hanging_nodes = self.hanging_nodes
        for i in range(n_rows):
            #
            # Iterate over all rows
            #
            if dofs[i] in hanging_nodes.keys():
                #
                # Row corresponds to hanging node
                #
                if not self.is_compressed():
                    #
                    # Replace equation in hanging node row with interpolation
                    # formula. 
                    # 
                    new_indices = [dof2idx[s_dof] for s_dof \
                                   in hanging_nodes[dofs[i]][0]] 
                    new_indices.append(i)
                    A.rows[i] = new_indices           
         
                    new_values = [-cs_j for cs_j in hanging_nodes[dofs[i]][1]] 
                    new_values.append(1)
                    A.data[i] = new_values
                    
                    b[i] = 0
                
            else:
                row = A.rows[i]
                data = A.data[i]
                
                for hn in hanging_nodes.keys():
                    #
                    # For each row, determine what hanging nodes are supported
                    #
                    if dof2idx[hn] in row:    
                        #
                        # If hanging node appears in row, modify
                        #
                        j_hn = row.index(dof2idx[hn])
                        for js,vs in zip(*hanging_nodes[hn]):
                            #
                            # Loop over supporting indices and coefficients
                            # 
                            if dof2idx[js] in row:
                                #
                                # Index exists: modify entry
                                #
                                j_js = row.index(dof2idx[js])
                                data[j_js] += vs*data[j_hn]
                            else:
                                #
                                # Insert new entry
                                # 
                                jj = bisect_left(row, dof2idx[js])
                                vi = vs*data[j_hn]
                                row.insert(jj,js)
                                data.insert(jj,vi)
                                j_hn = row.index(hn)  # find hn again
                        #
                        # Zero out column that contains the hanging node
                        #
                        print(row)
                        row.pop(j_hn)
                        data.pop(j_hn)
                        print(row)
                if self.is_compressed():
                    #
                    # Renumber entries to right of hanging nodes.
                    # 
                    for hn in hanging_nodes.keys():
                        j_hn = bisect_left(row, dof2idx[hn])
                        for j in range(j_hn,len(row)):
                            row[j] -= 1
        if self.is_compressed():
            #
            # Delete rows corresponding to hanging nodes
            #
            hn_list = [dof2idx[hn] for hn in hanging_nodes.keys()]
            n_hn = len(hn_list)    
            A.rows = np.delete(A.rows,hn_list,0)
            A.data = np.delete(A.data,hn_list,0)
            b = np.delete(b,hn_list,0)
            A._shape = (A._shape[0]-n_hn, A._shape[1]-n_hn)
        
        #
        # Store modified system 
        # 
        self.__A = A.tocoo()
        self.__b = b
            
     
    def resolve_hanging_nodes(self):
        """
        Enlarge the solution vector u to include hannging nodes  
        
        Inputs:
        
           u: double, (n,) numpy vector of nodal values, without hanging nodes.
            
           hanging_nodes: dict, {i_hn:[is_1,...,is_k], [cs_1,...,cs_k]}
                i_hn: hanging node index
                is_j: index of jth supporting node
                cs_j: coefficient of jth supporting basis function, i.e.
                
                phi_{i_hn} = cs_1*phi_{is_1} + ... + cs_k*phi_{is_k} 
    
                
        Outputs:
            
            uu: double, (n+k,) numpy vector of nodal values which includes 
                hanging nodes.
        """
        dofs = self.dofs()
        n_dofs = len(dofs)
        
        #
        # Vector for converting dofs to matrix indices
        #
        dof2idx = np.zeros(np.max(dofs)+1, dtype=np.int)
        dof2idx[dofs] = np.arange(n_dofs)
        
        for hn, supp in self.hanging_nodes.items():
            #
            # Iterate over hanging nodes and support dofs
            # 
            supp_dofs, supp_vals = supp
            
            self.__u[dof2idx[hn]] = \
                np.dot(self.__u[dof2idx[supp_dofs]], supp_vals)
                 
    
    def free_indices(self):
        """
        Returns boolean vector with 1s at all entries that are neither hanging
        nodes, nor previously encountered Dirichlet nodes  
        """ 
        n_dofs = len(self.dofs())
        if self.is_compressed():
            #
            # Collect all boolean masks applied so far
            # 
            unchanged_entries = np.ones(n_dofs, dtype=np.bool)
            if self.has_hanging_nodes():
                #
                # Mask from hanging nodes 
                # 
                unchanged_entries *= ~self.hanging_nodes_mask
            if len(self.dirichlet)!=0:
                #
                # Masks from previous dirichlet conditions
                # 
                for dirichlet in self.dirichlet:
                    unchanged_entries *= ~dirichlet['mask']
        else:
            #
            # No nodes have been removed, return a vector of ones
            # 
            unchanged_entries = np.ones(n_dofs, dtype=np.bool)
            
        return unchanged_entries
    '''    