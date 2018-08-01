import numpy as np
from scipy import sparse, linalg
import numbers
from mesh import Vertex, HalfEdge, Mesh2D
from mesh import Cell, QuadCell, Interval
from mesh import RHalfEdge, RInterval, RQuadCell
from mesh import convert_to_array
from bisect import bisect_left
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
                self.dofhandler.get_global_dofs(subforest_flag=subforest_flag)
            
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
                
                dofs = dofhandler.get_global_dofs(subforest_flag=subforest_flag)
                x = convert_to_array(dofhandler.get_dof_vertices(dofs=dofs), dim)
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
        self.__type = fn_type
 
 
    def assign(self, v, pos=None):
        """
        Assign function values to the function in the specified position
        
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
                              
 
         
    def global_dofs(self):
        """
        Returns the global dofs associated with the function values. 
        (Only appropriate for nodal type functions).
        """    
        if self.__type == 'nodal':
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
    
    
    def eval(self, x, cell=None, phi=None, derivative=(0,), samples='all'):
        """
        Evaluate function at an array of points x
        
        Inputs:
        
            x: double, function input in the form of an (n_points, dim) array,
                or a list of vertices or a list of tuples.
            
            cell: Cell, on which f is evaluated. If included, all points in x
                should be contained in it. 
                TODO: cell may be an ancestor of the cell on whose dofs the function is defined!!
            
            phi: shape functions (if function is nodal). 
                
            derivative: int, tuple, (order,i,j) where order specifies the order
                of the derivative, and i,j specify the variable wrt which we 
                differentiate, e.g. (2,0,0) computes d^2p/dx^2 = p_xx,
                (2,1,0) computes d^2p/dxdy = p_yx
            
            samples: int, (r, ) integer array specifying the samples to evaluate
                or use 'all' to denote all samples
        
        Output:
        
            f(x): If function is deterministic (i.e. n_samples is None), then 
                f(x) is an (n_points, ) numpy array. Otherwise, f(x) is an 
                (n_points, n_samples) numpy array of outputs
            
        """
        flag = self.__flag
        dim = self.__dim
        
        # =====================================================================
        # Parse Input
        # =====================================================================
        
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
                cell_list = [cell]
            #
            # Evaluate function within each cell
            #
            for cell in cell_list:
                #
                # Evaluate function at local dofs 
                # 
                idx_cell = [self.__global_dofs.index(i) for i in \
                            self.dofhandler.get_global_dofs(cell)]  
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
        dofs = dofhandler.get_global_dofs(subforest_flag=subforest_flag)
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
        dofs = dofhandler.get_global_dofs(subforest_flag=flag)
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
        self.__hanging_nodes = {}
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
        cell_dofs = self.get_global_dofs(cell)
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
            dofs = self.get_global_dofs(cell, pivot)
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
        cell_dofs = self.get_global_dofs(cell)  
        
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
                    vertex_dofs = self.get_global_dofs(cell, vertex)
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
                        he_dofs = self.get_global_dofs(cell, half_edge)
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
                    cell_dofs = self.get_global_dofs(cell, cell)
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
        cell_dofs = self.get_global_dofs(cell)  
        
                
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
            
        TODO: Include support for arbitrary Vertex
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
        
            cell: Cell, whose dofs we seek. 
            
            entity: get_verticesVertex/HalfEdge within cell, whose dofs we seek
            
            subforest_flag: flag, specifying submesh.
            
            mode: str, ['breadth-first'] or 'depth-first', mode of mesh traversal
            
            nested: bool, if true, get dofs for all cells, otherwise just for leaves 
            
            
        Outputs:
        
             global_dofs: list of global dofs 
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
            dofs = self.get_global_dofs(cell)
            
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
            
            *cells [None]: dictionary of (dof, Cell), specifying the cell
            in which the dofs must occur. 
        """
        is_singleton = False
        if type(dofs) is np.int:
            dofs = [dofs]
            is_singleton = True
        
        if dofs is None:
            dofs = self.get_global_dofs(subforest_flag=subforest_flag)
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
          
    
    def set_hanging_nodes(self, subforest_flag=None):
        """
        Set up the constraint matrix satisfied by the mesh's hanging nodes.
        
        Note: 
        
            - Hanging nodes can only be found once the mesh has been balanced.
        
            - Hanging nodes are never periodic
        """
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
                                    hanging_nodes[hn_dof[v.get_pos(1, i_child)]] = \
                                        (supporting_dofs, constraint_coefficients)
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
                        
    
    def eval(self, x, cell=None, phi=None):
        """
        Evaluate the kernel at the points stored in x 
        
        Inputs:
        
            x: (n_points, dim) array of points at which to evaluate the kernel
            
            cell: Cell/Interval within which the points x reside
            
            phi: dictionary, encoding the shape functions necessary to evaluate
                the shape function if the constituent functions are nodal. 
                
                form of phi: phi[regions][derivatives]
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
            fv = f.eval(x, cell=cell, derivative=dfdx, samples=self.samples)
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


    def derivatives(self):
        """
        Returns a set of all derivatives necessary for evaluating the kernel 
        """
        return set(self.dfdx)
        
    
    
class Form(object):
    """
    Constant, Linear, or Bilinear forms (integrals)
    
    Attributes:
        
        kernel: Kernel 
        
        trial:
        
        test:
        
        dx:
        
        flag:
        
                
    """
    def __init__(self, kernel, trial, test, dmu='dx', flag=None, samples=None):
        """
        Constructor
        
        Inputs:
        
            kernel: Kernel, specifying the form's kernel  
            
            trial: str, derivative of the trial function 
                'u', 'ux', 'uy', 'uxy', 'uyx', or 'uyy' 
            
            test: str, derivative of the test function 
                'v', 'vx', 'vy', 'vxy', 'vyx', or 'vyy' 
            
            dmu: str, area of integration
                'dx' - integrate over a cell
                'ds' - integrate over an edge
            
            flag: str/int/tuple cell/edge marker
            
        """
        self.trial = trial
        self.test = test
        self.dmu = dmu
        self.flag = flag
        self.kernel = kernel
        self.samples = samples


    def derivatives(self):
        """
        Determine all derivatives that must be evaluated (f, trial, and test)
        """
        derivatives = set()
        derivatives.update(self.kernel.derivatives())
        derivatives.add(parse_derivative_info(self.trial))
        derivatives.add(parse_derivative_info(self.test))
        
        return derivatives
    
    
    def eval(self, entity, xg, wg, phi=None):
        """
        Evaluates the local kernel, test, (and trial) functions of a (bi)linear
        form on a given entity.
        
        Inputs:
            
            entity: Cell/Interval/HalfEdge over which to integrate
            
            phi: shape functions evaluated at the Gauss quadrature points
            
            xg: array of Gaussian quadature points on entity
            
            wg: corresponding quadrature weights
        
        Outputs:
        
            Constant-, linear-, or bilinear forms
                            
        """
        #
        # kernel
        # 
        f = self.f
        assert isinstance(f, Function), \
            'Kernel should be a Function'
        #
        # Compute kernel, weight by quadrature weights    
        #
        kernel = f.eval(xg, samples=self.samples)
        wKer = (wg*kernel.T).T
          
        n_gauss = len(wg)
        n_samples = kernel.n_samples
        if self.test is not None:
            assert phi is not None, \
                'Evaluating (bi)linear form. Require shape functions'
            #
            # Need test function               
            # 
            drv = self.parse_derivative_info(self.test)
            test = phi[entity][drv] 
            n_dofs = test.shape[1]
            if self.trial is not None:
                #
                # Need trial function
                # 
                drv = self.parse_derivative_info(self.trial)
                trial = phi[entity][drv]
                
                #
                #  Bilinear form               
                #
                if n_samples is None:
                    #
                    # Deterministic Kernel
                    # 
                    f_loc = np.dot(test.T, np.dot(np.diag(wg*kernel),trial))
                else:
                    #
                    # Sampled kernel
                    # 
                    
                    f_loc = np.dot(test.T, np.reshape(np.kron(trial, wKer),(n_gauss,-1), order='F'))
                    f_loc.reshape((n_dofs**2,n_samples), order='F')
            else:
                #
                # Linear Form
                #
                f_loc = np.dot(test.T, wKer)                 
        else:
            #
            # Simple integral
            # 
            f_loc = np.sum(wKer, axis=0)           
        return f_loc

        
    def bilinear_loc(self,weight,kernel,trial,test):
        """
        Compute the local bilinear form over an element
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
        
    
    
class System(object):
    """
    (Non)linear system to be defined and solved 
    """
    def __init__(self, mesh, element, n_gauss=(4,16)):
        """
        Assemble the finite element system
        
        Inputs:
        
            mesh: Mesh, finite element mesh
            
            element: FiniteElement, shapefunctions
            
            n_gauss: int tuple, number of quadrature nodes in 1d and 2d respectively
                        
        """
        self.mesh = mesh
        self.element = element
        self.n_gauss_2d = n_gauss[1]
        self.n_gauss_1d = n_gauss[0]
        
        #
        # Initialize Gauss Quadrature Rule
        # 
        dim = self.element.dim()
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
            self.cell_rule = GaussRule(self.n_gauss_2d,shape=element.cell_type())
            
        #
        # Initialize DofHandler
        # 
        self.dofhandler = DofHandler(mesh,element)
        self.dofhandler.distribute_dofs()
        
        #
        # For storing values of shape functions at Gauss Points
        # 
        derivative_list = [(0,),(1,0),(1,1),(2,0,0),(2,0,1),(2,1,0),(2,1,1)]
        dof_list = [i for i in range(element.n_dofs())]
        self.__phi = dict.fromkeys(dof_list, dict.fromkeys(derivative_list, []))
 
 
    def assemble(self, problems, subforest_flag=None):
        """
        Assembles (bi)linear forms over computational mesh, incorporating
        boundary conditions. 
        
        Input:
        
            problems: A list of dictionaries that define a finite element 
                problem. Each problem contains the following fields:
                
                linear: list of tuples (f,'v*', dx, flag) defining the problem's
                    linear forms, where
                    
                    f: Function
                    
                    v*: is a string of the form 'v', 'vx', or 'vy' that
                        represents the test function
                
                    dx: string, that represents the integration region
                        'da' = integrate over cells/intervals
                        'ds' = integrate over half-edges
                        
                    flag: specifying the cells/half-edges over which to integrate
                    
                
                bilinear: list of 3 tuples (f,u*, v*, dx, flag) defining the 
                    problem's bilinear forms, where
                    
                    f: Function
                    
                    u*: is a string of the form 'u', 'ux', or 'uy' that 
                        represents the trial function
                        
                    v*: is a string of the form 'v', 'vx', or 'vy' that
                        represents the test function
                
                    dx: string, that represents the integration region
                        'da' = integrate over cells/intervals
                        'ds' = integrate over half-edges
            
                    flag: specifying the cells/half-edges over which to integrate 
                    
                
                bc: dictionary encoding boundary conditions, whose keys are
                
                    dirichlet: list of dictionaries encoding the parameters of
                        the problem's Dirichlet boundary conditions,
                        
                            u(xi) = g(xi) for xi in {Dirichlet nodes}
                        
                        marker: list of str or a boolean functions specifying 
                            the boundary segments on which Dirichlet conditions 
                            are to be applied.
                            
                        g: Function, values of the solution at the Dirichlet nodes
                    
                    neumann: list of dictionaries specifying the problem's 
                        Neumann boundary conditions
                    
                            -n*(A nabla(u)) = g(x) on Neumann edge
                            
                        marker: list of str or boolean functions specifying 
                            Neumann edges (or nodes in 1D).
                        
                        g: list of Functions, values of the fluxes on the 
                            Neumann edges
                    
                    robin: dictionary specifying the problem's Robin boundary
                        conditions. 
                        
                            -n*(A nabla(u)) = gma*( u - g )
                        
                        marker: list of str or a boolean functions specifying 
                            the boundary segments on which Robin conditions are
                            to be applied.
                        
                        gma: constant, list of proportionality constants   
                        
                        g: Function, list of Robin data functions
                                    
            subforest_flag: str/int, flag specifying the submesh on which to 
                assemble the problem.
                               
                               
        Output:
        
            assembled_forms: list of dictionaries (one for each problem), each of 
                which contains:
                
                'bf': dictionary summarizing assembled bilinear forms with fields
                    
                    'i': list of row entries 
                    
                    'j': list of column entries
                    
                    'val': list of matrix values 
                    TODO: Add support for samples for assembled matrices
                    
                    'dir_dofs': set, consisting of all dofs corresponding to 
                        Dirichlet vertices
            
                'lf': vector (or matrix), of assembled linear forms
                    
                
            A: double coo_matrix, system matrix determined by bilinear forms and 
                boundary conditions.
                
            b: double, right hand side vector determined by linear forms and 
                boundary conditions.
        
        
        TODO: Include option to assemble multiple problems
        TODO: Include option to assemble multiple realizations (sampled data)  
        """
        n_nodes = self.dofhandler.n_dofs(subforest_flag=subforest_flag)
        n_dofs = self.element.n_dofs() 
        dim = self.mesh.dim()
            
        #
        # Parse "problems" input
        # 
        if isinstance(problems, Form):
            #
            # Single problem
            # 
            problems = [problems]
            single_problem = True
        else: 
            assert type(problems) is list, \
            'Input "problems" should be a list or a dictionary.'
        n_problems = len(problems)    
        
        #
        # Initialize the forms to assemble
        #
        assembled_forms = []
        for dummy in range(n_problems):
            assembled_forms.append({'bf': {'rows': [], 'cols': [], 'vals': [], 
                                           'dir_dofs': set()}, 
                                    'lf': np.zeros(n_nodes,),
                                    'cf': 0})
        
        #
        # Assemble forms over mesh cells
        #            
        for cell in self.mesh.cells.get_leaves(subforest_flag=subforest_flag):
            # Get cell dofs
            cell_dofs = self.dofhandler.get_global_dofs(cell)
            
            #
            # Determine what derivatives to compute
            # 
            derivatives = self.problem_derivatives(problems, cell)
            regions = [region for region in derivatives.keys()]
            
            #
            # Compute shape functions and quadrature nodes on cell
            #  
            phi, xg, wg = self.shape_eval(regions, derivatives)
                    
            for i_problem in range(n_problems):
                problem = problems[i_problem]
                for form in problem:
                    if form.trial is not None:
                        #
                        # Bilinear form
                        # 
                        pass
                    elif form.test is not None:
                        #
                        # Linear form
                        #
                        pass
                    else:
                        #
                        # Integral
                        # 
                        pass
                
                #
                # Evaluate local linear and bilinear forms over cell
                # 
                bf_loc = np.zeros((n_dofs, n_dofs))
                lf_loc = np.zeros(n_dofs,)
                
                
                for bf in problem['bilinear']:
                    bf_loc += self.form_eval(bf, cell, phi, xg, wg)
                
                for lf in problem['linear']:
                    lf_loc += self.form_eval(lf, cell, phi, xg, wg)
        
        
                #
                # Local to global mapping
                # 
                for i in range(n_dofs):
                    #
                    # Update assembled linear form (vector)
                    #
                    if len(problem['lf'])>0:
                        assembled_forms[problem]['lf'][cell_dofs[i]] += lf_loc[i]
                    #
                    # Update assembled bilinear form (matrix)
                    # 
                    if len(problem['bf'])>0:
                        for j in range(n_dofs):
                            assembled_forms[i_problem]['bf']['i'].append(cell_dofs[i]) 
                            assembled_forms[i_problem]['bf']['j'].append(cell_dofs[j]) 
                            assembled_forms[i_problem]['bf']['val'].append(bf_loc[i,j]) 
                    
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
    
    def problem_derivatives(self, problems, cell):
        """
        Determine what shape functions must be computed and over what region
        
        Inputs:
        
            problems: list of problems (described in 'assemble' method)
            
            cell: cell over which to assemble
        """
        info = {}
        for problem in problems:
            for form in problem:
                if form.dmu == 'dx':
                    # 
                    # Integral over cell
                    # 
                    # Initialize cell key if necessary
                    if not info.has_key(cell):
                        info[cell] = set()
                    
                    # Update cell derivatives
                    info[cell].update(form.derivatives())
                elif form.dmu == 'ds':
                    #
                    # Integral over half-edge
                    # 
                    # Initialize 'edge' key if necessary
                    for half_edge in cell.get_half_edges():
                        if form.flag is None or half_edge.is_marked(form.flag):
                            # Initialize ith key if necessary
                            if not info.has_key(half_edge):
                                info[half_edge] = set()
                            
                            # Update edge derivatives 
                            form[half_edge].update(form.derivatives())
        return info
                
                                
                                    
    def shape_eval(self, regions, derivatives):
        """
        Evaluate the element shape functions (and their derivatives) at the 
        Gauss quadrature points in each region. 
        
        
        Inputs:
        
            regions: list of regions (QuadCell, Interval, or HalfEdge) over 
                which gaussian quadrature rules and shape functions are sought. 
            
            derivatives: dictionary, containing a list of shape function 
                derivatives sought over each region. 
             
        
        Output:
        
            phi: dictionary, of the form phi[region][derivative]
            
            xg: dictionary, of the form xg[region]
            
            wg: dictionary, of the form wg[region]
        """
        xg, wg, phi = {}, {}, {}
        for entity in regions:
            #
            # Map quadrature rule to entity (cell/halfedge)
            # 
            if isinstance(entity, Interval):
                #
                # Interval
                #
                # Get reference nodes and weights 
                x_ref = self.rule_1d.nodes()
                w_ref = self.rule_1d.weights()
                
                # Map reference quadrature nodes to cell 
                xg[entity], jac = entity.reference_map(x_ref, jacobian=True)
                
                # Modify the quadrature weights
                wg[entity] = w_ref*np.array(jac)
                
                # Specify cell
                cell = entity
                
            elif isinstance(entity, HalfEdge):
                #
                # Edge
                # 
                # Get reference quadrature nodes and weights 
                x_ref = self.rule_1d.nodes()
                w_ref = self.rule_1d.weights()
                
                # Map reference nodes to halfedge
                xg[entity], jac = entity.reference_map(x_ref, jacobian=True)
                
                # Modify the quadrature weights
                wg[entity] = w_ref*np.array(np.linalg.norm(jac[0]))
                
                # Define the enclosing cell
                cell = entity.cell()    
                
            elif isinstance(entity, QuadCell):
                #
                # Quadrilateral
                # 
                x_ref = self.rule_2d.nodes()
                w_ref = self.rule_2d.weights()
                
                # Map 
                xg[entity], jac = entity.reference_map(x_ref, jacobian=True)
                
                # Modify quadrature weights
                wg[entity] = w_ref*np.array([np.linalg.det(j) for j in jac])
                
                # Specify cell
                cell = entity
            else:
                raise Exception('Only Intervals, HalfEdges, & QuadCells supported')  
            #
            # Evaluate shape functions at quadrature points
            #
            p = self.element.shape(xg[entity], derivatives=derivatives[entity],
                                   cell=cell)
                
            # Convert list to dictionary
            count = 0
            for drv in range(derivatives[entity]):
                phi[entity][drv] = p[count]
                count += 1
                
        return phi, xg, wg        
            
    
    
    def extract_hanging_nodes(self,A,b, compress=False):
        """
        Incorporate hanging nodes into linear system.
    
        Inputs:
    
        A: double, (n,n) sparse matrix in coo format
        
        b: double, (n,1) vector of right hand sides
        
        hanging_nodes: dict, {i_hn:[is_1,...,is_k], [cs_1,...,cs_k]}
            i_hn: hanging node index
            is_j: index of jth supporting node
            cs_j: coefficient of jth supporting basis function, i.e.
            
            phi_{i_hn} = cs_1*phi_{is_1} + ... + cs_k*phi_{is_k}     
            
        compress: bool [False], flag for how the nodes should be accounted for
            True - remove the hanging nodes from the system (the solution 
                can then be reconstructed using "resolve_hanging_nodes").
            False - keep the size of the system, incorporating hanging nodes
                implicitly.
        """
        # Convert A to a lil matrix
        A = A.tolil() 
        
        hanging_nodes = self.dofhandler.get_hanging_nodes()
        n_rows = A.shape[0]
        for i in range(n_rows):
            #
            # Iterate over all rows
            #
            if i in hanging_nodes.keys():
                #
                # Row corresponds to hanging node
                #
                if not compress:
                    new_indices = [is_j for is_j in hanging_nodes[i][0]] 
                    new_indices.append(i)
                    A.rows[i] = new_indices           
         
                    new_values = [-cs_j for cs_j in hanging_nodes[i][1]] 
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
                    if hn in row:    
                        #
                        # If hanging node appears in row, modify
                        #
                        j_hn = row.index(hn)
                        for js,vs in zip(*hanging_nodes[hn]):
                            #
                            # Loop over supporting indices and coefficients
                            # 
                            if js in row:
                                #
                                # Index exists: modify entry
                                #
                                j_js = row.index(js)
                                data[j_js] += vs*data[j_hn]
                            else:
                                #
                                # Insert new entry
                                # 
                                jj = bisect_left(row,js)
                                vi = vs*data[j_hn]
                                row.insert(jj,js)
                                data.insert(jj,vi)
                                j_hn = row.index(hn)  # find hn again
                        #
                        # Zero out column that contains the hanging node
                        #
                        row.pop(j_hn)
                        data.pop(j_hn)
                        if compress:
                            #
                            # Renumber entries to hanging node's right.
                            # 
                            for j in range(j_hn,len(row)):
                                row[j] -= 1
        if compress:
            #
            # Delete rows corresponding to hanging nodes
            #
            hn_list = [hn for hn in hanging_nodes.keys()]
            n_hn = len(hn_list)    
            A.rows = np.delete(A.rows,hn_list,0)
            A.data = np.delete(A.data,hn_list,0)
            b = np.delete(b,hn_list,0)
            A._shape = (A._shape[0]-n_hn, A._shape[1]-n_hn)
        
        return A.tocoo(),b
            
     
    def resolve_hanging_nodes(self,u):
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
        hanging_nodes = self.__hanging_nodes
        hang = [hn for hn in hanging_nodes.keys()]
        n = len(u)
        not_hang = [i for i in range(n) if i not in hang]
        k = len(hang)
        uu = np.zeros((n+k,))
        uu.flat[not_hang] = u
        for i in range(k):
            i_s, c_s = hanging_nodes[hang[i]]
            uu[hang[i]] = np.dot(uu[i_s],np.array(c_s))
        return uu   
    
     
    def x_loc(self,cell):
        """
        Return the vertices corresponding to the local cell dofs 
        """   
        x_ref = self.element.reference_nodes()
        return cell.reference_map(x_ref)
         
    
    
  
        
    def f_eval_loc(self, f, node, edge_loc=None, derivatives=(0,), x=None):
        """
        TODO: Delete! 
        
        Evaluates a function (or its partial derivatives) at a set of 
        local nodes (quadrature nodes if none are specified).
        
        Inputs:
        
            f: function to be evaluated, either defined explicitly, or in
                terms of its LOCAL node values
                
            cell: QuadCell on which f is evaluated
            
            edge_loc: str, specifying edge location (W,E,S,N)
            
            derivatives: tuple specifying the order of the derivative and 
                the variable 
                [(0,)]: function evaluation, 
                (1,i) : 1st derivative wrt ith variable (i=0,1), or 
                (2,i,j) : 2nd derivative wrt ith and jth variables (i,j=0,1)
                
            x: Points (on physical entity) at which we evaluate f. By default,
                x are the Gaussian quadrature points.
        
        Output:  
        
            fv: vector of function values, at x points
        """
        cell = node.cell()
        #
        # Parse points x
        # 
        if x is None:
            #
            # Default: use quadrature points
            # 
            if edge_loc is None:
                x_ref = self.cell_rule().nodes()
                x = self.cell_rule().map(cell, x=x_ref)
            else:
                x_ref = self.edge_rule().nodes(direction=edge_loc)
                edge = cell.get_edges(edge_loc)
                x = self.edge_rule().map(edge)
        else:
            x_ref = self.cell_rule().inverse_map(cell,x)
              
        x =  np.array(x)
        #
        # Evaluate function
        #
        n_dofs = self.element.n_dofs() 
        if callable(f):
            #
            # f is a function
            # 
            if len(x.shape) == 1:
                # one dimensional input
                return f(x)
            elif len(x.shape) == 2:
                # two dimensional input
                return f(x[:,0],x[:,1])
            else: 
                raise Exception('Only 1D and 2D supported.')
        elif isinstance(f, Function):
            #
            # f is a Function object
            # 
            return f.eval(x, node=node)
        elif isinstance(f,numbers.Real):
            #
            # f is a constant
            #
            return f*np.ones(x.shape[0])
        elif len(f) == n_dofs:
            #
            # f is a nodal vector
            #            
            # Evaluate shape functions on reference entity
            phi = self.shape_eval(derivatives=derivatives,cell=cell,\
                                  edge_loc=edge_loc,x_ref=x_ref) 
            return np.dot(phi,f)                    
        else:
            fn_type = str('Function type for {0} not recognized.'.format(f))
            raise Exception(fn_type)
                
          
    def form_eval(self, form, entity, phi, xg, wg):
        """
        Evaluates the local kernel, test, (and trial) functions of a (bi)linear
        form on a given entity.
        
        Inputs:
        
            form: Form, (bi)linear form as tuple (f,'trial_type','test_type'), where
                
                f: Function
                
                trial_type: str, 'u','ux',or 'uy'
                
                test_type: str, 'v', 'vx', 'vy'    
                
            entity: Cell/Interval/HalfEdge  
            
            phi: shape functions evaluated at the Gauss quadrature points
            
            xg: array of Gaussian quadature points on entity
            
            wg: corresponding quadrature weights
        
        Outputs:
        
            (Bi)linear form
                            
        """
        #
        # kernel
        # 
        f = form.f
        assert isinstance(f, Function), \
            'Kernel should be a Function'
        kernel = f.eval(xg)
        
        if len(form) > 1:
            #
            # test function               
            # 
            drv = self.parse_derivative_info(form[1])
            test = phi[entity][drv] 
            if len(form) > 2:
                #
                # trial function
                # 
                drv = self.parse_derivative_info(form[2])
                trial = test.copy()
                test = phi[entity][drv]
                                
                if len(form) > 3:
                    raise Exception('Only Linear and Bilinear forms supported.')
                else:
                    return self.bilinear_loc(wg, kernel, trial, test) 
            else:
                return self.linear_loc(wg, kernel, test)
        else:
            return np.sum(kernel*wg)           
    

    
    '''
    def make_generic(self,entity):
        """
        Turn a specific entity (QuadCell or Edge) into a generic one
        e.g. Quadcell --> 'cell'
             (Edge, direction) --> ('edge',direction)
             
        TODO: Is this superfluous? 
        """ 
        if isinstance(entity, QuadCell):
            return 'cell'
        elif len(entity) == 2 and isinstance(entity[0], Edge):
            return ('edge', entity[1])
        else:
            raise Exception('Entity not supported.')
    '''    
        
        
        
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
            
        TODO: Move to Function Class
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