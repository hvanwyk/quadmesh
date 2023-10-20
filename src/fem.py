import numpy as np
from mesh import Vertex, HalfEdge
from mesh import Cell, QuadCell, Interval
from mesh import RInterval, RQuadCell
from mesh import convert_to_array
from mesh import Mesh
from diagnostics import Verbose


def parse_derivative_info(dstring):
        """
        Input:

            string: string of the form *,*x,*y,*xx, *xy, *yx, *yy, where *
                stands for any letter.

        Output:

            tuple, encoding derivative information
        """
        # Return tuples
        if type(dstring) is tuple:
            return dstring

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
                raise Exception('Use *xx,*xy,*yx, or *yy. * is any letter.')
        else:
            raise Exception('Higher order derivatives not supported.')



"""
Finite Element Classes
"""

class Element(object):
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


class QuadFE(Element):
    """
    Galerkin finite elements on quadrilateral cells
    """
    def __init__(self, dim, element_type):
        if dim==1:
            cell_type = 'interval'
        elif dim==2:
            cell_type = 'quadrilateral'

        Element.__init__(self, dim, element_type, cell_type)

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
            pxx = [lambda x: np.zeros(x.shape), lambda x: np.zeros(x.shape)]
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


    def shape(self, x_ref=None, cell=None, derivatives=(0,),
              jac_p2r=None, hess_p2r=None, local_dofs='all'):
        """
        Evaluate shape functions (and derivatives) at a given (reference) points

        Inputs:

            x_ref: double, points on reference domain at which shape functions
                are evaluated.

            cell [None]: optionally specify QuadCell or Interval to which
                the shape function is mapped. When computing derivatives,
                shape functions are modified to account for the coordinate
                mapping, using inverse Jacobians or Hessians. If region is None
                then the shape function is evaluated on reference element.

            derivatives: list of tuples, (order,i,j) where

                1. order specifies the order of the derivative,

                2. i,j specify the variable w.r.t which we differentiate
                    e.g. (2,0,0) computes d^2p/dx^2 = p_xx,
                         (2,1,0) computes d^2p/dxdy = p_yx

            jac_p2r: double, Jacobian of the inverse coordinate mapping

            hess_p2r: double, Hessian of the inverse coordinate mapping

            local_dofs: int, list of local dof indices whose entries in
                range(self.n_dofs).


        Output:

            phi: double, list of (n_points, len(local_dofs)) arrays of
                 (derivatives of ) shape functions, evaluated at the given
                 points.


        Note: To compute the reference points for a point x in the region,
            use the 'reference_map' function.

        """
        x_ref = convert_to_array(x_ref, self.dim())
        n_points = x_ref.shape[0]

        #
        # Determine whether to return singleton
        #
        if not type(derivatives) is list:
            derivatives = [derivatives]
            is_singleton = True
        else:
            is_singleton = False

        #
        # Points should lie in the reference domain
        #
        assert all(x_ref.ravel() >= 0), 'All entries should be nonnegative.'
        assert all(x_ref.ravel() <= 1), 'All entries should be at most 1.'

        if cell is not None:
            #
            # Determine whether necessary to compute inverse Jacobian or Hessian
            #
            if any(der[0]==1 or der[0]==2 for der in derivatives):
                assert jac_p2r is not None, 'When computing first or second '+\
                    'derivatives, input "jac_p2r" is needed.'

            if any(der[0]==2 for der in derivatives):
                assert hess_p2r is not None, 'When computing the second '+\
                    'derivative, input "hess_p2r" is needed.'

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

        #
        # Compute shape functions
        #
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
                        ds_dx = np.array(jac_p2r)
                        for i in range(n_dofs_loc):
                            p[:,i] = self.dphi(local_dofs[i], x_ref, var=i_var).ravel()
                            p[:,i] = ds_dx*p[:,i]
                    elif isinstance(cell, QuadCell):
                        if cell.is_rectangle():
                            #
                            # Rectangular cells are simpler
                            #
                            dst_dxy = np.array([Ji[i_var,i_var] for Ji in jac_p2r])
                            for i in range(n_dofs_loc):
                                p[:,i] = self.dphi(local_dofs[i], x_ref, var=i_var)
                                p[:,i] = dst_dxy*p[:,i]
                        else:
                            #
                            # Quadrilateral cells
                            #
                            ds_dxy = np.array([Ji[0,i_var] for Ji in jac_p2r])
                            dt_dxy = np.array([Ji[1,i_var] for Ji in jac_p2r])
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
                        ds_dx = np.array(jac_p2r)
                        for i in range(n_dofs_loc):
                            p[:,i] = (ds_dx)**2*self.d2phi(local_dofs[i], x_ref, der[1:]).ravel()
                    elif isinstance(cell, QuadCell):
                        if cell.is_rectangle():
                            #
                            # Rectangular cell: mixed derivatives 0
                            #
                            dri_dxi = np.array([Ji[i_var,i_var] for Ji in jac_p2r])
                            drj_dxj = np.array([Ji[j_var,j_var] for Ji in jac_p2r])
                            for i in range(n_dofs_loc):
                                p[:,i] = \
                                    dri_dxi*drj_dxj*self.d2phi(local_dofs[i], x_ref, der[1:])

                        else:
                            #
                            # General quadrilateral
                            #
                            # First partial derivatives of (s,t) wrt xi, xj
                            s_xi = np.array([Ji[0,i_var] for Ji in jac_p2r])
                            s_xj = np.array([Ji[0,j_var] for Ji in jac_p2r])
                            t_xi = np.array([Ji[1,i_var] for Ji in jac_p2r])
                            t_xj = np.array([Ji[1,j_var] for Ji in jac_p2r])

                            # Second mixed partial derivatives of (s,t) wrt xi, xj
                            s_xixj = np.array([Hi[i_var,j_var,0] for Hi in hess_p2r])
                            t_xixj = np.array([Hi[i_var,j_var,1] for Hi in hess_p2r])

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

        TODO: Delete?
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


class TriFE(Element):
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
        Element.__init__(self,dim,element_type)

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


class Basis():
    """
    Finite element basis function, which combines the dofhandler with a
    derivative and a subforest flag.
    """
    def __init__(self, dofhandler, derivative='v', subforest_flag=None):
        """
        Constructor

        Inputs:

            dofhandler: DofHandler (mesh and element)  associated with basis
                function

            derivative: str, derivative of the basis function
                'v', 'vx', 'vy', 'vxy', 'vyx', or 'vyy'
                (first letter is irrelevant)

            subforest_flag: str/int/tuple, submesh marker on which basis
                function is defined.
        """
        #
        # Parse dofhandler
        #
        assert isinstance(dofhandler, DofHandler), \
        'Input "dofhandler" should be of type "DofHandler".'

        self.__dofhandler = dofhandler
        self.__subforest_flag = subforest_flag
        self.__derivative = parse_derivative_info(derivative)

        dofs = self.dofs()
        d2i = -np.ones(dofs[-1]+1, dtype=np.int)
        d2i[dofs] = np.arange(len(dofs))

        self.__i2d = np.array(dofs)
        self.__d2i = d2i

    '''
    def __eq__(self, other):
        """
        Define equality
        """
        return self.__dict__ == other.__dict__
    '''

    def dofhandler(self):
        """
        Returns Basis function's dofhandler
        """
        return self.__dofhandler


    def mesh(self):
        """
        Returns
        -------
        mesh : Mesh
            The mesh on which the basis is defined
        """
        return self.dofhandler().mesh


    def subforest_flag(self):
        """
        Returns the subforest flag associated with the Basis
        """
        return self.__subforest_flag


    def set_subforest_flag(self, flag):
        """
        Description:
        ------------
        Set mesh flag on which basis is defined.

        Parameters
        ----------
        flag : double/str, default=None
            Flag used to identify submesh.

        Returns
        -------
        None.

        """
        self.__subforest_flag = flag


    def derivative(self):
        """
        Returns Basis functions' derivative information
        """
        return self.__derivative


    def same_mesh(self, basis=None, mesh=None, subforest_flag=None):
        """
        Determine whether input basis mesh/flag matches a reference mesh,
        specified either directly, or by via another basis function.

        Inputs:

            basis: Basis, to be compared.

            mesh: Mesh,

            subforest_flag: submesh flag

        Output:

            issame: bool, indicating whether basis functions share mesh
                and subforest_flag

        NOTE: We check only whether the Mesh adress matches. Basis functions
            defined on different instances of the identical mesh don't have
            the same mesh.
        """
        if basis is not None:
            #
            # Use basis
            #
            ref_mesh = basis.dofhandler().mesh
            ref_flag = basis.subforest_flag()
        else:
            #
            # Use mesh and flag
            #
            assert mesh is not None, 'Specify either basis or mesh'
            ref_mesh = mesh
            ref_flag = subforest_flag


        #
        # Basis functions differ in their dofhandlers
        #
        if self.dofhandler().mesh != ref_mesh:
            return False

        #
        # Basis functions differ in their subforest_flags
        #
        if self.subforest_flag() != ref_flag:
            return False

        # Same
        return True


    def same_dofs(self, basis):
        """
        Determine whether input basis has the same dofhandler and subforest
        flag as self.

        Inputs:

            basis: Basis, to be compared.

        Output:

            issame: bool, indicating whether basis functions share dofhandler
                and subforest_flag
        """
        #
        # Basis functions differ in their dofhandlers
        #
        if self.dofhandler() != basis.dofhandler():
            return False

        #
        # Basis functions differ in their subforest_flags
        #
        if self.subforest_flag() != basis.subforest_flag():
            return False

        # Same
        return True


    def dofs(self, cell=None):
        """
        Returns the dofs of the shape functions defined over a cell.
        """
        dofhandler = self.dofhandler()
        subforest_flag = self.subforest_flag()

        if cell is not None:
            #
            # Cell provided
            #

            # Determine smallest cell in subforest that contains given cell
            e_cell = cell.nearest_ancestor(subforest_flag)

            # Get dofs
            dofs = dofhandler.get_cell_dofs(e_cell)
        else:
            #
            # Get mesh dofs
            #
            dofs = dofhandler.get_region_dofs(subforest_flag=subforest_flag)
        return dofs


    def n_dofs(self):
        """
        Get numer of dofs
        """
        return self.dofhandler().n_dofs(subforest_flag=self.subforest_flag())


    def d2i(self, dofs):
        """
        Dof-to-index mapping
        """
        return self.__d2i[dofs]


    def i2d(self, idx):
        """
        Index-to-dof mapping
        """
        return self.__i2d[idx]


    def eval(self, x, cell, location='reference'):
        """
        Description
        -----------
        Evaluate a basis function at a set of points on the given cell. The 
        evaluation is achieved by evaluating shape functions at related
        reference points in the reference element (location='reference'). If
        the points lie in the cell (location='physical'), they must first be 
        mapped to the reference element. 
        
        
        Parameters
        ----------
        x : list of Vertex objects, or 2D array of points
            Points (either in the physical or reference cell at which to 
            evaluate the basis functions.
            
        cell : Cell,
            Cell at which to evaluate the basis function.
            
        location : {'reference' (default), 'physical'}, location of the points.
            This is not checked explicitly. 
        
        
        Returns
        -------
        phi : double
            The (n_points, n_dofs) array of n_dofs basis functions (defined on 
            the cell's nearest ancestor) evaluated at the n_points points.
             
        
        Notes
        -----
        Mapping points from physical- to reference cells can be computationally 
        costly when the cells are not rectangular. 
        """
        element = self.dofhandler().element
        subforest_flag = self.subforest_flag()
        
        #
        # Determine smallest cell in subforest that contains given cell
        #
        e_cell = cell.nearest_ancestor(subforest_flag)

        #
        # Map points to reference if necessary
        # 
        if location=='physical':
            #
            # Points lie in physical cell -> first map them to the reference
            # 
            x = cell.reference_map(x)
        #
        # Evaluate the Basis function at the given reference points
        #
        phi = element.shape(x, derivatives=self.derivative(), cell=e_cell)
        return phi


class DofHandler(object):
    """
    Degrees of freedom handler
    """
    def __init__(self, mesh, element):
        """
        Constructor
        """
        #
        # Check and store mesh
        #
        assert isinstance(mesh, Mesh), \
            'Input "mesh" should be of type "Mesh".'
        self.mesh = mesh
        
        #
        # Check element
        #
        assert isinstance(element, Element), \
            'Input "element" should be of type "Element".'
        self.element = element
        
        self.__global_dofs = {}
        self.constraints = {'constrained_dofs': [],
                            'supporting_dofs': [],
                            'coefficients': [],
                            'affine_terms': []}
        self.__hanging_nodes = {}
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

        Input:

            subforest_flag: str/int, identifies the submesh on which to
                distribute dofs
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
                        nbr = cell.get_neighbor(pivot, mode='level-wise')
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

    '''
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
    '''

    '''
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
        Returns all global dofs (sorted) of a specific entity type within a mesh region.


        Inputs:

            entity_type: str, specifying the type of entities whose dofs we seek.
                If None, then return all dofs within cell. Possible values:
                'cell', 'half_edge', 'interval', 'vertex'

            entity_flag: str/int/tuple, marker used to specify subset of entities

            interior: bool, if True only return dofs associated with entity interior
                (See "get_cell_dofs")

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
        dofs = self.get_region_dofs(subforest_flag=subforest_flag)
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
                                # TODO: This is not complete
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
                        dofs = self.get_cell_dofs(nb)
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
                                    hn_dof = self.get_cell_dofs(he_ch.cell())
                                    i_child = rhe_ch.cell().get_node_position()
                                    constrained_dof = hn_dof[v.get_pos(1, i_child)]

                                    hanging_nodes[constrained_dof] = \
                                        (supporting_dofs, constraint_coefficients)

                                    if constrained_dof not in self.constraints['constrained_dofs']:
                                        #
                                        # Update Constraints
                                        # TODO: Delete
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
            self.__hanging_nodes['default'] = hanging_nodes


    def get_hanging_nodes(self, subforest_flag=None):
        """
        Returns hanging nodes of current (sub)mesh
        """
        if subforest_flag is None:
            #
            # Default
            #
            if 'default' not in self.__hanging_nodes:
                #
                # Set hanging nodes first
                #
                self.set_hanging_nodes()
            return self.__hanging_nodes['default']
        else:
            #
            # With Subforest flag
            #
            if subforest_flag not in self.__hanging_nodes:
                #
                # Set hanging nodes first
                #
                self.set_hanging_nodes(subforest_flag=subforest_flag)
            return self.__hanging_nodes[subforest_flag]


    def has_hanging_nodes(self, subforest_flag=None):
        """
        Determine whether there are hanging nodes

        TODO: Delete
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