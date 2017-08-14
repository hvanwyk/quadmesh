import numpy as np
from scipy import sparse, linalg
import numbers
from mesh import QuadCell, Edge
from bisect import bisect_left       
from _operator import index
from itertools import count

"""
Finite Element Classes
"""

class FiniteElement(object):
    """
    Parent Class: Finite Elements
    """
    def __init__(self, dim, element_type):   
        self.__element_type = element_type
        self.__dim = dim    
        self._cell_type = None
        
    def dim(self):
        """
        Returns the spatial dimension
        """
        return self.__dim
     
     
    def cell_type(self):
        """
        Returns 'quadrilateral', 'triangle' or None
        """
        return self._cell_type
        
        
class QuadFE(FiniteElement):
    """
    Continuous Galerkin finite elements on quadrilateral cells 
    """
    def __init__(self, dim, element_type):
        FiniteElement.__init__(self, dim, element_type)
        
        if element_type == 'Q0':
            """
            -------------------------------------------------------------------
            Constant Elements
            -------------------------------------------------------------------
            
            -----     
            | 0 |
            -----
            
            TODO: Finish
            """
            p = lambda x: np.ones(shape=x.shape)
            px = lambda x: np.zeros(shape=x.shape)
            dofs_per_vertex = 0
            dofs_per_edge = 0
            dofs_per_cell = 1
         
        elif element_type == 'Q1':
            """
            -------------------------------------------------------------------
            Linear Elements
            -------------------------------------------------------------------
        
            2---3
            |   |
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
                ref_nodes = [0.0,1.0]
            elif dim == 2:
                #
                # Two Dimensional
                #
                dofs_per_vertex = 1
                dofs_per_edge = 0
                dofs_per_cell = 0
                basis_index = [(0,0),(1,0),(0,1),(1,1)]
                ref_nodes = np.array([[0,0],[1,0],[0,1],[1,1]])
                pattern = ['SW','SE','NW','NE']
        
        elif element_type == 'Q2':
            """
            -------------------------------------------------------------------
            Quadratic Elements 
            -------------------------------------------------------------------
        
            2---7---3
            |       |
            4   8   5 
            |       |
            0---6---1
        
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
                dofs_per_edge = 0
                dofs_per_cell = 1
                basis_index = [0,1,2]
                ref_nodes = np.array([0.0,1.0,0.5])
            elif dim == 2:
                #
                # Two Dimensional
                #
                dofs_per_vertex = 1 
                dofs_per_edge = 1
                dofs_per_cell = 1
                basis_index = [(0,0),(1,0),(0,1),(1,1),
                               (0,2),(1,2),(2,0),(2,1),(2,2)]
                ref_nodes = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0],
                                  [0.0,0.5],[1.0,0.5],[0.5,0.0],[0.5,1.0],
                                  [0.5,0.5]])
                pattern = ['SW','SE','NW','NE','W','E','S','N','I']
            else:
                raise Exception('Only 1D and 2D currently supported.')
             
        elif element_type == 'Q3':
            """
            -------------------------------------------------------------------
            Cubic Elements
            -------------------------------------------------------------------
            
            2---10---11---3
            |             |
            5   14   15   7
            |             |
            4   12   13   6
            |             |
            0----8---9----1 
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
                dofs_per_edge = 0
                dofs_per_cell = 2
                basis_index = [0,1,2,3]
                ref_nodes = np.array([0.0,1.0,1/3.0,2/3.0])
            elif dim == 2:
                #
                # Two Dimensional
                #
                dofs_per_vertex = 1 
                dofs_per_edge = 2
                dofs_per_cell = 4
                basis_index = [(0,0),(1,0),(0,1),(1,1),
                               (0,2),(0,3),(1,2),(1,3),(2,0),(3,0),(2,1),(3,1),
                               (2,2),(3,2),(2,3),(3,3)]
                ref_nodes = np.array([[0.0,0.0],[1.0,0.0],
                                      [0.0,1.0],[1.0,1.0],
                                      [0.0,1./3.],[0.0,2./3.],
                                      [1.0,1./3.],[1.0,2./3.], 
                                      [1./3.,0.0],[2./3.,0.0], 
                                      [1./3.,1.0],[2./3.,1.0],
                                      [1./3.,1./3.],[2./3.,1./3.],
                                      [1./3.,2./3.],[2./3.,2./3.]])
                pattern = ['SW','SE','NW','NE','W','W','E','E','S','S','N','N',
                           'I','I','I','I']
        self._cell_type = 'quadrilateral' 
        self.__dofs = {'vertex':dofs_per_vertex, 'edge':dofs_per_edge,'cell':dofs_per_cell}               
        self.__basis_index = basis_index
        self.__p = p
        self.__px = px
        self.__pxx = pxx
        self.__element_type = element_type
        self.__ref_nodes = ref_nodes
        if dim == 2:
            self.pattern = pattern
    
    
    def local_dof_matrix(self):
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
            local_dof_matrix[[0,0,-1,-1],[0,-1,0,-1]] = np.arange(4)
            count += 4
            
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
        
    def basis_index(self):
        return self.__basis_index
    
      
    #def cell_type(self):
    #    return self.__cell_type
    
    
    def element_type(self):
        """
        Return the finite element type (Q1, Q2, or Q3)
        """ 
        return self.__element_type
    
        
    def polynomial_degree(self):
        """
        Return the finite element's polynomial degree 
        """
        return int(list(self.__element_type)[1])
    
        
    def n_dofs(self,key=None):
        """
        Return the number of dofs per elementary entity
        """
        # Total Number of dofs
        if key is None:
            d = self.dim()
            return 2**d*self.__dofs['vertex'] + \
                   2*d*self.__dofs['edge'] + \
                   self.__dofs['cell']
        else:
            assert key in self.__dofs.keys(), 'Use "vertex","edge", "cell" for key'
            return self.__dofs[key]
    
    def loc_dofs_on_edge(self,direction):
        """
        Returns the local dofs on a given edge
        """    
        edge_dofs = []
        for i in range(self.n_dofs()):
            if direction in self.pattern[i]:
                edge_dofs.append(i)
        return edge_dofs
    
    
    def pos_on_edge(self, direction):
        """
        Returns the positions of dofs along each edge in order
        from left-to-right and low-to-high.
        
        Input:
        
            direction: str, cartographic direction (WESN)
            
        Output:
        
            pos_ordered: list, positions. 
        """
        assert direction in ['N','S','E','W'], 'Direction not supported.'
        positions = []
        count = 0
        for pos in self.pattern:
            if pos == direction:
                positions.append((pos,count))
                count += 1
            elif direction in pos:
                positions.append(pos)
            
        min_pos = 'S'+direction if direction in ['E','W'] else direction+'W'
        max_pos = 'N'+direction if direction in ['E','W'] else direction+'E'
  
        dpe = self.n_dofs('edge')
        pos_ordered = [min_pos] + [(direction,i) for i in range(dpe)] + [max_pos]       
        return pos_ordered
    
    
    def reference_nodes(self):
        """
        Returns vertices used to define nodal basis functions on reference cell
        """
        return self.__ref_nodes
    
     
        
    def phi(self, n, x):
        """
        Evaluate the nth basis function at the point x
        
        Inputs: 
        
            n: int, basis function number
            
            x: double, point at which function is to be evaluated
               (double if dim=1, or tuple if dim=2) 
        """
        x = np.array(x)
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
            # TODO: Doesn't work for tuples...
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
         
         
    def constraint_coefficients(self):
        """
        Returns the constraint coefficients of a typical bisected edge. 
        Vertices on the coarse edge are numbered in increasing order, e.g. 0,1,2,3 for Q2,
        
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
        
    
    def clear_dofs(self):
        """
        Clear all dofs
        """
        self.__global_dofs = {}
        self.__dof_count = 0
        
                
    def distribute_dofs(self, nested=False):
        """
        global enumeration of degrees of freedom
        
        Note: When root's children are in a grid, then the root has no DOFs 
        """
        #
        # Ensure the mesh is balanced
        # 
        assert self.mesh.is_balanced(), \
            'Mesh must be balanced before dofs can be distributed.'
            
        if not nested:
            for node in self.mesh.root_node().find_leaves():
                # 
                # Fill in own nodes
                # 
                self.fill_dofs(node)
                # 
                # Share dofs with neighbors
                # 
                self.share_dofs_with_neighbors(node)
        else:
            for node in self.mesh.root_node().traverse_depthwise():
                if node.type == 'ROOT' and node.grid_size() is not None:
                    pass
                else:       
                    #
                    # Fill in own nodes
                    # 
                    self.fill_dofs(node)
                    
                    #
                    # Share dofs with neighbors
                    # 
                    self.share_dofs_with_neighbors(node, nested=True)
                    
                    #
                    # Share dofs with children
                    # 
                    self.share_dofs_with_children(node)
            
    
    def share_dofs_with_children(self, node):
        """
        Assign shared degrees of freedom with children 
        
        Inputs:
        
            node: Node, whose global dofs are known
         
        Note: Cannot share dofs with children when children are in a grid
        """
        if node.has_children():
            
            cell_dofs = self.__global_dofs[node][:]
            cell_dofs = [-1 if c is None else c for c in cell_dofs]

            #
            # Construct useful array to store dofs of parent cell
            # 
            dps = self.element.n_dofs('edge')+2
            n_fine = 2*dps-1
            fine_dofs = -np.ones((n_fine,n_fine))
            m = self.element.local_dof_matrix().astype(np.int)
            i2 = [2*i for i in range(dps)]
            fine_dofs[np.ix_(i2,i2)] = np.array(cell_dofs)[m]
            #
            # Extract child dofs as sub-matrices
            # 
            for pos in node.children.keys():
                child = node.children[pos]
                if child is not None:
                    #
                    # Determine sub-indices
                    #
                    i_pos,j_pos = pos
                    if i_pos == 'S':
                        y_rng = np.arange(dps)
                    else:
                        y_rng = np.arange(dps-1,2*dps-1)
                    if j_pos == 'W':
                        x_rng = np.arange(dps)
                    else:
                        x_rng = np.arange(dps-1,2*dps-1)
                    #
                    # Select sub-array corresponding to child position 
                    #
                    child_dofs = fine_dofs[np.ix_(y_rng,x_rng)]
                    dofs = child_dofs[child_dofs!=-1]
                    position = self.element.local_dof_matrix()[child_dofs!=-1]
                    
                    #
                    # Assign global dofs to cell 
                    #
                    position = [int(p) for p in position] 
                    dofs = [int(d) for d in dofs]
                    self.assign_dofs(child, position, dofs) 
                                     
                    
    def share_dofs_with_neighbors(self, node, nested=False):
        """
        Assign shared degrees of freedom to neighboring cells
        
        Inputs: 
        
            node: Node, cell in quadmesh
            
            dof_list: list, complete list of cell's global degrees of freedom
            
        Notes:
            
            We assume that the mesh is balanced 
            
        """
        
        opposite = {'N':'S', 'S':'N', 'W':'E', 'E':'W', 
                    'SW':'NE','NE':'SW','SE':'NW','NW':'SE'}
        dof_list = self.__global_dofs[node][:]
        #
        # Diagonal directions
        #
        for diag_dir in ['SW','SE','NW','NE']:
            nb = node.find_neighbor(diag_dir)
            if nb != None:
                pos = self.pos_to_int(diag_dir)
                dof = dof_list[pos]
                opp_dir = opposite[diag_dir]
                opp_pos = self.pos_to_int(opp_dir)
                if nested:
                    self.assign_dofs(nb, opp_pos, dof)
                    self.share_dofs_with_children(nb)
                else:  
                    if nb.has_children(opp_dir):
                        nb = nb.children[opp_dir]
                    self.assign_dofs(nb,opp_pos,dof)    
                
        #
        # W, E, S, N
        # 
        sub_pos = {'E':['SE','NE'], 'W':['SW','NW'], 
                   'N':['NW','NE'], 'S':['SW','SE']}
        dpe = self.element.n_dofs('edge')
        ref_index = range(0,dpe+2) 
        coarse_index = [2*r for r in ref_index]
        for direction in ['W','E','S','N']:
            opp_dir = opposite[direction]
            n_pos = self.element.pos_on_edge(direction)
            dofs = [dof_list[i] for i in self.pos_to_int(n_pos)]
            nb = node.find_neighbor(direction)
            if nb != None:
                if not nested and nb.has_children():
                    #
                    # Neighboring cell has children
                    # 
                    ch_count = 0
                    for sp in sub_pos[opp_dir]:
                        child = nb.children[sp]
                        if child != None:
                            ch_pos = \
                                self.element.pos_on_edge(opp_dir)
                            fine_index = \
                                [r+(dpe+1)*ch_count for r in ref_index]
                            to_pos = []
                            to_dofs = []
                            for i in range(len(fine_index)):
                                if fine_index[i] in coarse_index:
                                    to_pos.append(ch_pos[i])
                                    j = coarse_index.index(fine_index[i])
                                    to_dofs.append(dofs[j])
                            self.assign_dofs(child,to_pos,to_dofs)
                        ch_count += 1
                elif nb.depth == node.depth:
                    #
                    # Same size cell
                    #
                    nb_pos = self.element.pos_on_edge(opp_dir)
                    self.assign_dofs(nb, nb_pos, dofs)
                    if nested:
                        self.share_dofs_with_children(nb)
                elif nb.depth < node.depth:
                    #
                    # Neighbor larger than self
                    # 
                    nb_pos = self.element.pos_on_edge(opp_dir)
                    offset = sub_pos[direction].index(node.position)
                    fine_index = [r+(dpe+1)*offset for r in ref_index]
                    to_pos = []
                    to_dofs = []
                    for i in range(len(coarse_index)):
                        if coarse_index[i] in fine_index:
                            to_pos.append(nb_pos[i])
                            j = fine_index.index(coarse_index[i])
                            to_dofs.append(dofs[j]) 
                    self.assign_dofs(nb, to_pos, to_dofs)

            
    def fill_dofs(self,node):
        """
        Fill in cell's dofs 
        
        Inputs:
        
            node: cell, whose global degrees of freedom are to be augmented
    
        Modify: 
        
            __global_dofs[node]: updated global dofs
            
            __n_global_dofs: updated global dofs count
        """
        dofs_per_cell = self.element.n_dofs()
        if not node in self.__global_dofs:
            #
            # Add node to dictionary if it's not there
            # 
            self.__global_dofs[node] = None
        cell_dofs = self.__global_dofs[node]
        if cell_dofs is None:
            #
            # Instantiate new list
            # 
            count = self.__dof_count
            self.__global_dofs[node] = list(range(count,count+dofs_per_cell))
            self.__dof_count += dofs_per_cell
        else:
            #
            # Augment existing list
            #
            count = self.__dof_count
            for k in range(dofs_per_cell):
                if cell_dofs[k] is None:
                    cell_dofs[k] = count
                    count += 1
            self.__global_dofs[node] = cell_dofs
            self.__dof_count = count        
                    
            
    def assign_dofs(self, node, positions, dofs):
        """
        Assign the degrees of freedom (dofs) to positions in cell (node). 
        The result is stored in the DofHandler's "global_dofs" dictionary. 
        
        Inputs:
        
            node: Node, represents the cell
            
            positions: str, list of positions given by cardinal directions.
            
            dofs: int, list (same length as positions) of degrees of freedom.    
        """    
        #
        # Preprocessing
        #
        if not node in self.__global_dofs:
            #
            # Node not in dictionary -> add it
            # 
            self.__global_dofs[node] = None
            
        cell_dofs = self.__global_dofs[node]
        if cell_dofs is None:
            # 
            # New doflist necessary
            # 
            cell_dofs = [None]*self.element.n_dofs()
        #
        # Turn positions and dofs into lists and check compatibility
        # 
        if not(type(positions) is list):
            positions = [positions]
        if not(type(dofs) is list):
            dofs = [dofs]
        lengths_do_not_match = 'Number of dofs and positions do not match.'
        assert len(positions)==len(dofs),lengths_do_not_match
        #
        # Ensure dofs have correct format
        # 
        dof_is_int = all([type(d) is np.int for d in dofs])
        assert dof_is_int, 'Degrees of freedom should be integers.' 
        dof_is_nonneg = all([d>=0 for d in dofs])
        assert dof_is_nonneg, 'Degrees of freedom should be nonnegative.'
        
        pos_is_int = all([type(p) is np.int for p in positions])
        if not pos_is_int:
            #
            # Convert positions to integers
            # 
            positions = self.pos_to_int(positions)
        
        for pos,dof in zip(positions,dofs):
            if cell_dofs[pos] != None:
                incompatible_dofs = 'Incompatible dofs. Something fishy.'
                assert cell_dofs[pos] == dof, incompatible_dofs
            else:
                cell_dofs[pos] = dof
        
        self.__global_dofs[node] = cell_dofs
        
    
    def pos_to_int(self, positions):
        """
        Convert a list of positions into indices 
        """
        return_int = False 
        if (not type(positions) is list):
            return_int = True
            positions = [positions]
        index = []
        p = self.element.pattern
        for pos in positions:
            if type(pos) is tuple:
                direction, offset = pos
                direction_error ='Only "W,E,S,N,I" admit multiple entries.'
                assert direction in ['W','E','S','N','I'], direction_error
                int_pos = p.index(direction) + offset
            elif type(pos) is np.int:
                int_pos = pos
            else:
                position_error = 'Position %s not recognized.'%(pos)
                assert pos in p, position_error
                int_pos = p.index(pos)
            index.append(int(int_pos))
        if return_int:
            #
            # Return single integer 
            # 
            return index[0]
        else:
            #
            # Return list of integers
            #
            return index
    
    
    def pos_to_dof(self, dof_list, positions):
        """
        Return a list of dofs corresponding to various positions within a cell
        
        Inputs: 
        
            dof_list: int, list of cell's global degrees of freedom
            
            positions: str, list of positions in the form
                
                Mixed: 'NW','NE','SW','SE'
                
                Directions: 'W', 'E', 'S', 'N' or (direction,i), 
                    i ordered from low high, left to right 
                    
                Interior: 'I' or (I,i), i=0,1,.. ordered row-wise bottom to top
            
        Outputs:
        
            dofs: list of degrees of freedom corresponding to given positions
         
        TODO: I want to get rid of this eventually. 
        """
        dofs = []
        p = self.element.pattern

        # Turn positions into list if only one entry
        if not(type(positions) is list):
            positions = [positions]
            
        for pos in positions:
            if type(pos) is tuple:
                direction, offset = pos
                direction_error ='Only "W,E,S,N,I" admit multiple entries.'
                assert direction in ['W','E','S','N','I'], direction_error
                index = p.index(direction) + offset
            else:
                direction_error = 'Position "%s" not recognized.'%(pos)
                assert pos in p, direction_error
                index = p.index(pos)
            dofs.append(dof_list[index])
        if not(type(positions) is list):
            dofs = dofs[0]
        return dofs    
    
    
    def get_global_dofs(self, node, edge_dir=None):
        """
        Return all global dofs corresponding to a given cell, or one of its 
        edges.
        
        Inputs:
        
            node: Node, quadtree node associated with cell
            
            edge_dir: str, edge direction (WESN)
            
            
        Outputs:
        
             global_dofs: list of cell dofs or edge dofs. 
        """
        if node in self.__global_dofs:
            cell_dofs = self.__global_dofs[node]
            if edge_dir is None:
                return cell_dofs
            else: 
                assert edge_dir in ['W','E','S','N'], \
                'Edge should be one of W, E, S, or N.'    
                edge_dofs = []
                for i in range(self.element.n_dofs()):
                    if edge_dir in self.element.pattern[i]:
                        #
                        # If edge appears in an entry, record the dof
                        # 
                        edge_dofs.append(cell_dofs[i])
                return edge_dofs
        else:
            return None
        
         
    
    def n_dofs(self, flag=None):
        """
        Return the total number of degrees of freedom distributed so far
        """
        if flag is None:
            return self.__dof_count
        else:
            #
            # Count dofs explicitly
            # 
            dof_set = set()
            for node in self.mesh.root_node().find_leaves(flag=flag):
                dof_set.update(self.get_global_dofs(node))
            return len(dof_set)
            
     
    def dof_vertices(self, node=None, flag=None):
        """
        Return the mesh vertices (or vertices corresponding to node).
        """
        assert hasattr(self, '_DofHandler__dof_count'), \
            'First distribute dofs.'
        rule = GaussRule(1,shape='quadrilateral')
        x_ref = self.element.reference_nodes()
        if node is not None:
            #
            # Vertices corresponding to a single Node->QuadCell
            # 
            g_dofs = self.get_global_dofs(node)
            x = rule.map(node.quadcell(),x=x_ref)
        else:
            #
            # Vertices over entire mesh
            # 
            x = np.empty((self.n_dofs(flag),2))
            for leaf in self.mesh.root_node().find_leaves(flag=flag):
                g_dofs = self.get_global_dofs(leaf)
                x[g_dofs,:] = rule.map(leaf.quadcell(),x=x_ref)
        return x
    
                
    def set_hanging_nodes(self):
        """
        Set up the constraint matrix satisfied by the mesh's hanging nodes.
        
        Note: Hanging nodes can only be found once the mesh has been balanced.
        """
     
        hanging_nodes = {}
        sub_pos = {'E':['SE','NE'], 'W':['SW','NW'], 
                   'N':['NW','NE'], 'S':['SW','SE']}
        opposite = {'E':'W','W':'E','N':'S','S':'N'}        
        cc = self.element.constraint_coefficients()
        for node, n_doflist in self.__global_dofs.items():
            #
            # Loop over cells in mesh
            #
            for direction in ['W','E','S','N']:
                #
                # Look in every direction
                # 
                n_dof_pos = self.element.pos_on_edge(direction)
                nb = node.find_neighbor(direction)
                if nb != None and nb.has_children():
                    #
                    # Neighbor has children -> resolve their hanging nodes
                    # 
                    opp = opposite[direction]
                    for i in range(2):
                        child = nb.children[sub_pos[opp][i]]
                        if child != None:
                            #
                            # For each of 2 children, get pos and dof info
                            #  
                            ch_dof_pos = self.element.pos_on_edge(opp)
                            ch_doflist = self.__global_dofs[child]
                            for hn in cc[i].keys():
                                #
                                # Loop over generic hanging nodes, store 
                                # global info in hanging_node dictionary.
                                # 
                                coarse_dofs = self.pos_to_dof(n_doflist, n_dof_pos)
                                hn_dof = self.pos_to_dof(ch_doflist,ch_dof_pos[hn])[0]
                                hanging_nodes[hn_dof] = (coarse_dofs,cc[i][hn])
                        else:
                            print('Child is None')
        self.__hanging_nodes = hanging_nodes
           
      
    def get_hanging_nodes(self):
        """
        Returns hanging nodes of current mesh
        """
        if hasattr(self,'__hanging_nodes'): 
            return self.__hanging_nodes
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
                2D rule: order in {1,4,16,25,36} for quadrilaterals
                                  {1,3,7,13} for triangles 
            
            element: FiniteElement object
            
                OR 
            
            shape: str, 'interval' (subset of R^1), 'edge' (subset of R^2), 
                        'triangle', or 'quadrilateral'
             
        """
        if element is None:
            #
            # Shape explicitly given
            # 
            assert shape in ['interval','edge','triangle','quadrilateral'], \
                "Use 'interval','edge', 'triangle', or 'quadrilateral'."
            if shape == 'interval' or shape == 'edge':
                dim = 1
            else:
                dim = 2
        else:  
            #
            # Shape given by element
            # 
            dim = element.dim()
            assert dim in [1,2], 'Only 1 or 2 dimensions supported.'
            shape = element.cell_type()
              
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
        
        
    def nodes(self, direction=None):
        """
        Return quadrature nodes 
        """
        if self.__cell_type == 'edge' and direction is not None:
            #
            # One dimensional rule over edges
            # 
            assert direction in ['W','E','S','N'], \
                'Only directions W,E,S, and N supported.'
            edge_dict = {'W':[(0,0),(0,1)], 
                         'E':[(1,0),(1,1)],
                         'S':[(0,0),(1,0)],
                         'N':[(0,1),(1,1)]}
            verts = edge_dict[direction]
            verts.sort()
            x0,y0 = verts[0]
            x1,y1 = verts[1]
            x_ref = self.__nodes 
            x = x0 + x_ref*(x1-x0)
            y = y0 + x_ref*(y1-y0)
            return np.array([x,y]).T
        else:
            #
            # Return 1D/2D nodes on reference entity 
            # 
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
    
        
    def map(self, entity, x=None):
        """
        Map from reference to physical cell
        
        Inputs:
        
            entity: QuadCell or Edge to which points are mapped
            
            x: double, points in reference cell in the form of 
                either (i) a length n list of dim-tuples or 
                (ii) an (n,dim) array  
                
        TODO: You cannot map reference edge nodes onto physical edge nodes
            with one dimensional rule. 
        Note: You can map a 1D rule onto an Edge by either 
            (i) directly mapping the 1d nodes onto an edge entity, OR
            (ii) specifying the edge location on the reference cell and then
                mapping the 2D nodes onto the cell. 
        """
        dim = self.__dim
        cell_type = self.__cell_type
        if x is None:
            x_ref = self.__nodes
        else:
            x_ref = np.array(x)
        if dim == 1:
            #
            # One dimensional mesh
            # 
            if cell_type == 'interval':
                #
                # Interval on real line
                # 
                x0, x1 = entity.box()
                x_phys = x0 + (x1-x0)*x_ref
            elif cell_type == 'edge':
                # 
                # Line segment in 2D
                # 
                x0,x1,y0,y1 = entity.box()
                x = x0 + x_ref*(x1-x0)
                y = y0 + x_ref*(y1-y0)
                x_phys = np.array([x,y]).T              
        elif dim == 2:
            #
            # Two dimensional mesh
            # 
            if cell_type == 'triangle':
                #
                # Triangles not supported yet
                #  
                pass
            elif cell_type == 'quadrilateral':
                x0,x1,y0,y1 = entity.box()
                x_phys = np.array([x0 + (x1-x0)*x_ref[:,0], 
                                   y0 + (y1-y0)*x_ref[:,1]]).T
        return x_phys


    def inverse_map(self, entity, x, mapto='2d'):
        """
        Return a point in the given entity, mapped to the standard reference
        domain. 
        
        Inputs:
        
            entity: QuadCell or (Edge,direction) containing points x
            
            x: (n_points,dim) numpy array of points in cell/on edge
            
            mapto: str, '2d' - map points to 2d reference domain
                        '1d' - map points on edge (or 1d cell) to 1d ref domain 
            
        Output:
        
            x_ref: numpy array of equivalent points on the reference cell 
        
        """
        dim = len(x.shape)
        cell_type = self.__cell_type
        x = np.array(x)
        n_points = x.shape[0]
        if dim==1:
            #
            # Map from interval to reference interval
            #
            x0, x1 = entity.box()
            x_ref = (x - x0)/(x1-x0)
        elif cell_type=='edge':
            #
            # Map from edge
            #
            assert len(entity)==2 and isinstance(entity[0],Edge),\
                'entity should be a tuple (Edge,direction)' 
            if mapto=='1d':
                #
                # Map to 1D reference           
                # 
                if entity[1] in ['W','E']:
                    y0,y1 = entity[0].box()
                    x_ref = (x[:,1]-y0)/(y1-y0)
                elif entity[1] in ['N','S']:
                    x0,x1 = entity[0].box()
                    x_ref = (x[:,0]-x0)/(x1-x0)
            elif mapto=='2d':
                x_ref = np.zeros((n_points,2))
                #
                # Map to 2D reference
                #
                if entity[1] in ['W','E']:
                    y0,y1 = entity[0].box()
                    x_ref[:,1] = (x[:,1]-y0)/(y1-y0)
                    if entity[1]=='E':
                        x_ref[:,0] = 1.0
                elif entity[1] in ['N','S']:
                    x0,x1 = entity[0].box()
                    x_ref[:,0] = (x[:,0]-x0)/(x1-x0)
                    if entity[1]=='N':
                        x_ref[:,1] = 1.0
        elif cell_type=='quadrilateral':
            #
            # Map from edge to 2d reference 
            #
            assert isinstance(entity, QuadCell), 'Entity should be a cell.'
            x0,x1,y0,y1 = entity.box()
            x_ref = np.zeros((n_points,2))
            x_ref[:,0] = (x[:,0]-x0)/(x1-x0)
            x_ref[:,1] = (x[:,1]-y0)/(y1-y0)
        elif cell_type == 'triangle':
            raise Exception('Triangles not supported (yet).')        
        return x_ref
        
        
    def jacobian(self, entity):
        """
        Jacobian of the Mapping from reference to physical cell
        """
        dim = self.__dim
        entity_type = self.__cell_type
        if dim == 1:
            #
            # One dimensional mesh
            # 
            if entity_type == 'interval':
                x0, x1 = entity.box()
                jac = x1-x0
            elif entity_type == 'edge':
                # Length of edge
                jac = entity.length()
                
        elif dim == 2:
            #
            # Two dimensional mesh
            #
            if entity_type == 'triangle':
                #
                # Triangles not yet supported
                # 
                pass
            elif entity_type == 'quadrilateral':
                x0,x1,y0,y1 = entity.box()
                jac = (x1-x0)*(y1-y0)
        return jac
        
 
    def rule(self, entity):
        """
        Returns the Gauss nodes and weights on a physical entity - QuadCell, 
        Edge, TriCell, or line segment -  
        """
        x_ref = self.map(entity)
        w = self.weights()
        jac = self.jacobian(entity)
        
        return x_ref, w*jac
        
    
class System(object):
    """
    (Non)linear system to be defined and solved 
    """
    def __init__(self, mesh, element, n_gauss=(4,16), nested=False):
        """
        Set up linear system
        
        Inputs:
        
            mesh: Mesh, finite element mesh
            
            element: FiniteElement, shapefunctions
            
            n_gauss: int tuple, number of quadrature nodes in 1d and 2d respectively
                        
            bnd_conditions: dictionary of boolean functions for marking boundaries
                and boundary data in the form
                {'dirichlet':[m_d,g_d],'neumann':[m_n,g_n], 
                 'robin':[m_r,(gamma,g_r)], 'periodic':m_p}
                where m_i maps a node/edge to a boolean and  
                
    
        """
        self.__mesh = mesh
        self.__element = element
        self.__n_gauss_2d = n_gauss[1]
        self.__n_gauss_1d = n_gauss[0]
        self.__rule_1d = GaussRule(self.__n_gauss_1d,shape='edge')
        self.__rule_2d = GaussRule(self.__n_gauss_2d,shape=element.cell_type())
        self.__dofhandler = DofHandler(mesh,element)
        self.__dofhandler.distribute_dofs(nested=nested)
        # Initialize reference shape functions
        dlist = [(0,),(1,0),(1,1),(2,0,0),(2,0,1),(2,1,0),(2,1,1)]
        self.__phi = {'cell':       dict.fromkeys(dlist, None),
                      ('edge','W'): dict.fromkeys(dlist, None),
                      ('edge','E'): dict.fromkeys(dlist, None),
                      ('edge','S'): dict.fromkeys(dlist, None),
                      ('edge','N'): dict.fromkeys(dlist, None)}  
    
    
    def dofhandler(self):
        """
        Returns dofhandler
        """
        return self.__dofhandler
    
    
    def assemble(self, bilinear_forms=None, linear_forms=None, 
                 boundary_conditions=None):
        """
        Assembles linear system associated with a weak form and accompanying
        boundary conditions. 
        
        Inputs: 
        
            bilinear_forms: 3-tuples (function,string,string), where
                function is the kernel function, the first string 
                ('u','ux',or 'uy) is the form of the trial function, and
                the second string ('v','vx','vy') is the form of the test
                functions. 
            
            linear_forms: 2-tuples (function, string), where the function
                is the kernel function, and the string ('v','vx','vy')
                is the form of the test function.
            
            boundary_conditions: dictionary whose keys are
                'dirichlet', 'neumann', 'robin', and 'periodic' (not implemented)
                and whose values are lists of tuples (m_bnd, d_bnd), where
                m_fun is used to identify a specific boundary (from either the
                edge [the case for neumann and robin conditions] or from nodal
                values [the case for dirichlet conditions], and d_bnd is the 
                data associated with the given boundary condition: 
                For 'dirichlet': u(x,y) = d_bnd(x,y) on bnd
                    'neumann'  : -n.q*nabla(u) = d_bnd(x,y) on bnd
                    'robin'    : d_bnd = (gamma, g_rob), so that 
                                -n.q*nabla(u) = gamma*(u(x,y)-d_bnd(x,y))
            
        Outputs:
        
            A: double coo_matrix, system matrix determined by bilinear forms and 
                boundary conditions.
                
            b: double, right hand side vector determined by linear forms and 
                boundary conditions.
        
        
        TODO: Include support for tensors. 
        TODO: Include option to assemble multiple matrices     
        """        
        n_nodes = self.__dofhandler.n_dofs()
        n_dofs = self.__element.n_dofs()   
  
        #
        # Determine the forms to assemble
        #
        if bilinear_forms is not None:
            assert type(bilinear_forms) is list, \
                'Bilinear form should be passed in list.'
            bivals = []
        
        if linear_forms is not None:
            linvec = np.zeros((n_nodes,))
 
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
        for node in self.__mesh.root_node().find_leaves():
            node_dofs = self.__dofhandler.get_global_dofs(node)
            cell = node.quadcell()            
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
                                        gamma_rob*self.form_eval((g_rob,'u','v'),\
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
                x_ref = self.__element.reference_nodes()
                x_cell = self.__rule_2d.map(cell,x=x_ref) 
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
        
        hanging_nodes = self.__dofhandler.get_hanging_nodes()
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
    
        
    def get_n_nodes(self):
        """
        Return the number of dofs 
        """
        return self.__dofhandler.n_dofs()
    
    
    def get_global_dofs(self,node):
        """
        Return the degrees of freedom associated with node
        """             
        return self.__dofhandler.get_global_dofs(node)
    
    
    def get_edge_dofs(self,node,direction):
        """
        Return the degrees of freedom associated with edge
        """
        return self.__dofhandler.get_global_dofs(node, direction)
    
    
    def dof_vertices(self):
        """
        Returns the locations of all degrees of freedom
        """
        return self.__dofhandler.dof_vertices()
     
     
    def x_loc(self,cell):
        """
        Return the vertices corresponding to the local cell dofs 
        """   
        x_ref = self.__element.reference_nodes()
        return self.__rule_2d.map(cell,x=x_ref)
        
        
    def bilinear_loc(self,weight,kernel,trial,test):
        """
        Compute the local bilinear form over an element
        """
        return np.dot(test.T, np.dot(np.diag(weight*kernel),trial))
    
    
    def linear_loc(self,weight,kernel,test):
        """
        Compute the local linear form over an element
        """
        return np.dot(test.T, weight*kernel)
    
    
    def shape_eval(self, derivatives=(0,), cell=None,\
                   edge_loc=None, x=None, x_ref=None):
        """
        Evaluate all shape functions at a set of reference points x. If x is 
        not specified, Gauss quadrature points are used. 
        
        Inputs: 
        
            derivatives: tuple specifying the order of the derivative and 
                the variable 
                (0,)  : function evaluation, (default) 
                (1,i) : 1st derivative wrt ith variable, or 
                (2,i,j): 2nd derivative wrt ith and jth variables (i,j in 0,1).
                
            cell: QuadCell, entity over which we evaluate the shape functions
            
            edge_loc: str, specifying edge location (W,E,S,N).
                                      
            x: double, np.array of points in the physical cell
            
            x_ref: double, np.array of points in the reference cell
             
                
        Output:
        
            phi: (n_points,n_dofs) array, the jth column of which is the 
                (derivative of) the jth shape function evaluated at the 
                specified points. 
        """
        #
        # Determine multiplier for derivatives (chain rule)
        # 
        c = 1
        if derivatives[0] in {1,2} and cell is not None:
            # 
            # There's a map and we're taking derivatives
            #
            x0,x1,y0,y1 = cell.box()
            for i in derivatives[1:]:
                if i==0:
                    c *= 1/(x1-x0)
                elif i==1:
                    c *= 1/(y1-y0)
        #
        # Determine entity
        #
        if edge_loc is None:
            entity = 'cell'
        else:
            assert edge_loc in ['W','E','S','N'], \
            'Edge should be one of "W","E","S", or "N"'
            entity = ('edge',edge_loc)
        #
        # Parse points x/x_ref
        # 
        for_quadrature = False
        if x is None and x_ref is None:
            #
            # Default: use quadrature points
            #
            for_quadrature = True
            if self.__phi[entity][derivatives] is not None:
                #
                # Phi already defined, return it
                # 
                return c*self.__phi[entity][derivatives]
            else:
                #
                # Must evaluate phi
                # 
                if edge_loc is None:
                    x_ref = self.cell_rule().nodes()
                else:
                    x_ref = self.edge_rule().nodes(direction=edge_loc)
        elif x_ref is None and x is not None:
            #
            # Points specified on physical cell, mapp them to reference
            # 
            x_ref = self.cell_rule().inverse_map(cell,x)
        else:
            #
            # x_ref specified directly
            #
            assert (x_ref is not None) and (x is None),\
            'Ambiguous. Points defined on reference and physical domains.'
        
        #
        # Evaluate shape functions at reference points
        #
        x_ref = np.array(x_ref)
        n_dofs = self.__element.n_dofs()               
        n_points = x_ref.shape[0] 
        phi = np.zeros((n_points,n_dofs))
        if derivatives[0] == 0:
            #
            # No derivatives
            #
            for i in range(n_dofs):
                phi[:,i] = self.__element.phi(i,x_ref)  
        elif derivatives[0] == 1:
            # 
            # First derivatives
            #
            i_var = derivatives[1]
            for i in range(n_dofs):
                phi[:,i] = self.__element.dphi(i,x_ref,var=i_var)
        elif derivatives[0]==2:
            #
            # Second derivatives
            #         
            for i in range(n_dofs):
                phi[:,i] = self.__element.d2phi(i,x_ref,derivatives[1:])
                    
        if for_quadrature and self.__phi[entity][derivatives] is None:
            #
            # Store shape function (at quadrature points) for future use
            # 
            self.__phi[entity][derivatives] = phi
        return c*phi
             
    
    def f_eval(self, f, x, derivatives=(0,)):
        """
        Evaluate a function (or its partial derivatives) at a set of points in
        the domain
        
        Inputs: 
        
            f: function to be evaluated, either defined explicitly, or by its 
                node values, or as a mesh function (cellwise)
                
            x: double, (n,dim) array of points at which to evaluate f.
            
            
        Output:
        
            f_vec: double, (n,) vector of function values at the interpolation
                points.
        """
        dim = 1 if len(x.shape)==1 else x.shape[1]
        if callable(f):
            #
            # Function explicitly given
            #
            if dim==1:
                f_vec = f(x)
            elif dim==2:
                assert derivatives==(0,), \
                    'Unable to take derivatives of function directly. Discretize'
                f_vec = f(x[:,0],x[:,1])
            else:
                raise Exception('Only 1D and 2D points supported.')
        elif len(f)==self.__mesh.get_number_of_cells():
            #
            # Mesh function
            # 
            f_vec = np.empty(x.shape[0])
            f_vec[:] = np.nan
            count = 0
            for node in self.__mesh.root_node().find_leaves():
                cell = node.quadcell()
                in_cell = cell.contains_point(x)
                f_vec[in_cell] = f[count]
                count += 1 
        elif len(f)==self.get_n_nodes():
            #
            # Nodal function
            # 
            f_vec = np.empty(x.shape[0])
            f_vec[:] = np.nan
            for node in self.__mesh.root_node().find_leaves():
                cell = node.quadcell()
                f_loc = f[self.get_global_dofs(node)]
                in_cell = cell.contains_point(x)
                x_loc = x[in_cell,:]
                f_vec[in_cell] = \
                    self.f_eval_loc(f_loc, cell, derivatives=derivatives,\
                                    x=x_loc)
        else:
            raise Exception('Function must be explicit, nodal, or cellwise.')
        
        return f_vec
            
        
    def f_eval_loc(self, f, cell, edge_loc=None, derivatives=(0,), x=None):
        """
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
        n_dofs = self.__element.n_dofs() 
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
                
          
    def form_eval(self, form, node, edge_loc=None):
        """
        Evaluates the local kernel, test, (and trial) functions of a (bi)linear
        form on a given entity.
        
        Inputs:
        
            form: (bi)linear form as tuple (f,'trial_type','test_type'), where
                
                f: function, constant, or vector of nodes
                
                trial_type: str, 'u','ux',or 'uy'
                
                test_type: str, 'v', 'vx', 'vy'    
                
            node: Node, tree node linked to physical cell  
            
            edge_loc: str, location of edge        
        
        Outputs:
        
            (Bi)linear form
                            
        """
        assert node.is_linked(), 'Tree node must be linked to cell.'
        cell = node.quadcell()
        #
        # Quadrature weights
        # 
        if edge_loc is not None:
            edge = cell.get_edges(edge_loc)
            weight = self.__rule_1d.jacobian(edge)*self.__rule_1d.weights()
        else:
            weight = self.__rule_2d.jacobian(cell)*self.__rule_2d.weights()
                   
        #
        # kernel
        # 
        f = form[0]
        if type(f) is tuple:
            #
            # Kernel already specified: f = (kernel,)
            # 
            kernel = f[0]
            kernel_size = len(kernel)
            assert kernel_size==self.__n_gauss_1d or\
                kernel_size==self.__n_gauss_2d, \
                'Kernel size not compatible with quadrature rule.'
        else:
            kernel = self.f_eval_loc(f,cell=cell, edge_loc=edge_loc)
        
        if len(form) > 1:
            #
            # test function               
            # 
            drv = self.parse_derivative_info(form[1])
            test = self.shape_eval(derivatives=drv, cell=cell, \
                                   edge_loc=edge_loc)
            if len(form) > 2:
                #
                # trial function
                # 
                drv = self.parse_derivative_info(form[2])
                trial = test.copy()
                test = self.shape_eval(derivatives=drv, cell=cell,\
                                        edge_loc=edge_loc)
                if len(form) > 3:
                    raise Exception('Only Linear and Bilinear forms supported.')
                else:
                    return self.bilinear_loc(weight, kernel, trial, test) 
            else:
                return self.linear_loc(weight,kernel,test)
        else:
            return np.sum(kernel*weight)   
        
            
    def cell_rule(self):
        """
        Return GaussRule for cell
        """
        
        return self.__rule_2d
    
    
    def edge_rule(self):
        """
        Return GaussRule for edge
        """
        return self.__rule_1d
    
    
    def make_generic(self,entity):
        """
        Turn a specific entity (QuadCell or Edge) into a generic one
        e.g. Quadcell --> 'cell'
             (Edge, direction) --> ('edge',direction)
        """ 
        if isinstance(entity, QuadCell):
            return 'cell'
        elif len(entity) == 2 and isinstance(entity[0], Edge):
            return ('edge', entity[1])
        else:
            raise Exception('Entity not supported.')
        
    def parse_derivative_info(self, dstring):
        """
        Input:
        
            dstring: string of the form u,ux,uy,v,vx, or vy.
            
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
        n_coarse =  self.__dofhandler.n_dofs(marker_coarse)
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
        for node in self.__mesh.root_node().find_leaves(marker_fine):
            if node.has_parent(marker_coarse):
                parent = node.get_parent(marker_coarse)
                node_dofs = self.__dofhandler.get_global_dofs(node)
                parent_dofs = self.__dofhandler.get_global_dofs(parent)
                x = self.__dofhandler.dof_vertices(node)
                phi = self.shape_eval(cell=parent.quadcell(), x=x)
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