import numpy as np
from scipy import sparse 
import numbers
from mesh import QuadCell, Edge
 
"""
To do with Finite Element Classes
"""

class FiniteElement(object):
    """
    Parent Class: Finite Elements
    """
    def __init__(self, dim, element_type):   
        self.__element_type = element_type
        self.__dim = dim    
    
    def dim(self):
        """
        Returns the spatial dimension
        """
        return self.__dim
     
    
class QuadFE(FiniteElement):
    """
    Continuous Galerkin finite elements on quadrilateral cells 
    """
    def __init__(self, dim, element_type):
        FiniteElement.__init__(self, dim, element_type)
        
        #
        # Linear Elements
        #
        if element_type == 'Q1':
            
            p  = [lambda x: 1-x, lambda x: x]
            px = [lambda x:-1.0, lambda x: 1.0]
            
            if dim == 1:
                dofs_per_vertex = 1
                dofs_per_edge = 0
                dofs_per_cell = 0
                basis_index  = [0,1]
                ref_nodes = [0.0,1.0]
            elif dim == 2:
                dofs_per_vertex = 1
                dofs_per_edge = 0
                dofs_per_cell = 0
                basis_index = [(0,0),(1,0),(0,1),(1,1)]
                ref_nodes = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]])
                pattern = ['SW','SE','NW','NE']
        #
        # Quadratic Elements 
        #        
        elif element_type == 'Q2':
            
            p =  [ lambda x: 2*x*x-3*x + 1.0, 
                   lambda x: 2*x*x-x, 
                   lambda x: 4.0*x-4*x*x ]
                
            px = [ lambda x: 4*x -3, 
                   lambda x: 4*x-1,
                   lambda x: 4.0-8*x ]
            
            if dim == 1: 
                dofs_per_vertex = 1
                dofs_per_edge = 0
                dofs_per_cell = 1
                basis_index = [0,1,2]
                ref_nodes = np.array([0.0,1.0,0.5])
            elif dim == 2:
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
        
        #
        # Cubic Elements
        #     
        elif element_type == 'Q3':
            
            p = [lambda x: -4.5*(x-1./3.)*(x-2./3.)*(x-1.),
                 lambda x:  4.5*x*(x-1./3.)*(x-2./3.),
                 lambda x: 13.5*x*(x-2./3.)*(x-1.),
                 lambda x: -13.5*x*(x-1./3.)*(x-1.) ]
                
            px = [lambda x: -13.5*x*x + 18*x - 5.5,
                  lambda x: 13.5*x*x - 9*x + 1.0,
                  lambda x: 40.5*x*x - 45*x + 9.0,
                  lambda x: -40.5*x*x + 36*x -4.5]
            
            if dim == 1:
                dofs_per_vertex = 1
                dofs_per_edge = 0
                dofs_per_cell = 2
                basis_index = [0,1,2,3]
                ref_nodes = np.array([0.0,1.0,1/3.0,2/3.0])
            elif dim == 2:
                dofs_per_vertex = 1 
                dofs_per_edge = 2
                dofs_per_cell = 4
                basis_index = [(0,0),(1,0),(0,1),(1,1),
                               (0,2),(0,3),(1,2),(1,3),(2,0),(3,0),(2,1),(3,1),
                               (2,2),(3,2),(2,3),(3,3)]
                ref_nodes = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0],
                                  [0.0,1./3.],[0.0,2./3.],
                                  [1.0,1./3.],[1.0,2./3.], 
                                  [1./3.,0.0],[2./3.,0.0], 
                                  [1./3.,1.0],[2./3.,1.0],
                                  [1./3.,1./3.],[2./3.,1./3.],
                                  [1./3.,2./3.],[2./3.,2./3.]])
                pattern = ['SW','SE','NW','NE','W','W','E','E','S','S','N','N',
                           'I','I','I','I']
        self.__cell_type = 'quadrilateral' 
        self.__dofs = {'vertex':dofs_per_vertex, 'edge':dofs_per_edge,'cell':dofs_per_cell}               
        self.__basis_index = basis_index
        self.__p = p
        self.__px = px
        self.__element_type = element_type
        self.__ref_nodes = ref_nodes
        if dim == 2:
            self.__pattern = pattern
      
    def cell_type(self):
        return self.__cell_type
    
    
    def polynomial_degree(self):
        """
        Return the finite element's polynomial degree 
        """
        return list(self.__element_type)[1]
    
    def element_type(self):
        """
        Return the finite element type (Q1, Q2, or Q3)
        """ 
        return self.__element_type
    
        
    def n_dofs(self,key=None):
        """
        Return the number of dofs per elementary entity
        """
        # Total Number of dofs
        if key == None:
            d = self.dim()
            return 2**d*self.__dofs['vertex'] + \
                   2*d*self.__dofs['edge'] + \
                   self.__dofs['cell']
        else:
            assert key in self.__dofs.keys(), 'Use "vertex","edge", "cell" for key'
            return self.__dofs[key]
    
    
    def reference_nodes(self):
        """
        Returns vertices used to define nodal basis functions on reference cell
        """
        return self.__ref_nodes
        
        
    def get_local_edge_dofs(self,direction):
        """
        Returns the local dofs on a given edge
        """    
        edge_dofs = []
        for i in range(self.n_dofs('cell')):
            if direction in self.__pattern[i]:
                edge_dofs.append(i)
        return edge_dofs
     
        
    def phi(self, n, x):
        """
        Evaluate the nth basis function at the point x
        
        Inputs: 
        
            n: int, basis function number
            
            x: double, point at which function is to be evaluated
               (double if dim=1, or tuple if dim=2) 
        """
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
            
            var: int, variable w.r.t. which we differentiate
            
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
   
   
    def constraint_coefficients(self):
        """
        Returns the constraint coefficients of a typical bisected edge. 
        Vertices on the coarse edge are numbered in increasing order, e.g. 0,1,2,3 for Q2,
        
        Output:
        
            constraint: double, dictionary whose keys are the fine node numbers  
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
    TODO: A lot of things can be handled more appropriately by the element
    """
    def __init__(self, mesh, element):
        """
        Constructor
        """
        etype = element.element_type()
        if etype == 'Q1':
            """
            2---3
            |   |
            0---1
            """
            dofs_per_vertex = 1
            dofs_per_edge = 0
            dofs_per_cell = 0
            n_dofs = 4
            pattern = ['SW','SE','NW','NE']
            ref_nodes = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]])
        elif etype == 'Q2':
            """
            2---7---3
            |       |
            4   8   5 
            |       |
            0---6---1
            """
            dofs_per_vertex = 1
            dofs_per_edge = 1
            dofs_per_cell = 1
            n_dofs = 9
            pattern = ['SW','SE','NW','NE',
                       'W','E','S','N','I']
            ref_nodes = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0],
                                  [0.0,0.5],[1.0,0.5],[0.5,0.0],[0.5,1.0],
                                  [0.5,0.5]])
        elif etype == 'Q3':
            """
            2---10---11---3
            |             |
            5   14   15   7
            |             |
            4   12   13   6
            |             |
            0----8---9----1 
            """
            dofs_per_vertex = 1
            dofs_per_edge = 2
            dofs_per_cell = 4
            n_dofs = 16
            pattern = ['SW','SE','NW','NE',
                       'W','W','E','E','S','S','N','N',
                       'I','I','I','I']
            ref_nodes = np.array([[0.0,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0],
                                  [0.0,1./3.],[0.0,2./3.],
                                  [1.0,1./3.],[1.0,2./3.], 
                                  [1./3.,0.0],[2./3.,0.0], 
                                  [1./3.,1.0],[2./3.,1.0],
                                  [1./3.,1./3.],[2./3.,1./3.],
                                  [1./3.,2./3.],[2./3.,2./3.]])
        else:
            raise Exception('Only Q1, Q2, or Q3 supported.')
        self.__element = element
        self.__element_type = etype
        self.__dim = element.dim()
        self.dofs_per_vertex = dofs_per_vertex 
        self.dofs_per_edge = dofs_per_edge
        self.dofs_per_cell = dofs_per_cell 
        self.__n_dofs = n_dofs
        self.__pattern = pattern
        self.__global_dofs = dict.fromkeys(mesh.root_node().find_leaves(),[None]*n_dofs) 
        self.__root_node = mesh.root_node()
        self.__hanging_nodes = []  
        self.__constraint_coefficients = element.constraint_coefficients()
        self.__reference_nodes = ref_nodes
        
    def n_dofs(self,key='cell'):
        """
        Return the number of dof's of cell
        
        Inputs: 
        
            key: str, specifying entity ['cell'],'edge', or 'vertex' 
        """
        if key == 'cell':
            return self.__n_dofs
        elif key == 'edge':
            return self.dofs_per_edge
        elif key == 'vertex':
            return 1
            
    def distribute_dofs(self):
        """
        global enumeration of degrees of freedom
        """
        count = 0  # possibly change this with nested meshes...
        opposite = {'N':'S', 'S':'N', 'W':'E', 'E':'W', 
                    'SW':'NE','NE':'SW','SE':'NW','NW':'SE'}

        for node in self.__root_node.find_leaves():
            # Initialize dofs list for node
            dof_list = self.__global_dofs[node][:] 
            
            # ========================
            # Fill in own nodes
            # ========================
            dof_list, count = self.fill_in_dofs(dof_list,count)
            self.__global_dofs[node] = dof_list

            # =========================
            # Share dofs with neighbors
            # =========================
            #
            # Diagonal directions
            #
            for diag_dir in ['SW','SE','NW','NE']:
                nb = node.find_neighbor(diag_dir)
                if nb != None:
                    dof = self.pos_to_dof(dof_list,diag_dir)
                    opp_dir = opposite[diag_dir] 
                    if nb.has_children(opp_dir):
                        nb = nb.children[opp_dir]
                    self.assign_dofs(nb,opp_dir,dof)    
            
            #
            # W, E, S, N
            # 
            sub_pos = {'E':['SE','NE'], 'W':['SW','NW'], 
                       'N':['NW','NE'], 'S':['SW','SE']}
            dpe = self.dofs_per_edge
            ref_index = range(0,dpe+2) 
            coarse_index = [2*r for r in ref_index]
            for direction in ['W','E','S','N']:
                opp_dir = opposite[direction]
                n_pos = self.positions_along_edge(direction)
                dofs = self.pos_to_dof(dof_list, n_pos)
                nb = node.find_neighbor(direction)
                if nb != None:
                    if nb.has_children():
                        #
                        # Neighboring cell has children
                        # 
                        ch_count = 0
                        for sp in sub_pos[opp_dir]:
                            child = nb.children[sp]
                            if child != None:
                                ch_pos = self.positions_along_edge(opp_dir)
                                fine_index = [r+(dpe+1)*ch_count for r in ref_index]
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
                        nb_pos = self.positions_along_edge(opp_dir)
                        self.assign_dofs(nb, nb_pos, dofs)
                    elif nb.depth < node.depth:
                        #
                        # Neighbor larger than self
                        # 
                        nb_pos = self.positions_along_edge(opp_dir)
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
        self.n_global_dofs = count
    
    
    def reference_nodes(self, key='cell'):
        """
        Return the nodes on the reference element
        
        Inputs: 
        
            key ['cell']: str, 'edge' 
        """
        if key == 'cell':
            return self.__reference_nodes
        elif key == 'edge':
            n_nodes = self.dofs_per_edge + 2*self.dofs_per_vertex
            return np.linspace(0.0, 1.0, n_nodes) 
        
    
    def n_nodes(self):
        """
        Return the total number of nodes
        """
        self.__getattribute__('n_global_dofs')
        if hasattr(self, 'n_global_dofs'):
            return self.n_global_dofs
        else:
            raise Exception('Dofs have not been distributed yet.')
            
     
    def mesh_nodes(self):
        """
        Return the mesh nodes
        """
        x = np.empty((self.n_global_dofs,2))
        rule = GaussRule(1,shape='quadrilateral')
        x_ref = self.reference_nodes()
        for leaf in self.__root_node.find_leaves():
            g_dofs = self.get_cell_dofs(leaf)
            x[g_dofs,:] = rule.map(leaf.quadcell(),x=x_ref)
        return x
                
    def fill_in_dofs(self,node_dofs, count):
        """
        Fill in node's dofs 
        """
        for i in range(self.__n_dofs):
            if node_dofs[i] == None:
                node_dofs[i] = count
                count += 1
        return node_dofs, count
    
    
    def positions_along_edge(self, direction):
        """
        Returns the positions of dofs along each edge in order
        from left-to-right and low-to-high.
        """
        assert direction in ['N','S','E','W'], 'Direction not supported.'
        positions = []
        count = 0
        for pos in self.__pattern:
            if pos == direction:
                positions.append((pos,count))
                count += 1
            elif direction in pos:
                positions.append(pos)
            
        min_pos = 'S'+direction if direction in ['E','W'] else direction+'W'
        max_pos = 'N'+direction if direction in ['E','W'] else direction+'E'
  
        dpe = self.dofs_per_edge
        ordering = [min_pos] + [(direction,i) for i in range(dpe)] + [max_pos]       
        return ordering 
        
            
    def assign_dofs(self, node, positions, dofs):
        """
        Assign dofs to node
        """    
        # Initialize positions
        p = self.__pattern
        dof_list = self.__global_dofs[node][:]
        
        # Turn positions and dofs into list
        if not(type(positions) is list):
            positions = [positions]
        if not(type(dofs) is list):
            dofs = [dofs]
        lengths_do_not_match = 'Number of dofs and positions do not match.'
        assert len(positions)==len(dofs),lengths_do_not_match
        for pos,dof in zip(positions,dofs):
            if type(pos) is tuple:
                direction, offset = pos
                direction_error ='Only "W,E,S,N,I" admit multiple entries.'
                assert direction in ['W','E','S','N','I'], direction_error
                index = p.index(direction) + offset
                if dof_list[index] != None:
                    incompatible_dofs = 'Incompatible dofs. Something fishy.'
                    assert dof_list[index] == dof, incompatible_dofs
                else:
                    dof_list[index] = dof
            else:
                position_error = 'Position %s not recognized.'%(pos)
                assert pos in p, position_error
                index = p.index(pos)
                if dof_list[index] != None:
                    incompatible_dofs = 'Incompatible dofs. Something fishy.'
                    assert dof_list[index] == dof, incompatible_dofs
                else:
                    dof_list[index] = dof
        self.__global_dofs[node] = dof_list
        
        
    def pos_to_dof(self, dof_list, positions):
        """
        Return a list of dofs corresponding to various positions 
        """
        dofs = []
        p = self.__pattern

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
    
    def get_cell_dofs(self, node):
        """
        Return all dofs corresponding to a given tree node 
        """
        return self.__global_dofs[node]
    
    
    def get_local_edge_dofs(self, direction):
        """
        Return all dofs on a given edge of a cell 
        """
        return self.__element.get_local_edge_dofs(direction)
        
        
    
    def get_global_edge_dofs(self, node, direction):
        """
        Return all global dofs of a given edge of a cell
        """
        cell_dofs = self.__global_dofs[node]
        edge_dofs = []
        for i in range(self.__n_dofs):
            if direction in self.__pattern[i]:
                edge_dofs.append(cell_dofs[i])
        return edge_dofs
    
                
    def make_hanging_node_constraints(self):
        """
        Return the constraint matrix satisfied by the mesh's hanging nodes.
        """
        rows = []
        cols = []
        vals = []
        sub_pos = {'E':['SE','NE'], 'W':['SW','NW'], 
                   'N':['NW','NE'], 'S':['SW','SE']}
        opposite = {'E':'W','W':'E','N':'S','S':'N'}
        n_rows = 0
        cc = self.__constraint_coefficients
        print('Constraint coefficients: {0}'.format(cc))
        n_verts = self.dofs_per_edge + 2
        for node, n_doflist in self.__global_dofs.items():
            for direction in ['W','E','S','N']:
                n_dof_pos = self.positions_along_edge(direction)
                nb = node.find_neighbor(direction)
                if nb != None and nb.has_children():
                    print('Node: {0} -> Neighbor: {1}'.format(node.address,nb.address))
                    opp = opposite[direction]
                    ignore_center = False
                    for i in range(2):
                        print('Child %s'%(sub_pos[opp][i]))
                        child = nb.children[sub_pos[opp][i]]
                        if child != None:
                            print(child.address)
                            ch_dof_pos = self.positions_along_edge(opp)
                            print('Child positions along edge {0}'.format(ch_dof_pos))
                            ch_doflist = self.__global_dofs[child]
                            print('Child doflist: {0}'.format(ch_doflist))
                            for hn in cc[i].keys():
                                coarse_dofs = self.pos_to_dof(n_doflist, n_dof_pos)
                                if not ignore_center:
                                    cols += coarse_dofs
                                    vals += cc[i][hn]
                                    hn_dofs = self.pos_to_dof(ch_doflist, 
                                                            ch_dof_pos[hn])
                                    cols += hn_dofs
                                    print('Hanging Node dofs: {0}'.format(hn_dofs))
                                    print('Coarse dofs: {0}'.format(coarse_dofs))
                                
                                    vals += [-1.0]
                                    rows += [n_rows]*(n_verts+1) 
                                    n_rows += 1
                            ignore_center = True
                        else:
                            print('Child is None')
        n_cols = self.__n_global_dofs
        print('%d Rows'%(n_rows))
        return -sparse.coo_matrix((vals,(rows,cols)),shape=(n_rows+1,n_cols))

  
        
        
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
                2D rule: order in {1,4,16,25} for quadrilaterals
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
                assert order in [1,4,9,16,25], 'Gauss rules over quads in 2D: 1,4,16,25'
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
       
        
    def map(self, cell, x=None):
        """
        Map from reference to physical cell
        
        Inputs:
        
            cell: QuadCell, used for its box coordinates
            
            x: double, a length n list of dim-tuples or an (n,dim) array  
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
                x0, x1 = cell.box()
                x_phys = x0 + (x1-x0)*x_ref
            elif cell_type == 'edge':
                # 
                # Line segment in 2D
                # 
                x0,x1,y0,y1 = cell.box()
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
                x0,x1,y0,y1 = cell.box()
                x_phys = np.array([x0 + (x1-x0)*x_ref[:,0], 
                                   y0 + (y1-y0)*x_ref[:,1]]).T
        return x_phys


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
    
    
    
class System(object):
    """
    (Non)linear system to be defined and solved 
    """
    def __init__(self, mesh, element, n_gauss=(3,9)):
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
        self.__dofhandler.distribute_dofs()
        # Initialize refernce shape functions
        self.__phi = {'cell':       {(0,): None, (1,0): None, (1,1): None},
                      ('edge','W'): {(0,): None, (1,0): None, (1,1): None},
                      ('edge','E'): {(0,): None, (1,0): None, (1,1): None},
                      ('edge','S'): {(0,): None, (1,0): None, (1,1): None},
                      ('edge','N'): {(0,): None, (1,0): None, (1,1): None}}  
    
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
                eddge [the case for neumann and robin conditions] or from nodal
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
              
        """        
        n_nodes = self.__dofhandler.n_nodes()
        n_dofs = self.__element.n_dofs()   
  
        #
        # Determine the forms to assemble
        #
        if bilinear_forms is not None:
            bivals = []
        
        if linear_forms is not None:
            linvec = np.zeros((n_nodes,))
 
        if boundary_conditions is not None:
            #
            # Unpack boundary data
            # 
            bc_dirichlet = boundary_conditions['dirichlet']
            bc_neumann = boundary_conditions['neumann']
            bc_robin = boundary_conditions['robin']
    
        rows = []
        cols = []
        dir_nodes_encountered = []
        for node in self.__mesh.root_node().find_leaves():
            node_dofs = self.__dofhandler.get_cell_dofs(node)
            cell = node.quadcell()            
            #
            # Assemble local system matrices/vectors
            # 
            if bilinear_forms is not None:
                bf_loc = np.zeros((n_dofs,n_dofs))
                for bf in bilinear_forms:
                    bf_loc += self.form_eval(bf, cell)
                    
            if linear_forms is not None:
                lf_loc = np.zeros((n_dofs,))
                for lf in linear_forms:
                    lf_loc += self.form_eval(lf,cell)
                    
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
                    for bc_neu in bc_neumann:
                        m_neu,g_neu = bc_neu 
                        if m_neu(edge):
                            print('Neumann Edge')
                            # -------------------------------------------------
                            # Neumann edge
                            # -------------------------------------------------
                            neumann_edge = True
                            #
                            # Update local linear form
                            #
                            print(self.form_eval((g_neu,'v'), (edge,direction)))
                            lf_loc += self.form_eval((g_neu,'v'), \
                                                     (edge,direction))
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
                                bf = (1,'u','v')
                                bf_loc += \
                                    gamma_rob*self.form_eval((1,'u','v'),\
                                                             (edge,direction))
                                #
                                # Update local linear form
                                # 
                                lf_loc += \
                                    gamma_rob*self.form_eval((g_rob,'v'),\
                                                             (edge,direction))
                                break                           
                #
                #  Check for Dirichlet Nodes
                #
                x_ref = self.__element.reference_nodes()
                x_cell = self.__rule_2d.map(cell,x=x_ref) 
                cell_dofs = np.arange(n_dofs)
                if bc_dirichlet is not None:
                    for bc_dir in bc_dirichlet:
                        m_dir,g_dir = bc_dir
                        is_dirichlet = m_dir(x_cell[:,0],x_cell[:,1])
                        if is_dirichlet.any():
                            dir_nodes_loc = x_cell[is_dirichlet,:]
                            dir_dofs_loc = cell_dofs[is_dirichlet] 
                            for j,x_dir in zip(dir_dofs_loc,dir_nodes_loc):
                                #
                                # Modify jth row 
                                #
                                notj = np.arange(n_dofs)!=j
                                uj = g_dir(x_dir[0],x_dir[1])
                                if node_dofs[j] not in dir_nodes_encountered: 
                                    bf_loc[j,j] = 1.0
                                    bf_loc[j,notj]=0.0
                                    lf_loc[j] = uj
                                    dir_nodes_encountered.append(node_dofs[j])
                                else:
                                    bf_loc[j,:] = 0.0  # make entire row 0
                                    lf_loc[j] = 0.0
                                #
                                # Modify jth column and right hand side
                                #
                                lf_loc[notj] -= bf_loc[notj,j]*uj 
                                bf_loc[notj,j] = 0.0            
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
    
    
    def shape_eval(self,derivatives=(0,),entity='cell',x=None):
        """
        Evaluate shape functions at a set of reference points x. If x is not
        specified, Gauss quadrature points are used. 
        
        Inputs: 
        
            derivatives: tuple specifying the order of the derivative and 
                the variable 
                [(0,)]: function evaluation, 
                (1,0) : 1st derivative wrt first variable, or 
                (1,1) : 1st derivative wrt second variable
        
            entity: type of entity containing the quadrature points (if x is 
                not specified).
                 'cell' : reference cell
                 ('edge',direction): edge of reference cell
            
            x: double, np.array of points in the reference cell
        
        Output:
        
            phi: (n_points,n_dofs) array, the jth column of which is the jth
                shape function evaluated at the specified points. 
        """
        n_dofs = self.__element.n_dofs()
        if x == None:
            #
            # Quadrature points
            #
            if self.__phi[entity][derivatives] is not None:
                return self.__phi[entity][derivatives]
            else: 
                if entity == 'cell':
                    x_ref = self.__rule_2d.nodes()
                    dofs_to_fill = range(n_dofs)
                elif type(entity) is tuple and entity[0] == 'edge':
                    x_ref = self.__rule_1d.nodes(direction=entity[1])
                    dofs_to_fill = self.__element.get_local_edge_dofs(entity[1])
        else:
            # Ensure that x is an array 
            x_ref = np.array(x)
            
        n_points = x_ref.shape[0] 
        phi = np.zeros((n_points,n_dofs))
        if len(derivatives) == 1:
            #
            # No derivatives
            #
            for i in dofs_to_fill:
                phi[:,i] = self.__element.phi(i,x_ref)  
        elif len(derivatives) == 2:
            # 
            # First derivatives
            #
            i_var = derivatives[1]
            for i in dofs_to_fill:
                phi[:,i] = self.__element.dphi(i,x_ref,i_var)        
        if x == None and self.__phi[entity][derivatives] is None:
            #
            # Store shape function (at quadrature points) for future use
            # 
            self.__phi[entity][derivatives] = phi
        return phi
    
    
    def f_eval_loc(self, f, entity, derivatives=(0,), x=None):
        """
        Evaluates a function (or its partial derivatives) at a set of 
        local nodes (quadrature nodes if none are specified).
        
        Inputs:
        
            f: function to be evaluated, either in the form of a 
                
            entity: cell or (edge,direction) on which we evaluate f
            
            derivatives: tuple specifying the order of the derivative and 
                the variable 
                [(0,)]: function evaluation, 
                (1,0) : 1st derivative wrt first variable, or 
                (1,1) : 1st derivative wrt second variable
                
            x: Points (on physical entity) at which we evaluate f.
        
        Output:  
        
            fv: vector of function values, at x points
        """
        n_dofs = self.__element.n_dofs()         
        if type(entity) is tuple:
            assert isinstance(entity[0], Edge), \
            'Entity should be an Edge.'
            ref_entity = ('edge',entity[1])
            if x is None:
                #
                # Use Gauss points
                #
                x = self.__rule_1d.map(entity[0])
        elif isinstance(entity, QuadCell):
            ref_entity = 'cell'
            if x is None:
                #
                # Use Gauss points
                #
                x = self.__rule_2d.map(entity)
        x = np.array(x)  # make sure it's an array
        # 
        # Evaluate the Kernel
        # 
        if callable(f):
            # f is a function
            if len(x.shape) == 1:
                # one dimensional input
                return f(x)
            elif len(x.shape) == 2:
                # two dimensional input
                return f(x[:,0],x[:,1])
        elif isinstance(f,numbers.Real):
            # f is a constant
            return f
        elif len(f) == n_dofs:
            # f is a nodal vector
            phi = self.shape_eval(derivatives=derivatives,\
                                  entity=ref_entity, x=x) 
            return np.dot(phi,f)
                
          
    def form_eval(self, form, entity):
        """
        Evaluates the local kernel, test, (and trial) functions of a (bi)linear
        form on a given entity.
        
        Inputs:
        
            form: (bi)linear form as tuple (f,'trial_type','test_type'), where
                
                f: function, constant, or vector of nodes
                
                trial_type: str, 'u','ux',or 'uy'
                
                test_type: str, 'v', 'vx', 'vy'    
                
            entity: cell or (edge,direction) over which we integrate       
        
        Outputs:
        
            (Bi)linear form
                            
        """
        #
        # Quadrature weights
        # 
        if type(entity) is tuple:
            assert isinstance(entity[0],Edge), \
                'Entity should be an Edge.'
            weight = self.__rule_1d.jacobian(entity[0])*self.__rule_1d.weights()
        elif isinstance(entity, QuadCell):
            weight = self.__rule_2d.jacobian(entity)*self.__rule_2d.weights()       
        #
        # kernel
        # 
        f = form[0]
        kernel = self.f_eval_loc(f,entity)
        
        if len(form) > 1:
            #
            # test function               
            # 
            drv = self.parse_derivative_info(form[1])
            test = self.shape_eval(derivatives=drv, \
                                   entity=self.make_generic(entity))
            if len(form) > 2:
                #
                # trial function
                # 
                drv = self.parse_derivative_info(form[2])
                trial = self.shape_eval(derivatives=drv, \
                                        entity=self.make_generic(entity))
                if len(form) > 3:
                    raise Exception('Only Linear and Bilinear forms supported.')
                else:
                    return self.bilinear_loc(weight, kernel, trial, test) 
            else:
                return self.linear_loc(weight,kernel,test)
        else: 
            return np.sum(kernel*weight)   
        
            
        """
        dim = x.shape[1]
        f = form[0]
        types = list(form[1:])
                   
        if dim == 1:
            #
            # Determine test and trial functions
            #
            tt = []
            for t_type in types:
                if t_type in ['u','v']:
                    tt.append(phi[0])
                elif t_type in ['ux','vx']:
                    tt.append(phi[1])
                else: 
                    raise Exception('Only "[u,v]" and "[u,v]x" allowed.')  
            #
            # Compute kernel
            # 
            if callable(f):
                # f is a function
                kernel = f(x[:,0],x[:,1])
            else:
                # f is a constant 
                kernel = f
                
        elif dim == 2:
            #
            # Determine test and trial functions
            #
            tt = []
            for t_type in types:
                if t_type in ['u','v']:
                    tt.append(phi[0])
                elif t_type in ['ux','vx']:
                    tt.append(phi[1][0])
                elif t_type in ['uy','vy']:
                    tt.append(phi[1][1])
                else:
                    raise Exception('Only "[u,v]" and "[u,v][x,y]" allowed.')
            #
            # Compute kernel
            #
            if callable(f):
                # f is a function
                kernel = f(x[:,0],x[:,1])
            else:
                kernel = f
        if len(form) == 3:
            #
            # Bilinear form         
            # 
            return kernel, tt[0], tt[1] # kernel, trial, test
        elif len(form) == 2:
            #
            # Linear form
            #
            return kernel, tt[0]  # kernel, test
        
        """
        
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
        else:
            raise Exception('Higher order derivatives not supported.')
        