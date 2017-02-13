import numpy as np
from scipy import sparse 
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
                
            elif dim == 2:
                dofs_per_vertex = 1
                dofs_per_edge = 0
                dofs_per_cell = 0
                basis_index = [(0,0),(1,0),(0,1),(1,1)]
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
                
            elif dim == 2:
                dofs_per_vertex = 1 
                dofs_per_edge = 1
                dofs_per_cell = 1
                basis_index = [(0,0),(1,0),(0,1),(1,1),
                               (0,2),(1,2),(2,0),(2,1),(2,2)]
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
                
            elif dim == 2:
                dofs_per_vertex = 1 
                dofs_per_edge = 2
                dofs_per_cell = 4
                basis_index = [(0,0),(1,0),(0,1),(1,1),
                               (0,2),(0,3),(1,2),(1,3),(2,0),(3,0),(2,1),(3,1),
                               (2,2),(3,2),(2,3),(3,3)]
        self.__cell_type = 'quadrilateral' 
        self.__dofs = {'vertex':dofs_per_vertex, 'edge':dofs_per_edge,'cell':dofs_per_cell}               
        self.__basis_index = basis_index
        self.__p = p
        self.__px = px
        self.__element_type = element_type
    
      
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
    
    
    def ref_vertices(self):
        """
        Returns vertices used to define nodal basis functions on reference cell
        """
        p = np.array(self.__basis_index)
        if list(self.__element_type)[0] == 'Q':
            n_dofs_per_dim = self.n_dofs('edge')+2
            x = np.linspace(0.0,1.0,n_dofs_per_dim)
            return x[p] 
        else:
            raise Exception('Only Q type elements currently supported.')
     
        
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
    """
    def __init__(self, mesh, element):
        """
        Constructor
        """
        etype = element.element_type()
        if etype == 'Q1':
            dofs_per_vertex = 1
            dofs_per_edge = 0
            dofs_per_cell = 0
            n_dofs = 4
            pattern = ['SW','SE','NW','NE']
        elif etype == 'Q2':
            dofs_per_vertex = 1
            dofs_per_edge = 1
            dofs_per_cell = 1
            n_dofs = 9
            pattern = ['SW','SE','NW','NE',
                       'W','E','S','N','I']
        elif etype == 'Q3':
            dofs_per_vertex = 1
            dofs_per_edge = 2
            dofs_per_cell = 4
            n_dofs = 16
            pattern = ['SW','SE','NW','NE',
                       'W','W','E','E','S','S','N','N',
                       'I','I','I','I']
        else:
            raise Exception('Only Q1, Q2, or Q3 supported.')
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
    
    
    def n_nodes(self):
        """
        Return the total number of nodes
        """
        print(self.n_global_dofs)
        self.__getattribute__('n_global_dofs')
        if hasattr(self, 'n_global_dofs'):
            return self.n_global_dofs
        else:
            raise Exception('Dofs have not been distributed yet.')
            
            
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
                x0,y0,x1,y1 = cell.box()
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


    def jacobian(self, cell):
        """
        Jacobian of the Mapping from reference to physical cell
        """
        dim = self.__dim
        cell_type = self.__cell_type
        if dim == 1:
            #
            # One dimensional mesh
            # 
            if cell_type == 'interval':
                x0, x1 = cell.box()
                jac = x1-x0
            elif cell_type == 'edge':
                # Length of edge
                jac = cell.length()
                
        elif dim == 2:
            #
            # Two dimensional mesh
            #
            if cell_type == 'triangle':
                #
                # Triangles not yet supported
                # 
                pass
            elif cell_type == 'quadrilateral':
                x0,x1,y0,y1 = cell.box()
                jac = (x1-x0)*(y1-y0)
        return jac
    
    
    
class System(object):
    """
    (Non)linear system to be defined and solved 
    """
    def __init__(self, mesh, element, n_gauss=(3,9), 
                 bnd_markers=None, bnd_functions=None):
        """
        Set up linear system
        
        Inputs:
        
            mesh: Mesh, finite element mesh
            
            element: FiniteElement, shapefunctions
            
            n_gauss: int tuple, number of quadrature nodes in 1d and 2d respectively
                        
            bnd_markers: dictionary of boolean functions for marking boundaries 
                {'dirichlet':m_d,'neumann':m_n,'robin':m_r, 'periodic':m_p},
                where m_i maps a node/edge? to a boolean 
                
            bnd_functions: dictionary of functions corresponding to the 
                boundary conditions, i.e.
                {'dirichlet':g_d,'neumann':g_n,'robin':g_r, 'periodic':g_p},
        """
        self.__mesh = mesh
        self.__element = element
        self.__n_gauss_2d = n_gauss[1]
        self.__n_gauss_1d = n_gauss[0]
        self.__bnd_markers = bnd_markers
        self.__bnd_functions = bnd_functions
          
    
    def assemble(self, bilinear_forms=None, linear_forms=None, 
                 bnd_conditions=False, separate_forms=False):
        """
        
        Inputs: 
        
            bilinear_forms: (q*u,v), where u,v can denote phi,phi_x, or phi_y 
            
            linear_forms: (f,v)
            
            bnd_conditions: bool, True if boundary conditions should be applied 
            
            separate_forms: bool, False if (bi)linear forms should be added to
                form a linear system.
            
        Outputs:
        
            b
            
        """
        # ---------------------------------------------------------------------
        # Set Element Spaces, Mesh, and DofHandler
        # ---------------------------------------------------------------------
        element_2d = self.__element
        print(element_2d.element_type())
        element_1d = QuadFE(1,element_2d.element_type())
        mesh = self.__mesh
        dof_handler = DofHandler(mesh,element_2d)
        dof_handler.distribute_dofs()
        n_nodes = dof_handler.n_nodes()
        
        # ---------------------------------------------------------------------
        # Define Quadrature Rule
        # ---------------------------------------------------------------------
        # One dimensional rule for edges 
        rule_1d = GaussRule(self.__n_gauss_1d,shape='edge')
        r_ref_1d = rule_1d.nodes()
        w_ref_1d = rule_1d.weights()
        
        # Two dimensional rule for cells
        rule_2d = GaussRule(self.__n_gauss_2d,shape='quadrilateral')
        r_ref_2d = rule_2d.nodes()
        w_ref_2d = rule_2d.weights()
        
        # ---------------------------------------------------------------------
        # Evaluate shape functions at Gauss nodes on reference element
        # ---------------------------------------------------------------------
        # 1D
        n_dofs_1d = element_1d.n_dofs()
        phi_ref_1d = [np.empty((self.__n_gauss_1d,n_dofs_1d))]*2 

        for i in range(n_dofs_1d):
            phi_ref_1d[0][:,i] = element_1d.phi(i,r_ref_1d)
            phi_ref_1d[1][:,i] = element_1d.dphi(i,r_ref_1d)
     
        # 2D
        n_dofs_2d = element_2d.n_dofs()  
        print('n_dofs per cell %i'%(n_dofs_2d))
        print('n_dofs per edge %i'%(n_dofs_1d))    
        phi_ref_2d = [np.empty((self.__n_gauss_2d,n_dofs_2d)), 
                      [np.empty((self.__n_gauss_2d,n_dofs_2d)),
                       np.empty((self.__n_gauss_2d,n_dofs_2d))]]
        
        for i in range(n_dofs_2d):
            phi_ref_2d[0][:,i] = element_2d.phi(i,r_ref_2d)
            phi_ref_2d[1][0][:,i] = element_2d.dphi(i,r_ref_2d,0)
            phi_ref_2d[1][1][:,i] = element_2d.dphi(i,r_ref_2d,1)
         
        #
        # Determine the forms to assemble
        #
        if bilinear_forms is not None:
            if type(bilinear_forms) is list:
                bivals = [[] for i in range(len(bilinear_forms))]
            else:
                bilinear_error_msg = 'bilinear_form should be a 3-tuple.'
                assert (type(bilinear_forms) is tuple and \
                        len(bilinear_forms)==3), bilinear_error_msg   
                bivals = [[]]
        print(bivals)
        
        if linear_forms is not None:
            if type(linear_forms) is list:
                linvecs = [np.empty((n_nodes,)) for i in range(len(linear_forms))]
            else:
                linear_error_msg = 'linear_form should be a 2-tuple.'
                assert(type(linear_forms) is tuple and \
                       len(linear_forms)==2), linear_error_msg 
                linvecs = [np.empty((n_nodes,))]
        

        
        rows = []
        cols = []
        for node in mesh.root_node().find_leaves():
            node_dofs = dof_handler.get_cell_dofs(node)
            cell = node.quadcell()
            
            #
            # Map quadrature info
            # 
            r_phys_2d = rule_2d.map(cell, r_ref_2d)
            w_phys_2d = w_ref_2d*rule_2d.jacobian(cell)
            
            #
            # Assemble local system matrices/vectors
            # 
            if bilinear_forms is not None:
                bf_loc = []
                for bf in bilinear_forms:
                    kernel, trial, test = \
                        self.local_eval(bf, phi_ref_2d, r_phys_2d)
                    bf_loc.append(self.bilinear_loc(w_phys_2d, kernel, \
                                                    trial, test)) 
            if linear_forms is not None:
                lf_loc = []
                for lf in linear_forms:
                    kernel, test = self.local_eval(lf, phi_ref_2d, r_phys_2d)
                    lf_loc.append(self.linear_loc(w_phys_2d, kernel, test))
          
            #
            # Boundary conditions
            # 
            for direction in ['W','E','S','N']:
                edge = cell.get_edges(direction)
                for key in ['dirichlet','neumann','robin','periodic']:
                    if self.__bnd_markers[key](edge):
                        pass
            #
            # Local to global mapping
            #
            for i in range(n_dofs_2d):

                #
                # Test Dofs
                #
                # Update linear forms
                # 
                for k in range(len(linvecs)):
                    linvecs[k][node_dofs[i]] = lf_loc[k][i]
                     
                for j in range(n_dofs_2d):
                    #
                    # Trial Dofs
                    #
                    rows.append(node_dofs[i]) 
                    cols.append(node_dofs[j])
                    #
                    # Update bilinear forms
                    # 
                    for k in range(len(bivals)):
                        bivals[k].append(bf_loc[k][i,j])
                        
        #            
        # Save results as sparse matrices 
        #
        bimats = []
        for bv in bivals:
            bimats.append(sparse.coo_matrix((bv,(rows,cols)))) 
        return bimats, linvecs
           

      
    
    def bilinear_loc(self,weight,kernel,trial,test):
        """
        Compute the local bilinear form over an element
        """
        return np.dot(test.T, np.dot(np.diag(weight*kernel),test))
    
    
    def linear_loc(self,weight,kernel,test):
        """
        Compute the local linear form over an element
        """
        return np.dot(test.T, weight*kernel)
    
        
    def local_eval(self, form, phi, x):
        """
        Evaluates the local kernel, test, and trial functions of a (bi)linear
        form on a given entity.
        
        Inputs:
        
            form: (bi)linear form as tuple (f,'trial_type','test_type'), where
                
                f: function, or constant
                
                trial_type: str, 'u','ux',or 'uy'
                
                test_type: str, 'v', 'vx', 'vy'
                
            phi: shape functions evaluated at the Gauss points on the reference
                element.    
                
            entity: local cell or edge
            
            x: Gauss points on pysical entity
            
        
        Outputs:
        
            kernel: function evaluated at the local quadrature nodes (n_quad,) 
            
            trial: element trial functionals, evaluated at quad nodes (n_quad, n_dof)
            
            test: element test functionals, evaluated at quadrature nodes (n_quad, n_dof)
                            
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
                kernel = f(x)
            else:
                # f is a constant (TODO: change this)
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
        
    def check_forms(self):
        """
        Make sure the (bi)linear forms are correctly formatted
        """ 
        