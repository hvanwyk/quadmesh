from mesh import convert_to_array
from function import Nodal
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import numbers
from diagnostics import Verbose    


class NonlinearSystem(object):
    """
    
    """
    def __init__(self, residual, Jacobian):
        """
        Inputs:
        
            residual: function, 
        """
        pass


class LinearSystem(object):
    """
    Linear system object consisting of a single coefficient matrix and possibly
    a right hand side vector, or matrix, together with the associated dof 
    information.
            
        
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
    def __init__(self, basis, A=None, b=None):
        """
        Constructor
        
        
        Inputs:
        
            basis: Basis vector containing dofhandler and subforest_flag info
            
            A: (sparse) square matrix whose size is compatible with number of dofs 
            
            b: vector whose size is compatible with basis (n_dofs)
                        
        """
        #
        # Store basis function
        # 
        self.__basis = basis
        
        #
        # Determine system dofs   
        # 
        dofs = self.get_basis().dofs()
        self.__dofs = dofs
        

        #
        # Form Dof-to-Equation Mapping
        # 
        n_dofs = len(dofs)
        dof2eqn = np.zeros(dofs[-1]+1, dtype=int)
        dof2eqn[dofs] = np.arange(n_dofs, dtype=int)
        
        #
        # Store Dof-to-Equation mapping 
        #
        self.__dof2eqn = dof2eqn
        
        #
        # Determine Hanging nodes 
        # 
        subforest_flag = self.get_basis().subforest_flag()
        self.get_dofhandler().set_hanging_nodes(subforest_flag=subforest_flag)
        
        # 
        # Store Bilinear Form
        # 
        self.set_matrix(A)
        
        #
        # Store Linear Form
        # 
        self.set_rhs(b)
        
        #
        # Initialize constraint matrix
        # 
        self.__C = None
        self.__d = None
              
        
    def get_basis(self):
        """
        Return the basis function 
        """
        return self.__basis
        
        
    def get_dofhandler(self):
        """
        Return the system's dofhandler
        """   
        return self.get_basis().dofhandler()
    
    
    def get_dofs(self):
        """
        Return system dofs
        """
        return self.__dofs

    
    def dof2eqn(self, dofs):
        """
        Convert vector of dofs to equivalent equations
        """
        return self.__dof2eqn[dofs]
    
    
    def get_C(self):
        """
        Return constraint matrix
        """ 
        return self.__C
    
    
    def get_d(self):
        """
        Return constraint affine term
        """
        return self.__d
     
        
    def add_dirichlet_constraint(self, bnd_marker, dirichlet_function=0, 
                                 on_boundary=True):
        """
        Modify an assembled bilinear/linear pair to account for Dirichlet 
        boundary conditions. The system matrix is modified "in place", 
        i.e. 
    
            a11 a12 a13 a14   u1     b1
            a21 a22 a23 a24   u2  =  b2 
            a31 a32 a33 a34   u3     b3
            a41 a42 a43 a44   u4     b4
            
        Suppose Dirichlet conditions u2=g2 and u4=g4 are prescribed. 
        The system is converted to
        
            a11  0  a13  0   u1     b1 - a12*g2 - a14*g4
             0   1   0   0   u2  =  0   
            a31  0  a33  0   u3     b3 - a32*g2 - a34*g4
             0   0   0   1   u4     0 
        
        The solution [u1,u3]^T of this system is then enlarged with the 
        dirichlet boundary values g2 and g4 by invoking 'resolve_constraints' 
        
    
        Inputs:
        
            bnd_marker: str/int flag to identify boundary
            
            dirichlet_function: Function, defining the Dirichlet boundary 
                conditions.
            
            on_boundary: bool, True if function values are prescribed on
                boundary.
            
            
        Notes:
        
        To maintain the dimensions of the matrix, the trial and test function 
        spaces must be the same, i.e. it must be a Galerkin approximation. 
        
        Specifying the Dirichlet conditions this way is necessary if there
        are hanging nodes, since a Dirichlet node may be a supporting node for
        one of the hanging nodes.  
                
                
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
        subforest_flag = self.get_basis().subforest_flag()
        dh = self.get_dofhandler()
        
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
            dirichlet_vals = dirichlet_function.eval(x_dir).ravel()
            
        constraints = dh.constraints
        for dof, val in zip(dirichlet_dofs, dirichlet_vals):
            constraints['constrained_dofs'].append(dof)
            constraints['supporting_dofs'].append([])
            constraints['coefficients'].append([])
            constraints['affine_terms'].append(val)

            
    def set_constraint_relation(self):
        """
        Define the constraint matrix C and affine term d so that 
        
            x = Cx + d,
            
        where the rows in C corresponding to unconstrained dofs are rows of the
        identity matrix.
        """
        dofs = self.get_dofs()
        n_dofs = len(dofs)
        
        #    
        # Define constraint matrix
        #
        constraints = self.get_dofhandler().constraints
        c_dofs = np.array(constraints['constrained_dofs'], dtype=int)
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
    
    
    def set_matrix(self, A):
        """
        Store system matrix
        
        Inputs:
        
            A: sparse matrix
        """
        if A is not None:
            # Check matrix shape
            assert A.shape[0]==A.shape[1], 'Matrix should be square.'
            
            # Check matrix size
            assert A.shape[0]==len(self.get_dofs()), \
            'Matrix size incompatible with Basis.'
        
        # Store matrix
        self.__A = A
        
        # Mark matrix as unfactored and unconstrained.              
        self.__A_is_factored = False
        self.__A_is_constrained = False
    
    
    def get_matrix(self):
        """
        Returns system matrix 
        """
        return self.__A
    
        
    def constrain_matrix(self):
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
        dofs = self.get_dofs()
        n_dofs = len(dofs)
        
        A = self.get_matrix()
        C = self.get_C()
        """
        d = self.d()
        """
        
        #
        # List within which to record modifications
        #
        self.column_records = []
        
        constraints = self.get_dofhandler().constraints
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
            
            """
            #
            # Modify b's rows
            #
            if self.has_rhs:
                one = sparse.csc_matrix(np.ones(b.shape[1]))
                b -= d[k]*ak.dot(one)
            """
            
            #
            # Record columns of A
            #  
            self.column_records.append(ak.copy())
                
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
            
            """            
            #
            # Modify b's rows
            # 
            if self.has_rhs:
                bk = b.getrow(k)
                #bk = b[k].toarray()[0,0]
                b += ck.T.dot(bk)
            """
                        
            #
            # Zero out row k 
            # vectorsvectors
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
            c_eqns = self.dof2eqn(c_dofs)
            ave_vec[c_eqns] = (np.sum(a_diag)-n_cdofs)/(n_dofs-n_cdofs)
        I_ave = sparse.dia_matrix((ave_vec,0), shape=(n_dofs,n_dofs))
        A = A.dot(I_ave)
        
        """  
        #
        # Set b = 0 at constraints 
        # 
        if self.has_rhs:
            zc = np.ones(n_dofs)
            zc[c_dofs] = 0
            Izc = sparse.dia_matrix((zc,0),shape=(n_dofs,n_dofs)) 
            b = Izc.dot(b)
         """       
        
        self.__A = A
        self.__A_is_constrained = True
        
        """
        if self.has_rhs:
            self.__b = b
        """
        
        
    def matrix_is_factored(self):
        """
        Determine whether the matrix has been factored
        """       
        return self.__A_is_factored
    
    
    def matrix_is_constrained(self):
        """
        Determine whether the system matrix has been constrained
        """
        self.__A_is_constrained
        
        
    def invert_matrix(self, b=None, factor=False):
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
        
        A = self.get_matrix()
        
        if b is None:
            #
            # Use stored b
            # 
            assert self.has_rhs(), 'Specify right hand side'
            assert self.rhs_is_constrained(), 'Constrain right hand side.'
        else:
            #
            # New b
            # 
            self.set_rhs(b)
            self.constrain_rhs()
                 
        b = self.get_rhs()
                
        #
        # Solve the linear system
        #
    
        # Convert to sparse column format
        if factor:
            A = A.tocsc()
            b = sparse.csc_matrix(b)
            u = self.__invA.solve(b.toarray())    
        else:
            u = linalg.spsolve(self.get_matrix(), self.get_rhs())    
        if len(u.shape)==1:
            u = u[:,None]
            
        self.__u = u
        
        
    def set_rhs(self, rhs):
        """
        Store right hand side
        
        Inputs:
        
            rhs: None, numpy array
        """
        dofs_error = 'Right hand side incompatible with system shape'
        
        if rhs is not None:
            #
            # Non-trivial rhs
            #
            assert type(rhs) is np.ndarray or sparse.issparse(rhs), \
            'Right hand side should be a numpy array.'
                 
            assert rhs.shape[0] == len(self.get_dofs()), dofs_error
            
            if len(rhs.shape)==1:
                # 
                # Convert to 2d array
                # 
                rhs = rhs[:,None]
            self.__b = sparse.csc_matrix(rhs)
        else:
            self.__b = None
        self.__b_is_constrained = False
        
    
    def get_rhs(self):
        """
        Return right hand side
        """
        return self.__b
    
    
    def has_rhs(self):
        """
        Check whether rhs has been specified
        """
        return self.__b is not None
    
    
    def n_rhs(self):
        """
        Returns the number of columns in the right hand side.
        """
        return self.__b.shape[1]
    
    
    def constrain_rhs(self):
        """
        Modify right hand side to incorporate constraints.
        """
        #
        # Check that system has rhs
        # 
        assert self.has_rhs(), 'Specify rhs, using "set_rhs".'
        
        #
        # Check that rhs not constrained 
        # 
        if self.rhs_is_constrained():
            return
        
        b = self.get_rhs()
        n_dofs = len(self.get_dofs())
            
        constraints = self.get_dofhandler().constraints
        c_dofs = constraints['constrained_dofs']
        C = self.get_C()
        d = self.get_d()
        i = 0
        
        for c_dof in c_dofs:
            k = self.dof2eqn(c_dof)
            
            ck = C.getrow(k)
            ak = self.column_records[i]
            
            #
            # Modify columns
            # 
            one = sparse.csc_matrix(np.ones(self.n_rhs()))
            b -= d[k]*ak.dot(one)
            
            #
            # Modify b's rows
            # 
            bk = b.getrow(k)
            b += ck.T.dot(bk)
            
            #
            # Set b=0 at the constraints
            # 
            zc = np.ones(n_dofs)
            zc[c_dofs] = 0
            Izc = sparse.dia_matrix((zc,0),shape=(n_dofs,n_dofs)) 
            b = Izc.dot(b)
            
            i += 1
        self.__b = b
        self.__b_is_constrained = True
    
    
    def rhs_is_constrained(self):
        """
        Check whether the rhs has been constrained.
        """
        return self.__b_is_constrained
                 
    
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
        C, d = self.get_C(), self.get_d()
        
        n_samples = self.n_rhs()
        drep = np.tile(d[:, np.newaxis], (1,n_samples))
        
        #
        # Modify dofs that don't depend on others
        # 
        sf = self.get_basis().subforest_flag()
        n_dofs = self.get_dofhandler().n_dofs(subforest_flag=sf)
        ec_dofs = [i for i in range(n_dofs) if C.getrow(i).nnz==0]        
        if type(u) is not np.ndarray:
            #
            # Convert to ndarray if necessary
            # 
            u = u.toarray()
        u[ec_dofs,:] = drep[ec_dofs,:]
        u = C.dot(u) + drep
              
        #
        # Store or return result       
        # 
        if x is None:
            self.__u = u
        else:
            return u
           
    
    def solve_system(self, b=None, factor=False):
        """
        Compute the solution of the linear system 
        
            Ax = b, subject to constraints "x=Cx+d"
        
        This method combines
        
            set_constraint_relation
            constrain_matrix
            constrain_rhs
            factor_matrix
            invert_matrix
            resolve_constraints
        """
        comment = Verbose()
        #
        # Parse right hand side
        # 
        if b is None:
            #
            # No b specified
            # 
            assert self.get_rhs() is not None, 'No right hand side specified.'
        else:
            #
            # New b
            # 
            self.set_rhs(b)            
            
        #
        # Define constraint system
        # 
        comment.tic('Setting constraint relation')
        if self.get_C() is None:
            self.set_constraint_relation()
        comment.toc()
        
        #
        # Apply constraints to A
        #
        comment.tic('Constraining matrix') 
        if not self.matrix_is_constrained():
            self.constrain_matrix()
        comment.toc()
        
        #
        # Apply constraints to b
        #
        comment.tic('constraining vector') 
        if not self.rhs_is_constrained():
            self.constrain_rhs()
        comment.toc()
        
        #
        # Factor matrix
        # 
        if factor:
            if not self.matrix_is_factored():
                self.factor_matrix()
            
        #
        # Solve the system
        #
        comment.tic('Inverting matrix') 
        self.invert_matrix(factor=factor)
        comment.toc()
        
        #
        # Resolve constraints
        #
        comment.tic('Resolving constraints')
        self.resolve_constraints()
        comment.toc()
        
    
    def get_solution(self, as_function=True):
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
            u = Nodal(data=self.__u, basis=self.get_basis())
            return u
