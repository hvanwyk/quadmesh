from mesh import convert_to_array
from assembler import AssembledForm
from function import Function
import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import numbers
           
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
    def __init__(self, assembler, problem_index=None, \
                 bilinear_form=None, linear_form=None):
        """
        Constructor
        
        
        Inputs:
        
            assembler: Assembler, containing the dof information and possibly
                storing the bilinear and linear forms. 
            
            problem_index: int, index to locate problem containing bilinear
                form.
            
            bilinear_form: AssembledForm, assembled bilinear form
            
            linear_form: AssembledForm, assembled linear form
            
        """
        self.assembler = assembler
        
        #
        # Parse bilinear/linear forms
        #
        if problem_index is None:
            #
            # No problem specified -> bilinear form must be specified directly
            #
            assert bilinear_form is not None, \
                'Must specify bilinear form or problem_index.'
            
        else:
            #
            # Problem number given -> check that there's a bilinear form
            # 
            assert 'bilinear' in assembler.af[problem_index], \
                'Problem must have a bilinear form.'
            
            bilinear_form = assembler.af[problem_index]['bilinear']
            
            if 'linear' in assembler.af[problem_index]:
                #
                # Linear form appears in problem
                # 
                linear_form = assembler.af[problem_index]['linear']
                
            else:
                #
                # No linear form appears in problem it may be specified explicitly
                # 
                linear_form = linear_form
                
        #
        # Determine element type  
        # 
        etype = bilinear_form.trial_etype 

        #
        # Check that the element type is consistent 
        #  
        assert etype==bilinear_form.test_etype, \
        'Trial and test spaces must have the same element type.'
        
        self.__etype = etype
        
        
        #
        # Determine system dofs   
        # 
        dofs = bilinear_form.row_dofs
        
        #
        # Check that dofs are consistent (should be if element type is)
        # 
        assert np.allclose(dofs, bilinear_form.col_dofs), \
        'Test and trial dofs should be the same.'
        
        self.__dofs = dofs
        

        #
        # Form Dof-to-Equation Mapping
        # 
        n_dofs = len(dofs)
        dof2eqn = np.zeros(dofs[-1]+1, dtype=np.int)
        dof2eqn[dofs] = np.arange(n_dofs, dtype=np.int)
        
        #
        # Store Dof-to-Equation mapping 
        #
        self.__dof2eqn = dof2eqn
        
        # 
        # Store Bilinear Form
        # 
        self.set_matrix(bilinear_form)
        
        #
        # Store Linear Form
        # 
        self.set_rhs(linear_form)
        
        #
        # Initialize constraint matrix
        # 
        self.__C = None
        self.__d = None
        
        """
        #
        # Initialize solution vector
        #
        if self.__b is not None:
            #
            # Solution vector
            # 
            n_samples = self.n_samples
            if self.n_samples is None:
                #
                # Sample size is None
                # 
                self.__u = np.zeros(n_dofs)
            else:
                #
                # Sample size is given
                # 
                self.__u = np.zeros((n_dofs,n_samples))
        else:
            #
            # No solution vector 
            #
            self.__u = None
        """                
            
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
        """
        # Initialize solution vector
        #
        if self.__b is not None:
            #
            # Solution vector
            # 
            n_samples = self.n_samples
            if self.n_samples is None:
                #
                # Sample size is None
                # 
                self.__u = np.zeros(n_dofs)
            else:
                #
                # Sample size is given
                # 
                self.__u = np.zeros((n_dofs,n_samples))
        else:
            #
            # No solution vector 
            #
            self.__u = None
        """
      
      
    def set_constraint_relation(self):
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
    
    
    def set_matrix(self, bilinear_form):
        """
        Store system matrix
        
        Inputs:
        
            bilinear_form: AssembledForm, bilinear assembled form
        """
        #
        # Ensure that there is only one bilinear form
        # 
        multiple_bfs_error = 'Only single bilinear form allowed.'
        bf_nsamples = bilinear_form.n_samples
        if bf_nsamples is not None:
            #
            # Sampled bilinear form 
            # 
            if bf_nsamples == 1:
                #
                # Only one sample
                # 
                A = bilinear_form.get_matrix()[0]
            elif bf_nsamples > 1:
                #
                # Multiple samples check that subsamples
                #   
                if len(bilinear_form.sub_samples)==1:
                    #
                    # Only one subsample good
                    # 
                    A = bilinear_form.get_matrix()[0]
                else:
                    #
                    # Multiple bilinear forms
                    #
                    raise Exception(multiple_bfs_error)
            else:
                #
                # Multiple bilinear forms
                #  
                raise Exception(multiple_bfs_error)
        else:
            #
            #  Deterministic bf
            # 
            A = bilinear_form.get_matrix()
        
        self.__A = A                
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
        dofs = self.dofs()
        n_dofs = len(dofs)
        
        A = self.get_matrix()
        C = self.C()
        """
        d = self.d()
        """
        
        #
        # List within which to record modifications
        #
        self.column_records = []
        
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
        
    
    def factor_matrix(self):
        """
        Factor system matrix
        """
        A = self.get_matrix()
        self.__invA = linalg.splu(A.tocsc())
        self.__A_is_factored = True
        
    
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
        
        
    def invert_matrix(self, b=None):
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
        A = A.tocsc()
        b = b.tocsc()
        self.__u = self.__invA.solve(b.toarray())    
            
        if self.n_samples is None:
            self.__u = self.__u.ravel()
            
            
    def set_rhs(self, rhs):
        """
        Store right hand side
        
        Inputs:
        
            rhs: AssembledForm, numpy array, or sparse array
        """
        dofs_error = 'Right hand side incompatible with system shape'
        if rhs is None:
            #
            # Rhs is not specified
            #
            b = rhs
            n_samples = None        
        elif isinstance(rhs, AssembledForm):
            #
            # Rhs given as linear form
            #
            assert rhs.type == 'linear', \
            'Right hand side must be linear form'
            
            assert np.allclose(self.dofs(), rhs.row_dofs), \
            'Test and trial dofs should be the same.'
                
            assert self.etype()==rhs.test_etype,\
           'Test and trial spaces must have same element type.'
            
            b = rhs.get_matrix()    
            n_samples = rhs.n_samples
                   
        elif type(rhs) is np.ndarray:
            #
            # Rhs given as (full) array
            #
            assert b.shape[0] == len(self.dofs()), dofs_error
            
            b = rhs
            
            if len(rhs.shape)==1:
                #
                # One dimensional array
                #
                n_samples = None
            elif len(rhs.shape)==2:
                #
                # Two dimensional array
                #
                n_samples = rhs.shape[1]
                
        elif sparse.issparse(rhs):
            #
            # Rhs is a sparse matrix   
            # 
            assert rhs.shape[0] == len(self.dofs()), dofs_error
            
            b = rhs
            n_samples = rhs.shape[1]
        
        #
        # Store information
        #  
        self.__b = b
        self.n_samples = n_samples 
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
        n_dofs = len(self.dofs())
            
        constraints = self.dofhandler().constraints
        c_dofs = constraints['constrained_dofs']
        C = self.C()
        d = self.d()
        i = 0
        
        for c_dof in c_dofs:
            k = self.dof2eqn(c_dof)
            
            ck = C.getrow(k)
            ak = self.column_records[i]
            
            #
            # Modify columns
            # 
            one = sparse.csc_matrix(np.ones(self.n_samples))
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
        C, d = self.C(), self.d()
        
        n_samples = self.n_samples
        if n_samples is not None:
            #
            # Sampled right hand side
            # 
            drep = np.tile(d[:, np.newaxis], (1,n_samples))
        
        #
        # Modify dofs that don't depend on others
        # 
        n_dofs = self.dofhandler().n_dofs()
        ec_dofs = [i for i in range(n_dofs) if C.getrow(i).nnz==0]        
        n_samples = self.n_samples
        if n_samples is None:
            #
            # Deterministic
            # 
            u[ec_dofs] = d[ec_dofs]
        else:
            # 
            # Sampled
            #
            if type(u) is not np.ndarray:
                #
                # Convert to ndarray if necessary
                # 
                u = u.toarray()
            u[ec_dofs,:] = drep[ec_dofs,:]
                
        #
        # Modify other dofs
        # 
        if n_samples is None:
            #
            # Deterministic
            # 
            u = C.dot(u) + d
        else:
            #
            # Sampled
            # 
            u = C.dot(u) + drep
              
        
        #
        # Store or return result       
        # 
        if x is None:
            self.__u = u
        else:
            return u
           
    
    def solve_system(self, b=None):
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
        if self.C() is None:
            self.set_constraint_relation()
            
        #
        # Apply constraints to A
        # 
        if not self.matrix_is_constrained():
            self.constrain_matrix()
            
        #
        # Apply constraints to b
        # 
        if not self.rhs_is_constrained():
            self.constrain_rhs()
            
        #
        # Factor matrix
        # 
        if not self.matrix_is_factored():
            self.factor_matrix()
            
        #
        # Solve the system
        # 
        self.invert_matrix()
        
        
        #
        # Resolve constraints
        #
        self.resolve_constraints()
        
    
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
            u = Function(self.__u, 'nodal', mesh=self.assembler.mesh, \
                         dofhandler=self.dofhandler(), \
                         subforest_flag=self.assembler.subforest_flag)
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