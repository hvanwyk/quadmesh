"""
Module for storing, factorizing, and solving systems with semi-positive definite matrices.
"""
# Built-in modules
from math import sqrt
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from scipy.sparse import linalg as spla


# CHOLMOD: Cholesky decomposition
from sksparse.cholmod import cholesky
from sksparse.cholmod import cholesky_AAt 
from sksparse.cholmod import Factor
from sksparse.cholmod import  CholmodNotPositiveDefiniteError

def diagonal_pseudo_inverse(d,eps=None):
    """
    Compute the (approximate) pseudo-inverse of a diagonal matrix of 
    eigenvalues.
    
    Inputs:

        d: double, (n,) array, the non-zero entries of a diagonal matrix 
        
        eps: double (>0), cut-off tolerance for zero entries. Default is 
            the machine epsilon.
    """
    if eps is None:
        #
        # Default tolerance
        # 
        eps = np.finfo(float).eps
    else:
        assert eps > 0, 'Input "eps" should be positive.'
            
    #
    # Compute the pseudo-inverse of the diagonal matrix of eigenvalues
    #
    d_inv = np.zeros(d.shape)
    i_nz = np.abs(d)>eps
    d_inv[i_nz] = 1/d[i_nz]
    D_inv = np.diag(d_inv)
    
    return D_inv


class SPDMatrix(object):
    def __init__(self, C):
        """
        Initialize the SPD matrix.

        Parameters:
        - C: numpy.ndarray or scipy.sparse matrix
            The input matrix to be stored as the SPD matrix.

        Raises:
        - AssertionError: If the input matrix is not square.

        """
        # Checks
        assert C.shape[0] == C.shape[1], 'Input "C" must be square.'

        # Set the sparsity of the matrix
        self.set_sparsity(sp.issparse(C))
        
        #assert linalg.issymmetric(C,atol=1e-12), 'Input "C" must be symmetric'
        
        # Store the matrix
        self.set_matrix(C)


    def set_matrix(self, C):
        """
        Store the SPD matrix.

        Parameters:
        - C: numpy.ndarray or scipy.sparse matrix
            The input matrix to be stored as the SPD matrix.

        """
        self.__C = C


    def get_matrix(self):
        """
        Return the SPD matrix.

        Returns:
        - numpy.ndarray or scipy.sparse matrix
            The stored SPD matrix.

        """
        return self.__C


    def size(self):
        """
        Return the number of rows (=columns) of C.

        Returns:
        - int
            The number of rows (=columns) of the SPD matrix.

        """
        return self.__C.shape[0]


    def set_rank(self,rank):
        """
        Store the rank of the matrix
        """
        assert isinstance(rank, int), 'Input "rank" should be an integer.'
        assert rank >= 0, 'Input "rank" should be non-negative.'
        assert rank <= self.size(), 'Input "rank" should be less than or equal to the size of the matrix.'
        self.__rank = rank


    def get_rank(self):
        """
        Return the rank of the matrix
        """
        return self.__rank


    def set_sparsity(self, is_sparse):
        """
        Set the sparsity of the matrix.

        Parameters:
        - is_sparse: bool
            True if the matrix is sparse, False otherwise.

        Raises:
        - AssertionError: If the input is_sparse is not a boolean.

        """
        assert isinstance(is_sparse, bool), \
        'Input "is_sparse" should be a boolean.'

        self.__is_sparse = is_sparse


    def is_sparse(self):
        """
        Return True if the matrix is sparse, False otherwise.

        Returns:
        - bool
            True if the matrix is sparse, False otherwise.

        """
        return self.__is_sparse


    def is_degenerate(self):
        """
        Return True if the matrix is degenerate, i.e. not truly positive 
        definite. This is the case if the matrix rank is less than its size.

        Returns:
        - bool
            True if the matrix is degenerate, False otherwise.
        """
        return self.rank() < self.size()


    def decompose(self):
        """
        Factorize the matrix
        """
        pass


    def reconstruct(self):
        """
        Reconstruct the matrix from its factorization
        """
        pass


    def dot(self):
        """
        Compute the matrix vector product C*b
        """
        pass


    def solve(self):
        """
        Solve the system Cx = b for x
        """
        pass


    def sqrt_dot(self):
        """
        Compute Sqrt(C)*b
        """
        pass


    def sqrt_solve(self):
        """
        Solve Sqrt(C)*x = b for x
        """
        pass



class CholeskyDecomposition(SPDMatrix):
    """
    Description:

    Cholesky Decomposition of a symmetric positive definite matrix of the form

        C = LL'

    Decompositions differ based on properties of C: 

    1. chol_sparse: C is sparse and non-degenerate: use cholmod, whose decomposition is of
        the form 
        
            PCP' = LL', 
            
        where 
        
            P is a permutation matrix, 
            L is lower triangular, and sparse

    2. chol_full: C is full and non-degnerate: use standard Cholesky, whose decomposition is
        of the form

            C = LL',

        where L is lower triangular.
    
            
    3. chol_mod: C is degenerate (convert to full if sparse): use modified Cholesky, whose 
        decomposition is of the form

            P*(C + E)*P' = L*D*L',

        where

            P is a permutation matrix,
            L is the cholesky factor (P*L is lower triangular)
            E is a perturbation matrix so that C+E is positive definite
            D is diagonal
            D0 block diagonal matrix so that C = L*D0*L'

    Attributes:

        __C: Symmetric positive definite matrix
        __L: Cholesky factor
        __D: Diagonal matrix
        __P: Permutation matrix
        __D0: Diagonal matrix so that C = L*D0*L'

    Methods:
        
        - decompose: Compute the Cholesky decomposition of the matrix
        - reconstruct: Reconstruct the matrix from its Cholesky decomposition
        - dot: Compute the matrix vector product C*b
        - solve: Solve the system Cx = b for x
        - sqrt: Compute Sqrt(C)*b
        - sqrt_solve: Solve Sqrt(C)*x = b for x 
    """
    def __init__(self,C,verbose=False):
        
        # Initialize the SPD matrix
        SPDMatrix.__init__(self,C)
        
        # 
        # Initialize the Cholesky decomposition
        # 
        # Default assumption: matrix is not degenerate
        self.set_degeneracy(False)
        
        # Compute the factorization
        self.decompose(C, verbose=verbose)          
    

    def set_degeneracy(self, is_degenerate):
        """
        Set the degeneracy of the matrix
        """
        assert isinstance(is_degenerate, bool), \
            'Input "is_degenerate" should be a boolean.'
        self.__is_degenerate = is_degenerate


    def is_degenerate(self):
        """
        Return True if the matrix is degenerate, i.e. not truly positive 
        definite.
        """
        return self.__is_degenerate


    def decompose(self,C,verbose=True):
        """
        Compute the Cholesky decomposition of the matrix C
        """
        if self.is_sparse():
            if verbose: print('Sparse matrix - using CHOLMOD')
            #
            # Sparse matrix - try efficient Cholesky factorization
            # 
            try:
                #
                # Try Cholesky (will fail if not PD)
                #
                L = cholesky(C.tocsc(), mode='supernodal')

                # Store Cholesky decomposition
                self.set_factors(L)
                
                # Record non-degeneracy
                self.set_degeneracy(False)

            except CholmodNotPositiveDefiniteError:
                if verbose: 
                    print('Matrix not positive definite - using modified Cholesky')

                # Sparse Cholesky failed - degenerate matrix
                self.set_degeneracy(True)
        else:
            #
            # Full Matrix - standard Cholesky
            # 
            if verbose: print('Full matrix - using standard Cholesky')
            try:
                #
                # Try Cholesky (will fail if not PD)
                # 
                L = linalg.cholesky(C,lower=True, check_finite=False)

                # Store Cholesky decomposition
                self.set_factors(L)

                # Record non-degeneracy
                self.set_degeneracy(False)

            except np.linalg.LinAlgError:
                if verbose: 
                    print('Matrix not positive definite - using modified Cholesky')

                # Standard Cholesky failed - degenerate matrix
                self.set_degeneracy(True)
            
        #
        # Not Strictly Positive Definite
        #         
        if self.is_degenerate():
            #
            # Use modified Cholesky
            # 
            if self.is_sparse():
                #
                # Sparse, degenerate matrix - convert to full first :(
                # 
                if verbose: print('Converting sparse matrix to full')
                C = C.toarray()
                
                # Update to non-sparse
                self.set_sparsity(False)

            # Compute modified Cholesky 
            # P*(C + E)*P' = L*D*L' or P*C*P' = L*D0*L'         
            L, D, P, D0 = self.modchol_ldlt(C)

            # 
            # Store Cholesky decomposition
            #
            self.set_factors((L, D, P, D0))  


    def modchol_ldlt(self,C,delta=None):
        """
        Modified Cholesky algorithm based on LDL' factorization.
        
            [L D,P,D0] = modchol_ldlt(A,delta) 
            
        computes a modified Cholesky factorization 
        
            P*(A + E)*P' = L*D*L', where 
        
        P is a permutation matrix, L is unit lower triangular, and D is block
        diagonal and positive definite with 1-by-1 and 2-by-2 
        diagonal blocks.  Thus A+E is symmetric positive definite, but E is
        not explicitly computed.  Also returned is a block diagonal D0 such
        that P*A*P' = L*D0*L'.  If A is sufficiently positive definite then 
        E = 0 and D = D0.  
        The algorithm sets the smallest eigenvalue of D to the tolerance
        delta, which defaults to sqrt(eps)*norm(A,'fro').
        The LDL' factorization is computed using a symmetric form of rook 
        pivoting proposed by Ashcraft, Grimes and Lewis.
        
        Reference:
        S. H. Cheng and N. J. Higham. A modified Cholesky algorithm based
        on a symmetric indefinite factorization. SIAM J. Matrix Anal. Appl.,
        19(4):1097-1110, 1998. doi:10.1137/S0895479896302898,

        Authors: Bobby Cheng and Nick Higham, 1996; revised 2015.
        """
        
        assert isinstance(C, np.ndarray), \
            'Input "C" should be a numpy array.'
        
        assert np.allclose(C, C.T, atol=1e-12), \
            'Input "C" must be symmetric'    

        if delta is None:
            eps = np.finfo(float).eps
            delta = np.sqrt(eps)*linalg.norm(C, 'fro')
            #delta = 1e-5*linalg.norm(A, 'fro')
        else:
            assert delta>0, 'Input "delta" should be positive.'

        n = max(C.shape)

        L,D,p = linalg.ldl(C)  # @UndefinedVariable
        DMC = np.eye(n)
            
        # Modified Cholesky perturbations.
        k = 0
        while k < n:
            one_by_one = False
            if k == n-1:
                one_by_one = True
            elif D[k,k+1] == 0:
                one_by_one = True
                
            if one_by_one:
                #            
                # 1-by-1 block
                #
                if D[k,k] <= delta:
                    DMC[k,k] = delta
                else:
                    DMC[k,k] = D[k,k]
                
                k += 1
        
            else:  
                #            
                # 2-by-2 block
                #
                E = D[k:k+2,k:k+2]
                T,U = linalg.eigh(E)
                T = np.diag(T)
                for ii in range(2):
                    if T[ii,ii] <= delta:
                        T[ii,ii] = delta
                
                temp = np.dot(U,np.dot(T,U.T))
                DMC[k:k+2,k:k+2] = (temp + temp.T)/2  # Ensure symmetric.
                k += 2
        P = np.eye(n)
        P = P[:,p]
        
        #P = sp.diags([1],0,shape=(n,n), format='coo') 
        #P.row = P.row[p]
        #P = P.tocsr()
        
        #ld = np.diagonal(P.dot(L))
        #if any(np.abs(ld)<1e-15):
        #    print('L is singular')
            
        return L, DMC, P, D


    def set_factors(self,L):
        """
        Store the Cholesky factorization
        """
        self.__L = L


    def get_factors(self,verbose=False):
        """
        Returns the Cholesky factorization of the matrix
        """                
        if self.__L is None:
            #
            # Return None if Cholesky decomposition not computed
            # 
            if verbose: print('Cholesky decomposition not computed.')
            return None 
        elif self.is_degenerate():
            #
            # Return the modified Cholesky decomposition
            # 
            if verbose: 
                print('Modified Cholesky decomposition')
                print('Returning L, D, P, D0, where')
                print('C = P*(C+E)*P\' = L*D*L\' and P*C*P\' = L*D0*L\'')

            return self.__L
        
        else:
            #
            # Return cholesky decomposition
            #            
            if verbose: 
                print('Cholesky factor')
                if self.is_sparse():
                    print('CHOLMOD factor')
                else:
                    print('Standard Cholesky factor')
            return self.__L
        
    
    def reconstruct(self, degenerate=False):
        """
        Reconstruct the matrix from its Cholesky decomposition
        """
        if self.is_sparse():
            n = self.size()
            #
            # Sparse
            # 
            f = self.get_factors()

            # Build permutation matrix
            P = f.P()
            I = sp.diags([1],0, shape=(n,n), format='csc')
            PP = I[P,:]
            
            # Compute P'L
            L = f.L()
            L = PP.T.dot(L)
            
            # Check reconstruction LL' = PAP'
            return L.dot(L.T) 
        elif not self.is_degenerate():
            #
            # Full, non-degenerate matrix
            # 
            L = self.get_factors()
            return L.dot(L.T)
        else:
            #
            # Full, degenerate matrix
            # 
            L, D, P, D0 = self.get_factors()
            if degenerate:
                #
                # Return C = L*D0*L'
                # 
                return P.T.dot(L.dot(D0.dot(L.T.dot(P))))
            else:
                #
                # Return P*(C+E)*P' = L*D*L'
                #   
                return P.T.dot(L.dot(D.dot(L.T.dot(P))))
        
   
    
    def dot(self,b):
        """
        Compute the matrix vector product C*b
        
        Input:
        
            b: double, (n,m) compatible array
        
            
        Output:
        
            C*b: double, (n,m) product
        """
        assert b.shape[0]==self.size(), 'Input "b" has incompatible shape.'+\
        'Size C: {0}, Size b: {1}'.format(self.size(), b.shape)
        if sp.issparse(b):
            #
            # b is a sparse matrix
            #
            C = self.get_matrix()
            b = b.tocsc()
            return b.T.dot(C).T
        else:
            return self.get_matrix().dot(b)


    def solve(self,b):
        """
        Solve the system C*x = b  by successively solving 
        
            Ly = b for y and hence L' x = y for x.
        
        Parameters:

            b: double, (n,m) array representing the right-hand side of the 
                system.
        
        Returns:

            The solution x of the system C*x = b.
        """
        if not self.is_degenerate():
            if self.is_sparse():
                #
                # Use CHOLMOD
                #
                L = self.get_factors()
                return L.solve_A(b)
            else:
                #
                # Use standard Cholesky
                # 
                L = self.get_factors()
                y = linalg.solve_triangular(L,b,lower=True)
                x = linalg.solve_triangular(L.T,y,lower=False)
                return x
        else:
            #
            # Use Modified Cholesky
            # 
            L, D, P, dummy = self.get_factors()
            PL = P.dot(L)
            y = linalg.solve_triangular(PL,P.dot(b),lower=True, unit_diagonal=True)
            Dinv = sp.diags(1./np.diagonal(D))
            z = Dinv.dot(y)
            x = linalg.solve_triangular(PL.T,z,lower=False,unit_diagonal=True)
            return P.T.dot(x)

    def sqrt_dot(self,b,transpose=False):
        """
        Returns L*b (or L'*b), where A = L*L'
        
        Parameters:

            b: double, The compatible vector/matrix to be multiplied.

            transpose: bool, (optional): If True, returns L'*b. 
                If False, returns L*b. Defaults to False.
        
        Returns:

            The result of multiplying L (or L') with b.
        """
        assert self.__L is not None, \
            'Cholesky factor not computed.'\
            
        n = self.size()
        if not self.is_degenerate():
            #
            # Non-degenerate matrix
            # 
            if self.is_sparse():
                #
                # Sparse non-degenerate matrix, use CHOLMOD (C = P'*L*L'*P)
                #
                f = self.get_factors()
                L = f.L()
                if transpose:
                    #
                    # W'*b, where W = P'*L => L'*P*b
                    # 
                    return L.T.dot(f.apply_P(b))
                else:
                    #
                    # W*b, where W = P'*L
                    # 
                    return f.apply_Pt(L.dot(b))
            else:
                # 
                # Full, non-degenerate matrix
                #
                L = self.__L
                if transpose:
                    #
                    # L'*b
                    # 
                    return L.T.dot(b)
                else:
                    #
                    # L*b
                    # 
                    return L.dot(b)   
        else:
            #
            # Degenerate matrix: 0 < C+E = P'*L*D*L'*P = W*W'
            # 
            L, D, P, D0 = self.get_factors()
            sqrtD = np.diag(np.sqrt(np.diag(D)))
            if transpose:
                #
                # W'*b
                # 
                return sqrtD.dot(L.T.dot(P.dot(b)))
            else:
                #
                # W*b
                # 
                return P.T.dot(L.dot(sqrtD.dot(b))) 
        

    def sqrt_solve(self,b,transpose=False):
        """
        Solve Sqrt(C)*x = b for x, i.e. L*x = b or L'*x = b, where C = LL'
        
        Note: The 'L' in CHOLMOD's solve_L 
            is the one appearing in the factorization LDL' = PCP'. 
            We first rewrite it as C = WW', where W = P'*L*sqrt(D)
        
        Parameters:

            b: double, compatible vector/matrix representing the right-hand side of the system.
            
            transpose: bool (optional), If True, solves L'x = b. 
                If False, solves Lx = b. Defaults to False.
        
        Returns:
            The solution x of the system Lx = b (or L'*x = b if transpose=True).
        """
        if not self.is_degenerate():
            #
            # Non-degenerate matrix
            # 
            if self.is_sparse():
                #
                # Sparse, non-degenerate matrix (CHOLMOD)
                # 
                f = self.get_factors()
                if transpose:
                    #
                    # Solve L' x = b
                    #
                    #return f.solve_Lt(b,use_LDLt_decomposition=False) 
                    return f.apply_Pt(f.solve_Lt(b,use_LDLt_decomposition=False))
                else:
                    #
                    # Solve Rx = b
                    #
                    #return f.solve_L(b,use_LDLt_decomposition=False) 
                    return f.solve_L(f.apply_P(b),use_LDLt_decomposition=False)
            else:
                #
                # Full, non-degenerate matrix
                # 
                L = self.__L
                if transpose:
                    #
                    # Solve R' x = b
                    # 
                    return linalg.solve_triangular(L.T,b,lower=False)
                else:
                    #
                    # Solve Rx = b
                    # 
                    return linalg.solve_triangular(L,b,lower=True)
        else:
            #
            # Degenerate matrix
            #
            #  C+E = P'*L*sqrtD*sqrtD*L'*P = W*W'
            # 
            L, D, P, D0 = self.get_factors()
            sqrtD = np.diag(np.sqrt(np.diag(D)))
            if transpose:
                #
                # Solve W' x = b
                # 
                return P.T.dot(linalg.solve_triangular(sqrtD.dot(L.T),b,lower=False))
                #return sqrtD.dot(linalg.solve_triangular(L.T,P.dot(b),lower=False))
            else:
                #
                # Solve Wx = b
                # 
                return linalg.solve_triangular(L.dot(sqrtD),P.dot(b)) 
                #return P.T.dot(linalg.solve_triangular(L,sqrtD.dot(b),lower=True))
        
        """
        if self.chol_type() == 'sparse':
            #
            # Sparse Matrix
            #
            f = self.__L
            sqrtDinv = sp.diags(1/np.sqrt(f.D()))
            if transpose:
                # Solve R' x = b
                return f.apply_Pt(f.solve_Lt(sqrtDinv.dot(b)))
            else:
                # Solve Rx = b 
                return sqrtDinv.dot(f.solve_L(f.apply_P(b)))
        else:
            #
            # Full Matrix
            # 
            L, D, P = self.__L, self.__D, self.__P
            PL = P.dot(L)
            sqrtDinv = sp.diags(1/np.sqrt(np.diagonal(D)))
            unit_diagonal = np.allclose(np.diagonal(PL),1)
            if transpose:
                #
                # Solve R' x = b
                # 
                y = sqrtDinv.dot(b)
                
                x = linalg.solve_triangular(PL.T,y, lower=False, 
                                             unit_diagonal=unit_diagonal)
                return P.T.dot(x)
            else:
                y = linalg.solve_triangular(PL, P.dot(b), lower=True, 
                                            unit_diagonal=unit_diagonal)
                
                return sqrtDinv.dot(y)
        """
        
    

class EigenDecomposition(SPDMatrix):
    """
    (Generalized) Eigenvalue decomposition of a symmetric positive definite 
    matrix. Unlike the Cholesky decomposition, the eigendecomposition also 
    stores the nullspace of the matrix (the eigenspace corresponding to zero
    eigenvalues), which is useful for conditional sampling.
    """
    def __init__(self,C,M=None,delta=None):
        """
        Constructor, initialize the (generalized) eigendecomposition of 

        Inputs:

            C: numpy.ndarray or scipy.sparse matrix
                The input matrix to be stored as the SPD matrix.

            M: numpy.ndarray or scipy.sparse matrix (optional mass matrix).

            delta: float, the smallest allowable eigenvalue in the decomposition.
        """
        # Initialize the SPD matrix
        SPDMatrix.__init__(self,C)
        
        # Store the mass matrix
        self.set_mass_matrix(M)
                    
        # Set the eigenvalue lower bound
        self.set_eigenvalue_lower_bound(delta)

        # Compute the eigendecomposition
        self.decompose()


    def set_mass_matrix(self, M):
        """
        Store the mass matrix
        """
        self.__M = M


    def get_mass_matrix(self):
        """
        Return the mass matrix
        """
        return self.__M


    def has_mass_matrix(self):
        """
        Return True if the mass matrix is available
        """
        return self.__M is not None
    

    def size(self):
        """
        Return the number of rows (=columns) of C
        """
        return self.get_matrix().shape[0]
   

    def set_eigenvalue_lower_bound(self, delta):
        """
        Store the eigenvalue lower bound

        Input:

            delta: float, the smallest allowable eigenvalue in the
                decomposition. Eigenvalues below this value are set to delta.

        Notes: 
        
            (i) Under the default value (None), the smallest eigenvalues
                is set to delta = sqrt(eps)*norm(C,'fro'), where eps is 
                the machine epsilon.

            (ii) If delta = 0, only negative eigenvalues are set to zero.
        """
        if delta is None:
            #
            # Default lower bound
            # 
            C = self.get_matrix()
            eps = np.finfo(float).eps
            delta = np.sqrt(eps)*linalg.norm(C, 'fro')

        # Check whether delta is non-negative
        assert delta >= 0, 'Input "delta" should be non-negative.'

        # Store the lower bound
        self.__delta = delta


    def get_eigenvalue_lower_bound(self):
        """
        Return the eigenvalue lower bound
        """
        return self.__delta
    

    def decompose(self):
        """
        Compute the (generalized) eigendecomposition of the matrix C, i.e. 

            C*vi = di*M*vi, i = 1,...,n

        """
        #
        # Preprocessing
        #
        if self.has_mass_matrix():
            #
            # Generalized eigendecomposition
            # 
            is_generalized = True
            M = self.get_mass_matrix()
        else:
            #
            # Standard eigendecomposition
            # 
            is_generalized = False
            
        
        # Get the matrix
        C = self.get_matrix()
        
        if self.is_sparse():
            #
            # Convert to full matrix
            # 
            C = C.toarray()
            
        if is_generalized:
            #
            # Compute the generalized eigendecomposition
            # 
            d, V = linalg.eigh(C,M)
        else:
            #
            # Compute the eigendecomposition
            # 
            d, V = linalg.eigh(C)      
        
        # Rearrange to ensure decreasing order
        d = d[::-1]
        V = V[:,::-1]
        
        
        # Modify negative eigenvalues
        delta = self.get_eigenvalue_lower_bound()

        # Ensure eigenvalues are at least delta
        d[d<=delta] = delta

        #
        # Store the eigendecomposition
        #         
        self.set_factors(d,V)


    def set_factors(self,d,V):
        """
        Store the eigendecomposition
        """
        self.__V = V
        self.__d = d


    def get_factors(self):
        """
        Return the eigendecomposition of the matrix
        """
        return self.__d, self.__V
    

    def get_eigenvectors(self):
        """
        Return the range of the matrix
        """
        return self.__V
    

    def get_eigenvalues(self):
        """
        Return the eigenvalues of the matrix
        """
        return self.__d 
        

    def reconstruct(self):
        """
        Reconstruct the matrix from its eigendecomposition
        """
        d, V = self.get_factors()
        C = V.dot(np.diag(d).dot(V.T))
        if self.has_mass_matrix():
            M = self.get_mass_matrix()
            return C, M
        else:
            return C
    

    def dot(self,b):
        """
        Compute the matrix vector product C*b
        """
        d, V = self.get_factors()
        return V.dot(np.diag(d).dot(V.T.dot(b)))


    def solve(self,b,eps=None,generalized=False):
        """
        Solve the system C*x = b for x or the generalized system Cx = Mb for x,
        using the eigendecomposition of C.

        For Cx = b, the solution is given by 
        
            x = V*Dinv*V'*b, 

        where Dinv is the pseudo-inverse of the diagonal matrix D, whereas for 
        the generalized problem Cx = Mb, the solution is given by

            x = V*Dinv*V'*M*b.
        
        Inputs:
        
            b: double, (n,m) array
            
            eps: double >0, cut off tolerance for zero entries in the diagonal

            generalized: bool, specifies whether to solve the generalized system

        Output:
            
                x: double, (n,m) solution of the (generalized) system.
        """
        d = self.get_eigenvalues()
        D_inv = diagonal_pseudo_inverse(d,eps=eps)
        V = self.get_eigenvectors()
        if generalized:
            #
            # Solve the generalized system
            # 
            M = self.get_mass_matrix()
            return V.dot(D_inv.dot(np.dot(V.T, M.dot(b))))
        else:
            #
            # Solve the system without mass matrix on the right
            #   
            return V.dot(D_inv.dot(np.dot(V.T, b)))


    def sqrt_dot(self,x, transpose=False):
        """
        Compute Sqrt(C)*x
        
        Compute Rx (or R'x), where C = RR'
        
        Inputs:
        
            x: double, (n,k) array
            
            transpose: bool, determine whether to compute Rx or R'x
            
        Output:
        
            b = Rx or R'x
        """
        d, V = self.get_factors()

        if transpose:
            # Sqrt(D)*V'x
            return np.diag(np.sqrt(d)).dot(V.T.dot(x))
        else:
            # V*Sqrt(D)*x
            return V.dot(np.diag(np.sqrt(d)).dot(x))


    def sqrt_solve(self,b,transpose=False,eps=None):
        """
        Solve Sqrt(S)*x = b for x
       
        Solve the system Rx=b (or R'x=b if transpose) where R = V*sqrt(D) in 
        the decomposition M^{-1}K = VDV' = RR' 
        
        Inputs:
        
            b: double, (n,k)  right hand side
            
            transpose: bool [False], specifies whether system or transpose is 
                to be solved.
        """
        d, V = self.get_factors()
        sqrtD_inv = diagonal_pseudo_inverse(np.sqrt(d),eps=eps)
        if transpose:
            #
            # Solve sqrtD*V'x = b
            # 
            return V.dot(sqrtD_inv.dot(b))
        else:
            #
            # Solve V*sqrtD x = b
            #
            return sqrtD_inv.dot(np.dot(V.T, b))


    def get_nullspace(self, tol=1e-13):
        """
        Determines an othornormal set of vectors spanning the nullspace
        """ 
        # Determine what eigenvalues are below tolerance
        d = self.get_eigenvalues()

        # Indices of small eigenvalues
        ix_null = np.abs(d)<tol 
        
        # Return the eigenvectors corresponding to small eigenvalues
        return self.get_eigenvectors()[:,ix_null]


    def get_rank(self, tol=1e-13):
        """
        Determines the rank of the matrix based on the number of non-zero
        eigenvalues
        
        Input:

            tol: double, tolerance for determining rank

        Output:

            rank: int, the approximate rank of the matrix
        """
        d = self.get_eigenvalues()
        return np.sum(np.abs(d)>tol)


    def get_range(self, tol=1e-13):
        """
        Determines an othornormal set of vectors spanning the range
        """ 
        # Determine what eigenvalues are below tolerance
        d = self.get_eigenvalues()

        # Indices of small eigenvalues
        ix_range = np.abs(d)>tol 
        
        # Return the eigenvectors corresponding to small eigenvalues
        return self.get_eigenvectors()[:,ix_range]