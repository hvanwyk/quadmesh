"""
Module for factorizations of positive definite matrices.
"""
# Built-in modules
import numpy as np
from scipy import linalg
import scipy.sparse as sp
from scipy.sparse import linalg as spla


# CHOLMOD: Cholesky decomposition
from sksparse.cholmod import cholesky
from sksparse.cholmod import cholesky_AAt 
from sksparse.cholmod import Factor
from sksparse.cholmod import  CholmodNotPositiveDefiniteError


class SPDMatrix(object):
    def __init__(self):
        """
        Initialize the SPD matrix
        """
        pass

    def size(self):
        pass

    def rank(self):
        pass

    def issparse(self):
        pass

    def isdegenerate(self):
        pass

    def decompose(self):
        pass

    def reconstruct(self):
        pass
    
    def dot(self):
        pass

    def solve(self):
        pass

    def sqrt(self):
        pass

    def sqrt_solve(self):
        pass



class CholeskyDecomposition(object):
    """
    Description:

    Cholesky Decomposition of a symmetric positive definite matrix of the form

        C = LL'

    Decompositions differ based on properties of C: 

    1. C is non-degenerate: use cholmod, whose decomposition is of
        the form 
        
            PCP' = LDL', 
            
        where 
        
            P is a permutation matrix, 
            L is lower triangular, sparse, and 
            D is diagonal.

    2. C is degenerate (convert to full if sparse): use modified Cholesky, whose 
        decomposition is of the form

            P*(C + E)*P' = L*D*L',

        where

            P is a permutation matrix,
            L is the cholesky factor (P*L is lower triangular)
            E is a perturbation matrix so that C+E is positive definite
            D is diagonal
            D0 diagonal matrix so that C = L*D0*L'

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
    def __init__(self,C,verbose=True):
        
        # Determine whether the matrix is sparse
        self.set_sparsity(sp.issparse(C))
        self.set_degeneracy(False)
        
        # Compute the factorization
        self.decompose(C, verbose=verbose)          
      
    def set_sparsity(self, is_sparse):
        """
        Set the sparsity of the matrix
        """
        assert isinstance(is_sparse, bool), 'Input "is_sparse" should be a boolean.'
        self.__is_sparse = is_sparse

    def issparse(self):
        """
        Return True if the matrix is sparse
        """
        return self.__is_sparse 

    def set_degeneracy(self, is_degenerate):
        """
        Set the degeneracy of the matrix
        """
        assert isinstance(is_degenerate, bool), 'Input "is_degenerate" should be a boolean.'
        self.__is_degenerate = is_degenerate

    def isdegenerate(self):
        """
        Return True if the matrix is degenerate, i.e. not truly positive 
        definite.
        """
        return self.__is_degenerate

    def size(self):
        """
        Return the number of rows (=columns) of C
        """
        return self.__L.shape[0]
   
    def decompose(self,C,verbose=True):
        """
        Compute the Cholesky decomposition of the matrix C
        """
        if self.issparse():
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
                L = np.linalg.cholesky(C)

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
        if self.isdegenerate():
            #
            # Use modified Cholesky
            # 
            if self.issparse():
                #
                # Sparse, degenerate matrix - convert to full first :(
                # 
                C = C.toarray()

            # Compute modified Cholesky            
            L, D, P, D0 = self.modchol_ldlt(C)

            # 
            # Store Cholesky decomposition
            #
            self.set_factors((L, D, P, D0))  
            

    def modchol_ldlt(S,delta=None):
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
        assert np.allclose(S, S.T, atol=1e-12), \
        'Input "A" must be symmetric'    

        if delta is None:
            eps = np.finfo(float).eps
            delta = np.sqrt(eps)*linalg.norm(S, 'fro')
            #delta = 1e-5*linalg.norm(A, 'fro')
        else:
            assert delta>0, 'Input "delta" should be positive.'

        n = max(S.shape)

        L,D,p = linalg.ldl(S)  # @UndefinedVariable
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

        P = sp.diags([1],0,shape=(n,n), format='coo') 
        P.row = P.row[p]
        P = P.tocsr()
        
        #ld = np.diagonal(P.dot(L))
        #if any(np.abs(ld)<1e-15):
        #    print('L is singular')
            
        return L, DMC, P, D

    def set_factors(self,L):
        """
        Store the Cholesky factorization
        """
        self.__L = L
        
    def reconstruct(self):
        """
        Reconstruct the matrix from its Cholesky decomposition
        """
        if self.issparse():
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
        else:
            #
            # Full matrix
            # 
            L, D = self.__L, self.__D
            return L.dot(D.dot(L.T))
        
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
        elif self.issparse():
            #
            # Return sparse cholesky decomposition
            #            
            if verbose: print('Sparse Cholesky decomposition')
            return self.__L
        elif self.chol_type()=='full':
            #
            # Return the modified Cholesky decomposition
            # 
            return self.__L, self.__D, self.__P, self.__D0 
    
    def dot(self,b):
        """
        Compute the matrix vector product C*b
        
        Input:
        
            b: double, (n,m) compatible array
        
            
        Output:
        
            Kb: double, (n,m) matrix-vector product
        """
        assert b.shape[0]==self.size(), 'Input "b" has incompatible shape.'+\
        'Size K: {0}, Size b: {1}'.format(self.size(), b.shape)
        if sp.issparse(b):
            #
            # b is a sparse matrix
            #
            K = self.get_matrix()
            b = b.tocsc()
            return b.T.dot(K).T
        else:
            return self.get_matrix().dot(b)

    def solve(self,b):
        """
        Solve the system C*x = b  by successively solving 
        Ly = b for y and hence L^T x = y for x.
        
        Parameters:
            b (double, (n,m) array): The right-hand side of the system.
        
        Returns:
            The solution x of the system C*x = b.
        """
        if self.chol_type() == 'sparse':
            #
            # Use CHOLMOD
            #
            return self.__L(b)
        else:
            #
            # Use Modified Cholesky
            # 
            L, D, P, dummy = self.get_chol_decomp()
            PL = P.dot(L)
            y = linalg.solve_triangular(PL,P.dot(b),lower=True, unit_diagonal=True)
            Dinv = sp.diags(1./np.diagonal(D))
            z = Dinv.dot(y)
            x = linalg.solve_triangular(PL.T,z,lower=False,unit_diagonal=True)
            return P.T.dot(x)

    def sqrt_dot(self,b,transpose=False):
        """
        Returns R*b, where A = R*R'
        
        Parameters:
            b (double, compatible vector/matrix): The vector/matrix to be multiplied.
            transpose (bool, optional): If True, returns R'*b. If False, returns R*b. Defaults to False.
        
        Returns:
            The result of multiplying R with b.
        """
        assert self.__L is not None, \
            'Cholesky factor not computed.'\
            
        n = self.size()
        if self.chol_type()=='sparse':
            #
            # Sparse matrix, use CHOLMOD
            #

            # Build permutation matrix
            P = self.__L.P()
            I = sp.diags([1],0, shape=(n,n), format='csc')
            PP = I[P,:]
                    
            # Compute P'L
            L = self.__L.L()
            R = PP.T.dot(L)
            
            if transpose:
                #
                # R'*b
                # 
                return R.T.dot(b)
            else:
                #
                # R*b
                # 
                return R.dot(b)
        
        elif self.chol_type()=='full':
            #
            # Cholesky Factor stored as full matrix
            # 
            L,D = self.__L, self.__D
            sqrtD = sp.diags(np.sqrt(np.diagonal(D)))
            if transpose:
                #
                # R'b
                # 
                return sqrtD.dot(L.T.dot(b))
            else:
                #
                # Rb
                # 
                return L.dot(sqrtD.dot(b))

    def sqrt_solve(self,b,transpose=False):
        """
        Solve Sqrt(S)*x = b for x
       
        Return the solution x of Rx = b, where C = RR'
        
        Note: The 'L' in CHOLMOD's solve_L 
            is the one appearing in the factorization LDL' = PQP'. 
            We first rewrite it as Q = WW', where W = P'*L*sqrt(D)*P
        
        Parameters:
            b (double, compatible vector/matrix): The right-hand side of the system.
            transpose (bool, optional): If True, solves R'x = b. If False, solves Rx = b. Defaults to False.
        
        Returns:
            The solution x of the system Rx = b.
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



    

class EigenDecomposition(object):
    """
    Eigenvalue decomposition of a symmetric positive definite matrix
    """
    def __init__(self):
        pass

    def size(self):
        """
        Return the number of rows (=columns) of K
        """
        return self.__K.shape[0]
   
    def decompose(self,S):
        """
        Compute the eigendecomposition of the matrix S
        
        Compute the singular value decomposition USV' of M^{-1}K
        
        Parameters:
            delta (float, optional): A small positive constant to add to the diagonal of K before computing the eigendecomposition. Defaults to None.
        """ 
        K = self.__K
        if self.issparse():
            K = K.toarray()
            
        # Compute eigendecomposition
        d, V = linalg.eigh(K)
        
        # Rearrange to ensure decreasing order
        d = d[::-1]
        V = V[:,::-1]
        
        
        # Modify negative eigenvalues
        if delta is None:
            eps = np.finfo(float).eps
            delta = np.sqrt(eps)*linalg.norm(K, 'fro')
        d[d<=delta] = delta
        
        
        # Store eigendecomposition
        self.__V = V
        self.__d = d

    def reconstruct(self):
        """
        Reconstruct the matrix from its eigendecomposition
        """
        d, V = self.get_eig_decomp()
        return V.dot(np.diag(d).dot(V.T))
    
    def dot(self,b):
        """
        Compute the matrix vector product S*b
        """
        pass

    def solve(self,b, eps=None):
        """
        Solve the system Sx = b for x
   
        Solve the linear system Kx = Mb by means of eigenvalue decomposition, 
        i.e. x = V'Dinv*V*b 
        
        Inputs:
        
            b: double, (n,m) array
            
            tol: double >0, 
        """
        # Check that eigendecomposition has been computed
        assert self.__d is not None, \
        'First compute eigendecomposition using "compute_eig_decomp".'
        
        V = self.__V  # eigenvectors
        d = self.__d  # eigenvalues
        D_inv = diagonal_inverse(d, eps=eps)
        return V.dot(D_inv.dot(np.dot(V.T, b)))

    def sqrt(self,x, transpose=False):
        """
        Compute Sqrt(S)*x
        
        Compute Rx (or R'x), where A = RR'
        
        Inputs:
        
            x: double, (n,k) array
            
            transpose: bool, determine whether to compute Rx or R'x
            
        Output:
        
            b = Rx/R'x
        """
        d, V = self.__d, self.__V
        if transpose:
            # Sqrt(D)*V'x
            return np.diag(np.sqrt(d)).dot(V.T.dot(x))
        else:
            # V*Sqrt(D)*x
            return V.dot(np.diag(np.sqrt(d)).dot(x))

    def sqrt_solve(self,b,transpose=False):
        """
        Solve Sqrt(S)*x = b for x
       
        Solve the system Rx=b (or R'x=b if transpose) where R = V*sqrt(D) in 
        the decomposition M^{-1}K = VDV' = RR' 
        
        Inputs:
        
            b: double, (n,k)  right hand side
            
            transpose: bool [False], specifies whether system or transpose is 
                to be solved.
        """
        V = self.__V  # eigenvectors
        d = self.__d  # eigenvalues
        sqrtD_inv = diagonal_inverse(np.sqrt(d))
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

    def compute_nullspace(self, tol=1e-13):
        """
        Determines an othornormal set of vectors spanning the nullspace
        """
        if not self.has_eig_decomp():
            #
            # Compute the eigendecomposition if necessary
            # 
            self.compute_eig_decomp()
            
        # Determine what eigenvalues are below tolerance
        d = self.__d
        ix = (np.abs(d)<tol) 
        
        self.__ix_nullspace = ix