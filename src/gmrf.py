'''
Created on Feb 8, 2017

@author: hans-werner
'''

# Internal
from assembler import Assembler
from assembler import Kernel
from assembler import IIForm
from assembler import Form
from assembler import IPForm
from assembler import GaussRule

from fem import Element
from fem import DofHandler
from fem import Basis

from function import Map
from function import Nodal
from function import Explicit
from function import Constant

from mesh import Mesh1D
from mesh import QuadMesh

# Builtins 
from numbers import Number, Real
import numpy as np
from scipy import linalg
from scipy.special import kv, gamma
import scipy.sparse as sp
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt





def modchol_ldlt(A,delta=None):
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
    assert np.allclose(A, A.T, atol=1e-12), \
    'Input "A" must be symmetric'    

    if delta is None:
        eps = np.finfo(float).eps
        delta = np.sqrt(eps)*linalg.norm(A, 'fro')
        #delta = 1e-5*linalg.norm(A, 'fro')
    else:
        assert delta>0, 'Input "delta" should be positive.'

    n = max(A.shape)

    L,D,p = linalg.ldl(A)  # @UndefinedVariable
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
    
    
def diagonal_inverse(d, eps=None):
    """
    Compute the (approximate) pseudo-inverse of a diagonal matrix with
    diagonal entries d. 
    
    Inputs:
    
        d: double, (n, ) vector of diagonal entries
        
        eps: cut-off tolerance for zero entries
    """
    if eps is None:
        eps = np.finfo(float).eps
    else:
        assert eps > 0, 'Input "eps" should be positive.'
    
    if len(d.shape)==2:
        #
        # Matrix
        # 
        d = d.diagonal()
        
    d_inv = np.zeros(d.shape)
    i_nz = np.abs(d)>eps
    d_inv[i_nz] = 1/d[i_nz]
    D_inv = np.diag(d_inv)
    
    return D_inv


# =============================================================================
# Covariance Functions
# =============================================================================
"""
Commonly used covariance functions

For each function, we assume the input is given by two d-dimensional
vectors of length n. 
"""
def distance(x, y, M=None, periodic=False, box=None):
    """
    Compute the Euclidean distance vector between rows in x and rows in y
    
    Inputs: 
    
        x,y: two (n,dim) arrays
        
        M: double, positive semidefinite anistropy coefficient 
        
        periodic: bool [False], indicates a toroidal domain
        
        box: double, tuple representing the bounding box, i.e. 
            1D: box = (x_min, x_max)
            2D: box = (x_min, x_max, y_min, y_max) 
            If periodic is True, then box should be specified.
        
    Outputs: 
    
        d: double, (n,1) vector ||x[i]-y[i]||_M of (M-weighted) 
            Euclidean distances
         
    """
    # Check wether x and y have the same dimensions 
    assert x.shape == y.shape, 'Vectors x and y have incompatible shapes.'
    dim = x.shape[1]
    
    if dim==1:
        #
        # 1D
        #
        # Periodicity
        if periodic:
            assert box is not None, \
            'If periodic, bounding box must be specified.'
            
            x_min, x_max = box
            w  = x_max - x_min
            dx = np.min(np.array([np.abs(x-y), w - np.abs(x-y)]),axis=0)
        else:
            dx = np.abs(x-y)
        # "Anisotropy"    
        if M is None:
            return dx
        else:
            assert isinstance(M, Real) and M>=0, \
            'For one dimensional covariance, input "M" '+\
            'is a positive number.'
            return np.sqrt(M)*dx
    elif dim==2:
        #
        # 2D
        #   
        dx = np.abs(x[:,0]-y[:,0])
        dy = np.abs(x[:,1]-y[:,1])
        if periodic:
            assert box is not None, \
            'If periodic, bounding box must be specified.'
            x_min, x_max, y_min, y_max = box
            dx = np.min(np.array([dx,(x_max-x_min)-dx]),axis=0)
            dy = np.min(np.array([dy,(y_max-y_min)-dy]),axis=0)
        
        if M is None:
            return np.sqrt(dx**2 + dy**2)
        else:
            assert all(np.linalg.eigvals(M)>=0) and \
                   np.allclose(M,M.transpose()),\
                   'M should be symmetric positive definite.'
            
            ddx = np.array([dx,dy])
            Mddx = np.dot(M, ddx).T
            return np.sqrt(np.sum(ddx.T*Mddx, axis=1))


def constant(x,y,sgm=1):
    """
    Constant covariance kernel
    
        C(x,y) = sgm
    
    Inputs: 
    
        x,y: double, two (n,d) arrays
        
        sgm: double >0, standard deviation
            
    Outputs:
    
        double, (n,) array of covariances  
    """
    assert x.shape == y.shape, \
    'Input arrays have incompatible shapes.'
    
    return sgm*np.ones(x.shape[0])

    
def linear(x,y,sgm=1, M=None):
    """
    Linear covariance
    
        C(x,y) = sgm^2 + <x,My>  (Euclidean inner product)
        
    Inputs: 
    
        x,y: double, (n,dim) np.array of points
        
        sgm: double >0, standard deviation
        
        M: double, positive definite anisotropy tensor 
     
    """
    dim = x.shape[1]
    if dim==1:
        #
        # 1D
        # 
        if M is None:
            sgm**2 + x*y
            return sgm**2 + x*y
        else:
            assert isinstance(M,Real), 'Input "M" should be a scalar.'
            return x*M*y
        
    elif dim==2:
        #
        # 2D
        #  
        if M is None:
            return sgm**2 + np.sum(x*y, axis=1)
        else:
            assert M.shape == (2,2), 'Input "M" should be a 2x2 matrix.'
            My = np.dot(M, y.T).T
            return sgm**2 + np.sum(x*My, axis=1)
    else: 
        raise Exception('Only 1D and 2D supported.')


def gaussian(x, y, sgm=1, l=1, M=None, periodic=False, box=None):
    """
    Squared exponential covariance function
    
        C(x,y) = exp(-|x-y|^2/(2l^2))
    
    """
    d = distance(x, y, M, periodic=periodic, box=box)
    return sgm**2*np.exp(-d**2/(2*l**2))


def exponential(x, y, sgm=1, l=0.1, M=None, periodic=False, box=None):
    """
    Exponential covariance function
    
        C(x,y) = exp(-|x-y|/l)
        
    Inputs: 
    
        x,y: np.array, spatial points
        
        l: range parameter
    """
    d = distance(x, y, M, periodic=periodic, box=box)
    return sgm**2*np.exp(-d/l)


def matern(x, y, sgm, nu, l, M=None, periodic=False, box=None):
    """
    Matern covariance function
    
    Inputs:
    
        x,y: np.array, spatial points
        
        sgm: variance
        
        nu: shape parameter (k times differentiable if nu > k)
        
        l: range parameter 
        
    Source: 
    """
    d = distance(x, y, M, periodic=periodic, box=box)
    K = sgm**2*2**(1-nu)/gamma(nu)*(np.sqrt(2*nu)*d/l)**nu*\
        kv(nu,np.sqrt(2*nu)*d/l)
    #
    # Modified Bessel function undefined at d=0, covariance should be 1
    #
    K[np.isnan(K)] = 1
    return K
    
    
def rational(x, y, a, M=None, periodic=False, box=None):
    """
    Rational covariance
    
        C(x,y) = 1/(1 + |x-y|^2)^a
         
    """
    d = distance(x, y, M, periodic=periodic, box=box)
    return (1/(1+d**2))**a   

'''
class CovKernel(Kernel):
    """
    Integral kernel
    
    TODO: It's better to define a class for covariance functions and integrate the Kernel
    into Covariance class (when assembling).
    """
    def __init__(self, name=None, parameters=None, dim=1, cov_fn=None):
        """
        Constructor
        
        Inputs:
        
            name: str, name of covariance kernel 
                'constant', 'linear', 'gaussian', 'exponential', 'matern', 
                or 'rational'
            
            parameters: dict, parameter name/value pairs (see functions for
                allowable parameters.
        
        """
        if cov_fn is None:
            assert name is not None, \
                'Covariance should either be specified '\
                ' explicitly or by a string.'
            #
            # Determine covariance kernel
            # 
            if name == 'constant':
                #
                # k(x,y) = sigma
                # 
                cov_fn = constant
            elif name == 'linear':
                #
                # k(x,y) = sigma + <x,My>
                # 
                cov_fn = linear
            elif name == 'gaussian':
                #
                # k(x,y) = sigma*exp(-0.5(|x-y|_M/l)^2)
                # 
                cov_fn = gaussian
            elif name == 'exponential':
                #
                # k(x,y) = sigma*exo(-0.5|x-y|_M/l)
                # 
                cov_fn = exponential
            elif name == 'matern':
                #
                # k(x,y) = 
                # 
                cov_fn = matern
            elif name == 'rational':
                #
                # k(x,y) = 1/(1 + |x-y|^2)^a
                # 
                cov_fn = rational
 
        # Store results
        k = Explicit(f=cov_fn, parameters=parameters, n_variables=2, dim=dim)
        Kernel.__init__(self, f=k)
'''        


class SPDMatrix(object):
    """
    Wrapper Class for semi-positive definite matrices
    """
    def __init__(self, K):
        """
        Constructor
             
        Inputs: 
        
            K: double, (n,n) symmetric positive semidefinite matrix
            
        """
        # Initialize Cholesky decomposition
        if isinstance(K, Factor):
            #
            # Cholesky factor already computed
            # 
            self.__L = K
            self.__chol_type = 'sparse'
            n = K.L().shape[0]
            self.__K = sp.identity(n)
            self.__K = self.chol_reconstruct()
        else:
            #
            # No Cholesky factor computed
            # 
            self.__L = None
            self.__K = K
        
        
        # Initialize eigendecomoposition
        self.__d = None
        self.__V = None
        
    
    def size(self):
        """
        Return the number of rows (=columns) of K
        
        Returns:
            The number of rows (=columns) of K.
        """
        return self.__K.shape[0]
    
    
    def rank(self):
        """
        Return the rank of the matrix
        
        Returns:
            The rank of the matrix.
        """
        if self.issparse():
            return 
        else:
            return np.linalg.matrix_rank(self.__K) 
    
    
    def issparse(self):
        """
        Return True if the matrix is sparse
        
        Returns:
            True if the matrix is sparse, False otherwise.
        """
        return sp.issparse(self.__K)

        
    def get_matrix(self):
        """
        Returns the underlying matrix
        
        Returns:
            The underlying matrix.
        """
        return self.__K
    
    
    def chol_decomp(self):
        """
        Compute the cholesky factorization C = LL', where C=M^{-1}K.
        
        Decompositions are grouped as follows: 
        
        Sparse      cholmod         
        Full        modchol_ldlt    
        
        
        The following quantities are stored:
        
        cholesky (full, non-degenerate): L, such that C = LL'
        
        cholmod (sparse): LDL' = PCP', where
            P: permutation matrix
            L: lower triangular matrix
            D: diagonal matrix
            
        modchol_ldlt (degenerate): factorization  P*(C + E)*P' = L*D*L', where
            P: permutation matrix
            L: cholesky factor (P*L = lower triangular) 
            D: diagonal matrix
            D0: diagonal matrix so that C = L*D0*L'
        """
        modified_cholesky = False
        if self.issparse():
            #
            # Sparse matrix
            # 
            try:
                #
                # Try Cholesky (will fail if not PD)
                #
                self.__L = cholesky(self.__K.tocsc(), 
                                    mode='supernodal')
                
                self.__chol_type = 'sparse'
                
            except CholmodNotPositiveDefiniteError:
                modified_cholesky = True
        else:
            #
            # Full Matrix 
            # 
            modified_cholesky = True
                
        if modified_cholesky:
            #
            # Use modified Cholesky
            # 
            if self.issparse():
                #
                # Sparse matrix - convert to full first :(
                # 
                L, D, P, D0 = modchol_ldlt(self.__K.toarray())
            else:
                #
                # Full matrix
                # 
                L, D, P, D0 = modchol_ldlt(self.__K)
            # 
            # Store Cholesky decomposition
            #  
            self.__L = L
            self.__D = D
            self.__P = P 
            self.__D0 = D0
            self.__chol_type = 'full'
                
        
    def chol_type(self):
        """
        Returns the type of Cholesky decomposition 
        (sparse_cholesky/full_cholesky)
        
        Returns:
            The type of Cholesky decomposition.
        """
        return self.__chol_type


    def has_chol_decomp(self):
        """
        Returns True is the Cholesky decomposition has been computed.
        
        Returns:
            True if the Cholesky decomposition has been computed, False otherwise.
        """
        if self.__L is None:
            return False
        else:
            return True
        
    
    def get_chol_decomp(self):
        """
        Returns the Cholesky decomposition of the matrix M^{-1}K
        
        Returns:
            The Cholesky decomposition of the matrix M^{-1}K.
        """
        if self.__L is None:
            #
            # Return None if Cholesky decomposition not computed
            # 
            return None 
        elif self.chol_type()=='sparse':
            #
            # Return sparse cholesky decomposition
            #            
            return self.__L
        elif self.chol_type()=='full':
            #
            # Return the modified Cholesky decomposition
            # 
            return self.__L, self.__D, self.__P, self.__D0 
        
    
    def chol_reconstruct(self):
        """
        Reconstructs the (modified) matrix K
        
        Returns:
            The reconstructed (modified) matrix K.
        """
        
        if self.issparse():
            n = self.size()
            #
            # Sparse
            # 
            f = self.get_chol_decomp()

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
            
    
    def chol_solve(self, b):
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
        
        
    
    def chol_sqrt(self, b, transpose=False):
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
        

    def chol_sqrt_solve(self, b, transpose=False):
        """
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
                
    
    def compute_eig_decomp(self, delta=None):
        """
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
    
    
    def set_eig_decomp(self, d, V):
        """
        Store an existing eigendecomposition of the matrix
        
        Inputs:
        
            d: double, (n,) vector of eigenvalues
            
            V: double, (n,n) array of orthonormal eigenvectors
            
        TODO: Add checks
        """
        self.__d = d
        self.__V = V
        
    
    def eig_reconstruct(self):
        """
        Reconstruct the (modified) matrix from its eigendecomposition
        """
        d, V = self.get_eig_decomp()
        return V.dot(np.diag(d).dot(V.T))
    
    
    def has_eig_decomp(self):
        """
        Returns True if the eigendecomposition of the matrix has been computed
        """
        if self.__d is None:
            return False
        else:
            return True
        
    
    def get_eig_decomp(self):
        """
        Returns the matrix's eigenvalues and vectors
        """
        # Check that eigendecomposition has been computed
        if self.__d is None:
            self.compute_eig_decomp()
        
        """    
        assert self.__d is not None, \
        'First compute eigendecomposition using "compute_eig_decomp".'
        """
        return self.__d, self.__V
        
       
    def eig_solve(self, b, eps=None):
        """
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
            
    
    def eig_sqrt(self, x, transpose=False):
        """
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
    
    
    def eig_sqrt_solve(self, b, transpose=False):
        """
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


    def dot(self, b):
        """
        Returns the matrix vector product K*b
        
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


    def solve(self, b, decomposition='eig'):
        """
        Solve the system Kx = b for x, using the specified decomposition 
        """
        if decomposition == 'chol':
            #
            # Solve using Cholesky decomposition
            # 
            return self.chol_solve(b)
        elif decomposition == 'eig':
            #
            # Solve using Eigendecomposition
            # 
            return self.eig_solve(b)
            
            
    def sqrt(self, x, transpose=False, decomposition='eig'):
        """
        Compute Rx (or R'x), where A = RR'
        """
        if decomposition=='chol':
            #
            # Cholesky decomposition
            # 
            return self.chol_sqrt(x, transpose=transpose)
        elif decomposition=='eig':
            #
            # Eigendecomposition
            # 
            return self.eig_sqrt(x, transpose=transpose)
    
    
    def sqrt_solve(self, b, transpose=False, decomposition='eig'):
        """
        Solve Rx = b (or R'x = b) where A = RR'
        """ 
        if decomposition=='chol':
            #
            # Cholesky decomposition
            # 
            return self.chol_sqrt_solve(b, transpose=transpose)
        elif decomposition=='eig':
            #
            # Eigendecomposition
            # 
            return self.eig_sqrt_solve(b, transpose=transpose)
        
        
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
            
            
    def get_nullspace(self):
        """
        Returns the nullspace of the matrix 
        """
        V = self.__V
        return V[:,self.__ix_nullspace]


class Covariance(SPDMatrix):
    """
    Discretized covariance operator
    
    TODO: Class to (i) incorporate CovKernel
    TODO: Initialize Covariance matrix based on an SPDMatrix
    """
    def __init__(self, dofhandler, discretization='interpolation', 
                 subforest_flag=None, name=None, parameters={}, cov_fn=None):
        """
        Constructor
        
        Inputs:
        
            dofhandler: DofHandler, specifying the space over which to assemble
                the covariance operator.
                
            method: str, method used to approximate the kernel
                (['interpolation'], 'collocation', 'galerkin')
            
                'interpolation': Covariance kernel k(x,y) is approximated by
                
                        kh(x,y) = sum_i sum_j k_ij phi_i(x) phi_j(y),
                    
                    so that the Fredholm equation Cu = lmd u becomes
                
                        MKM*V = M*Lmd*V.
                    
                    
                'collocation': Covariance operator C is approximated by
                
                        Ch u(x) = sum_i (int_D k(x_i,y) u(y) dy) phi_i(x)
                    
                    and Ch psi_j(x) = lmd*psi_j(x) is collocated at vertices 
                    to get
                
                        Kh V = Lmd*V 
                    
                    
                'galerkin': Covariance operator C is projected onto subspace
                    so that the Fredholm equation becomes 
                        
                        B*V = M*Lmd*V, 
                        
                    where 
                        
                        B_ij = int_D int_D phi_i(x) phi_j(y) k(x,y) dx dy 
                    
                Notes: 
                
                    -'interpolation' is 'galerkin' with an approximate kernel.
                    
                    -Both 'interpolation' and 'galerkin' give rise to 
                        orthogonal psi_i's, but not v's. 
            
            subforest_flag: str, submesh indicator
        
            name: str, name of predefined covariance kernel. 
                
                Supported kernels: 'constant', 'linear', 'gaussian', 
                    'exponential', 'matern', 'rational'
            
                Alternatively, the covariance function can be specified
                    directly using cov_fn.
                
            parameters: dict, parameter name/value pairs (see functions for
                allowable parameters.
                
            dim: int, dimension of the underlying physical domain
            
            cov_fn: Map, function used to define the covariance kernel
        """
        #
        # Store parameters
        # 
        self.__subforest_flag = subforest_flag
        dofhandler.distribute_dofs(subforest_flag=subforest_flag)
        dofhandler.set_dof_vertices(subforest_flag=subforest_flag)
        self.__dofhandler = dofhandler
        self.__discretization = discretization
        self.__dim = dofhandler.mesh.dim()
        
        #
        # Define covariance kernel
        # 
        self.set_kernel(name, parameters, cov_fn)
        
        #
        # Assemble discretized covariance
        # 
        self.assemble()
        
        
    def set_kernel(self, name, parameters, cov_fn):
        """
        Set covariance kernel
        
        Inputs:
        
            name: str, name of covariance kernel 
                'constant', 'linear', 'gaussian', 'exponential', 'matern', 
                or 'rational'
            
            parameters: dict, parameter name/value pairs (see functions for
                allowable parameters.
                
            cov_fn: Map, explicit function defining covariance kernel
        """
        if cov_fn is None:
            assert name is not None, \
                'Covariance should either be specified '\
                ' explicitly or by a string.'
            #
            # Determine covariance kernel
            # 
            if name == 'constant':
                #
                # k(x,y) = sigma
                # 
                cov_fn = constant
            elif name == 'linear':
                #
                # k(x,y) = sigma + <x,My>
                # 
                cov_fn = linear
            elif name == 'gaussian':
                #
                # k(x,y) = sigma*exp(-0.5(|x-y|_M/l)^2)
                # 
                cov_fn = gaussian
            elif name == 'exponential':
                #
                # k(x,y) = sigma*exp(-0.5|x-y|_M/l)
                # 
                cov_fn = exponential
            elif name == 'matern':
                #
                # k(x,y) = 
                # 
                cov_fn = matern
            elif name == 'rational':
                #
                # k(x,y) = 1/(1 + |x-y|^2)^a
                # 
                cov_fn = rational
 
        # Store results
        dim = self.dofhandler().mesh.dim()
        k = Explicit(f=cov_fn, parameters=parameters, n_variables=2, dim=dim)
        self.__kernel = Kernel(f=k)
 
        
    def kernel(self):
        """
        Return covariance kernel
        """
        return self.__kernel
            
        
    def assemble(self):
        """
        Assemble Covariance matrix
        """
        # Dofhandler
        dofhandler = self.dofhandler()
        
        # Submesh indicator
        sf = self.subforest_flag()

        # Size
        n_dofs = dofhandler.n_dofs(subforest_flag=sf)
                
        # Mesh 
        mesh = dofhandler.mesh
        
        # Basis
        u = Basis(dofhandler, 'u')
        
        # Mass matrix 
        m = Form(trial=u, test=u)
        
        # Kernel
        k = self.kernel()
        
        print(self.discretization())
        
        #
        # Assemble and decompose covariance operator
        # 
        if self.discretization()=='collocation':
        
            # Collocate integral kernel (not symmetric)
            c = IIForm(kernel=k, test=u, trial=u)
            assembler = Assembler([[c]], mesh, subforest_flag=sf)
            assembler.assemble()
            C = assembler.af[0]['bilinear'].get_matrix().toarray()
            
            # Compute eigendecomposition
            lmd, V = linalg.eig(C)
                   
        elif self.discretization()=='galerkin':
            
            # L2 projection (Galerkin method)
            c = IPForm(kernel=k, test=u, trial=u)
            assembler = Assembler([[m],[c]], mesh, subforest_flag=sf)
            assembler.assemble()
            C = assembler.get_matrix(1).toarray()
            M = assembler.get_matrix(0).toarray()
            
            # Generalized eigen-decomposition
            lmd, V = linalg.eigh(C,M)
            
               
        elif self.discretization()=='interpolation':
            
            #
            # Interpolate covariance kernel at dof vertices 
            # 
            
            x = dofhandler.get_dof_vertices(subforest_flag=sf)
            dim = dofhandler.mesh.dim()
            I,J = np.mgrid[0:n_dofs,0:n_dofs] 
            X = x[I,:].reshape((n_dofs**2,dim)) 
            Y = x[J,:].reshape((n_dofs**2,dim))
            K = k.eval((X,Y)).reshape((n_dofs,n_dofs))
                        
            # Assemble mass matrix
            assembler = Assembler([[m]], mesh, subforest_flag=sf)
            assembler.assemble()
            M = assembler.get_matrix(i_problem=0).toarray()
            
            # Define discretized covariance operator
            C = M.dot(K.dot(M.T))
            
            # Compute generalized eigendecomposition
            lmd, V = linalg.eigh(C,M)
         
        else:
            raise Exception('Only "interpolation", "galerkin", '+\
                            ' or "collocation" supported for input "method"')
    
    
        #
        # Construct covariance matrix using eigendecomposition
        #
        
        # Rearrange to ensure decreasing order
        lmd = lmd[::-1]
        V = V[:,::-1]
        covariance = V.dot(np.diag(lmd).dot(V.T))
        
        # 
        # Initialize as SPDMatrix  
        #
        SPDMatrix.__init__(self, covariance)
        
    
    def dim(self):
        """
        Return the dimension of the computational domain
        """
        return self.__dim
    
    
    def discretization(self):
        """
        Return the discretization scheme for the covariance operator
        """
        return self.__discretization
        

    def dofhandler(self):
        """
        Return dofhandler
        """
        return self.__dofhandler
    

    def subforest_flag(self):
        """
        Return the submesh flag
        """ 
        return self.__subforest_flag


class GaussianField(object):
    """
    Base class for Gaussian random fields
    """
    def __init__(self, size, mean=None, K=None, mode='covariance', 
                 support=None): 
        """
        Constructor
        
        Inputs:
        
            size: int, the size n of the Gaussian random vector.  
            
            mean: double, (n,1) numpy array representing the expectation. 
            
            b: double, (n,1) numpy array representing Q*mean.
            
            support: double, (n,k) numpy array whose columns form an 
                orthonormal basis for the support of the Gaussian vector.
            
            precision: double, (n,n) sparse/full precision matrix.
                    
            covariance: double, (n,n) sparse/full covariance matrix.
             
        NOTE: If the precision/covariance have a preferred decomposition, 
            decompose before defining the Gaussian field.
        """               
        # Store size
        self.__size = size
        
        # Set supporting subspace
        self.set_support(support)
        
        # Store covariance/precision
        self.set_dependence(K, mode=mode)
                 
        # Store mean 
        self.set_mean(mean)
            
            
    def set_mean(self, mean):
        """
        Store the mean
        
        Inputs:
        
            mean: double, (n,n_sample) mean (array)
        """
        if mean is None:
            #
            # Default mean is zero
            #
            mean = np.zeros((self.size(),1))
        else:
            #
            # non-trivial location vector 
            #
            assert isinstance(mean, np.ndarray)
            assert mean.shape[0]==self.size()
            
        self.__mean = mean
        self.__b = None
            
    
    def set_b(self):
        """
        Compute the convenience parameter b = precision*mu
        
        What about degenerate matrices? 
        """
        Q = self.precision()
        if Q is not None: 
            b = Q.dot(self.mean())
        else:
            K = self.covariance()
            b = K.solve(self.mean())
        self.__b = b
            
        
    def set_dependence(self, K, mode='covariance'):
        """
        Store the proper covariance matrix of the random field, i.e. the 
        covariance of the non-constant component of the random field. The 
        actual covariance is given by V*K*V^T  
        
        Inputs:
        
            covariance: double, (n,n) numpy array
        """
        V = self.support()
        if V is not None:
            #
            # Support on a restricted subspace
            # 
            K = V.T.dot(K.dot(V))
        
        #
        # Store as SPDMatrix
        # 
        if mode=='covariance':
            if isinstance(K, SPDMatrix):
                covariance = K
            else:
                covariance = SPDMatrix(K)
            precision = None
        elif mode=='precision':
            if isinstance(K,SPDMatrix):
                precision = K
            else:
                precision = SPDMatrix(K)
            covariance = None
        
        # Store
        self.__precision = precision
        self.__covariance = covariance
        
 
    def set_support(self, support):
        """
        Stores the support of the Gaussian field
        
        Input:
        
            support: double, (n,k) array whose columns form an orthonormal
                basis for the subspace in which the Gaussian field is not 
                constant.
        """
        if support is not None:
            # Check shape
            n,k = support.shape
            assert n==self.size(), 'Support subspace should have the same '+\
                'number of rows as the random vector.'
            assert k<=n, 'Number of columns in "support" cannot exceed '+\
                'number of rows.'
            
            # Check orthogonality
            I = np.identity(k)
            assert np.allclose(support.T.dot(support),I), 'Basis vectors '+\
                'support should be orthonormal.'
            
        # Store support vectors 
        self.__support = support
    
    
    def update_support(self, mode='covariance', tol=1e-12):
        """
        Updates the support subspace, based on the support of the projected
        covariance/precision matrix. 
        
        Inputs:
        
            mode: str, specifying matrix from which support is to be computed
                ('covariance'), or 'precision.
                
            tol: double>0, cut-off tolerance below which eigenvalue is 
                considered 0. 
        """
        
        if mode=='covariance':
            #
            # Use (reduced) covariance matrix
            # 
            cov = self.covariance()
            assert cov is not None, 'No covariance specified.'
            if not cov.has_eig_decomp():
                cov.compute_eig_decomp(delta=0)
            d, V = cov.get_eig_decomp()
        elif mode=='precision':
            #
            # Use (reduced) precision matrix
            # 
            prec = self.precision()
            assert prec is not None, 'No precision specified'
            if not prec.has_eig_decomp():
                prec.compute_eig_decomp(delta=0)
            d, V = prec.compute_eig_decomp()
        else:
            raise Exception('Input "mode" should be "covariance" or "precision"')
        
        # Determine non-zero eigenvalues 
        i_support = np.abs(d)>tol  
        m = np.sum(i_support)
        
        # Compute new (diagonal) covariance/precision on reduced subspace
        D = sp.spdiags(d[i_support], 0, m, m) 
        
        # Update precision/covariance matrix
        if mode=='covariance':
            self.__covariance = SPDMatrix(D)
        elif mode=='precision':
            self.__precision = SPDMatrix(D)
        
        # Define new set of support vectors
        W = self.support()
        if W is not None: 
            self.set_support(W.dot(V[:,i_support]))
        else:
            self.set_support(V[:,i_support])
     
     
    def project(self,b,space='range'):
        """
        Project the array b onto either the range of the covariance or its 
        nullspace.
        
        Inputs:
        
            b: (n,k) numpy array
            
            space: str, 'nullspace' or 'range'
        """
        V = self.support()
        Pb = V.dot(V.T.dot(b))
        if space=='range':
            return Pb
        elif space=='nullspace':
            return b - Pb 
        
        
    def size(self):
        """
        Returns the size of the random vector
        """ 
        return self.__size
        
    
    def mean(self, col=0, n_copies=None):
        """
        Return the mean of the random vector
        
        Inputs:
            
            col: int, column of mean vector to be used (default=0).
            
            n_copies: int, number of copies of the mean
            
        Output: 
        
            mu: (n,n_copies) mean
        """
        mu = self.__mean[:,col][:,None]
        if n_copies is not None:
            assert type(n_copies) is np.int, \
                'Number of copies should be an integer.'
            if n_copies == 1:
                return mu 
            else:
                return np.tile(mu, (1, n_copies))
        else:
            return mu
    
      
    def b(self):
        """
        Returns the vector of central tendency in canonical form
        """
        if self.__b is None:
            self.set_b()
        return self.__b


    def covariance(self):
        """
        Returns the covariance of the random field
        """
        return self.__covariance
    
        
    def precision(self):
        """
        Returns the precision of the random field
        """
        return self.__precision

        
    def support(self):
        """
        Returns a matrix of orthonormal vectors constituting the nullspace of
        the field's covariance/precision matrix.
        
        Input:
        
            compute: bool, compute the support if necessary
        """
        return self.__support 
        
    def truncate(self, level):
        """
        Description
        -----------
        Returns the truncated Karhunen Loeve expansion of the Gaussian field
        based on the existing covariance operator. 
        
        Parameters
        ----------
        level : int, 
            The truncation level for the Karhunen-Loeve expansion. 
            
        
        Returns
        -------
        tht : GaussianField, 
            Truncated Gaussian field defined in terms of the mean and truncated
            covariance of the given field.
        
        """
        # Check that the level is not too large
        assert level <= self.size(), \
            'The truncation level should be less than the fields dimension.'
            
        # Extract the mean
        mean = self.mean()
        
        # Truncate the eigen-decomposition
        d, V = self.covariance().get_eig_decomp()
        dk = d[:level]
        Vk = V[:,:level]
        
        # Define truncated Covariance matrix
        K = SPDMatrix(Vk.dot(np.diag(dk)).dot(Vk.T))
        K.set_eig_decomp(dk, Vk)
        
        # Return Gaussian field with truncated covariance
        tht = GaussianField(self.size(), mean=mean, K=K)
        tht.set_support(Vk)
        
        return tht
        
    
    
    def sample(self, n_samples=1, z=None, m_col=0,
               mode='covariance', decomposition='eig'):
        """
        Generate sample realizations from Gaussian random field.
        
        Inputs:
        
            n_samples: int, number of samples to generate
            
            z: (n,n_samples) random vector ~N(0,I).
            
            m_col: int, column of mean array (for multiple GMRFs)
            
            mode: str, specify parameters used to simulate random field
                ['precision', 'covariance', 'canonical']
                
            decomposition: str, specifying the decomposition type
                ['chol', 'eig']
                
              
        Outputs:
        
            x: (n,n_samples), samples paths of random field
            
                
        Note: Samples generated from the cholesky decomposition of Q are 
            different from those generated from that of eig. 
                
                Q = LL' (lower*upper)
                  
            =>  S = Q^(-1) = L'^(-1) L^(-1) (upper*lower)
            
            However, they have  the same distribution
        """         
        #
        # Retrieve SPDMatrix   
        #  
        if mode=='covariance':
            # Check that covariance matrix is specified 
            assert self.covariance() is not None, 'No covariance specified.'
            
            # Get covariance
            K = self.covariance()
            
        elif mode=='precision' or mode=='canonical':
            # Check that precision matrix is specified
            assert self.precision() is not None, 'No precision specified.'
            
            # Get precision
            K = self.precision()
            
        #    
        # Compute factorization L*L' if necessary
        #
        if decomposition=='chol' and not K.has_chol_decomp():
            #
            # Cholesky
            #
            K.chol_decomp()
        elif decomposition=='eig' and not K.has_eig_decomp():
            #
            # Eigendecomposition
            #
            K.compute_eig_decomp(delta=0)
        
        #
        # Parse samples
        # 
        if z is not None:
            #
            # Extract number of samples from z
            #  
            assert len(z.shape) == 2, \
                'Input "z" should have size (n, n_samples).'
            if z.shape[0] > K.size():
                z = z[:K.size(),:]
            
            assert z.shape[0] <= K.size(), \
                'Input "z" should have size (n, n_samples).'
            n_samples = z.shape[1]
        else:
            #
            # Generate z
            # 
            z = np.random.normal(size=(K.size(), n_samples))
        #
        #  Compute sample   
        # 
        if mode=='covariance':
            #
            # Return Lz + mean
            #
            Lz = K.sqrt(z, decomposition=decomposition)
            V = self.support()
            if V is not None:
                Lz = V.dot(Lz)
            return  Lz + self.mean(col=m_col, n_copies=n_samples)
                
        elif mode=='precision':
            #
            # Return L^{-T} z + mean
            #
            Lz = K.sqrt_solve(z,transpose=True, decomposition=decomposition)
            V = self.support()
            if V is not None:
                Lz = V.dot(Lz)  
            mu = self.mean(col=m_col, n_copies=n_samples)
            return Lz + mu
                       
        elif mode=='canonical':
            #
            # Cholesky decomposition of the precision matrix 
            #
            assert self.precision() is not None, 'No precision specified.'
            return self.sample(n_samples=n_samples, z=z, mode='precision', \
                               decomposition=decomposition, m_col=m_col)
        else:
            raise Exception('Input "mode" not recognized. '+\
                            'Use "covariance", "precision", or "canonical".')
            
    
    
    def condition(self, A, e, Ko=0, output='sample', n_samples=1, z=None, 
                  mode='covariance', decomposition='eig'):
        """
        Returns the conditional random field X|e, where e|X ~ N(AX, Ko).
        
            - (Hard Constraint) If Ko=0, then e|X = AX, i.e. AX = e and 
                the conditional mean and covariance are given by
                
                mu_{x|Ax=e} = mu - K*A^T*(A*K*A^T)^{-1}(A*mu-e)
                
                K_{x|Ax=e} = K - K*A^T*(A*K*A^T)^{-1}*A*K
            
            - Otherwise, the conditional mean and precision are given by
            
                     mu_{x|e} = Q*mu + A^T Ko\e
                     Q_{x|e}  = Q + A^T Ko\A.
            
        The resulting field has support on a reduced vector space.  
        """
        #
        # Measure of dependence
        # 
        if mode=='covariance':
            K = self.covariance()
            assert K is not None, 'No covariance specified.'
        elif mode=='precision':
            Q = self.precision()
            assert Q is not None, 'No precision specified.'
        
        # Mean
        mu = self.mean()  
                
        # Support
        Vk = self.support()
        
        #
        # Convert pointwise restrictions to sparse matrix
        #
        n = self.size()
        if len(A.shape)==1:
            k = len(A)
            rows = np.arange(k)
            cols = A
            vals = np.ones(k)
            A = sp.coo_matrix((vals, (rows,cols)),shape=(k,n))
             
        #
        # Determine whether constraints are hard or soft
        # 
        if isinstance(Ko, Real) and Ko==0:
            #
            # Hard constraints
            #
                          
            if Vk is not None:
                #
                # Reduce to active subspace 
                #
                
                # Compute reduced map Ak = A*Vk
                Ak = A.dot(Vk)
                
                # Component of mean onto support    
                mu_k = Vk.T.dot(mu)
                
                # Compute ek = e - P^*mu
                ek = e - A.dot(self.project(mu,'nullspace'))
            else:
                Ak = A.dot(Vk)
                mu_k = mu
                ek = e
                
            #
            # Compute K*A.T and A*K*A.T
            #  
            print('Computing KAT and AKAT') 
            if mode=='covariance':    
                KAT  = K.dot(Ak.T)
                AKAT = Ak.dot(KAT)
            elif mode=='precision':
                KAT = Q.solve(Ak.T)
                AKAT = Ak.dot(KAT)

            
            if output=='sample':
                #
                # Return Kriged Sample
                # 
                
                # Sample unconditioned field 
                print('Sampling from unconditioned field')
                Xs = self.sample(z=z, n_samples=n_samples, mode=mode, 
                                 decomposition=decomposition)            
                
                # Compute residual
                print('Computing residual')
                r = A.dot(Xs)-e
                
                # Conditional covariance 
                print('Computing conditional covariance')
                U = linalg.solve(AKAT,r)
                
                # Apply correction 
                print('Applying correction')
                X = Xs - Vk.dot(KAT.dot(U))
                
                return X
            
            elif output=='field':
                #
                # Return GaussianField
                # 
                
                # Conditional mean
                r = Ak.dot(mu_k)-ek
                mu_cnd = mu - Vk.dot(KAT.dot(linalg.solve(AKAT,r)))
                
                # Conditional covariance
                U = linalg.solve(AKAT, KAT.T)
                K_cnd = K.get_matrix() - KAT.dot(U)
                K_cnd = Vk.dot(K_cnd.dot(Vk.T))
                
                # Define random field 
                X = GaussianField(self.size(), mean=mu_cnd, \
                                  K=K_cnd,mode='covariance')
                X.update_support()
                
                return X
                
            else:
                raise Exception('Input "mode" should be "sample" or "field".')
            
        else:
            #
            # Soft constraint
            # 
            
            #
            # Compute K*A.T and A*K*A.T
            #   
            if mode=='covariance':    
                KAT  = K.dot(A.T)
                AKAT = A.dot(KAT)
            elif mode=='precision':
                KAT = Q.solve(A.T)
                AKAT = A.dot(KAT)
                
            if output=='sample':
                #
                # Return Kriged Sample            
                #

                # Sample unconditioned field 
                Xs = self.sample(z=z, n_samples=n_samples, mode=mode, 
                                 decomposition=decomposition)
                
                # Sample e|Ax
                eps = GaussianField(A.shape[0], mean=e, K=Ko, mode='covariance')
                e = eps.sample(z=z, n_samples=n_samples)
                
                # Compute residual
                r = A.dot(Xs)-e
                
                # Conditional covariance 
                U = linalg.solve(Ko+AKAT,r)
                
                # Apply correction 
                X = Xs - KAT.dot(U)
                
                return X
            
            elif output=='field':
                # Conditional mean
                mu_cnd = mu - A.T.dot(linalg.solve(Ko,e))
                
                # Conditional precision
                Q_cnd = Q + A.T.dot(linalg.solve(Ko,A))
                
                # Random field
                X = GaussianField(self.size(), mean=mu_cnd, K=Q_cnd, 
                                  mode='precision')
                
                return X
        """    
        #    
        # For hard constraints, check whether constraint is possible, reduce 
        # support 
        # 
        if not soft:
            #
            # A(P*x + P^*mu) = e must have a solution 
            # 
            tol = 1e-13
            
            
            
            
            
            # Solve Ay = r for y
            sgm_zero = s<tol
            assert linalg.norm(ek.T.dot(u[:,sgm_zero]))<tol, 'Projection onto '+\
                'nullspace non-zero.' 
                
                
        if output=='sample':
            # =================================================================
            # Return Kriged sample
            # =================================================================
            
        
             
        elif output=='field':
            # =================================================================
            # Return random field
            # =================================================================
            #
            # Conditional mean
            #
            mu = self.mean()
            if not soft:
                r = A.dot(mu) - ek
                mu_cnd = Vk.dot(mu_k - V.dot(linalg.solve(W,r)))
            else: 
                mu_cnd = mu - A.T.dot(linalg.solve(Ko,e))

            #
            # Support
            # 
            if not soft:
                #
                # New support = old supp intersect nullspace of A^T
                #
                
                # Orthonormal basis for range of A 
                AT_rng = vt[~sgm_zero,:].T
                
                # Project old support onto range of A^T (=orthogonal complement of
                # nullspace of A^T): what zeros out, we keep 
                R_spp = Vk - AT_rng.dot(AT_rng.T.dot(Vk))
                Q,R,dummy = linalg.qr(R_spp, pivoting='true', mode='economic')
                Rii = np.diag(R)
                print('Rii', Rii)
                i_spp = np.abs(Rii)>tol
                cnd_spp = Q[:,i_spp]   
                print('conditional support', cnd_spp)
            #
            # Conditional covariance/precision
            #
            if mode=='covariance':
                U = linalg.solve(Ko+AKAT, KAT.T) 
         
                # Conditional covariance
                K_cnd = K.get_matrix() - KAT.dot(U)
                K_cnd = Vk.dot(K_cnd.dot(Vk.T))
                
                if not soft:
                    # Define Gaussian field (hard constraint)
                    X = GaussianField(self.size(), mean=mu_cnd, \
                                      K=K_cnd,mode='covariance',\
                                      support=cnd_spp)
                else:
                    # Define Gaussian field (soft constraint)
                    X = GaussianField(self.size(), mean=mu_cnd, \
                                      K=K_cnd, mode='covariance')
                    
            elif mode=='precision':
                
                
                if not soft:
                    # Gaussian field (hard constraint)
                    X = GaussianField(mean=mu_cnd, precision=Q_cnd,\
                                      support=cnd_spp)
                else:
                    # Gaussian field (soft constraint)
                    X = GaussianField(mean=mu_cnd, precision=Q_cnd)
        else:
            raise Exception('Input "mode" should be "sample" or "field".')
        
        return X 
        
        """
        
    """            
                    
            elif mode=='covariance':
                K = self.covariance().get_matrix()
                V = K.dot(A.T)
                W = A.dot(V)
                U = linalg.solve(W,V.T)
                if output=='sample':
                    #
                    # Sample
                    # 
                    x = self.chol_sample(z=z, n_samples=n_samples, mode='covariance')
                    c = A.dot(x)-e
                    x = x - np.dot(U.T,c)
                    return x
                else:
                    #
                    # Random Field
                    # 
                    
                    # Conditional covariance
                    KK = K - V.dot(U) 
                    
                    # Conditional mean
                    m = self.mean()
                    c = A.dot(m)-e
                    mm = m - np.dot(U.T,c)
                    
                    return GaussianField(mean=mm, covariance=KK) 
        else:
            #
            # Soft Constraint
            # 
            KKo = SPDMatrix(Ko)
            mean_E = A.dot(self.mean())
            E = GaussianField(mean=mean_E, covariance=KKo)
            if mode=='precision':
                #
                # Precision 
                # 
                QQ = self.precision()
                if output=='sample':
                    #
                    # Generate sample from N(e,Ko) 
                    # 
                    ee = E.sample(n_samples=n_samples, z=z)
                    Qc = QQ.get_matrix() + A.T.dot(linalg.solve(Ko, A))
                    QQc = SPDMatrix(Qc)
                else:
                    #
                    # Store as Gaussian Field 
                    # 
                    bc = QQ.dot(self.mean()) + A.T.dot(Ko.solve(e))
                    Qc = QQ.get_matrix() + A.T.dot(Ko.solve(A))
                    
                    return GaussianField(b=bc, precision=Qc)
                
            elif mode=='covariance':
                #
                # Covariance
                #
                KK = self.covariance()
                if output=='sample':
                    pass
                else:
                    #
                    # Store as Gaussian Field
                    #  
                    bc = KK.solve(self.mean()) + A.T.dot(linalg.solve(Ko,e))
        """
        
    def chol_condition(self, A, e, Ko=0, output='sample', mode='precision', 
                       z=None, n_samples=1):
        """
        Computes the conditional covariance of X, given E ~ N(AX, Ko). 
        
        Inputs:
        
            A: double, (k,n) 
            
            Ko: double symm, covariance matrix of E.
            
            e: double, value
            
            output: str, type of output desired [sample/covariance]
            
            Z: double, array whose columns are iid N(0,1) random vectors 
                (ignored if output='gmrf')
            
            n_samples: int, number of samples (ignored if Z is not None)

        TODO: Soft constraints
        TODO: Test
        TODO: Delete
        """
        if Ko == 0:
            #
            # Hard Constraint
            #
            if mode=='precision':
                Q = self.precision()
                V = Q.chol_solve(A.T)
                W = A.dot(V)
                U = linalg.solve(W,V.T)
                if output=='sample':
                    #
                    # Sample
                    # 
                    x = self.chol_sample(z=z, n_samples=n_samples, mode='precision')                    
                    c = A.dot(x)-e
                    x = x - np.dot(U.T, c)
                    return x
                elif mode=='field':
                    #
                    # Random Field
                    #

                    # Conditional mean
                    m = self.mean()  
                    c = A.dot(m)-e
                    mm = m - np.dot(U.T,c)
                    
                    # Conditional covariance
                    K = Q.chol_solve(np.identity(self.size()))
                    KK = K - V.dot(U)
                    
                    # Return gaussian field
                    return GaussianField(mean=mm, covariance=KK)
                else:
                    raise Exception('Input "mode" should be "sample" or "gmrf".')
                    
            elif mode=='covariance':
                K = self.covariance().get_matrix()
                V = K.dot(A.T)
                W = A.dot(V)
                U = linalg.solve(W,V.T)
                if output=='sample':
                    #
                    # Sample
                    # 
                    x = self.sample(z=z, n_samples=n_samples, mode='covariance', 
                                    decomposition='chol')
                    c = A.dot(x)-e
                    x = x - np.dot(U.T,c)
                    return x
                else:
                    #
                    # Random Field
                    # 
                    
                    # Conditional covariance
                    KK = K - V.dot(U) 
                    
                    # Conditional mean
                    m = self.mean()
                    c = A.dot(m)-e
                    mm = m - np.dot(U.T,c)
                    
                    return GaussianField(mean=mm, covariance=KK) 
        else:
            #
            # Condition on Gaussian observations
            # 
            KKo = SPDMatrix(Ko)
            mean_E = A.dot(self.mean())
            E = GaussianField(mean=mean_E, covariance=KKo)
            if mode=='precision':
                #
                # Precision 
                # 
                QQ = self.precision()
                if output=='sample':
                    #
                    # Generate sample from N(e,Ko) 
                    # 
                    ee = E.chol_sample(n_samples=n_samples, z=z)
                    Qc = QQ.get_matrix() + A.T.dot(linalg.solve(Ko, A))
                    QQc = SPDMatrix(Qc)
                else:
                    #
                    #  
                    # 
                    pass
            elif mode=='covariance':
                #
                # Covariance
                #
                if output=='sample':
                    pass
                else:
                    pass

    
    def iid_gauss(self, n_samples=1):
        """
        Returns a matrix whose columns are N(0,I) vectors of length the 
        size of the covariance. 
        """
        V = self.support()
        if V is not None:
            n = V.shape[1]
        else:
            n = self.size()
            
        return np.random.normal(size=(n,n_samples)) 
      

class HaarField(GaussianField):
    """
    Multiresolution Gaussian random field parametrized by Haar wavelets over a nested mesh
    """
    def __init__(self):
        """
        """
        pass
    
    
'''    
class KLField(GaussianField):
    """
    Karhunen-Loeve expansion
    
        u(x) = mu(x) + sum_j sqrt(lmd_j)*psi_j(x)*Z_j, 
        
    where (lmd_j, psi_j(x)) are eigenpairs of approximations of a covariance
    operator C, defined by 
    
        Cu(x) = I_D k(x,y) u(y) dy 
            
    """
    def __init__(self, k, dofhandler, mean=None, method='interpolation', 
                 subforest_flag=None):
        """
        Constructor
        
        Inputs:
            
            k: Kernel, covariance kernel
            
            dofhandler: DofHandler, finite element dofhandler  
            
            mean: Nodal function representing the mean.
            
            method: str, method used to approximate the kernel
                (['interpolation'], 'collocation', 'galerkin')
            
                'interpolation': Covariance kernel k(x,y) is approximated by
                
                        kh(x,y) = sum_i sum_j k_ij phi_i(x) phi_j(y),
                    
                    so that the Fredholm equation Cu = lmd u becomes
                
                        MKM*V = M*Lmd*V.
                    
                    
                'collocation': Covariance operator C is approximated by
                
                        Ch u(x) = sum_i (int_D k(x_i,y) u(y) dy) phi_i(x)
                    
                    and Ch psi_j(x) = lmd*psi_j(x) is collocated at vertices 
                    to get
                
                        Kh V = Lmd*V 
                    
                    
                'galerkin': Covariance operator C is projected onto subspace
                    so that the Fredholm equation becomes 
                        
                        B*V = M*Lmd*V, 
                        
                    where 
                        
                        B_ij = int_D int_D phi_i(x) phi_j(y) k(x,y) dx dy 
                    
                Notes: 
                
                    -'interpolation' is 'galerkin' with an approximate kernel.
                    
                    -Both 'interpolation' and 'galerkin' give rise to 
                        orthogonal psi_i's, but not v's. 
            
            subforest_flag: str, submesh indicator
        """        
        #
        # Parse dofhandler
        #
        dofhandler.distribute_dofs(subforest_flag=subforest_flag)
        dofhandler.set_dof_vertices()
        self.__dofhandler = dofhandler
        n_dofs = dofhandler.n_dofs(subforest_flag=subforest_flag)
        
        #
        # Parse mean
        # 
        if mean is None:
            mean = Nodal(data=np.zeros((n_dofs,1)), dofhandler=dofhandler)
        else:
            assert isinstance(mean, Nodal), \
            'Input "mean" should be a "Nodal" object.'
        
            assert mean.dofhandler()==self.dofhandler(), \
            'Input "mean" should have the same dofhandler as the random field. '
        self.__mean_function = mean
        
        # Mesh 
        mesh = dofhandler.mesh
        
        # Basis
        u = Basis(dofhandler, 'u')
        
        # Mass matrix 
        m = Form(trial=u, test=u)
        
        # Store covariance kernel
        assert isinstance(k, Kernel), \
        'Input "cov_kernel" should be a Kernel object.'
        self.__kernel = k
        
        #
        # Assemble and decompose covariance operator
        # 
        if method=='collocation':
        
            # Collocate integral kernel (not symmetric)
            c = IIForm(kernel=k, test=u, trial=u)
            assembler = Assembler([[c]], mesh, subforest_flag=subforest_flag)
            assembler.assemble()
            C = assembler.af[0]['bilinear'].get_matrix().toarray()
            
            # Compute eigendecomposition
            lmd, V = linalg.eig(C)
                   
        elif method=='galerkin':
            
            # L2 projection (Galerkin method)
            c = IPForm(kernel=k, test=u, trial=u)
            assembler = Assembler([[m],[c]], mesh, subforest_flag=subforest_flag)
            assembler.assemble()
            C = assembler.af[1]['bilinear'].get_matrix().toarray()
            M = assembler.af[0]['bilinear'].get_matrix().toarray()
            
            # Generalized eigen-decomposition
            lmd, V = linalg.eigh(C,M)
            
               
        elif method=='interpolation':
            
            #
            # Interpolate covariance kernel at dof vertices 
            # 
            
            x = dofhandler.get_dof_vertices(subforest_flag=subforest_flag)
            dim = dofhandler.mesh.dim()
            I,J = np.mgrid[0:n_dofs,0:n_dofs] 
            X = x[I,:].reshape((n_dofs**2,dim)) 
            Y = x[J,:].reshape((n_dofs**2,dim))
            K = k.eval((X,Y)).reshape((n_dofs,n_dofs))
                        
            # Assemble mass matrix
            assembler = Assembler([[m]], mesh, subforest_flag=subforest_flag)
            assembler.assemble()
            M = assembler.af[0]['bilinear'].get_matrix().toarray()
            
            # Define discretized covariance operator
            C = M.dot(K.dot(M.T))
            
            # Compute generalized eigendecomposition
            lmd, V = linalg.eigh(C,M)
            
        else:
            raise Exception('Only "interpolation", "galerkin", '+\
                            ' or "collocation" supported for input "method"')
        
        #
        # Store eigendecomposition
        # 
        # Rearrange to ensure decreasing order
        lmd = lmd[::-1]
        V = V[:,::-1]
        self.__lmd = lmd                            # eigenvalues
        self.__V = V                                # eigenvectors
        
        #
        # Covariance matrix (X = mu + V*sqrt(Lmd)*Z
        # 
        covariance = V.dot(np.diag(lmd).dot(V.T))
        
        GaussianField.__init__(self,n_dofs, mean=mean.data(), 
                               K=covariance, mode='covariance')
        
        #
        # Store quantities
        # 
        self.__C = C                                # discretized covariance operator
        self.__discretization_method = method       # discretization method
        self.__assembler = assembler                # finite element assembler
        self.__subforest_flag = subforest_flag      # submesh flag
        self.__size = n_dofs                        # number of fem dofs

    
    def dim(self):
        """
        Return dimension of domain
        """
        return self.dofhandler().mesh.dim()
    
        
    def eigenvalues(self):
        """
        Return (generalized) eigenvalues of the discretized covariance operator 
        """
        return self.__lmd
    
    
    def eigenvectors(self):
        """
        Return (generalized) eigenvectors of the discretized covariance operator
        """
        return self.__V
    

    def mean_function(self):
        """
        Returns the mean function
        """
        return self.__mean_function
    
                  
    def discretization_method(self):
        """
        Returns the assembly/approximation method ('interpolation' or 'projection')
        """
        return self.__discretization_method
    
    
    def sample(self, n_samples=1, z=None):
        """
        """
        #
        # Parse samples
        # 
        if z is not None:
            #
            # Extract number of samples from z
            #  
            assert len(z.shape) == 2, \
                'Input "z" should have size (n, n_samples).'
            assert z.shape[0] == self.size(), \
                'Input "z" should have size (n, n_samples).'
            n_samples = z.shape[1]
        else:
            #
            # Generate z
            # 
            z = np.random.normal(size=(self.size(), n_samples))
    
        V = self.__V
        lmd = self.__lmd
        Lmd = np.diag(lmd)
        return V.dot(Lmd.dot(z))
        
'''

class EllipticField(GaussianField):
    """
    Elliptic Gaussian random field, defined as the solution of the elliptic 
    equation
    
        (k^2 u - div[T(x)grad(u)])^{gamma/2} X = W
    """
    def __init__(self, dofhandler, mean=None, gamma=1, kappa=None, tau=None, 
                 subforest_flag=None):
        """
        Constructor
            
        Inputs: 
        
            dofhandler: DofHandler, 
            
            gamma: int, positive integer (doubles not yet implemented).
            
            kappa: double, positive regularization parameter.
            
            tau: (Axx,Axy,Ayy) symmetric tensor or diffusion coefficient function.
                        
        """
        
        #
        # Parse dofhandler
        #
        dofhandler.distribute_dofs(subforest_flag=subforest_flag)
        dofhandler.set_dof_vertices()
        self.__dofhandler = dofhandler
        n_dofs = dofhandler.n_dofs(subforest_flag=subforest_flag)
        basis = Basis(dofhandler,'u')
        #
        # Parse mean
        # 
        if mean is None:
            mean = Nodal(data=np.zeros((n_dofs,1)), basis=basis)
        else:
            assert isinstance(mean, Nodal), \
            'Input "mean" should be a "Nodal" object.'
        
            assert mean.dofhandler()==self.dofhandler(), \
            'Input "mean" should have the same dofhandler as the random field. '
        self.__mean_function = mean
        
        dim = dofhandler.mesh.dim()
        
        #
        # Define basis
        # 
        ux = Basis(dofhandler, 'ux')
        uy = Basis(dofhandler, 'uy')
        u  = Basis(dofhandler, 'u')
        
        #
        # Define bilinear forms
        # 
        elliptic_forms = []
        if tau is not None:
            #
            # Test whether tau is a symmetric tensor
            # 
            if type(tau) is tuple:
                #
                # Tau is a tensor
                # 
                assert len(tau)==3, 'Symmetric tensor should have length 3.'
                dim == 2, 'Input "tau" cannot be a tuple when mesh dimension=1'
                axx, axy, ayy = tau
                Axx = Form(axx, trial=ux, test=ux)
                Axy = Form(axy, trial=ux, test=uy)
                Ayx = Form(axy, trial=uy, test=ux)
                Ayy = Form(ayy, trial=uy, test=uy)
                
                elliptic_forms = [Axx, Axy, Ayx, Ayy]
                
            else:
                #
                # tau is a function
                # 
                assert isinstance(tau, Map), 'Input "tau" should be a "Map".'
                
                if dim==1:
                    Ax = Form(tau, trial=ux, test=ux)
                    elliptic_forms = [Ax] 
                elif dim==2:
                    Ax = Form(tau, trial=ux, test=ux)
                    Ay = Form(tau, trial=uy, test=uy)
                    elliptic_forms = [Ax,Ay]
        else:
            #
            # Default tau=1
            # 
            if dim==1:
                Ax = Form(1, trial=ux, test=ux)
                elliptic_forms = [Ax]
            elif dim==2:
                Ax = Form(1, trial=ux, test=ux)
                Ay = Form(1, trial=uy, test=uy)
                elliptic_forms = [Ax, Ay]
        #
        # Regularization term
        # 
        elliptic_forms.append(Form(kappa, trial=u, test=u))
        
        #
        # Mass matrix 
        # 
        mass = [Form(1, trial=u, test=u)]
        
        #
        # Combine forms
        # 
        problems = [elliptic_forms, mass]
        
        #
        # Assemble matrices
        # 
        assembler = Assembler(problems, mesh=dofhandler.mesh)
        assembler.assemble()
        
        #
        # Get system matrix
        # 
        K = assembler.get_matrix(0)
        
        #
        # Lumped mass matrix
        # 
        M = assembler.get_matrix(1)
        m_lumped = np.array(M.sum(axis=1)).squeeze()
        
            
        if np.mod(gamma,2) == 0:
            #
            # Even power alpha
            # 
            Q = cholesky(K.tocsc())
            count = 1
        else:
            #
            # Odd power gamma
            # 
            Q = cholesky_AAt((K.dot(sp.diags(1/np.sqrt(m_lumped)))).tocsc())
            count = 2
        
        while count < gamma:
            #
            # Update Q
            #
            Q = cholesky_AAt((K.dot(sp.diags(1/m_lumped)).dot(Q.apply_Pt(Q.L()))).tocsc()) 
            count += 2
        # Save precision matrix
        GaussianField.__init__(self, n_dofs, mean=mean.data(), \
                               K=Q, mode='precision')
        
    '''    
    def sample(self, n_samples=1, z=None):
        """
        Generate sample realizations from an ellipt.
        
        Inputs:
        
            n_samples: int, number of samples to generate
            
            z: (n,n_samples) random vector ~N(0,I).
              
        Outputs:
        
            x: (n,n_samples), samples paths of random field
        
        """
        return GaussianField.sample(self, n_samples=n_samples, z=z, 
                                    mode='precision', decomposition='chol')
    '''    
            

    
            
         
'''   
class Covariance(object):
    """
    Covariance kernel for Gaussian random fields
    """        
    
    
            
    def __init__(self, name, parameters, mesh, element, n_gauss=9, 
                 assembly_type='projection', subforest_flag=None, lumped=False):
        """
        Construct a covariance matrix from the specified covariance kernel
        
        Inputs: 
        
            
            
            mesh: Mesh, object denoting physical mesh
            
            etype: str, finite element space (see Element for
                supported spaces).
                
            assembly_type: str, specifies type of approximation,
                projection, or collocation
                
            

        """
        
        self.__kernel = CovKernel(name, parameters)
        assert isinstance(element, Element), \
        'Input "element" must be of type Element.'
            
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        
        if assembly_type=='projection':
            #
            # Approximate covariance kernel by its projection
            #
            self.assemble_projection()
        elif assembly_type=='collocation':
            #
            # Approximate covariance kernel by collocation
            #
            self.assemble_collocation() 
        else:
            raise Exception('Use "projection" or "collocation" for'+\
                            ' input "assembly_type"')
        

    def assemble_projection(self):
        """
        Compute the discretization (C,M) of the covariance operator
        
        Ku(x) = I_D c(x,y) u(y) dy
        
        within a finite element projection framework. In particular, 
        compute the matrix pair (C,M), where 
        
            C = ((c(.,.)phi_i(x), phi_j(y))
            
            M = (phi_i(x), phi_j(x))
            
            So that K ~ M^{-1}C.
            
            
        Inputs:
        
            kernel: bivariate function, c(x,y, pars)
            
        """
        mesh = self.mesh
        subforest_flag = self.subforest_flag
        #
        # Iterate over outer integral
        # 
        for cell01 in mesh.cells.get_leaves(subforest_flag=subforest_flag):
            #
            # Iterate over inner integral
            # 
            for cell02 in mesh.cells.get_leaves(subforest_flag=subforest_flag):
                pass
    
          #
            # Assemble double integral
            #
            #  C(pi,pj) = II pi(xi) pj(xj) cov(xi,xj) dx 
            
            # Initialize 
            n_dofs = dofhandler.n_dofs()
            Sigma = np.zeros((n_dofs,n_dofs))
            m_row = []
            m_col = []
            m_val = []
            
            # Gauss rule on reference domain
            rule = GaussRule(9, element=element)
            xg_ref = rule.nodes()
            w_xg_ref = rule.weights()
            n_gauss = rule.n_nodes()
            
            # Iterate over mesh nodes: outer loop
            leaves = mesh.root_node().get_leaves()
            n_nodes = len(leaves)
            for i in range(n_nodes):
                # Local Gauss nodes and weights
                xnode = leaves[i]
                xcell = xnode.cell()
                xdofs = dofhandler.get_global_dofs(xnode)
                n_dofs_loc = len(xdofs)
                xg = xcell.map(xg_ref) 
                w_xg = rule.jacobian(xcell)*w_xg_ref
                
                # Evaluate shape functions and local mass matrix 
                xphi = element.shape(xg_ref)
                w_xphi = np.diag(w_xg).dot(xphi)
                m_loc = np.dot(xphi.T, np.dot(w_xphi))
                
                # Iterate over mesh nodes: inner loop
                for j in range(i,n_nodes):
                    ynode = leaves[j]
                    ycell = ynode.cell()
                    ydofs = dofhandler.get_global_dofs(ynode)
                    yg = xcell.map(xg_ref)
                    w_yg = rule.jacobian(ycell)*w_xg_ref
                    if i == j: 
                        yphi = xphi
                    else:
                        yphi = element.shape(xg_ref)
                    w_yphi = np.diag(w_yg).dot(yphi)
                    
                #
                # Evaluate covariance function at the local Gauss points
                # 
                ii,jj = np.meshgrid(np.arange(n_gauss),np.arange(n_gauss))
                if mesh.dim == 1:
                    x1, x2 = xg[ii.ravel()], yg[jj.ravel()]
                elif mesh.dim == 2:
                    x1, x2 = xg[ii.ravel(),:],yg[jj.ravel(),:]
                    
                C_loc = cov_fn(x1,x2,**cov_par).reshape(n_gauss,n_gauss)
                CC_loc = np.dot(w_yphi.T,C_loc.dot(w_xphi))
                    
            # Local to global mapping     
            for ii in range(n_dofs_loc):
                for jj in range(n_dofs_loc):
                    # Covariance 
                    Sigma[xdofs[ii],ydofs[jj]] += CC_loc[i,j]
                    Sigma[ydofs[jj],xdofs[ii]] += CC_loc[i,j]
                    
                    # Mass Matrix
                    m_row.append(ii)
                    m_col.append(jj)
                    m_val.append(m_loc[i,j])
                    
            
            # Define global mass matrix
            M = sp.coo_matrix((m_val,(m_row,m_col)))
            
            if lumped: 
                M_lumped = np.array(M.tocsr().sum(axis=1)).squeeze()
                #
                # Adjust covariance
                #
                Sigma = sp.diags(1/M_lumped)*Sigma
                return Sigma
            else:
                return Sigma, M
            
    
    
    def assemble_collocation(self):
        """
        Compute the discretization C of the covariance operator
        
        Ku(x) = I_D c(x,y) u(y) dy
        
        by collocation.
        
        Inputs:
        
            kernel
            
            pars
            
        
        Outputs:
            
            None
            
        
        Internal:
        
            self.__C
            
        """
        #
        # Interpolate the kernel at Dof-Vertices 
        # 
        
        
        u = Basis(element, 'u')
        
        assembler = Assembler()
        #
        # Assemble by finite differences
        # 
        dim = mesh.dim()
        element = QuadFE(dim, 'Q1')
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        x = dofhandler.dof_vertices()
        n = dofhandler.n_dofs()
        Sigma = np.empty((n,n))
        i,j = np.triu_indices(n)
        if dim == 1:
            Sigma[i,j] = cov_fn(x[i],x[j], **cov_par, \
                                periodic=periodic, M=M)
        if dim == 2:
            Sigma[i,j] = cov_fn(x[i,:],x[j,:], **cov_par, \
                                periodic=periodic, M=M)
        #
        # Reflect upper triangular part onto lower triangular part
        # 
        i,j = np.tril_indices(n,-1)
        Sigma[i,j] = Sigma[j,i]
        return Sigma      
'''
  


    
'''    
# =============================================================================
# Gaussian Markov Random Field Class
# =============================================================================
class GMRF(object):
    """
    Gaussian Markov Random Field
    
    Remark: Let's make this class completely independent of the discretization style.  
    
    Inputs (or important information) may be: 
        covariance/precision
        sparse/full
        full rank/degenerate
        finite difference / finite element
                   
    Modes:  
        
        Cholesky:
            Exploits sparsity
                    
        Singular value decomposition (KL)
            Computationally expensive
            Conditioning is easy
            
    Wishlist: 
    
        - Estimate the precision matrix from the covariance (Quic)
        - Log likelihood evaluation
        
               
    NOTES: 
    
    TODO: In what format should the sparse matrices be stored? consistency 
    TODO: Check: For sparse matrix A, Ax is computed by A.dot(x), not np.dot(A,x) 
    
    """ 
    def __init__(self, mean=0, precision=None, covariance=None):
        """
        Constructor
        
        Inputs:
        
            dofhandler: DofHandler, encoding mesh and element information
            
            subforest_flag: int/str/tuple, submesh indicator
        
            mean: Nodal, random field expectation (default=0)
            
            precision: SPDEMatrix, (n,n) sparse/full precision matrix
                    
            covariance: SPDEMatrix, (n,n) sparse/full covariance matrix

        """               
        # =====================================================================
        # Parse Precision and Covariance
        # =====================================================================
        # Check that at least one is not None
        assert precision is not None or covariance is not None, \
            'Specify precision or covariance (or both).'  
        
        #
        # Store covariance
        #
        self.__covariance = covariance
        
        # Check covariance type
        if self.covariance() is not None:
            assert isinstance(covariance, SPDMatrix), 'Input "covariance" '+\
                'must be an SPDMatrix object.'
            self.__size = covariance.size()
        #
        # Store precision
        #
        self.__precision = precision
        if precision is not None:
            assert isinstance(precision, SPDMatrix), 'Input "precision" '+\
                'must be an SPDMatrix object.'   
            if self.covariance() is not None:
                assert self.precision().size() == self.size()
            else:
                self.__size = self.precision().size()
        
        
        # =====================================================================
        # Mean
        # =====================================================================
        if mean is not None:
            #
            # Mean specified
            # 
            if isinstance(mean, Real):
                mean = mean*np.ones((self.size(),1))
            else:
                assert isinstance(mean, np.ndarray), \
                'Input "mean" should be a numpy array.'
            
                assert mean.shape[0] == self.size(), \
                'Mean incompatible with precision/covariance.'
        else: 
            #
            # Zero mean (default)
            # 
            mean = np.zeros((self.size(),1))
        self.__mean = mean
        
        # 
        # Convenience parameter b = Q\mu
        #
        if self.precision() is not None:
            #
            # Precision is given
            # 
            if not np.allclose(mean.data(), np.zeros(self.size()), 1e-10):
                #
                # mean is not zero
                b = self.precision.solve(self.mean())
                #
            else:
                b = np.zeros(self.size())
        else:
            b = None
        self.__b = b
        
    
    @classmethod
    def from_covariance_kernel(cls, cov_name, cov_par, mesh, \
                               mu=None, element=None):
        """
        Initialize GMRF from covariance function
        
        Inputs: 
        
            cov_name: string, name of one of the positive definite covariance
                functions that are supported 
                
                    ['constant', 'linear', 'sqr_exponential', 'exponential', 
                     'matern', 'rational'].
                     
            cov_par: dict, parameter name value pairs
            
            mesh: Mesh, computational mesh
            
            mu: double, expectation vector
            
            element: QuadFE, element (necessary for finite element discretization).
             
                     
        Note: In the case of finite element discretization, mass lumping is used. 
        """
        # Convert covariance name to function 
        #cov_fn = globals()['GMRF.'+cov_name+'_cov']
        cov_fn = locals()[cov_name+'_cov']
        #
        # Discretize the covariance function
        # 
        if element is None:
            #
            # Pointwise evaluation of the kernel
            #
            x = mesh.quadvertices()
            n_verts = x.shape[0]
            Y = np.repeat(x, n_verts, axis=0)
            X = np.tile(x, (n_verts,1))
            Sigma = cov_fn(X,Y,**cov_par).reshape(n_verts,n_verts)
            discretization = 'finite_differences' 
        else:
            #
            # Finite element discretization of the kernel
            # 
            discretization = 'finite_elements'
            #
            # Assemble double integral
            #

            system = Assembler(mesh, element) 
            n_dofs = system.n_dofs()
            Sigma = np.zeros((n_dofs,n_dofs))
            
            # Gauss points
            rule = system.cell_rule()
            n_gauss = rule.n_nodes()                  
            for node_1 in mesh.root_node().get_leaves():
                node_dofs_1 = system.get_global_dofs(node_1)
                n_dofs_1 = len(node_dofs_1)
                cell_1 = node_1.cell()
                
                
                weights_1 = rule.jacobian(cell_1)*rule.weights()
                x_gauss_1 = rule.map(cell_1, x=rule.nodes())
                phi_1 = system.shape_eval(cell=cell_1)    
                WPhi_1 = np.diag(weights_1).dot(phi_1)
                for node_2 in mesh.root_node().get_leaves():
                    node_dofs_2 = system.get_global_dofs(node_2)
                    n_dofs_2 = len(node_dofs_2)
                    cell_2 = node_2.cell()
                    
                    x_gauss_2 = rule.map(cell_2, x=rule.nodes())
                    weights_2 = rule.jacobian(cell_2)*rule.weights()
                    phi_2 = system.shape_eval(cell=cell_2)
                    WPhi_2 = np.diag(weights_2).dot(phi_2)
                    
                    i,j = np.meshgrid(np.arange(n_gauss),np.arange(n_gauss))
                    x1, x2 = x_gauss_1[i.ravel(),:],x_gauss_2[j.ravel(),:]
                    C_loc = cov_fn(x1,x2,**cov_par).reshape(n_gauss,n_gauss)
                
                    CC_loc = np.dot(WPhi_2.T,C_loc.dot(WPhi_1))
                    for i in range(n_dofs_1):
                        for j in range(n_dofs_2):
                            Sigma[node_dofs_1[i],node_dofs_2[j]] += CC_loc[i,j]
                        
                        
            
            #
            # Lumped mass matrix (not necessary!)
            #
            M = system.assemble(bilinear_forms=[(1,'u','v')]).tocsr()
            m_lumped = np.array(M.sum(axis=1)).squeeze()
            #
            # Adjust covariance
            #
            Sigma = sp.diags(1/m_lumped).dot(Sigma)
            
        return cls(mu=mu, covariance=Sigma, mesh=mesh, element=element, \
                   discretization=discretization)
    
    @classmethod
    def from_matern_pde(cls, alpha, kappa, mesh, element=None, tau=None):
        """
        Initialize finite element GMRF from matern PDE
        
        Inputs: 
        
            alpha: double >0, smoothness parameter
            
            kappa: double >0, regularization parameter
            
            mesh: Mesh, computational mesh 
            
            *element: QuadFE, finite element (optional)
            
            *tau: double, matrix-valued function representing the structure
                tensor tau(x,y) = [uxx uxy; uxy uyy].
        """
        #if element is not None: 
        #    discretization = 'finite_elements'
        #else:
        #    discretization = 'finite_differences'
            
        Q = Gmrf.matern_precision(mesh, element, alpha, kappa, tau)
        return cls(precision=Q, mesh=mesh, element=element)
    
    
    
               
                        
        
        #
        # Lumped mass matrix (not necessary!)
        #
        M = system.assemble(bilinear_forms=[(1,'u','v')]).tocsr()
        m_lumped = np.array(M.sum(axis=1)).squeeze()
        #
        # Adjust covariance
        #
        Sigma = sp.diags(1/m_lumped).dot(Sigma)
            
        return cls(mu=mu, covariance=Sigma, mesh=mesh, element=element, \
                   discretization=discretization)
    
    
    def precision(self):
        """
        Return the precision matrix
        """
        return self.__precision
    
    
    def covariance(self):
        """
        Return the covariance matrix
        """
        return self.__covariance
        
    
    def mean(self,n_copies=None):
        """
        Return the mean of the random vector
        
        Inputs:
        
            n_copies: int, number of copies of the mean
            
        Output: 
        
            mu: (n,n_copies) mean
        """
        if n_copies is not None:
            assert type(n_copies) is np.int, \
                'Number of copies should be an integer.'
            if n_copies == 1:
                return self.__mean
            else:
                return np.tile(self.__mean, (n_copies,1)).transpose()
        else:
            return self.__mean
        
    
    def size(self):
        """
        Return the size of the random vector 
        """
        return self.__size
    
    
       
    def L(self, b=None, mode='precision'):
        """
        Return lower triangular Cholesky factor L or compute L*b
        
            Inputs: 
            
                b: double, compatible vector
                
                mode: string, Specify the matrix for which to return the 
                    Cholesky factor: 'precision' (default) or 'covariance'
                    
                    
            Output:
            
                Lprec/Lcov: double, (sparse) lower triangular left Cholesky 
                    factor (if no b is specified) 
                    
                    or 
                
                y = Lprec*b / y = Lcov*b: double, vector.
                
        TODO: Move to Precision/Covariance
        """
        #
        # Parse mode
        #
        assert self.mode_supported(mode), \
            'Mode "'+mode+'" not supported by this random field.' 
        if mode == 'precision':
            #
            # Precision Matrix
            # 
            assert self.__f_prec is not None, \
                'Precision matrix not specified.'
            if sp.isspmatrix(self.__Q):
                #
                # Sparse matrix, use CHOLMOD
                #  
                P = self.__f_prec.P()
                L = self.__f_prec.L()[P,:][:,P]
            else:
                #
                # Cholesky Factor stored as full matrix
                # 
                L = self.__f_prec

        elif mode == 'covariance':
            #
            # Covariance Matrix
            # 
            assert self.__f_cov is not None, \
                'Covariance matrix not specified.'
            if sp.isspmatrix(self.__Sigma):
                #
                # Sparse Covariance matrix, use CHOLMOD
                # 
                P = self.__f_cov.P()
                L = self.__f_cov.L()[P,:][:,P]
            else:
                #
                # Cholesky Factor stored as full matrix
                # 
                L = self.__f_cov
        else:
            raise Exception('Mode not recognized. Use either' + \
                            '"precision" or "covariance".')
        #
        # Parse b   
        # 
        if b is None:
            return L 
        else: 
            return L.dot(b) 
        
        
    
    def b(self):
        """
        Return Q\mu
        """
        return self.__b
    
    def Q_solve(self, b):
        """
        Return the solution x of Qx = b by successively solving 
        Ly = b for y and hence L^T x = y for x.
        
        TODO: Move to precision
        """
        if sp.isspmatrix(self.__Q):
            return self.__f_prec(b)
        else:
            y = np.linalg.solve(self.__f_prec, b)
            return np.linalg.solve(self.__f_prec.transpose(),y)
    
    
    
    def L_solve(self, b, mode='precision'):
        """
        Return the solution x of Lx = b, where Q = LL' (or S=LL')
        
        Note: The 'L' CHOLMOD's solve_L is the one appearing in the 
            factorization LDL' = PQP'. We first rewrite it as 
            Q = WW', where W = P'*L*sqrt(D)*P
        """
        assert self.mode_supported(mode),\
            'Mode "'+ mode + '" not supported for this random field.'
        if mode == 'precision':
            if sp.isspmatrix(self.__Q):
                # Sparse
                f = self.__f_prec
                sqrtDinv = sp.diags(1/np.sqrt(f.D()))
                return f.apply_Pt(sqrtDinv*f.solve_L(f.apply_P(b))) 
            else: 
                # Full
                return np.linalg.solve(self.__f_prec,b)
        elif mode == 'covariance':
            if sp.isspmatrix(self.__Sigma):
                # Sparse
                f = self.__f_cov
                sqrtDinv = sp.diags(1/np.sqrt(f.D()))
                return f.apply_Pt(sqrtDinv*f.solve_L(f.apply_P(b)))
            else:
                # Full
                return np.linalg.solve(self.__f_cov,b)
    
    
    def Lt_solve(self, b, mode='precision'):
        """
        Return the solution x, of L'x = b, where Q = LL' (or S=LL')
        
        Note: The 'L' CHOLMOD's solve_L is the one appearing in the 
            factorization LDL' = PQP'. We first rewrite it as 
            Q = WW', where W' = P'*sqrt(D)*L'*P.
        """
        assert self.mode_supported(mode), \
            'Mode "'+ mode + '" not supported for this random field.'
        if mode == 'precision':
            #
            # Precision matrix
            # 
            if sp.isspmatrix(self.__Q):
                # Sparse
                f = self.__f_prec
                sqrtDinv = sp.diags(1/np.sqrt(f.D()))
                return f.apply_Pt(f.solve_Lt(sqrtDinv*(f.apply_P(b))))
            else:
                # Full
                return np.linalg.solve(self.__f_prec.transpose(),b)
        elif mode == 'covariance':
            #
            # Covariance matrix
            # 
            if sp.isspmatrix(self.__Sigma):
                # Sparse
                f = self.__f_cov
                sqrtDinv = sp.diags(1/np.sqrt(f.D()))
                return f.apply_Pt(f.solve_Lt(sqrtDinv*(f.apply_P(b))))
            else:
                # Full
                return np.linalg.solve(self.__f_cov.transpose(),b)
        else:
            raise Exception('For mode, use "precision" or "covariance".')
    
    
    def KL(self, precision=None, k=None):
        """
        Inputs:
        
        Outputs:
        
        """
        mesh = self.mesh()
        
    def chol_sample(self, n_samples=1, z=None, mode='covariance'):
        """
        Generate sample realizations from Gaussian random field.
        
        Inputs:
        
            n_samples: int, number of samples to generate
            
            z: (n,n_samples) random vector ~N(0,I).
            
            mode: str, specify parameters used to simulate random field
                ['precision', 'covariance', 'canonical']
              
        Outputs:
        
            x: (n,n_samples), samples paths of random field
            
                
        Note: Samples generated from the cholesky decomposition of Q are 
            different from those generated from that of Sigma. 
                
                Q = LL' (lower*upper)
                  
            =>  S = Q^(-1) = L'^(-1) L^(-1) (upper*lower)
            
            However, they have  the same distribution
        """
        #
        # Parse samples
        # 
        if z is not None:
            #
            # Extract number of samples from z
            #  
            assert len(z.shape) == 2, \
                'Input "z" should have size (n, n_samples).'
            assert z.shape[0] == self.size(), \
                'Input "z" should have size (n, n_samples).'
            n_samples = z.shape[1]
        else:
            #
            # Generate z
            # 
            z = np.random.normal(size=(self.size(), n_samples))
            
            
        if mode=='covariance':
            #
            # Cholesky decomposition of the covariance operator
            # 
            assert self.covariance() is not None, 'No covariance specified.'
            
            # Get covariance
            K = self.covariance()
            
            # Compute Cholesky factorization L*L' if necessary
            if not K.has_chol_decomp():
                K.chol_decomp()
            
            # Return Lz + mean
            return K.chol_sqrt(z) + self.mean(n_copies=n_samples)
            
        elif mode=='precision':
            #
            # Cholesky decomposition of the precision matrix
            # 
            assert self.precision() is not None, 'No precision specified.'
            
            # Get precision matrix
            Q = self.precision()
            
            # Compute Cholesky decomposition L*L' if necessary
            if not Q.has_chol_decomp():
                Q.chol_decomp()
            
            # Return L^{-T} z + mean
            return Q.sqrt_solve(z,transpose=True) \
                + self.mean(n_copies=n_samples)  
                       
        elif mode=='canonical':
            #
            # Cholesky decomposition of the precision matrix 
            #
            assert self.precision() is not None, 'No precision specified.'
            return self.chol_sample(z=z, mode='precision')
        else:
            raise Exception('Input "mode" not recognized. '+\
                            'Use "covariance", "precision", or "canonical".')
        
     
    
    
    def eig_sample(self, n_samples=1, z=None, mode='covariance'):
        """
        Generate a random sample based on the eigendecomposition of the 
        covariance/precision matrix
        
        Inputs:
        
            n_samples: int, number of samples to generate
            
            z: (n,n_samples) random vector ~N(0,I).
            
            mode: str, specify parameters used to simulate random field
                ['precision', 'covariance', 'canonical']
            
            
        Outputs:
        
            x: (n,n_samples), samples paths of random field
    
        
        Note: The eigendecomposition can generate samples from degenerate
            covariance/precision matrices.
        """
        #
        # Parse samples
        # 
        if z is not None:
            #
            # Extract number of samples from z
            #  
            assert len(z.shape) == 2, \
                'Input "z" should have size (n, n_samples).'
            assert z.shape[0] == self.size(), \
                'Input "z" should have size (n, n_samples).'
            n_samples = z.shape[1]
        else:
            #
            # Generate z
            # 
            z = np.random.normal(size=(self.size(), n_samples))
            
            
        if mode=='covariance':
            #
            # Eigen-decomposition of the covariance operator
            # 
            assert self.covariance() is not None, 'No covariance specified.'
            
            # Get covariance
            K = self.covariance()
            
            # Compute eigendecomposition if necessary
            if not K.has_eig_decomp():
                K.compute_eig_decomp()
            
            # Return Lz + mean
            return K.eig_sqrt(z) + self.mean(n_copies=n_samples)
            
        elif mode=='precision':
            #
            # Cholesky decomposition of the precision matrix
            # 
            assert self.precision() is not None, 'No precision specified.'
            
            # Get precision matrix
            Q = self.precision()
            
            # Compute Cholesky decomposition L*L' if necessary
            if not Q.has_chol_decomp():
                Q.chol_decomp()
            
            # Return L^{-T} z + mean
            return Q.sqrt_solve(z,transpose=True) \
                + self.mean(n_copies=n_samples)  
                       
        elif mode=='canonical':
            #
            # Cholesky decomposition of the precision matrix 
            #
            assert self.precision() is not None, 'No precision specified.'
            return self.chol_sample(z=z, mode='precision')
        else:
            raise Exception('Input "mode" not recognized. '+\
                            'Use "covariance", "precision", or "canonical".')
    
        
        
    def chol_condition(self, A, e, Ko=0, output='sample', mode='precision', 
                       z=None, n_samples=1):
        """
        Computes the conditional covariance of X, given E ~ N(AX, Ko). 
        
        Inputs:
        
            A: double, (k,n) 
            
            Ko: double symm, covariance matrix of E.
            
            e: double, value
            
            output: str, type of output desired [sample/covariance]
            
            Z: double, array whose columns are iid N(0,1) random vectors 
                (ignored if output='gmrf')
            
            n_samples: int, number of samples (ignored if Z is not None)
            
            
        Note: For 
        TODO: Test
        """
        if Ko == 0:
            #
            # Hard Constraint
            #
            if mode=='precision':
                Q = self.precision()
                V = Q.chol_solve(A.T)
                W = A.dot(V)
                U = linalg.solve(W,V.T)
                if output=='sample':
                    #
                    # Sample
                    # 
                    x = self.chol_sample(z=z, n_samples=n_samples, mode='precision')                    
                    c = A.dot(x)-e
                    x = x - np.dot(U.T, c)
                    return x
                elif mode=='gmrf':
                    #
                    # Random Field
                    #

                    # Conditional mean
                    m = self.mean()  
                    c = A.dot(m)-e
                    mm = m - np.dot(U.T,c)
                    
                    # Conditional covariance
                    K = Q.chol_solve(np.identity(self.size()))
                    KK = K - V.dot(U)
                else:
                    raise Exception('Input "mode" should be "sample" or "gmrf".')
                    
            elif mode=='covariance':
                K = self.covariance()
                V = K.dot(A.T)
                W = A.dot(V)
                U = linalg.solve(W,V.T)
                if output=='sample':
                    #
                    # Sample
                    # 
                    x = self.chol_sample(z=z, n_samples=n_samples, mode='covariance')
                    c = A.dot(x)-e
                    x = x - np.dot(U.T,c)
                    return x
                else:
                    #
                    # Random Field
                    # 
                    
                    # Conditional covariance
                    KK = K - V.dot(U) 
                    
                    # Conditional mean
                    m = self.mean()
                    c = A.dot(m)-e
                    mm = m - np.dot(U.T,c)
                    
                    return GMRF(mean=mm, covariance=KK) 
        else:
            #
            # Condition on Gaussian observations
            # 
            if mode=='precision':
                #
                # Precision 
                # 
                if output=='sample':
                    pass
                else:
                    pass
            elif mode=='covariance':
                #
                # Covariance
                #
                if output=='sample':
                    pass
                else:
                    pass
                
                
    
    def eig_condition(self, A, e, Ko=0, output='sample', mode='precision', 
                       z=None, n_samples=1):
        """
        Computes the conditional random field, X given E ~ N(AX, Ko). 
        
        Inputs:
        
            A: double, (k,n) 
            
            Ko: double symm, covariance matrix of E.
            
            e: double, value
            
            output: str, type of output desired [sample/covariance]
            
            Z: double, array whose columns are iid N(0,1) random vectors 
                (ignored if output='gmrf')
            
            n_samples: int, number of samples (ignored if Z is not None)
        """
        pass
    
    
    
        if constraint_type == 'pointwise':
            i_b, x_b = constraint
            i_a = [i not in i_b for i in range(self.n())]
            mu_a, mu_b = self.mu()[i_a], self.mu()[i_b]
            Q_aa = self.Q().tocsc()[np.ix_(i_a,i_a)]
            Q_ab = self.Q().tocsc()[np.ix_(i_a,i_b)]
            
            #
            # Conditional random field
            # 
            mu_agb = mu_a - spla.spsolve(Q_aa, Q_ab.dot(x_b-mu_b))
            if n_samples is None:
                return GMRF(mu=mu_agb, precision=Q_aa)
            else: 
                pass
            
        elif constraint_type == 'hard':
            A, e  = constraint
            assert self.mode_supported(mode), 'Mode not supported.'
            if output == 'gmrf':
                if mode == 'precision':
                    pass
                elif mode == 'covariance':
                    mu = self.mu()
                    S  = self.Sigma()
                    c =  A.dot(mu) - e
                    V = S.dot(A.T.dot(linalg.solve(A.dot(S.dot(A.T)),c)))
                    mu_gAx = self.mu() - V 
                     
            elif output == 'sample':
                #
                # Generate samples directly via Kriging
                # 
                if z is None:
                    # Z is not specified -> generate samples
                    z = self.iid_gauss(n_samples)
                if mode == 'precision':
                    #
                    # Use precision matrix
                    #
                    # Sample from unconstrained gmrf
                    v = self.Lt_solve(z)
                    x = self.mu(n_samples) + v
                    
                    # Compute [Sgm*A'*(A*Sgm*A')^(-1)]'
                    V = self.Q_solve(A.T)
                    W = A.dot(V)
                    U = linalg.solve(W, V.T)
                    
                    # Compute x|{Ax=e} = x - Sgm*A'*(A*Sgm*A')^(-1)(Ax-e)
                    if n_samples > 1:
                        e = np.tile(e, (n_samples,1)).transpose()
                    c = A.dot(x)-e
                    return x-np.dot(U.T,c) 
                           
                elif mode == 'covariance':
                    #
                    # Use covariance matrix
                    #
                    x = self.sample(n_samples=n_samples, z=z, 
                                    mode='covariance')
                    if n_samples > 1:
                        e = np.tile(e, (n_samples,1)).transpose()
                    c = A.dot(x)-e
                    
                    # Compute Sgm*A'*(A*Sgm*A')^(-1)
                    S = self.Sigma()
                    return x - S.dot(A.T.dot(linalg.solve(A.dot(S.dot(A.T)),c)))
            elif output == 'log_pdf':
                pass
            else:
                raise Exception('Variable "output" should be: '+\
                                '"gmrf","sample",or "log_pdf".')
        elif constraint_type == 'soft':
            pass
        else:
            raise Exception('Input "constraint_type" should be:' + \
                            ' "pointwise", "hard", or "soft"')
        '''
    

        