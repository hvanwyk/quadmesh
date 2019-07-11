import unittest
from gmrf import SPDMatrix
from gmrf import diagonal_inverse
from gmrf import modchol_ldlt  
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sp
from sklearn.datasets import make_sparse_spd_matrix

def test_matrix(n, sparse=False, d=-0.5):
    """
    Returns symmetric matrices on which to test algorithms
    
    Inputs:
    
        n: int, matrix size
        
        sparse: bool (False), sparsity
        
        rank: str/int, if 'full', then rank=n, otherwise rank=r in {1,2,...,n}.
        
    Output:
    
        A: double, symmetric positive definite matrix with specified rank
            (hopefully) and sparsity.
    """    
    if sparse:
        #
        # Sparse matrix 
        # 
        A = make_sparse_spd_matrix(dim=n, alpha=0.95, norm_diag=False,
                           smallest_coef=.1, largest_coef=.9);
        A = sp.csc_matrix(A)
    else:
        #
        # Full matrix
        #
        X = np.random.rand(n, n)
        X = X + X.T
        U, dummy, V = linalg.svd(np.dot(X.T, X))
        A = np.dot(np.dot(U, d + np.diag(np.random.rand(n))), V)
         
    return A
    


class TestSPDMatrix(unittest.TestCase):
    """
    Test the storage, inversion and factorization of matrices of the 
    form M^{-1} K
    """
    def test_modchol_ldlt(self):
        # Indefinite Matrix
        K = np.array([[1, 1, 0, 1], 
                      [1, 1, 1, 0], 
                      [0, 1, 1, 1], 
                      [1, 0, 1, 1]])
    
        # Compute modified Cholesky decomposition
        L, D, dummy, D0 = modchol_ldlt(K)
        
        self.assertTrue(np.allclose(L.dot(D0.dot(L.T)),K))
        self.assertFalse(np.allclose(D0,D))
              
        
    def test_constructor(self):
        n = 20
        for sparse in [True, False]:
            # Generate test matrix
            A = test_matrix(n, sparse)
            K = SPDMatrix(A)
            
            # Check size function
            self.assertEqual(K.size(),n)
            
            # Check sparsity function
            self.assertEqual(K.issparse(),sparse)
            
            # Check get_matrix function
            if sparse:
                self.assertTrue(np.allclose(K.get_matrix().toarray(), A.toarray()))
            else:
                self.assertTrue(np.allclose(K.get_matrix(), A))
       
        
    def test_diag_inverse(self):
        #
        # Compute the pseudo-inverse of a diagonal matrix
        # 
        I = np.eye(10)
        I[-1,-1] = 0        
        J = diagonal_inverse(np.diag(I))
        JJ = diagonal_inverse(I)
        self.assertTrue(np.allclose(I, J))
        self.assertTrue(np.allclose(I,JJ))

        
    def test_chol_types(self):
        
        n = 20        
        for sparsity in [False, True]:
            # Generate random SPD matrix
            A = test_matrix(n, sparsity)
            K = SPDMatrix(A)
            
            # Compute the Cholesky decomposition
            K.chol_decomp()
            
            # Check that the right algorithm was used.
            if sp.issparse(A):    
                A = A.toarray()
                
            # Check that matrix is full rank
            rank = np.linalg.matrix_rank(A)    
            self.assertEqual(rank, n)
            
            chol_type = 'sparse' if sparsity else 'full' 
            self.assertEqual(chol_type, K.chol_type())
                
                
    def test_get_chol_decomp(self):
            """
            Return L,
            """
            n = 10
            for sparsity in [False, True]:
                #
                # Cycle through sparsity
                #
                    
                # Generate random SPD matrix
                A = test_matrix(n, sparsity)
                K = SPDMatrix(A)
                
                # Compute the Cholesky decomposition
                K.chol_decomp()

                # Check that the decomposition reproduces the matrix
                if K.chol_type()=='full':
                    # Get Cholesky factor
                    L, D, P, D0 = K.get_chol_decomp()
                    
                    if not np.allclose(D,D0):
                        # Indefinite matrix - change to modified matrix
                        A = L.dot(D.dot(L.T))
                        
                    # Check reconstruction
                    self.assertTrue(np.allclose(L.dot(D.dot(L.T)),A))
                    

                    # Check that P*L is lower triangular with ones on diagonal
                    self.assertTrue(np.allclose(1, np.diagonal(P.dot(L))))
                    self.assertTrue(np.allclose(0, linalg.triu(P.dot(L),1)))
                    
                elif K.chol_type()=='sparse':
                    # Get Cholesky factor
                    L = K.get_chol_decomp()
                    P = L.P()
                    LL = L.L()
                    
                    # Build permutation matrix
                    I = sp.diags([1],0, shape=(n,n), format='csc')
                    PP = I[P,:]
                    
                    # Compute P'L
                    LL = PP.T.dot(LL)
                    
                    # Check reconstruction LL' = PAP'
                    self.assertTrue(np.allclose(LL.dot(LL.T).toarray(),
                                                A.toarray()))                    
                    
                
    def test_chol_sqrt(self):
        """
        Return R*b, where K = R*R'
        """
        n = 20
        b = np.random.rand(n)
        for sparsity in [False, True]:
            #
            # Cycle through sparsity
            #
                
            # Generate random SPD matrix
            A = test_matrix(n, sparsity)
            K = SPDMatrix(A)
            
            # Compute the Cholesky decomposition
            K.chol_decomp()
            
            # Compute R*b
            if K.chol_type()=='full':
                
                # Reconstruct (modified) matrix
                B = K.chol_reconstruct()
                
                # Identity matrix
                I = np.eye(n)
                
                # Compute R*I
                z = K.chol_sqrt(I)
                
                # Check that R*R' = B
                self.assertTrue(np.allclose(z.dot(z.T),B))
                
                # Compute R'*b 
                b = np.random.rand(n)
                z = K.chol_sqrt(b,transpose=True)
                
                # Check that b'Ab = (Rb)'(Rb)
                self.assertTrue(np.allclose(z.dot(z),b.T.dot(B.dot(b))))
               
                
            elif K.chol_type()=='sparse':
                # Identity matrix
                I = np.eye(n)
                
                # Compute R*I
                z = K.chol_sqrt(I)
                
                # Check that RR' = A
                # print(np.linalg.norm(z.dot(z.T) - A.toarray()))
                self.assertTrue(np.allclose(z.dot(z.T),A.toarray()))
                
                # Compute R'*b
                b = np.random.rand(n)
                z = K.chol_sqrt(b, transpose=True)
                
                # Check that b'Ab = (Rb)'(Rb)
                self.assertTrue(np.allclose(z.dot(z),b.T.dot(A.dot(b))))
                                  
               

    def test_sqrt_solve(self):
        n = 20
        
        for sparsity in [False, True]:
            #
            # Cycle through sparsity
            #
            
            # Generate random SPD matrix
            A = test_matrix(n, sparsity)
            K = SPDMatrix(A)
            
            # Compute the Cholesky decomposition
            K.chol_decomp()
            
            # Random vector
            x = np.random.rand(n)
            
            for transpose in [False, True]:
                # Compute b = Rx (or R'x)
                b = K.chol_sqrt(x, transpose=transpose)  
            
                # Solve for x             
                xx = K.chol_sqrt_solve(b, transpose=transpose)
                
                # Check that we've recovered the original x             
                self.assertTrue(np.allclose(xx,x))
                
        
    def test_chol_solve(self):
        n = 100
        
        for sparsity in [False, True]:
            #
            # Cycle through sparsity
            #
            
            # Generate random SPD matrix
            A = test_matrix(n, sparsity)
            K = SPDMatrix(A)
            
            # Compute the Cholesky decomposition
            K.chol_decomp()
            
            # Use modified A if necessary
            A = K.chol_reconstruct()
                    
            # Generate random solution 
            x = np.random.rand(n)
            b = A.dot(x)
            
            # Solve using Cholesky decomposition
            xx = K.chol_solve(b)
            
            # Check accuracy
            self.assertTrue(np.allclose(xx,x))
            
            
    
    def test_eig(self):
        # Form SPD matrix
        n = 20
        for sparse in range(False, True):
            A = test_matrix(n,sparse,1)
        K = SPDMatrix(A)
        
    
        # Compute eigendecomposition
        K.compute_eig_decomp()
        
        # Check reconstruction
        d, V = K.get_eig_decomp()
        AA = V.dot(np.diag(d).dot(V.T))
        A = A.toarray() if sparse else A
        self.assertTrue(np.allclose(AA,A))
       

    def test_eigsolve(self):
        n = 20
        for sparse in range(False, True):
            # Test matrix
            A = test_matrix(n, sparse)
            K = SPDMatrix(A)
            
            # Compute eigendecomposition
            K.compute_eig_decomp()
            
            # Reconstruct
            A = K.eig_reconstruct()
            
            # Make up system
            x = np.random.rand(K.size())
            b = A.dot(x)
        
            # Solve it
            xx = K.eig_solve(b)
            xxx = np.linalg.solve(A,b)
            
            # Check 
            self.assertTrue(np.allclose(xx,x))
            self.assertTrue(np.allclose(xxx,x))
    
    
    def test_eig_sqrt(self):
        n = 20
        for sparse in range(False, True):
            # Test matrix
            A = test_matrix(n, sparse)
            K = SPDMatrix(A)
            
            # Compute eigendecomposition
            K.compute_eig_decomp()               
    
            B = K.eig_reconstruct()
            
            #
            # Test Rx
            # 
            
            # Identity matrix
            I = np.eye(n)
            
            # Compute R*I
            z = K.eig_sqrt(I)
            
            # Check that R*R' = B
            self.assertTrue(np.allclose(z.dot(z.T),B))
            
            #
            # Compute R'*b
            #  
            b = np.random.rand(n)
            z = K.eig_sqrt(b,transpose=True)
            
            # Check that b'Ab = (Rb)'(Rb)
            self.assertTrue(np.allclose(z.dot(z),b.T.dot(B.dot(b))))

            
    def test_eig_sqrt_solve(self):
        n = 20
        
        for sparsity in [False, True]:
            #
            # Cycle through sparsity
            #
            
            # Generate random SPD matrix
            A = test_matrix(n, sparsity)
            K = SPDMatrix(A)
            
            # Compute the Eigen decomposition
            K.compute_eig_decomp()
            
            # Random vector
            x = np.random.rand(n)
            
            for transpose in [False, True]:
                # Compute b = Rx (or R'x)
                b = K.eig_sqrt(x, transpose=transpose)  
            
                # Solve for x             
                xx = K.eig_sqrt_solve(b, transpose=transpose)
                
                # Check that we've recovered the original x             
                self.assertTrue(np.allclose(xx,x))
                        
            
        
    def test_scalings(self):
        pass 