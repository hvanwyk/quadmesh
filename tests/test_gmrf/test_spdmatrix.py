import unittest
from gmrf import SPDMatrix
from gmrf import CovKernel
from gmrf import Covariance
from gmrf import diagonal_inverse
from gmrf import modchol_ldlt  
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sp
from scipy.sparse import linalg as spla
import matplotlib.pyplot as plt
from sksparse.cholmod import cholesky, cholesky_AAt, Factor, CholmodNotPositiveDefiniteError  # @UnresolvedImport

def test_matrix(n, sparse=False, rank='full'):
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
        if rank=='full':
            #
            # Full rank sparse matrix 
            # 
            delta = 1e-3
            L = sp.random(n,n,density=0.8, format='csr') 
            A = L.dot(L.T) + delta*sp.eye(n)
        else:
            #
            # Degenerate sparse matrix (at most rank=rank)
            # 
            L = sp.random(n,n,density=0.5, format='csr')
            one = np.ones(n)
            i_zeros = np.random.randint(n, size=n-rank)
            one[i_zeros] = 0
            D = sp.diags(one)
            A = L.dot(D.dot(L.T))
    else:
        #
        # Full matrix
        # 
        if rank=='full':
            #
            # Full rank matrix
            #
            delta = 1e-3
            L = np.random.rand(n,n)
            A = L.dot(L.T) + delta*np.eye(n)
        else:
            #
            # Degenerate matrix
            #
            L = np.random.rand(n,n)
            one = np.ones(n)
            i_zeros = np.random.randint(n, size=n-rank)
            one[i_zeros] = 0
            D = sp.diags(one)
            A = L.dot(D.dot(L.T))
        
    return A
    


class TestSPDMatrix(unittest.TestCase):
    """
    Test the storage, inversion and factorization of matrices of the 
    form M^{-1} K
    """
    def test_neg_definite(self):
        n = 40
        neg_definite = 0
        for dummy in range(10):
            A = test_matrix(n, sparse=True, rank=30)
            d = linalg.eigvalsh(A.toarray())
            if any([dd<-1e-12 for dd in d]):
                neg_definite += 1
        if neg_definite > 0:
            print('number of negative definite systems', neg_definite)
        
    
    
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
        n = 5
        for sparse in [True, False]:
            for rank in ['full', 3]:
                A = test_matrix(n, sparse, rank)
                K = SPDMatrix(A)
                
                # Check size function
                self.assertEqual(K.size(),5)
                
                # Check sparsity function
                self.assertEqual(K.issparse(),sparse)
                
                # Check get_matrix function
                if sparse:
                    self.assertTrue(np.allclose(K.get_matrix().toarray(), A.toarray()))
                else:
                    self.assertTrue(np.allclose(K.get_matrix(), A))
    
    
    def test_rank(self):
        
        A = np.array([[1, 1, 0, 1], 
                  [1, 1, 1, 0], 
                  [0, 1, 1, 1], 
                  [1, 0, 1, 1]])
        K = SPDMatrix(A)
        P, L, U = linalg.lu(A)
        x = linalg.solve(A,np.zeros(4))
        
        
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
            #
            # Cycle through sparsity
            #
            for rank in ['full', n-3]:
                #
                # Cycle through degeneracy
                #
                
                # Generate random SPD matrix
                A = test_matrix(n, sparsity, rank)
                K = SPDMatrix(A)
                
                # Compute the Cholesky decomposition
                K.chol_decomp()
                
                # Check that the right algorithm was used.
                if sp.issparse(A):    
                    A = A.toarray()
                    
                rank = np.linalg.matrix_rank(A)
                if rank < n:
                    chol_type = 'full'
                else:
                    if K.issparse():
                        chol_type = 'sparse'
                    else:
                        chol_type = 'full'
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
                for rank in ['full', n-2]:
                    #
                    # Cycle through degeneracy
                    #
                    
                    # Generate random SPD matrix
                    A = test_matrix(n, sparsity, rank)
                    K = SPDMatrix(A)
                    
                    # Compute the Cholesky decomposition
                    K.chol_decomp()
    
                    # Check that the decomposition reproduces the matrix
                    if K.chol_type()=='full_cholesky':
                        # Get Cholesky factor
                        L, D, P, D0 = K.get_chol_decomp()
                        
                        # Check reconstruction
                        self.assertTrue(np.allclose(L.dot(D.dot(L.T)),A))
                        
                        # Degenerate matrix: Diagonal matrices differ
                        if rank == n-2:
                            self.assertFalse(np.allclose(D,D0))
                            
                    elif K.chol_type()=='sparse_cholesky':
                        # Get Cholesky factor
                        L = K.get_chol_decomp()
                        P = L.P()
                        LL = L.L()[P,:][:,P]
                        
                        # Check reconstruction
                        self.assertTrue(np.allclose(LL.dot(LL.T).toarray(),A.toarray()))
                
                
    def test_chol_L(self):
        """
        Return L or L*b, where K = LL'
        """
        n = 20
        b = np.random.rand(n)
        for sparsity in [False, True]:
            #
            # Cycle through sparsity
            #
            for rank in ['full', n-3]:
                #
                # Cycle through degeneracy
                #
                
                # Generate random SPD matrix
                A = test_matrix(n, sparsity, rank)
                K = SPDMatrix(A)
                
                # Compute the Cholesky decomposition
                K.chol_decomp()

                # Compute
                if K.chol_type()=='full':
                    
                    #fig, axs = plt.subplots(2,2)
                    L, D, P, D0 = K.get_chol_decomp()
                    """
                    
                    axs[0,0].imshow(P.dot(L))
                    axs[0,0].set_title('PL')
                    
                    axs[0,1].imshow(D)
                    axs[0,1].set_title('D')
                    
                    axs[1,0].imshow(P)
                    axs[1,0].set_title('P')
                    
                    axs[1,1].imshow(D0)
                    axs[1,1].set_title('D0')
                    
                    plt.show()
                    """
                    
                    self.assertTrue(np.allclose(np.diagonal(P.dot(L)),1))
                    """
                    fig, axs = plt.subplots(2,2)
                    im1 = axs[0,0].imshow(A-L.dot(D0.dot(L.T)))
                    fig.colorbar(im1, ax=axs[0, 0])
                    
                    im2 = axs[0,1].imshow(A-L.dot(D.dot(L.T)))
                    fig.colorbar(im2, ax=axs[0, 1])
                    plt.show()
                    """
                    #plt.imshow(D)
                    #plt.imshow(D0)
                    #plt.colorbar()
                    #plt.show()
                elif K.chol_type()=='sparse':
                    # Return the lower triangular matrix L so that PAP' = LL'
                    L = K.chol_L()
                    
                    # Check that LL' = PAP'
                    
                    
                    # Evaluate L*b, where PAP' = LL'
                    b = np.random.rand(n)
                    Lb = K.chol_L(b)
                    
                    # Check that b'(PA'P')b = (Lb)'(Lb)
                
                
               

    def test_chol_Lsolve(self):
        n = 5
        
        for sparsity in [False, True]:
            #
            # Cycle through sparsity
            #
            for rank in ['full', np.int(n/2)]:
                #
                # Cycle through degeneracy
                #
                
                # Generate random SPD matrix
                A = test_matrix(n, sparsity, rank)
                K = SPDMatrix(A)
                
                # Compute the Cholesky decomposition
                K.chol_decomp()

    
    def test_chol_Ltsolve(self):
        n = 5
        
        for sparsity in [False, True]:
            #
            # Cycle through sparsity
            #
            for rank in ['full', np.int(n/2)]:
                #
                # Cycle through degeneracy
                #
                
                # Generate random SPD matrix
                A = test_matrix(n, sparsity, rank)
                K = SPDMatrix(A)
                
                # Compute the Cholesky decomposition
                K.chol_decomp()

    
    def test_chol_solve(self):
        n = 5
        
        for sparsity in [False, True]:
            #
            # Cycle through sparsity
            #
            for rank in ['full', np.int(n/2)]:
                #
                # Cycle through degeneracy
                #
                
                # Generate random SPD matrix
                A = test_matrix(n, sparsity, rank)
                K = SPDMatrix(A)
                
                # Compute the Cholesky decomposition
                K.chol_decomp()

    
    
    def test_eig(self):
        # Form SPD matrix
        A = np.array([[1, 1, 0, 1], 
                  [1, 1, 1, 0], 
                  [0, 1, 1, 1], 
                  [1, 0, 1, 1]])
        K = SPDMatrix(A)
        
        # Compute eigendecomposition
        K.eig_decomp()
        
        # Make up system
        x = np.random.rand(K.size())
        b = A.dot(x)
    
        # Solve it
        xx = K.eig_solve(b)
        xxx = np.linalg.solve(A,b)
        
        # Check 
        self.assertTrue(np.allclose(xx,x))
        self.assertTrue(np.allclose(xxx,x))
        
        
        
    def test_scalings(self):
        pass 