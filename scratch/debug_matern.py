from sksparse.cholmod import cholesky, cholesky_AAt
import scipy.sparse as sp
from scipy.sparse import linalg as spla
import numpy as np

def factor_of(factor, matrix):
    #
    # Check whether a CHOLMOD factor factors a given matrix
    #
    return np.allclose((factor.L() * factor.L().H).todense(),
                       matrix.todense()[factor.P()[:, np.newaxis],
                                        factor.P()[np.newaxis, :]])

# System matrix
Lk = sp.random(3,3,0.5, format='csc') + sp.eye(3,3,format='csc')
K  = Lk*Lk.T
assert np.allclose(K.toarray(), K.T.toarray(), 1e-10), 'K not symmetric.'
assert all(np.linalg.eigvals(K.toarray())>0), 'K is not positive definite.' 

# Lumped mass matrix
m = np.array([1,2,3])
M = sp.diags(m, format='csc')
assert np.allclose(M.toarray(), M.T.toarray(), 1e-10), 'M not symmetric.'

#
# alpha = 1
# 
Q1 =  K
f_Q1 = cholesky(K)
fs_Q1 = cholesky_AAt(Lk)

assert factor_of(f_Q1, K), 'Cannot use cholesky factor to reconstruct matrix.'
    
assert factor_of(fs_Q1, K), 'Cannot use cholesky factor to reconstruct matrix.'
    
assert np.allclose(f_Q1.L().toarray(),fs_Q1.L().toarray(),1e-10), \
    'Cholesky factors differ'
#
# alpha = 2
# 
Q2 = K*spla.inv(M)*K
f_Q2 = cholesky(Q2)
fs_Q2 = cholesky_AAt(K*sp.diags(1/np.sqrt(m)))

assert factor_of(f_Q2,Q2), 'Cannot use cholesky factor to reconstruct matrix.'
    
assert factor_of(fs_Q2, Q2), 'Cannot use cholesky factor to reconstruct matrix.'
    
assert np.allclose(f_Q2.L().toarray(),fs_Q2.L().toarray(),1e-10), \
    'Cholesky factors differ'
#
# alpha = 3
# 
Q3 = K*spla.inv(M)*Q1*spla.inv(M)*K 
f_Q3 = cholesky(Q3)
fs_Q3 = cholesky_AAt(K*sp.diags(1/m)*fs_Q1.apply_Pt(fs_Q1.L()))
assert factor_of(fs_Q3, Q3), 'Cannot use cholesky factor to reconstruct matrix.'
#
# alpha = 4
# 