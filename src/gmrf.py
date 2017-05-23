'''
Created on Feb 8, 2017

@author: hans-werner
'''
from finite_element import System, QuadFE
from mesh import Mesh
from numbers import Number
import scipy.sparse as sp
from sksparse.cholmod import cholesky, cholesky_AAt, Factor  # @UnresolvedImport
from scipy.special import kv, gamma
from scipy.sparse import linalg as spla
import numpy as np

# =============================================================================
# Covariance Functions
# =============================================================================
"""
Commonly used covariance functions

For each function, we assume the input is given by two d-dimensional
vectors of length n. 
"""
def constant_cov(x,y,sgm=1):
    """
    Constant covariance kernel
    
        C(x,y) = sgm
    
    Inputs: 
    
        x,y: double, two (n,d) arrays
        
        sgm: double >0, standard deviation
            
    Outputs:
    
        double, (n,) array of covariances  
    """
    return sgm*np.ones(x.shape[0])
    

def linear_cov(x,y,sgm=1, M=None):
    """
    Linear covariance
    
        C(x,y) = sgm^2 + <x,My>  (Euclidean inner product)
     
    """
    return sgm**2 + np.sum(x*y,axis=1)

    
def sqr_exponential_cov(x,y,sgm=1,l=1,M=None):
    """
    Squared exponential covariance function
    
        C(x,y) = exp(-|x-y|^2/(2l^2))
    
    """
    d = distance(x,y,M)
    return np.exp(-d**2/(2*l**2))

    
def exponential_cov(x,y,l,M=None):
    """
    Exponential covariance function
    
        C(x,y) = exp(-|x-y|/l)
        
    Inputs: 
    
        x,y: np.array, spatial points
        
        l: range parameter
    """
    d = distance(x,y,M)
    return np.exp(-d/l)

    
def matern_cov(x,y,sgm,nu,l,M=None):
    """
    Matern covariance function
    
    Inputs:
    
        x,y: np.array, spatial points
        
        sgm: variance
        
        nu: shape parameter (k times differentiable if nu > k)
        
        l: range parameter 
    """
    d = distance(x,y,M)
    K = sgm**2*2**(1-nu)/gamma(nu)*(np.sqrt(2*nu)*d/l)**nu*\
        kv(nu,np.sqrt(2*nu)*d/l)
    #
    # Modified Bessel function undefined at d=0, covariance should be 1
    #
    K[np.isnan(K)] = 1
    return K
    
    
def rational_cov(x,y,a,M):
    """
    Rational covariance
    
        C(x,y) = 1/(1 + |x-y|^2)^a
         
    """
    d = distance(x,y,M)
    return (1/(1+d**2))**a


def distance(x,y,M=None):
    """
    Compute the Euclidean distance vector between rows in x and rows in y
    
    Inputs: 
    
        x,y: (1,n) colum vectors
        
        M: double, (2,2) positive semidefinite matrix
        
        
    Outputs: 
    
        d: double, (n,1) vector |x[i]-y[i]| of Euclidean distances
         
    """
    #
    # Check wether x and y have the same dimensions
    # 
    assert x.shape == y.shape, 'Vectors x and y have incompatible shapes.'
    if M is None:
        return np.sqrt(np.sum((x-y)**2,axis=1))
    else:
        assert all(np.linalg.eigvals(M)>=0) and np.allclose(M,M.transpose()),\
        'M should be symmetric positive definite.'
        
        n = x.shape[0]
        d = np.empty(n)
        xmy = x-y
        for i in range(n):
            d[i] = xmy[i,:].dot(M.dot(xmy[i,:].transpose()))
        return np.sqrt(d)
             
        
     
def matern_precision(mesh, element, alpha, kappa, tau=None, 
                     boundary_conditions=None):
    """
    Return the precision matrix for the Matern random field defined on the 
    spatial mesh. The field X satisfies
    
        (k^2 - div[T(x)grad(.)])^{a/2} X = W
    
    Inputs: 
    
        mesh: Mesh, finite element mesh on which the field is defined
        
        element: QuadFE, finite element space of piecewise polynomials
        
        alpha: int, positive integer (doubles not yet implemented).
        
        kappa: double, positive regularization parameter.
        
        tau: (Axx,Axy,Ayy) symmetric tensor or diffusion coefficient function.
        
        boundary_conditions: tuple of boundary locator function and boundary value
            function (viz. finite_element.System)
        
        
    Outputs:
    
        Q: sparse matrix, in CSC format
        
    """
    system = System(mesh, element)
    
    #
    # Assemble (kappa * M + K)
    #
    bf = [(kappa,'u','v')]
    if tau is not None:
        #
        # Test whether tau is a symmetric tensor
        # 
        if type(tau) is tuple:
            assert len(tau)==3, 'Symmetric tensor should have length 3.'
            axx,axy,ayy = tau
            bf += [(axx,'ux','vx'),(axy,'uy','vx'),
                   (axy,'ux','vy'),(ayy,'uy','vy')]
        else:
            assert callable(tau) or isinstance(tau, Number)
            bf += [(tau,'ux','vx'),(tau,'uy','vy')]
    else:
        bf += [(1,'ux','vx'),(1,'uy','vy')]
    G = system.assemble(bilinear_forms=bf, 
                        boundary_conditions=boundary_conditions)
    G = G.tocsr()
    
    #
    # Lumped mass matrix
    # 
    M = system.assemble(bilinear_forms=[(1,'u','v')]).tocsr()
    m_lumped = np.array(M.sum(axis=1)).squeeze()
    
        
    if np.mod(alpha,2) == 0:
        #
        # Even power alpha
        # 
        Q = cholesky(G.tocsc())
        count = 1
    else:
        #
        # Odd power alpha
        # 
        Q = cholesky_AAt((G*sp.diags(1/np.sqrt(m_lumped))).tocsc())
        count = 2
    
    while count < alpha:
        #
        # Update Q
        #
        Q = cholesky_AAt((G*sp.diags(1/m_lumped)*Q.apply_Pt(Q.L())).tocsc()) 
        count += 2
    
    return Q

         
# =============================================================================
# Gaussian Markov Random Field Class
# =============================================================================
class Gmrf(object):
    """
    Gaussian Markov Random Field
    """
 
    def __init__(self, mu=None, precision=None, covariance=None, 
                 mesh=None, element=None, discretization=None):
        """
        Constructor
        
        Inputs:
        
        
            mu: double, (n,) vector of expectations (default=0)
            
            precision: double, (n,n) sparse/full precision matrix
                    
            covariance: double, (n,n) sparse/full covariance matrix
            
            mesh: Mesh, quadtree mesh
            
            element: QuadFE, finite element
            
            discretization: str, 'finite_elements' (default), or
                'finite_differences'.
                
            
        Attributes:
        
            __Q: double, precision matrix
            
            __Sigma: double, covariance matrix
            
            __mu: double, expected value
            
            __b: double, Q\mu (useful for sampling)
            
            __f_prec: double, lower triangular left cholesky factor of precision
                If Q is sparse, then use CHOLMOD.
                
            __f_cov: double, lower triangular left cholesky factor of covariance
                If Sigma is sparse, we use CHOLMOD.
                
            mesh: Mesh, Quadtree mesh
            
            element: QuadFE, finite element    
            
            discretization: str, 'finite_elements', or 'finite_differences' 
            
        """   
        n = None
        #
        # Need at least one
        #
        if precision is None and covariance is None:
            raise Exception('Specify precision or covariance (or both).')  
        #
        # Precision matrix
        # 
        Q = None
        if precision is not None:    
            if sp.isspmatrix(precision):
                #
                # Precision is sparse matrix
                # 
                n = precision.shape[0]
                Q = precision
                self.__f_prec = cholesky(Q.tocsc())
                #
                # Precision is cholesky factor
                # 
            elif type(precision) is Factor:
                n = len(precision.P())
                Q = (precision.L()*precision.L().transpose())
                self.__f_prec = precision
            else:
                #
                # Precision is full matrix
                #
                n = precision.shape[0]
                Q = precision 
                self.__f_prec = np.linalg.cholesky(precision)
        self.__Q = Q
        #
        # Covariance matrix
        # 
        self.__Sigma = covariance
        if covariance is not None:
            n = covariance.shape[0]
            if sp.isspmatrix(covariance):
                self.__f_cov = cholesky(covariance.tocsc())
            else:
                # Most likely
                self.__f_cov  = np.linalg.cholesky(covariance)    
        #
        # Check compatibility
        # 
        if covariance is not None and precision is not None:
            n_cov = covariance.shape[0]
            n_prc = precision.shape[0]
            assert n_prc == n_cov, \
                'Incompatibly shaped precision and covariance.'
            isI = precision.dot(covariance)
            if sp.isspmatrix(isI):
                isI = isI.toarray()
                assert np.allclose(isI, np.eye(n_prc),rtol=1e-10),\
               'Covariance and precision are not inverses.' 
        #
        # Mean
        # 
        if mu is not None:
            assert len(mu) == n, 'Mean incompatible with precision/covariance.'
        else: 
            mu = np.zeros(n)
        self.__mu = mu
        # 
        # b = Q\mu
        # 
        if not np.allclose(mu, np.zeros(n), 1e-10):
            # mu is not zero
            b = self.Q_solve(mu)
        else:
            b = np.zeros(n)
        self.__b = b
        #
        # Store
        # 
        self.__n = n    
        #
        # Store mesh and elements if available
        #
        if mesh is not None:
            self.mesh = mesh
        if element is not None:
            self.element = element
        
    @classmethod
    def from_covariance_kernel(cls, cov_fn, mesh, element=None):
        """
        Initialize Gmrf from covariance function
        """ 
        #
        # Discretize the covariance function
        # 
        if element is None:
            #
            # Pointwise evaluation of the kernel
            #
            x = mesh.quadvertices()
            X,Y = np.meshgrid(x,x)
             
        else:
            #
            # Finite element discretization of the kernel
            # 
            pass
    
    @classmethod
    def from_matern_pde(cls, alpha, kappa, mesh, element=None, tau=None):
        """
        Initialize finite element Gmrf from matern PDE
        
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
            
        Q = matern_precision(mesh, element, alpha, kappa, tau)
        return cls(precision=Q, mesh=mesh, element=element)
    
    
    def Q(self, i=None, j=None):
        """
        Return the precision matrix
        """
        return self.__Q
    
    
    def Sigma(self):
        """
        Return the covariance matrix
        """
        return self.__Sigma
        
    
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
        
        
    def mu(self,n_copies=None):
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
            return np.tile(self.__mu, (n_copies,1)).transpose()
        else:
            return self.__mu
        
    
    def b(self):
        """
        Return Q\mu
        """
        return self.__b
    
    
    def n(self):
        """
        Return the dimension of the random vector 
        """
        return self.__n
    
    
    def Q_solve(self, b):
        """
        Return the solution x of Qx = b by successively solving 
        Ly = b for y and hence L^T x = y for x.
        
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
    
    
    def kl_expansion(self, k=None):
        """
        Inputs:
        
        Outputs:
        
        """
        pass
    
    
    def sample(self, n_samples=None, z=None, mode='precision'):
        """
        Generate sample realizations from Gaussian random field.
        
        Inputs:
        
            n_samples: int, number of samples to generate
            
            z: (n,n_samples) random vector ~N(0,I).
            
            
        Outputs:
        
            x: (n,n_samples), samples paths of random field
            
            
        Note: Samples generated from the cholesky decomposition of Q are 
            different from those generated from that of Sigma. 
                
                Q = LL' (lower*upper)
                  
            =>  S = Q^(-1) = L'^(-1) L^(-1) (upper*lower) 
        """
        assert self.mode_supported(mode), \
            'Mode "'+ mode + '" not supported for this random field.'
        #
        # Preprocess z   
        # 
        if z is None:
            assert n_samples is not None, \
                'Specify either random array or sample size.'
            z = np.random.normal(size=(self.n(), n_samples))
            z_is_a_vector = False
        else:
            #
            # Extract number of samples from z
            #  
            if len(z.shape) == 1:
                nz = 1
                z_is_a_vector = True
            else:
                nz = z.shape[1]
                z_is_a_vector = False 
            assert n_samples is None or n_samples == nz, \
                'Sample size incompatible with given random array.'
            n_samples = nz
        #
        # Generate centered realizations
        # 
        if mode in ['precision','canonical']:
            v = self.Lt_solve(z, mode='precision')
        elif mode == 'covariance':
            v = self.L(z, mode='covariance')    
        #
        # Add mean
        # 
        if z_is_a_vector:
            return v + self.mu()
        else:
            return v + self.mu(n_samples)
        
    
    def mode_supported(self, mode):
        """
        Determine whether enough information is available to process given mode
        """
        if mode == 'precision':
            return self.__Q is not None
        elif mode == 'covariance':
            return self.__Sigma is not None
        elif mode == 'canonical':
            return self.__Q is not None
        else:
            raise Exception('For modes, use "precision", ' + \
                            '"covariance", or "canonical".')
            
    
    def condition(self, constraint=None, constraint_type='pointwise',
                  mode='precision'):
        """
        
        Inputs:
        
            constraint: tuple, parameters specifying the constraint, determined
                by the constraint type:
                
                'pointwise': (dof_indices, constraint_values) 
                
                'hard': (A, b) 
                
                'soft': (A, Q)
        
            constraint_type: str, 'pointwise' (default), 'hard', 'soft'.
            
            mode: str, 'precision' (default), or 'covariance'.
            
            
        Output:
        
            X: Gmrf, conditioned random field. 
        """
        if constraint_type == 'pointwise':
            i_b, x_b = constraint
            i_a = [i not in i_b for i in range(self.n())]
            mu_a, mu_b = self.mu()[i_a], self.mu()[i_b]
            #Q_aa = self.Q(i_a,i_a), Q_ab = self.Q(i_a,i_b)
            #
            # Conditional random field
            # 
            #mu_agb = mu_a - spla.spsolve(Q_aa, Q_ab.dot(x_b-mu_b))
            #return Gmrf(mu=mu_agb, precision=Q_aa)
            
        elif constraint_type == 'hard':
            pass
        elif constraint_type == 'soft':
            pass
        else:
            raise Exception('Input "constraint_type" should be:' + \
                            ' "pointwise", "hard", or "soft"')
    
    
