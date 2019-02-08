'''
Created on Feb 8, 2017

@author: hans-werner
'''
from fem import Assembler
from fem import FiniteElement
from fem import DofHandler
from fem import GaussRule
from fem import Function
from fem import Assembler
from mesh import Mesh
from mesh import Mesh1D
from mesh import QuadMesh

from numbers import Number, Real
import numpy as np
from scipy import linalg
from scipy.special import kv, gamma
import scipy.sparse as sp
from scipy.sparse import linalg as spla
from sksparse.cholmod import cholesky, cholesky_AAt, Factor  # @UnresolvedImport


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
    
        x,y: (n,dim) column vectors
        
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
    
    if len(x.shape) == 1:
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
    elif len(x.shape) == 2 and x.shape[1]==2:
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
    if len(x.shape) == 1:
        #
        # 1D
        # 
        if M is None:
            print('M is None')
            sgm**2 + x*y
            return sgm**2 + x*y
        else:
            assert isinstance(M,Real), 'Input "M" should be a scalar.'
            return x*M*y
        
    elif len(x.shape) == 2 and x.shape[1]==2:
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


def gaussian(x, y, sgm=1, l=1, M=None, periodic=False):
    """
    Squared exponential covariance function
    
        C(x,y) = exp(-|x-y|^2/(2l^2))
    
    """
    d = distance(x, y, M, periodic=periodic)
    return sgm**2*np.exp(-d**2/(2*l**2))


def exponential(x, y, sgm=1, l=0.1, M=None, periodic=False):
    """
    Exponential covariance function
    
        C(x,y) = exp(-|x-y|/l)
        
    Inputs: 
    
        x,y: np.array, spatial points
        
        l: range parameter
    """
    d = distance(x, y, M, periodic=periodic)
    return sgm**2*np.exp(-d/l)


def matern(x, y, sgm, nu, l, M=None, periodic=False):
    """
    Matern covariance function
    
    Inputs:
    
        x,y: np.array, spatial points
        
        sgm: variance
        
        nu: shape parameter (k times differentiable if nu > k)
        
        l: range parameter 
        
    Source: 
    """
    d = distance(x, y, M, periodic=periodic)
    K = sgm**2*2**(1-nu)/gamma(nu)*(np.sqrt(2*nu)*d/l)**nu*\
        kv(nu,np.sqrt(2*nu)*d/l)
    #
    # Modified Bessel function undefined at d=0, covariance should be 1
    #
    K[np.isnan(K)] = 1
    return K
    
    
def rational(x, y, a, M=None, periodic=False):
    """
    Rational covariance
    
        C(x,y) = 1/(1 + |x-y|^2)^a
         
    """
    d = distance(x, y, M, periodic=periodic)
    return (1/(1+d**2))**a   


class CovKernel(object):
    """
    Covariance kernel
    """
    def __init__(self, name, parameters):
        """
        Constructor
        
        Inputs:
        
            name: str, name of covariance kernel 
                'constant', 'linear', 'gaussian', 'exponential', 'matern', 
                or 'rational'
            
            parameters: dict, parameter name/value pairs (see functions for
                allowable parameters.
        
        """
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
        self.__fn = cov_fn
        self.__parameters = parameters
        
    
    def eval(self,x,y):
        """
        Evaluate kernel at the points x and y
        
        Inputs:
        
            x,y: (n_points, dim) array of spatial points
            
                
        Output:
        
            k: double, (n_points,) kernel value, at points 
        """
        return self.__fn(x,y, **self.__parameters)
    
    
    def slice(self, x, pos):
        """
        Evaluate kernel only in one argument
        
        
        Inputs:
        
            x: double, (n_points,dim) array of spatial points
            
            pos: int, variable position at which kernel is evaluated
            
            
        Output:
        
            slice: list, of n_points functions, each of which corresponds to
                an entry of the array x.
            
        """
        n_points = x.shape[0]
        kernel_slice = []
        for i in range(n_points):
            if pos==0:
                #
                # Resolve kernel in first argument
                # 
                f = lambda y: self.eval(np.repeat(x[i,:],n_points),y)
            elif pos==1:
                #
                # Resolve kernel in second argument
                # 
                f = lambda x: self.eval(x, np.repeat(x[i,:],n_points))
            #
            # Add function to list
            # 
            kernel_slice.append(f)
        return kernel_slice
              
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
            
            etype: str, finite element space (see FiniteElement for
                supported spaces).
                
            assembly_type: str, specifies type of approximation,
                projection, or collocation
                
            

        """
        
        self.__kernel = CovKernel(name, parameters)
        assert isinstance(element, FiniteElement), \
        'Input "element" must be of type FiniteElement.'
            
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

class Precision(object):
    """
    Precision Matrix for 
    """
    pass

# =============================================================================
# Gaussian Markov Random Field Class
# =============================================================================
class Gmrf(object):
    """
    Gaussian Markov Random Field
    
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

    
    
    @staticmethod
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
                function (viz. fem.Assembler)
            
            
        Outputs:
        
            Q: sparse matrix, in CSC format
            
        """
        system = Assembler(mesh, element)
        
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
 
 
    def __init__(self, mesh, mu=None, kernel=None, precision=None, 
                 covariance=None, element=None):
        """
        Constructor
        
        Inputs:
        
            mesh: Mesh, Computational mesh
        
            mu: Function, random field expectation (default=0)
            
            precision: double, (n,n) sparse/full precision matrix
                    
            covariance: double, (n,n) sparse/full covariance matrix
                    
            element: QuadFE, finite element
                
            
        Attributes:
        
            __Q: double, precision matrix
            
            __Sigma: double, covariance matrix
            
            __mu: double, expected value
            
            __b: double, Q\mu (useful for sampling)
            
            __f_prec: double, lower triangular left cholesky factor of precision
                If Q is sparse, then use CHOLMOD.
                
            __f_cov: double, lower triangular left cholesky factor of covariance
                If Sigma is sparse, we use CHOLMOD.
                
            __dim: int, effective dimension
            
                
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
                try:
                    self.__f_cov = cholesky(covariance.tocsc())
                except np.linalg.linalg.LinAlgError:
                    print('It seems a linalg error occured') 
            else:
                # Most likely
                try:
                    self.__f_cov = np.linalg.cholesky(covariance)
                except np.linalg.linalg.LinAlgError as ex:
                    if ex.__str__() == 'Matrix is not positive definite':
                        #
                        # Rank deficient covariance
                        # 
                        # TODO: Pivoted Cholesky
                        self.__f_cov = None
                        self.__svd = np.linalg.svd(covariance)  
                    else:
                        raise Exception('I give up.')
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
        # Store size of matrix
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
    def from_covariance_kernel(cls, cov_name, cov_par, mesh, \
                               mu=None, element=None):
        """
        Initialize Gmrf from covariance function
        
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
        #cov_fn = globals()['Gmrf.'+cov_name+'_cov']
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
            Sigma = sp.diags(1/m_lumped)*Sigma
            
        return cls(mu=mu, covariance=Sigma, mesh=mesh, element=element, \
                   discretization=discretization)
    
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
            
        Q = Gmrf.matern_precision(mesh, element, alpha, kappa, tau)
        return cls(precision=Q, mesh=mesh, element=element)
    
    
    
    
    
    def Q(self):
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
            if n_copies == 1:
                return self.__mu
            else:
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
    
    
    def rank(self):
        """
        Return the rank of the covariance/precision matrix
        
        Note: If the matrix is degenerate, we must use the covariance's
            or precision's eigendecomposition.
        """
        pass
    
    
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
    
    
    def KL(self, precision=None, k=None):
        """
        Inputs:
        
        Outputs:
        
        """
        mesh = self.mesh()
        
    
    
    def sample(self, n_samples=None, z=None, mode='precision'):
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
            if self.__f_cov is not None:
                v = self.L(z, mode='covariance')
            elif self.__svd is not None:
                U,s,_ = self.__svd
                v = U.dot(np.dot(np.sqrt(np.diag(s)), z))  
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
                  mode='precision', output='gmrf', n_samples=1, z=None):
        """
        
        Inputs:
        
            constraint: tuple, parameters specifying the constraint, determined
                by the constraint type:
                
                'pointwise': (dof_indices, constraint_values) 
                
                'hard': (A, b), where A is the (k,n) constraint matrix and 
                    b is the (k,m) array of realizations (usually m is None).
                
                'soft': (A, Q)
        
            constraint_type: str, 'pointwise' (default), 'hard', 'soft'.
            
            mode: str, 'precision' (default), or 'covariance', or 'svd'.
            
            output: str, type of output 'gmrf', 'sample', 'log_pdf' 
            
        Output:
        
            X: Gmrf, conditioned random field. 
            
        TODO: Unfinished
        """
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
                return Gmrf(mu=mu_agb, precision=Q_aa)
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
    
    
    def iid_gauss(self, n_samples):
        """
        Returns a matrix whose columns are N(0,I) vectors of length n 
        """
        if n_samples == 1:
            return np.random.normal(self.n())
        elif n_samples > 1:
            return np.random.normal(size=(self.n(),n_samples)) 
        