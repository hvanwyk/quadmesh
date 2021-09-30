from fem import DofHandler, Basis, QuadFE
from gmrf import GaussianField, Covariance
from assembler import Form, Assembler
from mesh import Mesh1D, QuadMesh
from plot import Plot
from function import Nodal
import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt

"""
Goal: 
    
Investigate optimal mesh density for reproducing the statistics of a given 
quantity of interest. Let q ~ Gaussian random field with expressed at some
maximum resolution and let 

    Q = Var(J(q))

where

    J(q) = I[0.75,1] q(x) dx. 
    
The field is discretized using piecewise constant basis over the fine mesh. 

Experiments: 

    1. Greedy algorithm: Start off with coarse mesh. At every stage, choose r 
        cells to refine. Use brute-force method, enumerating over all 
        possibilities. 
    2. Optimization: Set up a mesh optimization problem. 
    
Implementation:
    
    a. Calculate variance for a given mesh. 
        i. Use the intermediate mesh (assembly every time), 
        ii. or the fine mesh with a lifting operator (we probably need the 
          projection in any case).
    b. Calculate the gradient somehow.  

"""
def projection_matrix(dofhandler, fine_flag, coarse_flag):
    """
    Project a piecewise constant function, defined on a fine scale mesh onto 
    a coarse scale mesh. 
    
    Inputs:
        
        dofhandler: DofHandler, for discontinuous piecewise constant elements
        
        fine_flag: str/int, mesh-flag for fine mesh
        
        coarse_flag: str/int, mesh-flag for coarse mesh
    
    
    Outputs: 
        
        P: double, sparse (n_dofs_coars, n_dofs_fine) matrix representation of
            the projection, 
            
    """
    assert dofhandler.element.element_type()=='DQ0', \
        'Only piecewise constant approximations supported.'
    
    mesh = dofhandler.mesh
    rows, cols, vals = [], [], []
    for leaf in mesh.cells.get_leaves(subforest_flag=fine_flag):
        # Iterate over fine mesh
        
        # Add leaf dof to list of columns
        cols.extend(dofhandler.get_cell_dofs(leaf))
        
        # Search for nearest ancestor in coarse grid
        ancestor = leaf
        while not ancestor.is_marked(coarse_flag):
            ancestor = ancestor.get_parent()
        
        # Record coarse cell dof
        rows.extend(dofhandler.get_cell_dofs(ancestor))
            
        # Determine the ratio in areas
        if mesh.dim()==1:
            # One-dimensional interval
            multiplier = leaf.length()/ancestor.length()
        elif mesh.dim()==2:
            # Two-dimensional cell
            multiplier = leaf.area()/ancestor.area()
        
        # Store the value 
        vals.append(multiplier)
    
    # 
    # Re-index rows and columns
    # 
    
    # Compute unique dofs 
    col_dofs = list(set(cols))
    row_dofs = list(set(rows))

    # Re-index using unique dofs
    rows = [row_dofs.index(i) for i in rows]
    cols = [col_dofs.index(i) for i in cols]
    
    #
    # Define sparse projection matrix 
    # 
    n_rows = len(row_dofs)
    n_cols = len(col_dofs)
        
    P = sp.coo_matrix((vals,(rows,cols)), shape=(n_rows,n_cols))
    
    return P


def error(dofhandler, q, L,  ):
    """
    Compute the error E(|J(q)-J(qhat)|^2) for a given sample of q's.  
    
    """
    
    
def compute_variance(dofhandler, q, L, submesh_flag=None):
    """
    Compute the variance of the quantity of interest 
    
        J(q) = I[0.75,1] q(x,y) dx
        
    On a given submesh. 
    
    Inputs:
        
        dofhandler: DofHandler, associated with the problem
        
        L: double, vector representing the linear operator on the finest mesh
        
        submesh_flag: str/int, representing the coarser mesh.
        
    Output:
        
        v: double, variance given by  
        
            v = L^T (PT*P) K PT*P L
    """ 
    #
    # Get covariance matrix 
    #
    K = q.covariance().get_matrix()
    n = q.size()
    
    if submesh_flag is None:
        # Identity Matrix
        P = np.eye(n)
    else:
        #
        # Get projection and relaxation matrix
        # 
        P = projection_matrix(dofhandler, None, submesh_flag)  # projection
    Pt = P.transpose()  # lifting operator
    PtP = Pt.dot(P)
    
    # Compute variance
    return L.dot(PtP.dot(K.dot(PtP.dot(L))))


def test01_projection():
    """
    Test projection operator
    """
    pass


def test02_variance():
    """
    Compute the variance of J(q) for different mesh refinement levels
    and compare with MC estimates. 
    """
    l_max = 8
    for i_res in np.arange(2,l_max):
        
        # Computational mesh
        mesh = Mesh1D(resolution=(2**i_res,))
            
        # Element 
        element = QuadFE(mesh.dim(), 'DQ0')
        dofhandler = DofHandler(mesh, element)
        dofhandler.distribute_dofs()
        
        # Linear Functional
        mesh.mark_region('integrate', lambda x: x>=0.75, entity_type='cell', 
                         strict_containment=False)
        phi = Basis(dofhandler)
        assembler = Assembler(Form(4,test=phi, flag='integrate')) 
        assembler.assemble()
        L = assembler.get_vector()
        
        # Define Gaussian random field
        C = Covariance(dofhandler, name='gaussian', parameters={'l':0.05})
        C.compute_eig_decomp()
    
        eta = GaussianField(dofhandler.n_dofs(), K=C)
        eta.update_support()
    
        n_samples = 100000
        J_paths = L.dot(eta.sample(n_samples=n_samples))
        var_mc = np.var(J_paths)
        lmd, V = C.get_eig_decomp() 
        LV = L.dot(V)
        var_an = LV.dot(np.diag(lmd).dot(LV.transpose()))
        
        print(var_mc, var_an)
    
    
def experiment01():
    """
    Compute the quantity of interest, it's expectation and variance
    """
    #
    # FE Discretization
    # 
    
    # Computational mesh
    mesh = Mesh1D(resolution=(64,))
    
    # Element 
    element = QuadFE(mesh.dim(), 'DQ0')
    dofhandler = DofHandler(mesh, element)
    dofhandler.distribute_dofs()
    
    # Linear Functional
    mesh.mark_region('integrate', lambda x: x>0.75, entity_type='cell', 
                     strict_containment=False)
    phi = Basis(dofhandler)
    assembler = Assembler(Form(1,test=phi, flag='integrate')) 
    assembler.assemble()
    L = assembler.get_vector()
   
    # Gaussian field


if __name__ == '__main__':
    #%% Test 2: Variance
    
    test02_variance()
    
    #%% Simple projection Matrix
    
    #
    # Define and record coarse mesh
    # 
    mesh = Mesh1D(resolution=(16,))
    mesh.record(0)
    
    #
    # Refine mesh and record
    #  
    
    """
    # One level of refinement
    mesh.cells.find_node([0]).mark('r')
    mesh.cells.refine(refinement_flag='r')
    
    mesh.cells.find_node([0,0]).mark('r')
    mesh.cells.refine(refinement_flag='r')
    """
    l_max = 4
    for i in range(l_max):
        mesh.cells.refine()
        mesh.record(i+1)
      
    
    

    # plot meshes
    plot = Plot(quickview=False)
    fig, ax = plt.subplots(l_max,1)
    for i in range(l_max):
        ax[i] = plot.mesh(mesh, axis=ax[i], subforest_flag=i)
    #ax[1] = plot.mesh(mesh, axis=ax[1], subforest_flag=None)
    plt.show()
    # 
    # Define piecewise constant elements
    # 
    element = QuadFE(mesh.dim(), 'DQ0')
    dofhandler = DofHandler(mesh, element)
    dofhandler.distribute_dofs()
    
    # Get projection matrix
    P = projection_matrix(dofhandler, None, 0)
    
    fig, ax = plt.subplots(1,1)
    for i in range(l_max):
        CC = Covariance(dofhandler, subforest_flag=i, name='gaussian', 
                        parameters={'l':0.01})
        CC.compute_eig_decomp()
        d, V = CC.get_eig_decomp()
        print(d)
        lmd = np.arange(len(d))
        ax.semilogy(lmd, d, '.-', label='level=%d'%i)
    plt.legend()
    plt.show()
    
    #
    # Define random field on the fine mesh
    #     
    C = Covariance(dofhandler, name='gaussian', parameters={'l':0.05})
    C.compute_eig_decomp()
    
    eta = GaussianField(dofhandler.n_dofs(), K=C)
    eta.update_support()
    
    #eta_path = Nodal(data=eta.sample(), basis=phi)
    eta0 = P.dot(eta.sample())
    eg0 = eta.condition(P, eta0, n_samples=100)
    eg0_paths = Nodal(data=eg0, basis=Basis(dofhandler))
    e0_path = Nodal(data=eta0, basis=Basis(dofhandler, subforest_flag=0))
    plot = Plot(quickview=False)
    ax = plt.subplot(111)
    for i in range(30):
        ax = plot.line(eg0_paths, axis=ax, mesh=mesh, i_sample=i, 
                       plot_kwargs={'color':'k', 'linewidth':0.5})
    ax = plot.line(e0_path, axis=ax, mesh=mesh, 
                   plot_kwargs={'color':'r', 'linewidth':2})
        
    ax.set_ylim([-3,3])
    plt.tight_layout()
    plt.show()
    
    
    # 
    # Define linear functional 
    # 
    mesh.mark_region('integrate', lambda x: x>0.75, entity_type='cell', 
                     strict_containment=False)
    phi = Basis(dofhandler)
    assembler = Assembler(Form(1,test=phi, flag='integrate')) 
    assembler.assemble()
    L = assembler.get_vector()
    
    
    n_samples = 1000
    J = {}
    for l in range(l_max+1):
        P = projection_matrix(dofhandler, None, l)
        J[l] = L.dot(P.T.dot(P.dot(eta.sample(n_samples=n_samples))))    
        plt.hist(J[l], bins=40, density=False, alpha=0.5, label=l)
    plt.legend()
    plt.show()
    
    # Compute variance
    var = np.zeros(l_max+1)
    h   = np.zeros(l_max+1)
    for i in range(l_max+1):
        var[i] = compute_variance(dofhandler, eta, L, submesh_flag=i)
        h[i] = mesh.cells.get_leaves(subforest_flag=i)[0].length()
    print(var)
    plt.loglog(h,var)    
    #%% 
    #
    # Initial coarse mesh 
    # 
    l_max = 4
    mesh = Mesh1D()
    for i in range(l_max):
        mesh.record(i)
        mesh.cells.refine()
    
    mesh.mark_region('integrate', lambda x: x>0.75, entity_type='cell', 
                     strict_containment=False)
    
    
    plot = Plot(time=0.1)
    plot.mesh(mesh,regions=[('integrate','cell')])
    
    
        
    # Finite Element Space
    DQ0 = QuadFE(1,'DQ0')
    dh_0 = DofHandler(mesh,DQ0)
    dh_0.distribute_dofs()
    n = dh_0.n_dofs() 
    
    leaves = mesh.cells.get_leaves()
    print(len(leaves))
    for cell in mesh.cells.get_leaves():
        #print(cell.get_root().info())
        #print(cell.get_parent().is_marked(l_max-1))
        c_dof = dh_0.get_cell_dofs(cell)[0]
        #print(c_dof)
        
        
    phi_0 = Basis(dh_0)
    psi_0 = Basis(dh_0, subforest_flag=l_max-1)
        
    #plot.mesh(mesh, dofhandler=dh)
    C = Covariance(dh_0, name='gaussian', parameters={'l':0.05})
    eta = GaussianField(n,K=C)
    eta_path = Nodal(data=eta.sample(), basis=phi_0)
    
    
    assembler = Assembler(Form(1,test=phi_0, flag='integrate')) 
    assembler.assemble()
    L = assembler.get_vector()
    print(L)
    #
    # Coarsening 
    #
    
    rows = []
    cols = []
    vals = []
    for leaf in mesh.cells.get_leaves():
        rows.extend(dh_0.get_cell_dofs(leaf.get_parent()))
        cols.extend(dh_0.get_cell_dofs(leaf)) 
        vals.append(0.5)
    
    #
    # Map to index 
    #
    
    # Rows
    coarse_dofs = list(set(rows))
    dof2idx = dict()
    for (dof,i) in zip(coarse_dofs,range(len(coarse_dofs))):
        dof2idx[dof] = i 
    rows = [dof2idx[dof] for dof in rows]
    
    # Columns
    fine_dofs = list(set(cols))
    dof2idx = dict()
    for (dof,i) in zip(fine_dofs,range(len(fine_dofs))):
        dof2idx[dof] = i 
    cols = [dof2idx[dof] for dof in cols]
    
    # Local averaging matrix
    R = sp.coo_matrix((vals,(rows,cols))).tocsc()
    
    
    # Average data
    ave_data = R.dot(eta_path.data())
    eta_ave = Nodal(data=ave_data, basis=psi_0)
    
    
    #
    # Plots
    # 
    plot = Plot(quickview=False)
    ax = plt.subplot(111)
    ax = plot.line(eta_path, axis=ax, mesh=mesh)
    ax = plot.line(eta_ave, axis=ax, mesh=mesh)
    ax.set_ylim([-3,3])
    plt.tight_layout()
    plt.show()
    
    #%%
    mesh = Mesh1D()
    
    # Coarse mesh
    mesh.cells.refine()
    mesh.record(0)
    
    
    """
    for i in range(3):
        for leaf in mesh.cells.get_leaves():
            if np.random.rand()>0.3:
                leaf.mark('r')
        mesh.cells.refine(refinement_flag='r')
    mesh.record(0)
    """
    plot = Plot(quickview=False)
    fig, ax = plt.subplots(1,2)
    ax[0] = plot.mesh(mesh, axis=ax[0], subforest_flag=0)
    
    # Fine submesh
    for i in range(3):
        for leaf in mesh.cells.get_leaves():
            if np.random.rand()>0.4:
                leaf.mark('r')
        mesh.cells.refine(refinement_flag='r')
        
    mesh.record(1)
    #mesh.balance(0)
    #mesh.balance()
    ax[1] = plot.mesh(mesh, axis=ax[1], subforest_flag=1)
    
    mesh.cells.get_child(0).info()
    
    DQ0 = QuadFE(mesh.dim(),'DQ0')
    dh = DofHandler(mesh,DQ0)
    dh.distribute_dofs()
    
    #for leaf in mesh.cells.get_leaves(subforest_flag=1):
    #    print(dh.get_cell_dofs(leaf))
    
    leaves = mesh.cells.get_leaves(subforest_flag=0)
    while len(leaves)!=0:
        leaf = leaves.pop()
        
        
    print(mesh.cells.depth())
    d2i = [[] for _ in range(mesh.cells.depth()+1)];
    print(len(d2i))
    for cell in mesh.cells.traverse(mode='breadth-first', flag=1):
        # Add cell dofs to that level
        cell.get_depth()
        d2i[cell.get_depth()].extend(dh.get_cell_dofs(cell))
    
    for level in d2i:
        level.sort()
        
    print(d2i)