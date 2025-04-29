"""
Conditional Sampling

"""
import sys

if '/home/hans-werner/git/quadmesh/src' not in sys.path:
    sys.path.append('/home/hans-werner/git/quadmesh/src')

import gmrf
from mesh import QuadMesh
from fem import Basis, DofHandler, QuadFE   
from function import Nodal
from assembler import Assembler, Form
from gmrf import Covariance, GaussianField

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from plot import Plot

from scipy.sparse import linalg as spla
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np


def plot01_mesh(mesh,L):
    """
    Plot computational mesh at various levels

    Inputs:

        mesh: QuadMesh, computational mesh

        L: int, number of levels
    """
    plot = Plot(quickview=False)
    fig, ax = plt.subplots(1,L,figsize=(6,1.5))
    for l in range(L):
        ax[l] = plot.mesh(mesh,axis=ax[l], subforest_flag=l)
        ax[l].set_title('Level %d' % l)
    plt.tight_layout()
    foldername = '/home/hans-werner/Dropbox/work/research/projects/spatially_indexed_noise/phd_notes/fig/'
    filename = foldername+'mesh_refinement.eps'
    fig.savefig(filename, dpi=400)


def plot02_kl_expansion(rough_field, smooth_field, basis):
    """
    Plot Karhunen-Loeve expansions of a Gaussian field.

    (a) Plot the fields at medium, high, and full resolution
    (b) Plot the eigenvalue decay

    Inputs:

        rough_field: GaussianField, rough Gaussian field

        smooth_field : GaussianField, smooth Gaussian field
    """
    foldername = '/home/hans-werner/Dropbox/work/research/projects/spatially_indexed_noise/phd_notes/fig/'
    plot = Plot(quickview=False)
    
    #
    # Plot eigenvalue decay
    #
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    d_rough = rough_field.covariance().get_eigenvalues()
    d_smooth = smooth_field.covariance().get_eigenvalues()
    ax.semilogy(d_rough, '-', label='Rough')
    ax.semilogy(d_smooth, '-', label='Smooth')
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue')
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    filename = foldername+'eigenvalue_decay.eps'
    fig.savefig(filename) 
    
    #
    # Karhunen-Loeve expansion
    #
    i = 0
    eig_cutoffs = [0.75, 0.95, 1]
    for d, field in zip([d_rough, d_smooth], [rough_field, smooth_field]):
        # Get the levels from cutoffs
        n_eigs = []
        for cutoff in eig_cutoffs:
            n = np.sum(d.cumsum()/d.sum() < cutoff)
            n = min(n, len(d)-1)
            n_eigs.append(n)
    
        n_cutoffs = len(eig_cutoffs)
        z = np.random.normal(size=(field.size(),1))
        ax, fig = plt.subplots(1,n_cutoffs,figsize=(6,2))
        for i in range(n_cutoffs):
            zi = z[0:n_eigs[i]+1,:]
            q_smpl = field.KL_sample(i_max=n_eigs[i], z=zi) 
            qfn = Nodal(data=q_smpl, basis=basis)
            ax = plot.contour(qfn, axis=fig[i])
            
            ax.set_title(r'$\varepsilon = %.2f$ ($k=%d$)' % (eig_cutoffs[i], n_eigs[i]+1))
        plt.tight_layout()
        if i==0:
            figname = foldername+'kl_sample_rough.png'
        else:
            figname = foldername+'kl_sample_smooth.png'
        
        plt.savefig(figname, dpi=400)

        i += 1


def plot03_kl_conditional(field,basis):
    """
    Plot tree of conditional plots of the field at various levels
    """
    plot = Plot(quickview=False)
    foldername = '/home/hans-werner/Dropbox/work/research/projects/spatially_indexed_noise/phd_notes/fig/'

    eig_cutoffs = [0.75, 0.95, 1]
    d = field.covariance().get_eigenvalues()
    n_eigs = []
    for cutoff in eig_cutoffs:
        n = np.sum(d.cumsum()/d.sum() < cutoff)
        n = min(n, len(d)-1)
        n_eigs.append(n)


    fig, ax = plt.subplots(3,3,figsize=(6,5))
    for col in range(3):
        if col == 0:
            q_samples = field.KL_sample(i_max=n_eigs[col], n_samples=3)
            q_fn = Nodal(data=q_samples, basis=basis)
            for row in range(3):
                ax[row,col] = plot.contour(q_fn, axis=ax[row,col], n_sample=row)

            # Choose sample to condition on
            i_ref = np.random.randint(0,3)
            q_ref = q_samples[:,[i_ref]]
            ax[i_ref,col].patch.set_linewidth(2)
            ax[i_ref,col].patch.set_edgecolor('red')
        else:   
            dq_samples = field.KL_sample(i_min=n_eigs[col-1], i_max=n_eigs[col], n_samples=3)
            print('dq_samples',dq_samples.shape)
            print('q_ref',q_ref.shape)
            qq = np.tile(q_ref,(1,3))
            print('qq',qq.shape)
            q_samples = np.tile(q_ref,(1,3))+dq_samples
            q_fn = Nodal(data=q_samples, basis=basis)
            for row in range(3):
                ax[row,col] = plot.contour(q_fn, axis=ax[row,col], n_sample=row)

            # Choose sample to condition on
            i_ref = np.random.randint(0,3)
            q_ref = q_samples[:,[i_ref]]

            if col < 2:
                ax[i_ref,col].patch.set_linewidth(2)
                ax[i_ref,col].patch.set_edgecolor('red')

    for col in range(3):
        ax[0,col].set_title(r'$k=%d$' % (n_eigs[col]+1))
    plt.tight_layout()
    figname = foldername+'kl_conditional.png'
    plt.savefig(figname, dpi=400)
    plt.show()


def plot04_spatial(q, v0):
    """
    Plot projection of the field at various levels
    """
    plot = Plot(quickview=False)
    fig, ax = plt.subplots(1,3,figsize=(6,2))
    l = [1,2,4]

    # Figure out the projections
    Q = []
    for i in range(3):
        print('i',i)
        if i == 2:
            Q.append(np.eye(v0[l[i]].n_dofs()))
        else:
            Q.append(np.eye(v0[l[i+1]].n_dofs()))
            print('l[i]',l[i])
            print('l[i+1]',l[i+1])
            for ll in range(l[i+1]-1, l[i]-1, -1):
                print('ll',ll)
                Q[i] =P[ll].dot(Q[i])
                #print('Q[i]',Q[i].shape)
        print('Q[i]',Q[i].shape)       
        

    q_samples = q[l[-1]].sample(n_samples=1)
    q_fn = Nodal(data=q_samples, basis=v0[l[-1]])
    ax[2] = plot.contour(q_fn, axis=ax[2], n_sample=0)
    ax[2].set_title(r'$n=%d$' % (v0[l[-1]].n_dofs()))
    for i in range(1,-1,-1):
        q_samples = Q[i].dot(q_samples)
        q_fn = Nodal(data=q_samples, basis=v0[l[i]])
        ax[i] = plot.contour(q_fn, axis=ax[i], n_sample=0)
        ax[i].set_title(r'$n=%d$' % (v0[l[i]].n_dofs()))
    """
    for i in range(3):
        q_samples = q[l[i]].sample(n_samples=1)
        q_fn = Nodal(data=q_samples, basis=v0[l[i]])
        print('q_samples',q_samples.shape)
        n_dofs = v0[l[i]].n_dofs()
        print('basis dofs',n_dofs)
        ax[i] = plot.contour(q_fn, axis=ax[i], n_sample=0)
        ax[i].set_title(r'$n=%d$' % (n_dofs))
    """
    plt.tight_layout()
    figname = '/home/hans-werner/Dropbox/work/research/projects/spatially_indexed_noise/phd_notes/fig/field_refinement.png'
    plt.savefig(figname, dpi=400)
    plt.show()

def plot05_spatial_projection_conditional(q, v0, P):
    """
    Plot conditional sampling at various levels
    """
    # Figure out the projections
    plot = Plot(quickview=False)
    fw = 2  # figure width
    l = [0,1,2,3,4]
    L = len(l)
    fig, ax = plt.subplots(3,5,figsize=(fw*L,fw*3))
    

    # Figure out the projections
    Q = []
    for i in range(L):
        print('i',i)
        if i == L-1:
            Q.append(np.eye(v0[l[i]].n_dofs()))
        else:
            Q.append(np.eye(v0[l[i+1]].n_dofs()))
            for ll in range(l[i+1]-1, l[i]-1, -1):
                Q[i] =P[ll].dot(Q[i])

    
    for j in range(L):
        ii = np.random.randint(0,3)
    
        if j==0:
            q_samples = q[l[j]].sample(n_samples=3)
            q_fn = Nodal(data=q_samples, basis=v0[l[j]])
            for i in range(3):
                if i==ii:
                    ax[i,j] = plot.contour(q_fn, axis=ax[i,j], n_sample=i,colorbar=False)
                else:
                    ax[i,j] = plot.contour(q_fn, axis=ax[i,j], n_sample=i, cmap='gray',colorbar=False)
                ax[i,j].set_axis_off()
            ax[0,0].set_title(r'$n=%d$' % v0[l[j]].n_dofs())
        else:
            q_samples = q[l[j]].condition(Q[j-1], q_ref, n_samples=3)
            q_fn = Nodal(data=q_samples, basis=v0[l[j]])
            for i in range(3):
                if i==ii:
                    ax[i,j] = plot.contour(q_fn, axis=ax[i,j], n_sample=i,colorbar=False)
                else:
                    ax[i,j] = plot.contour(q_fn, axis=ax[i,j], n_sample=i, cmap='gray',colorbar=False)
                ax[i,j].set_axis_off()
            ax[0,j].set_title(r'$n=%d$' % v0[l[j]].n_dofs())
            
        #ii = np.random.randint(0,3)
        q_ref = q_samples[:,[ii]]
        if j < L-1:
            ax[ii,j].patch.set_linewidth(2)
            ax[ii,j].patch.set_edgecolor('red')

    plt.tight_layout()
    figname = '/home/hans-werner/Dropbox/work/research/projects/spatially_indexed_noise/phd_notes/fig/conditional_sampling_rough_v02.png'
    plt.savefig(figname, dpi=400)
    plt.show()

def plot06_conditional_independence(q, P):
    """
    Plot a row from (i) the fine-scale covariance matrix, and 
    (ii) the conditional covariance matrix.
    """
    plot = Plot(quickview=False)
    L = len(P)
    fig, ax = plt.subplots(2,L+1,figsize=(2*(L+1),2*2))
    


    # Fine scale covariance matrix
    K_fine = q[-1].covariance().get_matrix()
    sgm_fine = np.sqrt(np.diag(K_fine))
    R_fine = K_fine/np.outer(sgm_fine, sgm_fine)
    n_dofs = v0[-1].n_dofs()
    ax[0,0].imshow(R_fine,cmap='coolwarm', vmin=-1,vmax=1)
    ax[0,0].set_title(r'$R_{%d}$'%(n_dofs))
    ax[0,0].set_axis_off()
    q_fn_fine = Nodal(data=R_fine[:,500], basis=v0[-1])
    ax[1,0] = plot.contour(q_fn_fine, axis=ax[1,0], colorbar=False, n_sample=0, cmap='coolwarm',vmin=-1,vmax=1)
    ax[1,0].set_axis_off()
    for l in range(len(P)):
        if l == 0:
            A = P[-1].toarray()
        else:
            A = P[-(l+1)].toarray().dot(A)

        n_dofs = A.shape[0]

    #A = P[-1].toarray()

        KAT = K_fine.dot(A.T)
        AKAT = A.dot(KAT)
        K_cond = K_fine - KAT.dot(np.linalg.solve(AKAT, KAT.T))
        sgm_cond = np.sqrt(np.diag(K_cond))        
        R_cond = K_cond/np.outer(sgm_cond, sgm_cond)
        R_cond_row = Nodal(data=R_cond[:,500], basis=v0[-1])
        ax[0,L-l].imshow(R_cond, cmap='coolwarm', vmin=-1,vmax=1)
        ax[0,L-l].set_axis_off()
        ax[0,L-l].set_title(r'$R^c_{%d}$'%(n_dofs))
        ax[1,L-l] = plot.contour(R_cond_row, axis=ax[1,L-l], n_sample=0, colorbar=False, cmap='coolwarm',vmin=-1,vmax=1)
        ax[1,L-l].set_axis_off()


    q_sample = q[-2].sample(n_samples=1)
    
    K_cond = q[-1].condition(P[-1], q_sample, output='field').covariance().get_matrix().toarray()
    
    plt.tight_layout()
    figname = '/home/hans-werner/Dropbox/work/research/projects/spatially_indexed_noise/phd_notes/fig/conditional_independence_rough.png'
    plt.savefig(figname, dpi=400)
    plt.show()
    

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 11
})


# New mesh
mesh = QuadMesh(box=[-1,1,-1,1], resolution=(2,2))


# Refine the mesh
L = 5  # number of refinements
for l in range(L):
    if l==0:
        mesh.record(0)
    else:
        mesh.cells.refine(new_label=l)

# Generate plot of the mesh
#plot01_mesh(mesh, L)


# Define the function space 
Q0 = QuadFE(mesh.dim(), 'DQ0')
dhQ0 = DofHandler(mesh, Q0)
dhQ0.distribute_dofs()


v0 = [Basis(dhQ0, 'v',subforest_flag=l) for l in range(L)]

#
# Compute the spatial projection operators
# 
P = []
for l in range(L-1):
    #
    # Define the problem (v[l-1], v[l-1]) = (v[l], v[l-1])
    # 
    problems = [[Form(trial=v0[l],test=v0[l])], 
                [Form(trial=v0[l+1], test=v0[l])]]
    
    assembler = Assembler(problems, mesh=mesh, subforest_flag=l+1)
    assembler.assemble()

    M = assembler.get_matrix(i_problem=0).tocsc()
    A = assembler.get_matrix(i_problem=1).tocsc()
    Pl = spla.spsolve(M,A)
    P.append(Pl)
   

# Define the covariance and Gaussian field at various refinement levels
Cr, Cs = [], []
qr, qs = [], []
for l in range(L):
    Cr.append(Covariance(dhQ0, name='matern', parameters={'sgm': 1,'nu': 1, 'l':0.1},subforest_flag=l))
    Cs.append(Covariance(dhQ0, name='matern', parameters={'sgm': 1,'nu': 1, 'l':0.7},subforest_flag=l))
    qr.append(GaussianField(dhQ0.n_dofs(subforest_flag=l), covariance=Cr[l]))
    qs.append(GaussianField(dhQ0.n_dofs(subforest_flag=l), covariance=Cs[l]))

#plot02_kl_expansion(qr[-1], qs[-1],v0[-1])
#plot03_kl_conditional(qs[-1],v0[-1])
#plot04_spatial(qs, v0)
plot05_spatial_projection_conditional(qr, v0, P)
plot06_conditional_independence(qr, P)

n_cutoffs = len(eig_cutoffs)
z = np.random.normal(size=(q[-1].size(),1))
print(z.shape)
ax, fig = plt.subplots(1,n_cutoffs,figsize=(6,2))
for i in range(n_cutoffs):
    zi = z[0:n_eigs[i]+1,:]
    print(zi.shape)
    q_smpl = q[-1].KL_sample(i_max=n_eigs[i], z=zi) 
    qfn = Nodal(data=q_smpl, basis=v0[-1])
    ax = plot.contour(qfn, axis=fig[i])
    ax.set_title(r'$\varepsilon = %.2f$ ($k=%d$)' % (eig_cutoffs[i], n_eigs[i]+1))
plt.tight_layout()
figname = '/home/hans-werner/Dropbox/work/research/projects/spatially_indexed_noise/phd_notes/fig/kl_sample_rough.png'
#plt.savefig(figname, dpi=400)
plt.show()


#
# Conditional sampling
#
q = []
for l in range(L-1):
    q[l] = GaussianField(dhQ0.n_dofs(subforest_flag=l), covariance=C[l])
    if l == 0:
        q    
    else:
        q[l] = GaussianField(dhQ0.n_dofs(subforest_flag=l), covariance=C[l])
    q[l+1] = q[l].condition(P[l], q[l+1], n_samples=1)

# conditional sampling
#Sample at the coarsest level
"""
fig = plt.figure(figsize=(3*(L),9))

gs = GridSpec(3, L, figure=fig)
for i in range(3):
    q_data = []
    for l in range(L):
        if l==0:
            q_data.append(q[l].sample())
        else:
            q_data.append(q[l].condition(P[l-1],q_data[l-1],n_samples=1))
    for l in range(L):
        ax = fig.add_subplot(gs[i, l])
        q_fn = Nodal(data = q_data[l], basis=v0[l])
        ax = plot.contour(q_fn, axis=ax)

plt.tight_layout()


fig = plt.figure(figsize=(3*(L),9))
gs = GridSpec(3, L, figure=fig)
for l in range(3):
    if l==0:
        q_d = q[l].sample()
        q_fn = Nodal(data = q_d, basis=v0[l])
        ax = fig.add_subplot(gs[1, l])
        ax = plot.contour(q_fn, axis=ax)
    elif l==1:
        q_d = q[l].condition(P[l-1],q_d,n_samples=1)
        q_fn = Nodal(data=q_d, basis=v0[l])
        ax = fig.add_subplot(gs[1, l])
        ax = plot.contour(q_fn, axis=ax)
    elif l==2:
        q_d = q[l].condition(P[l-1],q_d,n_samples=3)
        q_fn = Nodal(data = q_d, basis=v0[l])
        for i in range(3):
            ax = fig.add_subplot(gs[i, l])
            ax = plot.contour(q_fn, axis=ax,n_sample=i)
        q0_data = q_d[:,0]
        q1_data = q_d[:,1]
        q2_data = q_d[:,2]

#q0_data = q[3].condition(P[2],q0_data,n_samples=1)
q0_fn = Nodal(data = q[3].condition(P[2],q0_data), basis=v0[3])
ax = fig.add_subplot(gs[0, 3])
ax = plot.contour(q0_fn, axis=ax)
    
"""    

"""
#q1_data = q[l].condition(P[l-1],q1_data)
q1_fn = Nodal(data = q1_data, basis=v0[2])
ax = fig.add_subplot(gs[1, l])
ax = plot.contour(q1_fn, axis=ax)

#q2_data = q[l].condition(P[l-1],q2_data)
q2_fn = Nodal(data = q2_data, basis=v0[2])
ax = fig.add_subplot(gs[2, l])
ax = plot.contour(q2_fn, axis=ax)
"""
plt.tight_layout()
plt.show()
            


"""
q0 = q.sample()
for l in range(L-2,-1,-1):
    q0 = P[l].dot(q0)
q0_fn = Nodal(data=q0, basis=v0[0])
"""


fig = plt.figure(figsize=(14, 6))
gs = GridSpec(3, 7, figure=fig)
ax01 = fig.add_subplot(gs[0, 3])
q0_fn = Nodal(data = q_data[0], basis=v0[0])
ax01 = plot.contour(q0_fn, axis=ax01)

#n1 = dhQ0.n_dofs(subforest_flag=1)
"""
q1 = GaussianField(n1, covariance=T1.dot(C.dot(T1.T)))
q1 = q.sample(n_samples=2)
for l in range(L-2,0,-1):
    q1 = P[l].dot(q1)
q1_fn = Nodal(data=q1, basis=v0[1])
"""

