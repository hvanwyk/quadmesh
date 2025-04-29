"""
Advection-Diffusion Equation with Hierarchical Random Diffusion Coefficients


TODO: 
- [ ] Generate projection operators 
- [ ] Evaluate statistical quantity of interest
- [ ] Implement Tasmanian sparse Gaussian grid interpolation/integration
"""
import sys

sys.path.append('/home/hans-werner/git/quadmesh/src/')

from mesh import QuadMesh
from fem import QuadFE, Basis, DofHandler
from function import Explicit, Nodal, Constant
from assembler import Assembler, Form, Kernel
from plot import Plot
import matplotlib.pyplot as plt
import numpy as np
from gmrf import Covariance, GaussianField
from diagnostics import Verbose
from scipy.sparse.linalg import spsolve
from solver import LinearSystem
from scipy.sparse import linalg as spla
from scipy import sparse
import Tasmanian as tas

# Initialize plot
plot = Plot(quickview=False)
comment = Verbose()

def solve_pde(mesh,v,vx,vy,a,xi):
    """
    Solve the underlying PDE
    
        -div[ xi(x,w) grad(u) ] + a*grad(u) = 0,  x in domain
        u(x) = 1, for x on inflow boundary
        u(x) = 0, for x on outflow boundary 
        
    Inputs:
    
        mesh: Quadmesh, on which problem is defined
                
        v, vx, vy: Basis, piecewise polynomial basis functions  
        
        a: double, (2,) vector of advection coefficients a = (ax,ay) 
        
        xi: Function, diffusivity coefficient.
        
        
    Output:
    
        u: finite element solution of the problem
        
    """
     # Weak form
    problem = [Form(kernel=xi,test=vx, trial=vx), 
               Form(kernel=xi,test=vy, trial=vy),
               Form(kernel=a1, test=v, trial=vx),
               Form(kernel=a2, test=v, trial=vy),
               Form(kernel=0, test=v)]
    
    # Initialize 
    assembler = Assembler(problem, mesh=mesh, subforest_flag=2)
    
    # Add Dirichlet conditions 
    assembler.add_dirichlet('inflow', 1)
    assembler.add_dirichlet('outflow', 0)
    
    # Assemble system
    assembler.assemble()
    
    # Solve system
    u_vec = assembler.solve()
    
    # Return finite element approximation
    return Nodal(basis=v, data=u_vec)


def local_average_operator(mesh, v0, v1, flag0=0, flag1=1):
    """
    Compute the local average operator from the fine to the coarse mesh.

    Inputs:

        mesh: QuadMesh, fine mesh with subforests labeled 0 (coarse) and 1 (fine).

        v0: Basis, piecewise constant basis on the coarse mesh.

        v1: Basis, piecewise linear basis on the fine mesh.

        flag0: int, subforest flag for the coarse mesh.

        flag1: int, subforest flag for the fine mesh.

    Outputs:

        M: scipy.sparse.csr_matrix, (n0, n1) local averaging operator
    """
        
    # Initialize local averaging operator
    M = np.zeros((v0.n_dofs(),v1.n_dofs()))
    
    problems = [[Form(trial=v1,test=v0)], [Form(trial=v0,test=v0)]]
    assembler = Assembler(problems, mesh=mesh, subforest_flag=flag1)
    assembler.assemble()
    M10 = assembler.get_matrix(i_problem=0).tocsr()
    M00 = assembler.get_matrix(i_problem=1)
    M = sparse.diags(1/M00.diagonal()).dot(M10)

    print('M:', 'type', type(M), 'size',M.shape, 'values', M.todense())
    


    return M


def average_flow_operator():
    pass


def conditional_diffusivity(q0, q, M, l_fine, n_samples=1):
    """
    Compute the conditional distribution of the fine scale, given the coarse scale.

    Inputs:

        q0: Nodal, (n0,1) coarse scale diffusion coefficient realization

        q: GaussianField, (n1,1) fine scale diffusion coefficient 

        M: scipy.sparse.csr_matrix, (n0, n1) local averaging operator

        l_fine: flag, indicating the level of the fine mesh

        n_samples: int, number of samples to draw from the conditional distribution


    Outputs:

        q_given_q0: Nodal, (n1, n_samples) conditional fine scale diffusion coefficient realizations
    """
    
    pass
#
# Mesh 
# 

# Computational domain
domain = [-2,2,-1,1]

# Boundary regions
infn = lambda x,y: (x==-2) and (-1<=y) and (y<=0)  # inflow boundary
outfn = lambda x,y: (x==2) and (0<=y) and (y<=1)  # outflow boundary

# Region of interest
dlt = 0.2
reg_fn = lambda x,y: (-dlt<=x) and (x<=dlt) and (-dlt<=y) and (y<=dlt)

# Define the mesh
mesh = QuadMesh(box=domain, resolution=(4,2))

# Various refinement levels
L = 4
for i in range(L):
    if i==0:
        mesh.record(0)
    else:
        mesh.cells.refine(new_label=i)
    
    # Mark inflow
    mesh.mark_region('inflow', infn, entity_type='half_edge', 
                     on_boundary=True, subforest_flag=i)
    
    # Mark outflow
    mesh.mark_region('outflow', outfn, entity_type='half_edge', 
                     on_boundary=True, subforest_flag=i)
    
    # Mark region of interest
    mesh.mark_region('roi', reg_fn, entity_type='cell', 
                     on_boundary=False, subforest_flag=i)
    
#
# Plot meshes 
#  
fig, ax = plt.subplots(L,4)  
for i in range(L):
    if i < L-1:
        ax[i,0] = plot.mesh(mesh,axis=ax[i,0], 
                        regions=[('inflow','edge'),('outflow','edge')],
                        subforest_flag=i)
    else:
        ax[i,0] = plot.mesh(mesh,axis=ax[i,0], 
                        regions=[('inflow','edge'),('outflow','edge'),('roi','cell')],
                        subforest_flag=i)    
    ax[i,0].set_xlabel('x')
    ax[i,0].set_ylabel('y')


#
# Define DofHandlers and Basis 
#

# Piecewise Constant Element
Q0 = QuadFE(2,'DQ0')  # element
dh0 = DofHandler(mesh,Q0)  # degrees of freedom handler
dh0.distribute_dofs()
v0 = [Basis(dh0, subforest_flag=i) for i in range(L)]

# Piecewise Linear 
Q2 = QuadFE(2,'Q2')  # linear element
dh1 = DofHandler(mesh,Q2)  # linear DOF handler
dh1.distribute_dofs()

v1   = [Basis(dh1,'v',i) for i in range(L)]   
v1_x = [Basis(dh1,'vx',i) for i in range(L)]
v1_y = [Basis(dh1,'vy',i) for i in range(L)]


# 
# Parameters
# 
a1 = Constant(1)  # advection parameters
a2 = Constant(-1) 

#
# Random diffusion coefficient
#
cov = Covariance(dh0,name='matern',parameters={'sgm': 1,'nu': 1, 'l':0.5})
Z = GaussianField(dh0.n_dofs(), covariance=cov)

Zs = Z.sample(n_samples=1)
print('Zs:', 'type', type(Zs), 'size',Zs.shape)

# Sample from the diffusion coefficient
qL = Nodal(basis=v0[-1], data=Z.sample())
 
print(qL.n_samples())
#
# Compute the spatial projection operators
# 
P = []
for l in range(L-1):
    #
    # Define the problem (v[l-1], v[l-1]) = (v[l], v[l+1])
    # 
    problems = [[Form(trial=v0[l],test=v0[l])], 
                [Form(trial=v0[l+1], test=v0[l])]]
    
    assembler = Assembler(problems, mesh=mesh, subforest_flag=l+1)
    assembler.assemble()

    M = assembler.get_matrix(i_problem=0).tocsc()
    A = assembler.get_matrix(i_problem=1)
    P.append(spla.spsolve(M,A))
    print('P:', 'type', type(P[-1]), 'size',P[-1].shape)




q = []
for i in range(L-1):
    #
    # Define Problem (v[i], v[i]) = (q, v[i]) 
    # 
    problem = [[Form(trial=v0[i],test=v0[i]), Form(kernel=qL, test=v0[i])]]
    assembler = Assembler(problem,mesh=mesh,subforest_flag=L-1)
    assembler.assemble()
    
    M = assembler.get_matrix()
    b = assembler.get_vector()
    
    solver = LinearSystem(v0[i], M, b)
    solver.solve_system()
    qi = solver.get_solution()
    q.append(qi)
q.append(qL)

print('plotting q')
# Plot realizations of the diffusion coefficient
for i,qi in enumerate(q):
    ax[i,1] = plot.contour(qi,axis=ax[i,1],colorbar=False)
    ax[i,1].set_axis_off()
#plt.show()
print('done plotting q')


#
# Solve the Linear System on Each Mesh
# 
xi = [Kernel(qi,F=lambda q: 0.01 + np.exp(q)) for qi in q]
u = []
for i in range(L):
    
    # Weak form
    problem = [Form(kernel=xi[i],test=v1_x[-1], trial=v1_x[-1]), 
               Form(kernel=xi[i],test=v1_y[-1], trial=v1_y[-1]),
               Form(kernel=a1, test=v1[-1], trial=v1_x[-1]),
               Form(kernel=a2, test=v1[-1], trial=v1_y[-1]),
               Form(kernel=0, test=v1[-1])]
    
    # Initialize 
    assembler = Assembler(problem, mesh=mesh, subforest_flag=L-1)
    
    # Add Dirichlet conditions 
    assembler.add_dirichlet('inflow', 1)
    assembler.add_dirichlet('outflow', 0)
    
    # Assemble system
    assembler.assemble()
    
    # Solve system
    ui = assembler.solve()
    u.append(Nodal(basis=v1[-1], data=ui))
    
    if i>0:
        ei = Nodal(basis=v1[-1], data = (u[i].data() - u[i-1].data()))
        ax[i,3] = plot.contour(ei,axis=ax[i,3],colorbar=True)  
        ax[i,3].set_axis_off()  
#fig, ax = plt.subplots(3,1)
for i,ui in enumerate(u):
    print(ui.basis().n_dofs())
    ax[i,2] = plot.contour(ui,axis=ax[i,2],colorbar=True)
    ax[i,2].set_axis_off()
plt.tight_layout()
plt.show()   


"""
#
# Conditional distribution of the fine scale, given the coarse scale. 
# 

# Unconditional distributions at each level

# Level 2:
K2 = Z.covariance()  # fine scale covariance matrix

# Level 1:
# Local averaging operator
M21 = local_average_operator(mesh, v0[1], v0[2], flag0=1, flag1=2).toarray()

# Level 1 scale covariance matrix
K1 = M21.dot(K2.dot(M21.transpose())) 
print('K1:', 'type', type(K1), 'size',K1.shape)


# Level 1 scale diffusion coefficient
Z1 = GaussianField(dh0.n_dofs(subforest_flag=1), covariance=K1)

# Level 0:
# Local averaging operator
M10 = local_average_operator(mesh, v0[0], v0[1], flag0=0, flag1=1).toarray()
K0 = M10.dot(K1.dot(M10.transpose()))

# Level 0 scale diffusion coefficient
Z0 = GaussianField(dh0.n_dofs(subforest_flag=0), covariance=K0)

fig, ax = plt.subplots(1,3)
ax[0] = plot.contour(Nodal(basis=v0[0], data=Z0.sample()),axis=ax[0],colorbar=True)
ax[1] = plot.contour(Nodal(basis=v0[1], data=Z1.sample()),axis=ax[1],colorbar=True)
ax[2] = plot.contour(Nodal(basis=v0[2], data=Z.sample()),axis=ax[2],colorbar=True)
for i in range(3):
    ax[i].set_axis_off()

plt.show()


# Coarse scale diffusion coefficient
#Z0 = q[0].data()
#ax[0,0] = plot.contour(q[0],axis=ax[0,0],colorbar=True)


# 
# Sequential Conditional Samples from q0
# 


# Step 1: Sample from q0
print('Sampling from q0')
z0 = Z0.sample()
q0 = Nodal(basis=v0[0], data=z0)

# Step 2: Condition q1 on q0
print('Conditioning q1 on q0')
z10 = Z1.condition(M10, z0, n_samples=1)
#q1 = Nodal(basis=v0[1], data=z10)

print('conditioning q2 on q1')
M20 = local_average_operator(mesh, v0[0], v0[2], flag0=0, flag1=2)
z21 = Z.condition(M21, z10, n_samples=1)
#q2 = Nodal(basis=v0[2], data=z21)

fig, ax = plt.subplots(2,3)

# Plot the Gausssian field samples 
ax[0,0] = plot.contour(Nodal(basis=v0[0], data=z0),axis=ax[0,0],colorbar=True)
ax[0,1] = plot.contour(Nodal(basis=v0[1], data=z10),axis=ax[0,1],colorbar=True)
ax[0,2] = plot.contour(Nodal(basis=v0[2], data=z21),axis=ax[0,2],colorbar=True)

# Plot the diffusion coefficient samples
ax[1,0] = plot.contour(Nodal(basis=v0[0], data=0.01 + np.exp(z0)),axis=ax[1,0],colorbar=True)
ax[1,1] = plot.contour(Nodal(basis=v0[1], data=0.01 + np.exp(z10)),axis=ax[1,1],colorbar=True)
ax[1,2] = plot.contour(Nodal(basis=v0[2], data=0.01 + np.exp(z21)),axis=ax[1,2],colorbar=True)

plt.show()

"""
"""
K = assembler.get_matrix().tocsr()
b = assembler.get_vector()
x0 = assembler.assembled_bnd()
u0 = np.zeros((v10.n_dofs(),1))
int_dofs = assembler.get_dofs('interior')

u0[int_dofs,0] = spsolve(K,b-x0)

# Resolve Dirichlet conditions
dir_dofs, dir_vals = assembler.get_dirichlet(asdict=False)
u0[dir_dofs] = dir_vals



solver = LinearSystem(v10,K,b)
solver.solve_system()

u0 = solver.get_solution()
"""