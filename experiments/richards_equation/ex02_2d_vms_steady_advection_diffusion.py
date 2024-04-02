"""
Variational Multiscale Method for Advection-Diffusion Equation


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

# Define the mesh
mesh = QuadMesh(box=domain, resolution=(10,5))

# Various refinement levels
for i in range(3):
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
    
    
#
# Plot meshes 
#  
fig, ax = plt.subplots(3,3)  
for i in range(3):

    ax[i,0] = plot.mesh(mesh,axis=ax[i,0], 
                      regions=[('inflow','edge'),('outflow','edge')],
                      subforest_flag=i)
    ax[i,0].set_xlabel('x')
    ax[i,0].set_ylabel('y')
#plt.show()


#
# Define DofHandlers and Basis 
#

# Piecewise Constant Element
Q0 = QuadFE(2,'DQ0')  # element
dh0 = DofHandler(mesh,Q0)  # degrees of freedom handler
dh0.distribute_dofs()
v0 = [Basis(dh0, subforest_flag=i) for i in range(3)]

# Piecewise Linear 
Q1 = QuadFE(2,'Q1')  # linear element
dh1 = DofHandler(mesh,Q1)  # linear DOF handler
dh1.distribute_dofs()

v1   = [Basis(dh1,'v',i) for i in range(3)]   
v1_x = [Basis(dh1,'vx',i) for i in range(3)]
v1_y = [Basis(dh1,'vy',i) for i in range(3)]


# 
# Parameters
# 
a1 = Constant(1)  # advection parameters
a2 = Constant(-0.5) 

#
# Random diffusion coefficient
#
cov = Covariance(dh0,name='matern',parameters={'sgm': 1,'nu': 1, 'l':0.1})
Z = GaussianField(dh0.n_dofs(), K=cov)

# Sample from the diffusion coefficient
q2 = Nodal(basis=v0[2], data=Z.sample())
 

#
# Compute the spatial average
# 
q = []
for i in range(2):
    #
    # Define Problem (v[i], v[i]) = (q, v[i]) 
    # 
    problem = [[Form(trial=v0[i],test=v0[i]), Form(kernel=q2, test=v0[i])]]
    assembler = Assembler(problem,mesh=mesh,subforest_flag=2)
    assembler.assemble()
    
    M = assembler.get_matrix()
    b = assembler.get_vector()
    
    solver = LinearSystem(v0[i], M, b)
    solver.solve_system()
    qi = solver.get_solution()
    q.append(qi)
q.append(q2)

# Plot realizations of the diffusion coefficient
for i,qi in enumerate(q):
    ax[i,1] = plot.contour(qi,axis=ax[i,1],colorbar=False)
    ax[i,1].set_axis_off()
#plt.show()



#
# Solve the Linear System on Each Mesh
# 
xi = [Kernel(qi,F=lambda q: 0.01 + np.exp(q)) for qi in q]
u = []
for i in range(3):
    
    # Weak form
    problem = [Form(kernel=xi[i],test=v1_x[2], trial=v1_x[2]), 
               Form(kernel=xi[i],test=v1_y[2], trial=v1_y[2]),
               Form(kernel=a1, test=v1[2], trial=v1_x[2]),
               Form(kernel=a2, test=v1[2], trial=v1_y[2]),
               Form(kernel=0, test=v1[2])]
    # Initialize 
    assembler = Assembler(problem, mesh=mesh, subforest_flag=2)
    
    # Add Dirichlet conditions 
    assembler.add_dirichlet('inflow', 1)
    assembler.add_dirichlet('outflow', 0)
    
    # Assemble system
    assembler.assemble()
    
    # Solve system
    ui = assembler.solve()
    u.append(Nodal(basis=v1[2], data=ui))
    

#fig, ax = plt.subplots(3,1)
for i,ui in enumerate(u):
    print(ui.basis().n_dofs())
    ax[i,2] = plot.contour(ui,axis=ax[i,2],colorbar=True)
    ax[i,2].set_axis_off()
plt.tight_layout()
plt.show()   


#
# Conditional distribution of the fine scale, given the coarse scale. 
# 
fig, ax = plt.subplots(3,3)

# Coarse scale diffusion coefficient
Z0 = q[0].data()
ax[0,0] = plot.contour(q[0],axis=ax[0,0],colorbar=True)

# 
# Sequential Conditional Samples from q0
# 

# Step 1: Sample from q0
print('Sampling from q0')
Z0 = Z.sample()

# Step 2: Condition q1 on q0
print('Condintioning q1 on q0')
M10 = local_average_operator(mesh, v0[0], v0[1], flag0=0, flag1=1)
Z1  = GaussianField(dh1.n_dofs(), K=cov)


Z10 = Z.condition(M10, Z0, n_samples=1)

print('conditioning q2 on q0')
M20 = local_average_operator(mesh, v0[0], v0[2], flag0=0, flag1=2)
Z20 = Z.condition(M20, Z0, n_samples=1)

print('conditioning q2 on q1')
M21 = local_average_operator(mesh, v0[1], v0[2], flag0=1, flag1=2)
Z21 = Z.condition(M21, Z10, n_samples=1)




ax[0,1] = plot.contour(Nodal(basis=v0[1], data=Z10),axis=ax[0,1],colorbar=True)
ax[0,2] = plot.contour(Nodal(basis=v0[2], data=Z20),axis=ax[0,2],colorbar=True)
ax[1,1] = plot.contour(Nodal(basis=v0[1], data=Z10),axis=ax[1,1],colorbar=True)
ax[1,2] = plot.contour(Nodal(basis=v0[2], data=Z21),axis=ax[1,2],colorbar=True)

plt.show()
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