"""
Compute QoI PDFs for the 2D advection-diffusion problem

    -div[ xi(x,w) grad(u) ] + a*grad(u) = 0,  x in domain
    u(x) = 1, for x on inflow boundary
    u(x) = 0, for x on outflow boundary 

where xi(x,w) is a random field with log-normal distribution.
"""


import numpy as np
import matplotlib.pyplot as plt

from mesh import QuadMesh
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

fig, ax = plt.subplots(L,1,figsize=(5,15))  
for i in range(L):
    if i < L-1:
        ax[i] = plot.mesh(mesh,axis=ax[i], 
                        regions=[('inflow','edge'),('outflow','edge')],
                        subforest_flag=i)
    else:
        ax[i] = plot.mesh(mesh,axis=ax[i], 
                        regions=[('inflow','edge'),('outflow','edge'),('roi','cell')],
                        subforest_flag=i)    
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('y')
plt.tight_layout()
plt.show()



#
# Define DofHandlers and Basis 
#

# Piecewise Constant Element
Q0 = QuadFE(2,'DQ0')  # element
dh0 = DofHandler(mesh,Q0)  # degrees of freedom handler
dh0.distribute_dofs()
v0 = [Basis(dh0, subforest_flag=i) for i in range(L)]

# Piecewise Linear 
Q1 = QuadFE(2,'Q1')  # linear element
dh1 = DofHandler(mesh,Q1)  # linear DOF handler
dh1.distribute_dofs()

v1   = [Basis(dh1,'v',i) for i in range(L)]   
v1_x = [Basis(dh1,'vx',i) for i in range(L)]
v1_y = [Basis(dh1,'vy',i) for i in range(L)]


# 
# Parameters
# 
a1 = Constant(0.5)  # advection parameters
a2 = Constant(-0.5) 

#
# Random diffusion coefficient
#
cov = Covariance(dh0,name='matern',parameters={'sgm': 1,'nu': 1, 'l':0.2})
Z = GaussianField(dh0.n_dofs(), covariance=cov)

#
# Compute the spatial projection operators
# 
M = []
A = []
for l in range(L-1):
    #
    # Define the problem (v[l-1], v[l-1]) = (v[l], v[l+1])
    # 
    problems = [[Form(trial=v0[l],test=v0[l])], 
                [Form(trial=v0[l+1], test=v0[l])]]
    
    assembler = Assembler(problems, mesh=mesh, subforest_flag=l+1)
    assembler.assemble()

    M.append(assembler.get_matrix(i_problem=0).tocsc())
    A.append(assembler.get_matrix(i_problem=1).tocsc())
    
    #P.append(spla.spsolve(M,A))




#
# Define Flux accross boundary (Quantity of interest)
#
def Flux(xi,u,ux,uy,a1,a2, region=None):
    """
    Compute -(xi*grad u + a*u) * n
    """
    n = region.unit_normal()
    return -xi*(ux*n[0]+uy*n[1])+u*(a1*n[0]+a2*n[1])


Q_ave_dom = [[] for l in range(L)]  # average over domain
Q_ave_reg = [[] for l in range(L)]  # average over region of interest
Q_outflow = [[] for l in range(L)]  # flux accross outflow


# Sample the random field
n_sample = 10
for i in range(n_sample):
    # Sample from the diffusion coefficient at the finest level
    #qL = Nodal(basis=v0[-1], data=Z.sample())

    # Upscale to the coarser levels
    #
    # Solve the Linear System on Each Mesh
    # 
    #xi = [Kernel(qi,F=lambda q: 0.01 + np.exp(q)) for qi in q]
    #u = []
    print('Sample %d' % i)
    for l in range(L-1,-1,-1):
        
        

        # Sample log-diffusion coefficient at level l
        if l==L-1:
            # Sample at the finest level
            logq = Z.sample()
        else:
            # Upscale log-diffusion coefficient 
            logq = spsolve(M[l],A[l].dot(logq))
            
        # Define diffusion coefficient
        xi = Nodal(basis=v0[l], data=0.01 + np.exp(logq))
        
        # Weak form
        problem = [Form(kernel=xi,test=v1_x[-1], trial=v1_x[-1]), 
                   Form(kernel=xi,test=v1_y[-1], trial=v1_y[-1]),
                   Form(kernel=a1, test=v1[-1], trial=v1_x[-1]),
                   Form(kernel=a2, test=v1[-1], trial=v1_y[-1]),
                   Form(kernel=0, test=v1[-1])]
        

        # Initialize assembler (at finest level)
        assembler = Assembler(problem, mesh=mesh, subforest_flag=L-1)
        
        # Add Dirichlet conditions 
        assembler.add_dirichlet('inflow', 1)
        assembler.add_dirichlet('outflow', 0)
        
        # Assemble system
        assembler.assemble()
        
        # Solve system
        u_vec = assembler.solve()
        
        u = Nodal(data=u_vec, basis=v1[-1])
        ux = Nodal(data=u_vec, basis=v1_x[-1])
        uy = Nodal(data=u_vec, basis=v1_y[-1])

        # Compute QoI's
        k_flux = Kernel(f=[xi,u,ux,uy,a1,a2], F=Flux) 
        p_flux = [Form(kernel=k_flux, flag='outflow', dmu='ds')]
        p_int_dom = [Form(kernel=Kernel(u))]
        p_int_reg = [Form(kernel=Kernel(u), flag='roi')]
        p_area_dom  = [Form(kernel=1)]
        p_area_reg  = [Form(kernel=1, flag='roi')]

        problems = [p_flux, p_int_dom, p_area_dom, p_int_reg, p_area_reg]
        assembler = Assembler(problems, mesh=mesh, subforest_flag=L-1)
        assembler.assemble()

        Q_outflow[l].append(assembler.get_scalar(i_problem=0))
        Q_ave_dom[l].append(assembler.get_scalar(i_problem=1)/assembler.get_scalar(i_problem=2))
        Q_ave_reg[l].append(assembler.get_scalar(i_problem=3)/assembler.get_scalar(i_problem=4))

        
        print(f'Flux={Q_outflow[l][-1]}')
        print(f'Average={Q_ave_dom[l][-1]}')
        print(f'Local Average={Q_ave_reg[l][-1]}')
            # First sample, plot the field, the solution