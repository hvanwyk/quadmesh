"""
Example 2: Adaptive mesh refinement for a simple diffusion dominated 
    steady state advection-diffusion problem
    
Step 1: Simulate forward problem 
Step 2: Compute adjoint solution
Step 3: Perform adaptive mesh refinement
Step 4: Sample forward problem for stochastic input
Step 5: Sample refinement indicators (classify)
Step 6: Try linearization (Yanzhao's method)    
"""
from finite_element import System, QuadFE
from mesh import Mesh
from plot import Plot
import matplotlib.pyplot as plt
from scipy.sparse import linalg as spla
import numpy as np

#
# Problem parameters
# 
k = 3.2  # diffusivity
v = 3
# Boundary marker functions
def bnd_inflow(x,y):
    return (np.abs(x-0)<1e-7)

def bnd_sides(x,y):
    return np.logical_or(np.abs(y-0)<1e-9, np.abs(y-1)<1e-9)

# Dirichlet boundary 
def g_hom(x,y):
    return 0

def g_inflow(x,y):
    return np.sin(np.pi*y) 

#
# Primal system
# 
element_u = QuadFE(2,'Q1')
bc_u = {'dirichlet': [(bnd_sides, g_hom), 
                    (bnd_inflow, g_inflow)]}
bf_u = [(k,'ux','vx'),(k,'uy','vy'),(v,'ux','v')]
lf_u = [(0,'v')]


#
# Adjoint system
# 
def J(x,y):
    """
    Kernel for evaluating int_A u dx, where A in [0,1]^2.
    """
    z = np.zeros(x.shape)
    x_rng = np.logical_and(x>=0.375, x<=0.625)   
    y_rng = np.logical_and(y>=0.375, y<=0.625) 
    in_block = np.logical_and(x_rng, y_rng)
    z[in_block] = 1
    return z

bc_z = {'dirichlet': [(bnd_sides, g_hom), (bnd_inflow, g_hom)]}
bf_z = [(k,'ux','vx'),(k,'uy','vy'),(-v,'ux','v')]
lf_z = [(J,'v')] 
element_z = QuadFE(2,'Q2')


#
# Initialize mesh
# 
mesh = Mesh.newmesh(grid_size=(4,4))
mesh.refine()


for i_refinement in range(6):
    # 
    # Solve primal system
    # 
    system_u = System(mesh, element_u, n_gauss=(4,16))
    A,b = system_u.assemble(bilinear_forms=bf_u, 
                            linear_forms=lf_u, 
                            boundary_conditions=bc_u)
    ua = spla.spsolve(A.tocsc(), b)


    #
    # Solve adjoint system using higher order fem
    # 
    system_z = System(mesh, element_z, n_gauss=(4,16))
    Az, bz = system_z.assemble(bilinear_forms=bf_z, 
                               linear_forms=lf_z, 
                               boundary_conditions=bc_z)
    za = spla.spsolve(Az.tocsc(),bz)


    # -------------------------------------------------------------------------
    # Error estimator
    # -------------------------------------------------------------------------
    #
    # Compute the residuals on each cell
    # 
    opposite = {'W':'E', 'E':'W', 'S':'N', 'N':'S'}
    cell_errors = []
    for node in mesh.root_node().find_leaves(): 
        cell = node.quadcell()
        
        #
        # Interpolate z -> zh
        #
        x_loc = system_u.x_loc(cell)
        z_dofs = system_z.get_global_dofs(node)
        zh_loc = system_z.f_eval_loc(za[z_dofs], cell, x=x_loc)  # evaluate z at linear dof vertices
        z_gauss = system_z.f_eval_loc(za[z_dofs], cell)
        zh_gauss = system_u.f_eval_loc(zh_loc, cell)
        dz = z_gauss - zh_gauss
        
        #
        # Compute cell residual
        # 
        u_dofs = system_u.get_global_dofs(node)
        u_loc = ua[u_dofs]
        uxx_gauss = system_u.f_eval_loc(u_loc, cell, derivatives=(2,0,0))
        uyy_gauss = system_u.f_eval_loc(u_loc, cell, derivatives=(2,1,1))
        ux_gauss = system_u.f_eval_loc(u_loc, cell, derivatives=(1,0))
        f_gauss = system_u.f_eval_loc(0, cell)
        
        res_cell = system_z.form_eval(((f_gauss*dz,),), node) + \
                   k*system_z.form_eval(((uxx_gauss*dz,),), node) + \
                   k*system_z.form_eval(((uyy_gauss*dz,),), node) - \
                   system_z.form_eval(((v*ux_gauss*dz,),), node)
        
        #
        # Compute edge residual
        # 
        res_edge = 0
        
        for direction in ['W','E','S','N']:
            #
            # Compute gradient of u on cell
            #                
            nu = cell.normal(cell.get_edges(direction))
            
             
                  
            nbr = node.find_neighbor(direction)
            subpos = {'W':['SW','NW'],'E':['NE','SE'],
                      'S':['SE','SW'], 'N':['NE','NW']}
            if nbr is not None:
                if nbr.has_children():
                    # Neighbor has children
                    for pos in subpos[opposite[direction]]:
                        child = nbr.children[pos]
                        
                        # Compute Gauss nodes on adjacent cell's edge
                        edge = child.quadcell().get_edges(opposite[direction])
                        edge_rule = system_u.edge_rule()
                        x_gauss = edge_rule.map(edge)
                        w_gauss = edge_rule.weights()*edge_rule.jacobian(edge)
                        
                        # Compute ux and uy at Gauss nodes 
                        ux_cell = \
                            system_u.f_eval_loc(u_loc, cell, \
                                                derivatives=(1,0), x=x_gauss) 
                        
                        uy_cell = \
                            system_u.f_eval_loc(u_loc, cell, \
                                                derivatives=(1,1), x=x_gauss)
                            
                        nbr_dofs = system_u.get_global_dofs(child)
                        u_nbr = ua[nbr_dofs]
                        ux_nbr = \
                            system_u.f_eval_loc(u_nbr, nbr.quadcell(), \
                                                derivatives=(1,0), x=x_gauss)
                            
                        uy_nbr = \
                            system_u.f_eval_loc(u_nbr, nbr.quadcell(), \
                                                derivatives=(1,1), x=x_gauss)
                            
                        jump = 0.5*nu[0]*k*(ux_cell-ux_nbr) + \
                               0.5*nu[1]*k*(uy_cell-uy_nbr)
                        
                        # Compute z-zh at the Gauss points
                        z_gauss = \
                            system_z.f_eval_loc(za[z_dofs], cell, x=x_gauss)
                            
                        zh_gauss = \
                            system_u.f_eval_loc(zh_loc, cell, x=x_gauss)
                            
                        # Update edge residual
                        res_edge += np.sum(w_gauss*jump*(z_gauss-zh_gauss))
                        
                elif nbr.depth == node.depth - 1:
                    # Neighbor is larger than cell
                    
                    # Compute Gauss node on own cell's edge
                    edge = cell.get_edges(direction)
                    edge_rule = system_u.edge_rule()
                    x_gauss = edge_rule.map(edge)
                    w_gauss = edge_rule.weights()*edge_rule.jacobian(edge)
                    
                    # Compute ux and uy at Gauss nodes 
                    ux_cell = \
                        system_u.f_eval_loc(u_loc, cell, \
                                            derivatives=(1,0), x=x_gauss) 
                    
                    uy_cell = \
                        system_u.f_eval_loc(u_loc, cell, \
                                            derivatives=(1,1), x=x_gauss)
                        
                    nbr_dofs = system_u.get_global_dofs(nbr)
                    u_nbr = ua[nbr_dofs]
                    ux_nbr = \
                        system_u.f_eval_loc(u_nbr, nbr.quadcell(), \
                                            derivatives=(1,0), x=x_gauss)
                        
                    uy_nbr = \
                        system_u.f_eval_loc(u_nbr, nbr.quadcell(), \
                                            derivatives=(1,1), x=x_gauss)
                        
                    jump = 0.5*nu[0]*k*(ux_cell-ux_nbr) + \
                           0.5*nu[1]*k*(uy_cell-uy_nbr)
                    
                    # Compute z-zh at the Gauss points
                    z_gauss = \
                        system_z.f_eval_loc(za[z_dofs], cell, x=x_gauss)
                        
                    zh_gauss = \
                        system_u.f_eval_loc(zh_loc, cell, x=x_gauss)
                        
                    # Update edge residual
                    res_edge += np.sum(w_gauss*jump*(z_gauss-zh_gauss))
                    
                else:
                    # Gauss nodes on edge
                    edge = cell.get_edges(direction)
                    edge_rule = system_u.edge_rule()
                    x_gauss = edge_rule.map(edge)
                    w_gauss = edge_rule.weights()*edge_rule.jacobian(edge)
                    
                    # Compute ux and uy at Gauss nodes 
                    ux_cell = \
                        system_u.f_eval_loc(u_loc, cell, \
                                            derivatives=(1,0), x=x_gauss) 
                    
                    uy_cell = \
                        system_u.f_eval_loc(u_loc, cell, \
                                            derivatives=(1,1), x=x_gauss)
                        
                    nbr_dofs = system_u.get_global_dofs(nbr)
                    u_nbr = ua[nbr_dofs]
                    ux_nbr = \
                        system_u.f_eval_loc(u_nbr, nbr.quadcell(), \
                                            derivatives=(1,0), x=x_gauss)
                        
                    uy_nbr = \
                        system_u.f_eval_loc(u_nbr, nbr.quadcell(), \
                                            derivatives=(1,1), x=x_gauss)
                        
                    jump = 0.5*nu[0]*k*(ux_cell-ux_nbr) + \
                           0.5*nu[1]*k*(uy_cell-uy_nbr)
                    
                    # Compute z-zh at the Gauss points
                    z_gauss = \
                        system_z.f_eval_loc(za[z_dofs], cell, x=x_gauss)
                        
                    zh_gauss = \
                        system_u.f_eval_loc(zh_loc, cell, x=x_gauss)
                        
                    # Update edge residual
                    res_edge += np.sum(w_gauss*jump*(z_gauss-zh_gauss))
            
                
        cell_errors.append(res_cell-res_edge)
    
    cell_errors = np.array(cell_errors)

    #
    # Mark cells
    # 
    i_sort = np.argsort(np.abs(cell_errors))
    cell_errors_sorted = np.abs(cell_errors[i_sort])
    total_error = np.sum(cell_errors_sorted)
    partial_error = 0
    tht = 0.8
    j = 0
    while partial_error < tht*total_error:
        partial_error += cell_errors_sorted[-(j+1)]
        j += 1
    threshold = cell_errors_sorted[-(j+1)]
    
    count = 0
    for node in mesh.root_node().find_leaves():
        if np.abs(cell_errors[count]) >= threshold:
            node.mark(i_refinement)
        count += 1
    
    plot = Plot()
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.semilogy(cell_errors_sorted)
    ax.plot(threshold, 'r.')
    ax = fig.add_subplot(122)
    
    fig, ax = plot.contour(ax, fig, np.abs(cell_errors), mesh, element_u)
    plt.show()
    
    mesh.refine(flag=i_refinement)
    mesh.balance()
    
    #ax = plot.mesh(ax,mesh,element_u, color_marked=[i_refinement])
    

# 
# Solve primal system
# 
system_u = System(mesh, element_u, n_gauss=(4,16))
A,b = system_u.assemble(bilinear_forms=bf_u, 
                        linear_forms=lf_u, 
                        boundary_conditions=bc_u)
ua = spla.spsolve(A.tocsc(), b)


#
# Solve adjoint system using higher order fem
# 
system_z = System(mesh, element_z, n_gauss=(4,16))
Az, bz = system_z.assemble(bilinear_forms=bf_z, 
                           linear_forms=lf_z, 
                           boundary_conditions=bc_z)
za = spla.spsolve(Az.tocsc(),bz)

#
# Plot results
# 
plot = Plot()

fig = plt.figure()
ax = fig.add_subplot(1,3,1)
fig, ax = plot.contour(ax, fig, J, mesh, element_u)
ax.set_title('primal')

ax = fig.add_subplot(1,3,2)
fig, ax = plot.contour(ax, fig, za, mesh, element_z)
ax.set_title('adjoint')
#ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0.1))

#ax = fig.add_subplot(1,3,3)
#fig, ax = plot.contour(ax, fig, np.abs(cell_errors), mesh, element_u)
#ax.set_title('errors')

#ax = fig.add_subplot(1,1,1)


#ax = plot.contour(ax, fig, u, mesh, element)
plt.show()