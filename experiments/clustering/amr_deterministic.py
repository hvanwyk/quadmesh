"""
Basic AMR using the DWR method 
"""
import numpy as np
from numpy import exp as e
import scipy.sparse.linalg as spla
from mesh import Mesh, QuadCell
from fem import QuadFE, System, DofHandler, GaussRule
from plot import Plot
import matplotlib.pyplot as plt


def mollifier(x,y):
    """
    Evaluate the mollifier rho with support within a ball at the point x,y
    """
    d = np.sqrt(x**2+y**2)
    in_ball = np.zeros(x.shape, dtype=np.bool)
    in_ball[d < 1] = 1
    z = np.zeros(x.shape)
    z[in_ball] = e(-1/(1-d[in_ball]**2))
    return z


def example_1():
    """
    Pointwise evaluation of a smooth function 
    """
    #
    # Specify exact solution and data
    # 
    u = lambda x,y: x*(1-x)*y*(1-y)
    f = lambda x,y: 2*y*(1-y) + 2*x*(1-x)
    
    #
    # Mesh and finite elements
    # 
    mesh = Mesh.newmesh(grid_size=(8,8))
    mesh.refine()
    element_u = QuadFE(2,'Q1')
    system_u = System(mesh, element_u, n_gauss=(6,36))
    
    # 
    # (Bi)linear forms and boundary conditions
    # 
    lf = [(f,'v')]
    bf = [(1,'ux','vx'),(1,'uy','vy')]
    dir_bnd = lambda x,y: (np.abs(x)<1e-8) + (np.abs(x-1)<1e-8) + \
                          (np.abs(y)<1e-8) + (np.abs(y-1)<1e-8)
     
    bc = {'dirichlet': [(dir_bnd,u)], \
          'neumann': None, \
          'robin': None}
    #
    # Assemble and solve 
    # 
    Au,bu = system_u.assemble(bilinear_forms=bf, linear_forms=lf, \
                          boundary_conditions=bc)
    ua = spla.spsolve(Au.tocsc(), bu)
    
    # 
    # Evaluate  exact solution at dof vertices
    # 
    x = system_u.dof_vertices()
    ue = u(x[:,0],x[:,1])
    
    
    # -------------------------------------------------------------------------
    # Adjoint Equation
    # -------------------------------------------------------------------------
    element_z = QuadFE(2,'Q2')
    system_z = System(mesh, element_z, n_gauss=(6,36))  # ensure we hit mollif!
    
    # 
    # Pointwise evaluation functional 
    # 
    # Compute the volume enclosed by the mollifier
    rule2d = GaussRule(36,shape='quadrilateral')
    cell = QuadCell(box=[-1,1,-1,1])
    x_phys = rule2d.map(cell)
    jac = rule2d.jacobian(cell)
    w = rule2d.weights()
    I = np.sum(w*jac*mollifier(x_phys[:,0], x_phys[:,1]))
    
    """
    Make sure eps doesn't change anything
    
    eps = 0.1
    x0, y0 = 0.5, 0.5    
    J = lambda x,y: mollifier((x0-x)/eps, (y0-y)/eps)/eps**2
    cell = QuadCell(box=[x0-eps,x0+eps,y0-eps,y0+eps])
    jac = rule2d.jacobian(cell)
    x_phys = rule2d.map(cell)
    w = rule2d.weights()
    I2 = np.sum(w*jac*J(x_phys[:,0],x_phys[:,1]))
    print(I2-I)
    """
    
    #
    # Define functional kernel
    # 
    eps = 1e-2
    x_point, y_point = 0.5, 0.5
    J = lambda x,y: mollifier((x_point-x)/eps, (y_point-y)/eps)/eps**2/I
    
    z_linear = [(J,'v')]
    z_bilinear = [(1,'ux','vx'),(1,'uy','vy')]
    zero = lambda x,y: np.zeros(x.shape)
    bnd = lambda x,y: (np.abs(x)<1e-8) + (np.abs(x-1)<1e-8) + \
                      (np.abs(y)<1e-8) + (np.abs(y-1)<1e-8)
    z_bc = {'dirichlet': [(bnd,zero)] , 'neumann': None , 'robin': None} 
    Az, bz = system_z.assemble(bilinear_forms=z_bilinear, \
                               linear_forms=z_linear, \
                               boundary_conditions=z_bc)
    za = spla.spsolve(Az.tocsc(), bz)
    
    # -------------------------------------------------------------------------
    # Error estimator
    # -------------------------------------------------------------------------
    #
    # Compute the residuals on each cell
    # 
    opposite = {'W':'E', 'E':'W', 'S':'N', 'N':'S'}
    cell_errors = []
    for node in mesh.root_node().find_leaves():
        #
        # Cell residual (f+uxx+uyy)
        # 
        cell = node.quadcell()
        
        # Interpolate z -> zh
        x_loc = system_u.x_loc(cell)
        z_dofs = system_z.get_global_dofs(node)
        zh_loc = system_z.f_eval_loc(za[z_dofs], node, x=x_loc)  # evaluate z at linear dof vertices
        z_gauss = system_z.f_eval_loc(za[z_dofs], node)
        zh_gauss = system_u.f_eval_loc(zh_loc, node)
        dz = z_gauss - zh_gauss
        
        u_dofs = system_u.get_global_dofs(node)
        uxx_gauss = system_u.f_eval_loc(ua[u_dofs], node, derivatives=(2,0,0))
        uyy_gauss = system_u.f_eval_loc(ua[u_dofs], node, derivatives=(2,1,1))
        f_gauss = system_u.f_eval_loc(f, node)
        
        res_cell = system_z.form_eval(((f_gauss*dz,),), node) + \
                   system_z.form_eval(((uxx_gauss*dz,),), node) + \
                   system_z.form_eval(((uyy_gauss*dz,),), node)
        
        res_edge = 0
        for direction in ['W','E','S','N']:
            #
            # Compute gradient of u on cell
            # 
            ux_cell = system_u.f_eval_loc(ua[u_dofs], node, edge_loc=direction, \
                                        derivatives=(1,0))
            uy_cell = system_u.f_eval_loc(ua[u_dofs], node, edge_loc=direction, \
                                        derivatives=(1,1))
            nu = cell.normal(cell.get_edges(direction))
            
            #
            # Compute  
            # 
            nbr = node.find_neighbor(direction)
            if nbr is not None:
                nbr_dofs = system_u.get_global_dofs(nbr)
                nbr_cell = nbr.quadcell()
                ux_nbr = system_u.f_eval_loc(ua[nbr_dofs], nbr, \
                                           edge_loc=opposite[direction], \
                                           derivatives=(1,0))
                
                uy_nbr = system_u.f_eval_loc(ua[nbr_dofs], nbr, \
                                           edge_loc=opposite[direction], \
                                           derivatives=(1,1))
                jump = 0.5*nu[0]*(ux_cell-ux_nbr)+0.5*nu[1]*(uy_cell-uy_nbr)
                
                #
                # z-zh at the edge
                # 
                z_gauss = system_z.f_eval_loc(za[z_dofs], node, edge_loc=direction)
                zh_gauss = system_u.f_eval_loc(zh_loc, node, edge_loc=direction) 
                res_edge += system_z.form_eval(((jump*(z_gauss-zh_gauss),),),\
                                                node, edge_loc=direction)
        
                
        cell_errors.append(res_cell-res_edge)
    
    cell_errors = np.array(cell_errors)
    
       
    x = np.linspace(0,1,1000)
    X,Y = np.meshgrid(x,x)
    #Z = mollifier(X, Y)
    Z = J(X,Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,Z) 
    
    
    
    #
    # "Point" evaluation at (0.5,0.5)
    #
    x0, y0 = 0.5, 0.5
    #rho = lambda x,y,eps: e(-1/(-eps**2))
    # 
    # Surface plot
    # 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot = Plot()
    plot.surface(ax, ua, mesh, element_u, shading=False, grid=True)
      
    x = system_z.dof_vertices()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot.surface(ax, J(x[:,0],x[:,1]), mesh, element_z, shading=False, grid=True) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot.surface(ax, np.array(cell_errors), mesh, element_z)
    
    plt.show()

def example_2():
    """
    Oden et al
    """
    u = lambda x,y: 5*x**2*(1-x)**2*(e(10*x**2)-1)*y**2*(1-y)**2*(e(10*y**2)-1)
    f = lambda x,y: 10*((e(10*x**2)-1)*(x-1)**2*x**2* (e(10*y**2)-1)*(y-1)**2\
                        + (e(10*x**2)-1)*(x-1)**2*x**2*(e(10*y**2)-1)*y**2\
                        + 50*(e(10*x**2)-1)*(x-1)**2*x**2*e(10*y**2)*(y-1)**2*y**2 
                        + 50*e(10*x**2)*(x-1)**2*x**2*(e(10*y**2)-1)*(y-1)**2*y**2\
                        + (e(10*x**2)-1)*x**2*(e(10*y**2)-1)*(y-1)**2*y**2\
                        + 4*(e(10*x**2)-1)*(x-1)**2*x**2*(e(10*y**2)-1)*(y-1)*y\
                        + 4*(e(10*x**2)-1)*(x-1)*x*(e(10*y**2)-1)*(y-1)**2*y**2\
                        + (e(10*x**2)-1)*(x-1)**2*(e(10*y**2)-1)*(y-1)**2*y**2\
                        + 200*(e(10*x**2)-1)*(x-1)**2*x**2*e(10*y**2)*(y-1)**2*y**4\
                        + 40*(e(10*x**2)-1)*(x-1)**2*x**2*e(10*y**2)*(y-1)*y**3\
                        + 200*e(10*x**2)*(x-1)**2*x**4*(e(10*y**2)-1)*(y-1)**2*y**2\
                        + 40*e(10*x**2)*(x-1)*x**3*(e(10*y**2)-1)*(y-1)**2*y**2)
        
    mesh = Mesh.newmesh(grid_size=(30,30))
    mesh.refine()
    element = QuadFE(2,'Q2')
    system = System(mesh, element)
    linear_forms = [(f,'v')]
    bilinear_forms = [(1,'ux','vx')]
    dir_bnd = lambda x,y: np.abs(y)<1e-10
    dir_fun = u
    boundary_conditions = {'dirichlet':[(dir_bnd, dir_fun)], 'neumann': None, 'robin': None}
    A,b = system.assemble(bilinear_forms, linear_forms, boundary_conditions)
    
    #A,b = system.extract_hanging_nodes(A, b, compress=True)
    ua = spla.spsolve(A.tocsc(), b)
    x = system.dof_vertices()
    ue = u(x[:,0],x[:,1])
    #ua = A.solve(b)
    
    
    
        
if __name__ == '__main__':
    """
    Run tests
    """
    example_1()



