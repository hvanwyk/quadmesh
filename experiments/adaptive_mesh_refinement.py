"""
Basic AMR using the DWR method 
"""
import numpy as np
from numpy import exp as e
import scipy.sparse.linalg as spla
from mesh import Mesh, QuadCell
from finite_element import QuadFE, System, DofHandler, GaussRule
from plot import Plot
import matplotlib.pyplot as plt
from pyatspi import tablecell

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
    element = QuadFE(2,'Q2')
    system = System(mesh, element)
    
    # 
    # (Bi)linear forms and boundary conditions
    # 
    lf = [(f,'v')]
    bf = [(1,'ux','vx'),(1,'uy','vy')]
    dir_bnd = lambda x,y: np.ones(x.shape, dtype=np.bool)
    boundary_conditions = {'dirichlet': [(dir_bnd,u)], \
                           'neumann': None, \
                           'robin': None}
    #
    # Assemble and solve 
    # 
    A,b = system.assemble(bilinear_forms=bf, linear_forms=lf, \
                          boundary_conditions=boundary_conditions)
    ua = spla.spsolve(A.tocsc(), b)
    
    # 
    # Evaluate  exact solution at dof vertices
    # 
    x = system.dof_vertices()
    ue = u(x[:,0],x[:,1])
    
    #
    # Compute the residuals on each cell
    # 
    for node in mesh.root_node().find_leaves():
        #
        # Cell residual (f+uxx+uyy)
        # 
        cell = node.quadcell()
        x_loc = system.x_loc(cell)
        dofs = system.get_global_dofs(node)
        uxx_gauss = system.f_eval_loc(ua[dofs], cell, derivatives=(2,0,0))
        uyy_gauss = system.f_eval_loc(ua[dofs], cell, derivatives=(2,1,1))
        res_cell = system.form_eval((f,'v'), node) + \
                   system.form_eval(((uxx_gauss,),'v'), node) + \
                   system.form_eval(((uyy_gauss,),'v'), node)
        
        for direction in ['W','E','S','N']:
            
    rule2d = GaussRule(9,shape='quadrilateral')
    cell = QuadCell(box=[-1,1,-1,1])
    x_phys = rule2d.map(cell)
    jac = rule2d.jacobian(cell)
    w = rule2d.weights()
    print(x_phys.shape, w.shape)
    I = np.sum(w*jac*mollifier(x_phys[:,0], x_phys[:,1]))
    
    x = np.linspace(-1,1,500)
    X,Y = np.meshgrid(x,x)
    Z = mollifier(X, Y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,Z) 
    print(I)
    
    
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
    plot.surface(ax, ua, mesh, element, shading=False, grid=True)
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



