"""
Solve the steady advection-diffusion equation in 2D:

    -div(q*grad(y)) + a . grad(y) = f  in  D, 

 with boundary conditions
 
    y = g_D       on dD
    q*dy/dn = g_N on dN

We are interested in computing the following quantities of interest (QoI):

1. The value of y at a specific point in the domain.
2. The flux of y across a specific boundary.
3. The average value of y over the entire domain.
"""

# built-in modules
import numpy as np
import matplotlib.pyplot as plt


# Quadmesh modules
from mesh import QuadMesh
from fem import QuadFE, DofHandler, Basis
from function import Nodal, Constant, Explicit
from assembler import Assembler, Form, Kernel
from plot import Plot


def test_accuracy(y_ref, yx1_ref, yx2_ref, y_apx):
    """
    Compute the L2- and H1-norm errors between the true and approximate 
    solutions. 

        E2 = int_D |y_ref - y_apx|**2 dx

        E1 = E2 + int_D |yx1_ref - yx1_apx|**2 + |yx2_ref - yx2_apx|**2 dx

    Inputs:

        y_ref: Explicit, reference solution

        yx1_ref: Explicit, x-derivative of reference solution

        yx2_ref: Explicit, y-derivative of reference solution

        y_apx: Nodal, approximate solution 

        dh: DofHandler, on which approximate solution is computed.


    Outputs:

        E2, E1: double, L2- and H1-errors

    """
    # 
    # L2 Error
    # 
    F2 = lambda f_ref, f_apx: (f_ref - f_apx)**2
    K2 = Kernel(f=[y_ref,y_apx],F=F2)
    L2_Form = Form(kernel=K2)

    #
    # H1 Error
    #
    F1 = lambda f_ref_x1, f_ref_x2,f_apx_x1,f_apx_x2: (f_ref_x1 - f_apx_x1)**2 + \
                                                      (f_ref_x2 - f_apx_x2)**2
    K1 = Kernel(f=[yx1_ref, yx2_ref, y_apx, y_apx], 
                derivatives=['v','v','vx','vy'], F=F1)
    H1_Form = Form(kernel=K1)

    #
    # Assembly
    # 
    problems = [[L2_Form],[H1_Form]]
    assembler = Assembler(problems, mesh=dofhandler.mesh)
    assembler.assemble()
    
    E_L2 = assembler.get_scalar(0)
    E_H1 = assembler.get_scalar(1)

    return E_L2, E_H1


# -----------------------------------------------------------------------------
# Geometry and Mesh
# -----------------------------------------------------------------------------
mesh = QuadMesh(box=[0, 2, 0, 1], resolution=(64, 64))

# Mark boundaries
mesh.mark_region('left', lambda x,y: abs(x) < 1e-6, 
                 entity_type='vertex', on_boundary=True)

mesh.mark_region('right', lambda x,y: abs(x-2) < 1e-6, 
                 entity_type='vertex',   on_boundary=True)

mesh.mark_region('bottom', lambda x,y: abs(y) < 1e-6, 
                 entity_type='half_edge', on_boundary=True)

mesh.mark_region('top', lambda x,y: abs(y-1) < 1e-6, 
                 entity_type='half_edge', on_boundary=True)


plot = Plot(quickview=False)
fig, ax = plt.subplots(figsize=(6,3))
regions = [('left','vertex'),('right','vertex'), ('bottom','edge'),('top','edge')]
ax = plot.mesh(mesh,regions=regions,axis=ax)
ax.set_title('Mesh')
plt.tight_layout()
plt.show()

# Elements and Basis
element = QuadFE(2, 'Q1')  # Linear elements

# Degree of freedom handler (mesh + element)
dofhandler = DofHandler(mesh, element)  
dofhandler.distribute_dofs() 

# Basis functions (mesh + element + function)
v = Basis(dofhandler, 'v')    # Test/trial function
vx = Basis(dofhandler, 'vx')  # x-derivative of test function
vy = Basis(dofhandler, 'vy')  # y-derivative of test function

# Coefficients and Source Terms
q = Constant(0.1)  # Diffusion coefficient
a0 = Constant(1)
a1 = Constant(0)
#f = Explicit(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]),dim=1)  # Source term

#

q = Constant(1)  # diffusion coefficient
a0,a1 = 2, 1.5 # advection coefficients
b0,b1 = 1*np.pi, 2*np.pi # frequency in x1 and x2 directions

# Reference solution
y_ref = Explicit(lambda x: np.sin(b0*x[:,0])*np.sin(b1*x[:,1]), dim=mesh.dim())
dy_dx1 = Explicit(lambda x: b0*np.cos(b0*x[:,0])*np.sin(b1*x[:,1]), dim=mesh.dim())
dy_dx2 = Explicit(lambda x: b1*np.sin(b0*x[:,0])*np.cos(b1*x[:,1]), dim=mesh.dim())


# Corresponding source term
f = Explicit(lambda x: 1*(b0**2+b1**2)*np.sin(b0*x[:,0])*np.sin(b1*x[:,1]) + \
                       a0*b0*np.cos(b0*x[:,0])*np.sin(b1*x[:,1]) + \
                       a1*b1*np.sin(b0*x[:,0])*np.cos(b1*x[:,1]),dim=mesh.dim())

g_D = Constant(0.0)  # Dirichlet BC

# Define flux functional (depending on q, dy_dx1, and dy_dx2)
F = lambda q,dy_dx1,dy_dx2, region=None: q*dy_dx1*region.unit_normal()[0] + q*dy_dx2*region.unit_normal()[1]  # Neumann flux function

# Define the kernel
g_N = Kernel(f=[q,dy_dx1,dy_dx2],F=F)

# Variational Forms
problems = [Form(kernel=q, test=vx, trial=vx),
            Form(kernel=q, test=vy, trial=vy),
            Form(kernel=a0, test=v, trial=vx),
            Form(kernel=a1, test=v, trial=vy),
            Form(kernel=f, test=v),
            Form(kernel=g_N, test=v, dmu='ds', flag='top'),
            Form(kernel=g_N, test=v, dmu='ds', flag='bottom')]

assembler = Assembler(problems,mesh)
assembler.add_dirichlet(dir_marker='left', dir_fn=y_ref)
assembler.add_dirichlet(dir_marker='right', dir_fn=y_ref)


assembler.assemble()


y_apx_vec = assembler.solve()

print('Size of output', y_apx_vec.shape)
print('Number of dofs', v.n_dofs())
y_apx = Nodal(data=y_apx_vec, basis=v)


fig, ax = plt.subplots(2,1, figsize=(6,3))
ax[0] = plot.contour(y_ref, mesh=mesh, axis=ax[0], cmap='viridis')
ax[1] = plot.contour(y_apx, mesh=mesh, axis=ax[1], cmap='viridis')


fig, ax = plt.subplots(2,2, figsize=(16,8))
# Row 1: derivatives of reference solution
ax[0,0] = plot.contour(dy_dx1,mesh=mesh, axis=ax[0,0],cmap='viridis')
ax[0,1] = plot.contour(dy_dx2,mesh=mesh, axis=ax[0,1],cmap='viridis')

# Row 2: derivatives of approximate solution
dya_dx1 = Nodal(data=y_apx_vec, basis=vx)
dya_dx2 = Nodal(data=y_apx_vec, basis=vy)

ax[1,0] = plot.contour(dya_dx1,mesh=mesh, axis=ax[1,0],cmap='viridis')
ax[1,1] = plot.contour(dya_dx2,mesh=mesh, axis=ax[1,1],cmap='viridis')

plt.show()

E_L2, E_H1 = test_accuracy(y_ref,dy_dx1,dy_dx2, y_apx)

print('L2 error', E_L2)
print('H1 error', E_H1)