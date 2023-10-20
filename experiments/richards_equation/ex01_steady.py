
from mesh import QuadMesh
from fem import QuadFE, Basis, DofHandler
from function import Explicit, Nodal, Constant
from assembler import Assembler, Form, Kernel
from plot import Plot
import matplotlib.pyplot as plt
import numpy as np
from gmrf import Covariance, GaussianField

mesh = QuadMesh(box=[-2,2,-1,1], resolution=(80,40))
plot = Plot(quickview=False)

# Mark Inflow Boundary
infn = lambda x,y: (x==-2) and (-1<=y) and (y<=0)
mesh.mark_region('inflow', infn, entity_type='half_edge', on_boundary=True)

# Mark Outflow Boundary
outfn = lambda x,y: (x==2) and (0<=y) and (y<=1) 
mesh.mark_region('outflow', outfn, entity_type='half_edge', on_boundary=True) 


fig, ax = plt.subplots(1,1)
ax = plot.mesh(mesh,axis=ax, regions=[('inflow','edge'),('outflow','edge')])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

Q1 = QuadFE(2,'Q1')
dhQ1 = DofHandler(mesh,Q1)
dhQ1.distribute_dofs()
vQ1 = Basis(dhQ1)
vQ1x = Basis(dhQ1,'vx')
vQ1y = Basis(dhQ1,'vy')

Q0 = QuadFE(2,'DQ0')
dhQ0 = DofHandler(mesh,Q0)
dhQ0.distribute_dofs()
vQ0 = Basis(dhQ0,'v')

cov = Covariance(dhQ0,name='matern',parameters={'sgm': 1,'nu': 1, 'l':1})
Z = GaussianField(dhQ0.n_dofs(), K=cov)
Zs = Nodal(basis=vQ0, data=np.exp(Z.sample()))
fig, ax = plt.subplots(1,1)
ax = plot.contour(Zs,axis=ax)
plt.show()


tol = 1e-6
kmax = 1

kpa = Constant(1)
u0 = Nodal(data=np.ones(dhQ1.n_dofs()), basis=vQ1)
for k in range(kmax):
    
    # Assemble the residual
    kerx = Kernel([kpa,u0,u0],derivatives=['k','u','ux'], F=lambda kpa,u,ux: kpa*ux/(1+np.abs(u)))
    kery = Kernel([kpa,u0,u0],derivatives=['k','u','uy'], F=lambda kpa,u,uy: kpa*uy/(1+np.abs(u)))
    resForms = [Form(kernel=kerx, test=vQ1x), 
               Form(kernel=kery, test=vQ1y)] 
    resAssembler = Assembler(resForms, mesh)
    resAssembler.assemble()
    r = resAssembler.get_vector()


    # Assemble the Jacobian
    pass
