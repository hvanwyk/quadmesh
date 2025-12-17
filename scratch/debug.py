from mesh import QuadMesh
from fem import DofHandler, Basis, QuadFE
from plot import Plot
import matplotlib.pyplot as plt

plot = Plot(quickview=False)

# Computational domain
domain = [-2,2,-1,1]

# Boundary regions
infn = lambda x,y: (x==-2) and (-1<=y) and (y<=0)  # inflow boundary
outfn = lambda x,y: (x==2) and (0<=y) and (y<=1)  # outflow boundary

# Define the mesh
mesh = QuadMesh(box=domain, resolution=(20,10))


# Mark inflow
mesh.mark_region('inflow', infn, entity_type='half_edge', on_boundary=True)
    
# Mark outflow
mesh.mark_region('outflow', outfn, entity_type='vertex', on_boundary=True)

