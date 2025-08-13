"""
Use active subspace methods to adaptively refine the parameter's mesh based on the sensitivity of the quantity of interest.

PDE: 2D advection-diffusion equation

- \nabla \cdot (D \nabla u) + v \cdot \nabla u = f


QOI: Local spatial average of the solution


"""
from mesh import QuadMesh
