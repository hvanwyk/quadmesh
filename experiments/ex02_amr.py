"""
Example 2: Adaptive mesh refinement for a simple diffusion dominated 
    steady state advection-diffusion problem
    
    
"""
from finite_element import System
from mesh import Mesh


mesh = Mesh.newmesh(grid_size=(8,8))
element = QuadFE(2,'Q1')
system = System(mesh, element)
