from mesh import Mesh1D
from mesh import QuadMesh
from fem import DofHandler
from function import Function
from function import Nodal
from fem import QuadFE
from assembler import Kernel
from assembler import Form
from fem import Basis
from assembler import Assembler
from solver import LinearSystem
from plot import Plot
import numpy as np
from mesh import HalfEdge
import matplotlib.pyplot as plt

"""
Debug conditioning
"""

V = np.array([[0.5, 1/np.sqrt(2), 0, 0.5], 
              [0.5, 0, -1/np.sqrt(2), -0.5],
              [0.5, -1/np.sqrt(2), 0, 0.5], 
              [0.5, 0, 1/np.sqrt(2), -0.5]])

s = np.array([4,3,2,0])
S = V.dot(np.dot(np.diag(s),V.T))

# Generate a sample of x
Z = np.random.normal(size=(4,1))
X = V.dot(np.dot(np.diag(np.sqrt(s)),Z))



Vk = V[:,:3]
Pi = np.dot(Vk, Vk.T)

A = np.eye(4,4)[0,None]
