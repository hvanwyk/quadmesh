"""
Investigate the accuracy and complexity of the sparse grid Gauss-Hermite rule.

References:

"""

import sys

if '/home/hans-werner/git/quadmesh/src' not in sys.path:
    sys.path.append('/home/hans-werner/git/quadmesh/src')

import Tasmanian

import numpy as np
import matplotlib.pyplot as plt

from mesh import QuadMesh

n_inputs = 5
n_outputs = 100
print(Tasmanian.lsTsgGlobalRules)
print(Tasmanian.lsTsgGlobalTypes)

grid = Tasmanian.makeGlobalGrid(n_inputs, n_outputs, 6, 'qpcurved','gauss-hermite-odd', fAlpha=0, fBeta=1)
print(grid.getNumPoints())
print(grid.getNeededPoints().shape)

