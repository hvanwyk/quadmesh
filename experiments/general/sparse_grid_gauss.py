"""
Investigate the accuracy and complexity of the sparse grid Gauss-Hermite rule.

 Reminder on how to use the Gauss-Hermite rule to compute integrals of the form

    .. math::
        I = \int_{-\infty}^{\infty} f(x) (x-a)^{\alpha} e^{-b(x-a)^2} dx,

    where :math:`\alpha` and :math:`b` are parameters that can be set in the sparse grid. 
    To compute the standard Gaussian integral, we thus set 
    
        :math:`\alpha=0`, :math:`a=0', and :math:`b=0.5`

    and pre-multiply the weights by :math:`\sqrt{2 \pi}`. 

    The growth (number of points) of the one-dimensional rule is: 
    
        m(l) = l + 1 (resp. 2l+1 for odd rules), 

    The precision of the one-dimensional rule is: 
    
        q(l) = 2l-1 (resp. 4l+1 for odd rules),  
        
    where l is the level of the rule.

    
References:

@TechReport{stoyanov2015tasmanian,
  title={User Manual: TASMANIAN Sparse Grids},
  author = "Stoyanov, M",
  institution = "Oak Ridge National Laboratory",
  address = "One Bethel Valley Road, Oak Ridge, TN",
  number = "ORNL/TM-2015/596",
  year = "2015"
  }
"""

from re import A
import sys

from matplotlib import axes
from sympy import rot_axis1

if '/home/hans-werner/git/quadmesh/src' not in sys.path:
    sys.path.append('/home/hans-werner/git/quadmesh/src')

import Tasmanian

import numpy as np
import matplotlib.pyplot as plt

from mesh import QuadMesh


def test00():
    """
    Make a table of complexities for the Gauss-Hermite rule.
    """
    n_max = 10
    l_max = 8
    n_points = np.zeros((n_max, l_max))
    for n in range(n_max):
        for l in range(l_max):
            # Create a sparse grid with the Gauss-Hermite rule
            # n: number of inputs, l: level of the rule
            grid = Tasmanian.makeGlobalGrid(n+1, 1, l, 'level', 'gauss-hermite-odd', fAlpha=0, fBeta=1)
            n_points[n, l] = grid.getNumPoints()
    print('Complexity of the Gauss-Hermite rule: Level')
    print('n_inputs\\l')
    for l in range(l_max):
        print(f'{l:3d}', end=' ')
    print()
    for n in range(n_max):
        print(f'{n:3d}', end=' ')
        for l in range(l_max):
            print(f'{int(n_points[n, l]):3d}', end=' ')
        print()

    fig, ax = plt.subplots()
    for n in range(n_max):
        ax.semilogy(np.arange(l_max), n_points[n,:], label=f'n={n+1}')
    ax.set_xlabel('Level of the rule (l)')
    ax.set_ylabel('Number of points')
    ax.set_title('Complexity of the Gauss-Hermite rule')
    ax.legend()
    ax.grid()
    plt.show()

def test01():
    """
    Test the one-dimensional Gauss-Hermite rule .
    """
    
    n_inputs = 5
    n_outputs = 100
    print(Tasmanian.lsTsgGlobalRules)
    print(Tasmanian.lsTsgGlobalTypes)

    grid = Tasmanian.makeGlobalGrid(n_inputs, n_outputs, 6, 'level','gauss-hermite-odd', fAlpha=0, fBeta=1)
    print(grid.getNumPoints())
    print(grid.getNeededPoints().shape)

test00()

n_inputs = 5
n_outputs = 100
print(Tasmanian.lsTsgGlobalRules)
print(Tasmanian.lsTsgGlobalTypes)

grid = Tasmanian.makeGlobalGrid(n_inputs, n_outputs, 6, 'qpcurved','gauss-hermite-odd', fAlpha=0, fBeta=1)
print(grid.getNumPoints())
print(grid.getNeededPoints().shape)

