from assembler import Assembler
from assembler import Form
from assembler import Kernel

from diagnostics import Verbose

from fem import QuadFE
from fem import DofHandler
from fem import Basis

from function import Constant
from function import Explicit
from function import Map
from function import Nodal

from gmrf import Covariance
from gmrf import GaussianField

from mesh import QuadMesh
from mesh import Mesh1D

from plot import Plot

from solver import LS

import numpy as np
from scipy import linalg as la
from scipy.stats import norm
import scipy.sparse as sp

import matplotlib.pyplot as plt
import TasmanianSG
#from tqdm import tqdm
"""
System 

    -div(exp(K)*grad(y)) = b + u,  x in D
                       y = g     ,  x in D_Dir
        exp(K)*grad(y)*n = 0     ,  x in D_Neu
    
    
Random field:
    
    K ~ GaussianField 

Cost Functional
    
    J(u) = E(|y(u)-y_d|**2) + alpha/2*|u|**2
"""
mesh = Mesh1D(resolution=(20,))
plot = Plot()
plot.mesh(mesh)
