from mesh import QuadMesh
from fem import DofHandler, Basis, QuadFE
from plot import Plot
import matplotlib.pyplot as plt
import numpy as np

# Sorting intervals 

a1 = [0.0,0.1, 0.3]
a2 = [0.3,0.4, 0.6]
a3 = [0.6,0.8, 1.0]

c1 = [1,2,3]
c2 = [7,8,9]
c3 = [4,5,6]

a = [a3, a2, a1]
print("Before sorting:", a)
print("Corresponding c:", [c3, c2, c1])

pairs = sorted(zip(a, [c3, c2, c1]))
a, c = zip(*pairs)

a.sorted()
print("After sorting:", a)
print("Corresponding c:", c)


b = []
b.extend(a1)
b.extend(a2)
b.extend(a3)
print("Extended list:", b)