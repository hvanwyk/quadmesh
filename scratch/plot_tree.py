#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:57:53 2021

@author: hans-werner
"""

from mesh import Tree
import matplotlib.pyplot as plt
import numpy as np



tree = Tree(n_children=2, regular=True)
tree.split()
for c in tree.get_children():
    c.split()
    
    for cc in c.get_children():
        cc.split()
        
        
print(tree.get_depth())

leaves = tree.get_leaves()
leaves.pop()


w = 0
max_depth = 0
for leaf in leaves:
    ld = leaf.get_depth()
    w += 2**(-leaf.get_depth())
    if ld > max_depth:
        max_depth = ld
h = 2**(-max_depth)

fig, ax = plt.subplots()

for node in tree.traverse(mode='breadth-first'):
    
    ax.plot([0,h/4],[0,0],'k')
    ax.plot([h/4,h/4],[-w,w],'k')    
    ax.plot([h/4,h/2],[-w,-w],'k')
    ax.plot([h/4,h/2],[w,w],'k')
    ax.plot([h/2,h/2],[-w-w/2,-w+w/2],'k')
    ax.plot([h/2,h/2],[w-w/2,w+w/2],'k')
    ax.plot([h/2,h/2+h/8],[w-w/2,w-w/2],'k')