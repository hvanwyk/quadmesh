'''
Created on Feb 8, 2017

@author: hans-werner
'''

import matplotlib.pyplot as plt

class Plot(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        fig, ax = plt.subplots()
        self.fig = fig 
        self.ax = ax