'''
Created on Apr 7, 2016

@author: hans-werner
'''
import unittest


class Test(unittest.TestCase):


    #def test_buoy_lookup(self):
        #from data.preprocess_data import build_buoy_lookup
        #data_file = '/home/hans-werner/files/work/code/ws/drifter/data/buoydata_test.dat'
        #lookup_file = '/home/hans-werner/files/work/code/ws/drifter/data/buoy_lookup_table.dat'
        #build_buoy_lookup(data_file,lookup_file)

    def test_grid(self):
        from grid.mesh import Mesh
        
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()