import unittest
from grid.mesh import Mesh
from transit.transit_matrix import TransitMatrix
from datetime import datetime
import os
import linecache

class TestData(unittest.TestCase):


    #def test_buoy_lookup(self):
        #from data.preprocess_data import build_buoy_lookup
        #data_file = '/home/hans-werner/files/work/code/ws/drifter/data/buoydata_test.dat'
        #lookup_file = '/home/hans-werner/files/work/code/ws/drifter/data/buoy_lookup_table.dat'
        #build_buoy_lookup(data_file,lookup_file)

    def test_transit_matrix_constructor(self):
        print os.getcwd()
        filename = '../../data/buoydata_test.dat'
        f = open(filename,'r')
        dates = []
        line = linecache.getline(filename, 2)
        line = line.split()
        print line
        for line in f:
            line = line.split()
            month = int(line[1])
            year = int(line[3])
            day_hour = float(line[2])
            day = int(day_hour)
            hour = int((day_hour - day)*24)
            dates.append(datetime(year,month,day,hour))
        #for i in range(len(dates)):
        #    print dates[i].date(), dates[i].time()
            
        f.close()
        
        
    def test_data(self):
        pass
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()