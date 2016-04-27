import unittest

class TestGrid(unittest.TestCase):
    '''
    Test grid module
    '''
      
    def test_vertex(self):
        from grid.vertex import Vertex
        v00 = Vertex((0.,0.))
        v10 = Vertex((1.,0.))
        v11 = Vertex((1.,1.))
        v01 = Vertex((0.,1.))
        
    def test_edge(self):
        pass
    
    def test_cell(self):
        from grid.cell import Cell
        import matplotlib.pyplot as plt
        
        #
        # Test cell constructor
        # 
        vertices = {'SW': (0.,0.), 'SE': (1.,0.), 'NE':(1.,1.), 'NW':(0.,1.)}
        cell = Cell(vertices)
        
        cell.mark()
        cell.coarsen()
        
        #
        # refine
        #  
        cell.mark()
        cell.refine()
        
                      
        cell.children['NW'].mark()
        print cell.children['NW'].flag
        cell.children['NW'].refine() 
        
        point = (.4,.6)
        if cell.contains_point(point):
            print 'Cell contains the point (%f,%f)' %(point[0],point[1])
        else:
            print 'Cell does not contain the point (%f,%f)' %(point[0],point[1])
        
        c = cell.locate_point(point)
        c.mark()
        
        c1 = c.find_neighbor('E')
        c1.mark()
        
        cell.coarsen()
        
        # Test cell plot
        # 
        fig, ax  = plt.subplots()        
        cell.plot(ax)
        plt.show()
        
        
    def test_mesh(self):
        pass
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()