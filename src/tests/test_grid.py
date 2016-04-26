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
        vertices = {'SW': (0.,0.), 'SE': (1.,0.), 'NE':(1.,1.), 'NW':(0.,1.)}
        cell = Cell(vertices)
        fig, ax  = plt.subplots()
        #cell.plot(ax)
        
        cell.mark()
        cell.refine()
        #cell.plot(ax)
        
        print cell.children
        cell.children['NW'].mark()
        print cell.children['NW'].flag
        cell.refine()
        
        cell.plot(ax)
        
    def test_mesh(self):
        pass
    
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()