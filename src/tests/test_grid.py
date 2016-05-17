import unittest
from grid.vertex import Vertex
from grid.edge import Edge
from grid.cell import Cell
from grid.mesh import Mesh
import matplotlib.pyplot as plt

class TestGrid(unittest.TestCase):
    """
    Test grid module
    """
    
    #
    # Vertex Class
    #   
    def test_vertex_constructor(self):
        v = Vertex((0.,0.))
        self.assertEqual(v.coordinate, (0.,0.), 'Vertex coordinate should be (0,0).')
        self.assertEqual(v.node_number, None, 'Node number should be set to None.')
            
    
    def test_vertex_set_node_number(self):
        v = Vertex((0.,1.))
        v.set_node_number(10)
        self.assertEqual(v.node_number, 10, 'Node number should be 10.')
        self.assertRaises(Warning, v.set_node_number, 20)
    
    
    # 
    # Edge Class
    #     
    def test_edge_constructor(self):
        vb = Vertex(0.,0.)
        ve = Vertex(1.,0)
        edge = Edge(vb, ve)
        self.assertEqual(edge.v_begin, vb, 'Initial vertex should be (0,0).')
        self.assertEqual(edge.v_end, ve, 'Terminal vertex should be (1,0).')
        
    # 
    # Cell Class
    # 
    def test_cell_contructor(self):
        vertices = {'SW': (0.,0.), 'SE': (1.,0.), 'NE':(1.,1.), 'NW':(0.,1.)}
        cell = Cell(vertices)
        self.assertEqual(cell.depth, 0, 'Depth should be zero.')
        self.assertEqual(cell.address, [], 'Address should be an empty list.') 
    
    
    def test_cell_box(self):
        vertices = {'SW': (0.,0.), 'SE': (0.,1.), 'NE': (1.,1.), 'NW': (0.,1.)}
        cell = Cell(vertices)
        x0, x1, y0, y1 = cell.box()
        self.assertEqual([x0,x1,y0,y1], [0.,1.,0.,1.], 'Cell dimensions should be [0,1]x[0,1].')
        
    
    def test_cell_find_neighbor(self):
        vertices = {'SW': (0.,0.), 'SE': (0.,1.), 'NE': (1.,1.), 'NW': (0.,1.)}
        cell = Cell(vertices)
        cell.split()
        sw_child = cell.children['SW']
        sw_child.split()
        nw_grandchild = sw_child.children['NW']
        #
        # Neighbor exterior to parent cell
        #  
        self.assertEqual(nw_grandchild.find_neighbor('N'), cell.children['NW'], 
                         'Neighbor should be NW child of ROOT cell.')
        #
        # Neighbor is sibling cell
        #  
        self.assertEqual(nw_grandchild.find_neighbor('S'), sw_child.children['SW'], 
                         'Neighbor should be SW sibling.')
        #
        # Neighbor is None
        # 
        self.assertEqual(nw_grandchild.find_neighbor('W'), None, 
                         'Neighbor should be None.')
                         
    
    def test_cell_find_leaves(self):
        pass
    
    def test_cell_find_cells_at_depth(self):
        pass
    
    def test_cell_find_root(self):
        pass
    
    def test_cell_has_children(self):
        pass
    
    def test_cell_has_parent(self):
        pass
    
    def test_cell_contains_point(self):
        pass
    
    def test_cell_locate_point(self):
        pass
    
    def test_cell_mark(self):
        pass
    
    def test_cell_mark_support_cell(self):
        pass
    
    def test_cell_unmark(self):
        pass
    
    def test_cell_split(self):
        pass
    
    def test_cell_merge(self):
        pass
    
    def test_cell_balance_tree(self):
        pass
    
    def test_cell_number_vertices(self):
        vertices = {'SW': (0.,0.), 'SE': (0.,1.), 'NE': (1.,1.), 'NW': (0.,1.)}
        cell = Cell(vertices)
        cell.split()
        sw_child = cell.children['SW']
        sw_child.split()
        vertex_list = cell.number_vertices(0)
        for i in range(len(vertex_list)):
            v = vertex_list[i]
            self.assertEqual(v.node_number, i, 'Node numbers should coincide with position in list.')
            
            
    def test_cell_pos2id(self):
        #
        # String input
        # 
        vertices = {'SW': (0.,0.), 'SE': (0.,1.), 'NE': (1.,1.), 'NW': (0.,1.)}
        cell = Cell(vertices)
        self.assertEqual(cell.pos2id('SE'), 1, 'Direction SE corresponds to index 1.')
        #
        # Numeric input
        # 
        self.assertEqual(cell.pos2id(1), 1, 'Index 1 should be preserved.')
       
       
    def test_cell_id2pos(self):
        #
        # Numeric input
        # 
        vertices = {'SW': (0.,0.), 'SE': (0.,1.), 'NE': (1.,1.), 'NW': (0.,1.)}
        cell = Cell(vertices)
        self.assertEqual(cell.id2pos(2), 'NE', 'Index 2 corresponds direction NE.')
        #
        # String input
        #
        self.assertEqual(cell.id2pos('NE'), 'NE', 'Direction NE should be preserved.') 

    
    def test_cell_plot(self):
        pass
    
    '''        
    def test_cell(self):
        #
        # Constructor
        # 
        print '1. Constructor'
         
        
        print 'Cell', cell.address
        #
        # Refinement
        #  
        print '2. Refinement'
        print '    split once'
        cell.mark()
 
        self.assertEqual(cell.flag, True, 'Cell should be flagged')
        
        cell.split()
        print cell.children
        sw_child = cell.children['SW']
        self.assertEqual(sw_child.position, 'SW', 'Position of child not correct.')
        self.assertEqual(sw_child.address, [0], 'Address of SW child should be [0]')
        self.assertEqual(sw_child.depth, 1, 'Depth should be 2')
        self.assertEqual(sw_child.parent.type,'ROOT','Parent should be ROOT.')
        self.assertEqual(sw_child.type, 'LEAF', 'Child should be a LEAF cell.')
        
        print '    split again: SW child' 
        sw_child.mark()
        sw_child.split()
        sw_child.unmark()
        self.assertFalse(sw_child.flag, 'Cell should be unmarked after refinement')   
        
        # 
        # Balancing
        #
        print '3. Balancing the tree'
        print '    split again: SW child -> SE grandchild'
        sw_grand_child = sw_child.children['SE']
        self.assertEqual(sw_grand_child.address, [0,1], 'Address of SW granchild should be [0,0].')
        sw_grand_child.mark()        
        cell.split()
        cell.balance_tree()
        _, ax = plt.subplots()
        cell.plot(ax)
        plt.show()
        
        #self.assertTrue(cell.children['SE'].has_children(), 'SE child should also have children.')    
        
        for pos in cell.children.keys():
            if pos == 'NW':
                print 'marking child', pos
                cell.children[pos].mark()
                
                print 'attempting to split child'
                cell.children[pos].split()
        
        for child in cell.children.itervalues():
            if child.position == 'NW':
                child.mark()
                child.split()
                                      
        cell.balance_tree()
        
        #
        # Point inclusion
        # 
        point = (.4,.6)
        if cell.contains_point(point):
            print 'Cell contains the point (%f,%f)' %(point[0],point[1])
        else:
            print 'Cell does not contain the point (%f,%f)' %(point[0],point[1])
        
        c = cell.locate_point(point)
        c.mark()
        
        c1 = c.find_neighbor('E')
        c1.mark()
    '''
    
    #
    # Mesh Class
    #
    def test_mesh_constructor(self):
        pass
    
    def test_mesh_find_leaves(self):
        pass
    
    def test_mesh_cells_at_depth(self):
        pass
    
    def test_mesh_number_cells(self):
        pass
    
    def test_mesh_number_vertices(self):
        pass
    
    def test_mesh_has_children(self):
        pass
    
    def test_mesh_get_max_depth(self):
        pass
    
    def test_mesh_refine(self):
        pass
    
    def test_mesh_coarsen(self):
        pass
    
    def test_mesh_balance_tree(self):
        pass
    
    def test_mesh_remove_supports(self):
        pass
    
    def test_mesh_build_connectivity(self):
        mesh = Mesh()
        for child in mesh.children.itervalues():
            print child.address
        child = mesh.children[0,0]
        child.split()
        mesh.number_vertices()
        #econn = mesh.build_connectivity()
        #for element in econn:
        #    print element
        _, ax = plt.subplots()
        mesh.plot_trimesh(ax)
        plt.show()
        
    def test_mesh_plot_quad_mesh(self):
        pass
    
    def test_mesh_plot_tri_mesh(self):
        mesh = Mesh()
        child = mesh.children[0,0]
        for _ in range(6):
            child.split()
            child = child.children['NE']
        _, ax = plt.subplots()
        mesh.plot_quadmesh(ax)
        plt.show()
        mesh.balance_tree()
        mesh.number_vertices()
        #mesh.build_connectivity()
        
        _, ax = plt.subplots()
        mesh.plot_quadmesh(ax)
        plt.show()
        
        _, ax = plt.subplots()
        mesh.plot_trimesh(ax)
        plt.show()

    
    def test_mesh(self):
        '''
        mesh = Mesh()
        
        sw_child = mesh.children[0,0]
        sw_child.mark()
        mesh.refine()
        sw_ne_grandchild = sw_child.children['NE']
        sw_ne_grandchild.mark()
        mesh.refine()
        mesh.balance_tree()
        
        _, ax = plt.subplots()
        mesh.plot_quadmesh(ax)
        
        print mesh.find_leaves()
        plt.show()
        
        sw_ne_grandchild.mark()
        mesh.coarsen()
        
        #c = mesh.children[0,0].children['NE'].children['SE']
        #print 'parent', c.parent.address
        #print 'parent neighbor ', c.parent.find_neighbor('E')
        #print c.find_neighbor('E')
        
        #mesh.balance_tree()
        
        _, ax = plt.subplots()
        mesh.plot_quadmesh(ax)
        plt.show()
        '''
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()