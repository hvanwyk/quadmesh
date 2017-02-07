from mesh import QuadCell
import matplotlib.pyplot as plt

_,ax = plt.subplots()
q = QuadCell()
q.split()
q_ne = q.children['NE']
q_nw = q.children['NW']
q_sw = q.children['SW']
q_se = q.children['SE']
#q_ne.split()
q_nw.plot(ax, edges=True)
q_ne.plot(ax, edges=True)
q_se.plot(ax, edges=True)
q_sw.plot(ax, edges=True)
"""
for key, vertex in q_nw.vertices.items():
    x,y = vertex.coordinate()
    print('Vertex %.2s: (%.2f,%.2f)'%(key,x,y))
for key, edge in q_nw.edges.items():
    v1,v2 = edge.vertices()
    x0,y0 = v1.coordinate()
    x1,y1 = v2.coordinate()
    if 'NE' in key:
        print('Edge (%s): (%.2f,%.2f)-(%.2f,%.2f)'%(key,x0,y0,x1,y1))
"""
q.plot(ax)

plt.show()