from igraph import *
import numpy as np
g=Graph()
n = 15
m = 100

a = Graph.Erdos_Renyi(n=n, m=m, directed=True)
rows, cols = zip(*a.get_edgelist())

rows = np.insert(np.array(rows), 0, n, axis=0)
cols = np.insert(np.array(cols), 0, m, axis=0)



b = np.random.uniform(-15, 15, size=(n))
x = np.stack([rows, cols], axis=1)
np.savetxt('e.txt', x, '%d', delimiter=' ')
np.savetxt('n.txt', b, '%.3f', delimiter=' ')
