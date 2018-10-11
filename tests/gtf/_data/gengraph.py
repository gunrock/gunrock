from igraph import *
import numpy as np
g=Graph()
n = 1000 #nodes
m = 2000 # edges

a = Graph.Erdos_Renyi(n=n, m=m, directed=False)
rows, cols = zip(*a.get_edgelist())

c = np.stack([rows, cols], axis=1).tolist()
c = sorted(c,key=lambda x: (x[0],x[1]))

with open('e_mine', 'w') as f: # not gunrock usage edge file
    f.write("%d %d\n" % (n,m))
    for (u,v) in c:
        f.write("%d %d\n" % (u, v))

rows_tem = rows + cols
cols = cols + rows
rows = rows_tem
a = np.stack([rows, cols], axis=1).tolist()
a = sorted(a,key=lambda x: (x[0],x[1]))

b = np.random.uniform(-15, 15, size=(n)).tolist()
with open('n', 'w') as f:
    for item in b:
        f.write("%.3f\n" % item)

with open('e', 'w') as f: # gunrock usage edge file, needs to call agnes.py too.
    f.write("%d %d\n" % (n,m*2))
    for (u,v) in a:
        f.write("%d %d\n" % (u, v))
