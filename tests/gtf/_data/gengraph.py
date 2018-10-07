from igraph import *
import numpy as np
g=Graph()
n = 10 # nodes
m = 10 # edges

a = Graph.Erdos_Renyi(n=n, m=m, directed=False)
rows, cols = zip(*a.get_edgelist())

c = np.stack([rows, cols], axis=1).tolist()
c = sorted(c,key=lambda x: (x[0],x[1]))

with open('e_mine', 'w') as f:
    f.write("%d %d\n" % (n,m))
    for (u,v) in c:
        f.write("%d %d\n" % (u, v))

rows_tem = rows + cols
cols = cols + rows
rows = rows_tem
print(rows, cols)
a = np.stack([rows, cols], axis=1).tolist()
a = sorted(a,key=lambda x: (x[0],x[1]))

b = np.random.uniform(-15, 15, size=(n)).tolist()
with open('n', 'w') as f:
    for item in b:
        f.write("%.3f\n" % item)

with open('e.mtx', 'w') as f:
    f.write("%d %d\n" % (n,m*2))
    for (u,v) in a:
        f.write("%d %d\n" % (u, v))
