import networkx as nx
import sys
from scipy.io import mmread

if len(sys.argv) != 2:
    print("Usage: python3 ./hits.py <file.mtx>")
    exit()

graph_mtx = mmread(sys.argv[1])
graph_nx = nx.from_scipy_sparse_matrix(graph_mtx)

max_iter = 100000
tol = 1.0e-8
hubs, auths = nx.hits(graph_nx, max_iter, tol, normalized=True)

print("Hubs: ")
for key, val in sorted(hubs.items(), key=lambda x: x[1], reverse=True):
    print(key, val)
print("Authorities: ")
for key, val in sorted(auths.items(), key=lambda x: x[1], reverse=True):
    print(key, val)
