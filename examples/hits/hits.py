import networkx as nx
import numpy as np
import sys
from scipy.io import mmread
from scipy.sparse import coo_matrix
np.set_printoptions(threshold=sys.maxsize)

if len(sys.argv) != 2:
    print("Usage: python3 ./hits.py <file.mtx>")
    exit()

graph_coo = mmread(sys.argv[1])
print("Loading COO matrix")
print(graph_coo.nnz, " edges")

graph_nx = nx.DiGraph(graph_coo)

print("Creating NetworkX Graph")
print("NetworkX is Directed: ", nx.is_directed(graph_nx))
print("NetworkX Graph has ", graph_nx.number_of_edges(), " edges")

max_iter = 10000
tol = 1e-6
hubs_nx, auths_nx = nx.hits(graph_nx, max_iter, tol, normalized=True)

# Numpy implementation
hrank = np.zeros((graph_coo.shape[0], 1))
arank = np.zeros((graph_coo.shape[0], 1))

hrank += 1/graph_coo.shape[0]
arank += 1/graph_coo.shape[0]

for _ in range(0, max_iter):
    hlast = hrank
    alast = arank
    hrank = np.zeros((graph_coo.shape[0], 1))
    arank = np.zeros((graph_coo.shape[0], 1))

    for edge in range(0, graph_coo.nnz):
        src = int(graph_coo.row[edge])
        dest = int(graph_coo.col[edge])
        arank[dest] += hlast[src]
        hrank[src] += alast[dest]

    # Normalize
    hrank = hrank / np.max(hrank)
    arank = arank / np.max(arank)

    err = np.sum(np.absolute(hrank-hlast))
    if err < tol:
        break

hrank = hrank / np.linalg.norm(hrank, ord=1)
arank = arank / np.linalg.norm(arank, ord=1)

hubs_np = {}
auths_np = {}

for i in range(0, graph_coo.shape[0]):
    hubs_np[i] = hrank[i]
    auths_np[i] = arank[i]

print("Hubs: ")
for key, val in sorted(hubs_nx.items(), key=lambda x: x[1], reverse=True):
    print(key, val, hubs_nx[key])
print("Authorities: ")
for key, val in sorted(auths_nx.items(), key=lambda x: x[1], reverse=True):
    print(key, val, auths_nx[key])