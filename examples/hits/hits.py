import networkx as nx
import numpy as np
import sys
from scipy.io import mmread
from scipy.sparse import coo_matrix
np.set_printoptions(threshold=sys.maxsize)

if len(sys.argv) != 2:
    print("Usage: python3 ./hits.py <file.mtx>")
    exit()

graph_mtx = mmread(sys.argv[1])

# print(graph_mtx.nnz)

graph_nx = nx.from_scipy_sparse_matrix(graph_mtx)
# print(nx.is_directed(graph_nx))
max_iter = 1
tol = 100
hubs_nx, auths_nx = nx.hits(graph_nx, max_iter, tol, normalized=True)

# Numpy implementation
curr_hrank = np.zeros((graph_mtx.shape[0], 1))
curr_arank = np.zeros((graph_mtx.shape[0], 1))

graph_coo = coo_matrix(graph_mtx)
curr_hrank += 1/graph_coo.shape[0]
curr_arank += 1/graph_coo.shape[0]

print(curr_hrank)
print(curr_arank)

for _ in range(0, max_iter):
    next_hrank = np.zeros((graph_coo.shape[0], 1))
    next_arank = np.zeros((graph_coo.shape[0], 1))

    for edge in range(0, graph_coo.nnz):
        src = int(graph_coo.row[edge])
        dest = int(graph_coo.col[edge])
        next_hrank[src] += curr_arank[dest]
        next_arank[dest] += curr_hrank[src]

    # Normalize

    print(next_hrank)
    print(next_arank)
    next_hrank = next_hrank / np.max(next_hrank)
    next_arank = next_arank / np.max(next_arank)
    next_hrank = next_hrank / np.linalg.norm(next_hrank, ord=1)
    next_arank = next_arank / np.linalg.norm(next_arank, ord=1)

    temp_hrank = next_hrank
    next_hrank = curr_hrank
    curr_hrank = temp_hrank

    temp_arank = next_arank
    next_arank = curr_arank
    curr_arank = temp_arank

hubs_np = {}
auths_np = {}

for i in range(0, graph_coo.shape[0]):
    hubs_np[i] = curr_hrank[i]
    auths_np[i] = curr_arank[i]

# print("Hubs: ")
# for key, val in sorted(hubs_nx.items(), key=lambda x: x[1], reverse=True):
#     print(key, val, hubs_np[key])
# print("Authorities: ")
# for key, val in sorted(auths_nx.items(), key=lambda x: x[1], reverse=True):
#     print(key, val, auths_np[key])