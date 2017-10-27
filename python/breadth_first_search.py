### sample python interface - breath-first search

from ctypes import *

### load gunrock shared library - libgunrock
gunrock = cdll.LoadLibrary('../build/lib/libgunrock.so')

### read in input CSR arrays from files
row_list = [int(x.strip()) for x in open('toy_graph/row.txt')]
col_list = [int(x.strip()) for x in open('toy_graph/col.txt')]

### convert CSR graph inputs for gunrock input
row = pointer((c_int * len(row_list))(*row_list))
col = pointer((c_int * len(col_list))(*col_list))
nodes = len(row_list) - 1
edges = len(col_list)

### output array
labels = pointer((c_int * nodes)())
preds  = pointer((c_int * nodes)())

### sources
sources= pointer((c_int * 1)(1))

### call gunrock function on device
gunrock.bfs(labels, preds, nodes, edges, row, col, 1, sources, 0, 0, 1)

### sample results
print ' bfs labels (depth):',
for idx in range(nodes): print labels[0][idx],
