### sample python interface - betweenness centrality

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
scores = pointer((c_float * nodes)())

### call gunrock function on device
gunrock.bc(scores, nodes, edges, row, col, 0)

### sample results
print ' node bc scores:',
for idx in range(nodes): print scores[0][idx],
