### sample python interface - single-source shortest path

from ctypes import *

### load gunrock shared library - libgunrock
gunrock = cdll.LoadLibrary('../../build/lib/libgunrock.so')

### read in input CSR arrays from files
row_list = [int(x.strip()) for x in open('toy_graph/row.txt')]
col_list = [int(x.strip()) for x in open('toy_graph/col.txt')]
val_list = [int(x.strip()) for x in open('toy_graph/val.txt')]

### convert CSR graph inputs for gunrock input
row = pointer((c_int  * len(row_list))(*row_list))
col = pointer((c_int  * len(col_list))(*col_list))
val = pointer((c_uint * len(val_list))(*val_list))
nodes = len(row_list) - 1
edges = len(col_list)

### output array
labels = pointer((c_uint * nodes)())

### call gunrock function on device
gunrock.sssp(labels, nodes, edges, row, col, val, 0)

### sample results
print ' sssp labels (distance):',
for idx in range(nodes): print labels[0][idx],
