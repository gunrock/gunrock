### sample python interface - connected components

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

### call gunrock function on device
num_components = gunrock.cc(labels, nodes, edges, row, col)

### sample results
print ' number of components: ' + str(num_components)
print ' component ids:',
for idx in range(nodes): print labels[0][idx],
