import numpy as np
np.set_printoptions(threshold=np.nan)
nodes = np.loadtxt('./n')
edges = np.loadtxt('./e')

lambda1 = 6

num_nodes, num_edges = edges[0, :].astype(np.int).tolist()
num_edges = 2*num_edges
edges = edges[1:]

rows, cols = edges.transpose(1,0).tolist()

rows_tem = rows + cols
cols = cols + rows
rows = rows_tem
a = np.stack([rows, cols], axis=1).tolist()
a = sorted(a,key=lambda x: (x[0],x[1]))
edges = np.stack([rows, cols], axis=1)

lambda1 = np.ones((num_edges, 1), dtype='float')*lambda1

print(lambda1.shape, edges.shape)
edges = np.concatenate([edges, lambda1], 1)


index = np.array([i for i in range(0, num_nodes)])
source = np.zeros((num_nodes, 3))
sink = np.zeros((num_nodes, 3))

source[:, 0] = num_nodes # from source: index num_nodes+1
source[:, 1] = index # to all nodes (exclusing sink)
source[:, 2] += (nodes > 0)*nodes

sink[:, 0] = index # from all nodes
sink[:, 1] = num_nodes+1 # to sink (excluding source)
sink[:, 2] -= (nodes < 0)*nodes

temp1 = np.zeros((num_nodes,3)) #duplicate for later usage, these are all 0
temp1[:,0] = index # connect from all nodes 
temp1[:,1] = num_nodes # to source

temp2 = np.zeros((num_nodes,3)) #duplicate for later usage, these are all 0
temp2[:,0] = num_nodes+1 # connect from sink
temp2[:,1] = index # to all nodes


edges_list = np.concatenate([edges, source, sink, temp1, temp2], 0).tolist()
edges = sorted(edges_list,key=lambda x: (x[0],x[1]))

with open('./std_added.mtx', 'w') as f:
    f.write("%d %d %d\n" % (num_nodes+2,num_nodes+2,num_edges+4*num_nodes))
    for (u,v,w) in edges:
        f.write("%d %d %.3f\n" % (u+1, v+1, w))
