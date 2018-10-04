import numpy as np
np.set_printoptions(threshold=np.nan)
nodes = np.loadtxt('n.txt')
edges = np.loadtxt('e.txt')

lambda1 = 6
num_nodes, num_edges = edges[0, :].astype(np.int).tolist()
for i in range(num_edges):
    edges[i+1, 0] += 1
    edges[i+1, 1] += 1

lambda1 = np.ones((num_edges+1, 1), dtype='float')*lambda1

print(lambda1.shape)
edges = np.concatenate([edges, lambda1], 1)
print(nodes.shape)


edges[0,0], edges[0,1], edges[0,2] = edges[0,0]+2, edges[0,0]+2, edges[0,1]+num_nodes*2

index = np.array([i for i in range(1, num_nodes+1)])
source = np.zeros((num_nodes, 3))
sink = np.zeros((num_nodes, 3))

source[:, 0] = num_nodes+1
source[:, 1] = index
source[:, 2] += (nodes > 0)*nodes

sink[:, 0] = index 
sink[:, 1] = num_nodes+2
sink[:, 2] -= (nodes < 0)*nodes

edges = np.concatenate([edges, source, sink], 0)
print(edges)
np.savetxt('st_added.txt', edges, '%i %i %.3f', delimiter=' ')
