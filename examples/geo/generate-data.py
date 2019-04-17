import pandas as pd
import numpy as np
from scipy import sparse
from scipy.io import mmwrite

# Load data
graph          = pd.read_csv('./graph', header=None, sep='\t')
user_locations = pd.read_csv('./user_locations', header=None, sep='\t')

# Figure out how to map nodes to sequential integers (0-based)
unodes = np.unique(np.hstack([
  graph[0].values,
  graph[1].values,
]))
num_nodes = unodes.shape[0]

lookup = dict(zip(unodes, range(len(unodes))))

# Convert graph nodes to sequential integers
graph['src'] = graph[0].apply(lookup.get)
graph['dst'] = graph[1].apply(lookup.get)
graph = graph.sort_values(['src', 'dst']).reset_index(drop=True)

# Convert user_location is to sequential integers
user_locations = user_locations[user_locations[0].isin(unodes)]
user_locations[0] = user_locations[0].apply(lookup.get)
user_locations = user_locations.sort_values(0).reset_index(drop=True)

# Convert to sparse matrix
row  = graph.src.values
col  = graph.dst.values
data = graph[2].values
m    = sparse.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

# Write to .mtx format
mmwrite('instagram', m)

user_locations[0] = user_locations[0].astype(int)
del user_locations[1]
del user_locations[4]
user_locations.to_csv('./instagram.labels', header=None, sep=' ', index=False)
