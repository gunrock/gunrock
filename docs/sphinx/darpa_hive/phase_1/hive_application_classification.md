# Application Classification

The application classification (AC) workflow is an implementation of probabalistic graph matching via belief propagation.  The workflow takes two node- and edge-attributed graphs as input -- a data graph `G = (U_G, E_G)` and a pattern graph `P = (U_P, E_P)`.  The goal is to find a subgraph `S` of `G` such that the dissimilarity between the node/edge features of `P` and `S` is minimized. The matching is optimized via loopy belief propagation, which consists of iteratively passing messages between nodes then updating beliefs about the optimal match.

## Summary of Results

Application classification involves a number of dense-matrix operations, which did not make it an obvious candidate for implementation in Gunrock.  However, our GPU implementation using the CUDA CUB library shows substantial speedups (10-50x) over the multi-threaded OpenMP implementations.

However, there are two neighbor reduce operations that may benefit from the kind of load balancing implemented in Gunrock.  Thus, it would be useful to either expose lightweight wrappers of high-performance Gunrock primitives for easy intergration into outside projects _or_ come up with a workflow inside of Gunrock that makes programming applications with lots of non-graph operations straightforward.

## Summary of CUDA Implementation

We implement application classification from scratch in CUDA using [CUB](https://nvlabs.github.io/cub/) rather than Gunrock. Application classification requires the following kernels:

 - compute pairwise distance between rows of dense matrices
 - normalize the columns of a dense matrix
 - compute maximum of columns in a dense matrix
 - gather/scatter rows of dense matrix
 - sum/max neighborhood reduce on the data/pattern graphs

Apart from the last one, these kernels do not obviously map to Gunrock's advance/filter model.

Pseudocode for the core belief propagation algorithm is as follows:

```
for iteration in range(num_iterations):

    # Update edge messages
    edge_msg_f = diff_r[data_edges.srcs] - edge_distances # data.num_edges x pattern.num_edges
    edge_msg_r = diff_f[data_edges.srcs] - edge_distances # data.num_edges x pattern.num_edges

    # Normalize edge messages
    edge_msg_f = normprob(edge_msg_f)
    edge_msg_r = normprob(edge_msg_r)

    # Max of forward/backward messages
    f_null = columnwise_max(diff_f)                           # 1 x pattern.num_edges
    r_null = columnwise_max(diff_r)                           # 1 x pattern.num_edges
    for edge_idx, (src, dst) in enumerate(data_edges):
        max_r[src] = max(max_r[src], edge_msg_r[edge_idx], f_null)
        max_f[dst] = max(max_f[dst], edge_msg_f[edge_idx], r_null)

    # Update beliefs
    belief = - node_distances                              # data.num_nodes x patt.num_nodes
    for edge_idx, (src, dst) in enumerate(pattern_edges):
        belief[:,dst] += max_f[:,edge_idx]
        belief[:,src] += max_r[:,edge_idx]

    # Normalize beliefs
    belief = normprob(belief)

    diff_f = belief[:,pattern_edges.dsts] - max_f          # data.num_nodes x pattern.num_edges
    diff_r = belief[:,pattern_edges.srcs] - max_r          # data.num_nodes x pattern.num_edges
```

where `normprob` is the column-wise log-softmax function.  That is, since `belief` is `data.num_nodes x patt.num_nodes`, each node in the `pattern` graph gets assigned a probability distribution over nodes in the `data` graph.

Our implementation is based on the [PNNL implementation](https://gitlab.hiveprogram.com/pnnl/ApplicationClassification) rather than the [distributed GraphX reference implementation](https://gitlab.hiveprogram.com/jcromano/applicationClassification).  On all of the graphs we've tested, the output of our implementation exactly matches the output of the PNNL code.  According to PNNL, their implementation may give different results than the HIVE reference implementation (due to e.g., different normalization schemes).

## How To Run This Application on DARPA's DGX-1

### Prereqs/input

```bash
git clone --recursive \
        https://github.com/owensgroup/application_classification
cd application_classification
make clean
make
```

### Running the application

#### Example Command

```bash
mkdir -p results
./main \
    ./data/georgiyData.Vertex.csv \
    ./data/georgiyData.Edges.csv \
    ./data/georgiyPattern.Vertex.csv \
    ./data/georgiyPattern.Edges.csv > results/georgiy
```

#### Example output
```bash
$ head -n 5 results/georgiy
-8.985810e+00
-3.859019e+01
-4.470994e+01
-1.673157e+01
-1.730952e+01
$ tail -n 5 results/georgiy
-2.886186e+01
-1.499165e+01
-4.034595e+01
-1.060496e+01
-7.684015e+00
$ cat results/georgiy | openssl md5
(stdin)= bd57a5126d5f943ad5c15408d410790d
```

### Expected Output

The output of the program is a `data.num_nodes x pattern.num_nodes` matrix, where each column represents a log-probability distribution of assignments of pattern node `j` to data node `i`.  This matrix is printed in row major order as a file with `data.num_nodes x pattern.num_nodes` lines.

In python, you could inspect the output like:

```python
import numpy as np

# load results
x = open('./results/georgiy').read().splitlines()
x = [float(xx) for xx in x]

# reshape to (data.num_nodes, pattern.num_nodes)
x = np.reshape(x, (1000, 10))

# exponentiate
x = np.exp(x)

# columns should sum to 1
x.sum(axis=0)
# array([1.0000001 , 1.00000001, 1.00000004, 0.99999999, 0.99999988,
       # 0.99999994, 0.99999985, 1.00000001, 1.        , 0.99999988])
```

As previously mentioned, our output exactly matches the output of the PNNL implementation on all of the graphs we have tested.

## Performance and Analysis

We measure performance by runtime of the algorithm given

 - a node- and edge-attributed data graph
 - a node- and edge-attributed pattern graph

Though AC computes probabilities of matches between data and pattern nodes, this is a deterministic and parameterless algorithm, so we do not measure performance in terms of accuracy.  Further, runtime does not depend on the _values_ of the node/edge attributes, so we can reasonably run experiments using randomly generated values.

### Implementation limitations

As currently implemented, the algorithm allocates a number of arrays of floats:

- 3 arrays of size `data.num_edges * pattern.num_edges`
- 3 arrays of size `data.num_nodes * pattern.num_edges`
- 2 arrays of size `data.num_nodes * pattern.num_nodes`

The combined memory usage of these arrays will be the primary bottleneck for scaling.  For example, if

 - data.num_nodes = 6M
 - data.num_edges = 18M
 - pattern.num_nodes = 10
 - pattern.num_edges = 20

then the memory footprint will be on the order of 13 GB.  (_Note:_ It's likely that reordering of operations could reduce the number of intermediate data structures required.)

AC is only applicable to graphs with node and edge features.  However, the runtime of the algorithm is only dependent on the _dimension_ of these features, rather than the values.  Thus, for benchmarking purposes, we can pick a `node_feature_dim` and `edge_feature_dim` and use randomly generated features.  This is helpful becaus we do not have a good real-world dataset for benchmarking.

The algorithm requires both a data graph and a smaller pattern graph.  Both the size and topology of the pattern graph may affect the runtime of the algorithm.  However, we do not have examples of actual pattern graphs apart from `georgiyPattern`, so we are forced to generate them synthetically.  Specifically, we do this by sampling a node `q`, sampling up to `max_pattern_nodes` number of `u` to form set `Q`, and using the subgraph induced by `Q + q` as the pattern graph.

We suspect the PNNL implementation may have a couple of minor bugs:

 - the `CV` matrix is log-softmax normalized in the initialization phase, updated with values from `FMax` and `RMax`, and log-softmax normalized again.  This kind of double normalization seems strange, and is perhaps incorrect.
 - In our `RepeatColumnsByDataEdges`, `FE` and `RE` both use the `src` of an edge, which conflicts with the algorithm description in the paper.  One of these should probably be using the `src` and the other the `dst`.

These bugs are easy to fix, but we left them "as is" because a) we have no easy way to verify which is correct and b) we'd like consistency of results w/ the PNNL implementation.

### Comparison against existing implementations

_The original reference implementation consisted of a large amount of distributed Spark and GraphX Scala code.  For ease of implementation, and to make sure performance comparisons are meaningful, we instead based our implementation on the [PNNL ApplicationClassification](https://gitlab.hiveprogram.com/pnnl/ApplicationClassification) OpenMP code._

Overall, the CUDA implementation is between 10x and 100x faster than the best run of the PNNL OpenMP code.

We compare our CUDA implementation to PNNL's C++ OpenMP implementation on several different graphs:

###### georgiyData
  - a small graph included w/ real (source unknown) node/edge features (included in PNNL repo)
  - `|U|=1000 |E|=20135 node_feat_dim=13 edge_feat_dim=17`

###### rmat18
  - a scale 18 RMAT synthetic graph w/ random node/edge features (included in PNNL repo)
  - `|U|=262145 |E|=2008660 node_feat_dim=13 edge_feat_dim=17`

######  JohnsHopkins_random
  - Social network graph w/ random node/edge features
  - `|U|=5157 |E|=186572 node_feat_dim=12 edge_feat_dim=16`

For the first two, we use the `georgiyPattern` pattern graph included in the PNNL repo.  For the latter, we generate a pattern graph using the neighborhood induction mentioned above.

#### georgiyData

| implementation  | threads | elapsed_ms  |
| --------------- | ------- | ----------- |
| PNNL OpenMP     | 1       | 1635.774136 |
| PNNL OpenMP     | 2       | 1405.072927 |
| PNNL OpenMP     | 4       | 1005.914927 |
| PNNL OpenMP     | 8       | 831.342936  |
| PNNL OpenMP     | 16      | 793.069839  |
| PNNL OpenMP     | 32      | 546.305180  |
| PNNL OpenMP     | 64      | 706.761122  |
| Our CUDA        | 1xP100  | __42.533__  |

__Takeaway:__ Our CUDA implementation is approximately 12.8x faster than the fastest OpenMP run (32 threads).  However, this problem is quite small and both implementations run in under 1 second.

#### rmat18

| implementation  | threads | elapsed_ms  |
| --------------- | ------- | ----------- |
| PNNL OpenMP     | 1       | 113337.949038 |
| PNNL OpenMP     | 2       | 142036.607981 |
| PNNL OpenMP     | 4       | 109564.634800 |
| PNNL OpenMP     | 8       | 95680.689096  |
| PNNL OpenMP     | 16      | 87083.579063  |
| PNNL OpenMP     | 32      | 88772.798061  |
| PNNL OpenMP     | 64      | 82495.028973  |
| Our CUDA        | 1xP100  | __827.573__  |

__Takeaway:__ Our CUDA implementation is approximately 99x faster than the fastest PNNL run (64 threads).  The absolute magnitude of the differences is much more substantial here --  the CUDA implementation runs in < 1 second while the OpenMP version runs in \~ 1.5 minutes.

#### JohnsHopkins_random

| implementation  | threads | elapsed_ms  |
| --------------- | ------- | ----------- |
| PNNL OpenMP     | 1       | 71190.566063 |
| PNNL OpenMP     | 2       | 44293.390989 |
| PNNL OpenMP     | 4       | 31736.392021 |
| PNNL OpenMP     | 8       | 24292.662144  |
| PNNL OpenMP     | 16      | 21556.239128  |
| PNNL OpenMP     | 32      | 17082.650900  |
| PNNL OpenMP     | 64      | 19473.608017  |
| Our CUDA        | 1xP100  | __450.897__  |

__Takeaway:__ Our CUDA implementation is approximately 37x faster than the fastest PNNL run (32 threads).  Again, the absolute difference in runtimes is more substantial, with our code running in < 1 second vs. \~ 20 seconds.

### Performance limitations

Profiling (on the RMAT graph) reveals that the distribution of runtime over kernels is flatter in AC than for many applications -- often, a single kernel will account for > 80% of runtime, but here the most expensive kernel only accounts for 12.8% of compute time.

 - 12.8% of time spent in `__reorderColumns`, which is a gather/scatter operation (memory bandwidth = 281 GB/s)
 - 10.5% of time spent in `__transpose`
 - 12.1% of time spend in `__rowSubLog`, an edgewise arithmetic operation
 - 9.5% of time computing `data.num_edges x pattern.num_edges` similarity matrix
 - 8.1% of time doing neighborhood reductions (293 GB/s)

All of these are memory bound operations.

As mentioned above, the memory use of the app could be reduced (if need be) by reducing the number of intermediate data structures.  This would come at the cost of increased (re)computation time.

## Next Steps

### Alternate approaches + Gunrock implications

In the CUDA implementation, there are a number of places where we could take advantage of multiple streams to reduce runtimes.  For example, in the initialization phase, we compute the pairwise distance between a) data/pattern node features and b) data/pattern edge features.  These operations are completely independent of one another, and so could happen asynchronously on different streams.

The application was implemented outside of the Gunrock framework because it had a large number of (dense matrix) operations that are not explicitly supported by Gunrock, and a relatively small number of kernels that map well to Gunrock's advance/filter paradigm.  However, the code uses CUB's `DeviceSegmentedReduce` a number of times -- Gunrock recently added a similar operator that is load-balanced for better performance.  In the future, it would be worthwhile to see what kind of speedup we could get from the Gunrock version, which should roughly be a drop-in replacement.

### Notes on multi-GPU parallelization

Most of the kernels are either row or column operations (reductions) over dense matrices, and thus relatively easy to partition over multiple nodes.  They would either end up being embarrassingly parallel or would require a per-node reduction and then a reduction across nodes.   Replacing CUB's `DeviceSegmentedReduce` with Gunrock's implementation would give us multi-GPU support for the remaining kernel.

Alternatively, depending on the topology of the graph, we may be able to partition the data graph so that we can duplicate the pattern graph across nodes and run an independent instance of application classification on each partition.  The partition would need to be constructed in a way that ensures that every subgraph is intact on _some_ GPU, which implies some partial duplication of the data graph.  If the data graph has a large diameter, or the pattern graph has a small diameter, this may be possible without excessive duplication.  If the data graph has a small diameter, we may still be able to partition the graph by e.g. removing edges that are particularly dissimilar from edges in the pattern graph.  This kind of approach is clearly very application specific, and may not be possible at all in some cases.

### Notes on dynamic graphs

In practice, it's likely that practitioners would like to run application classification on a dynamic graph (e.g., the HIVE use case was for detecting certain suspicious patterns of communication in a cybersecurity graph). However, it is not obvious how the current algorithm would be applied in a streaming fashion without relatively major modifications.  It's more likely that the current AC algorithm would be applied to a dynamic graph via some kind of sliding window.

### Notes on larger datasets

We may be able to use a partitioning scheme like the one described in the multi-GPU section above to handle data graphs that are larger than GPU memory.

### Notes on other pieces of this workload

The documentation on the wiki includes discussion of the various featurization methods used to produce the node/edge attributes.  These are beyond the scope of this work, but do include things such as computing degree, number of triangles, betweenness centrality, etc.  If we wanted to build a high-performance end-to-end application classification system, we would want to implement some of these featurization methods in Gunrock as well.
