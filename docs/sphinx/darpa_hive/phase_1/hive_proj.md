# Graph Projections

Given a (directed) graph `G`, graph projection outputs a graph `H` such that `H` contains edge `(u, v)` iff `G` contains edges `(w, u)` and `(w, v)` for some node `w`.  That is, graph projection creates a new graph where nodes are connected iff they are neighbors of the same node in the original graph.  Typically, the edge weights of `H` are computed via some (simple) function of the corresponding edge weights of `G`.

Graph projection is most commonly used when the input graph `G` is bipartitite with node sets `U1` and `U2` and directed edges `(u, v)`.  In this case, the operation yields a unipartite projection onto one of the node sets.  However, graph projection can also be applied to arbitrary (unipartite) graphs.

## Summary of Results

Because it has a natural representation in terms of sparse matrix operations, graph projections gave us an opportunity to compare ease of implementation and performance between Gunrock and another UC-Davis project, GPU [GraphBLAS](https://github.com/owensgroup/GraphBLAS).  

Overall, we found that Gunrock was more flexible and more performant than GraphBLAS, likely due to better load balancing.  However, in this case, the GraphBLAS application was substantially easier to program than Gunrock, and also allowed us to take advantage of some more sophisticated memory allocation methods available in the GraphBLAS cuSPARSE backend.  These findings suggest that addition of certain commonly used API functions to Gunrock could be a fruitful direction for further work.

## Summary of Gunrock Implementation

We implement two versions of graph projections: one using [Gunrock](https://github.com/gunrock/gunrock) and one using [GraphBLAS](https://github.com/owensgroup/GraphBLAS).

#### Gunrock

First, we can compute graph projection in Gunrock via a single `advance` operation from all nodes w/ nonzero outgoing degree:
```
def _advance_op(self, G, H_edges, src, dest):
    for neib in G.neighbors(src):
        if dest != neib:
            H_edges[dest * G.num_nodes + neib] += 1
```
That is, for each edge in the graph, we fetch the neighbors of the source node in `G`, then increment the weight of the edge between `dest` and each of those neighbors in `H_edges`.

Note that we have only implemented the unweighted case and a single method for computing the edgeweights of `H`, but the extension to weighted graphs and different weighting functions would be straightforward.

We use a dense `|V|x|V|` array to store the edges of the output matrix `H`.  This is simple and fast, but uses an unreasonably large amount of memory (a graph with 60k nodes requires 16 GB).  In the worst-case scenario, `H` may actually have all `|V|x|V|` possible edges, but any typical real-world graph has _far_ fewer edges in practice.

#### GraphBLAS

Second, we implement graph projection as a single sparse matrix-matrix multiply in our [GraphBLAS](https://github.com/owensgroup/GraphBLAS) GPU library, which wraps and extends cuSPARSE.

Graph projection admits a simple linear algebra formulation.  Given the adjacency matrix `A` of graph `G`, the projection is just:
```
H = matmul(transpose(A), A)
```
which can be concisely implemented via cuSPARSE's `csr2csc` and `csrgemm` functions.

The `csrgemm` functions in cuSPARSE allocate memory more intelligently than we do above, on the order of the number of edges in the output.  Thus, our GraphBLAS implementation can scale to substantially larger matrices than our Gunrock implementation.  However, implementing graph projection via a single call to `csrgemm` requires both the input graph `G` and output graph `H` to fit in GPU memory (16 GB on the DGX-1).  This limit can easily be hit, even for a moderately sized `G`, as the number of edges in `H` is often orders of magnitude larger than in `G`.

Thus, to scale to larger graphs, we implement graph projections via a chunked matrix multiply.  Specifically, to compute `matmul(X, Y)` w/ `X.shape = (n, m)` and `Y.shape = (m, k)`, we split `X` into `c` matrices `(X_1, ..., X_c)`, w/ `X_i.shape = (n / c, m)`.  Then we compute `matmul(X_i, Y)` for each `X_i`, moving the output of each multiplication from GPU to CPU memory as we go.  This implementation addresses the common case where we can fit both `X` and `Y` in GPU memory, but not `matmul(X, Y)`. Obviously, the chunked matrix multiply incurs a performance penalty, but allows us to run graph projections of much larger graphs on the GPU.

## How To Run This Application on DARPA's DGX-1

### Gunrock

#### Prereqs/input

```bash
git clone --recursive https://github.com/gunrock/gunrock -b dev-refactor
cd gunrock/tests/proj/
cp ../../gunrock/util/gitsha1.c.in ../../gunrock/util/gitsha1.c
make clean
make
```

#### Application specific parameters
None

#### Example Command
```bash
./bin/test_proj_9.1_x86_64 \
        --graph-type market \
        --graph-file ../../dataset/small/chesapeake.mtx
```

#### Example Output
```
Loading Matrix-market coordinate-formatted graph ...
Reading from ../../dataset/small/chesapeake.mtx:
  Parsing MARKET COO format edge-value-seed = 1539110067
 (39 nodes, 340 directed edges)...
Done parsing (0 s).
  Converting 39 vertices, 340 directed edges ( ordered tuples) to CSR format...
Done converting (0s).
__________________________
--------------------------
 Elapsed: 0.026941
Using advance mode LB
Using filter mode CULL
__________________________
0    0   0   queue3      oversize :  234 ->  342
0    0   0   queue3      oversize :  234 ->  342
--------------------------
Run 0 elapsed: 0.199080, #iterations = 1
edge_counter=1372
0->1 | GPU=9.000000 CPU=9.000000
0->2 | GPU=1.000000 CPU=1.000000
0->3 | GPU=2.000000 CPU=2.000000
...
38->35 | GPU=28.000000 CPU=28.000000
38->36 | GPU=2.000000 CPU=2.000000
38->37 | GPU=18.000000 CPU=18.000000
======= PASSED ======
[proj] finished.
 avg. elapsed: 0.199080 ms
 iterations: 38594739
 min. elapsed: 0.199080 ms
 max. elapsed: 0.199080 ms
 src: 0
 nodes_visited: 38578864
 edges_visited: 38578832
 nodes queued: 140734466796512
 edges queued: 140734466795232
 load time: 85.5711 ms
 preprocess time: 955.861000 ms
 postprocess time: 3.808022 ms
 total time: 960.005045 ms
```

#### Expected Output

When run in `verbose` mode, the app outputs the weighted edgelist of the graph projection `H`. When run in `quiet` mode, it only outputs performance statistics and the results of a correctness check.

### GraphBLAS

#### Prereqs/input
```bash
git clone --recursive https://github.com/bkj/graphblas_proj
cd graphblas_proj
make clean
make
```

#### Application specific parameters
```
--unweighted
 1 = convert entries of adjacency matrix to 1
 0 = leave entries of adjacency matrix as-is
--proj-debug
 1 = print debug information
 0 = don't print debug information
--print-results
 1 = print the edges of the projected graph
 0 = don't print edges
--onto-cols
 1 = given adjacency matrix A w/ `A.shape = (m, n)`,
     compute projection `H = matmul(transpose(A), A)` w/ `H.shape = (n, n)`
 0 = given adjacency matrix A w/ `A.shape = (m, n)`,
     compute projection `H = matmul(A, transpose(A))` w/ `H.shape = (m, m)`
--num-chunks
 <= 1 = do matrix multiply in one step
  > 1 = break matrix multiply into multiple chunks (eg, so we can compute
        projections larger than GPU memory)
```

#### Example Command
```bash
# generate some random data `data/X.mtx`
python data/make-random.py --seed 111 \
        --num-rows 1000 --num-cols 1000 --density 0.1

# run graph projection
./proj --X data/X.mtx --unweighted 1 --proj-debug 1
```

#### Example Output
```
proj.cu: loading data/X.mtx
  done
proj.cu: computing transpose
  done
proj.cu: computing projection
mxm analyze successful!
mxm compute successful!
  done
proj_num_edges          = 999946  # number of edges in H, including self loops
dim_out                 = 1000    # dimension of projected graph
proj_num_edges (noloop) = 998946  # number of edges in H, excluding self loops
timer                   = 208.98  # elapsed time (no IO)
```
#### Expected Output
The app will print the number of edges in the projected graph.
Additionally,

- When run w/ `--print-results=1`, the app prints the edges of the the graph projection `H`.
- When run w/ `--proj-debug=1`, the app prints a small number of progress messages.

## Validation

We compared the results of the Gunrock implementation to the [HIVE reference implementation](https://hiveprogram.com/wiki/display/WOR/V0+-+Application+Classification) and the [PNNL implementation](https://gitlab.hiveprogram.com/jfiroz/graph_projection).  These two implementations vary slightly in their output (e.g., handling of self loops). We validated the correctness of our results against the HIVE reference implementation.

## Performance and Analysis

Performance is measured by the runtime of the app, given:

 - an input graph `G` (possibly bipartite)
 - whether to project onto the rows or columns of the graph.

### Implementation limitations

#### Gunrock

The primary limitation of the current implementation is that it allocates a `|V|x|V|` array, where `|V|` is the number of nodes in the network.  This means that the memory requirements of the app can easily exceed the memory available on a single GPU (16 GB on the DGX-1).  The size of this array reflects the _worst case_ memory requirements of the graph projection workflow; while some graphs can become exceptionally large and dense when projected, we should be able to run the app on larger graphs if we store the output in a different data structure and/or allocate memory more efficiently.  Algorithms do exist for mitigating this issue: cuSPARSE's `csrgemm` method computes the row pointers in one pass, then allocates the exact amount of memory for `H`'s column and value arrays, then actually computes the matrix product.  An interesting future direction would be to integrate this sort of algorithm into Gunrock.

It may be possible to improve performance by making assumptions about the topology of the graph.  Graph projection is often used for bipartite graphs, but this app does not make any assumptions about the topology of the graph.  This choice was made in order to remain consistent with the [HIVE reference implementation](https://hiveprogram.com/wiki/display/WOR/V0+-+Application+Classification).

There are various ways that the edges of the output graph `H` can be weighted.  We only implement graph projections for unweighted graphs: the weight of the edge `(u, v)` in the output graph `H` is the count of (incoming) neighbors that `u` and `v` have in common in the original graph `G`.  Implementation of other weight functions would be fairly straightforward.

#### GraphBLAS

Currently, for the chunked matrix multiply, the CPU memory allocation and GPU to CPU memory copies for `matmul(X_i, Y)` block the computation of `matmul(X_[i+1], Y)`.  We could implement this in a non-blocking way using CUDA streams, but this would require some redesign of the GraphBLAS APIs.

Certain weighting functions are easily implemented by applying a transformation to the values of the sparse adjacency matrix `A`, but others cannot.  For instance, these weighting functions are easy to implement:
```
    weight_out = 1                             (by setting both matrices entries to 1)
    weight_out = weight_edge_1                 (by setting one matrix's entries to 1)
    weight_out = weight_edge_1 / weight_edge_2 (by setting one matrix's entries to 1 / x)
    ...
```
while this function (from the HIVE reference implementation) is not easy to implement in the `cuSPARSE` plus/multiply semiring:
```
        weight_out = weight_edge_1 / (weight_edge_1 + weight_edge_2)
```

Implementation of additional semirings in GraphBLAS is currently in progress, and will extend the family of weightings we could support.

### Comparison against existing implementations

When the graph fits in its memory, the Gunrock implementation is approx. 5x faster than the GraphBLAS implementation and approx. 100x faster than PNNL's OpenMP CPU implemention w/ 64 threads.  Somewhat surprisingly, PNNL's implementation is substantially slower than a single-threaded scipy sparse matrix multiplication.

When the graph does not fit in the Gunrock implementation's memory, our GPU GraphBLAS implementation is the fastest of the remaining implementations.

#### Existing implementations

##### PNNL

We compare our results against [PNNL's OpenMP reference implementation](https://gitlab.hiveprogram.com/jfiroz/graph_projection).  We make [minor modifications](https://gitlab.hiveprogram.com/bjohnson/graph_projection/commit/b37aabe2e56fe5207bc22c09c029b7e88c0327c1) to their code to handle unweighted graphs, in order to match the Gunrock and GraphBLAS implementations.  That is, the weight of the edge in the output graph `H` is the number of shared neighbors in the original graph.

There is a `--simple` flag in PNNL's CLI, but examining the code reveals that it just changes the order of operations.  Thus, all of our experiments are conducted with `--simple=1`, which is faster than `--simple=0` due to better data access patterns.

##### Scipy

A very simple baseline is sparse matrix-matrix multiplication as implemented in the popular `scipy` python package.  This is a single-threaded C++ implementation with a Python wrapper.  Note, this implementation comes with the same caveats about weighting functions as the Gunrock implementation.

#### Experiments

##### MovieLens

MovieLens is a bipartite graph `G=(U, V, E)` w/ `|U|=138493`, `|V|=26744` and `|E|=20000264`. We report results on the full graph, as well as several random subgraphs.

For PNNL's OpenMP implementation, we report results using {1, 2, 4, 8, 16, 32,  64} threads.

In all cases, we project onto the nodeset `|V|`, producing a `|V|x|V|` graph.

__Note:__ Small differences in the number of nonzero entries in the output (nnz_out) are due to small book-keeping differences (specifically, keeping or dropping self-loops).  These differences do not have any meaningful impact on runtimes.

###### 1M edge subgraph (|U|=6743 |V|=13950 |E|=1M)

| implementation | num_threads | nnz_out  | elapsed_seconds |
| -------------- | ----------- | ---------| --------------- |
| scipy          | 1           | 63104132 | 2.4912          |
| PNNL OpenMP    | 1           | 63090182 | 61.4852         |
| PNNL OpenMP    | 2           | 63090182 | 62.2842         |
| PNNL OpenMP    | 4           | 63090182 | 60.1542         |
| PNNL OpenMP    | 8           | 63090182 | 37.5853         |
| PNNL OpenMP    | 16          | 63090182 | 22.0257         |
| PNNL OpenMP    | 32          | 63090182 | 13.1482         |
| PNNL OpenMP    | 64          | 63090182 | 9.055           |
| Gunrock        | 1xP100 GPU  | 63090182 | __0.060__       |
| GraphBLAS      | 1xP100 GPU  | 63090182 | 0.366           |


###### 5M edge subgraph (|U|=34395 |V|=20402 |E|=5M)

| implementation | num_threads | nnz_out   | elapsed_seconds |
| -------------- | ----------- | ----------| --------------- |
| scipy          | 1           | 157071858 | 10.1052         |
| PNNL OpenMP    | 1           | 157051456 | 357.511         |
| PNNL OpenMP    | 2           | 157051456 | 309.723         |
| PNNL OpenMP    | 4           | 157051456 | 218.519         |
| PNNL OpenMP    | 8           | 157051456 | 113.987         |
| PNNL OpenMP    | 16          | 157051456 | 57.4606         |
| PNNL OpenMP    | 32          | 157051456 | 38.1186         |
| PNNL OpenMP    | 64          | 157051456 | 29.0056         |
| Gunrock        | 1xP100 GPU  | 157051456 | __0.3349__      |
| GraphBLAS      | 1xP100 GPU  | 157051456 | 1.221           |


###### MovieLens-20M graph (|U|=138493 |V|=26744 |E|=20M)

| implementation | num_threads | nnz_out   | elapsed_seconds |
| -------------- | ----------- | ----------| --------------- |
| scipy          | 1           | 286857534 | 39.181          |
| PNNL OpenMP    | 1           | 286830790 | _I killed before finish_ |
| PNNL OpenMP    | 2           | 286830790 | 1109.32         |
| PNNL OpenMP    | 4           | 286830790 | 727.224         |
| PNNL OpenMP    | 8           | 286830790 | 358.708         |
| PNNL OpenMP    | 16          | 286830790 | 188.701         |
| PNNL OpenMP    | 32          | 286830790 | 102.964         |
| PNNL OpenMP    | 64          | 286830790 | 163.731         |
| Gunrock        | 1xP100 GPU  | 286830790 | _out-of-memory_ |
| GraphBLAS      | 1xP100 GPU  | 286830790 | __5.012__       |

__Takeaway:__ When the graph is small enough, Gunrock graph projection is fastest, followed by GraphBLAS (approx. 5x slower).  The PNNL OpenMP implementation is consistently substantially slower than the single threaded scipy implementation, even when using 32+ threads.

##### RMAT

Next we test on a [scale 18 RMAT graph](https://graphchallenge.s3.amazonaws.com/synthetic/graph500-scale18-ef16/graph500-scale18-ef16_adj.tsv.gz).  This is _not_ a bipartite graph, but the graph projection algorithm can still be applied.

This graph was chosen because it was used in benchmarks in [PNNL's gitlab repo](https://gitlab.hiveprogram.com/jfiroz/graph_projection).  However, their command line parameters appear to be incorrect, so our results here are substantially different than reported in their README.

###### RMAT-18 (|V|=174147 |E|=7600696)

| implementation | num_threads | nnz_out    | elapsed_seconds |
| -------------- | ----------- | -----------| --------------- |
| scipy          | 1           | 2973926895 | 150.869                  |
| PNNL OpenMP    | 1           | 2973752748 | _I killed before finish_ |
| PNNL OpenMP    | 2           | 2973752748 | _I killed before finish_ |
| PNNL OpenMP    | 4           | 2973752748 | 812.453                  |
| PNNL OpenMP    | 8           | 2973752748 | 677.582                  |
| PNNL OpenMP    | 16          | 2973752748 | 419.468                  |
| PNNL OpenMP    | 32          | 2973752748 | 369.278                  |
| PNNL OpenMP    | 64          | 2973752748 | 602.69                   |
| Gunrock        | 1xP100 GPU  | 2973752748 | _out-of-memory_          |
| GraphBLAS      | 1xP100 GPU  | 2973752748 | __26.478__               |


__Takeaway:__ GraphBLAS is approx. 5.7x faster than scipy, the next fastest implementation. Again, the PNNL OpenMP implementation is substantially faster than the single-threaded scipy implementation.

_When the dataset can fit into memory_, Gunrock is \~ 4x faster than GraphBLAS.  Since the two implementations use slightly different algorithms, it's hard to tell where the Gunrock speedup comes from.  Our hunch is that Gunrock's superior load balancing gives better performance than GraphBLAS, but this is an interesting topic for further research.

### Performance information

#### Gunrock
  - Results of profiling indicate that the Gunrock implementation is bound by memory latency.
  - The device memory bandwidth is 297 GB/s -- within the expected range for Gunrock graph analytics.
  - 92% of the runtime is spent in the advance operator (pseudocode in implementation summary)

#### GraphBLAS
 - 99% of time is spent in cuSPARSE's `csrgemm` routines


## Next Steps

### Alternate approaches

As mentioned above, it would be worthwhile to implement a Gunrock version that does not require allocating the `|V|x|V|` array.  It should be possible to achieve this by implementing the same kind of two-pass approach that cuSPARSE uses for `spgemm` -- one pass computes the CSR offsets, then column and data values are inserted at the appropriate locations.

### Gunrock implications

Gunrock does not natively support bipartite graphs, but it is straightforward for programmers to implement algorithms that expect bipartite graphs by keeping track of the number of nodes in each node set.  However, for multi-GPU implementations, the fact that a graph is bipartite may be useful for determining the optimal graph partitioning.

### Notes on multi-GPU parallelization

Multiple GPU support for GraphBLAS is on the roadmap. Unlike the Seeded Graph Matching problem -- which requires the `mxm`, `mxv` and `vxm` primitives, which in turn necessitates possible changes in data layout -- this problem only requires the `mxm` primitive, so multiple GPU support for graph projections is easier than for SGM.

Even though extending matrix multiplication to multiple GPUs can be straightforward, doing so in a backend-agnostic fashion that abstracts away the placement (i.e. which part of matrix A goes on which GPU) may still be quite challenging.

Further discussion can be found [here](README.md#scaling-analysis-for-hive-applications).

### Notes on dynamic graphs

This workflow does not have an explicit dynamic component.  The graph projection operation seems like it would be fairly straightforward to adapt to dynamic graphs -- as nodes/edges are added to `G`, we create/increment the weight of the appropriate edges in `H`.  However, this adds an additional layer of complexity to the memory allocation step, as we can't use the two-pass approach to allocate memory conservatively.

### Notes on larger datasets

If the dataset were too big to fit into the aggregate GPU memory of multiple GPUs on a node, then two directions can be taken in order to be able to tackle these larger datasets:

 - Out-of-memory: Compute using part of the dataset at a time on the GPU, and save the completed result to CPU memory. This method is slower than distributed, but cheaper and easier to implement.
 - Distributed memory: If GPU memory of a single node is not enough, use multiple nodes. This method can be made to scale for infinitely large datasets provided the implementation is good enough (faster than out-of-memory, but more expensive and difficult).

### Notes on other pieces of this workload

This workload does not involve any non-graph components.
