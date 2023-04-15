# Local Graph Clustering (LGC)

From [Andersen et al.](https://projecteuclid.org/euclid.im/1243430567):

> A local graph partitioning algorithm finds a cut near a specified starting vertex, with a running time that depends largely on the size of the small side of the cut, rather than the size of the input graph.

A common algorithm for local graph clustering is called PageRank-Nibble (PRNibble), which solves the L1 regularized PageRank problem. We implement a coordinate descent variant of this algorithm found in [Fountoulakis et al.](https://arxiv.org/pdf/1602.01886.pdf), which uses the fast iterative shrinkage-thresholding algorithm (FISTA).

## Summary of Results

This variant of local graph clustering (L1 regularized PageRank via FISTA) is a natural fit for Gunrock's frontier-based programming paradigm.  We observe speedups of 2-3 orders of magnitude over the HIVE reference implementation.

The reference implementation of the algorithm was not explicitly written as `advance`/`filter`/`compute` operations, but we were able to quickly determine how to map the operations by using [a lightweight Python implementation of the Gunrock programming API](https://github.com/gunrock/pygunrock/blob/master/apps/pr_nibble.py) as a development environment.  Thus, LGC was a good exercise in implementing a non-trivial end-to-end application in Gunrock from scratch.

## Summary of Gunrock Implementation

We implement Algorithm 2 from [Fountoulakis et al.](https://arxiv.org/pdf/1602.01886.pdf), which maps nicely to Gunrock. We present the pseudocode below along with the corresponding Gunrock operations:

```
A: adjacency matrix of graph
D: diagonal degree matrix of graph
Q: D^(-1/2) x (D - (1 - alpha)/2 x (D + A)) x D^(-1/2)
s: teleportation distribution, a distribution over nodes of graph
d_i: degree of node i
p_0: PageRank vector at iteration 0
q_0:  D^(-1/2) x p term that coordinate descent optimizes over
f(q): 1/2<q, Qq> - alpha x <s, D^(-1/2) x q>
grad_f_i(q_0): i'th term of the gradient of f(q_0) using q at iteration 0
rho: constant used to ensure convergence
alpha: teleportation constant in (0, 1)

Initialize: rho > 0
Initialize: q_0 = [0 ... 0]
Initialize: grad_f(q_0) = -alpha x D^(-1/2) x s

For k = 0, 1, ..., inf
    // Implemented using Gunrock ForAll operator
    Choose an i such that grad_f_i(q_k) < - alpha x rho x d_i^(1/2)
    q_k+1(i) = q_k(i) - grad_f_i(q_k)
    grad_f_i(q_k+1) = (1 - alpha)/2 x grad_f_i(q_k)

    // Implemented using Gunrock Advance and Filter operator
    For each j such that j ~ i
        Set grad_f_j(q_k+1) = grad_f_j(q_k) +
            (1 - alpha)/(2d_i^(1/2) x d_j^(1/2)) x A_ij x grad_f_i(q_k)

    For each j such that j !~ j
        Set grad_f_j(q_k+1) = grad_f_j(q_k)

    // Implemented using Gunrock ForEach operator
    // Note: ||y||_inf is the infinity norm
    if (||D^(-1/2) x grad_f(q_k)||_inf > rho x alpha)
            break
EndFor

return p_k = D^(1/2) x q_k
```

## How To Run This Application on DARPA's DGX-1

### Prereqs/input

```bash
# clone gunrock
git clone --recursive https://github.com/gunrock/gunrock.git \
        -b dev-refactor

cd gunrock/tests/pr_nibble
cp ../../gunrock/util/gitsha1.c.in ../../gunrock/util/gitsha1.c
make clean
make
```

### Running the application

#### Example command

```bash
./bin/test_pr_nibble_9.1_x86_64 \
    --graph-type market \
    --graph-file ../../dataset/small/chesapeake.mtx \
    --src 0 \
    --max-iter 1
```

#### Example output
```
Loading Matrix-market coordinate-formatted graph ...
  Reading meta data from ../../dataset/small/chesapeake.mtx.meta
  Reading edge lists from ../../dataset/small/chesapeake.mtx.coo_edge_pairs
  Substracting 1 from node Ids...
  Edge doubleing: 170 -> 340 edges
  graph loaded as COO in 0.084587s.
Converting 39 vertices, 340 directed edges ( ordered tuples) to CSR format...Done (0s).
Degree Histogram (39 vertices, 340 edges):
    Degree 0: 0 (0.000000 %)
    Degree 2^0: 0 (0.000000 %)
    Degree 2^1: 1 (2.564103 %)
    Degree 2^2: 22 (56.410256 %)
    Degree 2^3: 13 (33.333333 %)
    Degree 2^4: 2 (5.128205 %)
    Degree 2^5: 1 (2.564103 %)

__________________________
pr_nibble::CPU_Reference: reached max iterations. breaking at it=10
--------------------------
 Elapsed: 0.103951
==============================================
 advance-mode=LB
Using advance mode LB
Using filter mode CULL
__________________________
0    2   0   queue3      oversize :  234 ->  342
0    2   0   queue3      oversize :  234 ->  342
pr_nibble::Stop_Condition: reached max iterations. breaking at it=10
--------------------------
Run 0 elapsed: 1.738071, #iterations = 10
0 errors occurred.
[pr_nibble] finished.
 avg. elapsed: 1.738071 ms
 iterations: 140733299213840
 min. elapsed: 1.738071 ms
 max. elapsed: 1.738071 ms
 src: 0
 nodes_visited: 41513344
 edges_visited: 140733299212960
 nodes queued: 140733299212992
 edges queued: 5424992
 load time: 116.627 ms
 preprocess time: 963.004000 ms
 postprocess time: 0.080824 ms
 total time: 965.005875 ms
```

### Expected Output

We do not print the actual output values of PRNibble, but we output the results of a correctness check of the GPU version against our CPU implementation. `0 errors occurred.` indicates that LGC has generated an output that exactly matches our CPU validation implementation.

Our implementations are validated against the [HIVE reference implementation](https://gitlab.hiveprogram.com/ggillary/local_graph_clustering_socialmedia).

For ease of exposition, and to help in mapping the workflow to Gunrock primitives, we also implemented [a version of PRNibble in pygunrock](https://github.com/gunrock/pygunrock/blob/master/apps/pr_nibble.py).  This implementation is nearly identical to the actual Gunrock app, but in a way that more clearly exposes the logic of the app and eliminates a lot of Gunrock scaffolding/memory management/etc.

## Performance and Analysis

Performance is measured by the runtime of the approximate PageRank solver, given

 - a graph `G=(U, E)`
 - a (set of) seed node(s) `S`
 - some parameters controlling e.g., the target conductivity of the output cluster (`rho`, `alpha`, ...)

The reference implementation also includes a sweep-cut step, where a threshold is applied to the approximate PageRank values to produce hard cluster assignments.  We do not implement this part of the workflow, as it is not fundamentally a graph operation.

### Implementation limitations

PageRank runs on arbitrary graphs -- it does not require any special conditions such as node attributes, etc.

- **Memory size**: The dataset is assumed to be an undirected graph (with no self-loops). We were able to run on graphs of up to 6.2 GB in size (7M vertices, 194M edges). The memory limitation should be the number of edges `2*|E| + 7*|U|`, which needs to be smaller than the GPU memory size (16 GB for a single P100 on DGX-1).

- **Data type**: We have only tested our implementations using an `int32` data type for node IDs.  However, we could also support `int64` node IDs for graphs with more than 4B edges.

### Comparison against existing implementations

We compare our Gunrock GPU implementation with two CPU reference implementations:

- [HIVE reference implementation (Python wrapper around C++ library)](https://gitlab.hiveprogram.com/ggillary/local_graph_clustering_socialmedia)
- [Gunrock CPU reference implementation (C++)](https://github.com/gunrock/gunrock/blob/dev-refactor/gunrock/app/pr_nibble/pr_nibble_test.cuh#L38)

We find the Gunrock implementation is 3 orders of magnitude faster than either reference CPU implementation. The minimum, geometric mean, and maximum speedups are 7.25x, 1297x, 32899x, respectively.

All runtimes are in milliseconds (ms):

Dataset          | HIVE Ref. C++  | Gunrock C++ | Gunrock GPU    | Speedup
---------------- | -------- | ------------ | ------------- | ------------- 
delaunay_n13     | 21.52  | 16.33  | 2.86     | 8 
ak2010           | 97.99  | 72.08  | 3.04     | 32 
coAuthorsDBLP    | 1004   | 1399   | 4.86     | 207 
belgium_osm      | 2270   | 1663   | 2.97     | 726 
roadNet-CA       | 3403   | 2475   | 3.03     | 1123 
delaunay_n21     | 5733   | 4084   | 2.98     | 1924 
cit-Patents      | 40574  | 22148  | 16.41    | 2472 
hollywood-2009   | 43024  | 30430  | 46.30    | 929 
road_usa         | 48232  | 31617  | 3.01     | 16024 
delaunay_n24     | 49299  | 34655  | 3.28     | 15030 
soc-LiveJournal1 | 63151  | 37936  | 19.29    | 3274 
europe_osm       | 97022  | 72973  | 2.95     | 32889 
indochina-2004   | 101877 | 71902  | 11.05    | 9220 
kron_g500-logn21 | 110309 | 89438  | 627.55   | 176 
soc-orkut        | 111391 | 89752  | 18.05    | 6171 

### Performance limitations

We profiled the Gunrock GPU primitives on the `kron_g500-logn21` graph. The profiler adds approx. 100 ms of overhead (728.48 ms with profiler vs. 627.55 ms without profiler). The breakdown of runtime by kernel looks like:

Gunrock Kernel     | Runtime (ms) | Percentage of Runtime
------------------ | ------------ | ---------------------
Advance (EdgeMap)  | 566.76       | 77.8%
Filter (VertexMap) | 10.85        | 1.49%
ForAll (VertexMap) | 2.90         | 0.40%
Other              | 147.89       | 20.3%

__Note:__ "Other" includes HtoD and DtoH memcpy, smaller kernels such as scan, reduce, etc.

By profiling the LB Advance kernel, we find that the performance of `advance` is bottlenecked by random memory accesses. In the first part of the computation -- getting row pointers and column indices -- memory accesses can be coalesced and the profilers says we perform 4.9 memory transactions per access, which is close to the ideal of 4. However, once we start processing these neighbors, the memory access becomes random and we perform 31.2 memory transactions per access.

## Next Steps

### Alternate approaches

PRNibble can also be implemented in terms of matrix operations using our GPU [GraphBLAS](https://github.com/owensgroup/GraphBLAS/) library -- this implementation is currently in progress.

In theory, local graph clustering is appealing because you don't have to "touch" the entire graph.  However, all LGC implementations that we are aware of first load the entire graph into CPU/GPU memory, which limits the size of the graph that can be analyzed.  Implementations that load data from disk "lazily" as computation happens would be interesting and practically useful.

[Ligra](https://github.com/jshun/ligra) includes some high performance implementations of similar algorithms.  In the future, it would be informative to benchmark our GPU implementation against those performance optimized multi-threaded CPU implementations.  

### Gunrock implications

Gunrock currently supports all of the operations needed for this application. In particular, the `ForAll` and `ForEach` operators were very useful for this application.

Additionally, `pygunrock` proved to be a useful tool for development -- correctly mapping the original (serial) algorithm to the Gunrock operators required a lot of attention to detail, and having an environment for rapid expedited experimentation facilitated the algorithmic development.

### Notes on multi-GPU parallelization

Since this problem maps well to Gunrock operations, we expect parallelization strategy would be similar to BFS and SSSP. The dataset can be effectively divided across multiple GPUs.

### Notes on dynamic graphs

It is not obvious how this algorithm would be extended to handle dynamic graphs.  At a high level, the algorithm iteratively spreads mass from the seed nodes to neighbors in the graph.  As the connectivity structure of the graph changes, the dynamics of this mass spreading could change arbitrarily -- imagine edges that form bridges between two previously distinct clusters.  Thus, we suspect that adapting the app to work on dynamic graphs may require substantial development and study of the underlying algorithms.

### Notes on larger datasets

If the data were too big to fit into the aggregate GPU memory of multiple GPUs on a single node, then we would need to look at multiple-node solutions. Getting the application to work on multiple nodes would not be challenging, because it is very similar to BFS. However, optimizing it to achieve good scalability may require asynchronous communication, an area where we have some experience ([Pan et al.](https://arxiv.org/pdf/1803.03922.pdf)). Asynchronous communication may be necessary in order to reach better scalability in multi-node, because this application which can be formulated as sparse-matrix vector multiplication has limited computational intensity (low computation-to-communication).

### Notes on other pieces of this workload

As mentioned previously, we do not implement the sweep-cut portion of the workflow where the PageRank values are discretized to produce hard cluster assignments.  Though it's not fundamentally a graph problem, parallelization of this step is a research question addressed in [Shun et al](https://arxiv.org/abs/1604.07515).

### Potential future academic work

The coordinate descent implementation (developed by Ben Johnson) shows that Gunrock can be used as a coordinate descent solver. There has been more interest in coordinate descent recently, because coordinate descent can be used in ML as an alternative to stochastic gradient descent for SVM training.

###### References
Prof. Cho-Jui Hsieh from UC Davis is an expert in this field (see [1](http://www.jmlr.org/proceedings/papers/v37/hsieha15-supp.pdf), [2](https://www.semanticscholar.org/paper/HogWild%2B%2B%3A-A-New-Mechanism-for-Decentralized-Zhang-Hsieh/183d421bfb807378bd0463894415f40e0fca64d6), [3](http://www.stat.ucdavis.edu/~chohsieh/passcode_fix.pdf)).
