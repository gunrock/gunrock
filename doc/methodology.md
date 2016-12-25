# Methodology for Graph Analytics Performance

We welcome comments from others on the methodology that we use for measuring Gunrock's performance.

Currently, Gunrock is a library that requires no preprocessing. By this we mean that Gunrock inputs graphs in a "standard" format, e.g., compressed sparse row or coordinate, such as those available on common graph repositories ([SNAP](https://snap.stanford.edu/data/index.html) or [SuiteSparse (UF)](http://www.cise.ufl.edu/research/sparse/matrices/)). In our experiments, we use [MatrixMarket](https://people.sc.fsu.edu/~jburkardt/data/mm/mm.html) format.

Other graph libraries may benefit from preprocessing of input datasets. We would regard any manipulation of the input dataset (e.g., reordering the input or more sophisticated preprocessing such as graph coloring or  [CuSha](http://farkhor.github.io/CuSha/)'s G-Shards) to be preprocessing. We think preprocessing is an interesting future direction for Gunrock, but have not yet investigated it. We hope that any graph libraries that do preprocessing report results with both preprocessed and unmodified input datasets.

(That being said, we do standardize input graphs in two ways: before running our experiments, we remove self-loops/duplicated edges. If the undirected flag is set, we convert the input graph to undirected. When we do so, that implies one edge in each direction, and we report edges for that graph accordingly. What we do here appears to be standard practice.)

In general, we try to report results in two ways:

- Throughput, measured in edges traversed per second (TEPS). We generally use millions of TEPS (MTEPS) as our figure of merit.
- Runtime, typically measured in ms. We measure runtime entirely on the GPU, with the expectation that the input data is already on the GPU and the output data will be stored on the GPU. This ignores transfer times (either disk to CPU or CPU to GPU), which are independent of the graph analytics system. It is our expectation that GPU graph analytics will be most effective when (a) they are run on complex primitives and/or (b) run on sequences of primitives, either of which would mitigate transfer times. GPU graph analytics are likely not well suited to running one single simple primitive; for a simple primitive like BFS, it is more expensive to transfer the graph from CPU to GPU than it is to complete the BFS.

To calculate TEPS, we require the number of edges traversed (touched), which we count dynamically. For traversal primitives, we note that non-connected components will not be visited, so the number of visited edges may be fewer than the number of edges in the graph. We note that precisely counting edges during the execution of a particular primitive may have performance implications, so we may approximate (see BFS).

Notes on specific primitives follow.

## BFS

When we count the number of edges traversed, we do so by summing the number of outbound edges for each visited vertex. For forward, non-idempotent BFS, this strategy should give us an exact count, since this strategy visits every edge incident to a visited vertex. When we enable idempotence, we may visit a node more than once and hence may visit an edge more than once. For backward (pull) BFS, when we visit a vertex, we count all edges incoming to that vertex even if we find a visited predecessor before traversing all edges (and terminate early). (To do so otherwise has performance implications.) Enterprise uses the same counting strategy.

If a comparison library does not measure MTEPS for BFS, we compute it by the number of edges visited divided by runtime; if the former is not available, we use Gunrock's edges-visited count.

## SSSP

In general we find MTEPS comparisons between different approaches to SSSP not meaningful: because an edge may be visited one or many times, there is no standard way to count edges traversed. Different algorithms may not only visit a very different number of edges (Dijkstra vs. Bellman-Ford will have very different edge visit counts) but may also have a different number of edges visited across different invocations of the same primitive.

When we report Gunrock's SSSP MTEPS, we use the number of edges queued as the edge-traversal count.

To have a meaningful SSSP experiment, it is critical to have varying edge weights. SSSP measured on uniform edge weights is not meaningful (it becomes BFS). In our experiments, we set edge weights randomly/uniformly between 1 and 64.

## BC

If a comparison library does not measure MTEPS for BC, we compute it by twice the number of edges visited in the forward phase divided by runtime (the same computation we use for Gunrock).

## PageRank

We measure PageRank elapsed time on one iteration of PageRank. (Many other engines measure results this way and it is difficult to extrapolate from this measurement to runtime of the entire algorithm.)
