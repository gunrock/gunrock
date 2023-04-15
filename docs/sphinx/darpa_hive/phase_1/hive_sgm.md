# Seeded Graph Matching (SGM)

From [Fishkind et al.](https://arxiv.org/pdf/1209.0367.pdf):

> Given two graphs, the graph matching problem is to align the two vertex sets so as to minimize the number of adjacency disagreements between the two graphs. The seeded graph matching problem is the graph matching problem when we are first given a partial alignment that we are tasked with completing.

That is, given two graphs `A` and `B`, we seek to find the permutation matrix `P` that maximizes the number of adjacency agreements between `A` and `P * B * P.T`, where `*` represents matrix multiplication.  The algorithm Fishkind et al. propose first relaxes the hard 0-1 constraints on `P` to the set of doubly stochastic matrices (each row and column sums to 1), then uses the Frank-Wolfe algorithm to minimize the objective function  `sum((A - P * B * P.T) ** 2)`.  Finally, the relaxed solution is projected back onto the set of permutation matrices to yield a feasible solution.

## Summary of Results

SGM is a fruitful workflow to optimize, because the existing implementations were not written with performance in mind.  By making minor modifications to the algorithm that allow use of sparse data structures, we enable scaling to larger datasets than previously possible.

The application involves solving a linear assignment problem (LSAP) as a subproblem.  Solving these problems on the GPU is an active area of research -- though papers have been written describing high-performance parallel LSAP solvers, reference implementations are not available.  We implement a GPU LSAP solver via Bertsekas' auction algorithm, and make it available as a [standalone library](https://github.com/bkj/cbert).

SGM is an approximate algorithm that minimizes graph adjacency disagreements via the Frank-Wolfe algorithm. Certain uses of the auction algorithm can introduce additional approximation in the gradients of the Frank-Wolfe iterations.  An interesting direction for future work would be a rigorous study of the effects of this kind of approximation on a variety of different graph tolopogies.  Understanding of those dynamics could allow further scaling beyond what our current implementations can handle.

## Summary of Gunrock Implementation

The SGM algorithm consists of:

 - several matrix-matrix and matrix-vector multiplies
 - solving a linear assignment problem at each iteration
 - computing the trace of matrix products (e.g., `trace(A * B))`)

The formulation of SGM in the [HIVE reference implementation](https://gitlab.hiveprogram.com/ggillary/seeded_graph_matching_brain_connectome/blob/master/sgm.py) does not take advantage of the sparsity of the problem.  This is due to two algorithmic design choices:

 1. In order to penalize adjacency disagreements, they convert the 0s in the input matrices `A` and `B` to -1s
 2. They initialize the solution matrix `P` near the barycenter of the Birkhoff polytope of doubly-stochastic matrices, so almost all entries are nonzero.

In order to take advantage of sparsity and make SGM more suitable for HIVE analysis, we make two relatively small modifications to the SGM algorithm:

 1. Rework the equations so we can express the objective function as a function of the sparse adjacency matrices plus some simple functions of node degree
 2. Initialize `P` to a vertex of the Birkhoff polytope

A nice property of the Frank-Wolfe algorithm is that the number of nonzero entries in the intermediate solutions grows slowly -- after the `n`th step, the solution is the convex combination of (at most) `n` vertices of the polytope (e.g., permutation matrices).  Thus, starting from a sparse initialization point means that all of the Frank-Wolfe steps are fairly sparse.

The reference implementation uses the Jonker-Volgenant algorithm to solve the linear assignment subproblem.  However, the JV algorithm (and the similar Hungarian algorithm) do not admit straightforward parallel implementations.  Thus, we replace the JV algorithm with [Bertsekas' auction algorithm](http://web.mit.edu/dimitrib/www/Auction_Encycl.pdf), which is much more natural to parallelize.

Because SGM consists of linear algebra plus an LSAP solver, we implement it outside of the Gunrock framework, using [a GPU GraphBLAS implementation](https://arxiv.org/abs/1804.03327) from John Owens' lab, as well as the [CUDA CUB](https://nvlabs.github.io/cub/) library.

## How To Run This Application on DARPA's DGX-1

### Prereqs/input

```bash
git clone --recursive https://github.com/owensgroup/csgm.git
cd csgm

# build
make clean
make

# make data (A = random sparse matrix, B = permuted version of A,
# except first `num-seeds` vertices)
python data/make-random.py --num-seeds 100
wc -l data/{A,B}.mtx
```

### Running the application
Command:

```bash
./csgm --A data/A.mtx --B data/B.mtx --num-seeds 100 --sgm-debug 1
```

Output:

```
===== iter=0 ================================
APB_num_entries=659238
counter=17
run_num=0 | h_numAssign=4096 | milliseconds=21.3737
APPB_trace = 196
APTB_trace = 7760
ATPB_trace = 7760
ATTB_trace = 109460
ps_grad_P  = 393620
ps_grad_T  = 16124208
ps_gradt_P = 407896
ps_gradt_T = 15879704
alpha      = -29.4213314
falpha     = 224003776
f1         = -15486084
num_diff   = 448756
------------
f1 < 0
iter_timer=74.0615005
...
===== iter=2 ================================
APB_num_entries=13464050
counter=1
run_num=0 | h_numAssign=4096 | milliseconds=5.71267223
APPB_trace = 333838
APTB_trace = 333838
ATPB_trace = 333838
ATTB_trace = 333838
ps_grad_P  = 16777216
ps_grad_T  = 16777216
ps_gradt_P = 16777216
ps_gradt_T = 16777216
alpha      = 0
falpha     = -1
f1         = 0
num_diff   = 0
------------
iter_timer=45.222271
total_timer=170.153473 | num_diff=0
```

__Note:__ Here, the final `num_diff=0` indicates that the algorithm has found a perfect match between the input graphs.

### Output

When run with `--sgm-debug 1`, we output information about the quality of the match in each iteration.  The most important number is `num_diff`, which gives the number of disagreements between `A` and `P * B * P.T`.  `num_diff=0` indicates that SGM has found a perfect matching between `A` and `B` (eg, there are no adjacency disagreements).

This implementation is validated against the [HIVE reference implementation](https://gitlab.hiveprogram.com/ggillary/seeded_graph_matching_brain_connectome/blob/master/sgm.py).  Additionally, since the original reference implementation code was posted, Ben Johnson has worked with Johns Hopkins to produce other more performant implementations of SGM, found [here](https://github.com/bkj/sgm/tree/v2).


## Performance and Analysis

Given:

 - two input graphs `A` and `B`
 - a set of seeds `S`
 - some parameters controlling convergence (`num_iters`, `tolerance`)

Performance is measured by runtime of the entire SGM procedure as well as the final number of adjacency disagreements between `A` and `P * B * P.T`.

Per-iteration runtime is not necessarily meaningful, because different iterations present dramatically more difficult LSAP instances than others.  In particular, the LSAP solver in the first iteration tends to take 10-100x longer than in subsequent iterations.

Bertsekas' auction algorithm allows us to make a tradeoff between runtime and accuracy.  With appropriate parameter settings, it produces the exact same answer as the JV or Hungarian algorithms.  However, with different parameter settings, the auction algorithm may run substantially faster (>10x), at the cost of a lower quality assignment.  Since SGM is already an approximate algorithm, _we do not currently know the SGM's sensitivity to this kind of approximation._  Experiments to explore these tradeoffs would be an interesting direction for future research.  In general, we run our SGM implementation with some approximation, and thus we rarely produce the exactly optimal solution for the LSAP subproblems.  However, we often produce _SGM solutions_ of comparable quality to those SGM implementations that exactly solve the LSAP subproblems.

### Implementation limitations

SGM is only appropriate for pairs of graphs w/ some kind of correspondence between the nodes.  This could be an explicit correspondance (users on Twitter and users on Instagram, people in a cell phone network and people on an email network), or an implicit correspondence (two people play similar roles at similar companies).

Currently, our implementation of SGM only supports undirected graphs -- an extension to directed graphs is mathematically straightforward, but requires a bit of code refactoring.  We have only tested on unweighted graphs, though the code should also work on weighted graphs out-of-the-box.

We also currently assume that the number of nodes in `A` and `B` are identical.  Fishkind et al. discuss various padding schemes to address the cases where `A` and `B` have a different number of nodes, but we assume all of these could be done in a preprocessing step before the graph is passed to our SGM implementation.

At the moment, the primary scaling bottleneck is that we allocate two `|V|x|V|` arrays as part of the LSAP auction algorithm.  These could be replaced w/ 3 `|E|` arrays without much effort.

Currently, the auction algorithm does not take advantage of all available parallelism.  Each CUDA thread gets a row of the cost matrix, and then does a serial reduction across the entries of the row.  As the auction algorithm runs, the number of "active" rows rapidly decreases.  This means that the majority of auction iterations have a small number of active rows, and thus use a small number of GPU threads.  We could better utilize the GPU by using multiple threads per row.  We have a preliminary implementation of this using the CUB library, but it has not been tested on various relevant edge cases. Preliminary experiments suggest the CUB implementation may be 2--5x faster than our current implementation.

### Comparison against existing implementations

#### Python SGM code

The [original SGM implementation](https://github.com/youngser/VN/) is a single-threaded R program.  Preliminary tests on small graphs show that this implementation is not very performant.  As part of other work, we've written a [modular SGM package](https://github.com/bkj/sgm) that allows the programmer to plug in different backends for the LSAP solver and the linear algebra operations.  This package includes modes that transform the SGM problem to take advantage of sparsity to improve runtime.  In particular, we benchmark our CUDA SGM implementation against the `scipy.sparse.jv` mode, which uses `scipy` for linear algebra and the [gatagat/lap](https://github.com/gatagat/lap) implementation of the JV algorithm for the LSAP solver.

#### Experiments

##### Connectome

The connectome graphs are generated from brain MRIs.  Nodes represent a voxel in the MRI and edges indicate some kind of flow between voxels. By using voxels of different spatial resolutions, the researchers that collected the data produced pairs of graphs at a variety of sizes (smaller voxel = larger graph).  Each graph in a pair represents one hemisphere of the brain.  Thus, we know there is an actual approximate correspondence between nodes.

Note that these graphs are already partially aligned -- the distance between the input graphs is far smaller than would be expected by chance.  We attempt to use SGM to further improve this initial alignment, using all nodes as "soft seeds" (see Fishkind et al. for further discussion).

The size of the connectome graphs we consider are as follows:

| name    | num_nodes | num_edges |
| ------- | --------- | --------- |
| DS00833 | 00833   |   12497   |
| DS01216 | 01216   |   19692   |
| DS01876 | 01876   |   34780   |
| DS03231 | 03231   |   64662   |
| DS06481 | 06481   |  150012   |
| DS16784 | 16784   |  445821   |
| DS72784 | 72784   | 2418304   |

In the tables below, `orig_dist` indicates the number of adjacency disagreements between `A` and `B`, and `final_dist` indicates the number of adjacency disagreements between `A` and `P * B * P.T` w/ `P` found by SGM.  We run CSGM w/ two values of `eps`, which controls the precision of the auction algorithm (lower values = more precise but slower).

###### Python runtimes

| name     | orig_dist | final_dist | time_ms    |
| -------- | --------- | ---------- | ---------- |
| DS00833  | 11650     | 11486      | 122.656    |
| DS01216  | 20004     | 19264      | 278.739    |
| DS01876  | 38228     | 36740      | 2275.141   |
| DS03231  | 78058     | 73282      | 8900.371   |
| DS06481  | 201084    | 183908     | 97658.378  |
| DS16784  | 677754    | 593590     | 920436.387 |

###### CSGM runtimes

| eps | name      | orig_dist | final_dist | time_ms    | speedup |
| --- | --------- | --------- | ---------- | ---------- | --------|
| 1.0 |  DS00833  | 11650     | 11538      | 181.481    | 0.6     |
| 1.0 |  DS01216  | 20004     | 19360      | 324.908    | 0.8     |
| 1.0 |  DS01876  | 38228     | 36936      | 807.148    | 2.8     |
| 1.0 |  DS03231  | 78058     | 73746      | 3078.78    | 2.9     |
| 1.0 |  DS06481  | 201084    | 184832     | 9056.55    | 10.7    |
| 1.0 |  DS16784  | 677754    | 596370     | 42220.5    | 21.8    |
| 1.0 |  DS72784  | ---       |  ---       | OOM        | ---     |
| 0.5 |  DS00833  | 11650     | 11466      | 378.056    | 0.3 *   |
| 0.5 |  DS01216  | 20004     | 19288      | 965.915    | 0.3     |
| 0.5 |  DS01876  | 38228     | 36764      | 1258.65    | 1.8     |
| 0.5 |  DS03231  | 78058     | 73346      | 6257.87    | 1.4     |
| 0.5 |  DS06481  | 201084    | 183796     | 25931.2    | 3.7 *   |
| 0.5 |  DS16784  | 677754    | 592822     | 120799     | 7.6 *   |
| 0.5 |  DS72784  |  ---      |  ---       | OOM        |  ---    |


__Takeaway:__ For small graphs (`|U| < ~2000`) the Python implementation is faster.  However, as the size of the graph grows, CSGM becomes significantly faster -- up to 20x faster in the low accuracy setting and up to 7.6x faster in the higher accuracy setting.  Also, though in general the auction algorithm does not compute exact solutions to the LSAP, in several cases CSGM's accuracy outperforms the Python implementation, which uses an exact LSAP solver -- these are denoted with a `*`.

##### Kasios

The Kasios call graph shows the communication pattern inside of a corporation -- nodes represent a person and edges indicate that two people spoke on the phone.  The whole graph has 10M edges and 500K nodes, which is too large for any of our SGM implementations to handle.  Thus, we sample subgraphs of size 2\*\*10 - 2\*\*15 by running a random walk until the desired number of nodes are observed, and then extracting the subgraph induced by those nodes.  We generate pairs of graphs by random permutations (except for the first `num_seeds=100` nodes).  Interestingly, with 100 seeds, SGM can almost always perfectly "reidentify" the permuted graph.

###### Python runtimes

| num_nodes | orig_dist | final_dist | time_ms |
| -------- | --------- | ---------- | -------- |
| 1024     | 66636     | 0          | 242 |
| 2048     | 208144    | 0          | 1275 |
| 4096     | 580988    | 0          | 6530 |
| 8192     | 1449712   | 0          | 30763 |
| 16384    | 3235356   | 0          | 118072 |
| 32768    | 6587656   | 0          | 479181 |


###### CSGM runtimes (`eps=1`)

| num_nodes | orig_dist | final_dist | time_ms  | speedup |
| --------- | --------- | ---------- | -------- | ------- |
| 1024      | 66636     | 0          | 182.445  | 1.3     |
| 2048      | 208144    | 4          | 310.566  | 4.1     |
| 4096      | 580988    | 4          | 1026.07  | 6.4     |
| 8192      | 1449712   | 8          | 3108.9   | 9.9     |
| 16384     | 3235356   | 16         | 4926.85  | 24.0    |
| 32768     | 6587656   | OOM!       |---       |---      |


__Takeaway:__ Similar to in the connectome experiments, we see the advantage of CSGM increase as the graphs grow larger.  In these examples, CSGM does not _quite_ match the performance of the exact LSAP solver.  However, this could be addressed by tuning and/or scheduling the `eps` parameter.

### Performance limitations

- Results from profiling indicate that all of the SGM kernels are memory latency bound.
- 50% of time is spent in sparse-sparse matrix-multiply (SpMM)
- 39% of time is spent in the auction algorithm. Of this 39%:
    - 65% of time is spent in `run_bidding`
    - 26% of time is spent in `cudaMemset`
    - 9% of time is spent in `run_assignment`

## Next Steps

### Alternate approaches

It would be worthwhile to look into parallel versions of the Hungarian or JV algorithms, as even a single-threaded CPU version of those algorithms is somewhat competitive with Bertseka's auction algorithm implemented on the GPU.  It's unclear whether it would be better to implement parallel JV/Hungarian as multi-threaded CPU programs or GPU programs.  If the former, SGM would be a good example of an application that makes good use of both the CPU (for LSAP) and GPU (for SpMM) at different steps.

### Gunrock implications

N/A

### Notes on multi-GPU parallelization

#### GraphBLAS

Multiple GPU support for GraphBLAS is on the roadmap. This will involve dividing the dataset across multiple GPUs, which can be challenging, because the GraphBLAS primitives required by SGM (`mxm`, `mxv` and `vxm`) have optimal layouts that vary depending on data and each other. There will need to be a tri-lemma between inter-GPU communication, layout transformation, and compute time for optimal vs. sub-optimal layout.

Although extending matrix-multiplication to multiple GPUs can be straightforward, doing so in a backend-agnostic fashion that abstracts away the placement (i.e., which part of matrix `A` goes on which GPU) from the user may be quite challenging. This can be done in two ways:

 1. Manually analyze the algorithm and specify the layout in a way that is application-specific to SGM (easier, but not as generalizable)
 2. Write a sophisticated runtime that will automatically build a directed acyclic graph (DAG), analyze the optimal layouts, communication volume and required layout transformations, and schedule them to different GPUs (difficult and may require additional research, but generalizable)

#### Auction algorithm

The auction algorithm can be parallelized across GPUs in several ways:

 1. Move data onto a single GPU and run the existing auction algorithm (simple, but not scalable).
 2. Bulk-synchronous algorithm: Run auction kernel, communicate, then run next iteration of auction kernel (medium difficulty, scalable).
 3. Asynchronous algorithm: Run auction kernel and communicate to other GPUs from within kernel (difficult, most scalable).

### Notes on dynamic graphs

Real-world applications of SGM to eg. social media or communications networks often involve dynamic graphs.  Application of SGM to streaming garphs could be a very interesting new research direction.  To our knowledge, this problem has not been studied by the JHU group responsible for the algorithm.  Given an initial view of two graphs, we could compute an initial match, and then update the match via a few iterations of SGM as new edges arrive.

### Notes on larger datasets

If the dataset were too big to fit into the aggregate GPU memory of multiple GPUs on a node, then two directions can be taken in order to be able to tackle these larger datasets:

 1. Out-of-memory: Compute using part of the dataset at a time on the GPU, and save the completed result to CPU memory. This method is slower than distributed, but cheaper and easier to implement.
 2. Distributed memory: If the GPU memory on a single node is not enough, use multiple nodes. This method can be made to scale for arbitrarily large datasets provided the implementation is good enough (faster than out-of-memory, but more expensive and difficult).

### Notes on other pieces of this workload

There are no non-graph pieces of the SGM workload.

### How this work can lead to a paper publication

Ben and Carl think this work can lead to a nice paper, because there aren't a lot of highly optimized parallel Linear Assignment Problem (LAP) solvers. A lot of the research Ben could find from 20+ years ago tends to assume that the input matrices are uniformly random. However, our use case is on cost matrices formed by the dot product of sparse matrices (so the `(i, j)`th entry is the number of neighbors node i and j have in common), which has a totally different distribution (closer to a power law). There may be some optimizations we can find that target this distribution (similar to how direction-optimized BFS targets power law graphs).

There is potential research in tie-breaking for the auction algorithm. In one of the popular Python/C++ LAP solvers, they're clearly not handling ties smartly, and the runtime can be improved ~10x by adding random values in a specific way. For these types of data, we find a lot of people assuming there aren't many ties. But with graphs, Ben notices many entries are ties, so some randomization is clearly beneficial.

Further development on the standalone auction algorithm will happen [here](https://github.com/bkj/cbert).  This will include porting the current implementation to CUDA CUB to take better advantage of available parallelism, as well as writing Python bindings for ease of use.
