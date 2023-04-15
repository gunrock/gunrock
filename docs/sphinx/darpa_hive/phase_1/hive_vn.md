# Vertex Nomination

The vertex nomination (VN) workflow is an implementation of the kind of algorithm discussed in [Coppersmith and Priebe](https://arxiv.org/abs/1201.4118).

Often, we have an attributed graph where we know of some "interesting" nodes, and we want to rank the rest of the nodes by their likelihood of being interesting.  Coppersmith and Priebe propose a general framework for using both node attributes ("content") and network features ("context") to rank nodes. The specific content, context, and fusion functions can be arbitrary, user-defined functions.

## Summary of Results

The term "vertex nomination" covers a variety of different node ranking schemes that fuse "content" and "context" information.  The HIVE reference code implements a "multiple-source shortest path" context scoring function, but uses a very suboptimal algorithm.  By using a more efficient algorithm, our serial CPU implementation achieves 1-2 orders of magnitude speedup over the HIVE implementation and our GPU implementation achieves another 1-2 orders of magnitude on top of that.  Implementation was straightforward, involving only a small modification to the existing Gunrock SSSP app.

## Summary of Gunrock Implementation

Since HIVE is focused on graph analytics, the content scoring function is not relevant, and we only implement the context scoring function.  Coppersmith and Priebe propose a number of possible network statistics that can be used for context scoring.  The [HIVE reference implementation](https://gitlab.hiveprogram.com/ggillary/vertex_nomination_Enron/blob/master/snap_vertex_nomination.py) ranks each node `u` in a graph `G = (U, E)` by the minimum distance from `u` to a node in a set of seed nodes `S`.  This is the VN variant we have implemented in Gunrock.

This choice of context scoring function ends up being nearly identical to a single source shortest paths (SSSP) problem.  The one difference is that we start from the set of seed nodes `S` instead of single node.

Because of this similarity to SSSP, the [Gunrock VN implementation](https://github.com/gunrock/gunrock/tree/dev-refactor/tests/vn) consists of making a minor modification to the [Gunrock SSSP implementation](https://github.com/gunrock/gunrock/tree/dev-refactor/tests/sssp), so that it can accept a list of source nodes instead of a single source node.  Thus, the core of the VN algorithm is a Gunrock advance operator implementing a parallel version of the Bellman-Ford algorithm.  Specifically, in `python`:

```python
class IterationLoop(BaseIterationLoop):
    def _advance_op(self, src, dest, problem, enactor_stats):
        src_distance = problem.distances[src]
        edge_weight  = problem.edge_weights[(src, dest)]
        new_distance = src_distance + edge_weight

        old_distance = problem.distances[dest]
        problem.distances[dest] = min(old_distance, new_distance)

        return new_distance < old_distance

    def _filter_op(self, src, dest, problem, enactor_stats):
        if problem.labels[dest] == enactor_stats['iteration']:
            return False

        problem.labels[dest] = enactor_stats['iteration']
        return True
```

Note we could have used the Gunrock SSSP implementation directly by

 1. adding a dummy node `d` to `G`
 2. adding an edge `(d, s)` between `d` and each node `s` in `S` with `weight(d, s) = 0`
 3. running SSSP from `d`

## How To Run This Application on DARPA's DGX-1

### Prereqs/input

```bash
git clone --recursive https://github.com/gunrock/gunrock -b dev-refactor
cd gunrock/tests/vn
cp ../../gunrock/util/gitsha1.c.in ../../gunrock/util/gitsha1.c
make clean
make
```

### Application specific parameters
```
--src
  Comma separated list of seed nodes (eg, `0,1,2`) OR `random` (see below)
--srcs-per-run
  If `src=random`, number of randomly chosen source nodes per run
--num-runs
  Number of runs
```

### Example command

```bash
./bin/test_vn_9.1_x86_64 \
    --graph-type market \
    --graph-file ../../dataset/small/chesapeake.mtx \
    --src random \
    --srcs-per-run 10 \
    --num-runs 2
```

### Example output
```
Loading Matrix-market coordinate-formatted graph ...
  Reading meta data from ../../dataset/small/chesapeake.mtx.meta
  Reading edge lists from ../../dataset/small/chesapeake.mtx.coo_edge_pairs
  Assigning 1 to all 170 edges
  Substracting 1 from node Ids...
  Edge doubleing: 170 -> 340 edges
  graph loaded as COO in 0.090935s.
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
--------------------------
Run 0 elapsed: 0.025034 ms, srcs = 21,19,3,28,20,25,23,26,13,38
__________________________
--------------------------
Run 1 elapsed: 0.021935 ms, srcs = 21,15,5,23,3,29,22,26,33,4
==============================================
 mark-pred=0 advance-mode=LB
Using advance mode LB
Using filter mode CULL
__________________________
0  1   0   queue3    oversize :  234 ->  246
0  1   0   queue3    oversize :  234 ->  246
--------------------------
Run 0 elapsed: 0.442028 ms, srcs = 21,19,3,28,20,25,23,26,13,38, #iterations = 3
__________________________
--------------------------
Run 1 elapsed: 0.329971 ms, srcs = 21,15,5,23,3,29,22,26,33,4, #iterations = 3
Distance Validity: PASS
First 40 distances of the GPU result:
[0:1 1:1 2:1 3:0 4:0 5:0 6:1 7:1 8:2 9:2 10:2 11:2 12:2 13:1 14:1 15:0 16:1 17:1 18:1 19:2 20:2 21:0 22:0 23:0 24:2 25:2 26:0 27:1 28:2 29:0 30:1 31:1 32:1 33:0 34:1 35:1 36:1 37:1 38:1 ]
First 40 distances of the reference CPU result.
[0:1 1:1 2:1 3:0 4:0 5:0 6:1 7:1 8:2 9:2 10:2 11:2 12:2 13:1 14:1 15:0 16:1 17:1 18:1 19:2 20:2 21:0 22:0 23:0 24:2 25:2 26:0 27:1 28:2 29:0 30:1 31:1 32:1 33:0 34:1 35:1 36:1 37:1 38:1 ]

[vn] finished.
 avg. elapsed: 0.386000 ms
 iterations: 3
 min. elapsed: 0.329971 ms
 max. elapsed: 0.442028 ms
 rate: 0.880830 MiEdges/s
 src: 3
 nodes_visited: 39
 edges_visited: 340
 load time: 113.31 ms
 preprocess time: 974.719000 ms
 postprocess time: 0.562906 ms
 total time: 976.519108 ms
```

### Expected Output

Currently, the VN app does not write any output to disk. It prints runtime statistics and the results of a correctness check.  A successful run will print  `Distance Validity: PASS` in the output.

## Validation

The Gunrock VN implementation was tested against the [HIVE reference implementation](https://gitlab.hiveprogram.com/ggillary/vertex_nomination_Enron/blob/master/snap_vertex_nomination.py) to verify correctness.  We also implemented a CPU reference implementation inside of the Gunrock VN app, with results that match the HIVE reference implementation.

Additionally, for ease of exposition, we implemented a [pure Python version of the Gunrock algorithm](https://github.com/gunrock/pygunrock/blob/master/apps/vn.py) that lets people new to Gunrock see the relevant logic without all of the complexity of C++/CUDA data structures, memory management, etc.  In this case, we already knew how to implement VN using Gunrock primitives.  However, in other cases, where we're writing a Gunrock app from scratch, translation from some arbitrary serial implementation to `advance`/`filter`/`compute` can be complex and involve some trial and error to handle edge cases.  In those cases, `pygunrock` has proven to be a useful tool for rapid prototyping and debugging.

Our implementation of VN is a deterministic algorithm, so all correct solutions have the same accuracy/quality.

## Performance and Analysis

Performance is measured by the runtime of the app, given:

- an input graph `G=(U, E)`
- a set of seed nodes (or size/number of random seed sets)

### Other implementations

#### Python reference implementation

The Python + SNAP reference implementation can be found [here](https://gitlab.hiveprogram.com/ggillary/vertex_nomination_Enron/blob/master/snap_vertex_nomination.py).  This is a very naive implementation of the context function -- rather than running the SSSP variant we describe above, it runs a separate BFS from each node `s` in the seed set `S` to each node `u` in `U`.  Thus, its algorithmic complexity is approximately `|S|x|U|` times larger than the Gunrock implementation.

#### Performer OpenMP implementations

We were unable to locate any C/OpenMP implementation of VN from TA1/TA2 performers at the time of writing. (2018-10-20)

#### Gunrock CPU implementation

For correctness checking, we implement VN in single-threaded C++ within the Gunrock testing framework.  This is a serial implementation of Djikstra's algorithm using a CSR graph representation and `std::priority_queue`.  We expect this to be substantially faster than the HIVE Python+SNAP reference implementation due to superior algorithmic complexity.

#### Experiments

##### HIVE Enron dataset (|U|=15056, |E|=57075)

The Enron graph is a graph of email communications between employees of Enron.

The HIVE reference implementation implementation does 10 runs w/ 5 random seeds each on the [Enron email dataset](https://hiveprogram.com/data/_v0/vertex_nomination_and_scan_statistics/).  Results are as follows:

| implementation | elapsed_ms (avg of 10 runs) |
| -------------- | --------------------------- |
| Python+SNAP    | 4115.545                    |
| Gunrock CPU    | 7.305                       |
| Gunrock GPU    | __0.921__                   |

<!--
#### Raw data:
```
num_nodes | num_edges | num_seeds | run_id | ms_elapsed

# HIVE reference
15056 57075 5 0 3326.16
15056 57075 5 1 3435.40
15056 57075 5 2 3509.23
15056 57075 5 3 3738.28
15056 57075 5 4 3791.98
15056 57075 5 5 3920.63
15056 57075 5 6 3951.03
15056 57075 5 7 4239.80
15056 57075 5 8 4371.74
15056 57075 5 9 6871.20
--
mean time = 4115.545

# Gunrock CPU
15056 57075 5 0 9.342909
15056 57075 5 1 7.142067
15056 57075 5 2 7.050037
15056 57075 5 3 7.044077
15056 57075 5 4 7.103205
15056 57075 5 5 7.117033
15056 57075 5 6 7.065058
15056 57075 5 7 7.049084
15056 57075 5 8 7.065058
15056 57075 5 9 7.073879
--
mean time = 7.305

# Gunrock GPU
15056 57075 5 0 1.070976
15056 57075 5 1 0.850201
15056 57075 5 2 0.936985
15056 57075 5 3 0.932932
15056 57075 5 4 0.855923
15056 57075 5 5 0.915051
15056 57075 5 6 0.982046
15056 57075 5 7 0.930071
15056 57075 5 8 0.837088
15056 57075 5 9 0.906944
--
mean time = 0.921
```
-->

__Takeaway:__ Due to improved algorithmic efficiency, the Gunrock CPU implementation is approximately 563x faster than the HIVE reference implementation.  The Gunrock GPU implementation is approximately 7.9x faster than the Gunrock CPU implementation.  However, this dataset may be too small for these numbers to be very precise.

##### Hollywood-2009 graph (|U|=1139905, |E|=57515616)

The Hollywood-2009 graph is a graph of Hollywood movie actors, where nodes are actors and edges indicate two actors appear in a movie together.

| implementation | elapsed_ms (avg of 10 runs) |
| -------------- | --------------------------- |
| Python+SNAP    | _> 10 minutes_              |
| Gunrock CPU    | 2035.45                     |
| Gunrock GPU    | __13.793__                  |

<!--
```
num_nodes | num_edges | num_seeds | run_id | ms_elapsed

# HIVE reference
# > 10 minutes

# Gunrock CPU
2258.028
2234.127
2028.867
1982.334
1988.029
1992.302
1984.540
1966.538
1952.640
1967.100
--
mean time = 2035.45

# Gunrock GPU
16.205
23.959
23.586
10.562
10.581
10.704
10.588
10.568
10.571
10.606
--
mean time = 13.793
```
-->

__Takeaway:__ Here, the Gunrock GPU implementation is approximately 150x faster than the Gunrock CPU implementation.  The HIVE reference implementation did not finish in 10 minutes.

##### Indochina-2004 graph (|U|=7414866, |E|= 191606827)

The Indochina-2004 graph is an internet hyperlink graph, generated by a crawl of Asian country domains.

| implementation | elapsed_ms (avg of 10 runs) |
| -------------- | --------------------------- |
| Python+SNAP    | _> 10 minutes_              |
| Gunrock CPU    | 9079.216                    |
| Gunrock GPU    | __22.743__                  |


<!--
```
num_nodes | num_edges | num_seeds | run_id | ms_elapsed

# HIVE reference
# > 10 minutes

# Gunrock CPU
9872.026367
9244.244141
9108.279297
8946.968750
8977.711914
9000.626953
8839.212891
8933.740234
8943.857422
8925.489258
--
mean time = 9079.216

# Gunrock GPU
26.900053
17.898083
27.159929
27.020216
18.491030
17.890930
27.776003
27.983904
18.074989
18.234015
--
mean time = 22.743
```
-->

__Takeaway:__ Here, the Gunrock GPU implementation is approximately 400x faster than the Gunrock CPU implementation.  The HIVE reference implementation did not finish in 10 minutes.

### Implementation limitations

The size of the graph that can be processed will (usually) be limited by the number of edges `|E|` in the input graph. The VN algorithm only allocates an additional 1--3 arrays of size `|U|`, and thus does not require a large amount of storage for temporary data.

The Gunrock VN algorithm works on weighted/unweighted directed/undirected graphs.  No particular graph topology or node/edge metadata is required.  In general, VN would be run on graphs with node and/or edge attributes, but since our Gunrock app only implements context scoring, we are not subject to those restrictions.

### Performance limitations

- Like SSSP, VN is bound by device memory latency.
- Profiling indicates that 64% of time is spent in the Gunrock `advance` operator and 20% of time is spent in in the `filter` operator (pseudocode above).
- The device memory bandwidth is 271 GB/s -- within the expected range for Gunrock graph analytics.  Random memory access means that we don't expect to get close to the reported maximum memory bandwidth.

## Next Steps

### Alternate approaches/further work

Because of its similarity to SSSP, this implementation of VN is fairly hardened.  However, given more time, we could implement more variations on context similiarity as described in Coppersmith and Priebe's paper.  Given the range of potential context similarity functions, this could involve implementing a wide variety of Gunrock operators.

### Gunrock implications

This was a straightforward adapatation of an existing Gunrock app.  SSSP is also one of the simpler apps -- only one advance/filter operation without  complex logic -- so implementing VN was not very difficult.  All of the core logic in VN is identical to SSSP.

### Notes on multi-GPU parallelization

Discussion of multi-GPU scalability of VN can be found [here](README.md#scaling-analysis-for-hive-applications).

### Notes on dynamic graphs

The reference implementation does not cover a dynamic graph version of this workflow, though one could imagine having a static set of seed nodes and a streaming graph on which one would like to compute context scores in real time.

### Notes on larger datasets

If the datasets are larger than a single or multi-GPU's aggregate memory, the straightforward solution would be to let Unified Virtual Memory (UVM) in CUDA automatically handle memory movement. Declaring the entire graph as managed memory for a single GPU implementation, will allow the users to simply oversubscribe for 1 GPU, and queue vertices and edges in from the host memory as needed (this will be very slow). Further optimizations can be done, where instead of utilizing host memory, we can leverage a multi-GPU implementation and move the entire graph over to device memory, using NVLink to move data between the devices' global memory. This can be further optimized by using MemAdvise hints such as pinning the memory to GPU's local memory where it is likely to be used, or create a direct map to all other GPUs to avoid page faulting on first touch.


### Notes on other pieces of this workload

The context scoring component of vertex nomination is incredibly general, and could include versions ranging in complexity from simple Euclidean distance metrics to the output of complex deep learning pipelines.  If we were to integrate these kinds of components more closely w/ Gunrock, we'd likely need to use other CUDA libraries (cuBLAS, cuDNN, etc.) as well as interface w/ higher-level machine learning libraries (TensorFlow, PyTorch, etc.).
