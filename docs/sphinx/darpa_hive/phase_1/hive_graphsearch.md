# GraphSearch

The graph search (GS) workflow is a walk-based method that searches a graph for nodes that score highly on some arbitrary indicator of interest.

The use case given by the HIVE government partner was sampling a graph: given some seed nodes, and some model that can score a node as "interesting", find lots of "interesting" nodes as quickly as possible.  Their algorithm attempts to solve this problem by implementing several different strategies for walking the graph.

 - `uniform`: given a node `u`, randomly move to one of `u`'s neighbors (ignoring scores)
 - `greedy`: given a node `u`, walk to neighbor with maximum score
 - `stochastic_greedy`: given a node `u`, choose neighbor to walk to with probability proportional to score

Use of these walk-based methods is motivated by the presence of homophily in many real world social networks: we expect interesting people to have relationships with interesting people.

## Summary of Results

Graph search is a relatively minor modification to Gunrock's random walk application, and was straightforward to implement.  Though random walks are a "worst case scenario" for GPU memory bandwidth, we still achieve 3--5x speedup over a modified version of the OpenMP reference implementation.

The original OpenMP reference implementation actually ran slower with more threads -- we fixed the bugs, but the benchmarking experience highlights the need for performant and hardened CPU baselines.

Until recently, Gunrock did not support parallelism _within_ the lambda functions run by the `advance` operator, so neighbor selection for a given step in the walk is done sequentially.  Methods for exposing more parallelism to the programmer are currently being developed via parallel neighbor reduce functions.

In an end-to-end graph search application, we'd need to implement the scoring function as well as the graph walk component.  For performance, we'd likely want to implement the scoring function on the GPU as well, which makes this a good example of a "Gunrock+X" app, where we'd need to integrate the high-performance graph processing component with arbitrary user code.

## Summary of Gunrock Implementation

The scoring model can be an arbitrary function (e.g., of node metadata).  For example, if we were running GS on the Twitter friends/followers graph, the scoring model might be the output of a text classifier on each users' messages.  Thus, we do not implement the scoring model in our Gunrock implementation -- instead, we read scores from an input file and access them as necessary.

GS is a generalization of a random walk implementation, where there can be more variety in the transition function between nodes.

The GS `uniform` mode is exactly a uniform random walk, so we can use the pre-existing Gunrock application.  Given a node, we compute the node to walk to as:

```python
r = random.uniform(0, 1)
neighbors = graph.get_neighbors(node)
next_node = neighbors[floor(r * len(neighbors))]
```

Both the `GraphSearch` `greedy` and `stochastic_greedy` consist of small modifications to this transition function.

For `greedy`, we find the neighbor with maximum score:

```python
neighbors = graph.get_neighbors(node)
next_node = neighbors[0]
next_node_score = scores[next_node]
for neighbor in neighbors:
    neighbor_score = scores[neighbor]
    if neighbor_score > next_node_score:
        next_node = neighbor
        next_node_score = neighbor_score
```

For `stochastic_greedy`, we sample neighbors proportional to their score -- e.g.:

```python
sum_neighbor_scores = 0
for neighbor in graph.neighbors(node):
   sum_neighbor_scores += scores[neighbor]

r *= sum_neighbor_scores

tmp = 0
for neighbor in graph.neighbors(node):
   tmp += scores[neighbor]
   if r < tmp:
       next_node = neighbor
       break
```

In Gunrock, we create a frontier containing all of the nodes we want to walk from.  Then we map the transition function over the frontier using Gunrock's `ForEach` operator.  Current nodes in the frontier are replaced with the chosen neighbor, and the walk is (optionally) recorded in an output array.

Because this is such a straightforward modification, we implement GS inside of the existing random walk `rw` Gunrock application.  GS just requires adding a couple of extra flags and one extra array of size `|V|` to store the node values.

## How To Run This Application on DARPA's DGX-1

### Prereqs/input

```bash
git clone --recursive https://github.com/gunrock/gunrock -b dev-refactor
cd gunrock/tests/rw/
cp ../../gunrock/util/gitsha1.c.in ../../gunrock/util/gitsha1.c
make clean
make
```

### Running the application

#### Application specific parameters
```
  --walk-mode
      0 = uniform
      1 = greedy
      2 = stochastic_greedy
  --node-value-path
      If --walk-mode != 0, this is the path to node scores
  --store-walks
      0 = just do the walk -- don't actually store it anywhere
      1 = store walks in memory
  --walk-length
      Length of each walk
  --walks-per-node
      Number of walks to do per seed node
  --seed
      Seed for random number generator
```

#### Example Command
```bash
# generate random features
python random-values.py 39 > chesapeake.values

# uniform random
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file \
  ../../dataset/small/chesapeake.mtx --walk-mode 0 --seed 123

# greedy
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file \
  ../../dataset/small/chesapeake.mtx --node-value-path chesapeake.values \
  --walk-mode 1

# stochastic greedy
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file \
  ../../dataset/small/chesapeake.mtx --node-value-path chesapeake.values \
  --walk-mode 2 --seed 123
```

#### Example Output
```
# ------------------------------------------------
# uniform random

Loading Matrix-market coordinate-formatted graph ...
Reading from ../../dataset/small/chesapeake.mtx:
  Parsing MARKET COO format
 (39 nodes, 340 directed edges)...
Done parsing (0 s).
  Converting 39 vertices, 340 directed edges ( ordered tuples) to CSR format...
Done converting (0s).
__________________________
--------------------------
 Elapsed: 0.001907
Using advance mode LB
Using filter mode CULL
num_nodes=39
__________________________
0    0   0   queue3      oversize :  234 ->  682
0    0   0   queue3      oversize :  234 ->  682
0    1   0   queue3      oversize :  682 ->  1085
0    1   0   queue3      oversize :  682 ->  1085
0    5   0   queue3      oversize :  1085 ->     1166
0    5   0   queue3      oversize :  1085 ->     1166
--------------------------
Run 0 elapsed: 4.551888, #iterations = 10
[[0, 38, 8, 35, 11, 25, 13, 27, 37, 7, ],
[1, 34, 1, 38, 30, 38, 29, 37, 7, 37, ],
[2, 17, 2, 38, 4, 38, 10, 18, 14, 28, ],
...
[36, 33, 0, 22, 38, 27, 37, 18, 38, 8, ],
[37, 21, 31, 17, 25, 17, 18, 32, 37, 26, ],
[38, 7, 8, 34, 6, 5, 6, 5, 38, 19, ]]
-------- NO VALIDATION -----[rw] finished.
 avg. elapsed: 4.551888 ms
 iterations: 10
 min. elapsed: 4.551888 ms
 max. elapsed: 4.551888 ms
 load time: 60.925 ms
 preprocess time: 964.890000 ms
 postprocess time: 0.715017 ms
 total time: 970.350027 ms

# ------------------------------------------------
# greedy
# !! In this case, the output is formatted as `GPU_result:CPU_result`, for correctness checking

Loading Matrix-market coordinate-formatted graph ...
Reading from ../../dataset/small/chesapeake.mtx:
  Parsing MARKET COO format
 (39 nodes, 340 directed edges)...
Done parsing (0 s).
  Converting 39 vertices, 340 directed edges ( ordered tuples) to CSR format...
Done converting (0s).
__________________________
--------------------------
 Elapsed: 0.085831
Using advance mode LB
Using filter mode CULL
num_nodes=39
__________________________
0    0   0   queue3      oversize :  234 ->  682
0    0   0   queue3      oversize :  234 ->  682
0    1   0   queue3      oversize :  682 ->  770
0    1   0   queue3      oversize :  682 ->  770
--------------------------
Run 0 elapsed: 0.695944, #iterations = 10
[[0:0, 22:22, 32:32, 18:18, 11:11, 18:18, 11:11, 18:18, 11:11, 18:18, ],
[1:1, 22:22, 32:32, 18:18, 11:11, 18:18, 11:11, 18:18, 11:11, 18:18, ],
[2:2, 17:17, 2:2, 17:17, 2:2, 17:17, 2:2, 17:17, 2:2, 17:17, ],
...
[36:36, 33:33, 36:36, 33:33, 36:36, 33:33, 36:36, 33:33, 36:36, 33:33, ],
[37:37, 18:18, 11:11, 18:18, 11:11, 18:18, 11:11, 18:18, 11:11, 18:18, ],
[38:38, 2:2, 17:17, 2:2, 17:17, 2:2, 17:17, 2:2, 17:17, 2:2, ]]
0 errors occurred.
[rw] finished.
 avg. elapsed: 0.695944 ms
 iterations: 10
 min. elapsed: 0.695944 ms
 max. elapsed: 0.695944 ms
 load time: 44.2419 ms
 preprocess time: 974.721000 ms
 postprocess time: 0.731945 ms
 total time: 976.338863 ms

# ------------------------------------------------
# stochastic_greedy
# Output same format as `uniform` above.
# No correctness checking is implemented due to stochasticity.
```

#### Expected Output

When run in `--verbose` mode, the app outputs the walks.  When run in `--quiet` mode, it outputs performance statistics (e.g., total number of steps taken).  If running `greedy` `GraphSearch`, the app also outputs the results of a correctness check.  Correctness checks for `uniform` and `stochastic_greedy` are omitted because of their inherent stochasticity.

### Validation

The correctness of the implementation has been validated in outside experiments, by making sure that the output walks are valid and the distribution of transitions is as expected.

## Performance and Analysis

Performance is measured by the runtime of the app, given

 - an input graph `G=(U, E)`
 - set of seed nodes (hardcoded to all nodes in `G`)
 - number of walks per seed
 - number of steps per walk
 - a transition function (e.g., `uniform|greedy|stochastic_greedy`)

### Implementation limitations

The output of the random walk is a dense array of size `(# seeds) * (steps per walk) * (walks per seed)`.  When we have a large graph _or_ long walks _or_ multiple walks per seed, this array may exceed the size of GPU memory.

At the moment, we only support walks starting from _all_ of the nodes in `G`.  It would be straightforward to add a parameter that would allow the use to specify a smaller set of seed nodes.

This app can only be used for graphs that have scores associated w/ each node.  In order to run benchmarks, if scores are not available we often assign uniformly random scores to nodes.  The distribution of these scores may affect the runtime of the algorithm by changing data access patterns -- we test on the provided Twitter dataset, but do not have a variety of other node attributed graphs to test on.

### Comparison against existing implementations

We measure runtime on the [HIVE graphsearch Twitter dataset](https://hiveprogram.com/data/_v0/graph_search/).  This graph has `|U|=9291392` nodes and `|E|=21741663` edges.

At a high level, the results show:

| Variant           | OpenMP w/ 64 threads | Gunrock GPU | Gunrock Speedup |
| ----------------- | -------------------- | ----------- | --------------- |
| Directed greedy   | 236ms                | __64ms__    | 3.7x
| Directed random   | 158ms                | __34ms__    | 4.6x
| Undirected random | 3186ms               | __630ms__   | 5.0x

The undirected random walks take \~ 10x longer because directed walks terminate when they encounter a node without any neighbors and thus have average length significantly shorter than the `--walk-length` parameter.

Details and raw data follow.

#### HIVE Python reference implementation

We run the HIVE Python reference implementation w/ the following settings:

 - undirected graph
 - uniform transition function
 - 1000 random seeds
 - 128 steps per walk

With the `uniform` transition function, the run took 41 seconds.  Walks are done sequentially, so runtime will scale linearly with the number of seeds.  This implementation is _substantially_ slower than even a single-threaded run of PNNLs OpenMP code.  Thus, we omit further analysis.

#### PNNL OpenMP implementation

We run the PNNL OpenMP implementation on the Twitter graph w/ the following settings:

 - commit: 69864383f0fc0e8aace52be34b329a2f8a58afb6
 - 1,2,4,8,16,32 or 64 threads
 - `greedy` or `uniform` transition function
 - directed or undirected graph

We omit the `greedy` undirected case because the algorithm gets stuck jumping between a local maximum and its highest-scoring neighbor.


| threads | method | directed? | nseeds   | elapsed_sec | nsteps   | steps_per_sec |
| --------| -------| ----------| ---------| ------------| ---------| --------------|
| 1       | greedy | yes       | 7199978  | 3.02876      | 16325873 | 5.39e+06   |
| 2       | greedy | yes       | 7199978  | 2.83467      | 16325873 | 5.75e+06   |
| 4       | greedy | yes       | 7199978  | 1.64405      | 16325873 | 9.93e+06   |
| 8       | greedy | yes       | 7199978  | 0.870028     | 16325873 | 1.87e+07   |
| 16      | greedy | yes       | 7199978  | 0.605769     | 16325873 | 2.69e+07   |
| 32      | greedy | yes       | 7199978  | 0.43742      | 16325873 | 3.73e+07   |
| 64      | greedy | yes       | 7199978  | __0.236701__ | 16325873 | 6.89e+07   |
| 1       | unif.  | yes       | 7199978  | __14.6291__  | 14510781 | __991915__     |
| 2       | unif.  | yes       | 7199978  | 24.2175      | 14186833 | 585809     |
| 4       | unif.  | yes       | 7199978  | 25.1764      | 14487202 | 575427     |
| 8       | unif.  | yes       | 7199978  | 27.7312      | 13937449 | 502591     |
| 16      | unif.  | yes       | 7199978  | 30.5377      | 14062226 | 460488     |
| 32      | unif.  | yes       | 7199978  | 32.1057      | 13906144 | 433137     |
| 64      | unif.  | yes       | 7199978  | 31.2754      | 13876284 | 443680     |
| 1       | unif.  | no        | 100000   | __12.3982__  | 12700000 | __1.024+06__   |
| 2       | unif.  | no        | 100000   | 19.7925      | 12700000 | 641658     |
| 4       | unif.  | no        | 100000   | 22.5432      | 12700000 | 563362     |
| 8       | unif.  | no        | 100000   | 26.1053      | 12700000 | 486491     |
| 16      | unif.  | no        | 100000   | 28.275       | 12700000 | 449160     |
| 32      | unif.  | no        | 100000   | 28.334       | 12700000 | 448224     |
| 64      | unif.  | no        | 100000   | 28.7419      | 12700000 | 441864     |

<!--
```
threads | method | num_nodes | num_edges | num_seeds | seconds | num_steps | steps_per_second | num_neighbors | neighbors_per_second

==> dir_greedy_gs_twitter <==
# 32 steps, 7199978 seeds

1 argmax 7199978 21741663 7199978 3.02876 16325873 5.39027e+06 298668024 9.86105e+07
2 argmax 7199978 21741663 7199978 2.83467 16325873 5.75936e+06 298668024 1.05363e+08
4 argmax 7199978 21741663 7199978 1.64405 16325873 9.9303e+06 298668024 1.81666e+08
8 argmax 7199978 21741663 7199978 0.870028 16325873 1.87648e+07 298668024 3.43286e+08
16 argmax 7199978 21741663 7199978 0.605769 16325873 2.69507e+07 298668024 4.93039e+08
32 argmax 7199978 21741663 7199978 0.43742 16325873 3.73231e+07 298668024 6.82794e+08
64 argmax 7199978 21741663 7199978 0.236701 16325873 6.89725e+07 298668024 1.26179e+09

==> dir_rand_gs_twitter <==
# 128 steps, 7199978 seeds

1 random 7199978 21741663 7199978 14.6291 14510781 991915 304759742 2.08325e+07
2 random 7199978 21741663 7199978 24.2175 14186833 585809 293767659 1.21304e+07
4 random 7199978 21741663 7199978 25.1764 14487202 575427 350743076 1.39314e+07
8 random 7199978 21741663 7199978 27.7312 13937449 502591 295315224 1.06492e+07
16 random 7199978 21741663 7199978 30.5377 14062226 460488 298425418 9.77237e+06
32 random 7199978 21741663 7199978 32.1057 13906144 433137 294176217 9.16275e+06
64 random 7199978 21741663 7199978 31.2754 13876284 443680 293351344 9.37961e+06

==> undir_rand_gs_twitter <==
# 128 steps, 10000 seeds
# !! Note, we run w/ 10K seeds because this is so slow

1 random 7199978 43483326 100000 12.3982 12700000 1.02434e+06 817537505 6.594e+07
2 random 7199978 43483326 100000 19.7925 12700000 641658 683870147 3.4552e+07
4 random 7199978 43483326 100000 22.5432 12700000 563362 579915539 2.57246e+07
8 random 7199978 43483326 100000 26.1053 12700000 486491 1044279401 4.00026e+07
16 random 7199978 43483326 100000 28.275 12700000 449160 1103122803 3.90141e+07
32 random 7199978 43483326 100000 28.334 12700000 448224 1120470606 3.95451e+07
64 random 7199978 43483326 100000 28.7419 12700000 441864 1041898336 3.62502e+07
```
-->

Note that we use fewer seeds for the undirected uniform case due to slow runtime.

Observe that the `rand` modes have very bad scaling as a function of cores.  After investigation, this was due to two issues.  First, the neighbors were being sampled incorrectly, which led to chaotic behavior.  Second, the app was using a slow random number generator w/ an excessive number of seed resets.  We created a PR to fix those issues [here](https://gitlab.hiveprogram.com/pnnl/graphsearch/merge_requests/1).

After these fixes, runtimes were as follows:

| threads | method | directed? | nseeds   | elapsed_sec | nsteps   | steps_per_sec |
| --------| -------| ----------| ---------| ------------| ---------| --------------|
| 1       | greedy | yes       | 7199978  | 3.02876      | 16325873 | 5.39e+06   |
| 2       | greedy | yes       | 7199978  | 2.83467      | 16325873 | 5.75e+06   |
| 4       | greedy | yes       | 7199978  | 1.64405      | 16325873 | 9.93e+06   |
| 8       | greedy | yes       | 7199978  | 0.870028     | 16325873 | 1.87e+07   |
| 16      | greedy | yes       | 7199978  | 0.605769     | 16325873 | 2.69e+07   |
| 32      | greedy | yes       | 7199978  | 0.43742      | 16325873 | 3.73e+07   |
| 64      | greedy | yes       | 7199978  | __0.236701__ | 16325873 | __6.89e+07__   |
| 1       | unif.  | yes       | 7199978  | 1.49886       | 16529694 | 1.10e+07    |
| 2       | unif.  | yes       | 7199978  | 1.60176       | 16533004 | 1.03e+07    |
| 4       | unif.  | yes       | 7199978  | 0.974128      | 16538957 | 1.69e+07    |
| 8       | unif.  | yes       | 7199978  | 0.455227      | 16534756 | 3.63+07     |
| 16      | unif.  | yes       | 7199978  | 0.257524      | 16528617 | 6.42e+07    |
| 32      | unif.  | yes       | 7199978  | __0.155722__  | 13906144 | __1.06e+08__    |
| 64      | unif.  | yes       | 7199978  | 0.158828      | 16537488 | 1.04e+08    |
| 1       | unif.  | no        | 7199978  | 125.963       | 914397206 | 1.92e+08   |
| 2       | unif.  | no        | 7199978  | 78.927        | 914397206 | 1.80e+08   |
| 4       | unif.  | no        | 7199978  | 39.7097       | 914397206 | 2.96e+08   |
| 8       | unif.  | no        | 7199978  | 22.5195       | 914397206 | 6.35e+08   |
| 16      | unif.  | no        | 7199978  | 11.0047       | 914397206 | 1.12e+09   |
| 32      | unif.  | no        | 7199978  | 5.56317       | 914397206 | __1.85e+09__   |
| 64      | unif.  | no        | 7199978  | __3.18615__   | 914397206 | 1.82e+09   |

<!--
```
==> dir_rand_gs_twitter <==
# 128 steps, 7199978 seeds

1 random 7199978 21741663 7199978 1.49886 16529694 1.10282e+07 288996705 1.92811e+08
2 random 7199978 21741663 7199978 1.60176 16533004 1.03218e+07 289219439 1.80563e+08
4 random 7199978 21741663 7199978 0.974128 16538957 1.69782e+07 289267606 2.9695e+08
8 random 7199978 21741663 7199978 0.455227 16534756 3.6322e+07 289233433 6.35361e+08
16 random 7199978 21741663 7199978 0.257524 16536579 6.42138e+07 289067097 1.12249e+09
32 random 7199978 21741663 7199978 0.155722 16528617 1.06142e+08 288840752 1.85485e+09
64 random 7199978 21741663 7199978 0.158828 16537488 1.04122e+08 289209005 1.8209e+09

==> undir_rand_gs_twitter <==
# 128 steps, 7199978 seeds

1 random 7199978 43483326 7199978 125.963 914397206 7.25927e+06 75440529120 5.98912e+08
2 random 7199978 43483326 7199978 78.927 914397206 1.15853e+07 75440651594 9.55828e+08
4 random 7199978 43483326 7199978 39.7097 914397206 2.3027e+07 75395470017 1.89867e+09
8 random 7199978 43483326 7199978 22.5195 914397206 4.06047e+07 75463823437 3.35104e+09
16 random 7199978 43483326 7199978 11.0047 914397206 8.30914e+07 75429254557 6.85427e+09
32 random 7199978 43483326 7199978 5.56317 914397206 1.64366e+08 75398245733 1.35531e+10
64 random 7199978 43483326 7199978 3.18615 914397206 2.86992e+08 75447277682 2.36798e+10
```
-->

Note the improved runtimes and scaling.  These experiments were run with [this branch](https://gitlab.hiveprogram.com/bjohnson/graphsearch/tree/gunrock_test) at commit 6c25a0687eecebfd4393e86fa4c7308d5594b73d.

All experiments are conducted on the HIVE DGX-1.

#### Gunrock GPU implementation

###### directed, greedy

```bash
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file dir_gs_twitter.mtx \
    --node-value-path gs_twitter.values \
    --walk-mode 1 \
    --walk-length 32 \
    --undirected=0 \
    --store-walks 0 \
    --quick \
    --num-runs 10
```
```
Loading Matrix-market coordinate-formatted graph ...
Reading from dir_gs_twitter.mtx:
  Parsing MARKET COO format
 (7199978 nodes, 21741663 directed edges)...
Done parsing (7 s).
  Converting 7199978 vertices, 21741663 directed edges ( ordered tuples) to CSR format...
Done converting (0s).
==============================================
 advance-mode=LB
Using advance mode LB
Using filter mode CULL
Run 0 elapsed: 65.273046, #iterations = 32
Run 1 elapsed: 64.157963, #iterations = 32
Run 2 elapsed: 64.009190, #iterations = 32
Run 3 elapsed: 64.055920, #iterations = 32
Run 4 elapsed: 64.069033, #iterations = 32
Run 5 elapsed: 64.002037, #iterations = 32
Run 6 elapsed: 64.031839, #iterations = 32
Run 7 elapsed: 64.036846, #iterations = 32
Run 8 elapsed: 64.065933, #iterations = 32
Run 9 elapsed: 64.047098, #iterations = 32
Validate_Results: total_neighbors_seen=298668024
Validate_Results: total_steps_taken=16325873
-------- NO VALIDATION --------
[rw] finished.
 avg. elapsed: 64.174891 ms
 iterations: 32
 min. elapsed: 64.002037 ms
 max. elapsed: 65.273046 ms
 load time: 7086.91 ms
 preprocess time: 1016.620000 ms
 postprocess time: 101.121902 ms
 total time: 2073.837996 ms
```

###### directed, uniform

```bash
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file dir_gs_twitter.mtx \
    --node-value-path gs_twitter.values \
    --walk-mode 0 \
    --walk-length 128 \
    --undirected=0 \
    --store-walks 0 \
    --quick \
    --num-runs 10 \
    --seed 123
```
```
Loading Matrix-market coordinate-formatted graph ...
Reading from dir_gs_twitter.mtx:
  Parsing MARKET COO format
 (7199978 nodes, 21741663 directed edges)...
Done parsing (7 s).
  Converting 7199978 vertices, 21741663 directed edges ( ordered tuples) to CSR format...
Done converting (1s).
==============================================
 advance-mode=LB
Using advance mode LB
Using filter mode CULL
__________________________
Run 0 elapsed: 38.613081, #iterations = 128
Run 1 elapsed: 34.458876, #iterations = 128
Run 2 elapsed: 34.530163, #iterations = 128
Run 3 elapsed: 33.849001, #iterations = 128
Run 4 elapsed: 33.759117, #iterations = 128
Run 5 elapsed: 33.967972, #iterations = 128
Run 6 elapsed: 33.873081, #iterations = 128
Run 7 elapsed: 33.970118, #iterations = 128
Run 8 elapsed: 33.756971, #iterations = 128
--------------------------
Run 9 elapsed: 33.715963, #iterations = 128
Validate_Results: total_neighbors_seen=289124779
Validate_Results: total_steps_taken=16530404
-------- NO VALIDATION --------
[rw] finished.
 avg. elapsed: 34.449434 ms
 iterations: 128
 min. elapsed: 33.715963 ms
 max. elapsed: 38.613081 ms
 load time: 7176.17 ms
 preprocess time: 1016.720000 ms
 postprocess time: 101.902962 ms
 total time: 1781.071901 ms
```

###### undirected, uniform

```bash
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file undir_gs_twitter.mtx \
    --node-value-path gs_twitter.values \
    --walk-mode 0 \
    --walk-length 128 \
    --store-walks 0 \
    --quick \
    --num-runs 10 \
    --seed 123
```
```
Loading Matrix-market coordinate-formatted graph ...
Reading from undir_gs_twitter.mtx:
  Parsing MARKET COO format
 (7199978 nodes, 43483326 directed edges)...
Done parsing (7 s).
  Converting 7199978 vertices, 43483326 directed edges ( ordered tuples) to CSR format...
Done converting (0s).
==============================================
 advance-mode=LB
Using advance mode LB
Using filter mode CULL
Run 0 elapsed: 636.021852, #iterations = 128
Run 1 elapsed: 631.129026, #iterations = 128
Run 2 elapsed: 631.053925, #iterations = 128
Run 3 elapsed: 631.713152, #iterations = 128
Run 4 elapsed: 631.028175, #iterations = 128
Run 5 elapsed: 631.374836, #iterations = 128
Run 6 elapsed: 631.196976, #iterations = 128
Run 7 elapsed: 632.030964, #iterations = 128
Run 8 elapsed: 631.026983, #iterations = 128
Run 9 elapsed: 630.996943, #iterations = 128
Validate_Results: total_neighbors_seen=75443835041
Validate_Results: total_steps_taken=914397206
-------- NO VALIDATION --------
[rw] finished.
 avg. elapsed: 631.757283 ms
 iterations: 128
 min. elapsed: 630.996943 ms
 max. elapsed: 636.021852 ms
 load time: 7705.9 ms
 preprocess time: 1010.830000 ms
 postprocess time: 102.057934 ms
 total time: 7755.448818 ms
```

### Performance limitations

- For the undirected uniform settings, profiling shows that 79% of compute time is spent in the `ForAll` operator and 20% is spent in the `curand` random number generator.  Device memory bandwidth in the `ForAll` kernel is 193 GB/s.
- For the directed greedy settings, profiling shows that 99.5% of compute time is spent in the `ForAll` operator.  Device memory bandwidth in the `ForAll` kernel is 136 GB/s.

When we do a large number of walks and/or the length of each walk is very long, there may not be enough GPU memory to store all of the walks in memory.  For now, we expose the `--store-walks` parameter -- when this is set to zero, the walk is discarded as it is computed and only the length of the walk is stored.  A better solution that could be implemented in the future would be to move walks from GPU to CPU memory as they grow too large.

__Optimization:__ In a directed walk, once we hit a node with no outgoing neighbors, we halt the walk.  In the current Gunrock implementation, the enactor runs for a fixed number of iterations, regardless of whether any of the nodes are still active.  It would be straightforward to add a check that terminates the app when no "living" nodes are left.

## Next Steps

### Alternate approaches

The size of the output array may become a significant bottleneck for large graphs.  However, since all of the transition functions do not depend on anything besides the current node, we could reasonably move the results of the walk from GPU to CPU memory every N iterations.  Properly executed, this should eliminate the largest bottleneck without unduely impacting performance.

### Gunrock implications

For the `greedy` and `stochastic_greedy` transition function, we have to sequentially iterate over all of a node's neighbors.  Simple wrappers for computing, e.g., the maximum of node scores across all of a node's neighbors could be helpful, both for ease of programming and performance.  Gunrock has a newly added `NeighborReduce` kernel that supports associative reductions -- it should be straightforward to implement (at least) the `greedy` transition function with this kernel.  The `stochastic_greedy` transition function would require a more complex reduction function along the lines of reservoir sampling.

### Notes on multi-GPU parallelization

If the graph is small enough to be duplicated on each GPU, the implementation is trivial: just do a subset of the walks on each GPU.  The scalability will be perfect, as there is no communication involved at all.

When the graph is distributed across multiple GPUs, we expect to have very poor scalability, as the ratio of computation to communication is very low.  A more detailed discussion is available [here](README.md#scaling-analysis-for-hive-applications).

### Notes on dynamic graphs

This workflow does not have an explicit dynamic component. However, because steps only depend on the current node, the underlying graph could change during the walks.

### Notes on larger datasets

The random accesses inherent to graph search make it a particularly difficult workflow for larger-than-GPU memory datasets.  The most straightforward solution would be to let Unified Virtual Memory (UVM) in CUDA automatically handle memory movement, but we should expect to see a substantial reduction in performance.

### Notes on other pieces of this workload

In real use cases, the scoring function would be computed lazily -- that is, we wouldn't have a precomputed array with scores for each of the nodes, and we would need to run the scoring function as the walk is running.  Thus, it would be critical for us to be able to call the scoring function from within Gunrock quickly and without excessive programmer overhead.
