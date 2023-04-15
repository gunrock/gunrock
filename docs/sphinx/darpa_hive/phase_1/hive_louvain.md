# Community Detection (Louvain)

Community detection in graphs means grouping vertices together, so that those vertices
that are closer (have more connections) to each other are placed in the same cluster.
A commonly used algorithm for community detection is Louvain
(<https://arxiv.org/pdf/0803.0476.pdf>).

## Summary of Results

The Gunrock implementation uses sort and segmented reduce to implement the
Louvain algorithm, different from the commonly used hash table mapping. The GPU
implementation is about ~1.5X faster than the OpenMP implementation, and also
faster than previous GPU works. It is still unknown whether the sort and
segmented reduce formulation map the problem better than hash table on the GPU. The
modularities resulting from the GPU implementation are within small differences
as the serial implementation, and are better when the graph is larger. A custom
hash table can potentially improve the running time. The GPU Louvain
implementation should have moderate scalability across multiple GPUs in an
DGX-1.

## Summary of Gunrock Implementation

The commonly used approach to implement the Louvain algorithm uses a hash table. However, the memory access pattern that results from a hash table is almost totally random, and not GPU-friendly (more in the Alternative approaches section). Instead of using a hash table to accumulate the values associated with the same key, the Gunrock implementation on GPU tries another method: sort all key-value pairs, and use segmented reduce to accumulate the values in the continuous segments. Because Louvain always visits all edges in the graph, there is no need to use Gunrock frontiers, and the `advance` operator with the `ALL_EDGES` advance mode or a simple `ForAll` loop should be sufficient. The pseudocode is listed below:

```
m2 <- sum(edge_weights);
//Outer-loop
Do
    // Pass-initialization, assign each vertex to its own community
    For each vertex v in graph:
        current_communities[v] <- v;
        weights_v2any      [v] <- 0;
        weights_v2self     [v] <- 0;
    For each edge e<v, u> in graph: //an advance operator on all edges
        weights_v2any[v] += edge_weights[e];
        if (v == u)
            weights_v2self[v] += edge_weights[e];
    For each vertex v in graph:
        weights_community2any[v] <- weights_v2any[v];
    pass_gain <- 0;

    // Modularity optimizing iterations
    Do
        // Get weights between vertex and community
        For each edge e<v, u> in graph:
            edge_pairs  [e] <- <v, current_communities[u]>;
            pair_weights[e] <- edge_weights[e];
        Sort(edge_pairs, pair_weights)
            by edge_pair.first, then edge_pair.second if tie;
        segment_offsets <- offsets of continuous edge_pairs;
        segment_weights <- SegmentedReduce(pair_weights, segment_offsets, sum);

        // Compute base modularity gains
        // if moving vertices out of current communities
        For each vertex v in graph:
            comm <- current_communities[v];
            w_v2comm <- Find weights from v to comm in segment_weights;
            gain_bases[v] = weights_v2self[v] - w_v2comm
                - (weights_v2any[v] - weights_community2any[comm])
                    * weights_v2any[v] / m2;

        // Find the max gains if moving vertices into adjacent communities
        For each vertex v in graph:
            gains[v] <- 0;
            next_communities[v] <- current_communities[v];
        For each seg<v, comm> segment:
            if (comm == current_communities[v])
                continue;
            gain <- gain_bases[v] + segment_weights[seg]
                - weights_community2any[comm] * weights_v2any[v] / m2;
            atomicMax(max_gains + v, gain);
            seg_gains[seg] <- gain;
        For each seg<v, comm> segment:
            if (seg_gains[seg] != max_gains[v])
                continue;
            next_communities[v] <- comm;

        // Update communities
        For each vertex v in graph:
            curr_comm <- current_communities[v];
            next_comm <- next_communities[v];
            if (curr_comm == next_comm)
                continue;

            atomicAdd(weights_community2any[next_comm],  weights_v2any[v]);
            atomicAdd(weights_community2any[curr_comm], -weights_v2any[v]);
            current_communities[v] <- next_comm;

        iteration_gain <- Reduce(max_gains, sum);
        pass_gain += iteration_gain;
    While iterations stop condition not met
    // End of modularity optimizing iterations

    // Contract the graph
    // renumber occupied communities
    new_communities <- Renumber(current_communities);
    For each edge e<v, u> in graph: //an advance operator on all edges
        edge_pairs[e] <- <new_communities[current_communities[v]],
                          new_communities[current_communities[u]]>
    Sort(edge_pairs, edge_weights) by pair.x, then pair.y if tie;
    segment_offsets <- offsets of continuous edge_pairs;
    new_graph.Allocate(|new_communities|, |segments|);
    new_graph.edges <- first pair of each segments;
    new_graph.edge_values <- SegmentedReduce(edge_weights, segment_offsets, sum);

While pass stop condition not met
```

## How To Run This Application on DARPA's DGX-1

### Prereqs/input

CUDA should have been installed; `$PATH` and `$LD_LIBRARY_PATH` should have been
set correctly to use CUDA. The current Gunrock configuration assumes boost
(1.58.0 or 1.59.0) and Metis are installed; if not, changes need to be made in
the Makefiles. DARPA's DGX-1 has both installed when the tests are performed.

```shell
git clone --recursive https://github.com/gunrock/gunrock/
cd gunrock
git checkout dev-refactor
git submodule init
git submodule update
mkdir build
cd build
cmake ..
cd ../tests/louvain
make
```
At this point, there should be an executable `louvain_main_<CUDA version>_x86_64` in `tests/louvain/bin`.

The datasets are assumed to have been placed in `/raid/data/hive`, and converted to proper matrix market format (.mtx). At the time of testing, `ca`, `amazon`, `akamai`, and `pokec` are available in that directory. `ca` and `amazon` are taken from PNNL's implementation, and originally use 0-based vertex indices; 1 is added to each vertex id to make them proper .mtx files.

The testing is done with Gunrock using the `dev-refactor` branch at commit `2699252`
(Oct. 18, 2018), using CUDA 9.1 with NVIDIA driver 390.30.

### Running the application

```shell
./bin/louvain_main_9.1_x86_64 --omp-threads=32 --iter-stats --pass-stats \
--advance-mode=ALL_EDGES --unify-segments=true --validation=each --num-runs=10 \
--graph-type=market --graph-file=/raid/data/hive/[DataSet]/[DataSet].mtx \
--jsondir=[LogDir] > [LogDir]/[DataSet].txt 2>&1
```

- Add `--undirected`, if the graph is indeed undirected.
- Remove `--iter-stats` or `--pass-stats`, if detailed timings are not required.
- Remove `--validation=each`, to only compute the modularity for the last run.

For example, when DataSet = `akamai`, and LogDir = `eval/DGX1-P100x1`, the command is
```shell
./bin/louvain_main_9.1_x86_64 --omp-threads=32 --iter-stats --pass-stats \
--advance-mode=ALL_EDGES --unify-segments=true --validation=each --num-runs=10 \
--graph-type=market --graph-file=/raid/data/hive/akamai/akamai.mtx \
--jsondir=eval/DGX1-P100x1 > eval/DGX1-P100x1/akamai.txt 2>&1
```

### Output

The outputs are in the [louvain](../attachments/louvain "louvain") directory. Look for the `.txt` files: running time is after `Run x elapsed:`, and the number of communities and the resulting modularity is in the line that starts with `Computed:`. There are 12 runs in each `.txt` file: 1 single thread CPU run for reference, 1 OpenMP multiple-thread (32 threads in the example, may not be optimal) run, and 10 GPU runs.

The output was compared against PNNL's results on the number of communities and
modularity for the amazon and ca datasets. Note that PNNL's code does not count
dangling vertices in communities. The results listed below use the number of
communities minus dangling vertices; the dataset details are can be found in
the next section.

The modularity of resulting communities:

| DataSet | Gunrock GPU | OMP (32T) | Serial | PNNL (8T) | PNNL (serial) |
|---------|----------|----------|----------|----------|----------|
| amazon  | 0.908073 | 0.925721 | 0.926442 | 0.923728 | 0.925557 |
| ca      | 0.711971 | 0.730217 | 0.731292 | 0.713885 | 0.727127 |

Note for these kind of small graphs, more parallelism could hurt the resulting modularity. Multi-thread CPU implementations from both Gunrock and PNNL yield modularities a little less than the serial versions, and the GPU implementation sees a ~0.02 drop. The reason could be concurrent updates to communities: vertex A moves to community C, thinking vertex B is in C; but B may have simultaneously moved to other communities.

However, when input graphs are larger in size, this issue seems to disappear, and modularities from the GPU implementation are sometimes even better than the serial implementation. (See detailed results below.)

The number of resulting communities:

| DataSet | Gunrock GPU | OMP (32T) | Serial | PNNL (8T) | PNNL (serial) |
|---------|------|-----|-----|-----|-----|
| amazon  | 7667 | 213 | 240 | 298 | 251 |
| ca      | 1120 | 616 | 617 | 654 | 623 |

More parallelism also affects the number of resulting communities. On these two small datasets, the GPU implementation produces significantly more communities than all CPU implementations; on large datasets, the differences in the number of communities are much smaller. We believe the reason may also be concurrent community updates, especially when whole-community migration happens: all vertices in community A decide to move to community B, and all vertices in community B decide to move to community A. In the serial implementation, we may see these two communities combine into a single community, but on the GPU, we instead see that community A and B just swap their labels and never become a single community.

## Performance and Analysis

We measure Louvain performance with three metrics: the number of resulting communities (#Comm), the modularity of resulting communities (Q), and the running time (Time, in seconds). Higher Q and lower running time are better. * indicates the graph is given as undirected, and the number of edges is counted after edge doubling and removing of self loops or duplicate edges; otherwise, the graph is taken as directed, and self loops or duplicate edges are also removed. If edge weights are available in the input graph, they follow the input; otherwise, the initial edge weights are set to 1.

Details of the datasets:

| DataSet          | #V    | #E | #dangling vertices|
|------------------|---------:|-----------:|-------:|
| ca               |   108299 |    186878  |  85166 |
| preferentialAttachment | 100000| 999970* |     0 |
| caidaRouterLevel |   192244 |   1218132* |      0 |
| amazon           |   548551 |   1851744  | 213688 |
| coAuthorsDBLP    |   299067 |   1955352* |      0 |
| webbase-1M       |  1000005 |   2105531  |   2453 |
| citationCiteseer |   268495 |   2313294* |      0 |
| cnr-2000         |   325557 |   3128710  |      0 |
| as-Skitter       |  1696415 |  22190596* |      0 |
| coPapersDBLP     |   540486 |  30481458* |      0 |
| pokec            |  1632803 |  30622564  |      0 |
| coPapersCiteseer |   434102 |  32073440* |      0 |
| akamai           | 16956250 |  53300364  |      0 |
| soc-LiveJournal1 |  4847571 |  68475391  |    962 |
| channel-500x100x100-b050 | 4802000 | 85362744 | 0 |
| europe_osm       | 50912018 | 108109320* |      0 |
| hollywood-2009   | 11399905 | 112751422* |  32662 |
| rgg_n_2_24_s0    | 16777216 | 265114400* |      1 |
| uk-2002          | 18520486 | 292243663  |  37300 |

Running time in seconds:

| GPU  | Dataset          | Gunrock GPU | Speedup vs. OMP | OMP | Serial |
|------|------------------|------------:|-----:|-------:|-------:|
| P100 | ca               |       0.108 | 0.24 | **0.026** |  0.065 |
| V100 | ca               |       0.089 | 0.33 | **0.029** |  0.067 |
| V100 | preferentialAttachment | **0.076** | 1.26 | 0.096 |  0.235 |
| V100 | caidaRouterLevel |     **0.063** | 1.03 | 0.065 |  0.229 |
| P100 | amazon           |     **0.160** | 1.27 | 0.203 |  0.648 |
| V100 | amazon           |     **0.122** | 1.62 | 0.198 |  0.631 |
| V100 | coAuthorsDBLP    |     **0.082** | 1.62 | 0.133 |  0.414 |
| V100 | webbase-1M       |     **0.107** | 1.57 | 0.168 |  0.318 |
| V100 | citationCiteseer |     **0.074** | 1.50 | 0.111 |  0.432 |
| V100 | cnr-2000         |       0.235 | 0.57 | **0.133** |  0.388 |
| V100 | as-Skitter       |     **0.376** | 1.76 | 0.660 |  2.480 |
| V100 | coPapersDBLP     |     **0.358** | 1.22 | 0.437 |  1.860 |
| P100 | pokec            |     **0.929** | 1.34 | 1.244 |  6.521 |
| V100 | pokec            |     **0.624** | 1.74 | 1.083 |  6.110 |
| V100 | coPapersCiteseer |     **0.353** | 1.11 | 0.391 |  1.592 |
| P100 | akamai           |     **1.278** | 5.13 | 6.560 | 14.427 |
| V100 | akamai           |     **0.934** | 6.79 | 6.343 | 13.266 |
| V100 | soc-LiveJournal1 |     **1.548** | 2.59 | 4.016 | 16.311 |
| V100 | channel-500x100x100-b050 | 1.133 | 0.68 | **0.768** | 4.449 |
| V100 | europe_osm       |     **4.902** | 7.11 | 34.875 |101.320 |
| V100 | hollywood-2009   |     **1.230** | 1.40 | 1.721 |  9.419 |
| V100 | rgg_n_2_24_s0    |       3.378 | 0.79 | **2.664** | 17.816 |
| V100 | uk-2002          |     **4.921** | 1.15 | 5.682 | 31.006 |

Resulting modularity:

| GPU  | DataSet | Gunrock GPU | Gunrock - Serial | OMP | OMP - Serial | Serial |
|------|------------------|-------:|--------:|-------:|--------:|-------:|
| P100 | ca               | 0.7112 | -0.0193 | 0.7302 | -0.0011 | **0.7313** |
| V100 | ca               | 0.7166 | -0.0147 | 0.7290 | -0.0025 | **0.7313** |
| V100 | preferentialAttachment | 0.1758 | -0.1095 | 0.2287 | -0.0565 | **0.2852** |
| V100 | caidaRouterLevel | **0.8500** | +0.0065 | 0.8362 | -0.0073 | 0.8436 |
| P100 | amazon           | 0.9081 | -0.0184 | 0.9257 | -0.0007 | **0.9264** |
| V100 | amazon           | 0.9089 | -0.0175 | 0.9258 | -0.0006 | **0.9264** |
| V100 | coAuthorsDBLP    | 0.8092 | -0.0179 | 0.8136 | -0.0135 | **0.8271** |
| V100 | webbase-1M       | 0.8945 | -0.0613 | 0.9471 | -0.0087 | **0.9558** |
| V100 | citationCiteseer | 0.7888 | -0.0137 | 0.7605 | -0.0420 | **0.8025** |
| V100 | cnr-2000         | 0.8766 | -0.0031 | 0.8784 | -0.0013 | **0.8797** |
| V100 | as-Skitter       | **0.8366** | +0.0234 | 0.8223 | +0.0091 | 0.8132 |
| V100 | coPapersDBLP     | **0.8494** | +0.0003 | 0.8440 | -0.0051 | 0.8492 |
| P100 | pokec            | 0.6933 | -0.0012 | 0.6914 | -0.0032 | **0.6945** |
| V100 | pokec            | 0.6741 | -0.0204 | 0.6763 | -0.0183 | **0.6945** |
| V100 | coPapersCiteseer | 0.9075 | -0.0035 | 0.9059 | -0.0051 | **0.9110** |
| P100 | akamai           | **0.9334** | +0.0329 | 0.9072 | +0.0067 | 0.9005 |
| V100 | akamai           | **0.9333** | +0.0328 | 0.9074 | +0.0070 | 0.9005 |
| V100 | soc-LiveJournal1 | **0.7336** | +0.0097 | 0.5456 | -0.1782 | 0.7239 |
| V100 | channel-500x100x100-b050 | 0.9004 | +0.0498 | **0.9512** | +0.1007 | 0.8505 |
| V100 | europe_osm       | **0.9979** | +0.0142 | 0.9844 | +0.0008 | 0.9836 |
| V100 | hollywood-2009   | 0.7432 | -0.0079 | 0.7502 | -0.0010 | **0.7511** |
| V100 | rgg_n_2_24_s0    | **0.9921** | +0.0026 | 0.9920 | +0.0024 | 0.9896 |
| V100 | uk-2002          | 0.9507 | -0.0098 | 0.9604 | +0.0000 | **0.9604** |

The number of resulting communities:

| GPU  | DataSet          | Gunrock GPU | OMP | Serial |
|------|------------------|-------:|-------:|-------:|
| P100 | ca               |   1120 |    616 |    617 |
| V100 | ca               |   1076 |    615 |    617 |
| V100 | preferentialAttachment | 18 |   14 |     39 |
| V100 | caidaRouterLevel |    410 |    467 |    745 |
| P100 | amazon           |   7667 |    213 |    240 |
| V100 | amazon           |   7671 |    233 |    240 |
| V100 | coAuthorsDBLP    |     95 |    138 |    273 |
| V100 | webbase-1M       |   4430 |   1469 |   1362 |
| V100 | citationCiteseer |     67 |     48 |    141 |
| V100 | cnr-2000         |  65621 |  59219 |  59253 |
| V100 | as-Skitter       |    924 |   1945 |   2531 |
| V100 | coPapersDBLP     |     70 |    111 |    237 |
| P100 | pokec            | 154988 | 161709 | 166156 |
| V100 | pokec            | 155100 | 162464 | 166156 |
| V100 | coPapersCiteseer |    108 |    110 |    358 |
| P100 | akamai           |  90285 | 130639 | 145785 |
| V100 | akamai           |  90245 | 127843 | 145785 |
| V100 | soc-LiveJournal1 | 506826 | 434272 | 447426 |
| V100 | channel-500x100x100-b050 | 24 | 54 |     12 |
| V100 | europe_osm       |  17320 | 784171 | 828662 |
| V100 | hollywood-2009   |  12218 |  12593 |  12741 |
| V100 | rgg_n_2_24_s0    |    344 |    359 |    311 |
| V100 | uk-2002          | 2402560| 2245355| 2245678|

### Implementation limitations

- **Memory usage** Each edge in the graph needs at least 88 bytes of GPU device memory:  32 bits for the destination vertex, 64 bits for edge weight, 64 bits x 2 x 4 for edge pairs and sorting, 32 bits for segment offsets and 64 bits for segment weights (assuming both vertex and edge ids are represented as 32-bit integers and edge weights are represented as 64-bit floating point numbers). So using a 32 GB GPU, the maximum graph the Louvain implementation can process is roughly 300 million edges. The largest graph successfully run so far is `uk-2002` with 292M edges and 18.5M  vertices.

- **Data types** The edge weights and all computation around them are double-precision floating point values (64-bit `double` in C/C++); we tried single precision and found it resulted in very poor modularities. The `weight * weight / m2` part in the modularity calculation may be the reason for this limitation. Vertex ids should be 32-bit, as the implementation uses 64-bit integers to represent an edge pair for the sort function; if fast GPU sorting is available for 128-bit integers, 64-bit vertex ids might be used.

### Comparison against existing implementations

We compare against serial CPU and OpenMP implementations; both are Gunrock's CPU reference routines. PNNL's results and previous published works are also referenced, to make sure the resulting modularities and running times are sane.

- **Modularity** The modularities have some variation from different
  implementations, mostly within +/- 0.05. On small graphs, the GPU
implementation sees some modularity drops; on large graphs, the GPU
implementation is more likely to yield modularity at least as good as the serial
implementation. The larger the graph is, the better relative modularity is
expected from Gunrock, as compared to the serial implementation. As mentioned
in the output section, that variation could be caused by concurrent movement of vertices. Small graphs could suffer more than larger graphs, as movements to a community have a higher chance to happen concurrently.

- **Running time** Overall, the Gunrock implementation is 2x to 5x faster than
previous work on GPU (Naim 2017, Cheong 2013). Our OMP implementation is a bit
faster than PNNL's, and much faster than previous work using multiple CPU
threads (Lu 2015, Naim 2016). The sequential CPU is an order of magnitude faster than previous
sequential CPU work (Naim 2017, Naim 2016, Blondel 2008, Cheong 2013, Lu 2015).
Comparing across different Gunrock implementations, the GPU is not always the
fastest: on small graphs, GPU could actually be slower, caused by GPU kernel
overheads and hardware underutilization; so for small graphs, the OpenMP
implementation may be a better choice. Gunrock's GPU implementation in practice
runs slower than OpenMP for mesh-like graphs, in which every vertex only has a
very small neighbor list; the parallel formulation Gunrock uses does not work
well on this kind of graph.

Published results (timing and modularity) from previous work are summarized in
the [louvain_results.xlsx](../attachments/louvain/louvain_results.xlsx "Louvain
results") file in the `louvain` directory.

### Performance limitations

The GPU performance bottleneck is the sort function, especially in the first pass. It's true that sort on GPU is much faster than CPU; but the CPU implementations use hash tables, which may not be suitable for the GPU; the alternative approaches section has more details on this.

Using the `akamai` dataset to profile the GPU Louvain implementation on a V100, an iteration in the first pass takes 64.08 ms, and the sort takes 42.05 ms, which is two-thirds of the iteration time. The Louvain implementation uses CUB's radix-sort-pair function. CUB is considered to be one of the GPU primitive libraries that provide the best performance. Further profiling shows during the sort kernel, the device memory controller is ~75% utilized; in other words, the sort is memory bound. This is as expected, as in each iteration of radix sort, the whole `edge_pairs` and `pair_weights` arrays are shuffled, with cost on the order of $O(|E|)$, although the memory accesses are mostly coalesced. The memory system utilizations are as below (from NVIDIA profiler):

|Type     | Transactions | Bandwidth | Utilization |
|--------------|--------:|----------:|-------------|
|Shared Loads  | 2141139 | 1168.859 GB/s | |
|Shared Stores | 2486946 | 1357.636 GB/s | |
|Shared Total  | 4628085 | 2526.495 GB/s | Low |
|L2 Reads      | 2533302 |  345.736 GB/s | |
|L2 Writes     | 2831732 |  386.464 GB/s | |
|L2 Total      | 5365034 |  732.200 GB/s | Low |
|Local Loads   |     588 |   80.248 MB/s | |
|Local Stores  |     588 |   80.248 MB/s | |
|Global Loads  | 2541637 |  346.873 GB/s | |
|Global Stores | 2675820 |  365.186 GB/s | |
|Texture Reads | 2515533 | 1373.242 GB/s | |
|Unified Cache Total| 7734166 | 2085.462 GB/s | Idle to Low |
|Device Reads  | 2469290 |  336.999 GB/s | |
|Device Writes | 2596403 |  354.347 GB/s | |
|**Device Memory Total**| **5065693** | **691.347 GB/s** | **High** |
|PCIe Reads    |       0 |        0 B/s  | None |
|PCIe Writes   |       5 |  682.381 kB/s | Idle to Low |

![Louvain_akamai](../attachments/louvain/Louvain_akamai.png "Memory statistics")

It's clear that the bottleneck is at the device memory: most data are read-once and write-once,  with little possibility to reuse the data. The kernel achieved 691 GB/s bandwidth utilization, ~77% of the 32GB HBM2's 900 GB/s capability. This high-bandwidth utilization fits the memory access pattern: mostly regular and coalesced.

The particular issue here is not the kernel implementation itself, it's the usage of sorting: fully sorting the whole edge list is perhaps overkill. One possible improvement is to use a custom hash table on the GPU, to replace the sort + segmented reduce part. The hash table could also cut the memory requirement by about 50%.

## Next Steps

### Alternate approaches

#### Things we tried that didn't work well

**Segmented sort and CUB segmented reduce**: the original idea for modularity optimization is to use CUB's segmented sort and segmented reduce within each vertex's neighborhood to get the vertex-community weights. However, CUB uses one block per each segment in these two routines, and that creates huge load imbalance, as the neighbor list length can have large differences. The solution is 1) use vertex-community pairs as keys, and CUB's unsegmented sort; 2) implement a load-balanced segmented-reduce kernel. This solution reduces the sort-reduce time by a few X, and can be enabled by the `--unify-segments` flag on the command line.

**STL hash table on CPU**: the `std::map` class has performance issues when used in a multi-threading environment, and yields poor scalability; it might be caused by using locks in the STL. The solution is to use a size $|V|$ array per thread for the vertices that thread processes. Since within a thread, vertices are processed one by one, and the maximum number of communities is $|V|$, so instead of a hash table, a flat size $|V|$ array can replace the hash table. This solution significantly reduces the running time, even using a single thread. The multi-thread scalability is also much better.

**Vanilla hash table on GPU**: this can't be used, as keys (vertex-community or community-community pair) and the values (edge weights) are both 64-bit, and there is currently no support of atomic-compare-and-exchange operations for 128-bit data. Even if that's available, a vanilla table only provides insert and query operations, but not accumulation of values for the same key.

#### Custom hash table

Hash tables can be used to accumulate the vertex-to-community weights during modularity optimization iterations, and to get the community-to-community edge weights during the graph contraction part of the algorithm. Previous implementations use hash tables as a common practice. However, as mentioned above, vanilla hash tables that store key-value pairs are not a good choice for Louvain.

Louvain not only needs to store the key-value pairs, but also to accumulate  values associated with the same key. That key may be the same vertex-community pair for modularity optimization, or the same community-community pair for graph contraction. Ideally, if the hash table provides the following functionality, it would be much more suitable for Louvain:

- Only insert the key (in the first phase of using the hash table).
- In the next phase, query the _positions_ of the keys, and use atomics to accumulate the values belonging to the same key, in a separate array.
- There must be a barrier between the two phases; all insertions of the first phase need to be visible to the second phase.

This kind of hash table removes the strong restriction that the key-value pair needs to be able to support an atomic compare-and-exchange operation, which is imposed by vanilla hash table implementations. The custom hash table is also more value-accumulation-friendly. It replaces the sort-segmented reduce part, and can reduce the workload from about $O(6|E|)$ to $O(2|E|)$, and the memory requirement from $48|E|$ bytes to $24|E|$. However, the memory access pattern now becomes irregular and uncoalesced, so it is still unknown whether it can actually yield a performance gain.

#### Iteration and pass stop conditions

The concurrent nature of the GPU implementation makes modularity gain
calculation inaccurate. Community migrations are also observed. Because of
these, the optimization iterations and the passes may run more than needed,
and resulted in longer running times, especially in the first pass. Preliminary
experiments to cap the number of iterations for the first pass can reduce the
running time by about 40%, with the modularity value roughly unchanged. The
actual cap is dataset-dependent, and more investigation is needed to get a
better understanding of how to set the cap.

### Gunrock implications

The core of the Louvain implementation mainly uses all-edges advance, sort, segmented reduce, and for loops. The sort-segmented reduce operation is actually a segmented keyed reduce; if that's a common operation that appears  in more algorithms, it could be made into a new operator. The all-edges advance is used quite often in several applications, so wrapping it with a simpler operator interface could be helpful.

### Notes on multi-GPU parallelization

When parallelizing across multiple GPUs, the community assignment of local vertices and their neighbors needs to available locally, either explicitly by moving them using communication functions, or implicitly by peer memory access or unified virtual memory. The communication volume is in the order of $O(|V|)$ and the computation workload is at least in $O(|E|)$, so the scalability should be manageable.

When the number of GPUs is small, 1D partitioning can be used to divide the edges, and replicate the vertices, so there is no need to do vertex id conversion across multiple GPUs. When the number of GPUs is large, the high-low degree vertex separation and partitioning scheme can be used: edges are still distributed across GPUs, high degree vertices are duplicated, and low degree vertices are owned by one GPU each. The boundary to use different partitioning scheme is still unclear, but it's likely that 8 GPUs within a DGX-1 can still be considered as a small number, and use the simple 1D partitioning.

### Notes on dynamic graphs

Louvain is not directly related to dynamic graphs. But it should be able to run on a dynamic graph, provided the way to access all the edges is the same. Community assignment from the previous graph can be used as a good starting point, if the vertex ids are consistent and the graph is not dramatically changed.

### Notes on larger datasets

The bottleneck of memory usage of the current implementation is in the edge pair-weight sort function: that makes up about half of the memory usage. Replacing sort-segmented reduce with a custom hash table could significantly lower the memory usage.

Louvain needs to go over the whole graph once in each modularity optimization iteration. If the graph is larger than the combined GPU memory space, which forces streaming of the graph in each modularity optimization iteration, the performance bottleneck will be CPU-GPU bandwidth. That can be an order of magnitude slower than when the graph can fully fit in GPU memory. Considering the OpenMP implementation is not that slow, using that may well be faster than moving the graph multiple times across the CPU-GPU bridge.

### Notes on other pieces of this workload

All parts of Louvain are graph related, and fully implemented in Gunrock.

### Research potential

Community detection on graphs is generally an interesting research topic,
and others have targeted good GPU implementations for it.
Our work is the first time Louvain is mapped to sort and segmented reduce, so more
comparisons are needed to know whether it's better than the hash table mapping.
The custom hash table implementation is worth trying out, and comparing with the
current sort and segmented reduce implementation. Multi-GPU implementations,
particularly the graph contraction part, can be a place for research
investigation: it may be better to contract the graph onto a single GPU,
or even the CPU, when the graph is small; but where is the threshold and how to
do the multi-GPU to single-GPU or CPU switch?

Label propagation is another algorithm for community detection. It has a similar
structure to Louvain, but has a simpler way to decide how to move the vertices.
Comparing them side by side for both result quality (the modularity value) and
computation speed (the running time) will be beneficial.

**References**

[1] Hao Lu, Mahantesh Halappanavar, Ananth Kalyanaraman. "Parallel Heuristics for Scalable Community Detection", <https://arxiv.org/abs/1410.1237> (2015).

[2] Cheong C.Y., Huynh H.P., Lo D., Goh R.S.M. "Hierarchical Parallel Algorithm for Modularity-Based Community Detection Using GPUs". Euro-Par 2013.

[3] Md. Naim, Fredrik Manne, Mahantesh Halappanavar, and Antonio Tumeo. "Highly Scalable Community Detection Using a GPU", <https://www.eecs.wsu.edu/~assefaw/CSC16/abstracts/naim-CSC16_paper_14.pdf> (2016).

[4] M. Naim, F. Manne, M. Halappanavar and A. Tumeo, "Community Detection on the GPU," IPDPS `17.

[5] Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Etienne Lefebvre, "Fast unfolding of communities in large networks". <https://arxiv.org/abs/0803.0476> (2008).
