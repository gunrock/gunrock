# Scan Statistics

Scan statistics, as described in [Priebe et al.](http://www.cis.jhu.edu/~parky/CEP-Publications/PCMP-CMOT2005.pdf), is the generic method that computes a statistic for the neighborhood of each node in the graph, and looks for anomalies in those statistics. In this workflow, we implement a specific version of scan statistics where we compute the number of edges in the subgraph induced by the one-hop neighborhood of each node $u$ in the graph. It turns out that this statistic is equal to the number of triangles that node $u$ participates in plus the degree of $u$. Thus, we are able to implement scan statistics by making relatively minor modifications to our existing Gunrock triangle counting (TC) application.

## Summary of Results

Scan statistics applied to static graphs fits perfectly into the Gunrock framework. Using a combination of `ForAll` and Intersection operations, we outperform the parallel OpenMP CPU reference by up to 45.4 times speedup on the small Enron graph (provided as part of the HIVE workflows) and up to a 580 times speedup on larger graphs that feature enough computation to saturate the throughput of the GPU.

## Algorithm: Scan Statistics
Input is an undirected graph w/o self-loops.

```python
scan_stats = [len(graph.neighbors(u)) for u in graph.nodes]
for (u, v) in graph.edges:
    if u < v:
        u_neibs = graph.neighbors(u)
        v_neibs = graph.neighbors(v)
        for shared_neib in intersect(u_neibs, v_neibs):
            scan_stats[shared_neib] += 1
return argmax([scan_stats(node) for node in graph.nodes])
```

## Summary of Gunrock implementation

```
max_scan_stat = -1
node = -1
src_node_ids[nodes]
scan_stats[nodes]

ForAll (src_node_ids, scan_stats): fill scan_stats with the degree of each node.
Intersection (src_node_ids, scan_stats): intersect neighboring nodes of both
  nodes of each edge; add 1 to scan_stats[common_node] for each common_node
  we get from the intersection.

return [scan_stats]
```

## How To Run This Application on DARPA's DGX-1

### Prereqs/input

```bash
git clone --recursive https://github.com/gunrock/gunrock \
    -b dev-refactor
cd gunrock/tests/ss/
make clean && make
```

### Running the application
Application specific parameters:

Example command-line:

```bash
./bin/test_ss_main_10.0_x86_64 \
    --graph-type=market \
    --graph-file=./enron.mtx \
        --undirected --num_runs=10
```

### Output

The output of this app is an array of uint values: the scan statistics values for each node.  The output file will be in `.txt` format with the aforementioned values.

We compare our GPU output with the HIVE CPU reference implementation implemented using OpenMP.

Details of the datasets:

| DataSet          | $\cardinality{V}$    | $\cardinality{E}$ | #dangling vertices|
|------------------|---------:|-----------:|-------:|
| enron            |    15056 |     57074  |      0 |
| ca               |   108299 |    186878  |  85166 |
| amazon           |   548551 |   1851744  | 213688 |
| coAuthorsDBLP    |   299067 |   1955352* |      0 |
| citationCiteseer |   268495 |   2313294* |      0 |
| as-Skitter       |  1696415 |  22190596* |      0 |
| coPapersDBLP     |   540486 |  30481458* |      0 |
| pokec            |  1632803 |  30622564  |      0 |
| coPapersCiteseer |   434102 |  32073440* |      0 |
| akamai           | 16956250 |  53300364  |      0 |
| soc-LiveJournal1 |  4847571 |  68475391  |    962 |
| europe_osm       | 50912018 | 108109320* |      0 |
| hollywood-2009   | 11399905 | 112751422* |  32662 |
| rgg_n_2_24_s0    | 16777216 | 265114400* |      1 |


Running time in milliseconds:

| GPU  | Dataset          | Gunrock GPU | Speedup vs. OMP | OMP |
|------|------------------|------------:|-----:|-------:|
| P100 | enron            |     **0.461** | 45.4 | 20.95 |
| P100 | ca               |     **0.219** | 71.6 | 15.681 |
| P100 | amazon           |     **1.354** | 74.5 | 100.871 |
| P100 | coAuthorsDBLP    |     **1.569** | 88.0 | 138.111 |
| P100 | citationCiteseer |     **3.936** | 65.22 | 256.694 |
| P100 | as-Skitter       |     **111.738**| 579.22  | 64721.414 |
| P100 | coPapersDBLP     |     **226.672** | 25.4 | 5766.4 |
| P100 | pokec            |     **202.185** | 80.7 | 16316.474 |
| P100 | coPapersCiteseer |     **451.582** | 16.34 | 7378.188 |
| P100 | akamai           |     **151.47** | 5.13 | 12596.36 |
| P100 | soc-LiveJournal1 |     **1.548** | 2.59 | 4.016 |
| P100 | europe_osm       |     **9.59** | 153.35 | 1470.632 |
| P100 | hollywood-2009   |     **10032.46** | 18.76 | 188206.234 |
| P100 | rgg_n_2_24_s0    |     **539.559** | 29.45 | 15887.536 |

## Performance and Analysis

We measure the performance by runtime. The whole process runs on the GPU; we don't include data copy time and regard it as part of preprocessing. The  algorithm is deterministic so does not require any accuracy comparison.

### Implementation limitations

Since the implementation only needs `number of nodes`'s size of memory allocated on the GPU, so the largest dataset that can fit on a single GPU is limited by GPU on-die memory size / number of nodes.

Our implementation regards the graph as undirected and is limited to graphs of that type.

### Performance limitations

We currently use atomic adds to accumulate the number of triangles for each node; atomic operations are relatively slow.

## Next Steps

### Alternate approaches

The most time-consuming operation for scan statistics is the intersection operator. We believe we can improve its performance in future work.

Currently we divide the edge lists into two groups:
(1) small neighbor lists and (2) large neighbor lists. We implement two kernels (`TwoSmall` and `TwoLarge`) that cooperatively compute intersections. Our `TwoSmall` kernel uses one thread to compute the intersection of a node pair. Our `TwoLarge` kernel partitions the edge list into chunks and assigns each chunk to a separate thread block. Then each block uses the balanced path primitive to cooperatively compute intersections. However, these two kernels do not cover the case of one small neighbor list and one large neighbor list. By using a 3-kernel strategy and carefully choosing a threshold value to divide the edge list into two groups, we could potentially process intersections with the same level of workload together, gaining better load balancing and a higher GPU resource utilization.

### Gunrock implications

Gunrock is well-suited for this implementation. The major challenge is the need for accumulation when doing intersection. Intersection was originally designed to count the total number of triangles. But in scan statistics, we instead need the triangle count for each node, which introduces an extra atomic add when doing the accumulation if multiple edges share the same node, since we count triangles in parallel for each edge.

### Notes on multi-GPU parallelization

Non-ideal graph partitioning could be a bottleneck on a multi-GPU parallelization. Because we need the info of each node's two-hop neighbors, an unbalanced workload will decrease the performance and increase the communication bottleneck.

### Notes on dynamic graphs

Scan statistics as a workload certainly has a dynamic-graph component: in its general form, it inputs a time-series graph and tries to identify any abnormal behavior in any statistics change (as mentioned in the referenced papers). To support this general workload, we would need Gunrock to support dynamic graphs in its advance and intersection operators. The application could be easily changed to record all history scan statistics and then to try to find any significant changes given a certain threshold.

### Notes on larger datasets

For a dataset which is too large to fit into a single GPU, we can leverage the multi-GPU implementation of Gunrock to make it work on multiple GPUs on a single node. The implementation won't change a lot since Gunrock already has good support in its multi-GPU implementation. We expect the performance and memory usage to scale linearly with good graph partionling method.

For a dataset that cannot fit onto multiple GPUs on a single node, we need a distributed level of computation, which Gunrock doesn't support yet. However, we can leverage open-source libraries such as NCCL and Horovod that support this. Performance-wise, the way of partitioning the graph as well as the properties of a graph will affect the communication bottleneck. Since we need to calculate the total number of triangles each node is involved in, if we cannot fit the entire neighborhood of a node on a single node, we need other compute resources' help to compute its scan statistic. The worst-case senario is if the graph is fully connected, in which case we must to wait for the counting results from all other compute resources and then sum them up. In this case, if we can do good load-balanced scheduling, we can potentially minimize the communication bottleneck and reduce latency.

### Notes on other pieces of this workload

The other important piece of this work is statistical time series analysis on dynamic changes of the graph. We hope to add dynamic graph capability in the future.

### Research value

This application leverages a classic triangle-counting problem to solve a more complex statistics problem. Instead of directly using the existing solution, we solve it from a different angle: rather than counting the total number of triangles in the graph, we instead count triangles from each node's perspective.

From this app, we see the flexibility of Gunrock's intersection operation. This operation could potentially be used in other graph analytics problems such as community detection, etc.

## References

[Scan Statistics on Enron Graphs](http://www.cis.jhu.edu/~parky/CEP-Publications/PCMP-CMOT2005.pdf)

[Statistical inference on random graphs: Comparative power analyses via Monte Carlo](http://cis.jhu.edu/~parky/CEP-Publications/PCP-JCGS-2010.pdf)
