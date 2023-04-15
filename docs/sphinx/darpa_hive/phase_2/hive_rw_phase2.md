# GraphSearch

The Phase 1 report for GraphSearch can be found [here](hive_graphsearch).

> The graph search (GS) workflow is a walk-based method that searches a graph for nodes that score highly on some arbitrary indicator of interest.

>The use case given by the HIVE government partner was sampling a graph: given some seed nodes, and some model that can score a node as "interesting", find lots of "interesting" nodes as quickly as possible. Their algorithm attempts to solve this problem by implementing several different strategies for walking the graph.

## Scalability Summary

Bottlenecked by network bandwidth between GPUs

## Summary of Results

We rely on a Gunrock's multi-GPU `ForALL` operator to implement GraphSearch as the entire behavior can be described within a single-loop like structure. The core computation focuses on determining which neighbor to visit next based on uniform, greedy, or stochastic functions. Each GPU is given an equal number of vertices to process. No scaling is observed, and in general we see a pattern of decreased performance as we move from 1 to 16 GPUs due to random neighbor access across GPU interconnects.



## Summary of Gunrock Implementation

The Phase 1 single-GPU implementation is [here](hive_graphsearch).

We parallelize across GPUs by using a multi-GPU `ForAll` operator that splits arrays equally across GPUs. For more detail on how `ForAll` was written to be multi-GPU can be found in [Gunrock's `ForAll` Operator](#gunrocks-forall-operator) section of the report.

### Differences in implementation from Phase 1

No change from Phase 1.

## How To Run This Application on NVIDIA's DGX-2

### Prerequisites
```
git clone  https://github.com/gunrock/gunrock -b multigpu
mkdir build
cd build/
cmake ..
make -j16 rw
```
**Verify git SHA:** `commit d70a73c5167c5b59481d8ab07c98b376e77466cc`

### Partitioning the input dataset

How did you do this? Command line if appropriate.

<code>
include a transcript
</code>

### Running the application (default configurations)

From the `build` directory

```
cd ../examples/rw/
./hive-mgpu-run.sh
```

This will launch jobs that sweep across 1 to 16 GPU configurations per dataset and application option as specified across three different test scripts:

* `hive-rw-undirected-uniform.sh`
* `hive-rw-directed-uniform.sh`
* `hive-rw-directed-greedy.sh`

Please see [Running the Applications](#running-the-applications) for additional information.

#### Datasets
**Default Locations:**

```
/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/graphsearch
```

**Names:**

```
dir_gs_twitter
gs_twitter.values
```
### Running the application (alternate configurations)

#### hive-mgpu-run.sh

Modify `OUTPUT_DIR` to store generated output and json files in an alternate location.

#### Additional hive-rw-\*.sh scripts

This application relies on Gunrock's random walk `rw` primitive. Modify `WALK_MODE` to control the application's `--walk-mode` parameter and specify `--undirected` as `true` or `false`. Please see the Phase 1 single-GPU implementation details [here](hive_graphsearch) for additional parameter information.

### Output

No change from Phase 1.


## Performance and Analysis

No change from Phase 1.


### Implementation limitations

No change from Phase 1.

### Performance limitations

**Single-GPU:** No change from Phase 1.

**Multiple-GPUs:** Performance bottleneck is the remote memory accesses from one GPU to another GPU's memory through NVLink.

## Scalability behavior

GraphSearch scales poorly due to low compute (not enough computation per memory access) and high communication costs due to random access patterns (across multiple GPUs) characteristic to the underlying "random walk" algorithm used.
