# Scan Statistics

From the [Phase 1 report](hive_scan_statistics) for Scan Statistics:

> Scan statistics, as described in [Priebe et al.](http://www.cis.jhu.edu/~parky/CEP-Publications/PCMP-CMOT2005.pdf), is the generic method that computes a statistic for the neighborhood of each node in the graph, and looks for anomalies in those statistics. In this workflow, we implement a specific version of scan statistics where we compute the number of edges in the subgraph induced by the one-hop neighborhood of each node $u$ in the graph. It turns out that this statistic is equal to the number of triangles that node $u$ participates in plus the degree of $u$. Thus, we are able to implement scan statistics by making relatively minor modifications to our existing Gunrock triangle counting (TC) application.

## Scalability Summary

Bottlenecked by single-GPU and communication

## Summary of Results

We rely on Gunrock's multi-GPU `ForALL` operator to implement Scan Statistics. We see no scaling and in general performance degrades as we sweep from one to sixteen GPUs. The application is likely bottlenecked by the single GPU intersection operator that requires a two-hop neighborhood lookup and accessing an array distributed across multiple GPUs.

## Summary of Gunrock Implementation

The Phase 1 single-GPU implementation is [here](hive_scan_statistics).


We parallelize Scan Statistics by utilizing a multi-GPU `ForAll` operator that splits the `scan_stats` array evenly across all available GPUs. Additional information on multi-GPU `ForAll` can be found in [Gunrock's `ForAll` Operator](#gunrocks-forall-operator) section of the report. Furthermore, this application depends on triangle counting and an intersection operator that have not been parallelized (i.e., across multiple GPUs). It is not clear that simply parallelizing these functions would lead to scalability due to the communication patterns they exhibit.

### Differences in implementation from Phase 1

No change from Phase 1.

## How To Run This Application on NVIDIA's DGX-2

### Prerequisites
```
git clone  https://github.com/gunrock/gunrock -b multigpu
mkdir build
cd build/
cmake ..
make -j16 ss
```
**Verify git SHA:** `commit d70a73c5167c5b59481d8ab07c98b376e77466cc`

### Partitioning the input dataset

Partitioning is handled automatically as Scan Statistics relies on Gunrock's multi-GPU `ForALL` operator and its `scan_stats` array is split evenly across all available GPUs (see [`ForAll` ](#gunrocks-forall-operator) for details).

### Running the application (default configurations)

From the `build` directory

```
cd ../examples/ss/
./hive-mgpu-run.sh
```

This will launch jobs that sweep across 1 to 16 GPU configurations per dataset and application option as specified in `hive-ss-test.sh`. See [Running the Applications](#running-the-applications) for more details.


#### Datasets
**Default Locations:**

```
/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB
```

**Names:**

```
pokec
```

### Running the application (alternate configurations)

#### hive-mgpu-run.sh


Modify `OUTPUT_DIR` to store generated output and json files in an alternate location.

#### hive-ss-test.sh

Modify `APP_OPTIONS` to specify alternate `--undirected` and `--num-runs` values.  Please see the Phase 1 single-GPU implementation details [here](hive_scan_statistics) for additional parameter information.

Please review the provided script and see "Running the Applications" chapter for details on running with additional datasets.

### Output

No change from Phase 1.

## Performance and Analysis

No change from Phase 1.


### Implementation limitations

No change from Phase 1.


### Performance limitations

**Single-GPU:** No change from Phase 1.

**Multiple-GPUs:** Performance bottleneck is likely the single-GPU implementation of triangle counting and intersection and the need to randomly access an array distributed across multiple GPUs. Though once parallelized across multiple GPUs, the random access patterns of these functions (e.g., two-hop neighborhoods) would bottleneck communication over NVLink.

## Scalability behavior

We observe no scaling with the current Scan Statistics implementation. Please see the chapter on [Gunrock's `ForAll` Operator](#gunrocks-forall-operator) for a discussion on future directions around more specialized operators to be designed with communication patterns in mind.
