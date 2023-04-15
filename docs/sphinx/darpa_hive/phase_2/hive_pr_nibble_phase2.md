# Local Graph Clustering (LGC)

The [Phase 1 writeup]((hive_pr_nibble)) contains a detailed description of the application.

From the Phase 1 writeup:

>>From [Andersen et al.](https://projecteuclid.org/euclid.im/1243430567):

>> A local graph partitioning algorithm finds a cut near a specified starting vertex, with a running time that depends largely on the size of the small side of the cut, rather than the size of the input graph.

>A common algorithm for local graph clustering is called PageRank-Nibble (PRNibble), which solves the L1 regularized PageRank problem. We implement a coordinate descent variant of this algorithm found in [Fountoulakis et al.](https://arxiv.org/pdf/1602.01886.pdf), which uses the fast iterative shrinkage-thresholding algorithm (FISTA).

## Scalability Summary

Bottlenecked by single-GPU and communication

## Summary of Results

We rely on Gunrock's multi-GPU `ForALL` operator to implement Local Graph Clustering and observe no scaling as we increase from one to sixteen GPUs. The application is likely bottlenecked by single-GPU filter and advance operators and communication across NVLink necessary to access arrays distributed across GPUs.

## Summary of Gunrock Implementation

The Phase 1 single-GPU implementation is [here](hive_pr_nibble).

We parallelize Local Graph Clustering by utilizing a multi-GPU `ForAll` operator that splits necessary arrays evenly across multiple GPUs. Additional information on multi-GPU `ForAll` can be found in [Gunrock's `ForAll` Operator](#gunrocks-forall-operator) section of the report. In addition, this application depends on single-GPU implementations of Gunrock's advance and filter operations.

### Differences in implementation from Phase 1

No change from Phase 1.

## How To Run This Application on NVIDIA's DGX-2

### Prerequisites
```
git clone  https://github.com/gunrock/gunrock -b multigpu
mkdir build
cd build/
cmake ..
make -j16 pr_nibble
```
**Verify git SHA:** `commit 3e7d4f29f0222e9fd1f4e768269b704d6ebcd02c`

### Partitioning the input dataset

Partitioning is handled automatically as Local Graph Clustering relies on Gunrock's multi-GPU `ForALL` operator and its frontier vertices are split evenly across all available GPUs. Please refer to the chapter on [Gunrock's `ForAll` Operator](#gunrocks-forall-operator) for additional information.

### Running the application (default configurations)

From the `build` directory

```
cd ../examples/pr_nibble/
./hive-mgpu-run.sh
```

This will launch jobs that sweep across 1 to 16 GPU configurations per dataset and application option as specified in `hive-pr_nibble-test.sh`.  See [Running the Applications](#running-the-applications) for additional information.


#### Datasets
**Default Locations:**

```
/home/u00u7u37rw7AjJoA4e357/data/gunrock/gunrock_dataset/mario-2TB/large
```

**Names:**

```
hollywood-2009
europe_osm
```

### Running the application (alternate configurations)

#### hive-mgpu-run.sh


Modify `OUTPUT_DIR` to store generated output and json files in an alternate location.

#### hive-geo-test.sh

Modify `APP_OPTIONS` to specify alternate `--src` and `--max-iter` values.  Please see the Phase 1 single-GPU implementation details [here](hive_pr_nibble) for additional parameter information.

Please review the provided script and see [Running the Applications](#running-the-applications) for details on running with additional datasets.

### Output

No change from Phase 1.

## Performance and Analysis

No change from Phase 1.


### Implementation limitations

No change from Phase 1.

### Performance limitations

**Single-GPU:** No change from Phase 1.

**Multiple-GPUs:** The performance bottleneck is likely due to single-GPU implementations of advance and filter operations randomly accessing numerous arrays distributed across multiple GPUs.


## Scalability behavior

We observe no scaling with Local Graph Clustering as currently implemented. Please see the chapter on [Gunrock's `ForAll` Operator](#gunrocks-forall-operator) for a discussion on future directions around more specialized operators to be designed with communication patterns in mind.
