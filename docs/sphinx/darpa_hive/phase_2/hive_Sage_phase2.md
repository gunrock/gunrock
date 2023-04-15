# GraphSAGE

The [Phase 1 writeup]((hive_graphSage)) contains a detailed description of the application.

From the Phase 1 writeup:

> GraphSAGE is a way to fit graphs into a neural network: instead of getting the embedding of a vertex from all its neighbors' features as in conventional implementations, GraphSAGE selects some 1-hop neighbors, some 2-hop neighbors connected to those 1-hop neighbors, and computes the embedding based on the features of the 1-hop and 2-hop neighbors. The embedding can be considered as a vector containing hash values describing the interesting properties of a vertex.

## Scalability Summary

Bottlenecked by network bandwidth between GPUs

## Summary of Results

We rely on Gunrock's multi-GPU `ForALL` operator to implement GraphSAGE. We see no scaling as we sweep from one to sixteen GPUs due to communication over GPU interconnects.

## Summary of Gunrock Implementation

The Phase 1 single-GPU implementation is [here](hive_graphSage).

We parallelize across GPUs by utilizing a multi-GPU `For-All` operator and evenly distribute relevant arrays across multiple GPUs. Please see [Gunrock's `ForAll` Operator](#gunrocks-forall-operator) for more details.

### Differences in implementation from Phase 1

No change from Phase 1.

## How To Run This Application on NVIDIA's DGX-2

### Prerequisites
```
git clone  https://github.com/gunrock/gunrock -b multigpu
mkdir build
cd build/
cmake ..
make -j16 sage
```
**Verify git SHA:** `commit d70a73c5167c5b59481d8ab07c98b376e77466cc`

### Partitioning the input dataset

Partitioning is handled automatically as GraphSage relies on Gunrock's multi-GPU `ForALL` operator and its frontier vertices are split evenly across all available GPUs. Please refer to the chapter on [Gunrock's `ForAll` Operator](#gunrocks-forall-operator) for additional information.

### Running the application (default configurations)

From the `build` directory

```
cd ../examples/sage/
./hive-mgpu-run.sh
```

This will launch jobs that sweep across 1 to 16 GPU configurations per dataset and application option as specified in `hive-sage-test.sh`.

  [Running the Applications](#running-the-applications) chapter for details on running with additional datasets

for additional parameter information, review the provided script, and see [Running the Applications](#running-the-applications) chapter for details on running with additional datasets.

#### Datasets
**Default Locations:**

```
/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB
/home/u00u7u37rw7AjJoA4e357/data/gunrock/gunrock_dataset/mario-2TB/large
```

**Names:**

```
pokec
dir_gs_twitter
europe_osm
```

### Running the application (alternate configurations)

#### hive-mgpu-run.sh

Modify `OUTPUT_DIR` to store generated output and json files in an alternate location.

#### hive-sage-test.sh

Modify `APP_OPTIONS` to specify alternate `--undirected` and `--batch-size` options.  Please see the Phase 1 single-GPU implementation details [here](hive_graphSage) for additional parameter information, review the provided script, and see [Running the Applications](#running-the-applications) chapter for details on running with additional datasets.

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

We observe no scaling with the current GraphSAGE implementation. Please see the chapter on [Gunrock's `ForAll` Operator](#gunrocks-forall-operator) for a discussion on future directions around more specialized operators to be designed with communication patterns in mind.
