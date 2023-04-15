# Community Detection (Louvain)

The [Phase 1 writeup]((hive_louvain)) contains a detailed description of the application.

From the Phase 1 writeup:

> Community detection in graphs means grouping vertices together, so that those vertices that are closer (have more connections) to each other are placed in the same cluster. A commonly used algorithm for community detection is Louvain (<https://arxiv.org/pdf/0803.0476.pdf>).

## Scalability Summary

Application is nonfunctional

## Summary of Results

The application has a segmentation fault and is currently nonfunctional.

## Summary of Gunrock Implementation

The Phase 1 single-GPU implementation is [here](hive_louvain).

We parallelize across GPUs by utilizing Gunrock's multi-GPU `ForAll` operator described [here](#gunrocks-forall-operator).

### Differences in implementation from Phase 1

No change from Phase 1.


## How To Run This Application on NVIDIA's DGX-2

### Prerequisites
```
git clone  https://github.com/gunrock/gunrock -b multigpu
mkdir build
cd build/
cmake ..
make -j16 louvain
```
**Verify git SHA:** `commit d70a73c5167c5b59481d8ab07c98b376e77466cc`

### Partitioning the input dataset

Partitioning is handled automatically as Community Detection relies on Gunrock's multi-GPU `ForALL` operator and its data is split evenly across all available GPUs

### Running the application

Once functional, the application will follow the two script approach described in [Running the Applications](#running-the-applications) (i.e., using `hive-mgpu-run.sh` and `hive-louvain-test.sh` scripts).

#### Datasets

Final datasets will be listed when the application is functional.


### Output

No change from Phase 1.


## Performance and Analysis

No change from Phase 1.


### Implementation limitations

Currently nonfunctional.

### Performance limitations

Currently nonfunctional.

## Scalability behavior

Currently unavailable, but unlikely to scale given its `ForAll` based implementation. See [Gunrock's `ForAll` Operator](#gunrocks-forall-operator) for additional information.
