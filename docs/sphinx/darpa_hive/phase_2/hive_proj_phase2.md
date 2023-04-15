# Graph Projections

The [Phase 1 writeup]((hive_proj)) contains a detailed description of the application.

From the Phase 1 writeup:

> Given a (directed) graph `G`, graph projection outputs a graph `H` such that `H` contains edge `(u, v)` iff `G` contains edges `(w, u)` and `(w, v)` for some node `w`.  That is, graph projection creates a new graph where nodes are connected iff they are neighbors of the same node in the original graph.  Typically, the edge weights of `H` are computed via some (simple) function of the corresponding edge weights of `G`.

> Graph projection is most commonly used when the input graph `G` is bipartitite with node sets `U1` and `U2` and directed edges `(u, v)`.  In this case, the operation yields a unipartite projection onto one of the node sets.  However, graph projection can also be applied to arbitrary (unipartite) graphs.

Note that mathematically this reduces to a sparse-sparse matrix multiplication of `G`'s adjacency matrix.

## Scalability Summary

Limited by load imbalance

## Summary of Results

We implemented a multi-GPU version of sparse-sparse matrix multiplication, based on chunking the rows of the left hand matrix.  This yields a communication-free implementation with good scaling properties.  However, our current implementation remains partially limited by load imbalance across GPUs.

## Summary of Gunrock Implementation

The Phase 1 single-GPU implementation is [here](hive_proj).

In Phase 1, we had two implementations: one using GraphBLAS and one using Gunrock.  The GraphBLAS implementation is more obviously distributed across GPUs, so we build off of that implementation.

`graph_projections` for a symmetric graph is mathematically `H = A @ A`, where `A` is the adjacency matrix of graph `G`.  One way to easily parallelize this operation across GPUs is by partitioning on the rows of the left hand matrix:
```python
H = row_stack([A[start_row:end_row] @ A for start_row, end_row in partition(n_rows)])
```

We parallelize across GPUs by copying the adjacency matrix of `G` to each GPU.  Then, for each GPU, we determine the chunk of rows of the left hand matrix that will be computed on, and each GPU computes `A[start_row:end_row] @ A` for it's respective chunk.  No communication between GPUs is required, except for the initial scatter.

The adjacency matrix `A` is assumed to be randomly permuted and the number of rows in a chunk is constant.  This leads to a coarse-grained load balancing -- each chunk has _roughly_ the same number of nonzero entries.  However, some rows in a power law graph may have orders of magnitude more non-zero entries than others, which does lead to some load imbalance in this application.

### Differences in implementation from Phase 1

The multi-GPU implementation consists of wrapper code around the Phase 1 implementation that distributes `A` across all of the GPUs, launches independent computation on each GPU, and collects the results.

## How To Run This Application on NVIDIA's DGX-2

### Prerequisites

```
git clone https://github.com/owensgroup/graphblas_proj -b dev/mgpu2
cd graphblas_proj
make -j16
```

**Verify git SHA:** `commit c55074593fac49de088ca9afa9d2e82422bccda4`

### Partitioning the input dataset

Data partitioning occurs at runtime whereby matrix `A` is distributed across all available GPUs. Please see the summary above for more information.

### Running the application (default configurations)

```
./hive-mgpu-run.sh
```

This will launch jobs that sweep across 1 to 16 GPU configurations per dataset as specified in `hive-proj-test.sh`.  See [Running the Applications](#running-the-applications) for additional information.

#### Datasets

**Default Locations:**

```
/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/proj_movielens
```

**Names:**

```
ml_1000000
ml_5000000
ml_full
```

### Running the application (alternate configurations)

#### hive-mgpu-run.sh

Modify `OUTPUT_DIR` to store generated output and json files in an alternate location.

#### hive-proj-test.sh

Please review the provided script and see [Running the Applications](#running-the-applications) for details on running with additional datasets. In addition matrix market `.mtx` must first be converted to binary as follows:

```
# convert data to binary
python data/mtx2bin.py --inpath data/ml_full.mtx
```

### Output

No change from Phase 1.

## Performance and Analysis

No change from Phase 1.

### Implementation limitations

Implementation limitations are largely the same as in Phase 1.

The input graph still must fit onto a single GPU, as this parallelization strategy requires the adjacency matrix `A` to be replicated across all GPUs.

However, in the multi-GPU implementation, only `1 / num_gpus` of the _output_ adjacency matrix `H` must fit on a GPU.  This is important, because `H` tends to be a dense matrix, which causes us to run out of GPU memory for even medium-sized graphs `G`.  Thus, the multi-GPU implementation does allow us to run `graph_projections` on larger graphs, approximately linearly with the number of GPUs used.

### Performance limitations

No change from Phase 1 -- in the multi-GPU setting, each GPU is doing almost exactly the same operations as the single-GPU setting, albeit on a subset of the left hand matrix rows.

## Scalability behavior

Scaling is predominantly limited by the presence of load imbalance due to the constant size chunking of rows.  To attain perfect scaling, we would want to use a dynamically allocated chunk of the left hand matrix such that the number of nonzero elements is approximately equal, rather than such that the number of _rows_ is approximately equal.  This is a somewhat non-trivial optimization -- we'd need either some heuristic for creating chunks of rows with _approximately_ the same number of nonzero elements _or_ we'd need to add support for accumulating values across GPUs.  However, we do expect that one of these approaches would lead to further improvements in scaling.

The time it takes to copy the input adjacency matrix `A` to each GPU also contributes to some imperfect scaling, though the cost of this operation tends to be small compared to the cost of the actal computation.
