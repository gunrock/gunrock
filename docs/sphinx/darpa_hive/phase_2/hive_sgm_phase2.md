# Seeded Graph Matching (SGM)

The Phase 1 report for SGM can be found [here](hive_sgm).

From [Fishkind et al.](https://arxiv.org/pdf/1209.0367.pdf):

> Given two graphs, the graph matching problem is to align the two vertex sets so as to minimize the number of adjacency disagreements between the two graphs. The seeded graph matching problem is the graph matching problem when we are first given a partial alignment that we are tasked with completing.

That is, given two graphs `A` and `B`, we seek to find the permutation matrix `P` that maximizes the number of adjacency agreements between `A` and `P * B * P.T`, where `*` represents matrix multiplication.  The algorithm Fishkind et al. propose first relaxes the hard 0-1 constraints on `P` to the set of doubly stochastic matrices (each row and column sums to 1), then uses the Frank-Wolfe algorithm to minimize the objective function  `sum((A - P * B * P.T) ** 2)`.  Finally, the relaxed solution is projected back onto the set of permutation matrices to yield a feasible solution.

## Scalability Summary

We observe great scaling

## Summary of Results

Multi-GPU SGM experiences considerable speed-ups over single GPU implementation with a near linear scaling if the dataset being processed is large enough to fill up the GPU. We notice that ~$1$ million nonzeros sparse-matrix is a decent enough size for us to show decent scaling as we increase the number of GPUs. The misalignment for this implementation is also synthetically generated (just like it was for Phase 1, the bottleneck is still the `|V|x|V|` allocation size).

## Summary of Gunrock Implementation

The Phase 1 single-GPU implementation is [here](../hive/hive_sgm).

We parallelize across GPUs by scaling the per-iteration linear assignment problem. In our multi-GPU implementation we ignore the preprocessing step of sparse general matrix multiplication of given input matrices and the trace of matrix products at the very end. For the assignment problem, we use the auction algorithm (also described in the Phase 1 report), where each CUDA block gets a row of the cost matrix and does parallel reductions across the entries of the row using all available threads (with the help of NVIDIA’s CUB library). This allows us to map our rows to each block and explore parallelism within a single row of the matrix in a single-GPU, and split the number of rows across multiple GPUs. Our auction algorithm is implemented using a 2-step process (2-kernels with one fill operation to reset the maximum bids):

1. **Bidding:** Each bidder chooses an object  which brings him/her the best value (benefit-price).
2. **Assign:** Each object chooses a bidder which has the highest bid, and assigns itself to him/her as well as increases the object’s price.

Our experiments conclude that this “bidding” step was the bottleneck for our auction algorithm, and is the only kernel needed to be parallelized across multiple GPUs. For our assignment kernel, it was more effective to use one block to do the final assignment and use one volatile variable to compute the convergence metric.

### Differences in implementation from Phase 1

We now assign each row of the matrix to an entire block instead of a CUDA thread, and process the row in parallel instead of sequentially.

## How To Run This Application on NVIDIA's DGX-2

### Prerequisites
```
git clone https://github.com/owensgroup/SGM -b mgpu
cd SGM/test/
make
```
**Verify git SHA:** `commit d41a43d5653455c1adc59841499ce84a63ecd2db`

### Partitioning the input dataset

Data partitioning occurs at runtime whereby matrix rows are split across multiple GPUs. Please see the summary above for more information.

### Running the application (default configurations)

From the `test` directory

```
./hive-mgpu-run.sh
```

This will launch jobs that sweep across 1 to 16 GPU configurations per dataset as specified in `hive-sgm-test.sh`. **(see `hive_run_apps_phase2.md` for more info)**.

**Please note:** due to an intermittent bug (occassional infinite loop) in the implementation, the scheduled SLURM job is set to timeout after three minutes (all used datasets should complete in under one minute).

#### Datasets
**Default Locations:**

```
/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/seeded-graph-matching/connectome
```

**Names:**

```
DS00833
DS01216
DS01876
DS03231
DS06481
DS16784
```

### Running the application (alternate configurations)

#### hive-mgpu-run.sh

Due to the bug mentioned above, a user may wish to increase or decrease the SLURM job cancellation time. Modify the `--time` options shown here:

```
SLURM_CMD="srun --cpus-per-gpu 2 -G $i -p $PARTITION_NAME -N 1 --time=3:00 "
```

Modify `OUTPUT_DIR` to store generated output and json files in an alternate location.

#### hive-sgm-test.sh

A tolerance value can be specified by setting a value in `APP_OPTIONS`

Please review the provided script and see [Running the Applications](#running-the-applications) for information on running with additional datasets.

### Output

No change from Phase 1.

## Performance and Analysis

No change from Phase 1.


### Implementation limitations

No change from Phase 1.

### Performance limitations

**Single-GPU:** No change from Phase 1.

**Multiple-GPUs:** Our multi-GPU implementation does not consider the SpGEMM preprocessing step. As SpGEMM is one of the core computations for many other algorithms, one future opportunity will be to scale a load-balanced SpGEMM to a multi-GPU system using merge-based decomposition. CUDA’s new virtual memory APIs also allow us to map and unmap physical memory chunks to a contiguous virtual memory array, which can be used to perform and store SpGEMM in its sparse-format without relying on an intermediate dense representation and a conversion to sparse output.

## Scalability behavior

We observe great scaling for our bidding kernel as we increase the number of GPUs. If the input matrix is large enough, the rows can be easily split across multiple GPUs, and each GPU processes its equal share of rows, where within a GPU, each CUDA block processes one complete row.
