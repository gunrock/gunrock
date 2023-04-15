# Vertex Nomination

The [Phase 1 writeup](hive_vn) contains a detailed description of the application.  The most important point to note is that `vertex_nomination` is a "multiple-source shortest paths" algorithm.  The algorithm description and implementation are identical to canonical single-source shorest paths (SSSP), with the minor modification that the search starts from multiple vertices instead of one.

## Scalability Summary

We observe weak scaling

## Summary of Results

We implemented `vertex_nomination` as a standalone CUDA program, and achieve good weak scaling performance by eliminating communication during the `advance` phase of the algorithm and using a frontier representation that allows an easy-to-compute reduction across devices.

## Summary of Gunrock Implementation and Differences from Phase 1

The Phase 1 single-GPU implementation is [here](hive_vn).

In Phase 1, `vertex_nomination` was implemented for a single GPU using the Gunrock framework.  However, The Phase 2 multi-GPU implementation required some functionality that is not currently available in Gunrock, so we implemented it as a standalone CUDA program (using the `thrust` and `NCCL` libraries).

Specifically, the multi-GPU `vertex_nomination` uses a fixed-size (boolean or integer) array to represent the input and output frontiers, while Gunrock predominantly uses a dynamically-sized list of vertex IDs.  The fixed-size representation admits a more natural multi-GPU implementation, and avoids a complex merge / deduplication step in favor of a cheap `or` reduce step.

As described in the Phase 1 report, the core kernel in `vertex_nomination` is the following advance:
```python
def _advance_op(src, dst, distances):
    src_distance   = distances[src]
    edge_weight    = edge_weights[(src, dst)]
    new_distance   = src_distance + edge_weight
    old_distance   = distances[dst]
    distances[dst] = min(old_distance, new_distance)
    return new_distance < old_distance
```

which runs in a loop like the following pseudocode:
```python
# advance
thread_parallel for src in input_frontier:
  thread_parallel for dst in src.neighbors():
    if _advance_op(src, dst, distances):
      output_frontier.add(dst)
```

In the multi-GPU implementation, the loop instead looks like the following pseudocode:

```python
# advance per GPU
device_parallel for device in devices:
  thread_parallel for src in local_input_frontiers[device].get_chunk(device):
    thread_parallel for dst in src.neighbors():
      if _advance_op(src, dst, local_distances[device]):
        local_output_frontiers[device][dst] = True

# reduce across GPUs
local_input_frontiers = all_reduce(local_output_frontiers, op="or")
device_parallel for device in devices:
  local_output_frontiers[device][:] = False

local_distances = all_reduce(local_distances, op="min")
```

In the per-GPU `advance` phase, each device has

- a _local_ replica of the complete input graph
- a chunk of nodes it is responsible for computing on
- a _local_ copy of the `input_frontier` that is read from
- a _local_ copy of the `output_frontier` that is written to
- a _local_ copy of the `distance` array that is read / written

This data layout means that no communication between devices is required during the advance phase.

During the `reduce` phase,

- the local output frontiers are reduced with the `or` operator (remember they are boolean masks)
- the local `distances` arrays are reduced with the `min` operator

After this phase, the copies of the input frontiers and the computed distances are the same on each device.  In our implementation, these reduces uses the `ncclAllReduce` function from NVIDIA's `nccl` library.

## How To Run This Application on NVIDIA's DGX-2

### Prerequisites

The setup process assumes [Anaconda](https://www.anaconda.com/products/individual) is already installed.

```
git clone  git clone https://github.com/porumbes/mgpu_sssp -b main
cd mgpu_sssp
bash install.sh # downloads and compiles NVIDIA's nccl library
make
```
**Verify git SHA:** `commit 4f93307e7a0aa7f71e8ab024771e950e40247a4e`

### Partitioning the input dataset

The input graph is replicated across all devices.

Each device is reponsible for running the `advance` operation on a subset of nodes in the graph (eg, `GPU:0` operates on node range `[0, n_nodes / n_gpus]`, `GPU:1` on `[n_nodes / n_gpus + 1, 2 * n_nodes / n_gpus]`, etc).  Assuming a random node labeling, this correspond to a random partition of nodes across devices.

### Running the application (default configurations)

```
./hive-mgpu-run.sh
```

This will launch jobs that sweep across 1 to 16 GPU configurations per dataset and application options as specified in `hive-vn-test.sh` **(see `hive_run_apps_phase2.md` for more info)**.


#### Datasets
**Default Locations:**

```
/home/u00u7u37rw7AjJoA4e357/data/gunrock/gunrock_dataset/mario-2TB/large
```

**Names:**

```
chesapeake
rmat18
rmat20
rmat22
rmat24
enron
hollywood-2009
indochina-2004
```

### Running the application (alternate configurations)

#### hive-mgpu-run.sh

Modify `NUM_SEEDS` to specify the number of seed locations to be used by `hive-vn-test.sh`.

Modify `OUTPUT_DIR` to store generated output and json files in an alternate location.

#### hive-vn-test.sh

Please see the Phase 1 single-GPU implementation details [here](hive_vn) for additional parameter information, review the provided script, and see [Running the Applications](#running-the-applications) chapter for details on running with additional datasets.

### Output

No change from Phase 1.

## Performance and Analysis

No change from Phase 1.

### Implementation limitations

Implementation limitations are largely the same as in the Phase 1 Gunrock-based implementation.

Note that in the current implementation, the _entire_ input graph is replicated across all devices.  That means that this implementation cannot run on datasets that are large than the memory of a single GPU.

### Performance limitations

The `advance` phase does not include any communication between devices, so the performance limitations are the same as in Phase 1.

The `reduce` phase requires copying and reducing `local_output_frontiers` and `local_distances` across GPUs, and is memory bandwidth bound.

## Scalability behavior

Scaling is not perfectly ideal because of the time taken by the `reduce` phase, which is additional work in the multi-GPU setting that is not present in the single-GPU case.  As the number of GPUs increases, the cost of this communication increases relative to the per-GPU cost of computation, which limits weak scaling of our implementation.

Scaling is primarily limited by the current restriction that the entire input graph must fit in a single GPU's memory.  From a programming perspective, it would be straightforward to partition the input graph across GPUs; however, this would lead to remote memory accesses in the `advance` phase and impact performance substantially.
