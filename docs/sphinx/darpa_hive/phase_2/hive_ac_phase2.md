# Application Classification

The [Phase 1 writeup]((hive_application_classification)) contains a detailed description of the application.

From the Phase 1 writeup:

> The application classification (AC) workflow is an implementation of probabalistic graph matching via belief propagation.  The workflow takes two node- and edge-attributed graphs as input -- a data graph `G = (U_G, E_G)` and a pattern graph `P = (U_P, E_P)`.  The goal is to find a subgraph `S` of `G` such that the dissimilarity between the node/edge features of `P` and `S` is minimized. The matching is optimized via loopy belief propagation, which consists of iteratively passing messages between nodes then updating beliefs about the optimal match.

## Scalability Summary

Bottlenecked by network bandwidth between GPUs

## Summary of Results

We re-forumlate the `application_classification` workload to improve memory locality and admit a natural multi-GPU implementation.  We then parallelized the core computational region of `application_classification` across GPUs.  For the kernels in that region that do not require communication between GPUs, we attain near-perfect scaling.  Runtime of the entire application remains bottlenecked by network bandwidth between GPUs.  However, mitigating this bottleneck should be possible further optimization of the memory layout.

## Summary of Implementation

The Phase 1 single-GPU implementation is [here](hive_application_classification).

`application_classification` consists of two regions:
  - Region 1: initialization of distance and feature matrices
  - Region 2: iterative loop consisting of of message passing operations and matrix normalization operations

Region 2 accounts for the majority of runtime.  For example, in our single-GPU implementation running on the `rmat18` `application_classification` benchmark dataset, Region 1 takes 37ms (20% of runtime) and Region 2 takes 157ms (80% of runtime).  As such, we focused on parallelizing Region 2 across GPUs.  A multi-GPU implementation of Region 1 would also be possible, but with diminishing returns.

Upon examination of the Phase 1 `application_classification` implementation, we determined that most of the matrices could be transposed to attain better memory locality.  In the original implementation, there were a number of column-wise operations (max reduce on columns; softmax normalization of columns).  Transposing these matrices converts these into row-wise operations, and yields a substantial speedup.  For example, on the `rmat18` benchmark dataset, this reformulation yields a 6.44x speedup on a single GPU.

"Transposing" the problem also makes it more suitable for multi-GPU parallelism, via row-wise chunking of the data matrices.  Chunks are manually scattered across GPUs using `cudaMemcpy`.  Most of the kernels in Region 2 require _no_ communication between GPUs, which leads to good scaling.  The small amount of communication that is required is done by enabling peer access, with remote memory loads / stores happening over NVLink.

Because it is not a canonical graph workload, `application_classification` is written outside of Gunrock using the `thrust` and `cub` libraries (as in HIVE Phase 1).


## How To Run This Application on NVIDIA's DGX-2

### Prerequisites

The setup process assumes [Anaconda](https://www.anaconda.com/products/individual) is already installed.

```
git clone \
    https://github.com/porumbes/application_classification \
    -b dev/mgpu_manual_reduce

cd application_classification

# prep binary input data
./hive-gen-data.sh

# build
make -j16
```
**Verify git SHA:** `commit 7e20dd05126c174c51b7155cb1f2f9e3084080b3`

### Partitioning the input dataset

Partitioning is done automatically by the application.

### Running the application (default configurations)

```
./hive-mgpu-run.sh
```

This will launch jobs that sweep across 1 to 16 GPU configurations per dataset and application options as specified in `hive-ac-test.sh`. See [Running the Applications](#running-the-applications) for additional information.


#### Datasets

**Default Locations:**

```
/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/application_classification/
```
with subdirectory: `ac_JohnsHopkins_random`


**Names:**

```
rmat18
georgiyPattern
JohnsHopkins
```

### Running the application (alternate configurations)

#### hive-mgpu-run.sh

Modify `OUTPUT_DIR` to store generated output and json files in an alternate location.

#### hive-gen-data.sh

Unlike most of the other applications, Application Classification makes use of an additional script, `hive-gen-data.sh`, to generate necessary input. Please review the chapter on [Running the Applications](#running-the-applications) for information on running with additional datasets.

#### hive-ac-test.sh

Please see the Phase 1 single-GPU implementation details [here](hive_application_classification) for additional parameter information and review the provided script.

Given the setup in `hive-gen-data.sh`, modify the key-value store, `DATA_PATTERN` with the generated `rmat18_data.bin` as the key and the generated `georgiyPattern_pattern.bin` as the value. For example:

```
DATA_PATTERN["rmat18"]="georgiyPattern"
```

### Output

No change from Phase 1.

## Performance and Analysis

No change from Phase 1.

### Implementation limitations

Performance limitations regarding the size of the data matrices are mitigated by the multi-GPU approach -- with this implementation, the maximum size of a problem instance should theoretically scale linearly with the number of GPUs.  Practically, the current implementation still does Region 1 on a single GPU, which would create a bottleneck in terms of available memory.

Other performance limitations remain the same as in Phase 1.

### Performance limitations

From the perspective of a single GPU, there is no change from Phase 1.

From the perspective of the multi-GPU system, we are primarily bottlenecked by bandwidth across the NVLink network, which impacts both the runtime of the row scatter operation and the Region 2 kernels that require communication.  This could be (partially) mitigated by additional optimizations -- more details below.

## Scalability behavior

Scaling of the whole workload's runtime is not ideal, primarily because:
  a) because Region 1 is not parallelized across GPUs
  b) because scattering the rows of the matrices across GPUs takes time.

Region 1 would be relatively straightforward to distribute across GPUs.  The runtime of the scatter could also be reduced via asynchronous memory copies (possibly launched from multiple CPU threads).

The scalability of Region 2 is limited by the couple of kernels that require communication between GPUs, which take ~5x longer to run w/ 4 GPUs than on a single GPU.  Currently, we're bottlenecked by the bandwidth into GPU0 -- scattering an additional datastructure across GPUs would reduce this load by a factor of `num_gpus`, and provide further speedup.  However, this is slightly more complex than the current method, and has not yet been implemented.
