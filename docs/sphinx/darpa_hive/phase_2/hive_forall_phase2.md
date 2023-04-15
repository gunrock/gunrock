# Gunrock's `ForAll` Operator
Gunrock's `ForAll` operator is a `compute` operator type, meaning, it takes in an input array and applies user-defined function on every element of the input in **parallel**. This input for the `ForAll` operator can in a sense be any element of an array, vertices or edges of a frontier, or all the vertices or edges of the entire graph.
In HIVE's phase I, due to the intuitive nature and simple implementation of the the parallel `ForAll` operator, we found that the operator was very useful in implementing single-GPU versions of several of the HIVE application workloads such as Geolocation, Graph Search, Random Walk, GraphSAGE, computation elements of Local Graph Clustering, Louvain, and Graph Trend Filtering.

The following pseudocode shows a simple-sequential implementation of the `ForAll` operator:
```cpp
template <typename ArrayT, typename ApplyLambda>
void ForAll(ArrayT* array, ApplyLambda apply, std::size_t size) {
  for(std::size_t i = 0; i < size; ++i)
    apply(array, i);
}
```

## Summary of Multi-GPU `ForAll`
In this write-up, we show how Gunrock's `ForAll` operator can be extended to support multiple GPU execution. We also explain what kind of scaling is expected with the new multi-GPU `ForAll` versus the kind of scaling we observe in real-world problems (such as the HIVE applications). We elaborate on what the performance bottlenecks are for our current implementation and what can we do better in the future with specialized-scalable operators targetting interesting patterns present in these applications.

## `ForAll` Implementation
### Approach to Single-GPU
CUDA-based implementation of a parallel `ForAll` operator is a simple extension to the sequential version described above, where instead of looping over the array in a sequential loop, we launch `ceil_div(size, BLOCK_SIZE)` blocks, with $128$ or $256$ threads per block, and each element of the array gets processed by each thread of the parallel CUDA grid launch.
This effectively makes a simple loop-operator, a parallel operator with the ability to apply any arbitrary user-defined operator on every element of the array. Given a single-GPU parallel `ForAll` operator, the users working on the graph algorithms can then write their custom user-defined operators to implement the apps. One example of an application implemented entirely using `ForAll` is Geolocation (described in detail [here](hive_geolocation)). The following snippet is the CUDA-kernel call for Gunrock's `ForAll` operator.

```cpp
template <typename ArrayT, typename ApplyLambda>
__global__ void ForAll_Kernel(ArrayT array, ApplyLambda apply, std::size_t size) {
  const std::size_t STRIDE = blockDim.x * gridDim.x;
  auto thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  while (thread_idx < size) {
    apply(array, thread_idx);
    i += STRIDE;
  }
}
```

### Approach to Multi-GPU
Extending Gunrock's `ForAll` operator from single-GPU to multiple GPUs can be achieved by using CUDA's multi-stream model. A *stream* in CUDA programming model introduces asynchrony such that independent tasks (or kernels) can run concurrently. If, no stream is specified, CUDA assumes the kernels are all running under a special **default** stream called the `NULL` stream. `NULL` stream's behavior is such that each task on the `NULL` stream synchronizes before running the next task, effectively making it sequential. However, it is important for multiple GPU streams to all execute in parallel, therefore, we create a stream for each GPU and a "master" stream, which every stream synchronizes to at the very end to signal that the task has been completed.

The following simplified snippet shows how one can create, launch and synchronize a stream per GPU for the `ForAll` operator:
```cpp
// Create a stream per GPU:
std::vector<cudaStream_t> streams(num_gpus);
for(int i = 0; i < num_gpus; ++i) {
  cudaSetDevice(i);
  cudaStreamCreate(&streams[i]);
}

// Launch kernels on individual streams:
for(int i = 0; i < num_gpus; ++i) {
  cudaSetDevice(i);
  ForAll<<<GRID_DIM, BLOCK_DIM, 0, streams[i]>>>(...);
}

// Synchronize each streams:
for(int i = 0; i < num_gpus; ++i) {
  cudaSetDevice(i);
  cudaStreamSynchronize(streams[i]);
}
```

The above is a great initial formulation to achieve asynchronous _device-side_ launch of our `ForAll` kernel, but we can do better! Even though the device-side execution is now asynchronous with the multi-streams abstraction, on the CPU-side, we are still launching kernels sequentially. We can remedy that by using muiltiple CPU threads to asynchronously launch our kernels on multiple streams from the CPU using OpenMP or C++ threads:
```cpp
// We can use openmp or C++ thread to achieve the multithreaded launch:
#pragma omp parallel for
for(int i = 0; i < num_gpus; ++i) {
  cudaSetDevice(i);
  ForAll<<<GRID_DIM, BLOCK_DIM, 0, streams[i]>>>(...);
}
```

Now, to be able to actually work on individual data elements per GPU, we simply offset the input array by `gpu_id * (size / num_gpus)`, such that each GPU gets a unique section of the work to process.

## Scalability Analysis
### Expected vs. Observed Scaling
Multiple GPUs `ForAll` operator was initially intended as a `transform` operator, where given an array we apply a user-defined transformation on every element of the array. If the user-defined operations are restricted to the array/elements being processed and are simple, the observed scaling is linear. Each GPU gets an embarassingly parallel chunk of work to do independent of every other GPU on the system, therefore, expected scaling to be perfect-linear.

However, what we observe in practice is that the user-defined functions can be complex computations used to implement some of the HIVE workloads. An example pattern that the user may want can be described as following:

1. "Array" being processed in the `ForAll` is an active vertex set of the graph,
2. Therefore, giving access to each vertex in a frontier within the user-defined operation,
3. And in the operation itself, the user may do any random access to other arrays in the algorithm's problem.

These random accesses are observed in many applications, for example, in Geolocation you may want to get the latitude and longitude for each vertex in the graph, and get the latitude and longitude of each of the neighbors of that given vertex to find a spatial-distance. In an ideal case, the neighbor's vertices data is local to each GPU, but in practice, that neighbor could live in any of the GPUs in a system, which causes the GPU processing the neighbor, to incur remote memory transaction causing our expected perfectly linear scaling to fail.

### Performance limitations
For our multi-GPU work, we deploy three different memory schemes for allocating/managing the data that gets split equally among all the GPUs in the systems, these schemes are:

1. Unified Memory Addressing Space (`cudaEnablePeerAccess()`)
2. Unified Memory Management (`cudaMallocManaged()`)
3. CUDA Virtual Memory Management (`cuMemAddressReserve()`)

With the help of prefetching the managed memory (2), all three APIs can perform nearly the same and achieve the goal of allowing all the data within Gunrock to be accessed by all of the GPUs in the system. One additional optimization we deploy is replicating the input graph to all GPUs if the graph is small enough to fit in a single-GPU memory space. As hinted earlier, the performance bottleneck is not within these memory schemes, as they simply split the data and allow it to be accessed from a unified address space, it is within the memory accesses that the user defines within the compute lambda. When lots of remote memory accesses occur from a single-GPU, it can saturate the memory bus causing operations to halt until these transactions are completed. This makes it so that the problems take longer to solve in a multiple GPU system versus a single GPU, because many of the computations are waiting on memory to arrive from remote GPUs. The problem can be reduced by using faster interconnects, such as the new NVLink in the Ampere A100s, but due to Ampere A100s having more compute units as well, the interconnects’ bandwidth is still not enough to saturate the device.

We found that although there are some accesses that are entirely random, many of the user-defined lambdas can be split into multiple parts and common patterns can be further extracted into operators. Once we switch to this specialized-operator model, we can scale our problems better (as further explained in the following section).

### Optimizations and Future Work
One lesson learned from implementing a multiple GPU `ForAll` operator is that there is a need to identify common patterns within the `ForAll` user-defined implementations to be made into operators that can potentially scale. Continuing the previously mentioned Geolocation example, we can look into implementing Geolocation with `NeighborReduction`, where `Reduction` is not a simple reduce, but more complex user-defined operations (such as `spatial-median`). Another reason why moving onto specialized graph operators instead of a general `ForAll` will be better is that we can then map communication patterns within these operators to be able to transfer information at a per-iteration basis between different GPUs using gather, scatter, broadcast (can be achieved using [NCCL](https://developer.nvidia.com/nccl) primitives.) We show one such example with Vertex Nomination, implemented using NCCL, an NVIDIA communication library for multiple GPUs.
