#pragma once
#include <wrapper/cuda/kernel_policy.cuh>

namespace wrapper {
namespace cuda {

template <typename KernelPolicy,
          typename ProblemData,
          typename Functor,
          bool VALID = (KernelPolicy::CUDA_ARCH >= 300)>
struct Dispatch
{
    typedef typename KernelPolicy::Value Value;
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ void Compute(
            int   *&d_in_queue,
            int     &num_elements,
            DataSlice *&problem) {}
};

template <typename KernelPolicy,
          typename ProblemData,
          typename Functor>
struct Dispatch<KernelPolicy, ProblemData, Functor, true>
{
    typedef typename KernelPolicy::Value Value;
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ void Compute(
            int   *&d_in_queue,
            int     &num_elements,
            DataSlice *&problem)
    {
        int tid = threadIdx.x;
        int bid = blockIdx.x;

        int my_id = bid*blockDim.x + tid;
        if (my_id >= num_elements)
            return;
        Functor::Apply(d_in_queue[my_id], problem);
    }
};

template <typename KernelPolicy, typename ProblemData, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::MIN_CTA_OCCUPANCY)
__global__
void Compute(
    int   *d_in_queue,
    int     num_elements,
    typename ProblemData::DataSlice *problem) {
        Dispatch<KernelPolicy, ProblemData, Functor>::Compute(
                d_in_queue,
                num_elements,
                problem);
    }

} //cuda
} //wrapper
