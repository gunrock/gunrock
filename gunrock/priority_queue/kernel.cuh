// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kernel.cuh
 *
 * @brief Priority Queue Kernel
 */

#pragma once

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

namespace gunrock {
namespace priority_queue {

/**
 * Arch dispatch
 */

/**
 * Not valid for this arch (default)
 */
template<
    typename    KernelPolicy,
    typename    ProblemData,
    typename    PriorityQueue,
    typename    Functor,
    bool        VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch
{
};

template<
    typename KernelPolicy,
    typename ProblemData,
    typename PriorityQueue,
    typename Functor>
struct Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor, true>
{
    typedef typename KernelPolicy::VertexId     VertexId;
    typedef typename KernelPolicy::SizeT        SizeT;
    typedef typename ProblemData::DataSlice     DataSlice;

    static __device__ __forceinline__ void MarkNF()
    {
    }

    static __device__ __forceinline__ void Compact()
    {
    }
};

template<typename KernelPolicy, typename ProblemData, typename PriorityQueue, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::BLOCKS)
    __global__
void MarkNF()
{
    Dispatch<KernelPolicy, ProblemData, Functor>::MarkNF();
}

template<typename KernelPolicy, typename ProblemData, typename PriorityQueue, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::BLOCKS)
    __global__
void Compact()
{
    Dispatch<KernelPolicy, ProblemData, Functor>::Compact();
}

template <typename KernelPolicy, typename ProblemData, typename PriorityQueue, typename Functor>
    void Bisect()
{
}

} //priority_queue
} //gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
