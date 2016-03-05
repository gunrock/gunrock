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
 * @brief bypass filter kernel
 */

#pragma once

#include <gunrock/oprtr/bypass_filter/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {
namespace bypass_filter {

template <
    typename KernelPolicy,
    typename Problem,
    typename Functor,
    bool     VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch
{

};

template <
    typename KernelPolicy,
    typename Problem,
    typename Functor>
struct Dispatch<KernelPolicy, Problem, Functor, true>
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename KernelPolicy::Value    Value;
    typedef typename Problem::DataSlice     DataSlice;
    typedef typename Functor::LabelT        LabelT;

    static __device__ __forceinline__ void  Kernel(
            //VertexId &iteration,
            LabelT &label,
            bool &queue_reset,
            VertexId &queue_index,
            //int &num_gpus,
            SizeT &num_elements,
            //volatile int *&d_done,
            VertexId *&d_in_queue,
            VertexId *&d_out_queue,
            DataSlice *&d_data_slice,
            util::CtaWorkProgress<SizeT> &work_progress,
            SizeT &max_in_frontier,
            SizeT &max_out_frontier,
            util::KernelRuntimeStats &kernel_stats)
    {
        SizeT pos = (SizeT)threadIdx.x + blockIdx.x * blockDim.x;
        const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;

        while (pos < num_elements)
        {
            Functor::ApplyFilter(
                util::InvalidValue<VertexId>(),
                d_in_queue[pos],
                d_data_slice, /*iteration*/
                util::InvalidValue<SizeT>(),
                label,
                pos,
                pos);
            d_out_queue[pos] = d_in_queue[pos];
            if (pos >= num_elements - STRIDE) break;
            pos += STRIDE;
        }
    }
};

template <typename KernelPolicy, typename Problem, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void Kernel(
    //typename KernelPolicy::VertexId         iteration,
    typename Functor::LabelT                label,
    bool                                    queue_reset,
    typename KernelPolicy::VertexId         queue_index,
    typename KernelPolicy::SizeT            num_elements,
    typename KernelPolicy::VertexId         *d_in_queue,
    //typename KernelPolicy::Value            *d_in_value_queue,
    typename KernelPolicy::VertexId         *d_out_queue,
    typename Problem::DataSlice             *d_data_slice,
    //unsigned char                           *d_visited_mask,
    util::CtaWorkProgress<typename KernelPolicy::SizeT> work_progress,
    typename KernelPolicy::SizeT            max_in_queue,
    typename KernelPolicy::SizeT            max_out_queue,
    util::KernelRuntimeStats                kernel_stats)
    //bool                                    filtering_flag = true)
{
    Dispatch<KernelPolicy, Problem, Functor>::Kernel(
        //iteration,
        label,
        queue_reset,
        queue_index,
        //num_gpus,
        num_elements,
        //d_done,
        d_in_queue,
        //d_in_value_queue,
        d_out_queue,
        d_data_slice,
        //d_visited_mask,
        work_progress,
        max_in_queue,
        max_out_queue,
        kernel_stats);
}

template <typename KernelPolicy, typename Problem, typename Functor>
void LaunchKernel(
    unsigned int                            grid_size,
    unsigned int                            block_size,
    size_t                                  shared_size,
    cudaStream_t                            stream,
    //long long                               iteration,
    typename Functor::LabelT                label,
    bool                                    queue_reset,
    unsigned int                            queue_index,
    typename KernelPolicy::SizeT            num_elements,
    typename KernelPolicy::VertexId         *d_in_queue,
    //typename KernelPolicy::Value            *d_in_value_queue,
    typename KernelPolicy::VertexId         *d_out_queue,
    typename Problem::DataSlice             *d_data_slice,
    //unsigned char                           *d_visited_mask,
    util::CtaWorkProgress<typename KernelPolicy::SizeT> work_progress,
    typename KernelPolicy::SizeT            max_in_queue,
    typename KernelPolicy::SizeT            max_out_queue,
    util::KernelRuntimeStats                kernel_stats)
{
    if (queue_reset)
        work_progress.Reset_(0, stream);

    Kernel<KernelPolicy, Problem, Functor>
        <<<grid_size, block_size, shared_size, stream>>>(
        label,
        queue_reset,
        queue_index,
        num_elements,
        d_in_queue,
        //d_in_value_queue,
        d_out_queue,
        d_data_slice,
        work_progress,
        max_in_queue,
        max_out_queue,
        kernel_stats);
}

} // namespace bypass_filter
} // namespace oprtr
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
