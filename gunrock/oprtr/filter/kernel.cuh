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
 * @brief Filter Kernel
 */

#pragma once

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/filter/cta.cuh>

namespace gunrock {
namespace oprtr {
namespace filter {

/**
 * @brief Structure for invoking CTA processing tile over all elements.
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
struct SweepPass
{
    static __device__ __forceinline__ void Invoke(
        typename KernelPolicy::VertexId         &iteration,
        typename KernelPolicy::VertexId         &queue_index,
        //int                                     &num_gpus,
        typename KernelPolicy::VertexId         *&d_in,
        typename KernelPolicy::VertexId         *&d_pred_in,
        typename KernelPolicy::VertexId         *&d_out,
        typename ProblemData::DataSlice         *&problem,
        unsigned char                           *&d_visited_mask,
        typename KernelPolicy::SmemStorage      &smem_storage,
        util::CtaWorkProgress                   &work_progress,
        util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
        typename KernelPolicy::SizeT            &max_out_frontier)
        //texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> *&t_bitmask)
    {
        typedef Cta<KernelPolicy, ProblemData, Functor>     Cta;
        typedef typename KernelPolicy::SizeT                SizeT;

        // Determine our threadblock's work range
        util::CtaWorkLimits<SizeT> work_limits;
        work_decomposition.template GetCtaWorkLimits<
            KernelPolicy::LOG_TILE_ELEMENTS,
            KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

        // Return if we have no work to do
        if (!work_limits.elements) {
            return;
        }

        // CTA processing abstraction
        Cta cta(
            iteration,
            queue_index,
            //num_gpus,
            smem_storage,
            d_in,
            d_pred_in,
            d_out,
            problem,
            d_visited_mask,
            work_progress,
            max_out_frontier);
	    //t_bitmask);

        // Process full tiles
        while (work_limits.offset < work_limits.guarded_offset) {

            cta.ProcessTile(work_limits.offset);
            work_limits.offset += KernelPolicy::TILE_ELEMENTS;
        }

        // Clean up last partial tile with guarded-i/o
        if (work_limits.guarded_elements) {
            cta.ProcessTile(
                work_limits.offset,
                work_limits.guarded_elements);
        }
    }
};

/******************************************************************************
 * Arch dispatch
 ******************************************************************************/

/**
 * Not valid for this arch (default)
 */
template <
    typename    KernelPolicy,
    typename    ProblemData,
    typename    Functor,
    bool        VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT SizeT;
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ void Kernel(
        VertexId                    &iteration,
        bool                        &queue_reset,
        VertexId                    &queue_index,
        //int                         &num_gpus,
        SizeT                       &num_elements,
        //volatile int                *&d_done,
        VertexId                    *&d_in,
        VertexId                    *&d_pred_in,
        VertexId                    *&d_out,
        DataSlice                   *&problem,
        unsigned char               *&d_visited_mask,
        util::CtaWorkProgress       &work_progress,
        SizeT                       &max_in_frontier,
        SizeT                       &max_out_frontier,
        util::KernelRuntimeStats    &kernel_stats)
        //texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> *&ts_bitmask)
    {
        // empty
    }
};


/**
 * @brief Kernel dispatch code for different architectures
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
struct Dispatch<KernelPolicy, ProblemData, Functor, true>
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT SizeT;
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ void Kernel(
        VertexId                    &iteration,
        bool                        &queue_reset,
        VertexId                    &queue_index,
        //int                         &num_gpus,
        SizeT                       &num_elements,
        //volatile int                *&d_done,
        VertexId                    *&d_in,
        VertexId                    *&d_pred_in,
        VertexId                    *&d_out,
        DataSlice                   *&problem,
        unsigned char               *&d_visited_mask,
        util::CtaWorkProgress       &work_progress,
        SizeT                       &max_in_frontier,
        SizeT                       &max_out_frontier,
        util::KernelRuntimeStats    &kernel_stats)
        //texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> *&t_bitmask)
    {

        // Shared storage for the kernel
        __shared__ typename KernelPolicy::SmemStorage smem_storage;

        if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
            kernel_stats.MarkStart();
        }
    
        // workprogress reset
        if (queue_reset)
        {
            if (threadIdx.x < gunrock::util::CtaWorkProgress::COUNTERS) {
                //Reset all counters
                work_progress.template Reset<SizeT>();
            }
        }


        // Determine work decomposition
        if (threadIdx.x == 0) {
            // Obtain problem size
            if (queue_reset)
            {
                work_progress.template StoreQueueLength<SizeT>(num_elements, queue_index);
            }
            else
            {
                num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);

                // Check if we previously overflowed
                if (num_elements >= max_in_frontier) {
                    printf(" num_elements >= max_in_frontier, num_elements = %d, max_in_frontier = %d\n", num_elements, max_in_frontier);
                    num_elements = 0;
                }

                // Signal to host that we're done
                //if ((num_elements == 0) ||
                //        (KernelPolicy::SATURATION_QUIT && (num_elements <= gridDim.x * KernelPolicy::SATURATION_QUIT)))
                //{
                //    if (d_done) d_done[0] = num_elements;
                //}
            }
            
            // Initialize work decomposition in smem
            smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
                    num_elements, gridDim.x);

            // Reset our next outgoing queue counter to zero
            work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

        }

        // Barrier to protect work decomposition
        __syncthreads();

        SweepPass<KernelPolicy, ProblemData, Functor>::Invoke(
                iteration,
                queue_index,
                //num_gpus,
                d_in,
                d_pred_in,
                d_out,
                problem,
                d_visited_mask,
                smem_storage,
                work_progress,
                smem_storage.state.work_decomposition,
                max_out_frontier);
		//t_bitmask);

        if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
            kernel_stats.MarkStop();
            kernel_stats.Flush();
        }
    }
};

/**
 * @brief Filter kernel entry point.
 *
 * @tparam KernelPolicy Kernel policy type for filter.
 * @tparam ProblemData Problem data type for filter.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] iteration     Current graph traversal iteration
 * @param[in] queue_reset   If reset queue counter
 * @param[in] queue_index   Current frontier queue counter index
 * @param[in] num_gpus      Number of GPUs
 * @param[in] num_elements  Number of elements
 * @param[in] d_done        pointer of volatile int to the flag to set when we detect incoming frontier is empty
 * @param[in] d_in_queue    pointer of VertexId to the incoming frontier queue
 * @param[in] d_in_predecessor_queue pointer of VertexId to the incoming predecessor queue (only used when both mark_predecessor and enable_idempotence are set)
 * @param[in] d_out_queue   pointer of VertexId to the outgoing frontier queue
 * @param[in] problem       Device pointer to the problem object
 * @param[in] d_visited_mask Device pointer to the visited mask queue
 * @param[in] work_progress queueing counters to record work progress
 * @param[in] max_in_queue  Maximum number of elements we can place into the incoming frontier
 * @param[in] max_out_queue Maximum number of elements we can place into the outgoing frontier
 * @param[in] kernel_stats  Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT is set)
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void Kernel(
    typename KernelPolicy::VertexId         iteration,
    bool                                    queue_reset,
    typename KernelPolicy::VertexId         queue_index,                
    //int                                     num_gpus,                  
    typename KernelPolicy::SizeT            num_elements,             
    //volatile int                            *d_done,                 
    typename KernelPolicy::VertexId         *d_in_queue,            
    typename KernelPolicy::VertexId         *d_in_predecessor_queue,
    typename KernelPolicy::VertexId         *d_out_queue,          
    typename ProblemData::DataSlice         *problem,
    unsigned char                           *d_visited_mask,
    util::CtaWorkProgress                   work_progress,        
    typename KernelPolicy::SizeT            max_in_queue,        
    typename KernelPolicy::SizeT            max_out_queue,      
    util::KernelRuntimeStats                kernel_stats)
    //texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> *ts_bitmask)
{
    Dispatch<KernelPolicy, ProblemData, Functor>::Kernel(
        iteration,
        queue_reset,
        queue_index,
        //num_gpus,
        num_elements,
        //d_done,
        d_in_queue,
        d_in_predecessor_queue,
        d_out_queue,
        problem,
        d_visited_mask,
        work_progress,
        max_in_queue,
        max_out_queue,
        kernel_stats);
        //ts_bitmask);
}


} // namespace filter
} // namespace oprtr
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
