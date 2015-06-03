// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------


// Add Functor into Kernel Call (done)

/**
 * @file
 * kernel.cuh
 *
 * @brief Forward Edge Map Kernel Entrypoint
 */

#pragma once
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/edge_map_forward/cta.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {
namespace edge_map_forward {

/**
 * @brief Structure for invoking CTA processing tile over all elements.
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
struct Sweep
{
    static __device__ __forceinline__ void Invoke(
        typename KernelPolicy::VertexId         &queue_index,
        //int                                     &num_gpus,
        int                                     &label,
        typename KernelPolicy::VertexId         *&d_in_queue,
        typename KernelPolicy::VertexId         *&d_pred_out,
        typename KernelPolicy::VertexId         *&d_out_queue,
        typename KernelPolicy::SizeT            *&d_row_offsets,
        typename KernelPolicy::VertexId         *&d_column_indices,
        typename KernelPolicy::VertexId         *&d_inverse_column_indices,
        typename ProblemData::DataSlice         *&problem,
        typename KernelPolicy::SmemStorage      &smem_storage,
        util::CtaWorkProgress                   &work_progress,
        util::CtaWorkDistribution<typename KernelPolicy::SizeT> &work_decomposition,
        typename KernelPolicy::SizeT            &max_out_frontier,
        gunrock::oprtr::advance::TYPE           &ADVANCE_TYPE,
        bool                                    &inverse_graph,
        gunrock::oprtr::advance::REDUCE_TYPE    &R_TYPE,
        gunrock::oprtr::advance::REDUCE_OP      &R_OP,
        typename KernelPolicy::Value            *&d_value_to_reduce,
        typename KernelPolicy::Value            *&d_reduce_frontier)
        {
            typedef Cta<KernelPolicy, ProblemData, Functor>     Cta;
            typedef typename KernelPolicy::SizeT                SizeT;

            // Determine threadblock's work range
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
                queue_index,
                label,
                smem_storage,
                d_in_queue,
                d_pred_out,
                d_out_queue,
                d_row_offsets,
                d_column_indices,
                d_inverse_column_indices,
                problem,
                work_progress,
                max_out_frontier,
                ADVANCE_TYPE,
                inverse_graph,
                R_TYPE,
                R_OP,
                d_value_to_reduce,
                d_reduce_frontier);

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

/**
 * Arch dispatch
 */

/**
 * Not valid for this arch (default)
 */
template<
    typename    KernelPolicy,
    typename    ProblemData,
    typename    Functor,
    bool        VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ void Kernel(
        bool                        &queue_reset,
        VertexId                    &queue_index,
        int                         &label,
        SizeT                       &num_elements,
        VertexId                    *&d_in_queue,
        VertexId                    *&d_pred_out,
        VertexId                    *&d_out_queue,
        SizeT                       *&d_row_offsets,
        VertexId                    *&d_column_indices,
        VertexId                    *&d_inverse_column_indices,
        DataSlice                   *&problem,
        util::CtaWorkProgress       &work_progress,
        SizeT                       &max_in_frontier,
        SizeT                       &max_out_frontier,
        util::KernelRuntimeStats    &kernel_stats,
        gunrock::oprtr::advance::TYPE &ADVANCE_TYPE,
        bool                        &inverse_graph,
        gunrock::oprtr::advance::REDUCE_TYPE    &R_TYPE,
        gunrock::oprtr::advance::REDUCE_OP      &R_OP,
        typename KernelPolicy::Value            *&d_value_to_reduce,
        typename KernelPolicy::Value            *&d_reduce_frontier)
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
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename ProblemData::DataSlice DataSlice;

    static __device__ __forceinline__ void Kernel(
        bool                        &queue_reset,
        VertexId                    &queue_index,
        int                         &label,
        SizeT                       &num_elements,
        VertexId                    *&d_in_queue,
        VertexId                    *&d_pred_out,
        VertexId                    *&d_out_queue,
        SizeT                       *&d_row_offsets,
        VertexId                    *&d_column_indices,
        VertexId                    *&d_inverse_column_indices,
        DataSlice                   *&problem,
        util::CtaWorkProgress       &work_progress,
        SizeT                       &max_in_frontier,
        SizeT                       &max_out_frontier,
        util::KernelRuntimeStats    &kernel_stats,
        gunrock::oprtr::advance::TYPE &ADVANCE_TYPE,
        bool                        &inverse_graph,
        gunrock::oprtr::advance::REDUCE_TYPE    &R_TYPE,
        gunrock::oprtr::advance::REDUCE_OP      &R_OP,
        typename KernelPolicy::Value            *&d_value_to_reduce,
        typename KernelPolicy::Value            *&d_reduce_frontier)

    {
        // Shared storage for the kernel
        __shared__ typename KernelPolicy::SmemStorage smem_storage;

        // If instrument flag is set, track kernel stats
        if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
            kernel_stats.MarkStart();
        }

        // Reset work_progress
        if (queue_reset)
        {
            if (threadIdx.x < util::CtaWorkProgress::COUNTERS) {
                //Reset all counters
                work_progress.template Reset<SizeT>();
            }   
        }

        // Determine work decomposition
        if (threadIdx.x == 0) { 
        
            // Obtain problem size
            if (queue_reset)
            {
                work_progress.StoreQueueLength<SizeT>(num_elements, queue_index);
            }
            else
            {
                num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);

                // Check if we previously overflowed
                if (num_elements >= max_in_frontier) {
                    num_elements = 0;
                }

                // Signal to host that we're done
                //if (num_elements == 0) {
                //    if (d_done) d_done[0] = num_elements;
                //}
            }

            // Initialize work decomposition in smem
            smem_storage.state.work_decomposition.template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
                    num_elements, gridDim.x);

            // Reset our next outgoing queue counter to zero
            work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

            work_progress.template PrepResetSteal<SizeT>(queue_index + 1);

        }

        // Barrier to protect work decomposition
        __syncthreads(); 

        Sweep<KernelPolicy, ProblemData, Functor>::Invoke(
                queue_index,
                label,
                d_in_queue,
                d_pred_out,
                d_out_queue,
                d_row_offsets, 
                d_column_indices,
                d_inverse_column_indices,
                problem,
                smem_storage,
                work_progress,
                smem_storage.state.work_decomposition,
                max_out_frontier,
                ADVANCE_TYPE,
                inverse_graph,
                R_TYPE,
                R_OP,
                d_value_to_reduce,
                d_reduce_frontier);

        if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
            kernel_stats.MarkStop();
            kernel_stats.Flush();
        }
    }

};

/**
 * @brief Forward edge map kernel entry point.
 *
 * @tparam KernelPolicy Kernel policy type for forward edge mapping.
 * @tparam ProblemData Problem data type for forward edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] queue_reset       If reset queue counter
 * @param[in] queue_index       Current frontier queue counter index
 * @param[in] label             Distance from source (label) of current frontier
 * @param[in] num_elements      Number of elements
 * @param[in] d_in_queue        Device pointer of VertexId to the incoming frontier queue
 * @param[in] d_pred_out         Device pointer of VertexId to the outgoing predecessor queue (only used when both mark_pred and enable_idempotence are set)
 * @param[in] d_out_queue       Device pointer of VertexId to the outgoing frontier queue
 * @param[in] d_column_indices  Device pointer of VertexId to the column indices queue  
 * @param[in] problem           Device pointer to the problem object
 * @param[in] work_progress     queueing counters to record work progress
 * @param[in] max_in_queue      Maximum number of elements we can place into the incoming frontier
 * @param[in] max_out_queue     Maximum number of elements we can place into the outgoing frontier
 * @param[in] kernel_stats      Per-CTA clock timing statistics (used when KernelPolicy::INSTRUMENT is set)
 */
    template <typename KernelPolicy, typename ProblemData, typename Functor>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
    __global__
void Kernel(
        bool                                    queue_reset,
        typename KernelPolicy::VertexId         queue_index,
        int                                     label,
        typename KernelPolicy::SizeT            num_elements,
        typename KernelPolicy::VertexId         *d_in_queue,
        typename KernelPolicy::VertexId         *d_pred_out,
        typename KernelPolicy::VertexId         *d_out_queue,
        typename KernelPolicy::SizeT            *d_row_offsets,
        typename KernelPolicy::VertexId         *d_column_indices,
        typename KernelPolicy::VertexId         *d_inverse_column_indices,
        typename ProblemData::DataSlice         *problem,
        util::CtaWorkProgress                   work_progress,
        typename KernelPolicy::SizeT            max_in_frontier,
        typename KernelPolicy::SizeT            max_out_frontier,
        util::KernelRuntimeStats                kernel_stats,
        gunrock::oprtr::advance::TYPE           ADVANCE_TYPE = gunrock::oprtr::advance::V2V,
        bool                                    inverse_graph = false,
        gunrock::oprtr::advance::REDUCE_TYPE    R_TYPE = gunrock::oprtr::advance::EMPTY,
        gunrock::oprtr::advance::REDUCE_OP      R_OP = gunrock::oprtr::advance::NONE,
        typename KernelPolicy::Value            *d_value_to_reduce = NULL,
        typename KernelPolicy::Value            *d_reduce_frontier = NULL)
{
    Dispatch<KernelPolicy, ProblemData, Functor>::Kernel(
            queue_reset,    
            queue_index,
            label,
            num_elements,
            d_in_queue,
            d_pred_out,
            d_out_queue,
            d_row_offsets,
            d_column_indices,
            d_inverse_column_indices,
            problem,
            work_progress,
            max_in_frontier,
            max_out_frontier,
            kernel_stats,
            ADVANCE_TYPE,
            inverse_graph,
            R_TYPE,
            R_OP,
            d_value_to_reduce,
            d_reduce_frontier);
}

} //edge_map_forward
} //oprtr
} //gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
