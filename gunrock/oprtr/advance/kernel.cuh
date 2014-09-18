#pragma once
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/soa_tuple.cuh>
#include <gunrock/util/srts_grid.cuh>
#include <gunrock/util/srts_soa_details.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/operators.cuh>

#include <gunrock/util/test_utils.cuh>

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/enactor_base.cuh>

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_backward/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned_backward/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned/kernel.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>

#include <moderngpu.cuh>

namespace gunrock {
namespace oprtr {
namespace advance {

/**
 * @brief Advance operator kernel entry point.
 *
 * @tparam KernelPolicy Kernel policy type for advance operator.
 * @tparam ProblemData Problem data type for advance operator.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_done                    Pointer of volatile int to the flag to set when we detect incoming frontier is empty
 * @param[in] enactor_stats             EnactorStats object to store enactor related variables and stast
 * @param[in] frontier_attribute        FrontierAttribute object to store frontier attribute while doing the advance operation
 * @param[in] data_slice                Device pointer to the problem object's data_slice member
 * @param[in] backward_index_queue      If backward mode is activated, this is used to store the vertex index. (deprecated)
 * @param[in] backward_frontier_map_in  If backward mode is activated, this is used to store input frontier bitmap
 * @param[in] backward_frontier_map_out If backward mode is activated, this is used to store output frontier bitmap
 * @param[in] partitioned_scanned_edges If load balanced mode is activated, this is used to store the scanned edge number for neighbor lists in current frontier
 * @param[in] d_in_key_queue            Device pointer of input key array to the incoming frontier queue
 * @param[in] d_out_key_queue           Device pointer of output key array to the outgoing frontier queue
 * @param[in] d_in_value_queue          Device pointer of input value array to the incoming frontier queue
 * @param[in] d_out_value_queue         Device pointer of output value array to the outgoing frontier queue
 * @param[in] d_row_offsets             Device pointer of SizeT to the row offsets queue  
 * @param[in] d_column_indices          Device pointer of VertexId to the column indices queue
 * @param[in] d_column_offsets          Device pointer of SizeT to the row offsets queue for inverse graph
 * @param[in] d_row_indices             Device pointer of VertexId to the column indices queue for inverse graph
 * @param[in] max_in_queue              Maximum number of elements we can place into the incoming frontier
 * @param[in] max_out_queue             Maximum number of elements we can place into the outgoing frontier
 * @param[in] work_progress             queueing counters to record work progress
 * @param[in] context                   CudaContext pointer for moderngpu APIs
 * @param[in] ADVANCE_TYPE              enumerator of advance type: V2V, V2E, E2V, or E2E
 * @param[in] inverse_graph             whether this iteration of advance operation is in the opposite direction to the previous iteration (false by default)
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
    void LaunchKernel(
            volatile int                            *d_done,
            gunrock::app::EnactorStats              &enactor_stats,
            gunrock::app::FrontierAttribute         &frontier_attribute,
            typename ProblemData::DataSlice         *data_slice,
            typename ProblemData::VertexId          *backward_index_queue,
            bool                                    *backward_frontier_map_in,
            bool                                    *backward_frontier_map_out,
            unsigned int                            *partitioned_scanned_edges,
            typename KernelPolicy::VertexId         *d_in_key_queue,
            typename KernelPolicy::VertexId         *d_out_key_queue,
            typename KernelPolicy::VertexId         *d_in_value_queue,
            typename KernelPolicy::VertexId         *d_out_value_queue,
            typename KernelPolicy::SizeT            *d_row_offsets,
            typename KernelPolicy::VertexId         *d_column_indices,
            typename KernelPolicy::SizeT            *d_column_offsets,
            typename KernelPolicy::VertexId         *d_row_indices,
            typename KernelPolicy::SizeT            max_in,
            typename KernelPolicy::SizeT            max_out,
            util::CtaWorkProgress                   work_progress,
            CudaContext                             &context,
            TYPE                                    ADVANCE_TYPE,
            bool                                    inverse_graph = false)
{
    switch (KernelPolicy::ADVANCE_MODE)
    {
        case TWC_FORWARD:
        {
            // Load Thread Warp CTA Forward Kernel
            gunrock::oprtr::edge_map_forward::Kernel<typename KernelPolicy::THREAD_WARP_CTA_FORWARD, ProblemData, Functor>
                <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_FORWARD::THREADS>>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    enactor_stats.iteration,
                    frontier_attribute.queue_length,
                    d_done,
                    d_in_key_queue,              // d_in_queue
                    d_out_value_queue,          // d_pred_out_queue
                    d_out_key_queue,            // d_out_queue
                    d_column_indices,
                    d_row_indices,
                    data_slice,
                    work_progress,
                    max_in,                   // max_in_queue
                    max_out,                 // max_out_queue
                    enactor_stats.advance_kernel_stats,
                    ADVANCE_TYPE,
                    inverse_graph);
            break;
        }
        case LB_BACKWARD:
        {
            // Load Thread Warp CTA Backward Kernel
            typedef typename ProblemData::SizeT         SizeT;
            typedef typename ProblemData::VertexId      VertexId;
            typedef typename KernelPolicy::LOAD_BALANCED LBPOLICY;
            // Load Load Balanced Kernel
            // Get Rowoffsets
            // Use scan to compute edge_offsets for each vertex in the frontier
            // Use sorted sort to compute partition bound for each work-chunk
            // load edge-expand-partitioned kernel
            //util::DisplayDeviceResults(d_in_key_queue, frontier_attribute.queue_length);
            int num_block = (frontier_attribute.queue_length + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
            gunrock::oprtr::edge_map_partitioned_backward::GetEdgeCounts<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
            <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        d_column_offsets,
                                        d_row_indices,
                                        d_in_key_queue,
                                        partitioned_scanned_edges,
                                        frontier_attribute.queue_length,
                                        max_in,
                                        max_out,
                                        ADVANCE_TYPE);

            Scan<mgpu::MgpuScanTypeInc>((int*)partitioned_scanned_edges, frontier_attribute.queue_length, (int)0, mgpu::plus<int>(),
            (int*)0, (int*)0, (int*)partitioned_scanned_edges, context);

            SizeT *temp = new SizeT[1];
            cudaMemcpy(temp,partitioned_scanned_edges+frontier_attribute.queue_length-1, sizeof(SizeT), cudaMemcpyDeviceToHost);
            SizeT output_queue_len = temp[0];
            //printf("input queue:%d, output_queue:%d\n", frontier_attribute.queue_length, output_queue_len);

            if (frontier_attribute.selector == 1) {
                // Edge Map
                gunrock::oprtr::edge_map_partitioned_backward::RelaxLightEdges<LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                        frontier_attribute.queue_reset,
                        frontier_attribute.queue_index,
                        enactor_stats.iteration,
                        d_column_offsets,
                        d_row_indices,
                        (VertexId*)NULL,
                        partitioned_scanned_edges,
                        d_done,
                        d_in_key_queue,
                        backward_frontier_map_in,
                        backward_frontier_map_out,
                        data_slice,
                        frontier_attribute.queue_length,
                        output_queue_len,
                        max_in,
                        max_out,
                        work_progress,
                        enactor_stats.advance_kernel_stats,
                        ADVANCE_TYPE,
                        inverse_graph);
            } else {
                // Edge Map
                gunrock::oprtr::edge_map_partitioned_backward::RelaxLightEdges<LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                        frontier_attribute.queue_reset,
                        frontier_attribute.queue_index,
                        enactor_stats.iteration,
                        d_column_offsets,
                        d_row_indices,
                        (VertexId*)NULL,
                        partitioned_scanned_edges,
                        d_done,
                        d_in_key_queue,
                        backward_frontier_map_out,
                        backward_frontier_map_in,
                        data_slice,
                        frontier_attribute.queue_length,
                        output_queue_len,
                        max_in,
                        max_out,
                        work_progress,
                        enactor_stats.advance_kernel_stats,
                        ADVANCE_TYPE,
                        inverse_graph);
            }
            break;
        }
        case TWC_BACKWARD:
        {
            // Load Thread Warp CTA Backward Kernel
            if (frontier_attribute.selector == 1) {
                // Edge Map
                gunrock::oprtr::edge_map_backward::Kernel<typename KernelPolicy::THREAD_WARP_CTA_BACKWARD, ProblemData, Functor>
                    <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_BACKWARD::THREADS>>>(
                            frontier_attribute.queue_reset,
                            frontier_attribute.queue_index,
                            enactor_stats.num_gpus,
                            frontier_attribute.queue_length,
                            d_done,
                            d_in_key_queue,              // d_in_queue
                            backward_index_queue,            // d_in_index_queue
                            backward_frontier_map_in,
                            backward_frontier_map_out,
                            d_column_offsets,
                            d_row_indices,
                            data_slice,
                            work_progress,
                            enactor_stats.advance_kernel_stats,
                            ADVANCE_TYPE);
            } else {
                // Edge Map
                gunrock::oprtr::edge_map_backward::Kernel<typename KernelPolicy::THREAD_WARP_CTA_BACKWARD, ProblemData, Functor>
                    <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_BACKWARD::THREADS>>>(
                            frontier_attribute.queue_reset,
                            frontier_attribute.queue_index,
                            enactor_stats.num_gpus,
                            frontier_attribute.queue_length,
                            d_done,
                            d_in_key_queue,              // d_in_queue
                            backward_index_queue,            // d_in_index_queue
                            backward_frontier_map_out,
                            backward_frontier_map_in,
                            d_column_offsets,
                            d_row_indices,
                            data_slice,
                            work_progress,
                            enactor_stats.advance_kernel_stats,
                            ADVANCE_TYPE);
            }
            break;   
        }
        case LB:
        {
            typedef typename ProblemData::SizeT         SizeT;
            typedef typename ProblemData::VertexId      VertexId;
            typedef typename KernelPolicy::LOAD_BALANCED LBPOLICY;
            // Load Load Balanced Kernel
            // Get Rowoffsets
            // Use scan to compute edge_offsets for each vertex in the frontier
            // Use sorted sort to compute partition bound for each work-chunk
            // load edge-expand-partitioned kernel
            int num_block = (frontier_attribute.queue_length + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
            gunrock::oprtr::edge_map_partitioned::GetEdgeCounts<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
            <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        d_row_offsets,
                                        d_column_indices,
                                        d_in_key_queue,
                                        &partitioned_scanned_edges[1],
                                        frontier_attribute.queue_length,
                                        max_in,
                                        max_out,
                                        ADVANCE_TYPE);

            Scan<mgpu::MgpuScanTypeInc>((int*)&partitioned_scanned_edges[1], frontier_attribute.queue_length, (int)0, mgpu::plus<int>(),
            (int*)0, (int*)0, (int*)&partitioned_scanned_edges[1], context);

            SizeT *temp = new SizeT[1];
            cudaMemcpy(temp,partitioned_scanned_edges+frontier_attribute.queue_length, sizeof(SizeT), cudaMemcpyDeviceToHost);
            SizeT output_queue_len = temp[0];
            //printf("input_queue_len:%d\n", frontier_attribute.queue_length);
            //printf("output_queue_len:%d\n", output_queue_len);

            /*if (output_queue_len < LBPOLICY::LIGHT_EDGE_THRESHOLD)
            {
                gunrock::oprtr::edge_map_partitioned::RelaxLightEdges<LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                        frontier_attribute.queue_reset,
                        frontier_attribute.queue_index,
                        enactor_stats.iteration,
                        d_row_offsets,
                        d_column_indices,
                        d_row_indices,
                        &partitioned_scanned_edges[1],
                        d_done,
                        d_in_key_queue,
                        d_out_key_queue,
                        data_slice,
                        frontier_attribute.queue_length,
                        output_queue_len,
                        max_in,
                        max_out,
                        work_progress,
                        enactor_stats.advance_kernel_stats,
                        ADVANCE_TYPE,
                        inverse_graph);
            }
            else*/
            {
                /*unsigned int split_val = (output_queue_len + KernelPolicy::LOAD_BALANCED::BLOCKS - 1) / KernelPolicy::LOAD_BALANCED::BLOCKS;
                int num_block = (output_queue_len >= 256) ? KernelPolicy::LOAD_BALANCED::BLOCKS : 1;
                int nb = (num_block + 1 + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
                gunrock::oprtr::edge_map_partitioned::MarkPartitionSizes<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
                    <<<nb, KernelPolicy::LOAD_BALANCED::THREADS>>>(
                            enactor_stats.d_node_locks,
                            split_val,
                            num_block+1,
                            output_queue_len);
                //util::MemsetIdxKernel<<<128, 128>>>(enactor_stats.d_node_locks, KernelPolicy::LOAD_BALANCED::BLOCKS, split_val);

                SortedSearch<MgpuBoundsLower>(
                        enactor_stats.d_node_locks,
                        KernelPolicy::LOAD_BALANCED::BLOCKS,
                        partitioned_scanned_edges,
                        frontier_attribute.queue_length,
                        enactor_stats.d_node_locks_out,
                        context);*/
                LoadBalanceSearch(output_queue_len, partitioned_scanned_edges, frontier_attribute.queue_length, d_out_key_queue, context);

                //util::DisplayDeviceResults(enactor_stats.d_node_locks_out, KernelPolicy::LOAD_BALANCED::BLOCKS);
                int num_block = (output_queue_len + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;

                gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges<typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        frontier_attribute.queue_reset,
                                        frontier_attribute.queue_index,
                                        enactor_stats.iteration,
                                        d_row_offsets,
                                        d_column_indices,
                                        d_row_indices,
                                        partitioned_scanned_edges,
                                        enactor_stats.d_node_locks_out,
                                        KernelPolicy::LOAD_BALANCED::BLOCKS,
                                        d_done,
                                        d_in_key_queue,
                                        d_out_key_queue,
                                        data_slice,
                                        frontier_attribute.queue_length,
                                        output_queue_len,
                                        max_in,
                                        max_out,
                                        work_progress,
                                        enactor_stats.advance_kernel_stats,
                                        ADVANCE_TYPE,
                                        inverse_graph);
            }
            break;
        }
    }
}


} //advance
} //oprtr
} //gunrock/
