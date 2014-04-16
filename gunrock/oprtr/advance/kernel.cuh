#pragma once
#include <gunrock/util/basic_utils.cuh>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/soa_tuple.cuh>
#include <gunrock/util/srts_grid.cuh>
#include <gunrock/util/srts_soa_details.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/operators.cuh>

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/enactor_base.cuh>

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/edge_map_forward/cta.cuh>
#include <gunrock/oprtr/edge_map_backward/cta.cuh>
#include <gunrock/oprtr/edge_map_partitioned/cta.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>

#include <moderngpu.cuh>

namespace gunrock {
namespace oprtr {
namespace advance {

//TODO: finish LaucnKernel, should load diferent kernels according to their AdvanceMode
//AdvanceType is the argument to send into each kernel call
template <typename KernelPolicy, typename ProblemData, typename Functor>
    void LaunchKernel(EnactorStats              enactor_stats,
            FrontierAttribute                   frontier_attribute,
            ProblemData                         *problem,
            util::CtaWorkProgress               work_progress,
            util::KernelRuntimeStats            kernel_stats,
            CudaContext                         &context,
            KernelPolicy::TYPE                  ADVANCE_TYPE)
{
            
    typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[enactor_stats.gpu_id];
    typename ProblemData::DataSlice *data_slice = problem->data_slices[enactor_stats.gpu_id];

    switch (KernelPolicy::ADVANCE_MODE)
    {
        case THREAD_WARP_CTA_FORWARD:
        {
            // Load Thread Warp CTA Forward Kernel
            gunrock::oprtr::edge_map_forward::Kernel<THREAD_WARP_CTA_FORWARD, ProblemData, Functor>
                <<<enactor_stats.advance_grid_size, THREAD_WARP_CTA_FORWARD::THREADS>>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    enactor_stats.iteration,
                    frontier_attribute.queue_length,
                    enactor_stats.d_done,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],              // d_in_queue
                    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],          // d_pred_out_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],            // d_out_queue
                    graph_slice->d_column_indices,
                    data_slice,
                    work_progress,
                    graph_slice->frontier_elements[selector],                   // max_in_queue
                    graph_slice->frontier_elements[selector^1],                 // max_out_queue
                    enactor_stats.advance_kernel_stats);
            break;
        }
        case THREAD_WARP_CTA_BACKWARD:
        {
            // Load Thread Warp CTA Backward Kernel
            break;
        }
        case LOAD_BALANCED:
        {
            // Load Load Balanced Kernel
            // Get Rowoffsets
            // Use scan to compute edge_offsets for each vertex in the frontier
            // Use sorted sort to compute partition bound for each work-chunk
            // load edge-expand-partitioned kernel
            int num_block = (frontier_attribute.queue_length + KernelPolicy::LOAD_BALANCED::THREADS - 1)/EdgeMapPolicy::THREADS;
            gunrock::oprtr::edge_map_partitioned::GetEdgeCounts<KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
            <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        graph_slice->d_row_offsets,
                                        graph_slice->frontier_queues.d_keys[frontier_attribute.selector],
                                        problem->data_slices[enactor_stats.gpu_id]->d_scanned_edges,
                                        frontier_attribute.queue_length,
                                        graph_slice->frontier_elements[frontier_attribute.selector],
                                        graph_slice->frontier_elements[frontier_attribute.selector^1]);

            Scan<mgpu::MgpuScanTypeInc>((int*)problem->data_slices[enactor_stats.gpu_id]->d_scanned_edges, frontier_attribute.queue_length, (int)0, mgpu::plus<int>(),
            (int*)0, (int*)0, (int*)problem->data_slices[enactor_stats.gpu_id]->d_scanned_edges, context);

            SizeT *temp = new SizeT[1];
            cudaMemcpy(temp,problem->data_slices[enactor_stats.gpu_id]->d_scanned_edges+frontier_attribute.queue_length-1, sizeof(SizeT), cudaMemcpyDeviceToHost);
            SizeT output_queue_len = temp[0];

            //if (output_queue_len < EdgeMapPolicy::LIGHT_EDGE_THRESHOLD)
            {
                gunrock::oprtr::edge_map_partitioned::RelaxLightEdges<KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                        frontier_attribute.queue_reset,
                        frontier_attribute.queue_index,
                        enactor_stats.iteration,
                        graph_slice->d_row_offsets,
                        graph_slice->d_column_indices,
                        problem->data_slices[enactor_stats.gpu_id]->d_scanned_edges,
                        enactor_stats.d_done,
                        graph_slice->frontier_queues.d_keys[frontier_attribute.selector],
                        graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],
                        data_slice,
                        frontier_attribute.queue_length,
                        output_queue_len,
                        graph_slice->frontier_elements[selector],
                        graph_slice->frontier_elements[selector^1],
                        work_progress,
                        enactor_stast.advance_kernel_stats);
            }
            //else
            /*{
                unsigned int split_val = (output_queue_len + KernelPolicy::LOAD_BALANCED::BLOCKS - 1) / KernelPolicy::LOAD_BALANCED::BLOCKS;
                util::MemsetIdxKernel<<<128, 128>>>(enactor_stats.d_node_locks, KernelPolicy::LOAD_BALANCED::BLOCKS, split_val);
                SortedSearch<MgpuBoundsLower>(
                enactor_stats.d_node_locks,
                KernelPolicy::LOAD_BALANCED::BLOCKS,
                problem->data_slices[enactor_stats.gpu_id]->d_scanned_edges,
                frontier_attribute.queue_length,
                enactor_stats.d_node_locks_out,
                context);

                gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges<KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
                <<< KernelPolicy::LOAD_BALANCED::BLOCKS, KernelPolicy::LOAD_BALANCED::THREADS >>>(
                                        frontier_attribute.queue_reset,
                                        frontier_attribute.queue_index,
                                        enactor_stats.iteration,
                                        graph_slice->d_row_offsets,
                                        graph_slice->d_column_indices,
                                        problem->data_slices[enactor_stats.gpu_id]->d_scanned_edges,
                                        enactor_stats.d_node_locks_out,
                                        KernelPolicy::LOAD_BALANCED::BLOCKS,
                                        enactor_stats.d_done,
                                        graph_slice->frontier_queues.d_keys[frontier_attribute.selector],
                                        graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],
                                        data_slice,
                                        frontier_attribute.queue_length,
                                        output_queue_len,
                                        split_val,
                                        graph_slice->frontier_elements[frontier_attribute.selector],
                                        graph_slice->frontier_elements[frontier_attribute.selector^1],
                                        work_progress,
                                        enactor_stats.advance_kernel_stats);
            }*/
            break;
        }
    }
}

} //advance
} //oprtr
} //gunrock/
