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
#include <gunrock/util/multithread_utils.cuh>

#include <moderngpu.cuh>

namespace gunrock {
namespace oprtr {
namespace advance {

/*
 * @brief Compute output frontier queue length.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] frontier_attribute Pointer to the frontier attribute.
 * @param[in] d_offsets Pointer to the offsets.
 * @param[in] d_indices Pointer to the indices.
 * @param[in] d_in_key_queue Pointer to the input mapping queue.
 * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
 * @param[in] max_in Maximum input queue size.
 * @param[in] max_out Maximum output queue size.
 * @param[in] context CudaContext for ModernGPU API.
 * @param[in] stream CUDA stream.
 * @param[in] ADVANCE_TYPE Advance kernel mode.
 * @param[in] express Whether or not enable express mode.
 * @param[in] in_inv Input inverse.
 * @param[in] out_inv Output inverse.
 *
 * \return cudaError_t object Indicates the success of all CUDA calls.
 */
template <typename KernelPolicy, typename Problem, typename Functor>
cudaError_t ComputeOutputLength(
    // int                             num_block,
    gunrock::app::FrontierAttribute<typename Problem::SizeT>
                               *frontier_attribute,
    typename Problem::SizeT    *d_offsets,
    typename Problem::VertexId *d_indices,
    typename Problem::SizeT    *d_inv_offsets,
    typename Problem::VertexId *d_inv_indices,
    typename Problem::VertexId *d_in_key_queue,
    typename Problem::SizeT    *partitioned_scanned_edges,
    typename Problem::SizeT     max_in,
    typename Problem::SizeT     max_out,
    CudaContext                &context,
    cudaStream_t                stream,
    TYPE                        ADVANCE_TYPE,
    bool                        express = false,
    bool                        in_inv = false,
    bool                        out_inv = false)
{
    // Load Load Balanced Kernel
    // Get Rowoffsets
    // Use scan to compute edge_offsets for each vertex in the frontier
    // Use sorted sort to compute partition bound for each work-chunk
    // load edge-expand-partitioned kernel
    //util::DisplayDeviceResults(d_in_key_queue, frontier_attribute.queue_length);
    typedef typename Problem::SizeT         SizeT;
    if (frontier_attribute->queue_length == 0)
    {
        //printf("setting output_length to 0");
        util::MemsetKernel<SizeT><<<1,1,0,stream>>>(frontier_attribute->output_length.GetPointer(util::DEVICE),0,1);
        return cudaSuccess;
    }

    SizeT num_block = (frontier_attribute->queue_length 
        + KernelPolicy::LOAD_BALANCED::THREADS - 1)
        /KernelPolicy::LOAD_BALANCED::THREADS;
    //if (num_block > 256) num_block = 256;
    if (KernelPolicy::ADVANCE_MODE == LB_BACKWARD || 
        KernelPolicy::ADVANCE_MODE == TWC_BACKWARD)
    {
        gunrock::oprtr::edge_map_partitioned_backward::GetEdgeCounts
            <typename KernelPolicy::LOAD_BALANCED, Problem, Functor>
            <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS, 0, stream>>>(
                d_inv_offsets,
                d_inv_indices,
                d_in_key_queue,
                partitioned_scanned_edges,
                frontier_attribute->queue_length, // TODO: +1?
                max_in,
                max_out,
                ADVANCE_TYPE);
        //util::DisplayDeviceResults(partitioned_scanned_edges, frontier_attribute->queue_length);
    }
    else if (KernelPolicy::ADVANCE_MODE == LB ||
             KernelPolicy::ADVANCE_MODE == LB_LIGHT ||
             KernelPolicy::ADVANCE_MODE == TWC_FORWARD)
    {
        gunrock::oprtr::edge_map_partitioned::GetEdgeCounts
            <typename KernelPolicy::LOAD_BALANCED, Problem, Functor>
            <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS, 0, stream>>>(
                d_offsets,
                d_indices,
                d_inv_offsets,
                d_inv_indices,
                d_in_key_queue,
                partitioned_scanned_edges,
                frontier_attribute->queue_length, // TODO: +1?
                max_in,
                max_out,
                ADVANCE_TYPE,
                in_inv,
                out_inv);
        //util::DisplayDeviceResults(partitioned_scanned_edges, frontier_attribute->queue_length);
    }

    //if (KernelPolicy::ADVANCE_MODE == LB ||
    //    KernelPolicy::ADVANCE_MODE == LB_LIGHT ||
    //    KernelPolicy::ADVANCE_MODE == LB_BACKWARD)
    //{
        Scan<mgpu::MgpuScanTypeInc>(
            (SizeT*)partitioned_scanned_edges,
            frontier_attribute->queue_length, // TODO: +1?
            (SizeT)0,
            mgpu::plus<SizeT>(),
            (SizeT*)0,
            (SizeT*)0,
            (SizeT*)partitioned_scanned_edges,
            context);

        return util::GRError(cudaMemcpyAsync(
            frontier_attribute->output_length.GetPointer(util::DEVICE),
            partitioned_scanned_edges + frontier_attribute->queue_length - 1, // TODO: +1?
            sizeof(SizeT), cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsync failed", __FILE__, __LINE__);
    //} else {
    //    util::MemsetKernel<<<1,1,0,stream>>>(
    //        frontier_attribute->output_length.GetPointer(util::DEVICE),
    //        0, 1);
    //}
    //return cudaSuccess;
}

/**
 * @brief Advance operator kernel entry point.
 *
 * @tparam KernelPolicy Kernel policy type for advance operator.
 * @tparam ProblemData Problem data type for advance operator.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam Op Operation for gather reduce. mgpu::plus<int> by default.
 *
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
 * @param[in] REDUCE_OP                 enumerator of available reduce operations: plus, multiplies, bit_or, bit_and, bit_xor, maximum, minimum. none by default.
 * @param[in] REDUCE_TYPE               enumerator of available reduce types: EMPTY(do not do reduce) VERTEX(extract value from |V| array) EDGE(extract value from |E| array)
 * @param[in] d_value_to_reduce         array to store values to reduce
 * @param[out] d_reduce_frontier        neighbor list values for nodes in the output frontier
 * @param[out] d_reduced_value          array to store reduced values
 */

//TODO: Reduce by neighbor list now only supports LB advance mode.
//TODO: Add a switch to enable advance+filter (like in BFS), pissibly moving idempotent ops from filter to advance?

template <typename KernelPolicy, typename ProblemData, typename Functor>
    void LaunchKernel(
        gunrock::app::EnactorStats              &enactor_stats,
        gunrock::app::FrontierAttribute<typename KernelPolicy::SizeT>
                                                &frontier_attribute,
        typename ProblemData::DataSlice         *data_slice,
        typename ProblemData::VertexId          *backward_index_queue,
        bool                                    *backward_frontier_map_in,
        bool                                    *backward_frontier_map_out,
        typename KernelPolicy::SizeT            *partitioned_scanned_edges,
        typename KernelPolicy::VertexId         *d_in_key_queue,
        typename KernelPolicy::VertexId         *d_out_key_queue,
        typename KernelPolicy::VertexId         *d_in_value_queue,
        typename KernelPolicy::VertexId         *d_out_value_queue,
        typename KernelPolicy::SizeT            *d_row_offsets,
        typename KernelPolicy::VertexId         *d_column_indices,
        typename KernelPolicy::SizeT            *d_column_offsets,
        typename KernelPolicy::VertexId         *d_row_indices,
        typename KernelPolicy::SizeT             max_in,
        typename KernelPolicy::SizeT             max_out,
        util::CtaWorkProgress                    work_progress,
        CudaContext                             &context,
        cudaStream_t                             stream,
        TYPE                                     ADVANCE_TYPE,
        bool                                     input_inverse_graph  = false,
        bool                                     output_inverse_graph = false,
        bool                                     get_output_length = true,
        REDUCE_OP                                R_OP              = gunrock::oprtr::advance::NONE,
        REDUCE_TYPE                              R_TYPE            = gunrock::oprtr::advance::EMPTY,
        typename KernelPolicy::Value            *d_value_to_reduce = NULL,
        typename KernelPolicy::Value            *d_reduce_frontier = NULL,
        typename KernelPolicy::Value            *d_reduced_value   = NULL)
{
    if (frontier_attribute.queue_length == 0) return;

    switch (KernelPolicy::ADVANCE_MODE)
    {
        case TWC_FORWARD:
        {
            // Load Thread Warp CTA Forward Kernel
            gunrock::oprtr::edge_map_forward::Kernel
                <typename KernelPolicy::THREAD_WARP_CTA_FORWARD, ProblemData, Functor>
                <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_FORWARD::THREADS, 0, stream>>>(
                frontier_attribute.queue_reset,
                frontier_attribute.queue_index,
                enactor_stats.iteration,
                frontier_attribute.queue_length,
                d_in_key_queue,              // d_in_queue
                d_out_value_queue,          // d_pred_out_queue
                d_out_key_queue,            // d_out_queue
                d_row_offsets,
                d_column_indices,
                d_row_indices,
                data_slice,
                work_progress,
                max_in,                   // max_in_queue
                max_out,                 // max_out_queue
                enactor_stats.advance_kernel_stats,
                ADVANCE_TYPE,
                input_inverse_graph,
                R_TYPE,
                R_OP,
                d_value_to_reduce,
                d_reduce_frontier);

            // Do segreduction using d_scanned_edges and d_reduce_frontier
            typedef typename ProblemData::SizeT         SizeT;
            typedef typename ProblemData::VertexId      VertexId;
            typedef typename ProblemData::Value         Value;
            //TODO: For TWC_Forward, Find a way to get the output_queue_len,
            //also, try to get the scanned_edges array too. Then the following code will work.
            /*if (R_TYPE != gunrock::oprtr::advance::EMPTY && d_value_to_reduce && d_reduce_frontier) {
              switch (R_OP) {
                case gunrock::oprtr::advance::PLUS: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)0, mgpu::plus<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MULTIPLIES: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)1, mgpu::multiplies<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MAXIMUM: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)INT_MIN, mgpu::maximum<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MINIMUM: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)INT_MAX, mgpu::minimum<typename KernelPolicy::Value>(), context);
                      break;
                }
                default:
                    //default operator is plus
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)0, mgpu::plus<typename KernelPolicy::Value>(), context);
                      break;
              }
            }*/
            break;
        }
        case LB_BACKWARD:
        {
            // Load Thread Warp CTA Backward Kernel
            typedef typename ProblemData::SizeT         SizeT;
            typedef typename ProblemData::VertexId      VertexId;
            typedef typename KernelPolicy::LOAD_BALANCED LBPOLICY;
            SizeT num_block = (frontier_attribute.queue_length + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
            if (get_output_length)
            {
                ComputeOutputLength<KernelPolicy, ProblemData, Functor>(
                    &frontier_attribute,
                    d_row_offsets,
                    d_column_indices,
                    d_column_offsets,
                    d_row_indices,
                    d_in_key_queue,
                    partitioned_scanned_edges,  // TODO: +1?
                    max_in,
                    max_out,
                    context,
                    stream,
                    ADVANCE_TYPE,
                    false,
                    input_inverse_graph,
                    output_inverse_graph);
            }

            // Edge Map
            gunrock::oprtr::edge_map_partitioned_backward::RelaxLightEdges<LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS, 0, stream >>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.iteration,
                    d_column_offsets,
                    d_row_indices,
                    (VertexId*)NULL,
                    partitioned_scanned_edges,  // TODO: +1?
                    d_in_key_queue,
                    frontier_attribute.selector == 1 ? backward_frontier_map_in  : backward_frontier_map_out,
                    frontier_attribute.selector == 1 ? backward_frontier_map_out : backward_frontier_map_in ,
                    data_slice,
                    frontier_attribute.queue_length,
                    frontier_attribute.output_length.GetPointer(util::DEVICE),
                    max_in,
                    max_out,
                    work_progress,
                    enactor_stats.advance_kernel_stats,
                    ADVANCE_TYPE,
                    input_inverse_graph);
            break;
        }
        case TWC_BACKWARD:
        {
            // Load Thread Warp CTA Backward Kernel
            // Edge Map
            gunrock::oprtr::edge_map_backward::Kernel
                <typename KernelPolicy::THREAD_WARP_CTA_BACKWARD, ProblemData, Functor>
                <<<enactor_stats.advance_grid_size, KernelPolicy::THREAD_WARP_CTA_BACKWARD::THREADS>>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    //enactor_stats.num_gpus,
                    frontier_attribute.queue_length,
                    //d_done,
                    d_in_key_queue,              // d_in_queue
                    backward_index_queue,            // d_in_index_queue
                    frontier_attribute.selector == 1 ? backward_frontier_map_in  : backward_frontier_map_out,
                    frontier_attribute.selector == 1 ? backward_frontier_map_out : backward_frontier_map_in ,
                    d_column_offsets,
                    d_row_indices,
                    data_slice,
                    work_progress,
                    enactor_stats.advance_kernel_stats,
                    ADVANCE_TYPE);
            break;
        }
        case LB:
        {
            typedef typename ProblemData::SizeT         SizeT;
            typedef typename ProblemData::VertexId      VertexId;
            typedef typename ProblemData::Value         Value;
            typedef typename KernelPolicy::LOAD_BALANCED LBPOLICY;
            // load edge-expand-partitioned kernel
            SizeT num_block = (frontier_attribute.queue_length + KernelPolicy::LOAD_BALANCED::THREADS - 1)/KernelPolicy::LOAD_BALANCED::THREADS;
            if (get_output_length)
            {
                ComputeOutputLength<KernelPolicy, ProblemData, Functor>(
                    &frontier_attribute,
                    d_row_offsets,
                    d_column_indices,
                    d_column_offsets,
                    d_row_indices,
                    d_in_key_queue,
                    partitioned_scanned_edges,
                    max_in,
                    max_out,
                    context,
                    stream,
                    ADVANCE_TYPE,
                    false,
                    input_inverse_graph,
                    output_inverse_graph);
            }
            if (!get_output_length || (get_output_length && 
                frontier_attribute.output_length[0] < LBPOLICY::LIGHT_EDGE_THRESHOLD))
            {
                gunrock::oprtr::edge_map_partitioned::RelaxLightEdges
                <LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS, 0, stream>>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.iteration,
                    d_row_offsets,
                    d_column_offsets,
                    d_column_indices,
                    d_row_indices,
                    partitioned_scanned_edges, // TODO: +1?
                    d_in_key_queue,
                    d_out_key_queue,
                    data_slice,
                    frontier_attribute.queue_length,
                    frontier_attribute.output_length.GetPointer(util::DEVICE),
                    max_in,
                    max_out,
                    work_progress,
                    enactor_stats.advance_kernel_stats,
                    ADVANCE_TYPE,
                    input_inverse_graph,
                    output_inverse_graph,
                    R_TYPE,
                    R_OP,
                    d_value_to_reduce,
                    d_reduce_frontier);
            }
            else if (/*get_output_length &&*/ frontier_attribute.output_length[0] >= LBPOLICY::LIGHT_EDGE_THRESHOLD)
            {
                unsigned int split_val = (frontier_attribute.output_length[0] + 
                    KernelPolicy::LOAD_BALANCED::BLOCKS - 1) / KernelPolicy::LOAD_BALANCED::BLOCKS;
                util::MemsetIdxKernel<unsigned int, int> <<<128, 128, 0, stream>>>(
                    enactor_stats.node_locks.GetPointer(util::DEVICE),
                    (int)KernelPolicy::LOAD_BALANCED::BLOCKS, 
                    split_val);
                SortedSearch<MgpuBoundsLower>(
                    enactor_stats.node_locks.GetPointer(util::DEVICE),
                    KernelPolicy::LOAD_BALANCED::BLOCKS,
                    partitioned_scanned_edges,
                    frontier_attribute.queue_length,
                    enactor_stats.node_locks_out.GetPointer(util::DEVICE),
                    context);

                gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges2
                    <typename KernelPolicy::LOAD_BALANCED, ProblemData, Functor>
                    <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS, 0, stream>>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.iteration,
                    d_row_offsets,
                    d_column_offsets,
                    d_column_indices,
                    d_row_indices,
                    partitioned_scanned_edges,
                    enactor_stats.node_locks_out.GetPointer(util::DEVICE),
                    KernelPolicy::LOAD_BALANCED::BLOCKS,
                    //d_done,
                    d_in_key_queue,
                    d_out_key_queue,
                    data_slice,
                    frontier_attribute.queue_length,
                    frontier_attribute.output_length.GetPointer(util::DEVICE),
                    split_val,
                    max_in,
                    max_out,
                    work_progress,
                    enactor_stats.advance_kernel_stats,
                    ADVANCE_TYPE,
                    input_inverse_graph,
                    output_inverse_graph,
                    R_TYPE,
                    R_OP,
                    d_value_to_reduce,
                    d_reduce_frontier);
                //util::DisplayDeviceResults(d_out_key_queue, output_queue_len);
            }
            // TODO: switch REDUCE_OP for different reduce operators
            // Do segreduction using d_scanned_edges and d_reduce_frontier
            /*if (R_TYPE != gunrock::oprtr::advance::EMPTY && d_value_to_reduce && d_reduce_frontier) {
              switch (R_OP) {
                case gunrock::oprtr::advance::PLUS: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)0, mgpu::plus<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MULTIPLIES: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)1, mgpu::multiplies<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MAXIMUM: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)INT_MIN, mgpu::maximum<typename KernelPolicy::Value>(), context);
                      break;
                }
                case gunrock::oprtr::advance::MINIMUM: {
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)INT_MAX, mgpu::minimum<typename KernelPolicy::Value>(), context);
                      break;
                }
                default:
                    //default operator is plus
                    SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges, output_queue_len,frontier_attribute.queue_length,
                      false, d_reduced_value, (Value)0, mgpu::plus<typename KernelPolicy::Value>(), context);
                      break;
              }
            }*/
            break;
        }
        case LB_LIGHT:
        {
            typedef typename ProblemData::SizeT          SizeT;
            typedef typename ProblemData::VertexId       VertexId;
            typedef typename ProblemData::Value          Value;
            typedef typename KernelPolicy::LOAD_BALANCED LBPOLICY;
            // load edge-expand-partitioned kernel
            SizeT num_block = (frontier_attribute.queue_length +
                               KernelPolicy::LOAD_BALANCED::THREADS - 1) /
                               KernelPolicy::LOAD_BALANCED::THREADS;

            if (get_output_length)
            {
                ComputeOutputLength<KernelPolicy, ProblemData, Functor>(
                    &frontier_attribute,
                    d_row_offsets,
                    d_column_indices,
                    d_column_offsets,
                    d_row_indices,
                    d_in_key_queue,
                    partitioned_scanned_edges,
                    max_in,
                    max_out,
                    context,
                    stream,
                    ADVANCE_TYPE,
                    false,
                    input_inverse_graph,
                    output_inverse_graph);
            }

            gunrock::oprtr::edge_map_partitioned::RelaxLightEdges
                <LBPOLICY, ProblemData, Functor>
                <<< num_block, KernelPolicy::LOAD_BALANCED::THREADS, 0, stream>>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.iteration,
                    d_row_offsets,
                    d_column_offsets,
                    d_column_indices,
                    d_row_indices,
                    partitioned_scanned_edges, // TODO: +1?
                    d_in_key_queue,
                    d_out_key_queue,
                    data_slice,
                    frontier_attribute.queue_length,
                    frontier_attribute.output_length.GetPointer(util::DEVICE),
                    max_in,
                    max_out,
                    work_progress,
                    enactor_stats.advance_kernel_stats,
                    ADVANCE_TYPE,
                    input_inverse_graph,
                    output_inverse_graph,
                    R_TYPE,
                    R_OP,
                    d_value_to_reduce,
                    d_reduce_frontier);
            break;
        }
    }
}


} //advance
} //oprtr
} //gunrock/
