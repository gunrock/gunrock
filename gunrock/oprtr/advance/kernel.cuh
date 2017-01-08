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

#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_backward/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned_backward/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned/kernel.cuh>
#include <gunrock/oprtr/edge_map_partitioned_cull/kernel.cuh>
#include <gunrock/oprtr/all_edges_advance/kernel.cuh>

#include <gunrock/util/multithread_utils.cuh>

#include <moderngpu.cuh>

namespace gunrock {
namespace oprtr {
namespace advance {

// Steps for adding reduce:
// fix block_dim=512/1024 and items_per_thread=7 or 11, make block number a variable
// creating flag, value
// flag comes from whether two threads' smem_storage.iter_input_start + v_index equal
// value is sent in
// load BlockScan with special reduce operation
// __syncthreads()
// block store
// __syncthreads()
// for each item and next per thread, if smem_storage.iter_input_start + v_index are different
// atomicAdd/Min/Max item to global mem according to reduction type
//

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
template <typename KernelPolicy, typename Problem, typename Functor,
    gunrock::oprtr::advance::TYPE        ADVANCE_TYPE = gunrock::oprtr::advance::V2V,
    gunrock::oprtr::advance::REDUCE_TYPE R_TYPE       = gunrock::oprtr::advance::EMPTY,
    gunrock::oprtr::advance::REDUCE_OP   R_OP         = gunrock::oprtr::advance::NONE>
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
    //util::Array1D<typename Problem::SizeT, unsigned char> &cub_scan_space,
    typename Problem::SizeT     max_in,
    typename Problem::SizeT     max_out,
    CudaContext                &context,
    cudaStream_t                stream,
    //TYPE                        ADVANCE_TYPE,
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
    cudaError_t retval = cudaSuccess;
    if (frontier_attribute->queue_length == 0)
    {
        //printf("setting output_length to 0");
        util::MemsetKernel<SizeT><<<1,1,0,stream>>>(frontier_attribute->output_length.GetPointer(util::DEVICE),0,1);
        return retval;
    }

    SizeT num_block = frontier_attribute->queue_length/KernelPolicy::THREADS + 1;
    //if (num_block > 256) num_block = 256;
    //printf("queue_length = %lld, num_blocks = %lld, block_size = %d\n",
    //    (long long)frontier_attribute -> queue_length,
    //    (long long)num_block,
    //    KernelPolicy::THREADS);
    if (isBackward<KernelPolicy::ADVANCE_MODE>())
    {
        gunrock::oprtr::edge_map_partitioned_backward::GetEdgeCounts
            <KernelPolicy, Problem, Functor>
            <<< num_block, KernelPolicy::THREADS, 0, stream>>>(
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
    //else if (KernelPolicy::ADVANCE_MODE == LB ||
    //         KernelPolicy::ADVANCE_MODE == LB_LIGHT ||
    //         KernelPolicy::ADVANCE_MODE == TWC_FORWARD ||
    //         KernelPolicy::ADVANCE_MODE == LB_CULL)
    else {
        gunrock::oprtr::edge_map_partitioned::GetEdgeCounts
            <KernelPolicy, Problem, Functor, ADVANCE_TYPE, R_TYPE, R_OP>
            <<< num_block, KernelPolicy::THREADS, 0, stream>>>(
                d_offsets,
                d_indices,
                d_inv_offsets,
                d_inv_indices,
                d_in_key_queue,
                partitioned_scanned_edges,
                frontier_attribute->queue_length, // TODO: +1?
                max_in,
                max_out,
                //ADVANCE_TYPE,
                in_inv,
                out_inv);
        //util::DisplayDeviceResults(partitioned_scanned_edges, frontier_attribute->queue_length);
    }

    //if (KernelPolicy::ADVANCE_MODE == LB ||
    //    KernelPolicy::ADVANCE_MODE == LB_LIGHT ||
    //    KernelPolicy::ADVANCE_MODE == LB_BACKWARD)
    //{
        Scan<mgpu::MgpuScanTypeInc>(
            partitioned_scanned_edges,
            frontier_attribute->queue_length, // TODO: +1?
            (SizeT)0,
            mgpu::plus<SizeT>(),
            /*(SizeT*)NULL,*/ frontier_attribute -> output_length.GetPointer(util::DEVICE),
            (SizeT*)NULL,
            partitioned_scanned_edges,
            context);

        //if (retval = util::GRError(cudaMemcpyAsync(
        //    frontier_attribute->output_length.GetPointer(util::DEVICE),
        //    partitioned_scanned_edges + frontier_attribute->queue_length - 1, // TODO: +1?
        //    sizeof(SizeT), cudaMemcpyDeviceToDevice, stream),
        //    "cudaMemcpyAsync failed", __FILE__, __LINE__)) reutrn retval;

        return retval;
    //} else {
    //    util::MemsetKernel<<<1,1,0,stream>>>(
    //        frontier_attribute->output_length.GetPointer(util::DEVICE),
    //        0, 1);
    //}
    //return cudaSuccess;
}

template <
    typename    _KernelPolicy,
    typename    _Problem,
    typename    _Functor,
    TYPE        _ADVANCE_TYPE,
    REDUCE_OP   _R_OP        ,
    REDUCE_TYPE _R_TYPE      >
struct KernelParameter
{
    typedef _KernelPolicy KernelPolicy;
    typedef _Problem      Problem;
    typedef _Functor      Functor;
    static const TYPE        ADVANCE_TYPE = _ADVANCE_TYPE;
    static const REDUCE_OP   R_OP         = _R_OP;
    static const REDUCE_TYPE R_TYPE       = _R_TYPE;

    gunrock::app::EnactorStats<typename KernelPolicy::SizeT>
                                            *enactor_stats;
    gunrock::app::FrontierAttribute<typename KernelPolicy::SizeT>
                                            *frontier_attribute;
    typename Problem::DataSlice             *h_data_slice;
    typename Problem::DataSlice             *d_data_slice;
    typename Problem::VertexId              *d_backward_index_queue;
    bool                                    *d_backward_frontier_map_in;
    bool                                    *d_backward_frontier_map_out;
    typename KernelPolicy::SizeT            *d_partitioned_scanned_edges;
    typename KernelPolicy::VertexId         *d_in_key_queue;
    typename KernelPolicy::VertexId         *d_out_key_queue;
    typename KernelPolicy::Value            *d_in_value_queue;
    typename KernelPolicy::Value            *d_out_value_queue;
    typename KernelPolicy::SizeT            *d_row_offsets;
    typename KernelPolicy::VertexId         *d_column_indices;
    typename KernelPolicy::SizeT            *d_column_offsets;
    typename KernelPolicy::VertexId         *d_row_indices;
    typename KernelPolicy::SizeT             max_in;
    typename KernelPolicy::SizeT             max_out;
    typename Functor::LabelT                 label;
    util::CtaWorkProgress<typename KernelPolicy::SizeT> *work_progress;
    CudaContext                             *context;
    cudaStream_t                             stream;
    //TYPE                                     ADVANCE_TYPE,
    bool                                     input_inverse_graph ;
    bool                                     output_inverse_graph;
    bool                                     get_output_length   ;
    //REDUCE_OP                                R_OP              = gunrock::oprtr::advance::NONE,
    //REDUCE_TYPE                              R_TYPE            = gunrock::oprtr::advance::EMPTY,
    typename KernelPolicy::Value            *d_value_to_reduce;
    typename KernelPolicy::Value            *d_reduce_frontier;
    typename KernelPolicy::Value            *d_reduced_value  ;
};

template <typename Parameter>
cudaError_t ComputeOutputLength(Parameter* parameter)
{
    return ComputeOutputLength<
        typename Parameter::KernelPolicy,
        typename Parameter::Problem,
        typename Parameter::Functor,
        Parameter::ADVANCE_TYPE,
        Parameter::R_TYPE,
        Parameter::R_OP>(
        parameter -> frontier_attribute,
        parameter -> d_row_offsets,
        parameter -> d_column_indices,
        parameter -> d_column_offsets,
        parameter -> d_row_indices,
        parameter -> d_in_key_queue,
        parameter -> d_partitioned_scanned_edges,
        parameter -> max_in,
        parameter -> max_out,
        parameter -> context[0],
        parameter -> stream,
        parameter -> get_output_length,
        parameter -> input_inverse_graph,
        parameter -> output_inverse_graph);
}

template <typename Parameter, gunrock::oprtr::advance::MODE ADVANCE_MODE>
struct LaunchKernel_
{
    static cudaError_t Launch(Parameter *parameter)
    {
        extern void UnSupportedAdvanceMode();
        UnSupportedAdvanceMode();
        return util::GRError(cudaErrorInvalidDeviceFunction,
            "UnSupportedAdvanceMode", __FILE__, __LINE__);
    }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::advance::TWC_FORWARD>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value;

    static cudaError_t Launch(Parameter *parameter)
    {
        cudaError_t retval = cudaSuccess;
         // Load Thread Warp CTA Forward Kernel
        gunrock::oprtr::edge_map_forward::Kernel
            <typename Parameter::KernelPolicy::THREAD_WARP_CTA_FORWARD,
            typename Parameter::Problem,
            typename Parameter::Functor,
            Parameter::ADVANCE_TYPE,
            Parameter::R_TYPE,
            Parameter::R_OP>
            <<< parameter -> enactor_stats -> advance_grid_size,
            Parameter::KernelPolicy::THREAD_WARP_CTA_FORWARD::THREADS,
            0,
            parameter -> stream>>>(
            parameter -> frontier_attribute -> queue_reset,
            (VertexId) parameter -> frontier_attribute -> queue_index, // TODO: match this type
            parameter -> label,//(int) parameter -> enactor_stats -> iteration,        // TODO: match this type
            parameter -> d_row_offsets,
            parameter -> d_column_offsets,
            parameter -> d_column_indices,
            parameter -> d_row_indices,
            parameter -> d_in_key_queue,              // d_in_queue
            parameter -> d_out_key_queue,            // d_out_queue
            parameter -> d_out_value_queue,          // d_pred_out_queue
            parameter -> d_data_slice,
            parameter -> frontier_attribute -> queue_length,
            parameter -> max_in,                   // max_in_queue
            parameter -> max_out,                 // max_out_queue
            parameter -> work_progress[0],
            parameter -> enactor_stats -> advance_kernel_stats,
            //Parameter::ADVANCE_TYPE,
            parameter -> input_inverse_graph,
            //Parameter::R_TYPE,
            //Parameter::R_OP,
            parameter -> d_value_to_reduce,
            parameter -> d_reduce_frontier);

        // Do segreduction using d_scanned_edges and d_reduce_frontier
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
        return retval;
    }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::advance::TWC_BACKWARD>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value;

    static cudaError_t Launch(Parameter *parameter)
    {
        cudaError_t retval = cudaSuccess;
        // Load Thread Warp CTA Backward Kernel
        // Edge Map
        gunrock::oprtr::edge_map_backward::Kernel
            <typename Parameter::KernelPolicy::THREAD_WARP_CTA_BACKWARD,
            Parameter::Problem, Parameter::Functor>
            <<< parameter -> enactor_stats -> advance_grid_size,
            Parameter::KernelPolicy::THREAD_WARP_CTA_BACKWARD::THREADS,
            0, parameter -> stream>>>(
            parameter -> frontier_attribute -> queue_reset,
            parameter -> frontier_attribute -> queue_index,
            parameter -> frontier_attribute -> queue_length,
            parameter -> d_in_key_queue,              // d_in_queue
            parameter -> d_backward_index_queue,            // d_in_index_queue
            parameter -> frontier_attribute -> selector == 1 ?
                parameter -> d_backward_frontier_map_in  :
                parameter -> d_backward_frontier_map_out,
            parameter -> frontier_attribute -> selector == 1 ?
                parameter -> d_backward_frontier_map_out :
                parameter -> d_backward_frontier_map_in ,
            parameter -> d_column_offsets,
            parameter -> d_row_indices,
            parameter -> d_data_slice,
            parameter -> work_progress[0],
            parameter -> enactor_stats -> advance_kernel_stats,
            Parameter::ADVANCE_TYPE);
        return retval;
   }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::advance::LB_BACKWARD>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value;
    typedef typename Parameter::KernelPolicy::LOAD_BALANCED LBPOLICY;

    static cudaError_t Launch(Parameter *parameter)
    {
        cudaError_t retval = cudaSuccess;
         // Load Thread Warp CTA Backward Kernel
        SizeT num_block = parameter -> frontier_attribute -> queue_length/
            LBPOLICY::THREADS + 1;
        if (parameter -> get_output_length)
        {
            if (retval = ComputeOutputLength<Parameter>(parameter))
                return retval;
        }

        // Edge Map
        gunrock::oprtr::edge_map_partitioned_backward::RelaxLightEdges<
            LBPOLICY, Parameter::Problem, Parameter::Functor>
            <<< num_block, LBPOLICY::THREADS, 0, parameter -> stream >>>(
            parameter -> frontier_attribute -> queue_reset,
            parameter -> frontier_attribute -> queue_index,
            parameter -> label, //parameter -> enactor_stats -> iteration,
            parameter -> d_column_offsets,
            parameter -> d_row_indices,
            (VertexId*)NULL,
            parameter -> d_partitioned_scanned_edges,  // TODO: +1?
            parameter -> d_in_key_queue,
            (parameter -> frontier_attribute -> selector == 1) ?
                parameter -> d_backward_frontier_map_in  :
                parameter -> d_backward_frontier_map_out,
            (parameter -> frontier_attribute -> selector == 1) ?
                parameter -> d_backward_frontier_map_out :
                parameter -> d_backward_frontier_map_in ,
            parameter -> d_data_slice,
            parameter -> frontier_attribute -> queue_length,
            parameter -> frontier_attribute -> output_length.GetPointer(util::DEVICE),
            parameter -> max_in,
            parameter -> max_out,
            parameter -> work_progress[0],
            parameter -> enactor_stats -> advance_kernel_stats,
            Parameter::ADVANCE_TYPE,
            parameter -> input_inverse_graph);
        return retval;
   }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::advance::LB>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value;
    typedef typename Parameter::KernelPolicy::LOAD_BALANCED LBPOLICY;

    static cudaError_t Launch(Parameter *parameter)
    {
        cudaError_t retval = cudaSuccess;
        // load edge-expand-partitioned kernel
        if (parameter -> get_output_length)
        {
            if (retval = ComputeOutputLength(parameter))
                return retval;
            if (retval = util::GRError(cudaStreamSynchronize(parameter -> stream),
                "cudaStreamSynchronize failed", __FILE__, __LINE__))
                return retval;
        }
        //printf("output_length = %lld\n", (long long)frontier_attribute.output_length[0]);
        if (/*!parameter -> get_output_length || (parameter -> get_output_length &&*/
            //parameter -> frontier_attribute -> output_length[0] < LBPOLICY::LIGHT_EDGE_THRESHOLD)//)
            parameter -> frontier_attribute -> output_length[0] < 64 * 2 * LBPOLICY::THREADS)
        {
            SizeT num_block = parameter -> frontier_attribute -> queue_length / LBPOLICY::SCRATCH_ELEMENTS + 1;
            //printf("using RelaxLightEdges\n");
            gunrock::oprtr::edge_map_partitioned::RelaxLightEdges
                <LBPOLICY,
                typename Parameter::Problem,
                typename Parameter::Functor,
                Parameter::ADVANCE_TYPE,
                Parameter::R_TYPE,
                Parameter::R_OP>
                <<< num_block, LBPOLICY::THREADS, 0, parameter -> stream>>>(
                parameter -> frontier_attribute -> queue_reset,
                parameter -> frontier_attribute -> queue_index,
                parameter -> label, //parameter -> enactor_stats -> iteration,
                parameter -> d_row_offsets,
                parameter -> d_column_offsets,
                parameter -> d_column_indices,
                parameter -> d_row_indices,
                parameter -> d_partitioned_scanned_edges, // TODO: +1?
                parameter -> d_in_key_queue,
                parameter -> d_out_key_queue,
                parameter -> d_out_value_queue,
                parameter -> d_data_slice,
                parameter -> frontier_attribute -> queue_length,
                parameter -> frontier_attribute -> output_length.GetPointer(util::DEVICE),
                parameter -> max_in,
                parameter -> max_out,
                parameter -> work_progress[0],
                parameter -> enactor_stats -> advance_kernel_stats,
                //ADVANCE_TYPE,
                parameter -> input_inverse_graph,
                parameter -> output_inverse_graph,
                //R_TYPE,
                //R_OP,
                parameter -> d_value_to_reduce,
                parameter -> d_reduce_frontier);
        }
        else //if (/*get_output_length &&*/ parameter -> frontier_attribute -> output_length[0] >= LBPOLICY::LIGHT_EDGE_THRESHOLD)
        {
            int num_blocks = parameter -> frontier_attribute -> output_length[0] / 2 / LBPOLICY::THREADS + 1; // LBPOLICY::BLOCKS
            if (num_blocks > LBPOLICY::BLOCKS)
                num_blocks = LBPOLICY::BLOCKS;
            SizeT split_val = (parameter -> frontier_attribute -> output_length[0] +
                num_blocks - 1) / num_blocks;
            //printf("using RelaxLightEdges2, input_length = %lld, ouput_length = %lld, split_val = %lld\n",
            //    (long long)parameter -> frontier_attribute -> queue_length,
            //    (long long)parameter -> frontier_attribute -> output_length[0],
            //    (long long)split_val);
            util::MemsetIdxKernel<<<1, 256, 0, parameter -> stream>>>(
                parameter -> enactor_stats -> node_locks.GetPointer(util::DEVICE),
                num_blocks,
                split_val);
            SortedSearch<MgpuBoundsLower>(
                parameter -> enactor_stats -> node_locks.GetPointer(util::DEVICE),
                num_blocks,
                parameter -> d_partitioned_scanned_edges,
                parameter -> frontier_attribute -> queue_length,
                parameter -> enactor_stats -> node_locks_out.GetPointer(util::DEVICE),
                parameter -> context[0]);
            //util::cpu_mt::PrintGPUArray("node_locks_out",
            //    parameter -> enactor_stats -> node_locks_out.GetPointer(util::DEVICE),
            //    num_blocks + 1, -1, -1, -1, parameter -> stream);
            gunrock::oprtr::edge_map_partitioned::RelaxPartitionedEdges2
                <LBPOLICY,
                typename Parameter::Problem,
                typename Parameter::Functor,
                Parameter::ADVANCE_TYPE,
                Parameter::R_TYPE,
                Parameter::R_OP>
                <<< num_blocks, LBPOLICY::THREADS, 0, parameter -> stream>>>(
                parameter -> frontier_attribute -> queue_reset,
                parameter -> frontier_attribute -> queue_index,
                parameter -> label, //parameter -> enactor_stats -> iteration,
                parameter -> d_row_offsets,
                parameter -> d_column_offsets,
                parameter -> d_column_indices,
                parameter -> d_row_indices,
                parameter -> d_partitioned_scanned_edges,
                parameter -> enactor_stats -> node_locks_out.GetPointer(util::DEVICE),
                num_blocks,
                parameter -> d_in_key_queue,
                parameter -> d_out_key_queue,
                parameter -> d_out_value_queue,
                parameter -> d_data_slice,
                parameter -> frontier_attribute -> queue_length,
                parameter -> frontier_attribute -> output_length.GetPointer(util::DEVICE),
                split_val,
                parameter -> max_in,
                parameter -> max_out,
                parameter -> work_progress[0],
                parameter -> enactor_stats -> advance_kernel_stats,
                parameter -> input_inverse_graph,
                parameter -> output_inverse_graph,
                parameter -> d_value_to_reduce,
                parameter -> d_reduce_frontier);
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
        return retval;
    }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::advance::LB_LIGHT>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value;
    typedef typename Parameter::KernelPolicy::LOAD_BALANCED LBPOLICY;

    static cudaError_t Launch(Parameter *parameter)
    {
        cudaError_t retval = cudaSuccess;
        // load edge-expand-partitioned kernel
        SizeT num_block = (parameter -> frontier_attribute -> queue_length +
             LBPOLICY::SCRATCH_ELEMENTS - 1) / LBPOLICY::SCRATCH_ELEMENTS;

        if (parameter -> get_output_length)
        {
            if (retval = ComputeOutputLength(parameter))
                return retval;
        }

        gunrock::oprtr::edge_map_partitioned::RelaxLightEdges
            <LBPOLICY,
            typename Parameter::Problem,
            typename Parameter::Functor,
            Parameter::ADVANCE_TYPE,
            Parameter::R_TYPE,
            Parameter::R_OP>
            <<< num_block, LBPOLICY::THREADS, 0, parameter -> stream>>>(
            parameter -> frontier_attribute -> queue_reset,
            parameter -> frontier_attribute -> queue_index,
            parameter -> label, //parameter -> enactor_stats -> iteration,
            parameter -> d_row_offsets,
            parameter -> d_column_offsets,
            parameter -> d_column_indices,
            parameter -> d_row_indices,
            parameter -> d_partitioned_scanned_edges, // TODO: +1?
            parameter -> d_in_key_queue,
            parameter -> d_out_key_queue,
            parameter -> d_out_value_queue,
            parameter -> d_data_slice,
            parameter -> frontier_attribute -> queue_length,
            parameter -> frontier_attribute -> output_length.GetPointer(util::DEVICE),
            parameter -> max_in,
            parameter -> max_out,
            parameter -> work_progress[0],
            parameter -> enactor_stats -> advance_kernel_stats,
            parameter -> input_inverse_graph,
            parameter -> output_inverse_graph,
            parameter -> d_value_to_reduce,
            parameter -> d_reduce_frontier);
        return retval;
   }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::advance::LB_CULL>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value;
    typedef typename Parameter::KernelPolicy::LOAD_BALANCED_CULL KernelPolicy;

    static cudaError_t Launch(Parameter *parameter)
    {
        cudaError_t retval = cudaSuccess;
        // load edge-expand-partitioned kernel


        if (parameter -> get_output_length)
        {
            if (retval = ComputeOutputLength(parameter))
                return retval;
            if (retval = util::GRError(cudaStreamSynchronize(parameter -> stream),
                "cudaStreamSynchronize failed", __FILE__, __LINE__))
                return retval;
        }
        //printf("output_length = %lld\n", (long long)frontier_attribute.output_length[0]);
        if (/*!parameter -> get_output_length || (parameter -> get_output_length &&*/
            //parameter -> frontier_attribute -> output_length[0] < LBPOLICY::LIGHT_EDGE_THRESHOLD)//)
            parameter -> frontier_attribute -> output_length[0] < 64LL * 2 * KernelPolicy::THREADS)
        {
            SizeT num_blocks = (parameter -> frontier_attribute -> queue_length +
                KernelPolicy::SCRATCH_ELEMENTS - 1) / KernelPolicy::SCRATCH_ELEMENTS;
            //if (num_blocks > 480 ) num_blocks = 480;
            //printf("using RelaxLightEdges, num_bolocks = %d, labels = %p\n", num_blocks, parameter->h_data_slice->labels.GetPointer(util::DEVICE));
            gunrock::oprtr::edge_map_partitioned_cull::RelaxLightEdges
                <KernelPolicy,
                typename Parameter::Problem,
                typename Parameter::Functor,
                Parameter::ADVANCE_TYPE,
                Parameter::R_TYPE,
                Parameter::R_OP>
                <<< num_blocks, KernelPolicy::THREADS, 0, parameter -> stream>>>(
                parameter -> frontier_attribute -> queue_reset,
                parameter -> frontier_attribute -> queue_index,
                parameter -> label, //parameter -> enactor_stats -> iteration,
                parameter -> d_row_offsets,
                parameter -> d_column_offsets,
                parameter -> d_column_indices,
                parameter -> d_row_indices,
                parameter -> d_partitioned_scanned_edges, // TODO: +1?
                parameter -> d_in_key_queue,
                parameter -> d_out_key_queue,
                parameter -> d_out_value_queue,
                parameter -> d_data_slice,
                parameter -> frontier_attribute -> queue_length,
                parameter -> frontier_attribute -> output_length.GetPointer(util::DEVICE),
                parameter -> max_in,
                parameter -> max_out,
                parameter -> work_progress[0],
                parameter -> enactor_stats -> advance_kernel_stats,
                //ADVANCE_TYPE,
                parameter -> input_inverse_graph,
                parameter -> output_inverse_graph,
                //R_TYPE,
                //R_OP,
                parameter -> d_value_to_reduce,
                parameter -> d_reduce_frontier);
        }
        else //if (/*get_output_length &&*/ parameter -> frontier_attribute -> //output_length[0] >= LBPOLICY::LIGHT_EDGE_THRESHOLD)
        {
            SizeT num_blocks = parameter -> frontier_attribute -> output_length[0] / 2 / KernelPolicy::THREADS + 1; // LBPOLICY::BLOCKS
            if (num_blocks > 840)
                num_blocks = 840;
            if (num_blocks < 1) num_blocks = 1;
            SizeT split_val = (parameter -> frontier_attribute -> output_length[0] +
                num_blocks - 1) / num_blocks;
            //printf("using RelaxLightEdges2, input_length = %lld, ouput_length = %lld, split_val = %lld\n",
            //    (long long)parameter -> frontier_attribute -> queue_length,
            //    (long long)parameter -> frontier_attribute -> output_length[0],
            //    (long long)split_val);
            util::MemsetIdxKernel<<<1, num_blocks, 0, parameter -> stream>>>(
                parameter -> enactor_stats -> node_locks.GetPointer(util::DEVICE),
                num_blocks,
                split_val);
            SortedSearch<MgpuBoundsLower>(
                parameter -> enactor_stats -> node_locks.GetPointer(util::DEVICE),
                num_blocks,
                parameter -> d_partitioned_scanned_edges,
                parameter -> frontier_attribute -> queue_length,
                parameter -> enactor_stats -> node_locks_out.GetPointer(util::DEVICE),
                parameter -> context[0]);
            //util::cpu_mt::PrintGPUArray("node_locks_out",
            //    parameter -> enactor_stats -> node_locks_out.GetPointer(util::DEVICE),
            //    num_blocks + 1, -1, -1, -1, parameter -> stream);
            gunrock::oprtr::edge_map_partitioned_cull::RelaxPartitionedEdges2
                <KernelPolicy,
                typename Parameter::Problem,
                typename Parameter::Functor,
                Parameter::ADVANCE_TYPE,
                Parameter::R_TYPE,
                Parameter::R_OP>
                <<< num_blocks,
                KernelPolicy::THREADS,
                0, parameter -> stream>>>(
                parameter -> frontier_attribute -> queue_reset,
                parameter -> frontier_attribute -> queue_index,
                parameter -> label, //parameter -> enactor_stats -> iteration,
                parameter -> d_row_offsets,
                parameter -> d_column_offsets,
                parameter -> d_column_indices,
                parameter -> d_row_indices,
                parameter -> d_partitioned_scanned_edges,
                parameter -> enactor_stats -> node_locks_out.GetPointer(util::DEVICE),
                num_blocks,
                parameter -> d_in_key_queue,
                parameter -> d_out_key_queue,
                parameter -> d_out_value_queue,
                parameter -> d_data_slice,
                parameter -> frontier_attribute -> queue_length,
                parameter -> frontier_attribute -> output_length.GetPointer(util::DEVICE),
                split_val,
                parameter -> max_in,
                parameter -> max_out,
                parameter -> work_progress[0],
                parameter -> enactor_stats -> advance_kernel_stats,
                parameter -> input_inverse_graph,
                parameter -> output_inverse_graph,
                parameter -> d_value_to_reduce,
                parameter -> d_reduce_frontier);
            //util::DisplayDeviceResults(d_out_key_queue, output_queue_len);
        }
        return retval;
    }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::advance::LB_LIGHT_CULL>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value;
    typedef typename Parameter::KernelPolicy::LOAD_BALANCED_CULL KernelPolicy;

    static cudaError_t Launch(Parameter *parameter)
    {
        cudaError_t retval = cudaSuccess;
        // load edge-expand-partitioned kernel


        if (parameter -> get_output_length)
        {
            if (retval = ComputeOutputLength(parameter))
                return retval;
            if (retval = util::GRError(cudaStreamSynchronize(parameter -> stream),
                "cudaStreamSynchronize failed", __FILE__, __LINE__))
                return retval;
        }

        SizeT num_blocks = (parameter -> frontier_attribute -> queue_length +
            KernelPolicy::SCRATCH_ELEMENTS - 1) / KernelPolicy::SCRATCH_ELEMENTS;
        //if (num_blocks > 480 ) num_blocks = 480;
        //printf("using RelaxLightEdges, num_bolocks = %d, labels = %p\n", num_blocks, parameter->h_data_slice->labels.GetPointer(util::DEVICE));
        gunrock::oprtr::edge_map_partitioned_cull::RelaxLightEdges
            <KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor,
            Parameter::ADVANCE_TYPE,
            Parameter::R_TYPE,
            Parameter::R_OP>
            <<< num_blocks, KernelPolicy::THREADS, 0, parameter -> stream>>>(
            parameter -> frontier_attribute -> queue_reset,
            parameter -> frontier_attribute -> queue_index,
            parameter -> label, //parameter -> enactor_stats -> iteration,
            parameter -> d_row_offsets,
            parameter -> d_column_offsets,
            parameter -> d_column_indices,
            parameter -> d_row_indices,
            parameter -> d_partitioned_scanned_edges, // TODO: +1?
            parameter -> d_in_key_queue,
            parameter -> d_out_key_queue,
            parameter -> d_out_value_queue,
            parameter -> d_data_slice,
            parameter -> frontier_attribute -> queue_length,
            parameter -> frontier_attribute -> output_length.GetPointer(util::DEVICE),
            parameter -> max_in,
            parameter -> max_out,
            parameter -> work_progress[0],
            parameter -> enactor_stats -> advance_kernel_stats,
            //ADVANCE_TYPE,
            parameter -> input_inverse_graph,
            parameter -> output_inverse_graph,
            //R_TYPE,
            //R_OP,
            parameter -> d_value_to_reduce,
            parameter -> d_reduce_frontier);
       return retval;
    }
};

template <typename Parameter>
struct LaunchKernel_<Parameter, gunrock::oprtr::advance::ALL_EDGES>
{
    typedef typename Parameter::Problem::SizeT         SizeT;
    typedef typename Parameter::Problem::VertexId      VertexId;
    typedef typename Parameter::Problem::Value         Value;
    typedef typename Parameter::KernelPolicy::EDGES KernelPolicy;

    static cudaError_t Launch(Parameter *parameter)
    {
        cudaError_t retval = cudaSuccess;
        // load edge-expand-partitioned kernel
        SizeT num_blocks = (parameter -> frontier_attribute -> queue_length +
             KernelPolicy::THREADS - 1) / KernelPolicy::THREADS;
        if (num_blocks > 480) num_blocks = 480;

        gunrock::oprtr::all_edges_advance::Advance_Edges
            <KernelPolicy,
            typename Parameter::Problem,
            typename Parameter::Functor,
            Parameter::ADVANCE_TYPE,
            Parameter::R_TYPE,
            Parameter::R_OP>
            <<< num_blocks, KernelPolicy::THREADS, 0, parameter -> stream>>>(
            parameter -> frontier_attribute -> queue_reset,
            parameter -> frontier_attribute -> queue_index,
            parameter -> label, //parameter -> enactor_stats -> iteration,
            parameter -> max_in,
            parameter -> max_out,
            parameter -> d_row_offsets,
            parameter -> d_column_indices,
            (SizeT*) NULL,
            (SizeT*) NULL,
            parameter -> d_data_slice,
            parameter -> frontier_attribute -> queue_length,
            parameter -> frontier_attribute -> output_length.GetPointer(util::DEVICE),
            parameter -> work_progress[0]);
        return retval;
   }
};


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

template <typename KernelPolicy, typename Problem, typename Functor,
    TYPE        ADVANCE_TYPE,
    REDUCE_OP   R_OP         = gunrock::oprtr::advance::NONE,
    REDUCE_TYPE R_TYPE       = gunrock::oprtr::advance::EMPTY>
cudaError_t LaunchKernel(
    gunrock::app::EnactorStats<typename KernelPolicy::SizeT>
                                            &enactor_stats,
    gunrock::app::FrontierAttribute<typename KernelPolicy::SizeT>
                                            &frontier_attribute,
    typename Functor::LabelT                 label,
    typename Problem::DataSlice             *h_data_slice,
    typename Problem::DataSlice             *d_data_slice,
    typename Problem::VertexId              *d_backward_index_queue,
    bool                                    *d_backward_frontier_map_in,
    bool                                    *d_backward_frontier_map_out,
    typename KernelPolicy::SizeT            *d_partitioned_scanned_edges,
    typename KernelPolicy::VertexId         *d_in_key_queue,
    typename KernelPolicy::VertexId         *d_out_key_queue,
    typename KernelPolicy::Value            *d_in_value_queue,
    typename KernelPolicy::Value            *d_out_value_queue,
    typename KernelPolicy::SizeT            *d_row_offsets,
    typename KernelPolicy::VertexId         *d_column_indices,
    typename KernelPolicy::SizeT            *d_column_offsets,
    typename KernelPolicy::VertexId         *d_row_indices,
    typename KernelPolicy::SizeT             max_in,
    typename KernelPolicy::SizeT             max_out,
    util::CtaWorkProgress<typename KernelPolicy::SizeT> &work_progress,
    CudaContext                             &context,
    cudaStream_t                             stream,
    bool                                     input_inverse_graph  = false,
    bool                                     output_inverse_graph = false,
    bool                                     get_output_length = true,
    typename KernelPolicy::Value            *d_value_to_reduce = NULL,
    typename KernelPolicy::Value            *d_reduce_frontier = NULL,
    typename KernelPolicy::Value            *d_reduced_value   = NULL)
{
    cudaError_t retval = cudaSuccess;
    if (frontier_attribute.queue_reset)
    {
        if (retval = work_progress.Reset_(0, stream))
            return retval;
    }
    if (frontier_attribute.queue_length == 0) return retval;

    typedef KernelParameter<KernelPolicy, Problem, Functor, ADVANCE_TYPE, R_OP, R_TYPE> Parameter;
    Parameter parameter;
    parameter. enactor_stats                = &enactor_stats;
    parameter. frontier_attribute           = &frontier_attribute;
    parameter. label                        =  label;
    parameter. h_data_slice                 =  h_data_slice;
    parameter. d_data_slice                 =  d_data_slice;
    parameter. d_backward_index_queue       =  d_backward_index_queue;
    parameter. d_backward_frontier_map_in   =  d_backward_frontier_map_in;
    parameter. d_backward_frontier_map_out  =  d_backward_frontier_map_out;
    parameter. d_partitioned_scanned_edges  =  d_partitioned_scanned_edges;
    parameter. d_in_key_queue               =  d_in_key_queue;
    parameter. d_out_key_queue              =  d_out_key_queue;
    parameter. d_in_value_queue             =  d_in_value_queue;
    parameter. d_out_value_queue            =  d_out_value_queue;
    parameter. d_row_offsets                =  d_row_offsets;
    parameter. d_column_indices             =  d_column_indices;
    parameter. d_column_offsets             =  d_column_offsets;
    parameter. d_row_indices                =  d_row_indices;
    parameter. max_in                       =  max_in;
    parameter. max_out                      =  max_out;
    parameter. work_progress                = &work_progress;
    parameter. context                      = &context;
    parameter. stream                       =  stream;
    parameter. input_inverse_graph          =  input_inverse_graph;
    parameter. output_inverse_graph         =  output_inverse_graph;
    parameter. get_output_length            =  get_output_length;
    parameter. d_value_to_reduce            =  d_value_to_reduce;
    parameter. d_reduce_frontier            =  d_reduce_frontier;
    parameter. d_reduced_value              =  d_reduced_value;

    if (retval = LaunchKernel_<Parameter, KernelPolicy::ADVANCE_MODE>::Launch(&parameter))
        return retval;
    return retval;
}


} //advance
} //oprtr
} //gunrock/
