// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * advance.cuh
 *
 * @brief API interfaces for advance calls
 */

#pragma once

#include <gunrock/oprtr/advance/advance_base.cuh>
#include <gunrock/oprtr/LB_advance/kernel.cuh>
#include <gunrock/oprtr/LB_CULL_advance/kernel.cuh>
#include <gunrock/oprtr/TWC_advance/kernel.cuh>
#include <gunrock/oprtr/AE_advance/kernel.cuh>

namespace gunrock {
namespace oprtr {
namespace advance {

// Steps for adding reduce:
// fix block_dim=512/1024 and items_per_thread=7 or 11, make block number a
// variable creating flag, value flag comes from whether two threads'
// smem_storage.iter_input_start + v_index equal value is sent in load BlockScan
// with special reduce operation
// __syncthreads()
// block store
// __syncthreads()
// for each item and next per thread, if smem_storage.iter_input_start + v_index
// are different atomicAdd/Min/Max item to global mem according to reduction
// type
//

/*template <typename Parameter, gunrock::oprtr::advance::MODE ADVANCE_MODE>
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
            (VertexId) parameter -> frontier_attribute -> queue_index, // TODO:
match this type parameter -> label,//(int) parameter -> enactor_stats ->
iteration,        // TODO: match this type parameter -> d_row_offsets, parameter
-> d_column_offsets, parameter -> d_column_indices, parameter -> d_row_indices,
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
            parameter -> d_reduce_frontier);*/

// Do segreduction using d_scanned_edges and d_reduce_frontier
// TODO: For TWC_Forward, Find a way to get the output_queue_len,
// also, try to get the scanned_edges array too. Then the following code will
// work.
/*if (R_TYPE != gunrock::oprtr::advance::EMPTY && d_value_to_reduce &&
d_reduce_frontier) { switch (R_OP) { case gunrock::oprtr::advance::PLUS: {
        SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges,
output_queue_len,frontier_attribute.queue_length, false, d_reduced_value,
(Value)0, mgpu::plus<typename KernelPolicy::Value>(), context); break;
    }
    case gunrock::oprtr::advance::MULTIPLIES: {
        SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges,
output_queue_len,frontier_attribute.queue_length, false, d_reduced_value,
(Value)1, mgpu::multiplies<typename KernelPolicy::Value>(), context); break;
    }
    case gunrock::oprtr::advance::MAXIMUM: {
        SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges,
output_queue_len,frontier_attribute.queue_length, false, d_reduced_value,
(Value)INT_MIN, mgpu::maximum<typename KernelPolicy::Value>(), context); break;
    }
    case gunrock::oprtr::advance::MINIMUM: {
        SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges,
output_queue_len,frontier_attribute.queue_length, false, d_reduced_value,
(Value)INT_MAX, mgpu::minimum<typename KernelPolicy::Value>(), context); break;
    }
    default:
        //default operator is plus
        SegReduceCsr(d_reduce_frontier, partitioned_scanned_edges,
output_queue_len,frontier_attribute.queue_length, false, d_reduced_value,
(Value)0, mgpu::plus<typename KernelPolicy::Value>(), context); break;
  }
}*/
/*return retval;
}
};*/

/*template <typename Parameter>
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
            parameter -> frontier_attribute ->
output_length.GetPointer(util::DEVICE), parameter -> max_in, parameter ->
max_out, parameter -> work_progress[0], parameter -> enactor_stats ->
advance_kernel_stats, Parameter::ADVANCE_TYPE, parameter ->
input_inverse_graph); return retval;
   }
};*/

/**
 * @brief Advance operator kernel entry point.
 *
 * @tparam KernelPolicy Kernel policy type for advance operator.
 * @tparam ProblemData Problem data type for advance operator.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam Op Operation for gather reduce. mgpu::plus<int> by default.
 *
 * @param[in] enactor_stats             EnactorStats object to store enactor
 * related variables and stast
 * @param[in] frontier_attribute        FrontierAttribute object to store
 * frontier attribute while doing the advance operation
 * @param[in] data_slice                Device pointer to the problem object's
 * data_slice member
 * @param[in] backward_index_queue      If backward mode is activated, this is
 * used to store the vertex index. (deprecated)
 * @param[in] backward_frontier_map_in  If backward mode is activated, this is
 * used to store input frontier bitmap
 * @param[in] backward_frontier_map_out If backward mode is activated, this is
 * used to store output frontier bitmap
 * @param[in] partitioned_scanned_edges If load balanced mode is activated, this
 * is used to store the scanned edge number for neighbor lists in current
 * frontier
 * @param[in] d_in_key_queue            Device pointer of input key array to the
 * incoming frontier queue
 * @param[in] d_out_key_queue           Device pointer of output key array to
 * the outgoing frontier queue
 * @param[in] d_in_value_queue          Device pointer of input value array to
 * the incoming frontier queue
 * @param[in] d_out_value_queue         Device pointer of output value array to
 * the outgoing frontier queue
 * @param[in] d_row_offsets             Device pointer of SizeT to the row
 * offsets queue
 * @param[in] d_column_indices          Device pointer of VertexId to the column
 * indices queue
 * @param[in] d_column_offsets          Device pointer of SizeT to the row
 * offsets queue for inverse graph
 * @param[in] d_row_indices             Device pointer of VertexId to the column
 * indices queue for inverse graph
 * @param[in] max_in_queue              Maximum number of elements we can place
 * into the incoming frontier
 * @param[in] max_out_queue             Maximum number of elements we can place
 * into the outgoing frontier
 * @param[in] work_progress             queueing counters to record work
 * progress
 * @param[in] context                   CudaContext pointer for moderngpu APIs
 * @param[in] ADVANCE_TYPE              enumerator of advance type: V2V, V2E,
 * E2V, or E2E
 * @param[in] inverse_graph             whether this iteration of advance
 * operation is in the opposite direction to the previous iteration (false by
 * default)
 * @param[in] REDUCE_OP                 enumerator of available reduce
 * operations: plus, multiplies, bit_or, bit_and, bit_xor, maximum, minimum.
 * none by default.
 * @param[in] REDUCE_TYPE               enumerator of available reduce types:
 * EMPTY(do not do reduce) VERTEX(extract value from |V| array) EDGE(extract
 * value from |E| array)
 * @param[in] d_value_to_reduce         array to store values to reduce
 * @param[out] d_reduce_frontier        neighbor list values for nodes in the
 * output frontier
 * @param[out] d_reduced_value          array to store reduced values
 */

// TODO: Reduce by neighbor list now only supports LB advance mode.
// TODO: Add a switch to enable advance+filter (like in BFS), pissibly moving
// idempotent ops from filter to advance?

/*template <typename KernelPolicy, typename Problem, typename Functor,
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

    typedef KernelParameter<KernelPolicy, Problem, Functor, ADVANCE_TYPE, R_OP,
R_TYPE> Parameter; Parameter parameter; parameter. enactor_stats =
&enactor_stats; parameter. frontier_attribute           = &frontier_attribute;
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

    if (retval = LaunchKernel_<Parameter,
KernelPolicy::ADVANCE_MODE>::Launch(&parameter)) return retval; return retval;
}*/

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOpT,
          typename FilterOpT>
cudaError_t Launch(const GraphT &graph, const FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   AdvanceOpT advance_op, FilterOpT filter_op) {
  if (parameters.advance_mode == "LB")
    return LB::Launch<FLAG>(graph, frontier_in, frontier_out, parameters,
                            advance_op, filter_op);
  if (parameters.advance_mode == "LB_LIGHT")
    return LB::Launch_Light<FLAG>(graph, frontier_in, frontier_out, parameters,
                                  advance_op, filter_op);
  if (parameters.advance_mode == "LB_CULL")
    return LB_CULL::Launch<FLAG>(graph, frontier_in, frontier_out, parameters,
                                 advance_op, filter_op);
  if (parameters.advance_mode == "LB_LIGHT_CULL")
    return LB_CULL::Launch_Light<FLAG>(graph, frontier_in, frontier_out,
                                       parameters, advance_op, filter_op);
  if (parameters.advance_mode == "TWC")
    return TWC::Launch<FLAG>(graph, frontier_in, frontier_out, parameters,
                             advance_op, filter_op);
  if (parameters.advance_mode == "ALL_EDGES")
    return AE::Launch<FLAG>(graph, frontier_in, frontier_out, parameters,
                            advance_op, filter_op);

  return util::GRError(cudaErrorInvalidValue,
                       "AdvanceMode " + parameters.advance_mode + " undefined.",
                       __FILE__, __LINE__);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename OpT>
cudaError_t Launch(const GraphT &graph, const FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   OpT op) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  auto dummy_filter = [] __host__ __device__(
                          const VertexT &src, VertexT &dest,
                          const SizeT &edge_id, const VertexT &input_item,
                          const SizeT &input_pos,
                          SizeT &output_pos) -> bool { return true; };
  return oprtr::advance::Launch<FLAG>(graph, frontier_in, frontier_out,
                                      parameters, op, dummy_filter);
}

}  // namespace advance
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
