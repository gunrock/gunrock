// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file lp_enactor.cuh
 * @brief Primitive problem enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/lp/lp_problem.cuh>
#include <gunrock/app/lp/lp_functor.cuh>

namespace gunrock {
namespace app {
namespace lp {

/**
 * @brief Primitive enactor class.
 *
 * @tparam _Problem
 * @tparam INSTRUMWENT
 * @tparam _DEBUG
 * @tparam _SIZE_CHECK
 */
template <typename _Problem>
// bool _INSTRUMENT,
// bool _DEBUG,
// bool _SIZE_CHECK>
class LpEnactor
    : public EnactorBase<typename _Problem::SizeT /*, _DEBUG, _SIZE_CHECK*/> {
 protected:
  /**
   * @brief Prepare the enactor for kernel call.
   *
   * @param[in] problem Problem object holds both graph and primitive data.
   *
   * \return cudaError_t object indicates the success of all CUDA functions.
   */
  /*template <typename ProblemData>
  cudaError_t Setup(ProblemData *problem)
  {
      typedef typename ProblemData::SizeT    SizeT;
      typedef typename ProblemData::VertexId VertexId;

      cudaError_t retval = cudaSuccess;

      GraphSlice<SizeT, VertexId, Value>*
          graph_slice = problem->graph_slices[0];
      typename ProblemData::DataSlice*
          data_slice = problem->data_slices[0];

      return retval;
  }*/

 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexId VertexId;
  typedef typename Problem::Value Value;
  // static const bool INSTRUMENT = _INSTRUMENT;
  // static const bool DEBUG      = _DEBUG;
  // static const bool SIZE_CHECK = _SIZE_CHECK;
  typedef EnactorBase<SizeT> BaseEnactor;
  Problem *problem;
  ContextPtr *context;

  /**
   * @brief Primitive Constructor.
   *
   * @param[in] gpu_idx GPU indices
   */
  LpEnactor(int num_gpus = 1, int *gpu_idx = NULL, bool instrument = false,
            bool debug = false, bool size_check = true)
      : BaseEnactor(EDGE_FRONTIERS, num_gpus, gpu_idx, instrument, debug,
                    size_check),
        problem(NULL),
        context(NULL) {}

  /**
   * @brief Primitive Destructor.
   */
  virtual ~LpEnactor() {}

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /** @} */

  template <typename AdvanceKernelPolicy, typename FilterKernelPolicy>
  cudaError_t InitLp(ContextPtr *context, Problem *problem,
                     int max_grid_size = 0) {
    cudaError_t retval = cudaSuccess;

    if (retval =
            BaseEnactor::Init(max_grid_size, AdvanceKernelPolicy::CTA_OCCUPANCY,
                              FilterKernelPolicy::CTA_OCCUPANCY))
      return retval;

    this->problem = problem;
    this->context = context;

    // typename Problem::DataSlice
    //    *data_slice  = problem -> data_slices [0].GetPointer(util::HOST);

    // TODO:
    // One pass of connected component step to get rid of ocillation of
    // community ID assignment.
    // prepare edge frontier as input.
    // only set edge(u,v) whose weight(u)==weight(v) as active
    // assign smaller community ID to larger one
    // prepare node frontier and for each node do pointer jumping
    //
    // The above two steps will update the community ID for certain edges
    // whose node weights are equal.

    return retval;
  }

  /**
   * @brief Enacts computing on the specified graph.
   *
   * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
   * @tparam FilterKernelPolicy Kernel policy for filter operator.
   * @tparam Problem Problem type.
   *
   * @param[in] context CudaContext pointer for ModernGPU APIs
   * @param[in] problem Problem object.
   * @param[in] max_grid_size Max grid size for kernel calls.
   *
   * \return cudaError_t object indicates the success of all CUDA functions.
   */
  template <typename AdvanceKernelPolicy, typename FilterKernelPolicy>
  // typename Problem >
  cudaError_t EnactLp()
  // ContextPtr  context,
  // Problem*    problem,
  // int         max_grid_size = 0)
  {
    // Define functors for primitive
    typedef LpHookFunctor<VertexId, SizeT, Value, Problem> HookFunctor;
    typedef LpPtrJumpFunctor<VertexId, SizeT, Value, Problem> PtrJumpFunctor;
    typedef LpWeightUpdateFunctor<VertexId, SizeT, Value, Problem>
        WeightUpdateFunctor;
    typedef LpAssignWeightFunctor<VertexId, SizeT, Value, Problem>
        AssignWeightFunctor;
    typedef LpSwapLabelFunctor<VertexId, SizeT, Value, Problem>
        SwapLabelFunctor;

    typedef typename Problem::DataSlice DataSlice;
    typedef util::DoubleBuffer<VertexId, SizeT, Value> Frontier;
    typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;

    Problem *problem = this->problem;
    EnactorStats<SizeT> *enactor_stats = &this->enactor_stats[0];
    DataSlice *data_slice = problem->data_slices[0].GetPointer(util::HOST);
    DataSlice *d_data_slice = problem->data_slices[0].GetPointer(util::DEVICE);
    GraphSliceT *graph_slice = problem->graph_slices[0];
    Frontier *frontier_queue = &data_slice->frontier_queues[0];
    FrontierAttribute<SizeT> *frontier_attribute = &this->frontier_attribute[0];
    util::CtaWorkProgressLifetime<SizeT> *work_progress =
        &this->work_progress[0];
    cudaStream_t stream = data_slice->streams[0];
    ContextPtr context = this->context[0];
    cudaError_t retval = cudaSuccess;
    SizeT *d_scanned_edges = NULL;  // Used for LB

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::ArgMax(
        d_temp_storage, temp_storage_bytes,
        data_slice->final_weights.GetPointer(util::DEVICE),
        data_slice->argmax_kv.GetPointer(util::DEVICE), graph_slice->nodes,
        (int *)data_slice->offsets.GetPointer(util::DEVICE),
        (int *)data_slice->offsets.GetPointer(util::DEVICE) + 1);
    // Note that if num_edges < num_nodes (chain), there will be problem.
    do {
      // single-GPU graph slice
      if (retval = util::GRError(cudaMalloc((void **)&d_scanned_edges,
                                            graph_slice->edges * sizeof(SizeT)),
                                 "Problem cudaMalloc d_scanned_edges failed",
                                 __FILE__, __LINE__)) {
        return retval;
      }
      // TODO:
      // one pass of connected component.

      // initialize frontier as edge index using memsetidx
      util::MemsetIdxKernel<<<128, 128>>>(
          frontier_queue->keys[0].GetPointer(util::DEVICE), graph_slice->edges);
      frontier_attribute->queue_length = graph_slice->edges;
      frontier_attribute->queue_reset = true;
      //
      // launch filter, for edge(u,v) u=froms[node], v=tos[node]
      // if node_weights[u] == node_weights[v],
      // assign labels[u] = labels[v] = min(labels[u],labels[v])
      gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                           HookFunctor>(
          enactor_stats[0], frontier_attribute[0],
          typename HookFunctor::LabelT(), data_slice, d_data_slice, NULL,
          (unsigned char *)NULL,
          frontier_queue->keys[frontier_attribute->selector].GetPointer(
              util::DEVICE),  // d_in_queue
          frontier_queue->keys[frontier_attribute->selector ^ 1].GetPointer(
              util::DEVICE),  // d_out_queue
          (Value *)NULL, (Value *)NULL, frontier_attribute->queue_length,
          graph_slice->nodes, work_progress[0], context[0], stream,
          util::MaxValue<SizeT>(), util::MaxValue<SizeT>(),
          enactor_stats->filter_kernel_stats);

      frontier_attribute->selector ^= 1;
      frontier_attribute->queue_reset = false;
      frontier_attribute->queue_index++;

      if (enactor_stats->retval = work_progress->GetQueueLength(
              frontier_attribute->queue_index, frontier_attribute->queue_length,
              false, stream, true))
        return retval;
      //
      // launch filter, do pointer jumping for each froms[node]
      //
      while (!data_slice->stable_flag[0]) {
        data_slice->stable_flag[0] = 1;
        data_slice->stable_flag.Move(util::HOST, util::DEVICE, 1, 0, stream);
        gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                             PtrJumpFunctor>(
            enactor_stats[0], frontier_attribute[0],
            typename HookFunctor::LabelT(), data_slice, d_data_slice, NULL,
            (unsigned char *)NULL,
            frontier_queue->keys[frontier_attribute->selector ^ 1].GetPointer(
                util::DEVICE),  // d_in_queue
            frontier_queue->keys[frontier_attribute->selector].GetPointer(
                util::DEVICE),  // d_out_queue
            (Value *)NULL, (Value *)NULL, frontier_attribute->queue_length,
            graph_slice->nodes, work_progress[0], context[0], stream,
            util::MaxValue<SizeT>(), util::MaxValue<SizeT>(),
            enactor_stats->filter_kernel_stats);

        data_slice->stable_flag.Move(util::HOST, util::DEVICE, 1, 0, stream);
      }

      frontier_attribute->queue_reset = true;
      // initialize frontier as edge index using memsetidx
      //
      util::MemsetIdxKernel<<<128, 128>>>(
          frontier_queue->keys[0].GetPointer(util::DEVICE), graph_slice->edges);
      frontier_attribute->queue_length = graph_slice->edges;
      frontier_attribute->selector = 0;
      while (data_slice->stable_flag[0] == 0 ||
             enactor_stats->iteration < data_slice->max_iter) {
        data_slice->stable_flag[0] = 1;
        data_slice->stable_flag.Move(util::HOST, util::DEVICE, 1, 0, stream);
        //
        // filter for each edge use atomicAdd to accumulate weight_reg
        // weight_reg[label[v]] -= degrees[u]/2*num_edges
        gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                             WeightUpdateFunctor>(
            enactor_stats[0], frontier_attribute[0],
            typename HookFunctor::LabelT(), data_slice, d_data_slice, NULL,
            (unsigned char *)NULL,
            frontier_queue->keys[frontier_attribute->selector].GetPointer(
                util::DEVICE),  // d_in_queue
            (VertexId *)
                NULL,  // frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                       // // d_out_queue
            (Value *)NULL, (Value *)NULL, frontier_attribute->queue_length,
            graph_slice->nodes, work_progress[0], context[0], stream,
            util::MaxValue<SizeT>(), util::MaxValue<SizeT>(),
            enactor_stats->filter_kernel_stats, false);

        // this is simple. For each node n, final_weights[n] =
        // edge_weights[labels[tos[n]]]*weight_reg[labels[tos[n]]]
        frontier_attribute->queue_length = graph_slice->edges;
        gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                             AssignWeightFunctor>(
            enactor_stats[0], frontier_attribute[0],
            typename HookFunctor::LabelT(), data_slice, d_data_slice, NULL,
            (unsigned char *)NULL,
            frontier_queue->keys[frontier_attribute->selector].GetPointer(
                util::DEVICE),  // d_in_queue
            (VertexId *)
                NULL,  // frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                       // // d_out_queue
            (Value *)NULL, (Value *)NULL, frontier_attribute->queue_length,
            graph_slice->nodes, work_progress[0], context[0], stream,
            util::MaxValue<SizeT>(), util::MaxValue<SizeT>(),
            enactor_stats->filter_kernel_stats, false);
        //
        // reduce by key to find largest_weight for each node store into
        // reduced_weight use advance to choose argmax and store in
        // labels_argmax (can we use customized reduction op to compute argmax?)
        //
        cub::DeviceSegmentedReduce::ArgMax(
            d_temp_storage, temp_storage_bytes,
            data_slice->final_weights.GetPointer(util::DEVICE),
            data_slice->argmax_kv.GetPointer(util::DEVICE), graph_slice->nodes,
            (int *)data_slice->offsets.GetPointer(util::DEVICE),
            (int *)data_slice->offsets.GetPointer(util::DEVICE) + 1);

        // swap labels_argmax and labels (also return if all_label_stable,
        // simple kernel)
        frontier_attribute->queue_length = graph_slice->nodes;
        gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                             SwapLabelFunctor>(
            enactor_stats[0], frontier_attribute[0],
            typename HookFunctor::LabelT(), data_slice, d_data_slice, NULL,
            (unsigned char *)NULL,
            frontier_queue->keys[frontier_attribute->selector].GetPointer(
                util::DEVICE),  // d_in_queue
            (VertexId *)
                NULL,  // frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                       // // d_out_queue
            (Value *)NULL, (Value *)NULL, frontier_attribute->queue_length,
            graph_slice->nodes, work_progress[0], context[0], stream,
            util::MaxValue<SizeT>(), util::MaxValue<SizeT>(),
            enactor_stats->filter_kernel_stats, false);
        //
        // clear weight_reg to 1, edge_weights to 0
        util::MemsetKernel<<<256, 1024>>>(
            data_slice->edge_weights.GetPointer(util::DEVICE), 0.0f,
            graph_slice->nodes);
        util::MemsetKernel<<<256, 1024>>>(
            data_slice->weight_reg.GetPointer(util::DEVICE), 1.0f,
            graph_slice->nodes);

        data_slice->stable_flag.Move(util::DEVICE, util::HOST, 1, 0, stream);
        enactor_stats->iteration++;
      }

      if (d_scanned_edges) cudaFree(d_scanned_edges);
      if (retval) break;
    } while (0);

    if (this->debug) {
      printf("\nGPU Primitive Enact Done.\n");
    }

    return retval;
  }

  typedef gunrock::oprtr::filter::KernelPolicy<
      Problem,                          // Problem data type
      300,                              // CUDA_ARCH
      0,                                // SATURATION QUIT
      true,                             // DEQUEUE_PROBLEM_SIZE
      (sizeof(VertexId) == 4) ? 8 : 4,  // MIN_CTA_OCCUPANCY
      8,                                // LOG_THREADS
      2,                                // LOG_LOAD_VEC_SIZE
      0,                                // LOG_LOADS_PER_TILE
      5,                                // LOG_RAKING_THREADS
      5,                                // END_BITMASK_CULL
      8,                                // LOG_SCHEDULE_GRANULARITY
      gunrock::oprtr::filter::CULL>
      FilterKernelPolicy;

  typedef gunrock::oprtr::advance::KernelPolicy<
      Problem,  // Problem data type
      300,      // CUDA_ARCH
      // INSTRUMENT,         // INSTRUMENT
      8,         // MIN_CTA_OCCUPANCY
      10,        // LOG_THREADS
      9,         // LOG_BLOCKS
      32 * 128,  // LIGHT_EDGE_THRESHOLD
      1,         // LOG_LOAD_VEC_SIZE
      0,         // LOG_LOADS_PER_TILE
      5,         // LOG_RAKING_THREADS
      32,        // WARP_GATHER_THRESHOLD
      128 * 4,   // CTA_GATHER_THRESHOLD
      7,         // LOG_SCHEDULE_GRANULARITY
      gunrock::oprtr::advance::LB_CULL>
      AdvanceKernelPolicy;

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief Reset enactor
   *
   * \return cudaError_t object Indicates the success of all CUDA calls.
   */
  cudaError_t Reset() { return BaseEnactor::Reset(); }

  /**
   * @brief Primitive enact kernel initaliization.
   *
   * @param[in] context CudaContext pointer for ModernGPU APIs.
   * @param[in] problem Pointer to Problem object.
   * @param[in] max_grid_size Max grid size for kernel calls.
   *
   * \return cudaError_t object indicates the success of all CUDA functions.
   */
  cudaError_t Init(ContextPtr *context, Problem *problem,
                   int max_grid_size = 0) {
    int min_sm_version = -1;
    for (int i = 0; i < this->num_gpus; i++) {
      if (min_sm_version == -1 ||
          this->cuda_props[i].device_sm_version < min_sm_version) {
        min_sm_version = this->cuda_props[i].device_sm_version;
      }
    }
    cudaError_t ret = cudaSuccess;

    if (min_sm_version >= 300) {
      ret = InitLp<AdvanceKernelPolicy, FilterKernelPolicy>(context, problem,
                                                            max_grid_size);
      return ret;
    }

    // to reduce compile time, get rid of other architecture for now
    // TODO: add all the kernel policy setting for all architectures

    printf("Not yet tuned for this architecture.\n");
    return cudaErrorInvalidDeviceFunction;
  }

  /**
   * @brief Primitive enact kernel entry.
   *
   * \return cudaError_t object indicates the success of all CUDA functions.
   */
  cudaError_t Enact()
  // ContextPtr context,
  // Problem*   problem,
  // int        max_grid_size = 0)
  {
    int min_sm_version = -1;
    for (int i = 0; i < this->num_gpus; i++) {
      if (min_sm_version == -1 ||
          this->cuda_props[i].device_sm_version < min_sm_version) {
        min_sm_version = this->cuda_props[i].device_sm_version;
      }
    }

    if (min_sm_version >= 300) {
      return EnactLp<AdvanceKernelPolicy, FilterKernelPolicy>();
      // context, problem, max_grid_size);
    }

    // to reduce compile time, get rid of other architecture for now
    // TODO: add all the kernel policy setting for all architectures

    printf("Not yet tuned for this architecture.\n");
    return cudaErrorInvalidDeviceFunction;
  }

  /** @} */
};

}  // namespace lp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
