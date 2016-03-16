// -----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// -----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// -----------------------------------------------------------------------------

/**
 * @file
 * tc_enactor.cuh
 *
 * @brief Problem enactor for Triangle Counting
 */

#pragma once

#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/intersection/kernel.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/global_indicator/tc/tc_problem.cuh>
#include <gunrock/global_indicator/tc/tc_functor.cuh>

#include <moderngpu.cuh>
#include <cub/cub.cuh>

namespace gunrock {
namespace global_indicator {
namespace tc {

using namespace gunrock::app;
using namespace mgpu;
using namespace cub;

/**
 * @brief TC enactor class.
 *
 * @tparam _Problem
 * @tparam _INSTRUMWENT
 * @tparam _DEBUG
 * @tparam _SIZE_CHECK
 */
template <
  typename _Problem,
  bool _INSTRUMENT,
  bool _DEBUG,
  bool _SIZE_CHECK >
class TCEnactor :
  public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK> {
 protected:

 public:
   typedef _Problem                    TCProblem;
   typedef typename TCProblem::SizeT       SizeT;
   typedef typename TCProblem::VertexId VertexId;
   typedef typename TCProblem::Value       Value;
   static const bool INSTRUMENT   =   _INSTRUMENT;
   static const bool DEBUG        =        _DEBUG;
   static const bool SIZE_CHECK   =   _SIZE_CHECK;

  /**
   * @brief TCEnactor constructor.
   */
  TCEnactor(int *gpu_idx) :
    EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>(
      EDGE_FRONTIERS, 1, gpu_idx)
  {
  }

  /**
   * @brief TCEnactor destructor
   */
  virtual ~TCEnactor()
  {
  }

  /**
   * @brief Enacts a TC computing on the specified graph.
   *
   * @tparam Advance Kernel policy for forward advance kernel.
   * @tparam Filter Kernel policy for filter kernel.
   * @tparam Intersection Kernel policy for intersection kernel.
   * @tparam TCProblem TC Problem type.
   *
   * @param[in] context CudaContext for moderngpu library
   * @param[in] problem TCProblem object.
   * @param[in] max_grid_size Max grid size for TC kernel calls.
   *
   * \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename IntersectionKernelPolicy,
    typename TCProblem>
  cudaError_t EnactTC(
    ContextPtr  context,
    TCProblem* problem,
    int         max_grid_size = 0)
  {
    typedef typename TCProblem::VertexId VertexId;
    typedef typename TCProblem::SizeT    SizeT;
    typedef typename TCProblem::Value    Value;

    typedef TCFunctor <VertexId, SizeT, Value, TCProblem> TCFunctor;

    cudaError_t retval = cudaSuccess;
    SizeT *d_scanned_edges = NULL;  // Used for LB

    FrontierAttribute<SizeT>* attributes = &this->frontier_attribute[0];
    EnactorStats* statistics = &this->enactor_stats[0];
    typename TCProblem::DataSlice* data_slice = problem->data_slices[0];
    util::DoubleBuffer<SizeT, VertexId, Value>*
      queue = &data_slice->frontier_queues[0];
    util::CtaWorkProgressLifetime*
      work_progress = &this->work_progress[0];
    cudaStream_t stream = data_slice->streams[0];

    do
    {
      // initialization
      if (retval = EnactorBase<
        typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>::Setup(
          problem,
          max_grid_size,
          AdvanceKernelPolicy::CTA_OCCUPANCY,
          FilterKernelPolicy::CTA_OCCUPANCY)) break;

      // single-GPU graph slice
      GraphSlice<SizeT,VertexId,Value>* graph_slice = problem->graph_slices[0];
      typename TCProblem::DataSlice* d_data_slice = problem->d_data_slices[0];

      if (retval = util::GRError(cudaMalloc(
        (void**)&d_scanned_edges, graph_slice->edges * sizeof(SizeT)),
        "Problem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__))
      {
        return retval;
      }

      attributes->queue_index = 0;
      attributes->selector = 0;
      attributes->queue_length = graph_slice->nodes;
      attributes->queue_reset = true; 

      // TODO: Add TC algorithm here.
      
      // Prepare src node_ids for edge list
      // TODO: move this to problem. need to send CudaContext to problem too. 

      // 1) Do advance/filter to get rid of neighbors with a smaller #ofdegree.
      gunrock::oprtr::advance::LaunchKernel
      <AdvanceKernelPolicy, TCProblem, TCFunctor>(
      statistics[0],
      attributes[0],
      d_data_slice,
      (VertexId*)NULL,
      (bool*)NULL,
      (bool*)NULL,
      d_scanned_edges,
      queue->keys[attributes->selector].GetPointer(util::DEVICE),
      queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
      (VertexId*)NULL,
      (VertexId*)NULL,
      graph_slice->row_offsets.GetPointer(util::DEVICE),
      graph_slice->column_indices.GetPointer(util::DEVICE),
      (SizeT*)NULL,
      (VertexId*)NULL,
      graph_slice->nodes,
      graph_slice->edges,
      work_progress[0],
      context[0],
      stream,
      gunrock::oprtr::advance::V2V);

      //SegReduce
      SegReduceCsr(
        data_slice->d_src_node_ids.GetPointer(util::DEVICE),
        graph_slice->row_offsets.GetPointer(util::DEVICE),
        graph_slice->edges,
        graph_slice->nodes,
        false,
        data_slice->d_edge_list.GetPointer(util::DEVICE),
        (int)0,
        mgpu::plus<int>(),
        context[0]);

      //Scan
      Scan<mgpu::MgpuScanTypeExc>(
        data_slice->d_edge_list.GetPointer(util::DEVICE),
        graph_slice->nodes+1,
        (int)0,
        mgpu::plus<int>(),
        (int*)0,
        (int*)0,
        graph_slice->row_offsets.GetPointer(util::DEVICE),
        context[0]);

      IntervalExpand(
            graph_slice->edges/2,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            graph_slice->nodes,
            data_slice->d_src_node_ids.GetPointer(util::DEVICE),
            context[0]);
      
      if (retval = work_progress->GetQueueLength(++attributes->queue_index, attributes->queue_length, false, stream, true)) return retval;
      attributes->selector ^= 1;

      //Filter to get edge_list (done)
      //declare edge_list in problem (done)
      //modify intersection operator (done)
      //cubPartition the coarse_count is on device, need to change (done)
      gunrock::oprtr::filter::Kernel<FilterKernelPolicy, TCProblem, TCFunctor>
      <<<statistics->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
      statistics->iteration,
      attributes->queue_reset,
      attributes->queue_index,
      attributes->queue_length,
      queue->keys[attributes->selector].GetPointer(util::DEVICE),
      NULL,
      graph_slice->column_indices.GetPointer(util::DEVICE),
      d_data_slice,
      NULL,
      work_progress[0],
      graph_slice->edges,
      graph_slice->edges,
      statistics->filter_kernel_stats);

      graph_slice->edges /= 2;

      //GetQueueLength of the new edge_list
      if (retval = work_progress->GetQueueLength(++attributes->queue_index, attributes->queue_length, false, stream, true)) return retval;

      printf("queue length:%d\n", attributes->queue_length);


      util::MemsetMadVectorKernel<<<256, 2014>>>(
              data_slice->d_degrees.GetPointer(util::DEVICE),
              graph_slice->row_offsets.GetPointer(util::DEVICE),
              graph_slice->row_offsets.GetPointer(util::DEVICE)+1, -1, graph_slice->nodes);
      // 2) Do intersection using generated edge lists from the previous step.
      //gunrock::oprtr::intersection::LaunchKernel
      //<IntersectionKernelPolicy, TCProblem, TCFunctor>(
      //);
      // Reuse d_scanned_edges
      SizeT *d_output_counts = d_scanned_edges;

      util::MemsetKernel<<<256, 1024>>>(d_output_counts, (SizeT)0, attributes->queue_length);

      // Should make tc_count a member var to TCProblem
      long tc_count = gunrock::oprtr::intersection::LaunchKernel
      <IntersectionKernelPolicy, TCProblem, TCFunctor>(
      statistics[0],
      attributes[0],
      d_data_slice,
      graph_slice->row_offsets.GetPointer(util::DEVICE),
      graph_slice->column_indices.GetPointer(util::DEVICE),
      data_slice->d_src_node_ids.GetPointer(util::DEVICE),
      graph_slice->column_indices.GetPointer(util::DEVICE),
      data_slice->d_degrees.GetPointer(util::DEVICE),
      d_output_counts,
      attributes->queue_length,
      graph_slice->nodes,
      graph_slice->edges,
      work_progress[0],
      context[0],
      stream);

      //tc_count /= 3;

      printf("tc count:%ld\n", tc_count);

      // end of the TC

      if (d_scanned_edges) cudaFree(d_scanned_edges);
      if (retval) break;

    } while(0);
    return retval;
  }

  /**
   * @brief MST Enact kernel entry.
   *
   * @tparam MSTProblem MST Problem type. @see MSTProblem
   *
   * @param[in] context CudaContext pointer for ModernGPU APIs.
   * @param[in] problem Pointer to Problem object.
   * @param[in] max_grid_size Max grid size for kernel calls.
   *
   * \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  template <typename TCProblem, int NL_SIZE>
  cudaError_t Enact(
    ContextPtr  context,
    TCProblem* problem,
    int         max_grid_size = 0)
  {
    int min_sm_version = -1;
    for (int i = 0; i < this->num_gpus; i++)
    {
      if (min_sm_version == -1 ||
        this->cuda_props[i].device_sm_version < min_sm_version)
      {
        min_sm_version = this->cuda_props[i].device_sm_version;
      }
    }

    if (min_sm_version >= 300)
    {
      typedef gunrock::oprtr::filter::KernelPolicy<
        TCProblem,         // Problem data type
        300,                // CUDA_ARCH
        INSTRUMENT,         // INSTRUMENT
        0,                  // SATURATION QUIT
        true,               // DEQUEUE_PROBLEM_SIZE
        8,                  // MIN_CTA_OCCUPANCY
        8,                  // LOG_THREADS
        1,                  // LOG_LOAD_VEC_SIZE
        0,                  // LOG_LOADS_PER_TILE
        5,                  // LOG_RAKING_THREADS
        5,                  // END_BITMASK_CULL
        8>                  // LOG_SCHEDULE_GRANULARITY
        FilterKernelPolicy;

      typedef gunrock::oprtr::advance::KernelPolicy<
        TCProblem,         // Problem data type
        300,                // CUDA_ARCH
        INSTRUMENT,         // INSTRUMENT
        8,                  // MIN_CTA_OCCUPANCY
        10,                 // LOG_THREADS
        8,                  // LOG_BLOCKS
        32 * 128,           // LIGHT_EDGE_THRESHOLD
        1,                  // LOG_LOAD_VEC_SIZE
        0,                  // LOG_LOADS_PER_TILE
        5,                  // LOG_RAKING_THREADS
        32,                 // WARP_GATHER_THRESHOLD
        128 * 4,            // CTA_GATHER_THRESHOLD
        7,                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
        AdvanceKernelPolicy;

      typedef gunrock::oprtr::intersection::KernelPolicy<
        TCProblem,         // Problem data type
        300,                // CUDA_ARCH
        INSTRUMENT,         // INSTRUMENT
        1,                  // MIN_CTA_OCCUPANCY
        10,                 // LOG_THREADS
        8,                  // LOG_BLOCKS
        NL_SIZE>                  // NL_SIZE_THRESHOLD
        IntersectionKernelPolicy;

      return EnactTC<AdvanceKernelPolicy, FilterKernelPolicy, IntersectionKernelPolicy,
        TCProblem>(context, problem, max_grid_size);
    }

    // to reduce compile time, get rid of other architectures for now
    // TODO: add all the kernel policy settings for all architectures

    printf("Not yet tuned for this architecture\n");
    return cudaErrorInvalidDeviceFunction;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /** @} */

};

} // namespace tc
} // namespace global_indicator
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
