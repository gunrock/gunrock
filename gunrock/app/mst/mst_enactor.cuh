// -----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// -----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// -----------------------------------------------------------------------------

/**
 * @file
 * mst_enactor.cuh
 *
 * @brief Problem enactor for Minimum Spanning Tree
 */

#pragma once

#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/sort_utils.cuh>
#include <gunrock/util/select_utils.cuh>
#include <gunrock/util/mark_segment.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/mst/mst_problem.cuh>
#include <gunrock/app/mst/mst_functor.cuh>

#include <moderngpu.cuh>
#include <limits>

namespace gunrock {
namespace app {
namespace mst {

using namespace mgpu;

/**
 * @brief MST enactor class.
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
class MSTEnactor :
  public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK> {
 protected:

  /**
   * A pinned, mapped word that the traversal kernels will signal when done
   */
  int *vertex_flag;

  /**
   * @brief Prepare the enactor for MST kernel call.
   * Must be called prior to each MST iteration.
   *
   * @param[in] problem MST Problem object which holds the graph data and
   * MST problem data to compute.
   *
   * \return cudaError_t object which indicates the success of all CUDA calls.
   */
  template <typename ProblemData>
  cudaError_t Setup(ProblemData *problem)
  {
    typedef typename ProblemData::Value    Value;
    typedef typename ProblemData::VertexId VertexId;
    typedef typename ProblemData::SizeT    SizeT;

    cudaError_t retval = cudaSuccess;

    do
    {
      // Graph slice
      // GraphSlice<SizeT, VertexId, Value>*
      //   graph_slice = problem->graph_slices[0];
      // Data slice
      // typename ProblemData::DataSlice*
      //   data_slice = problem->data_slices[0];
    } while (0);

    return retval;
  }

 public:
   typedef _Problem                    MSTProblem;
   typedef typename MSTProblem::SizeT       SizeT;
   typedef typename MSTProblem::VertexId VertexId;
   typedef typename MSTProblem::Value       Value;
   static const bool INSTRUMENT   =   _INSTRUMENT;
   static const bool DEBUG        =        _DEBUG;
   static const bool SIZE_CHECK   =   _SIZE_CHECK;

  /**
   * @brief MSTEnactor constructor.
   */
  MSTEnactor(int *gpu_idx) :
    EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>(
      EDGE_FRONTIERS, 1, gpu_idx)
  {
    vertex_flag = new int;
    vertex_flag[0] = 0;
  }

  /**
   * @brief MSTEnactor destructor
   */
  virtual ~MSTEnactor()
  {
    if (vertex_flag) delete vertex_flag;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @ brief Obtain statistics about the last MST search enacted.
   *
   * @ param[out] total_queued Total queued elements in MST kernel.
   * @ param[out] search_depth Search depth of MST algorithm.
   * @ param[out] avg_duty Average kernel running duty (kernel run time / kernel lifetime).
   * spaces between @ and name are to eliminate doxygen warnings
   */
  /*template <typename VertexId>
  void GetStatistics(
    long long &total_queued,
    VertexId  &search_depth,
    double    &avg_duty)
  {
    cudaDeviceSynchronize();

    total_queued = this->enactor_stats->total_queued[0];
    search_depth = this->enactor_stats->iteration;

    avg_duty = (this->enactor_stats->total_lifetimes >0) ?
      double(this->enactor_stats->total_runtimes) /
        this->enactor_stats->total_lifetimes : 0.0;
  }*/

  /** @} */

  /**
   * @brief Enacts a MST computing on the specified graph.
   *
   * @tparam Advance Kernel policy for forward advance kernel.
   * @tparam Filter Kernel policy for filter kernel.
   * @tparam MSTProblem MST Problem type.
   *
   * @param[in] context CudaContext for ModernGPU library
   * @param[in] problem MSTProblem object.
   * @param[in] max_grid_size Max grid size for MST kernel calls.
   *
   * \return cudaError_t object which indicates the success of
   * all CUDA function calls.
   */
  template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename MSTProblem>
  cudaError_t EnactMST(
    ContextPtr  context,
    MSTProblem* problem,
    int         max_grid_size = 0)
  {
    typedef typename MSTProblem::VertexId VertexId;
    typedef typename MSTProblem::SizeT    SizeT;
    typedef typename MSTProblem::Value    Value;

    typedef SuccFunctor <VertexId, SizeT, VertexId, MSTProblem> SuccFunctor;
    typedef EdgeFunctor <VertexId, SizeT, VertexId, MSTProblem> EdgeFunctor;
    typedef CyRmFunctor <VertexId, SizeT, VertexId, MSTProblem> CyRmFunctor;
    typedef PJmpFunctor <VertexId, SizeT, VertexId, MSTProblem> PJmpFunctor;
    typedef EgRmFunctor <VertexId, SizeT, VertexId, MSTProblem> EgRmFunctor;
    typedef RIdxFunctor <VertexId, SizeT, VertexId, MSTProblem> RIdxFunctor;
    typedef SuRmFunctor <VertexId, SizeT, VertexId, MSTProblem> SuRmFunctor;
    typedef EIdxFunctor <VertexId, SizeT, VertexId, MSTProblem> EIdxFunctor;
    typedef MarkFunctor <VertexId, SizeT, VertexId, MSTProblem> MarkFunctor;

    cudaError_t retval = cudaSuccess;
    SizeT *d_scanned_edges = NULL;  // Used for LB

    FrontierAttribute<SizeT>* attributes = &this->frontier_attribute[0];
    EnactorStats* statistics = &this->enactor_stats[0];
    typename MSTProblem::DataSlice* data_slice = problem->data_slices[0];
    util::DoubleBuffer<SizeT, VertexId, Value>*
      queue = &data_slice->frontier_queues[0];
    util::CtaWorkProgressLifetime*
      work_progress = &this->work_progress[0];
    cudaStream_t stream = data_slice->streams[0];

    do
    {
      // initialization
      if (retval = Setup(problem)) break;
      if (retval = EnactorBase<
        typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>::Setup(
          problem,
          max_grid_size,
          AdvanceKernelPolicy::CTA_OCCUPANCY,
          FilterKernelPolicy::CTA_OCCUPANCY)) break;

      // single-GPU graph slice
      GraphSlice<SizeT,VertexId,Value>* graph_slice = problem->graph_slices[0];
      typename MSTProblem::DataSlice* d_data_slice = problem->d_data_slices[0];

      if (retval = util::GRError(cudaMalloc(
        (void**)&d_scanned_edges, graph_slice->edges * sizeof(SizeT)),
        "Problem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__))
      {
        return retval;
      }

      // debug configurations
      //SizeT num_edges_origin = graph_slice->edges;
      bool debug_info = 0;   // used for debug purpose
      //int tmp_select  = 0; // used for debug purpose
      //int tmp_length  = 0; // used for debug purpose
      unsigned int *num_selected = new unsigned int; // used in cub select

      //////////////////////////////////////////////////////////////////////////
      // recursive Loop for minimum spanning tree implementation
      while (graph_slice->nodes > 1)  // more than ONE super-vertex
      {
        if (DEBUG)
        {
          printf("\nBEGIN ITERATION: %lld #NODES: %d #EDGES: %d\n",
            statistics->iteration+1,graph_slice->nodes,graph_slice->edges);
        }

        if (debug_info)
        {
          printf(":: initial read in row_offsets ::");
          util::DisplayDeviceResults(
           graph_slice->row_offsets.GetPointer(util::DEVICE),
           graph_slice->nodes + 1);
        }

        // generate flag_array from d_row_offsets using MarkSegment kernel
        util::MarkSegmentFromIndices<<<128, 128>>>(
          problem->data_slices[0]->flag_array.GetPointer(util::DEVICE),
          graph_slice->row_offsets.GetPointer(util::DEVICE),
          graph_slice->nodes);

        if (DEBUG) printf("* finished mark segmentation >> flag_array.\n");

        // generate d_keys_array from flag_array using sum inclusive scan
        Scan<MgpuScanTypeInc>(
          (int*)problem->data_slices[0]->flag_array.GetPointer(util::DEVICE),
          graph_slice->edges, (int)0, mgpu::plus<int>(), (int*)0, (int*)0,
          (int*)problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
          context[0]);

        if (DEBUG) printf("* finished segmented sum scan >> d_keys_array.\n");
        if (DEBUG) printf("A. MARKING THE MST EDGES ...\n");
        if (DEBUG) printf(" a. Finding Minimum Weighted Edges\n");

        ////////////////////////////////////////////////////////////////////////
        // each vertex u finds the minimum weighted edge to another vertex v
        // select minimum edge_weights and keys using mgpu::ReduceByKey
        int num_segments;
        ReduceByKey(
          problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
          problem->data_slices[0]->edge_value.GetPointer(util::DEVICE),
          graph_slice->edges,
          std::numeric_limits<Value>::max(),
          mgpu::minimum<Value>(),
          mgpu::equal_to<Value>(),
          problem->data_slices[0]->reduce_key.GetPointer(util::DEVICE),
          problem->data_slices[0]->reduce_val.GetPointer(util::DEVICE),
          &num_segments, (int*)0, context[0]);

        if (DEBUG) printf("  * finished segmented reduction: keys & weight.\n");

        if (debug_info)
        {
          printf(":: origin flag_array ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->flag_array.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: origin d_keys_array ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: origin d_col_indices ::");
          util::DisplayDeviceResults(
            graph_slice->column_indices.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: origin d_edge_weights ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->edge_value.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: reduced keys array - d_reduced_keys ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->reduce_key.GetPointer(util::DEVICE),
            num_segments);
          printf(":: reduced edge weights - d_reduced_vals ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->reduce_val.GetPointer(util::DEVICE),
            num_segments);
        }

        if (DEBUG) printf(" (b). Finding and Removing Cycles.\n");

        ////////////////////////////////////////////////////////////////////////
        // generate successor array using SuccFunctor - advance
        // successor array holds the outgoing v for each u
        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset  = true;

        util::MemsetKernel<<<128, 128>>>(
          problem->data_slices[0]->successors.GetPointer(util::DEVICE),
          std::numeric_limits<int>::max(),
          graph_slice->nodes);
        util::MemsetKernel<<<128, 128>>>(
          problem->data_slices[0]->temp_index.GetPointer(util::DEVICE),
          std::numeric_limits<VertexId>::max(),
          graph_slice->nodes);
        util::MemsetIdxKernel<<<128, 128>>>(
          queue->keys[attributes->selector].GetPointer(util::DEVICE),
          graph_slice->nodes);

        gunrock::oprtr::advance::LaunchKernel
          <AdvanceKernelPolicy, MSTProblem, SuccFunctor>(
          statistics[0],
          attributes[0],
          d_data_slice,
          (VertexId*)NULL,
          (bool*)NULL,
          (bool*)NULL,
          d_scanned_edges,
          queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
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

        if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
          "advance::Kernel failed", __FILE__, __LINE__))) break;

        if (DEBUG) printf("  * finished min weighted edges >> successors.\n");

        ////////////////////////////////////////////////////////////////////////
        // finding original edge ids with the corresponding d_id
        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset  = true;

        gunrock::oprtr::advance::LaunchKernel
          <AdvanceKernelPolicy, MSTProblem, EdgeFunctor>(
          statistics[0],
          attributes[0],
          d_data_slice,
          (VertexId*)NULL,
          (bool*)NULL,
          (bool*)NULL,
          d_scanned_edges,
          queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
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

        if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
          "advance::Kernel failed", __FILE__, __LINE__))) break;

        ////////////////////////////////////////////////////////////////////////
        // mark MST output edges
        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset  = true;

        gunrock::oprtr::advance::LaunchKernel
          <AdvanceKernelPolicy, MSTProblem, MarkFunctor>(
          statistics[0],
          attributes[0],
          d_data_slice,
          (VertexId*)NULL,
          (bool*)NULL,
          (bool*)NULL,
          d_scanned_edges,
          queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
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
          gunrock::oprtr::advance::V2E);

        if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
          "advance::Kernel failed", __FILE__, __LINE__))) break;

        ////////////////////////////////////////////////////////////////////////
        // remove cycles - vertices with S(S(u)) = u forms cycles
        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset  = true;

        gunrock::oprtr::advance::LaunchKernel
          <AdvanceKernelPolicy, MSTProblem, CyRmFunctor>(
          statistics[0],
          attributes[0],
          d_data_slice,
          (VertexId*)NULL,
          (bool*)NULL,
          (bool*)NULL,
          d_scanned_edges,
          queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
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
          gunrock::oprtr::advance::V2E);

        if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
          "advance::Kernel failed", __FILE__, __LINE__))) break;

        if (DEBUG) printf("  * finished removing cycles >> new successors.\n");

        if (debug_info)
        {
          printf(":: remove cycles from successors ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->successors.GetPointer(util::DEVICE),
            graph_slice->nodes);
        }

        if (DEBUG) printf("B. GRAPH CONSTRUCTION ...\n");
        if (DEBUG) printf(" (a). Merging Vertices\n");

        ////////////////////////////////////////////////////////////////////////
        // Then, we combine vertices to form a super-vertex by employing
        // pointer doubling to achieve this result, iteratively setting
        // S(u) = S(S(u)) until no further change occurs in S
        // using filter kernel: PJmpFunctor
        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset  = true;

        vertex_flag[0] = 0;
        while (!vertex_flag[0])
        {
          vertex_flag[0] = 1;
          problem->data_slices[0]->done_flags.SetPointer(vertex_flag);
          if (retval = problem->data_slices[0]->done_flags.Move(
            util::HOST, util::DEVICE)) return retval;

          gunrock::oprtr::filter::LaunchKernel
            <FilterKernelPolicy, MSTProblem, PJmpFunctor>(
            statistics->filter_grid_size,
            FilterKernelPolicy::THREADS, 
            0, stream,
            statistics->iteration + 1,
            attributes->queue_reset,
            attributes->queue_index,
            attributes->queue_length,
            queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
            NULL,
            queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
            d_data_slice,
            NULL,
            work_progress[0],
            queue->keys[attributes->selector  ].GetSize(),
            queue->keys[attributes->selector^1].GetSize(),
            statistics->filter_kernel_stats);

          if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
            "filter PointerJumping failed", __FILE__, __LINE__))) break;

          // prepare for next iteration, only reset once
          if (attributes->queue_reset)
          {
            attributes->queue_reset = false;
          }
          attributes->queue_index++;
          attributes->selector ^= 1;

          problem->data_slices[0]->done_flags.SetPointer(vertex_flag);
          if (retval = problem->data_slices[0]->done_flags.Move(
            util::DEVICE, util::HOST)) return retval;

          // check if finished pointer jumping
          if (vertex_flag[0]) break;
        }

        if (DEBUG) printf("  * finished pointer doubling: representatives.\n");
        if (DEBUG) printf(" (b).Assigning IDs to Super-vertices\n");

        ////////////////////////////////////////////////////////////////////////
        // each vertex of a super-vertex now has a representative, but the
        // super-vertices are not numbered in order. The vertices assigned
        // to a super-vertex are also not placed in order in the successor

        // bring all vertices of a super-vertex together by sorting
        util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->super_idxs.GetPointer(util::DEVICE),
          problem->data_slices[0]->successors.GetPointer(util::DEVICE),
          graph_slice->nodes);

        util::MemsetIdxKernel<<<128, 128>>>(
        problem->data_slices[0]->original_n.GetPointer(util::DEVICE),
        graph_slice->nodes);

        util::CUBRadixSort<VertexId, VertexId>(
          true, graph_slice->nodes,
          problem->data_slices[0]->super_idxs.GetPointer(util::DEVICE),
          problem->data_slices[0]->original_n.GetPointer(util::DEVICE));

        if (debug_info)
        {
          printf(":: pointer jumping: representatives ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->successors.GetPointer(util::DEVICE),
            graph_slice->nodes);
          printf(":: bring all vertices of a super-vertex together ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->super_idxs.GetPointer(util::DEVICE),
            graph_slice->nodes);
        }

        ////////////////////////////////////////////////////////////////////////
        // create a flag to mark the boundaries of representative vertices
        util::MarkSegmentFromKeys<<<128, 128>>>(
          problem->data_slices[0]->flag_array.GetPointer(util::DEVICE),
          problem->data_slices[0]->super_idxs.GetPointer(util::DEVICE),
          graph_slice->nodes);

        if (DEBUG) printf("  * finished mark super-vertices: super flags.\n");

        ////////////////////////////////////////////////////////////////////////
        // sum scan of the super flags to assign new super-vertex ids
        Scan<MgpuScanTypeInc>(
          (int*)problem->data_slices[0]->flag_array.GetPointer(util::DEVICE),
          graph_slice->nodes, (int)0, mgpu::plus<int>(), (int*)0, (int*)0,
          (int*)problem->data_slices[0]->super_idxs.GetPointer(util::DEVICE),
          context[0]);

        if (DEBUG) printf("  * finished assign super ids:   .\n");

        if (debug_info)
        {
          printf(":: super flags (a.k.a. c flag) ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->flag_array.GetPointer(util::DEVICE),
            graph_slice->nodes);
          printf(":: new assigned super-vertex ids ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->super_idxs.GetPointer(util::DEVICE),
            graph_slice->nodes);
        }

        ////////////////////////////////////////////////////////////////////////
        // used for finding super-vertex ids for next iteration
        util::CUBRadixSort<VertexId, VertexId>(
          true, graph_slice->nodes,
          problem->data_slices[0]->original_n.GetPointer(util::DEVICE),
          problem->data_slices[0]->super_idxs.GetPointer(util::DEVICE));

        ////////////////////////////////////////////////////////////////////////
        // update graph_slice->nodes with number of super-vertices
        SizeT current_nodes = graph_slice->nodes;
        // the first segment in flag was set to 0 instead of 1
        util::MemsetKernel<unsigned int><<<1, 1>>>(
          problem->data_slices[0]->flag_array.GetPointer(util::DEVICE), 1, 1);
        graph_slice->nodes = Reduce(
          problem->data_slices[0]->flag_array.GetPointer(util::DEVICE),
          graph_slice->nodes, context[0]);

        if (DEBUG)
          printf("  * finished update #nodes: %d left.\n", graph_slice->nodes);

        // terminate the loop if there is only one super-vertex left
        if (graph_slice->nodes == 1)
        {
          if (DEBUG) printf("\nTERMINATE THE MST ALGORITHM ENACTOR.\n\n");
          break;  // break the MST recursive loop
        }

        if (DEBUG) printf(" (c). Removing Edges & Forming the new Edge List\n");

        ////////////////////////////////////////////////////////////////////////
        // shorten the edge list by removing self edges in the new graph
        // advance kernel remove edges belonging to the same super-vertex
        // each edge examines the super-vertex id of both end vertices and
        // removes itself if the id is the same
        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = current_nodes;
        attributes->queue_reset  = true;

        gunrock::oprtr::advance::LaunchKernel
          <AdvanceKernelPolicy, MSTProblem, EgRmFunctor>(
          statistics[0],
          attributes[0],
          d_data_slice,
          (VertexId*)NULL,
          (bool*)NULL,
          (bool*)NULL,
          d_scanned_edges,
          queue->keys[attributes->selector  ].GetPointer(util::DEVICE),
          queue->keys[attributes->selector^1].GetPointer(util::DEVICE),
          (VertexId*)NULL,
          (VertexId*)NULL,
          graph_slice->row_offsets.GetPointer(util::DEVICE),
          graph_slice->column_indices.GetPointer(util::DEVICE),
          (SizeT*)NULL,
          (VertexId*)NULL,
          current_nodes,
          graph_slice->edges,
          work_progress[0],
          context[0],
          stream,
          gunrock::oprtr::advance::V2E);

        if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
          "advance::Kernel failed", __FILE__, __LINE__))) break;

        if (DEBUG) printf("  * finished mark edges in same super-vertex.\n");

        ////////////////////////////////////////////////////////////////////////
        // filter to remove all -1 in d_col_indices
        util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->temp_index.GetPointer(util::DEVICE),
          problem->data_slices[0]->colindices.GetPointer(util::DEVICE),
          graph_slice->edges);
        util::CUBSelect<VertexId, SizeT>(
          problem->data_slices[0]->temp_index.GetPointer(util::DEVICE),
          graph_slice->edges,
          problem->data_slices[0]->colindices.GetPointer(util::DEVICE),
          num_selected);

        ////////////////////////////////////////////////////////////////////////
        // filter to remove all -1 in d_edge_weights
        util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->temp_value.GetPointer(util::DEVICE),
          problem->data_slices[0]->edge_value.GetPointer(util::DEVICE),
          graph_slice->edges);
        util::CUBSelect<Value, SizeT>(
          problem->data_slices[0]->temp_value.GetPointer(util::DEVICE),
          graph_slice->edges,
          problem->data_slices[0]->edge_value.GetPointer(util::DEVICE),
          num_selected);

        ////////////////////////////////////////////////////////////////////////
        // filter to remove all -1 in d_keys_array
        util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->temp_index.GetPointer(util::DEVICE),
          problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
          graph_slice->edges);
        util::CUBSelect<VertexId, SizeT>(
          problem->data_slices[0]->temp_index.GetPointer(util::DEVICE),
          graph_slice->edges,
          problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
          num_selected);

        ////////////////////////////////////////////////////////////////////////
        // filter to remove all -1 in d_origin_edges
        util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->temp_index.GetPointer(util::DEVICE),
          problem->data_slices[0]->original_e.GetPointer(util::DEVICE),
          graph_slice->edges);
        util::CUBSelect<VertexId, SizeT>(
          problem->data_slices[0]->temp_index.GetPointer(util::DEVICE),
          graph_slice->edges,
          problem->data_slices[0]->original_e.GetPointer(util::DEVICE),
          num_selected);

        if (DEBUG) printf("  * finished remove edges in one super-vertex.\n");

        ////////////////////////////////////////////////////////////////////////
        // update edge list length in graph_slice [1]
        graph_slice->edges = *num_selected;

        if (DEBUG) printf("  * finished update #edge: %d\n",graph_slice->edges);

        if (debug_info)
        {
          printf(":: edge removal in one super-vertex (d_keys_array) ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: edge removal in one super-vertex (d_col_indices) ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->colindices.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: edge removal in one super-vertex (d_edge_weights) ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->edge_value.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: edge removal in one super-vertex (d_origin_edges) ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->original_e.GetPointer(util::DEVICE),
            graph_slice->edges);
        }

        ////////////////////////////////////////////////////////////////////////
        // find super-vertex ids for d_keys_array and d_col_indices
        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = graph_slice->edges;
        attributes->queue_reset  = true;

        gunrock::oprtr::filter::LaunchKernel
          <FilterKernelPolicy, MSTProblem, EgRmFunctor>(
          statistics->filter_grid_size,
          FilterKernelPolicy::THREADS, 
          0, stream,
          statistics->iteration + 1,
          attributes->queue_reset,
          attributes->queue_index,
          attributes->queue_length,
          queue->values[attributes->selector].GetPointer(util::DEVICE),
          NULL,
          queue->values[attributes->selector^1].GetPointer(util::DEVICE),
          d_data_slice,
          NULL,
          work_progress[0],
          queue->keys[attributes->selector  ].GetSize(),
          queue->keys[attributes->selector^1].GetSize(),
          statistics->filter_kernel_stats);

        if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
          "filter::Kernel failed", __FILE__, __LINE__))) break;

        if (DEBUG) printf("  * finished find ids for keys and col_indices. \n");

        if (debug_info)
        {
          printf(":: keys_array found super-vertex ids ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: edgeId_list found super-vertex ids ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->colindices.GetPointer(util::DEVICE),
            graph_slice->edges);
        }

        ////////////////////////////////////////////////////////////////////////
        // bring edges, weights, origin_eids together according to keys
        util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->temp_index.GetPointer(util::DEVICE),
          problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
          graph_slice->edges);

        // used super_edge as temp_index here
        util::MemsetCopyVectorKernel<<<128, 128>>>(
          problem->data_slices[0]->super_edge.GetPointer(util::DEVICE),
          problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
          graph_slice->edges);

        util::CUBRadixSort<VertexId, VertexId>(
          true, graph_slice->edges,
          problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
          problem->data_slices[0]->colindices.GetPointer(util::DEVICE));

        util::CUBRadixSort<VertexId, Value>(
          true, graph_slice->edges,
          problem->data_slices[0]->temp_index.GetPointer(util::DEVICE),
          problem->data_slices[0]->edge_value.GetPointer(util::DEVICE));

        // used super_edge as temp_index here
        util::CUBRadixSort<VertexId, VertexId>(
          true, graph_slice->edges,
          problem->data_slices[0]->super_edge.GetPointer(util::DEVICE),
          problem->data_slices[0]->original_e.GetPointer(util::DEVICE));

        if (DEBUG) printf("  * finished sort according to new vertex ids.\n");
        if (DEBUG) printf(" (d). Constructing the Vertex List.\n");

        ////////////////////////////////////////////////////////////////////////
        // flag array used for getting row_offsets for next iteration
        util::MarkSegmentFromKeys<<<128, 128>>>(
          problem->data_slices[0]->flag_array.GetPointer(util::DEVICE),
          problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
          graph_slice->edges);

        util::MemsetKernel<unsigned int><<<1, 1>>>(
          problem->data_slices[0]->flag_array.GetPointer(util::DEVICE), 0, 1);

        if (DEBUG) printf("  * finished scan of keys: flags next iteration.\n");

        ////////////////////////////////////////////////////////////////////////
        // generate row_offsets for next iteration
        attributes->queue_index  = 0;
        attributes->selector     = 0;
        attributes->queue_length = graph_slice->edges;
        attributes->queue_reset  = true;

        gunrock::oprtr::filter::LaunchKernel
          <FilterKernelPolicy, MSTProblem, RIdxFunctor>(
          statistics->filter_grid_size,
          FilterKernelPolicy::THREADS, 
          0, stream,
          statistics->iteration + 1,
          attributes->queue_reset,
          attributes->queue_index,
          attributes->queue_length,
          queue->values[attributes->selector  ].GetPointer(util::DEVICE),
          NULL,
          queue->values[attributes->selector^1].GetPointer(util::DEVICE),
          d_data_slice,
          NULL,
          work_progress[0],
          queue->keys[attributes->selector  ].GetSize(),
          queue->keys[attributes->selector^1].GetSize(),
          statistics->filter_kernel_stats);

        if (DEBUG && (retval = util::GRError(cudaDeviceSynchronize(),
          "filter::Kernel failed", __FILE__, __LINE__))) break;

        ////////////////////////////////////////////////////////////////////////
        // copy back d_col_indices back to column indices in graph_slice
        util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->column_indices.GetPointer(util::DEVICE),
          problem->data_slices[0]->colindices.GetPointer(util::DEVICE),
          graph_slice->edges);

        ////////////////////////////////////////////////////////////////////////
        // set last element of row_offsets manually and copy back to graph_slice
        util::MemsetKernel<<<128, 128>>>(
          problem->data_slices[0]->row_offset.GetPointer(util::DEVICE)
          + graph_slice->nodes, graph_slice->edges, 1);
        util::MemsetCopyVectorKernel<<<128, 128>>>(
          graph_slice->row_offsets.GetPointer(util::DEVICE),
          problem->data_slices[0]->row_offset.GetPointer(util::DEVICE),
          graph_slice->nodes + 1);

        if (DEBUG) printf("  * finished row_offset for next iteration.\n");

        if (debug_info)
        {
          printf(":: final graph_slice d_column_indices ::");
          util::DisplayDeviceResults(
            graph_slice->column_indices.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: final keys for current iteration ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->keys_array.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: final edge_values for current iteration ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->edge_value.GetPointer(util::DEVICE),
            graph_slice->edges);
          printf(":: final d_origin_edges for current iteration ::");
          util::DisplayDeviceResults(
            problem->data_slices[0]->original_e.GetPointer(util::DEVICE),
            graph_slice->edges);
        }

        if (DEBUG)
        {
          printf("END ITERATION: %lld #NODES LEFT: %d #EDGES LEFT: %d\n",
            statistics->iteration+1,graph_slice->nodes,graph_slice->edges);
        }

        statistics->iteration++;

      }  // end of the MST recursive loop

      if (d_scanned_edges) cudaFree(d_scanned_edges);
      if (num_selected) delete num_selected;
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
  template <typename MSTProblem>
  cudaError_t Enact(
    ContextPtr  context,
    MSTProblem* problem,
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
        MSTProblem,         // Problem data type
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
        MSTProblem,         // Problem data type
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

      return EnactMST<AdvanceKernelPolicy, FilterKernelPolicy,
        MSTProblem>(context, problem, max_grid_size);
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

} // namespace mst
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
