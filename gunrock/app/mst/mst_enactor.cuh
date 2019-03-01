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
#include <gunrock/oprtr/filter/kernel.cuh>

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
template <typename _Problem>
// bool _INSTRUMENT,
// bool _DEBUG,
// bool _SIZE_CHECK >
class MSTEnactor : public EnactorBase<typename _Problem::SizeT> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexId VertexId;
  typedef typename Problem::Value Value;
  typedef EnactorBase<SizeT> BaseEnactor;
  Problem* problem;
  ContextPtr* context;
  // static const bool INSTRUMENT   =   _INSTRUMENT;
  // static const bool DEBUG        =        _DEBUG;
  // static const bool SIZE_CHECK   =   _SIZE_CHECK;

 protected:
  /**
   * A pinned, mapped word that the traversal kernels will signal when done
   */
  // int *vertex_flag;

 public:
  /**
   * @brief MSTEnactor constructor.
   */
  MSTEnactor(int num_gpus = 1, int* gpu_idx = NULL, bool instrument = false,
             bool debug = false, bool size_check = true)
      : BaseEnactor(EDGE_FRONTIERS, num_gpus, gpu_idx, instrument, debug,
                    size_check),
        problem(NULL),
        context(NULL) {
    // vertex_flag = new int[1];
    // vertex_flag[0] = 0;
  }

  /**
   * @brief MSTEnactor destructor
   */
  virtual ~MSTEnactor() {
    // if (vertex_flag) delete[] vertex_flag;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /** @} */

  template <typename AdvanceKernelPolicy, typename FilterKernelPolicy>
  cudaError_t InitMST(ContextPtr* context, Problem* problem,
                      int max_grid_size = 0) {
    cudaError_t retval = cudaSuccess;

    // Lazy initialization
    if (retval = BaseEnactor::Init(
            // problem,
            max_grid_size, AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY))
      return retval;

    this->problem = problem;
    this->context = context;
    return retval;
  }

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
  template <typename AdvanceKernelPolicy, typename FilterKernelPolicy>
  // typename MSTProblem>
  cudaError_t EnactMST()
  // ContextPtr  context,
  // MSTProblem* problem,
  // int         max_grid_size = 0)
  {
    // typedef typename MSTProblem::VertexId VertexId;
    // typedef typename MSTProblem::SizeT    SizeT;
    // typedef typename MSTProblem::Value    Value;

    typedef SuccFunctor<VertexId, SizeT, Value, Problem> SuccFunctor;
    typedef EdgeFunctor<VertexId, SizeT, Value, Problem> EdgeFunctor;
    typedef CyRmFunctor<VertexId, SizeT, Value, Problem> CyRmFunctor;
    typedef PJmpFunctor<VertexId, SizeT, Value, Problem> PJmpFunctor;
    typedef EgRmFunctor<VertexId, SizeT, Value, Problem> EgRmFunctor;
    typedef RIdxFunctor<VertexId, SizeT, Value, Problem> RIdxFunctor;
    typedef SuRmFunctor<VertexId, SizeT, Value, Problem> SuRmFunctor;
    typedef EIdxFunctor<VertexId, SizeT, Value, Problem> EIdxFunctor;
    typedef MarkFunctor<VertexId, SizeT, Value, Problem> MarkFunctor;
    typedef typename Problem::DataSlice DataSlice;
    typedef util::DoubleBuffer<VertexId, SizeT, Value> Frontier;
    typedef GraphSlice<VertexId, SizeT, Value> GraphSliceT;

    Problem* problem = this->problem;
    FrontierAttribute<SizeT>* attributes = &this->frontier_attribute[0];
    EnactorStats<SizeT>* statistics = &this->enactor_stats[0];
    GraphSliceT* graph_slice = problem->graph_slices[0];
    DataSlice* d_data_slice = problem->d_data_slices[0];
    DataSlice* data_slice = problem->data_slices[0];
    Frontier* queue = &data_slice->frontier_queues[0];
    util::CtaWorkProgressLifetime<SizeT>* work_progress =
        &this->work_progress[0];
    cudaStream_t stream = data_slice->streams[0];
    ContextPtr context = this->context[0];
    cudaError_t retval = cudaSuccess;
    SizeT* d_scanned_edges = NULL;  // Used for LB

    do {
      // single-GPU graph slice

      if (retval = util::GRError(cudaMalloc((void**)&d_scanned_edges,
                                            graph_slice->edges * sizeof(SizeT)),
                                 "Problem cudaMalloc d_scanned_edges failed",
                                 __FILE__, __LINE__)) {
        return retval;
      }

      // debug configurations
      // SizeT num_edges_origin = graph_slice->edges;
      bool debug_info = 0;  // used for debug purpose
      // int tmp_select  = 0; // used for debug purpose
      // int tmp_length  = 0; // used for debug purpose
      unsigned int* num_selected = new unsigned int;  // used in cub select

      //////////////////////////////////////////////////////////////////////////
      // recursive Loop for minimum spanning tree implementation
      while (graph_slice->nodes > 1)  // more than ONE super-vertex
      {
        if (this->debug) {
          printf("\nBEGIN ITERATION: %lld #NODES: %lld #EDGES: %lld\n",
                 statistics->iteration + 1, (long long)graph_slice->nodes,
                 (long long)graph_slice->edges);
          printf("keys size = %d, %d\n", queue->keys[0].GetSize(),
                 queue->keys[1].GetSize());
        }

        if (debug_info) {
          printf(":: initial read in row_offsets ::");
          util::DisplayDeviceResults(
              graph_slice->row_offsets.GetPointer(util::DEVICE),
              graph_slice->nodes + 1);
        }

        // generate flag_array from d_row_offsets using MarkSegment kernel
        util::MarkSegmentFromIndices<<<128, 128, 0, stream>>>(
            data_slice->flag_array.GetPointer(util::DEVICE),
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->nodes);

        if (this->debug)
          printf("* finished mark segmentation >> flag_array.\n");

        // generate d_keys_array from flag_array using sum inclusive scan
        Scan<MgpuScanTypeInc>(
            data_slice->flag_array.GetPointer(util::DEVICE), graph_slice->edges,
            (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)NULL, (SizeT*)NULL,
            data_slice->keys_array.GetPointer(util::DEVICE), context[0]);

        if (this->debug) {
          printf(
              "* finished segmented sum scan >> d_keys_array.\n"
              "A. MARKING THE MST EDGES ...\n"
              " a. Finding Minimum Weighted Edges\n");
        }

        ////////////////////////////////////////////////////////////////////////
        // each vertex u finds the minimum weighted edge to another vertex v
        // select minimum edge_weights and keys using mgpu::ReduceByKey
        int num_segments;
        ReduceByKey(data_slice->keys_array.GetPointer(util::DEVICE),
                    data_slice->edge_value.GetPointer(util::DEVICE),
                    graph_slice->edges, std::numeric_limits<Value>::max(),
                    mgpu::minimum<Value>(), mgpu::equal_to<Value>(),
                    data_slice->reduce_key.GetPointer(util::DEVICE),
                    data_slice->reduce_val.GetPointer(util::DEVICE),
                    &num_segments, (int*)NULL, context[0]);

        if (retval = util::GRError(cudaStreamSynchronize(stream),
                                   "cudaStreamSynchronize failed", __FILE__,
                                   __LINE__))
          break;
        if (this->debug)
          printf("  * finished segmented reduction: keys & weight.\n");

        if (debug_info) {
          printf(":: origin flag_array ::");
          util::DisplayDeviceResults(
              data_slice->flag_array.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: origin d_keys_array ::");
          util::DisplayDeviceResults(
              data_slice->keys_array.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: origin d_col_indices ::");
          util::DisplayDeviceResults(
              graph_slice->column_indices.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: origin d_edge_weights ::");
          util::DisplayDeviceResults(
              data_slice->edge_value.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: reduced keys array - d_reduced_keys ::");
          util::DisplayDeviceResults(
              data_slice->reduce_key.GetPointer(util::DEVICE), num_segments);

          printf(":: reduced edge weights - d_reduced_vals ::");
          util::DisplayDeviceResults(
              data_slice->reduce_val.GetPointer(util::DEVICE), num_segments);
        }

        if (this->debug) printf(" (b). Finding and Removing Cycles.\n");

        ////////////////////////////////////////////////////////////////////////
        // generate successor array using SuccFunctor - advance
        // successor array holds the outgoing v for each u
        attributes->queue_index = 0;
        attributes->selector = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset = true;

        util::MemsetKernel<<<128, 128, 0, stream>>>(
            data_slice->successors.GetPointer(util::DEVICE),
            std::numeric_limits<VertexId>::max(), graph_slice->nodes);
        util::MemsetKernel<<<128, 128, 0, stream>>>(
            data_slice->temp_index.GetPointer(util::DEVICE),
            std::numeric_limits<VertexId>::max(), graph_slice->nodes);
        util::MemsetIdxKernel<<<128, 128, 0, stream>>>(
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            graph_slice->nodes);

        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem,
                                              SuccFunctor,
                                              gunrock::oprtr::advance::V2V>(
            statistics[0], attributes[0], util::InvalidValue<VertexId>(),
            data_slice, d_data_slice, (VertexId*)NULL, (bool*)NULL, (bool*)NULL,
            d_scanned_edges,
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            queue->keys[attributes->selector ^ 1].GetPointer(util::DEVICE),
            (Value*)NULL, (Value*)NULL,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE), (SizeT*)NULL,
            (VertexId*)NULL, graph_slice->nodes, graph_slice->edges,
            work_progress[0], context[0], stream);

        if (this->debug) {
          if (retval =
                  util::GRError(cudaStreamSynchronize(stream),
                                "advance::Kernel failed", __FILE__, __LINE__))
            break;
          printf("  * finished min weighted edges >> successors.\n");
        }

        ////////////////////////////////////////////////////////////////////////
        // finding original edge ids with the corresponding d_id
        attributes->queue_index = 0;
        attributes->selector = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset = true;

        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem,
                                              EdgeFunctor,
                                              gunrock::oprtr::advance::V2V>(
            statistics[0], attributes[0], util::InvalidValue<VertexId>(),
            data_slice, d_data_slice, (VertexId*)NULL, (bool*)NULL, (bool*)NULL,
            d_scanned_edges,
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            queue->keys[attributes->selector ^ 1].GetPointer(util::DEVICE),
            (Value*)NULL, (Value*)NULL,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE), (SizeT*)NULL,
            (VertexId*)NULL, graph_slice->nodes, graph_slice->edges,
            work_progress[0], context[0], stream);

        if (this->debug && (retval = util::GRError(
                                cudaStreamSynchronize(stream),
                                "advance::Kernel failed", __FILE__, __LINE__)))
          break;

        ////////////////////////////////////////////////////////////////////////
        // mark MST output edges
        attributes->queue_index = 0;
        attributes->selector = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset = true;

        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem,
                                              MarkFunctor,
                                              gunrock::oprtr::advance::V2E>(
            statistics[0], attributes[0], util::InvalidValue<VertexId>(),
            data_slice, d_data_slice, (VertexId*)NULL, (bool*)NULL, (bool*)NULL,
            d_scanned_edges,
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            queue->keys[attributes->selector ^ 1].GetPointer(util::DEVICE),
            (Value*)NULL, (Value*)NULL,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE), (SizeT*)NULL,
            (VertexId*)NULL, graph_slice->nodes, graph_slice->edges,
            work_progress[0], context[0], stream);

        if (this->debug && (retval = util::GRError(
                                cudaStreamSynchronize(stream),
                                "advance::Kernel failed", __FILE__, __LINE__)))
          break;

        ////////////////////////////////////////////////////////////////////////
        // remove cycles - vertices with S(S(u)) = u forms cycles
        attributes->queue_index = 0;
        attributes->selector = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset = true;

        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem,
                                              CyRmFunctor,
                                              gunrock::oprtr::advance::V2E>(
            statistics[0], attributes[0], util::InvalidValue<VertexId>(),
            data_slice, d_data_slice, (VertexId*)NULL, (bool*)NULL, (bool*)NULL,
            d_scanned_edges,
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            queue->keys[attributes->selector ^ 1].GetPointer(util::DEVICE),
            (Value*)NULL, (Value*)NULL,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE), (SizeT*)NULL,
            (VertexId*)NULL, graph_slice->nodes, graph_slice->edges,
            work_progress[0], context[0], stream);

        if (this->debug) {
          if (retval =
                  util::GRError(cudaStreamSynchronize(stream),
                                "advance::Kernel failed", __FILE__, __LINE__))
            break;
          printf("  * finished removing cycles >> new successors.\n");
        }

        if (debug_info) {
          printf(":: remove cycles from successors ::");
          util::DisplayDeviceResults(
              data_slice->successors.GetPointer(util::DEVICE),
              graph_slice->nodes);
        }

        if (this->debug)
          printf(
              "B. GRAPH CONSTRUCTION ...\n"
              " (a). Merging Vertices\n");

        ////////////////////////////////////////////////////////////////////////
        // Then, we combine vertices to form a super-vertex by employing
        // pointer doubling to achieve this result, iteratively setting
        // S(u) = S(S(u)) until no further change occurs in S
        // using filter kernel: PJmpFunctor
        attributes->queue_index = 0;
        attributes->selector = 0;
        attributes->queue_length = graph_slice->nodes;
        attributes->queue_reset = true;

        data_slice->done_flags[0] = 0;
        while (data_slice->done_flags[0] == 0) {
          data_slice->done_flags[0] = 1;
          if (retval = data_slice->done_flags.Move(util::HOST, util::DEVICE, 1,
                                                   0, stream))
            return retval;

          gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                               PJmpFunctor>(
              statistics[0], attributes[0], (VertexId)statistics->iteration + 1,
              data_slice, d_data_slice, (SizeT*)NULL,
              data_slice->visited_mask.GetPointer(util::DEVICE),
              queue->keys[attributes->selector].GetPointer(util::DEVICE),
              queue->keys[attributes->selector ^ 1].GetPointer(util::DEVICE),
              (Value*)NULL, (Value*)NULL,
              attributes->queue_length,  // attributes ->output_length[0],
              graph_slice->nodes, work_progress[0], context[0], stream,
              queue->keys[attributes->selector].GetSize(),
              queue->keys[attributes->selector ^ 1].GetSize(),
              statistics->filter_kernel_stats, true, false);

          // prepare for next iteration, only reset once
          attributes->queue_reset = false;
          attributes->queue_index++;
          attributes->selector ^= 1;

          // problem->data_slices[0]->done_flags.SetPointer(vertex_flag);
          if (retval = data_slice->done_flags.Move(util::DEVICE, util::HOST, 1,
                                                   0, stream))
            return retval;

          if (retval = util::GRError(cudaStreamSynchronize(stream),
                                     "filter PointerJumping failed", __FILE__,
                                     __LINE__))
            break;

          // check if finished pointer jumping
          if (data_slice->done_flags[0] != 0) break;
        }

        if (this->debug)
          printf(
              "  * finished pointer doubling: representatives.\n"
              " (b).Assigning IDs to Super-vertices\n");

        ////////////////////////////////////////////////////////////////////////
        // each vertex of a super-vertex now has a representative, but the
        // super-vertices are not numbered in order. The vertices assigned
        // to a super-vertex are also not placed in order in the successor
        if (debug_info) {
          printf("after ptrjump before radixsort\n");
          util::DisplayDeviceResults(
              data_slice->super_idxs.GetPointer(util::DEVICE),
              graph_slice->nodes);
          util::DisplayDeviceResults(
              data_slice->successors.GetPointer(util::DEVICE),
              graph_slice->nodes);
        }

        // bring all vertices of a super-vertex together by sorting
        util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
            data_slice->super_idxs.GetPointer(util::DEVICE),
            data_slice->successors.GetPointer(util::DEVICE),
            graph_slice->nodes);

        util::MemsetIdxKernel<<<128, 128, 0, stream>>>(
            data_slice->original_n.GetPointer(util::DEVICE),
            graph_slice->nodes);

        util::CUBRadixSort<VertexId, VertexId>(
            true, graph_slice->nodes,
            data_slice->super_idxs.GetPointer(util::DEVICE),
            data_slice->original_n.GetPointer(util::DEVICE));

        if (debug_info) {
          printf(":: pointer jumping: representatives ::");
          util::DisplayDeviceResults(
              data_slice->successors.GetPointer(util::DEVICE),
              graph_slice->nodes);
          printf(":: bring all vertices of a super-vertex together ::");
          util::DisplayDeviceResults(
              data_slice->super_idxs.GetPointer(util::DEVICE),
              graph_slice->nodes);
        }

        ////////////////////////////////////////////////////////////////////////
        // create a flag to mark the boundaries of representative vertices
        util::MarkSegmentFromKeys<<<128, 128, 0, stream>>>(
            data_slice->flag_array.GetPointer(util::DEVICE),
            data_slice->super_idxs.GetPointer(util::DEVICE),
            graph_slice->nodes);

        if (this->debug)
          printf("  * finished mark super-vertices: super flags.\n");

        ////////////////////////////////////////////////////////////////////////
        // sum scan of the super flags to assign new super-vertex ids
        Scan<MgpuScanTypeInc>(
            data_slice->flag_array.GetPointer(util::DEVICE), graph_slice->nodes,
            (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)NULL, (SizeT*)NULL,
            data_slice->super_idxs.GetPointer(util::DEVICE), context[0]);

        if (this->debug) printf("  * finished assign super ids:   .\n");

        if (debug_info) {
          printf(":: super flags (a.k.a. c flag) ::");
          util::DisplayDeviceResults(
              data_slice->flag_array.GetPointer(util::DEVICE),
              graph_slice->nodes);
          printf(":: new assigned super-vertex ids ::");
          util::DisplayDeviceResults(
              data_slice->super_idxs.GetPointer(util::DEVICE),
              graph_slice->nodes);
        }

        // Back to default stream, since CUBRadixSort does not support stream
        if (retval = util::GRError(cudaStreamSynchronize(stream),
                                   "cudaStreamSynchronize failed", __FILE__,
                                   __LINE__))
          break;

        ////////////////////////////////////////////////////////////////////////
        // used for finding super-vertex ids for next iteration
        util::CUBRadixSort<VertexId, VertexId>(
            true, graph_slice->nodes,
            data_slice->original_n.GetPointer(util::DEVICE),
            data_slice->super_idxs.GetPointer(util::DEVICE));

        ////////////////////////////////////////////////////////////////////////
        // update graph_slice->nodes with number of super-vertices
        SizeT current_nodes = graph_slice->nodes;
        // the first segment in flag was set to 0 instead of 1
        util::MemsetKernel<unsigned int>
            <<<1, 1>>>(data_slice->flag_array.GetPointer(util::DEVICE), 1, 1);
        graph_slice->nodes =
            Reduce(data_slice->flag_array.GetPointer(util::DEVICE),
                   graph_slice->nodes, context[0]);

        if (retval = util::GRError(cudaDeviceSynchronize(),
                                   "cudaDeviceSynchronize failed", __FILE__,
                                   __LINE__))
          break;

        if (this->debug)
          printf("  * finished update #nodes: %d left.\n", graph_slice->nodes);

        // terminate the loop if there is only one super-vertex left
        if (graph_slice->nodes == 1) {
          if (this->debug) printf("\nTERMINATE THE MST ALGORITHM ENACTOR.\n\n");
          break;  // break the MST recursive loop
        }

        if (this->debug)
          printf(" (c). Removing Edges & Forming the new Edge List\n");

        ////////////////////////////////////////////////////////////////////////
        // shorten the edge list by removing self edges in the new graph
        // advance kernel remove edges belonging to the same super-vertex
        // each edge examines the super-vertex id of both end vertices and
        // removes itself if the id is the same
        attributes->queue_index = 0;
        attributes->selector = 0;
        attributes->queue_length = current_nodes;
        attributes->queue_reset = true;

        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem,
                                              EgRmFunctor,
                                              gunrock::oprtr::advance::V2E>(
            statistics[0], attributes[0], util::InvalidValue<VertexId>(),
            data_slice, d_data_slice, (VertexId*)NULL, (bool*)NULL, (bool*)NULL,
            d_scanned_edges,
            queue->keys[attributes->selector].GetPointer(util::DEVICE),
            queue->keys[attributes->selector ^ 1].GetPointer(util::DEVICE),
            (Value*)NULL, (Value*)NULL,
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE), (SizeT*)NULL,
            (VertexId*)NULL, current_nodes, graph_slice->edges,
            work_progress[0], context[0], stream);

        // Back to default stream, as Cub calls do not support GPU steam for now
        if (retval =
                util::GRError(cudaStreamSynchronize(stream),
                              "advance::Kernel failed", __FILE__, __LINE__))
          break;

        if (this->debug)
          printf("  * finished mark edges in same super-vertex.\n");

        if (debug_info) {
          printf(":: edge removal in one super-vertex (d_keys_array) ::");
          util::DisplayDeviceResults(
              data_slice->keys_array.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: edge removal in one super-vertex (d_col_indices) ::");
          util::DisplayDeviceResults(
              data_slice->colindices.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: edge removal in one super-vertex (d_edge_weights) ::");
          util::DisplayDeviceResults(
              data_slice->edge_value.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: edge removal in one super-vertex (d_origin_edges) ::");
          util::DisplayDeviceResults(
              data_slice->original_e.GetPointer(util::DEVICE),
              graph_slice->edges);
        }

        ////////////////////////////////////////////////////////////////////////
        // filter to remove all -1 in d_col_indices
        util::MemsetCopyVectorKernel<<<128, 128>>>(
            data_slice->temp_index.GetPointer(util::DEVICE),
            data_slice->colindices.GetPointer(util::DEVICE),
            graph_slice->edges);
        util::CUBSelect<VertexId, SizeT>(
            data_slice->temp_index.GetPointer(util::DEVICE), graph_slice->edges,
            data_slice->colindices.GetPointer(util::DEVICE), num_selected);

        ////////////////////////////////////////////////////////////////////////
        // filter to remove all -1 in d_edge_weights
        util::MemsetCopyVectorKernel<<<128, 128>>>(
            data_slice->temp_value.GetPointer(util::DEVICE),
            data_slice->edge_value.GetPointer(util::DEVICE),
            graph_slice->edges);
        util::CUBSelect<Value, SizeT>(
            data_slice->temp_value.GetPointer(util::DEVICE), graph_slice->edges,
            data_slice->edge_value.GetPointer(util::DEVICE), num_selected);

        ////////////////////////////////////////////////////////////////////////
        // filter to remove all -1 in d_keys_array
        util::MemsetCopyVectorKernel<<<128, 128>>>(
            data_slice->temp_index.GetPointer(util::DEVICE),
            data_slice->keys_array.GetPointer(util::DEVICE),
            graph_slice->edges);
        util::CUBSelect<VertexId, SizeT>(
            data_slice->temp_index.GetPointer(util::DEVICE), graph_slice->edges,
            data_slice->keys_array.GetPointer(util::DEVICE), num_selected);

        ////////////////////////////////////////////////////////////////////////
        // filter to remove all -1 in d_origin_edges
        util::MemsetCopyVectorKernel<<<128, 128>>>(
            data_slice->temp_index.GetPointer(util::DEVICE),
            data_slice->original_e.GetPointer(util::DEVICE),
            graph_slice->edges);
        util::CUBSelect<VertexId, SizeT>(
            data_slice->temp_index.GetPointer(util::DEVICE), graph_slice->edges,
            data_slice->original_e.GetPointer(util::DEVICE), num_selected);

        if (retval = util::GRError(cudaDeviceSynchronize(),
                                   "cudaDeviceSynchronize failed", __FILE__,
                                   __LINE__))
          break;

        if (this->debug)
          printf("  * finished remove edges in one super-vertex.\n");

        ////////////////////////////////////////////////////////////////////////
        // update edge list length in graph_slice [1]
        graph_slice->edges = *num_selected;

        if (this->debug)
          printf("  * finished update #edge: %lld\n",
                 (long long)graph_slice->edges);

        if (debug_info) {
          printf(":: edge removal in one super-vertex (d_keys_array) ::");
          util::DisplayDeviceResults(
              data_slice->keys_array.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: edge removal in one super-vertex (d_col_indices) ::");
          util::DisplayDeviceResults(
              data_slice->colindices.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: edge removal in one super-vertex (d_edge_weights) ::");
          util::DisplayDeviceResults(
              data_slice->edge_value.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: edge removal in one super-vertex (d_origin_edges) ::");
          util::DisplayDeviceResults(
              data_slice->original_e.GetPointer(util::DEVICE),
              graph_slice->edges);
        }

        ////////////////////////////////////////////////////////////////////////
        // find super-vertex ids for d_keys_array and d_col_indices
        attributes->queue_index = 0;
        attributes->selector = 0;
        attributes->queue_length = graph_slice->edges;
        attributes->queue_reset = true;

        if (retval = util::GRError(cudaStreamSynchronize(stream),
                                   "memset queue failed", __FILE__, __LINE__))
          break;

        gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                             EgRmFunctor>(
            statistics[0], attributes[0], (VertexId)statistics->iteration + 1,
            data_slice, d_data_slice, (SizeT*)NULL,
            data_slice->visited_mask.GetPointer(util::DEVICE),
            queue->values[attributes->selector].GetPointer(
                util::DEVICE),  //(VertexId*)NULL,
            queue->values[attributes->selector ^ 1].GetPointer(
                util::DEVICE),         //(VertexId*)NULL,
            (Value*)NULL,              // queue -> keys[attributes -> selector
                                       // ].GetPointer(util::DEVICE),
            (Value*)NULL,              // queue -> keys[attributes ->
                                       // selector^1].GetPointer(util::DEVICE),
            attributes->queue_length,  // attributes -> output_length[0], //
                                       // attributes -> quque_length
            graph_slice->nodes,  // attribute -> queue_length
            work_progress[0], context[0], stream,
            queue->values[attributes->selector].GetSize(),
            queue->values[attributes->selector ^ 1].GetSize(),
            statistics->filter_kernel_stats, true, false);

        // Back to default stream, as Cub calls do not support GPU steam for now
        if (retval = util::GRError(cudaStreamSynchronize(stream),
                                   "filter::Kernel failed", __FILE__, __LINE__))
          break;

        if (retval = util::GRError(cudaDeviceSynchronize(),
                                   "cudaDeviceSynchronie() failed", __FILE__,
                                   __LINE__))
          break;

        if (this->debug)
          printf("  * finished find ids for keys and col_indices. \n");

        if (debug_info) {
          printf(":: keys_array found super-vertex ids ::");
          util::DisplayDeviceResults(
              data_slice->keys_array.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: edgeId_list found super-vertex ids ::");
          util::DisplayDeviceResults(
              data_slice->colindices.GetPointer(util::DEVICE),
              graph_slice->edges);
        }

        ////////////////////////////////////////////////////////////////////////
        // bring edges, weights, origin_eids together according to keys
        util::MemsetCopyVectorKernel<<<128, 128>>>(
            data_slice->temp_index.GetPointer(util::DEVICE),
            data_slice->keys_array.GetPointer(util::DEVICE),
            graph_slice->edges);

        // used super_edge as temp_index here
        util::MemsetCopyVectorKernel<<<128, 128>>>(
            data_slice->super_edge.GetPointer(util::DEVICE),
            data_slice->keys_array.GetPointer(util::DEVICE),
            graph_slice->edges);

        util::CUBRadixSort<VertexId, VertexId>(
            true, graph_slice->edges,
            data_slice->keys_array.GetPointer(util::DEVICE),
            data_slice->colindices.GetPointer(util::DEVICE));

        util::CUBRadixSort<VertexId, Value>(
            true, graph_slice->edges,
            data_slice->temp_index.GetPointer(util::DEVICE),
            data_slice->edge_value.GetPointer(util::DEVICE));

        // used super_edge as temp_index here
        util::CUBRadixSort<VertexId, VertexId>(
            true, graph_slice->edges,
            data_slice->super_edge.GetPointer(util::DEVICE),
            data_slice->original_e.GetPointer(util::DEVICE));

        // back to stream
        if (retval = util::GRError(cudaDeviceSynchronize(),
                                   "cudaDeviceSynchronize failed", __FILE__,
                                   __LINE__))
          break;

        if (this->debug)
          printf(
              "  * finished sort according to new vertex ids.\n"
              " (d). Constructing the Vertex List.\n");

        ////////////////////////////////////////////////////////////////////////
        // flag array used for getting row_offsets for next iteration
        util::MarkSegmentFromKeys<<<128, 128, 0, stream>>>(
            data_slice->flag_array.GetPointer(util::DEVICE),
            data_slice->keys_array.GetPointer(util::DEVICE),
            graph_slice->edges);

        util::MemsetKernel<unsigned int><<<1, 1, 0, stream>>>(
            data_slice->flag_array.GetPointer(util::DEVICE), 0, 1);

        if (this->debug)
          printf("  * finished scan of keys: flags next iteration.\n");

        ////////////////////////////////////////////////////////////////////////
        // generate row_offsets for next iteration
        attributes->queue_index = 0;
        attributes->selector = 0;
        attributes->queue_length = graph_slice->edges;
        attributes->queue_reset = true;

        gunrock::oprtr::filter::LaunchKernel<FilterKernelPolicy, Problem,
                                             RIdxFunctor>(
            statistics[0], attributes[0], (VertexId)statistics->iteration + 1,
            data_slice, d_data_slice, (SizeT*)NULL,
            data_slice->visited_mask.GetPointer(util::DEVICE),
            queue->values[attributes->selector].GetPointer(
                util::DEVICE),  //(VertexId*)NULL, // values...
            queue->values[attributes->selector ^ 1].GetPointer(
                util::DEVICE),  //(VertexId*)NULL, // values...
            (Value*)NULL,       // queue->values[attributes->selector
                           // ].GetPointer(util::DEVICE), // NULL
            (Value*)NULL,  // queue->values[attributes->selector^1].GetPointer(util::DEVICE),
                           // // NULL
            attributes->queue_length,  // attributes->output_length[0], //
                                       // queue_length
            graph_slice->nodes,  // queue_length
            work_progress[0], context[0], stream,
            queue->values[attributes->selector].GetSize(),      // values.Size
            queue->values[attributes->selector ^ 1].GetSize(),  // values.Size
            statistics->filter_kernel_stats, true, false);

        ////////////////////////////////////////////////////////////////////////
        // copy back d_col_indices back to column indices in graph_slice
        util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
            graph_slice->column_indices.GetPointer(util::DEVICE),
            data_slice->colindices.GetPointer(util::DEVICE),
            graph_slice->edges);

        ////////////////////////////////////////////////////////////////////////
        // set last element of row_offsets manually and copy back to graph_slice
        util::MemsetKernel<<<128, 128, 0, stream>>>(
            data_slice->row_offset.GetPointer(util::DEVICE) +
                graph_slice->nodes,
            graph_slice->edges, 1);
        util::MemsetCopyVectorKernel<<<128, 128, 0, stream>>>(
            graph_slice->row_offsets.GetPointer(util::DEVICE),
            data_slice->row_offset.GetPointer(util::DEVICE),
            graph_slice->nodes + 1);

        // Back to default stream, as Cub calls do not support GPU steam for now
        if (retval = util::GRError(cudaStreamSynchronize(stream),
                                   "cudaStreamSynchronize failed", __FILE__,
                                   __LINE__))
          break;

        if (this->debug)
          printf("  * finished row_offset for next iteration.\n");

        if (debug_info) {
          printf(":: final graph_slice d_column_indices ::");
          util::DisplayDeviceResults(
              graph_slice->column_indices.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: final keys for current iteration ::");
          util::DisplayDeviceResults(
              data_slice->keys_array.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: final edge_values for current iteration ::");
          util::DisplayDeviceResults(
              data_slice->edge_value.GetPointer(util::DEVICE),
              graph_slice->edges);

          printf(":: final d_origin_edges for current iteration ::");
          util::DisplayDeviceResults(
              data_slice->original_e.GetPointer(util::DEVICE),
              graph_slice->edges);
        }

        if (this->debug) {
          printf("END ITERATION: %lld #NODES LEFT: %lld #EDGES LEFT: %lld\n",
                 statistics->iteration + 1, (long long)graph_slice->nodes,
                 (long long)graph_slice->edges);
        }

        statistics->iteration++;
        // break;

      }  // end of the MST recursive loop

      if (d_scanned_edges) cudaFree(d_scanned_edges);
      if (num_selected) delete num_selected;
      if (retval) break;

    } while (0);
    return retval;
  }

  typedef gunrock::oprtr::filter::KernelPolicy<Problem,  // Problem data type
                                               300,      // CUDA_ARCH
                                               0,        // SATURATION QUIT
                                               true,     // DEQUEUE_PROBLEM_SIZE
                                               8,        // MIN_CTA_OCCUPANCY
                                               8,        // LOG_THREADS
                                               2,        // LOG_LOAD_VEC_SIZE
                                               0,        // LOG_LOADS_PER_TILE
                                               5,        // LOG_RAKING_THREADS
                                               5,        // END_BITMASK_CULL
                                               8>  // LOG_SCHEDULE_GRANULARITY
      FilterKernelPolicy;

  typedef gunrock::oprtr::advance::KernelPolicy<
      Problem,  // Problem data type
      300,      // CUDA_ARCH
      // INSTRUMENT,         // INSTRUMENT
      8,         // MIN_CTA_OCCUPANCY
      10,        // LOG_THREADS
      8,         // LOG_BLOCKS
      32 * 128,  // LIGHT_EDGE_THRESHOLD
      1,         // LOG_LOAD_VEC_SIZE
      0,         // LOG_LOADS_PER_TILE
      5,         // LOG_RAKING_THREADS
      32,        // WARP_GATHER_THRESHOLD
      128 * 4,   // CTA_GATHER_THRESHOLD
      7,         // LOG_SCHEDULE_GRANULARITY
      gunrock::oprtr::advance::LB_LIGHT>
      AdvanceKernelPolicy;

  /**
   * @brief Reset enactor
   *
   * \return cudaError_t object Indicates the success of all CUDA calls.
   */
  cudaError_t Reset() { return BaseEnactor::Reset(); }

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
  cudaError_t Init(ContextPtr* context, Problem* problem,
                   int max_grid_size = 0) {
    int min_sm_version = -1;
    for (int i = 0; i < this->num_gpus; i++) {
      if (min_sm_version == -1 ||
          this->cuda_props[i].device_sm_version < min_sm_version) {
        min_sm_version = this->cuda_props[i].device_sm_version;
      }
    }

    if (min_sm_version >= 300) {
      return InitMST<AdvanceKernelPolicy, FilterKernelPolicy>(context, problem,
                                                              max_grid_size);
    }

    // to reduce compile time, get rid of other architectures for now
    // TODO: add all the kernel policy settings for all architectures

    printf("Not yet tuned for this architecture\n");
    return cudaErrorInvalidDeviceFunction;
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
  cudaError_t Enact(
        /*ContextPtr  context,
        MSTProblem* problem,
        int         max_grid_size = 0*/)
    {
    int min_sm_version = -1;
    for (int i = 0; i < this->num_gpus; i++) {
      if (min_sm_version == -1 ||
          this->cuda_props[i].device_sm_version < min_sm_version) {
        min_sm_version = this->cuda_props[i].device_sm_version;
      }
    }

    if (min_sm_version >= 300) {
      return EnactMST<AdvanceKernelPolicy, FilterKernelPolicy>(
          /*context, problem, max_grid_size*/);
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

}  // namespace mst
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
