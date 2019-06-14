// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mf_problem.cuh
 *
 * @brief GPU Storage management Structure for Max Flow Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/oprtr/1D_oprtr/for_all.cuh>
#include <queue>

#define debug_aml(a...)
//#define debug_aml(a...) {printf("%s:%d ", __FILE__, __LINE__); printf(a);\
    printf("\n");}

namespace gunrock {
namespace app {
namespace mf {

/**
 * @brief Speciflying parameters for MF Problem
 * @param  parameters  The util::Parameter<...> structure holding all
 *			parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  return retval;
}

/**
 * @brief Max Flow Problem structure stores device-side arrays
 * @tparam _GraphT  Type of the graph
 * @tparam _ValueT  Type of signed integer to use as capacity and flow
                    of edges and as excess and height values of vertices.
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::GpT GpT;
  typedef _ValueT ValueT;
  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // Helper structures

  /**
   * @brief Data structure containing MF-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // MF-specific storage arrays:
    util::Array1D<SizeT, ValueT> flow;                 // edge flow
    util::Array1D<SizeT, ValueT> residuals;            // edge residuals
    util::Array1D<SizeT, ValueT> excess;               // vertex excess
    util::Array1D<SizeT, VertexT> height;              // vertex height
    util::Array1D<SizeT, VertexT> reverse;             // id reverse edge
    util::Array1D<SizeT, SizeT> lowest_neighbor;       // id lowest neighbor
    util::Array1D<SizeT, VertexT> local_vertices;      // set of vertices
    util::Array1D<SizeT, SizeT, util::PINNED> active;  // flag active vertices

    util::Array1D<SizeT, VertexT> head;
    util::Array1D<SizeT, VertexT> tail;
    VertexT head_;
    VertexT tail_;

    util::Array1D<SizeT, bool> reachabilities;
    util::Array1D<SizeT, VertexT> queue0;
    util::Array1D<SizeT, VertexT> queue1;
    util::Array1D<SizeT, bool> mark;

    VertexT source;  // source vertex
    VertexT sink;    // sink vertex
    int num_repeats;
    SizeT num_updated_vertices;
    bool was_changed;  // flag relabeling
    util::Array1D<SizeT, SizeT, util::PINNED> changed;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      source = util::PreDefinedValues<VertexT>::InvalidValue;
      sink = util::PreDefinedValues<VertexT>::InvalidValue;
      num_repeats = 10000;
      num_updated_vertices = 1;
      was_changed = false;
      reverse.SetName("reverse");
      excess.SetName("excess");
      flow.SetName("flow");
      residuals.SetName("residuals");
      height.SetName("height");
      lowest_neighbor.SetName("lowest_neighbor");
      local_vertices.SetName("local_vertices");
      active.SetName("active");

      head.SetName("head");
      tail.SetName("tail");

      reachabilities.SetName("reachabilities");
      queue0.SetName("queue0");
      queue1.SetName("queue1");
      mark.SetName("mark");
      changed.SetName("changed");
    }

    /*
     * @brief Default destructor
     */
    virtual ~DataSlice() { Release(); }

    /*
     * @brief Releasing allocated memory space
     * @param[in] target      The location to release memory from
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL) {
      cudaError_t retval = cudaSuccess;
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx));

      GUARD_CU(excess.Release(target));
      GUARD_CU(flow.Release(target));
      GUARD_CU(residuals.Release(target));
      GUARD_CU(height.Release(target));
      GUARD_CU(reverse.Release(target));
      GUARD_CU(lowest_neighbor.Release(target));
      GUARD_CU(local_vertices.Release(target));
      GUARD_CU(active.Release(target));

      GUARD_CU(head.Release(target));
      GUARD_CU(tail.Release(target));

      GUARD_CU(reachabilities.Release(target));
      GUARD_CU(queue0.Release(target));
      GUARD_CU(queue1.Release(target));
      GUARD_CU(mark.Release(target));

      GUARD_CU(BaseDataSlice::Release(target));

      return retval;
    }

    /**
     * @brief initializing MF-specific Data Slice a on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus = 1, int gpu_idx = 0,
                     util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      debug_aml("DataSlice Init");

      cudaError_t retval = cudaSuccess;
      SizeT nodes_size = sub_graph.nodes;
      SizeT edges_size = sub_graph.edges;

      was_changed = false;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      //
      // Allocate data on Gpu
      //
      GUARD_CU(flow.Allocate(edges_size, util::HOST | target));
      GUARD_CU(residuals.Allocate(edges_size, target));
      GUARD_CU(reverse.Allocate(edges_size, target));
      GUARD_CU(excess.Allocate(nodes_size, target));
      GUARD_CU(height.Allocate(nodes_size, util::HOST | target));
      GUARD_CU(lowest_neighbor.Allocate(nodes_size, target));
      GUARD_CU(local_vertices.Allocate(nodes_size, target));
      GUARD_CU(active.Allocate(2, util::HOST | target));

      GUARD_CU(head.Allocate(1, target));
      GUARD_CU(tail.Allocate(1, target));

      GUARD_CU(reachabilities.Allocate(nodes_size, target));

      GUARD_CU(queue0.Allocate(nodes_size, target));
      GUARD_CU(queue1.Allocate(nodes_size, target));
      GUARD_CU(mark.Allocate(nodes_size, target));

      GUARD_CU(changed.Allocate(1, util::HOST | target));

      GUARD_CU(util::SetDevice(gpu_idx));
      GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));

      return retval;
    }  // Init Data Slice

    /**
     * @brief Reset DataSlice function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(const GraphT &graph, const VertexT source,
                      const VertexT sink, VertexT *h_reverse,
                      util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      typedef typename GraphT::CsrT CsrT;

      debug_aml("DataSlice Reset");

      SizeT nodes_size = graph.nodes;
      SizeT edges_size = graph.edges;

      // Ensure data are allocated
      GUARD_CU(active.EnsureSize_(2, target | util::HOST));
      GUARD_CU(flow.EnsureSize_(edges_size, target));
      GUARD_CU(residuals.EnsureSize_(edges_size, target));
      GUARD_CU(reverse.EnsureSize_(edges_size, target));
      GUARD_CU(excess.EnsureSize_(nodes_size, target));
      GUARD_CU(height.EnsureSize_(nodes_size, target | util::HOST));
      GUARD_CU(lowest_neighbor.EnsureSize_(nodes_size, target));
      GUARD_CU(local_vertices.EnsureSize_(nodes_size, target));

      GUARD_CU(head.EnsureSize_(1, target));
      GUARD_CU(tail.EnsureSize_(1, target));

      GUARD_CU(reachabilities.EnsureSize_(nodes_size, target));
      GUARD_CU(queue0.EnsureSize_(nodes_size, target));
      GUARD_CU(queue1.EnsureSize_(nodes_size, target));
      GUARD_CU(mark.EnsureSize_(nodes_size, target));

      GUARD_CU(changed.EnsureSize_(1, target | util::HOST));

      GUARD_CU(util::SetDevice(this->gpu_idx));
      GUARD_CU(reverse.SetPointer(h_reverse, edges_size, util::HOST));
      GUARD_CU(reverse.Move(util::HOST, target, edges_size, 0, this->stream));

#if MF_DEBUG
      debug_aml("reverse on CPU\n");
      for (int i = 0; i < edges_size; ++i)
        debug_aml("reverse[%d] = %d\n", i, h_reverse[i]);

      debug_aml("reverse after coping to device\n");
      GUARD_CU(reverse.ForAll(
          [] __host__ __device__(VertexT * r, const VertexT &pos) {
            debug_aml("reverse[%d] = %d\n", pos, r[pos]);
          },
          edges_size, target, this->stream));
#endif

      this->num_updated_vertices = 1;
      // Reset data
      GUARD_CU(height.ForAll(
          [source, sink, nodes_size] __host__ __device__(VertexT * h,
                                                         const VertexT &pos) {
            if (pos == source)
              h[pos] = nodes_size;
            else  // if (pos == sink)
              h[pos] = 0;
            // else
            // h[pos] = 2 * nodes_size + 1;
          },
          nodes_size, target, this->stream));

      GUARD_CU(flow.ForAll(
          [] __host__ __device__(ValueT * f, const VertexT &pos) {
            f[pos] = (ValueT)0;
          },
          edges_size, target, this->stream));

      GUARD_CU(excess.ForAll(
          [] __host__ __device__(ValueT * e, const VertexT &pos) {
            e[pos] = (ValueT)0;
          },
          nodes_size, target, this->stream));

      GUARD_CU(active.ForAll(
          [] __host__ __device__(SizeT * active_, const VertexT &pos) {
            active_[pos] = 1;
          },
          2, target | util::HOST, this->stream));

      GUARD_CU(lowest_neighbor.ForAll(
          [graph, source] __host__ __device__(VertexT * lowest_neighbor,
                                              const VertexT pos) {
            lowest_neighbor[pos] =
                util::PreDefinedValues<VertexT>::InvalidValue;
          },
          nodes_size, target, this->stream));

      GUARD_CU(local_vertices.ForAll(
          [] __host__ __device__(VertexT * local_vertex, const VertexT pos) {
            local_vertex[pos] = pos;
          },
          nodes_size, target));

      GUARD_CU(mark.ForAll(
          [] __host__ __device__(bool *mark_, const VertexT &pos) {
            mark_[pos] = false;
          },
          nodes_size, target, this->stream));

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      return retval;
    }

  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief MFProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief MFProblem default destructor
   */
  virtual ~Problem() { Release(); }

  /*
   * @brief Releasing allocated memory space
   * @param[in] target      The location to release memory from
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    if (data_slices == NULL) return retval;
    for (int i = 0; i < this->num_gpus; i++)
      GUARD_CU(data_slices[i].Release(target));

    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(BaseProblem::Release(target));
    return retval;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief Copy result flow computed on GPUs back to host-side arrays.
   * @param[out] h_flow Host array to store computed flow on edges
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(ValueT *h_flow, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    auto &data_slice = data_slices[0][0];
    SizeT eN = this->org_graph->edges;

    // Set device
    if (target == util::DEVICE) {
      GUARD_CU(util::SetDevice(this->gpu_idx[0]));
      GUARD_CU(data_slice.flow.SetPointer(h_flow, eN, util::HOST));
      GUARD_CU(data_slice.flow.Move(util::DEVICE, util::HOST));
    } else if (target == util::HOST) {
      GUARD_CU(data_slice.flow.ForEach(
          h_flow,
          [] __host__ __device__(const ValueT &f, ValueT &h_f) {
            { h_f = f; }
          },
          eN, util::HOST));
    }
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    return retval;
  }

  /**
   * @brief Init MF Problem
   * @param     graph       The graph that MF processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    debug_aml("Problem Init");
    cudaError_t retval = cudaSuccess;

    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      auto gpu_name = std::to_string(gpu);
      data_slices[gpu].SetName("data_slices[" + gpu_name + "]");

      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));
      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag));

      GUARD_CU2(cudaStreamSynchronize(data_slices[gpu]->stream),
                "sync failed.");

    }  // end for (gpu)
    return retval;
  }  // End Init MF Problem

  /**
   * @brief Reset Problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(GraphT &graph, VertexT *h_reverse,
                    util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    debug_aml("Problem Reset");

    auto source_vertex = this->parameters.template Get<VertexT>("source");
    auto sink_vertex = this->parameters.template Get<VertexT>("sink");
    auto num_repeats = this->parameters.template Get<int>("num-repeats");

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      auto &data_slice = data_slices[gpu][0];
      data_slice.source = source_vertex;
      data_slice.sink = sink_vertex;
      data_slice.num_repeats = num_repeats;

      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(graph, source_vertex, sink_vertex,
                                       h_reverse, target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    // Filling the initial input_queue for MF problem

    int gpu;
    VertexT src_;
    if (this->num_gpus <= 1) {
      gpu = 0;
      src_ = source_vertex;
    } else {
      gpu = this->org_graph->partition_table[source_vertex];
      if (this->flag & partitioner::Keep_Node_Num)
        src_ = source_vertex;
      else
        src_ = this->org_graph->GpT::convertion_table[source_vertex];
    }
    GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    return retval;
  }

  /** @} */
};

}  // namespace mf
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
