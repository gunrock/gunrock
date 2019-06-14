// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * gtf_problem.cuh
 *
 * @brief GPU Storage management Structure for Max Flow Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/oprtr/1D_oprtr/for_all.cuh>

// MF includes
#include <gunrock/app/mf/mf_enactor.cuh>
#include <gunrock/app/mf/mf_test.cuh>

#define debug_aml(a...)
//#define debug_aml(a...) {printf("%s:%d ", __FILE__, __LINE__); printf(a);\
    printf("\n");}

namespace gunrock {
namespace app {
namespace gtf {

/**
 * @brief Speciflying parameters for GTF Problem
 * @param  parameters  The util::Parameter<...> structure holding all
 *			parameter info
 * \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  // TODO: Add problem specific command-line parameter usages here, e.g.:
  GUARD_CU(parameters.Use<bool>(
      "mark-pred",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to mark predecessor info.", __FILE__, __LINE__));

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
   * @brief Data structure containing GTF-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // GTF-specific storage arrays:
    int num_nodes;
    int num_org_nodes;
    int num_edges;
    double error_threshold;  // = parameters.Get<double>("error_threshold");

    util::Array1D<SizeT, VertexT>
        next_communities;  //= new VertexT[num_nodes]; // nextlabel
    util::Array1D<SizeT, VertexT>
        curr_communities;  //= new VertexT[num_nodes]; // label
    util::Array1D<SizeT, VertexT>
        community_sizes;  //= new VertexT[num_nodes]; // nums
    util::Array1D<SizeT, ValueT>
        community_weights;  //= new ValueT [num_nodes]; // averages
    util::Array1D<SizeT, bool>
        community_active;  //= new bool   [num_nodes]; // !inactivelable
    util::Array1D<SizeT, ValueT>
        community_accus;  //  = new ValueT [num_nodes]; // values
    util::Array1D<SizeT, bool>
        vertex_active;  //    = new bool   [num_nodes]; // alive
    util::Array1D<SizeT, bool> vertex_reachabilities;  // = new bool[num_nodes];
    util::Array1D<SizeT, ValueT>
        edge_residuals;  //   = new ValueT [num_edges]; // graph
    util::Array1D<SizeT, ValueT>
        edge_flows;  //       = new ValueT [num_edges]; // edge flows
    util::Array1D<SizeT, SizeT> active;  // flag active vertices
    util::Array1D<SizeT, VertexT> num_comms;
    util::Array1D<SizeT, VertexT> previous_num_comms;  // flag active vertices
    // util::Array1D<SizeT, VertexT> num_comms;	      // flag active vertices
    util::Array1D<SizeT, SizeT> reverse;  // for storing mf h_reverse

    util::Array1D<SizeT, ValueT> Y;  // for storing mf h_reverse
    SizeT num_updated_vertices;

    VertexT source;  // source vertex
    VertexT sink;    // sink vertex

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      source = util::PreDefinedValues<VertexT>::InvalidValue;
      sink = util::PreDefinedValues<VertexT>::InvalidValue;

      num_nodes = util::PreDefinedValues<VertexT>::InvalidValue;
      num_org_nodes = util::PreDefinedValues<VertexT>::InvalidValue;
      num_edges = util::PreDefinedValues<VertexT>::InvalidValue;
      num_updated_vertices = 1;

      next_communities.SetName("next_communities");
      curr_communities.SetName("curr_communities");
      community_sizes.SetName("community_sizes");
      community_weights.SetName("community_weights");
      community_active.SetName("community_active");
      community_accus.SetName("community_accus");
      vertex_active.SetName("vertex_active");
      vertex_reachabilities.SetName("vertex_reachabilities");
      edge_residuals.SetName("edge_residuals");
      edge_flows.SetName("edge_flows");
      active.SetName("active");
      num_comms.SetName("num_comms");
      previous_num_comms.SetName("previous_num_comms");
      reverse.SetName("reverse");
      Y.SetName("Y");
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

      GUARD_CU(next_communities.Release(target));
      GUARD_CU(curr_communities.Release(target));
      GUARD_CU(community_sizes.Release(target));
      GUARD_CU(community_weights.Release(target));
      GUARD_CU(community_active.Release(target));
      GUARD_CU(community_accus.Release(target));
      GUARD_CU(vertex_active.Release(target));
      GUARD_CU(vertex_reachabilities.Release(target));
      GUARD_CU(edge_residuals.Release(target));
      GUARD_CU(edge_flows.Release(target));
      GUARD_CU(BaseDataSlice::Release(target));
      GUARD_CU(active.Release(target));
      GUARD_CU(num_comms.Release(target));
      GUARD_CU(previous_num_comms.Release(target));
      GUARD_CU(reverse.Release(target));
      GUARD_CU(Y.Release(target));
      return retval;
    }

    /**
     * @brief initializing GTF-specific Data Slice a on each gpu
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

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      // Allocate data on Gpu
      GUARD_CU(next_communities.Allocate(nodes_size, target));
      GUARD_CU(curr_communities.Allocate(nodes_size, target));
      GUARD_CU(community_sizes.Allocate(nodes_size, target));
      GUARD_CU(community_weights.Allocate(nodes_size, target));
      GUARD_CU(community_active.Allocate(nodes_size, target));
      GUARD_CU(community_accus.Allocate(nodes_size, target));
      GUARD_CU(vertex_active.Allocate(nodes_size, target));
      GUARD_CU(vertex_reachabilities.Allocate(nodes_size, target));
      GUARD_CU(edge_residuals.Allocate(edges_size, target));
      GUARD_CU(edge_flows.Allocate(edges_size, target));
      GUARD_CU(active.Allocate(1, util::HOST | target));
      GUARD_CU(num_comms.Allocate(1, target));
      GUARD_CU(previous_num_comms.Allocate(1, target));
      GUARD_CU(reverse.Allocate(edges_size, util::HOST));

      GUARD_CU(Y.Allocate(nodes_size, target));

      GUARD_CU(util::SetDevice(gpu_idx));
      GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));
      return retval;
    }  // Init Data Slice

    /**
     * @brief Reset DataSlice function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(const GraphT &graph, ValueT *h_community_accus,
                      util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;

      typedef typename GraphT::CsrT CsrT;

      debug_aml("DataSlice Reset");

      SizeT nodes_size = graph.nodes;
      SizeT edges_size = graph.edges;

      // Ensure data are allocated
      GUARD_CU(next_communities.EnsureSize_(nodes_size, target));
      GUARD_CU(curr_communities.EnsureSize_(nodes_size, target));
      GUARD_CU(community_sizes.EnsureSize_(nodes_size, target));
      GUARD_CU(community_weights.EnsureSize_(nodes_size, target));
      GUARD_CU(community_active.EnsureSize_(nodes_size, target));
      GUARD_CU(community_accus.EnsureSize_(nodes_size, target));
      GUARD_CU(vertex_active.EnsureSize_(nodes_size, target));
      GUARD_CU(vertex_reachabilities.EnsureSize_(nodes_size, target));
      GUARD_CU(edge_residuals.EnsureSize_(edges_size, target));
      GUARD_CU(edge_flows.EnsureSize_(edges_size, target));
      GUARD_CU(active.EnsureSize_(1, target | util::HOST));
      GUARD_CU(num_comms.EnsureSize_(1, target));
      GUARD_CU(previous_num_comms.EnsureSize_(1, target));
      GUARD_CU(reverse.EnsureSize_(edges_size, util::HOST));

      GUARD_CU(util::SetDevice(this->gpu_idx));

      ///////////////////////////////
      num_org_nodes = graph.nodes - 2;
      SizeT offset = graph.edges - num_org_nodes * 2;
      printf("offset is %d num edges %d \n", offset, edges_size);

      // bool* h_vertex_active = (bool*)malloc(sizeof(bool)*graph.edges);
      // bool* h_community_active = (bool*)malloc(sizeof(bool)*graph.nodes);
      // VertexT* h_curr_communities =
      // (VertexT*)malloc(sizeof(VertexT)*graph.nodes); VertexT*
      // h_next_communities = (VertexT*)malloc(sizeof(VertexT)*graph.nodes); for
      // (VertexT v = 0; v < num_org_nodes; v++)
      // {
      //     h_vertex_active   [v] = true;
      //     h_community_active[v] = true;
      //     h_curr_communities[v] = 0;
      //     h_next_communities[v] = 0; //extra
      // }

      GUARD_CU(vertex_active.ForAll(
          [] __host__ __device__(bool *v_active, const SizeT &pos) {
            v_active[pos] = true;
          },
          graph.nodes, target, this->stream));

      GUARD_CU(community_active.ForAll(
          [] __host__ __device__(bool *c_active, const SizeT &pos) {
            c_active[pos] = true;
          },
          graph.nodes, target, this->stream));

      GUARD_CU(curr_communities.ForAll(
          [] __host__ __device__(VertexT * c_communities, const SizeT &pos) {
            c_communities[pos] = 0;
          },
          graph.nodes, target, this->stream));

      GUARD_CU(next_communities.ForAll(
          [] __host__ __device__(VertexT * n_communities, const SizeT &pos) {
            n_communities[pos] = 0;
          },
          graph.nodes, target, this->stream));

      GUARD_CU(vertex_reachabilities.ForAll(
          [] __host__ __device__(bool *vertex_reachabilities,
                                 const SizeT &pos) {
            vertex_reachabilities[pos] = 0;
          },
          graph.nodes, target, this->stream));

      // GUARD_CU(community_accus.ForAll([h_community_accus]
      //    __host__ __device__(ValueT *community_accus, const SizeT &pos)
      // {
      //   community_accus[0] = h_community_accus[0];
      // }, 1, target, this -> stream));

      // GUARD_CU(vertex_active.SetPointer(h_vertex_active, num_org_nodes,
      // util::HOST)); GUARD_CU(vertex_active.Move(util::HOST, target,
      // num_org_nodes, 0, this->stream));
      // GUARD_CU(community_active.SetPointer(h_community_active, num_org_nodes,
      // util::HOST)); GUARD_CU(community_active.Move(util::HOST, target,
      // num_org_nodes, 0, this->stream));
      // GUARD_CU(curr_communities.SetPointer(h_curr_communities, num_org_nodes,
      // util::HOST)); GUARD_CU(curr_communities.Move(util::HOST, target,
      // num_org_nodes, 0, this->stream));
      // GUARD_CU(next_communities.SetPointer(h_next_communities, num_org_nodes,
      // util::HOST)); GUARD_CU(next_communities.Move(util::HOST, target,
      // num_org_nodes, 0, this->stream));
      //
      printf("h_community_accus is %f \n", h_community_accus[0]);
      GUARD_CU(community_accus.SetPointer(h_community_accus, graph.nodes,
                                          util::HOST));
      GUARD_CU(community_accus.Move(util::HOST, target, graph.nodes, 0,
                                    this->stream));

      this->num_updated_vertices = 1;

      GUARD_CU(active.ForAll(
          [] __host__ __device__(SizeT * active_, const VertexT &pos) {
            active_[pos] = 1;
          },
          1, target, this->stream));

      GUARD_CU(num_comms.ForAll(
          [] __host__ __device__(SizeT * num_comm, const VertexT &pos) {
            num_comm[pos] = 1;
          },
          1, target, this->stream));

      //////////////////////////////
      // GUARD_CU(reverse.SetPointer(h_reverse, edges_size, util::HOST));
      // GUARD_CU(reverse.Move(util::HOST, target, edges_size, 0,
      // this->stream));

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
      // free(h_vertex_active);
      // free(h_community_active);
      // free(h_curr_communities);
      // free(h_next_communities);
      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;
  typedef mf::Problem<GraphT, ValueT, FLAG> MfProblemT;
  MfProblemT mf_problem;

  // Methods

  /**
   * @brief GTFProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag),
        mf_problem(_parameters, _flag),
        data_slices(NULL) {}

  /**
   * @brief GTFProblem default destructor
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
    for (int i = 0; i < this->num_gpus; i++) {
      GUARD_CU(data_slices[i].Release(target));
      GUARD_CU(mf_problem.data_slices[i].Release(target));
    }
    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(mf_problem.Release(target));
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
  cudaError_t Extract(ValueT *h_Y, ValueT *edge_values,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    auto &data_slice = data_slices[0][0];
    SizeT vN = this->org_graph->nodes;
    SizeT vE = this->org_graph->edges;

    // Set device
    if (target == util::DEVICE) {
      printf("transfering to host!!!: %d \n", vN);
      GUARD_CU(util::SetDevice(this->gpu_idx[0]));
      GUARD_CU(data_slice.Y.SetPointer(h_Y, vN, util::HOST));
      GUARD_CU(data_slice.Y.Move(util::DEVICE, util::HOST));

      GUARD_CU(util::SetDevice(this->gpu_idx[0]));
      GUARD_CU(
          data_slice.edge_residuals.SetPointer(edge_values, vE, util::HOST));
      GUARD_CU(data_slice.edge_residuals.Move(util::DEVICE, util::HOST));
    } else if (target == util::HOST) {
      GUARD_CU(data_slice.Y.ForEach(
          h_Y,
          [] __host__ __device__(const ValueT &f, ValueT &h_f) {
            { h_f = f; }
          },
          vN, util::HOST));
    }
    return retval;
  }

  /**
   * @brief Init GTF Problem
   * @param     graph       The graph that GTF processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    debug_aml("Problem Init");
    cudaError_t retval = cudaSuccess;

    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];
    GUARD_CU(mf_problem.Init(graph, target));

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
  }  // End Init GTF Problem

  /**
   * @brief Reset Problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(GraphT &graph, ValueT *h_community_accus, SizeT *h_reverse,
                    util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    debug_aml("Problem Reset");
    auto &reverse = data_slices[0][0].reverse;
    for (auto i = 0; i < graph.edges; i++) {
      reverse[i] = h_reverse[i];
    }

    auto source_vertex = graph.nodes - 2;
    auto sink_vertex = graph.nodes - 1;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      auto &data_slice = data_slices[gpu][0];
      data_slice.source = source_vertex;
      data_slice.sink = sink_vertex;

      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(graph, h_community_accus, target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    // Filling the initial input_queue for GTF problem

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

}  // namespace gtf
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
