// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * pr_problem.cuh
 *
 * @brief GPU Storage management Structure for PageRank Problem Data
 */

#pragma once

#include <gunrock/util/track_utils.cuh>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/oprtr/1D_oprtr/for_all.cuh>

namespace gunrock {
namespace app {
namespace pr {

/**
 * @brief Speciflying parameters for PR Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));
  GUARD_CU(parameters.Use<bool>(
      "normalize",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      true, "Whether to normalize ranking values.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "compensate",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      true, "Whether to compensate for zero-degree vertices.", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "scale",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to scale the ranking values during computation.",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<double>(
      "delta",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      0.85, "Damping factor of PageRank.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<double>(
      "threshold",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      0.01, "Error threshold of PageRank.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int64_t>(
      "max-iter",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      50, "Maximum number of PageRank iterations.", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief PageRank Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _ValueT  Type of ranking values
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CscT CscT;
  typedef typename GraphT::CooT CooT;
  typedef typename GraphT::GpT GpT;
  typedef _ValueT ValueT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // Helper structures

  /**
   * @brief Data slice structure which contains PR problem specific data.
   */
  struct DataSlice : BaseDataSlice {
    // PR-specific storage arrays
    util::Array1D<SizeT, ValueT> rank_curr;  // Ping-pong ranking values
    util::Array1D<SizeT, ValueT> rank_next;  // Ping-pong ranking values
    util::Array1D<SizeT, ValueT>
        rank_temp;  // Temp ranking values for neighborreduce
    util::Array1D<SizeT, ValueT> rank_temp2;  // Another temp ranking values
    util::Array1D<SizeT, SizeT> degrees;      // Out-degree for each vertex
    util::Array1D<SizeT, VertexT> node_ids;
    util::Array1D<SizeT, VertexT> local_vertices;
    util::Array1D<SizeT, VertexT> *remote_vertices_out;
    util::Array1D<SizeT, VertexT> *remote_vertices_in;

    SizeT org_nodes;  // Number of vertices in the orginal graph
    bool normalize;   // Whether to normalize the ranking value
    bool compensate;  // Whether to compensate for zero-degree vertices
    bool scale;       // Whether to scale the ranking values during computation
    bool pull;        // Whether to use pull direction PR

    ValueT threshold;   // Threshold for ranking errors
    ValueT delta;       // Damping factor
    SizeT max_iter;     // Maximum number of PR iterations
    ValueT init_value;  // Initial ranking value
    ValueT reset_value;
    VertexT src_node;  // Source vertex for personalized PageRank

    bool to_continue;
    SizeT num_updated_vertices;
    bool final_event_set;

    DataSlice *data_slices;

    util::Array1D<int, SizeT> in_counters;
    util::Array1D<int, SizeT> out_counters;
    util::Array1D<uint64_t, char> cub_sort_storage;
    util::Array1D<SizeT, VertexT> temp_vertex;

    /*
     * @brief Default constructor
     */
    DataSlice()
        : BaseDataSlice(),
          org_nodes(0),
          normalize(true),
          compensate(true),
          scale(false),
          pull(false),
          threshold(0),
          delta(0),
          init_value(0),
          reset_value(0),
          src_node(util::PreDefinedValues<VertexT>::InvalidValue),
          to_continue(true),
          max_iter(0),
          num_updated_vertices(0),
          final_event_set(false),
          remote_vertices_in(NULL),
          remote_vertices_out(NULL) {
      rank_curr.SetName("rank_curr");
      rank_next.SetName("rank_next");
      rank_temp.SetName("rank_temp");
      rank_temp2.SetName("rank_temp2");
      degrees.SetName("degrees");
      node_ids.SetName("node_ids");
      local_vertices.SetName("local_vertices");
      in_counters.SetName("in_counters");
      out_counters.SetName("out_counters");
      cub_sort_storage.SetName("cub_sort_storage");
      temp_vertex.SetName("temp_vertex");
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

      GUARD_CU(rank_curr.Release(target));
      GUARD_CU(rank_next.Release(target));
      GUARD_CU(rank_temp.Release(target));
      GUARD_CU(rank_temp2.Release(target));
      GUARD_CU(degrees.Release(target));
      GUARD_CU(node_ids.Release(target));
      GUARD_CU(in_counters.Release(target));
      GUARD_CU(out_counters.Release(target));
      GUARD_CU(cub_sort_storage.Release(target));
      GUARD_CU(temp_vertex.Release(target));

      if (remote_vertices_in != NULL) {
        for (int peer = 0; peer < this->num_gpus; peer++) {
          GUARD_CU(remote_vertices_in[peer].Release(target));
        }
        delete[] remote_vertices_in;
        remote_vertices_in = NULL;
      }
      if (remote_vertices_out != NULL) {
        for (int peer = 0; peer < this->num_gpus; peer++) {
          GUARD_CU(remote_vertices_out[peer].Release(target));
        }
        delete[] remote_vertices_out;
        remote_vertices_out = NULL;
      }

      GUARD_CU(BaseDataSlice::Release(target));
      return retval;
    }

    /**
     * @brief initializing PR-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, SizeT org_nodes, int num_gpus = 1,
                     int gpu_idx = 0, util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = sub_graph.nodes;
      SizeT edges = sub_graph.edges;
      this->org_nodes = org_nodes;

      util::PrintMsg("nodes = " + std::to_string(nodes));
      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      GUARD_CU(rank_curr.Allocate(nodes, target));
      GUARD_CU(rank_next.Allocate(nodes, target));
      if (pull) {
        GUARD_CU(rank_temp.Allocate(edges, target));
        GUARD_CU(rank_temp2.Allocate(nodes, target));
      }
      GUARD_CU(degrees.Allocate(nodes + 1, target));
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");

      // Compute degrees
      // auto &sub_graph = this -> sub_graph[0];
      if (num_gpus == 1) {
        GUARD_CU(local_vertices.Allocate(nodes, target));
        GUARD_CU(local_vertices.ForAll(
            [] __host__ __device__(VertexT * l_vertices, const SizeT &pos) {
              l_vertices[pos] = pos;
            },
            nodes, target, this->stream));
      } else {
        GUARD_CU(out_counters.Allocate(num_gpus, util::HOST));
        GUARD_CU(in_counters.Allocate(num_gpus, util::HOST));
        remote_vertices_out = new util::Array1D<SizeT, VertexT>[num_gpus];
        remote_vertices_in = new util::Array1D<SizeT, VertexT>[num_gpus];
        for (int peer = 0; peer < num_gpus; peer++) {
          out_counters[peer] = 0;
          remote_vertices_out[peer].SetName("remote_vetices_out[]");
          remote_vertices_in[peer].SetName("remote_vertces_in []");
        }

        for (VertexT v = 0; v < nodes; v++)
          out_counters[sub_graph.GpT::partition_table[v]]++;

        for (int peer = 0; peer < num_gpus; peer++) {
          GUARD_CU(remote_vertices_out[peer].Allocate(out_counters[peer],
                                                      util::HOST | target));
          out_counters[peer] = 0;
        }

        for (VertexT v = 0; v < nodes; v++) {
          int target = sub_graph.GpT::partition_table[v];
          remote_vertices_out[target][out_counters[target]] = v;
          out_counters[target]++;
        }

        for (int peer = 0; peer < num_gpus; peer++) {
          GUARD_CU(remote_vertices_out[peer].Move(util::HOST, target));
        }
        GUARD_CU(local_vertices.SetPointer(
            remote_vertices_out[0].GetPointer(util::HOST), out_counters[0],
            util::HOST));
        GUARD_CU(
            local_vertices.SetPointer(remote_vertices_out[0].GetPointer(target),
                                      out_counters[0], target));
      }

      if (pull) {
        GUARD_CU(sub_graph.CscT::Move(util::HOST, target, this->stream));
      } else {
        GUARD_CU(sub_graph.CooT::Move(util::HOST, target, this->stream));
      }
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");

      if (GraphT::FLAG & gunrock::graph::HAS_CSR) {
        GUARD_CU(degrees.ForAll(
            [sub_graph] __host__ __device__(SizeT * degrees, const SizeT &pos) {
              degrees[pos] = sub_graph.GetNeighborListLength(pos);
            },
            sub_graph.nodes, target, this->stream));
      } else if (GraphT::FLAG &
                 (gunrock::graph::HAS_COO | gunrock::graph::HAS_CSC)) {
        bool pull = this->pull;
        GUARD_CU(degrees.ForEach(
            [] __host__ __device__(SizeT & degree) { degree = 0; }, nodes + 1,
            target, this->stream));

        GUARD_CU(degrees.ForAll(
            [sub_graph, nodes, pull] __host__ __device__(SizeT * degrees,
                                                         const SizeT &e) {
              VertexT src, dest;
              if (pull) {
                sub_graph.CscT::GetEdgeSrcDest(e, src, dest);
                SizeT old_val = atomicAdd(degrees + dest, 1);
                // if (util::isTracking(dest))
                //    printf("degree[%d] <- %d, edge %d : %d -> %d\n",
                //        dest, old_val + 1, e, src, dest);
              } else {
                sub_graph.CooT::GetEdgeSrcDest(e, src, dest);
                SizeT old_val = atomicAdd(degrees + src, 1);
                // if (util::isTracking(src))
                //    printf("degree[%d] <- %d, edge %d : %d -> %d\n",
                //        src, old_val + 1, e, src, dest);
              }
            },
            sub_graph.edges, target, this->stream));

        // GUARD_CU(oprtr::ForAll((VertexT*)NULL,
        //    [degrees]
        //    __host__ __device__ (VertexT *dummy, const SizeT &e)
        //{
        //    printf("degree[42029] = %d\n", degrees[42029]);
        //}, 1, target, this -> stream));
      }

      return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;

      // TODO: Move to EnactorSlice::Reset()
      /*if (this -> num_gpus > 1)
      for (int peer = 0; peer < this -> num_gpus; peer++)
      {
          if (retval = this -> keys_out[peer].Release()) return retval;
          if (retval = this -> keys_in[0][peer].Release()) return retval;
          if (retval = this -> keys_in[1][peer].Release()) return retval;
      }*/

      // Ensure data allocation is sufficient
      GUARD_CU(rank_curr.EnsureSize_(nodes, target));
      GUARD_CU(rank_next.EnsureSize_(nodes, target));
      // GUARD_CU(degrees  .EnsureSize_(nodes, target));

      // Initial rank_next = 0
      init_value = normalize ? (scale ? 1.0 : (1.0 / (ValueT)(org_nodes)))
                             : (1.0 - delta);
      reset_value = normalize ? (scale ? (1.0 - delta)
                                       : ((1.0 - delta) / (ValueT)(org_nodes)))
                              : (1.0 - delta);

      bool &normalize = this->normalize;
      ValueT &delta = this->delta;
      GUARD_CU(rank_next.ForEach(
          [normalize, delta] __host__ __device__(ValueT & rank) {
            rank = normalize ? (ValueT)0.0 : (ValueT)(1.0 - delta);
          },
          nodes, target, this->stream));

      ValueT &init_value = this->init_value;
      GUARD_CU(rank_curr.ForAll(
          degrees,
          [init_value] __host__ __device__(ValueT * ranks, SizeT * degrees_,
                                           const SizeT &v) {
            SizeT degree = degrees_[v];
            ranks[v] = (degree == 0) ? init_value : (init_value / degree);
            // if (v == 42029)
            //    printf("rank[%d] = %f = %f / %d\n",
            //       v, ranks[v], init_value, degree);
          },
          nodes, target, this->stream));

      this->to_continue = true;
      this->final_event_set = false;
      // this -> PR_queue_length = 1;
      this->num_updated_vertices = 1;

      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // whether to use the scaling feature
  // bool scaled;

  // Methods

  /**
   * @brief PRProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief PRProblem default destructor
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
   * @brief Copy result ranking values and vertex orders to host-side vectors.
   * @param[out] h_node_id host vector to store node Vertex ID.
   * @param[out] h_rank host vector to store page rank values.
   * @param[in]  target where the results are stored
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(VertexT *h_node_ids, ValueT *h_ranks,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    auto &data_slice = data_slices[0][0];
    SizeT nodes = this->org_graph->nodes;

    if (target == util::DEVICE) {
      GUARD_CU(util::SetDevice(this->gpu_idx[0]));
      data_slice.rank_curr.SetPointer(h_ranks, nodes, util::HOST);
      data_slice.node_ids.SetPointer(h_node_ids, nodes, util::HOST);
      GUARD_CU(data_slice.rank_curr.Move(util::DEVICE, util::HOST));
      GUARD_CU(data_slice.node_ids.Move(util::DEVICE, util::HOST));
    } else if (target == util::HOST) {
      GUARD_CU(data_slice.rank_curr.ForEach(
          h_ranks,
          [] __host__ __device__(const ValueT &rank, ValueT &h_rank) {
            h_rank = rank;
          },
          nodes, util::HOST));
      GUARD_CU(data_slice.node_ids.ForEach(
          h_node_ids,
          [] __host__ __device__(const VertexT &node_id, VertexT &h_node_id) {
            h_node_id = node_id;
          },
          nodes, util::HOST));
    }
    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that PageRank processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      // GUARD_CU(this -> graph_slices[gpu] -> out_degrees    .Release());
      // GUARD_CU(this -> graph_slices[gpu] -> original_vertex.Release());
      // GUARD_CU(this -> graph_slices[gpu] -> convertion_table.Release());

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));
      auto &data_slice = data_slices[gpu][0];
      data_slice.normalize = this->parameters.template Get<bool>("normalize");
      data_slice.compensate = this->parameters.template Get<bool>("compensate");
      data_slice.scale = this->parameters.template Get<bool>("scale");
      data_slice.pull = this->parameters.template Get<bool>("pull");
      data_slice.threshold = this->parameters.template Get<ValueT>("threshold");
      data_slice.delta = this->parameters.template Get<ValueT>("delta");
      data_slice.max_iter = this->parameters.template Get<SizeT>("max-iter");

      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], graph.nodes,
                               this->num_gpus, this->gpu_idx[gpu], target,
                               this->flag));
    }

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (target & util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
        GUARD_CU2(cudaStreamSynchronize(data_slices[gpu]->stream),
                  "cudaStreamSynchronize failed");
      }
    }
    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(VertexT src = util::PreDefinedValues<VertexT>::InvalidValue,
                    util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      auto &data_slice = data_slices[gpu][0];
      data_slice.src_node = src;

      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slice.Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));

      if (gpu == 0 && this->num_gpus > 1) {
        for (int peer = 1; peer < this->num_gpus; peer++) {
          GUARD_CU(data_slice.remote_vertices_in[peer].Move(
              util::HOST, target, data_slice.in_counters[peer]));
        }
      }
    }

    return retval;
  }

  /** @} */
};  // Problem

}  // namespace pr
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
