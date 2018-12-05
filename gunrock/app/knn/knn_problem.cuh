// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * knn_problem.cuh
 *
 * @brief GPU Storage management Structure for knn Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace knn {

/**
 * @brief Speciflying parameters for knn Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  return retval;
}

/**
 * @brief KNN Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // ----------------------------------------------------------------
  // Dataslice structure

  /**
   * @brief Data structure containing problem specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // struct Point()
    util::Array1D<SizeT, SizeT> keys;
    util::Array1D<SizeT, VertexT> distances;
    util::Array1D<SizeT, SizeT> adj;
    util::Array1D<SizeT, SizeT> core_point;
    util::Array1D<SizeT, SizeT *> cluster;
    util::Array1D<SizeT, SizeT> cluster_id;

    // Nearest Neighbors
    util::Array1D<SizeT, SizeT> knns;

    // Number of neighbors
    SizeT k;
    SizeT eps;
    SizeT min_pts;

    // Reference Point
    VertexT point_x;
    VertexT point_y;

    // CUB Related storage
    util::Array1D<uint64_t, char> cub_temp_storage;

    // Sorted
    util::Array1D<SizeT, SizeT> keys_out;
    util::Array1D<SizeT, VertexT> distances_out;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      keys.SetName("keys");
      distances.SetName("distances");
      adj.SetName("adj");
      core_point.SetName("core_point");
      cluster.SetName("cluster");
      cluster_id.SetName("cluster_id");

      knns.SetName("knns");

      cub_temp_storage.SetName("cub_temp_storage");
      keys_out.SetName("keys_out");
      distances_out.SetName("distances_out");
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

      GUARD_CU(keys.Release(target));
      GUARD_CU(distances.Release(target));
      GUARD_CU(adj.Release(target));
      GUARD_CU(core_point.Release(target));
      GUARD_CU(cluster_id.Release(target));
      GUARD_CU(cluster.Release(target));

      GUARD_CU(knns.Release(target));

      GUARD_CU(cub_temp_storage.Release(target));
      GUARD_CU(keys_out.Release(target));
      GUARD_CU(distances_out.Release(target));

      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing sssp-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus, int gpu_idx, SizeT k,
                     util::Location target, ProblemFlag flag) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      SizeT nodes = sub_graph.nodes;
      SizeT edges = sub_graph.edges;
      // Point ()
      GUARD_CU(keys.Allocate(edges, target));
      GUARD_CU(distances.Allocate(edges, target));
      GUARD_CU(adj.Allocate(nodes * nodes, target));
      GUARD_CU(core_point.Allocate(nodes, target));
      GUARD_CU(cluster.Allocate(nodes, target));
      GUARD_CU(cluster_id.Allocate(nodes * nodes, target));

      // k-nearest neighbors
      GUARD_CU(knns.Allocate(k * nodes, target));

      GUARD_CU(cub_temp_storage.Allocate(1, target));

      GUARD_CU(keys_out.Allocate(edges, target));
      GUARD_CU(distances_out.Allocate(edges, target));

      if (target & util::DEVICE) {
        GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
      }
      return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(VertexT point_x_, VertexT point_y_, SizeT k_, SizeT eps_,
                      SizeT min_pts_, util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;
      SizeT edges = this->sub_graph->edges;
      auto &cluster_id = this->cluster_id;
      auto &graph = this->sub_graph[0];
      typedef typename GraphT::CsrT CsrT;

      // Number of knns
      this->k = k_;
      this->eps = eps_;
      this->min_pts = min_pts_;

      // Reference point
      this->point_x = point_x_;
      this->point_y = point_y_;

      // Ensure data are allocated
      GUARD_CU(keys.EnsureSize_(edges, target));
      GUARD_CU(distances.EnsureSize_(edges, target));
      GUARD_CU(cluster_id.EnsureSize_(nodes * nodes, target));
      GUARD_CU(cluster_id.ForAll(
          [nodes] __host__ __device__(SizeT * c, const SizeT &p) {
            if (p < nodes)
              c[p] = p;
            else
              c[p] = util::PreDefinedValues<SizeT>::InvalidValue;
          },
          nodes * nodes, util::DEVICE, this->stream));
      GUARD_CU(cluster.EnsureSize_(nodes, target));
      GUARD_CU(cluster.ForAll(
          [cluster_id, nodes] __host__ __device__(SizeT * *c, const SizeT &p) {
            c[p] = &cluster_id[p];
          },
          nodes, util::DEVICE, this->stream));
      GUARD_CU(core_point.EnsureSize_(nodes, target));
      GUARD_CU(core_point.ForAll(
          [] __host__ __device__(SizeT * c, const SizeT &p) { c[p] = 0; },
          nodes, util::DEVICE, this->stream));
      GUARD_CU(adj.EnsureSize_(nodes * nodes, target));
      GUARD_CU(adj.ForAll(
          [graph] __host__ __device__(SizeT * a, const SizeT &pos) {
            auto nodes = graph.nodes;
            for (auto n = 0; n < nodes; ++n) a[nodes * pos + n] = 0;
            auto e_start = graph.CsrT::GetNeighborListOffset(pos);
            auto num_neighbors = graph.CsrT::GetNeighborListLength(pos);
            for (auto e = e_start; e < e_start + num_neighbors; ++e) {
              auto n = graph.CsrT::GetEdgeDest(e);
              a[nodes * pos + n] = 1;
            }
          },
          nodes, target, this->stream));

      // K-Nearest Neighbors
      GUARD_CU(knns.EnsureSize_(k, target));

      return retval;
    }
  };  // DataSlice

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;
  SizeT k;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief knn default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {
    k = _parameters.Get<int>("k");
  }

  /**
   * @brief knn default destructor
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
   * @brief Copy result distancess computed on GPUs back to host-side arrays.
...
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(SizeT n, SizeT *h_cluster,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));
        GUARD_CU(data_slice.cluster_id.Move(util::DEVICE, util::HOST));
        for (int i = 0; i < n; ++i) {
          h_cluster[i] = data_slice.cluster_id[i];
        }
      }
    } else if (target == util::HOST) {
    }

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that SSSP processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, SizeT k,
                   util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    // <TODO> get problem specific flags from parameters, e.g.:
    // if (this -> parameters.template Get<bool>("mark-pred"))
    //    this -> flag = this -> flag | Mark_Predecessors;
    // </TODO>

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], k, target, this->flag));
    }

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(VertexT point_x, VertexT point_y, SizeT k, SizeT eps,
                    SizeT min_pts, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    // Reset data slices
    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(
          data_slices[gpu]->Reset(point_x, point_y, k, eps, min_pts, target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    return retval;
  }
};

}  // namespace knn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
