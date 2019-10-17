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
#include <unordered_set>

//KNN includes
#include <gunrock/app/knn/knn_enactor.cuh>
#include <gunrock/app/knn/knn_test.cuh>

//#define SNN_DEBUG 1

#ifdef SNN_DEBUG
#define debug(a...) printf(a)
#else
#define debug(a...)
#endif

namespace gunrock {
namespace app {
namespace snn {

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
 * @brief SNN Problem structure.
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
    util::Array1D<SizeT, ValueT> distances;
    util::Array1D<SizeT, SizeT> core_point_mark_0;
    util::Array1D<SizeT, SizeT> core_point_mark;
    util::Array1D<SizeT, SizeT> core_points;
    util::Array1D<SizeT, SizeT> cluster_id;
    util::Array1D<SizeT, SizeT> snn_density;
    util::Array1D<SizeT, SizeT, util::PINNED> core_points_counter;

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

    // Perform SNN if enabled
    bool snn;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      keys.SetName("keys");
      distances.SetName("distances");
      core_point_mark.SetName("core_point_mark");
      core_point_mark_0.SetName("core_point_mark_0");
      core_points.SetName("core_points");
      cluster_id.SetName("cluster_id");
      snn_density.SetName("snn_density");

      knns.SetName("knns");
      core_points_counter.SetName("core_points_counter");

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
      GUARD_CU(core_point_mark_0.Release(target));
      GUARD_CU(core_point_mark.Release(target));
      GUARD_CU(core_points_counter.Release(target | util::HOST));
      GUARD_CU(cluster_id.Release(target));
      GUARD_CU(snn_density.Release(target));

      GUARD_CU(knns.Release(target));

      GUARD_CU(cub_temp_storage.Release(target));
      GUARD_CU(keys_out.Release(target));
      GUARD_CU(distances_out.Release(target));

      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing snn-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus, int gpu_idx, SizeT k,
                     bool snn_, util::Location target, ProblemFlag flag) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      SizeT nodes = sub_graph.nodes;
      SizeT edges = sub_graph.edges;

      // Perform SNN
      snn = snn_;

      // Point ()
      //GUARD_CU(keys.Allocate(edges, target));
      //GUARD_CU(distances.Allocate(edges, target));
      GUARD_CU(core_point_mark_0.Allocate(nodes, target));
      GUARD_CU(core_point_mark.Allocate(nodes, target));
      GUARD_CU(cluster_id.Allocate(nodes, target));
      GUARD_CU(snn_density.Allocate(nodes, target));

      // k-nearest neighbors
      //GUARD_CU(knns.Allocate(k * nodes, target));

      GUARD_CU(cub_temp_storage.Allocate(1, target));
      GUARD_CU(core_points_counter.Allocate(1, target | util::HOST));

      //GUARD_CU(keys_out.Allocate(edges, target));
      //GUARD_CU(distances_out.Allocate(edges, target));

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
      GUARD_CU(snn_density.EnsureSize_(nodes, target));
      GUARD_CU(snn_density.ForAll(
          [nodes] __host__ __device__(SizeT * s, const SizeT &p) { s[p] = 0; },
          nodes, util::DEVICE, this->stream));
      //GUARD_CU(keys.EnsureSize_(edges, target));
      GUARD_CU(distances.EnsureSize_(edges, target));

      GUARD_CU(cluster_id.EnsureSize_(nodes, target));
      GUARD_CU(cluster_id.ForAll(
          [nodes] __host__ __device__(SizeT * c, const SizeT &p) { c[p] = p; },
          nodes, util::DEVICE, this->stream));

      GUARD_CU(core_point_mark_0.EnsureSize_(nodes, target));
      GUARD_CU(core_point_mark_0.ForAll(
          [] __host__ __device__(SizeT * c, const SizeT &p) { c[p] = 0; },
          nodes, util::DEVICE, this->stream));

      GUARD_CU(core_point_mark.EnsureSize_(nodes, target));
      GUARD_CU(core_point_mark.ForAll(
          [] __host__ __device__(SizeT * c, const SizeT &p) { c[p] = 0; },
          nodes, util::DEVICE, this->stream));

      GUARD_CU(core_points.EnsureSize_(nodes, target));
      GUARD_CU(core_points.ForAll(
          [] __host__ __device__(SizeT * c, const SizeT &p) {
            c[p] = util::PreDefinedValues<SizeT>::InvalidValue;
          },
          nodes, util::DEVICE, this->stream));

      GUARD_CU(core_points_counter.EnsureSize_(1, target | util::HOST));
      GUARD_CU(core_points_counter.ForAll(
          [] __host__ __device__(SizeT * c, const SizeT &p) { c[p] = 0; }, 1,
          util::DEVICE, this->stream));
/*
      // K-Nearest Neighbors
      GUARD_CU(knns.EnsureSize_(k * nodes, target));
      GUARD_CU(knns.ForAll(
          [] __host__ __device__(SizeT * k_, const SizeT &p) { 
            k_[p] = util::PreDefinedValues<SizeT>::InvalidValue;
          },
          k * nodes, util::DEVICE, this->stream));
*/
      return retval;
    }
  };  // DataSlice

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;
  typedef knn::Problem<GraphT, FLAG> KnnProblemT;
  KnnProblemT knn_problem;
  SizeT k;
  bool snn;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief snn default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), 
        knn_problem(_parameters, _flag),
        data_slices(NULL) {
    k = _parameters.Get<int>("k");
    snn = _parameters.Get<bool>("snn");
  }

  /**
   * @brief snn default destructor
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
    for (int i = 0; i < this->num_gpus; i++){
      GUARD_CU(data_slices[i].Release(target));
      GUARD_CU(knn_problem.data_slices[i].Release(target));
    }

    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(knn_problem.Release(target));
    GUARD_CU(BaseProblem::Release(target));
    return retval;
  }

  /**
   * @brief Copy result distancess computed on GPUs back to host-side arrays.
...
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(SizeT nodes, SizeT k, /*SizeT *h_knns,*/ SizeT *h_cluster,
                      SizeT *h_core_point_counter, SizeT *h_cluster_counter,
                      bool snn = true, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    auto &data_slice = data_slices[0][0];

    if (this->num_gpus == 1) {
      // Set device
      if (target == util::DEVICE) {
        // Extract SNN clusters
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));
        if (snn) {
          GUARD_CU(
              data_slice.cluster_id.SetPointer(h_cluster, nodes, util::HOST));
          GUARD_CU(data_slice.cluster_id.Move(util::DEVICE, util::HOST));

          SizeT *h_core_points = (SizeT *)malloc(sizeof(SizeT) * nodes);
          GUARD_CU(data_slice.core_points.SetPointer(h_core_points, nodes,
                                                     util::HOST));
          GUARD_CU(data_slice.core_points.Move(util::DEVICE, util::HOST));

          h_cluster_counter[0] = 0;

          std::unordered_set<SizeT> set;
          printf("gpu clusters: ");
          for (SizeT i = 0; i < nodes; ++i) {
            if (util::isValid(h_core_points[i])) {
              SizeT c = h_cluster[i];
              if (set.find(c) == set.end()) {
                set.insert(c);
                printf("%d ", c);
                ++h_cluster_counter[0];
              }
            }
          }
          printf("\n");

          delete[] h_core_points;

          h_core_point_counter[0] = data_slice.core_points_counter[0];
          printf("Core points: %d, Clusters: %d\n", h_core_point_counter[0],
                 h_cluster_counter[0]);
        }

        // Extract KNNs
        // GUARD_CU(data_slice.knns.SetPointer(h_knns, nodes * k, util::HOST));
        // GUARD_CU(data_slice.knns.Move(util::DEVICE, util::HOST));
      }

    } else if (target == util::HOST) {
      // auto &data_slice = data_slices[0][0];
      GUARD_CU(data_slice.cluster_id.ForEach(
          h_cluster,
          [] __host__ __device__(const SizeT &device_val, SizeT &host_val) {
            host_val = device_val;
          },
          nodes, util::HOST));

      /*GUARD_CU(data_slice.knns.ForEach(
          h_knns,
          [] __host__ __device__(const SizeT &device_val, SizeT &host_val) {
            host_val = device_val;
          },
          nodes * k, util::HOST));*/
    }

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that SNN processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, SizeT k,
                   util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];
    GUARD_CU(knn_problem.Init(graph, target));

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], k, this->snn, target,
                               this->flag));
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

}  // namespace snn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
