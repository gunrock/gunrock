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
    
    util::Array1D<SizeT, SizeT> snn_density;
    util::Array1D<SizeT, SizeT> cluster_id;
    util::Array1D<SizeT, SizeT> core_point_mark_0;
    util::Array1D<SizeT, SizeT> core_point_mark;
    util::Array1D<SizeT, SizeT, util::PINNED> core_points_counter;
    util::Array1D<SizeT, SizeT, util::PINNED> flag;
    util::Array1D<SizeT, SizeT> noise_points;
    util::Array1D<SizeT, SizeT> core_points;
    util::Array1D<SizeT, char> visited;

    // Nearest Neighbors
    util::Array1D<SizeT, SizeT> knns;

    // Number of neighbors
    SizeT num_points;
    SizeT k;
    SizeT eps;
    SizeT min_pts;

    // CUB Related storage
    util::Array1D<uint64_t, char> cub_temp_storage;
    util::Array1D<SizeT, SizeT> knns_out;
    util::Array1D<SizeT, SizeT> offsets;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      knns.SetName("knns");
      core_point_mark.SetName("core_point_mark");
      core_point_mark_0.SetName("core_point_mark_0");
      core_points.SetName("core_points");
      core_points_counter.SetName("core_points_counter");
      flag.SetName("flag");
      noise_points.SetName("noise_points");
      cluster_id.SetName("cluster_id");
      snn_density.SetName("snn_density");
      cub_temp_storage.SetName("cub_temp_storage");
      knns_out.SetName("knns_out");
      offsets.SetName("offsets");
      visited.SetName("visited");
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

      GUARD_CU(core_point_mark_0.Release(target));
      GUARD_CU(core_point_mark.Release(target));
      GUARD_CU(core_points_counter.Release(target | util::HOST));
      GUARD_CU(flag.Release(target | util::HOST));
      GUARD_CU(noise_points.Release(target | util::HOST));
      GUARD_CU(cluster_id.Release(target));
      GUARD_CU(snn_density.Release(target));
      GUARD_CU(knns.Release(target));
      GUARD_CU(cub_temp_storage.Release(target));
      GUARD_CU(knns_out.Release(target));
      GUARD_CU(BaseDataSlice ::Release(target));
      GUARD_CU(offsets.Release(target));
      GUARD_CU(visited.Release(target));
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
    cudaError_t Init(GraphT &sub_graph,  
            SizeT num_points_, SizeT k_, SizeT eps_, SizeT min_pts_, 
            int num_gpus = 1, int gpu_idx = 0,
            util::Location target=util::DEVICE, 
            ProblemFlag flag_ = Problem_None) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag_));

      num_points = num_points_;
      k = k_;
      eps = eps_;
      min_pts = min_pts_;

      GUARD_CU(knns.Allocate(k * num_points, target));
      GUARD_CU(snn_density.Allocate(num_points, target));
      GUARD_CU(cluster_id.Allocate(num_points, target));
      GUARD_CU(core_points.Allocate(num_points, target));
      GUARD_CU(core_point_mark_0.Allocate(num_points, target));
      GUARD_CU(core_point_mark.Allocate(num_points, target));
      GUARD_CU(core_points_counter.Allocate(1, target | util::HOST));
      GUARD_CU(flag.Allocate(1, target | util::HOST));
      GUARD_CU(noise_points.Allocate(1, target | util::HOST));
      GUARD_CU(cub_temp_storage.Allocate(1, target));
      GUARD_CU(knns_out.Allocate(k * num_points, target));
      GUARD_CU(offsets.Allocate(num_points+1, target));
      GUARD_CU(visited.Allocate(num_points, target));
//      if (target & util::DEVICE) {
//        GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
//      }

      return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(SizeT* h_knns, util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      auto &cluster_id = this->cluster_id;
      typedef typename GraphT::CsrT CsrT;

      auto k_number = k;

      // Ensure data are allocated
      GUARD_CU(snn_density.EnsureSize_(num_points, target));
      GUARD_CU(snn_density.ForAll(
          [] 
          __host__ __device__(SizeT * s, const SizeT &p) { s[p] = 0; },
          num_points, util::DEVICE, this->stream));
      
      GUARD_CU(cluster_id.EnsureSize_(num_points, target));
      GUARD_CU(cluster_id.ForAll(
          [] 
          __host__ __device__(SizeT * c, const SizeT &p) { 
            c[p] = util::PreDefinedValues<SizeT>::InvalidValue;
          }, num_points, util::DEVICE, this->stream));

      GUARD_CU(core_point_mark_0.EnsureSize_(num_points, target));
      GUARD_CU(core_point_mark_0.ForAll(
          [] __host__ __device__(SizeT * c, const SizeT &p) { c[p] = 0; },
          num_points, util::DEVICE, this->stream));

      GUARD_CU(core_point_mark.EnsureSize_(num_points, target));
      GUARD_CU(core_point_mark.ForAll(
          [] __host__ __device__(SizeT * c, const SizeT &p) { c[p] = 0; },
          num_points, util::DEVICE, this->stream));

      GUARD_CU(core_points.EnsureSize_(num_points, target));
      GUARD_CU(core_points.ForAll(
          [] __host__ __device__(SizeT * c, const SizeT &p) {
            c[p] = util::PreDefinedValues<SizeT>::InvalidValue;
          },
          num_points, util::DEVICE, this->stream));

      GUARD_CU(visited.EnsureSize_(num_points, target));
      GUARD_CU(visited.ForAll(
          [] __host__ __device__ (char *v, const SizeT &p){
            v[p] = (char)0;
          },
          num_points, util::DEVICE, this->stream));

      GUARD_CU(flag.EnsureSize_(1, target | util::HOST));
      GUARD_CU(core_points_counter.EnsureSize_(1, target | util::HOST));
      GUARD_CU(core_points_counter.ForAll(
          [] __host__ __device__(SizeT * c, const SizeT &p) { c[p] = 0; }, 1,
          util::DEVICE, this->stream));

      GUARD_CU(noise_points.EnsureSize_(1, target | util::HOST));
      GUARD_CU(noise_points.ForAll(
          [] __host__ __device__(SizeT * c, const SizeT &p) { c[p] = 0; }, 1,
          util::DEVICE, this->stream));
      
      GUARD_CU(knns_out.EnsureSize_(num_points*k, target));

      GUARD_CU(offsets.EnsureSize_(num_points+1, target));
      GUARD_CU(offsets.ForAll(
        [k_number] __host__ __device__ (SizeT *ro, const SizeT &pos){
            ro[pos] = pos*k_number;
        }, num_points+1, util::DEVICE, this->stream));

      GUARD_CU(util::SetDevice(this->gpu_idx));
      GUARD_CU(knns.SetPointer(h_knns, num_points*k, util::HOST));
      GUARD_CU(knns.Move(util::HOST, target, num_points*k, 0, this->stream));

      return retval;
    }
  };  // DataSlice struct

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;
  SizeT num_points;
  SizeT k;
  SizeT eps;
  SizeT min_pts;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief snn default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), 
        data_slices(NULL) {
    num_points = _parameters.Get<int>("n");
    k = _parameters.Get<int>("k");
    eps = _parameters.Get<int>("eps");
    min_pts = _parameters.Get<int>("min-pts");
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
    }

    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(BaseProblem::Release(target));
    return retval;
  }

  /**
   * @brief Copy snn results computed on GPUs back to host-side arrays.
...
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(SizeT num_points, SizeT k, SizeT *h_cluster,
                      SizeT *h_core_point_counter, SizeT *h_noise_point_counter,
                      SizeT *h_cluster_counter,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    auto &data_slice = data_slices[0][0];

    auto cluster_id = data_slice.cluster_id;
    auto core_points = data_slice.core_points;
    auto noise_points = data_slice.noise_points;

    if (this->num_gpus == 1) {
      // Set device
      if (target == util::DEVICE) {
        // Extract SNN clusters
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        GUARD_CU(cluster_id.SetPointer(h_cluster, num_points, util::HOST));
        GUARD_CU(cluster_id.Move(util::DEVICE, util::HOST));
        
        SizeT *h_core_points = (SizeT*)malloc(sizeof(SizeT) * num_points);
        GUARD_CU(core_points.SetPointer(h_core_points,num_points,util::HOST));
        GUARD_CU(core_points.Move(util::DEVICE, util::HOST));
        
        h_cluster_counter[0] = 0;
        
        std::unordered_set<SizeT> set;
        debug("gpu clusters: ");
        for (SizeT i = 0; i < num_points; ++i) {
          if (util::isValid(h_cluster[i])) {
            SizeT c = h_cluster[i];
            if (set.find(c) == set.end()) {
              set.insert(c);
              debug("%d ", c);
              ++h_cluster_counter[0];
            }
          }
        }
        debug("\n");
/*
        printf("gpu clusters ids: ");
        for (auto x :set){
            printf("%d ", x);
        }
        printf("\n");*/
          
        delete[] h_core_points;

        h_core_point_counter[0] = data_slice.core_points_counter[0];
        GUARD_CU(noise_points.SetPointer(h_noise_point_counter, 1, util::HOST));
        GUARD_CU(noise_points.Move(util::DEVICE, util::HOST));
        util::PrintMsg("[GPU] Core points: " + 
                std::to_string(h_core_point_counter[0]) +
                " Noise points: " + 
                std::to_string(h_noise_point_counter[0]) +
                ". Clusters: " + std::to_string(h_cluster_counter[0]),false);
      }
    } else if (target == util::HOST) {
      GUARD_CU(data_slice.cluster_id.ForEach(
          h_cluster,
          [] __host__ __device__(const SizeT &device_val, SizeT &host_val) {
            host_val = device_val;
          },
          num_points, util::HOST));
    }

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that SNN processes on
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
      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));
      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_points, 
                               this->k, this->eps, this->min_pts, 
                               this->num_gpus, this->gpu_idx[gpu], 
                               target, this->flag));
    }

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(SizeT* h_knns, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    // Reset data slices
    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(
          data_slices[gpu]->Reset(h_knns, target));
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
