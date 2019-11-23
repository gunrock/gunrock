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

#include <gunrock/util/array_utils.cuh>
#include <gunrock/app/problem_base.cuh>
#include <unordered_set>

namespace gunrock {
namespace app {
namespace knn {

/**
 * @brief Speciflying parameters for KNN Problem
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
  typedef typename util::Array1D<SizeT, ValueT> ArrayT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // ----------------------------------------------------------------
  // Dataslice structure

  /**
   * @brief Data structure containing problem specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    
    util::Array1D<SizeT, ValueT> points;
    util::Array1D<SizeT, SizeT> keys;
    util::Array1D<SizeT, ValueT> distance;
    util::Array1D<SizeT, SizeT> offsets;

    // Nearest Neighbors
    util::Array1D<SizeT, SizeT> knns;

    // Number of neighbors
    SizeT k;
    // Number of points
    SizeT num_points;
    // Dimension of points labels 
    SizeT dim;

    // CUB Related storage
    util::Array1D<uint64_t, char> cub_temp_storage;

    // Sorted
    util::Array1D<SizeT, SizeT> keys_out;
    util::Array1D<SizeT, ValueT> distance_out;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      points.SetName("points");
      keys.SetName("keys");
      distance.SetName("distance");
      offsets.SetName("offsets");
      knns.SetName("knns");
      cub_temp_storage.SetName("cub_temp_storage");
      keys_out.SetName("keys_out");
      distance_out.SetName("distance_out");
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
      GUARD_CU(distance.Release(target));
      GUARD_CU(offsets.Release(target));
      GUARD_CU(knns.Release(target));
      GUARD_CU(cub_temp_storage.Release(target));
      GUARD_CU(keys_out.Release(target));
      GUARD_CU(distance_out.Release(target));

      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing KNN-specific Data Slice on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param     num_points_ Number of points
     * @param     k_          Number of Nearest Neighbors
     * @param     dim_        Dimension of the points labels
     * @param     num_gpus    Number of GPU devices
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT sub_graph, SizeT num_points_, SizeT k_, SizeT dim_, 
            int num_gpus = 1, int gpu_idx = 0, util::Location target = util::DEVICE, 
            ProblemFlag flag = Problem_None) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      // Basic problem parameters
      num_points = num_points_;
      k = k_;
      dim = dim_;

      //keys need for cub sorting, the same size like distance array
      GUARD_CU(keys.Allocate(k*num_points, target));
      GUARD_CU(keys_out.Allocate(k*num_points, target));
      
      GUARD_CU(distance.Allocate(k * num_points, target));
      GUARD_CU(distance_out.Allocate(k * num_points, target));

      // k-nearest neighbors
      GUARD_CU(knns.Allocate(k * num_points, target));

      // GUARD_CU(cub_temp_storage.Allocate(1, target));

      GUARD_CU(offsets.Allocate(num_points+1, target));

      return retval;
    }

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(ValueT* h_points, util::Location target = util::DEVICE) {

      cudaError_t retval = cudaSuccess;
      SizeT num_points = this->num_points;
      typedef typename GraphT::CsrT CsrT;

      // Ensure data are allocated
      GUARD_CU(keys.EnsureSize_(k * num_points, target));
      GUARD_CU(keys_out.EnsureSize_(k * num_points, target)); 
      // GUARD_CU(cub_temp_storage.EnsureSize_(1, target));
      
      GUARD_CU(distance.EnsureSize_(k * num_points, target));
      GUARD_CU(distance.ForAll(
            [] __host__ __device__(ValueT * d, const SizeT &p) { 
                d[p] = util::PreDefinedValues<ValueT>::InvalidValue;
            },
            k * num_points, target, this->stream));

      // K-Nearest Neighbors
      GUARD_CU(knns.EnsureSize_(k * num_points, target));
      GUARD_CU(knns.ForAll(
          [] __host__ __device__(SizeT * k_, const SizeT &p) { 
            k_[p] = util::PreDefinedValues<SizeT>::InvalidValue;
          },
          k * num_points, target, this->stream));

      GUARD_CU(distance_out.EnsureSize_(k * num_points, target));
      GUARD_CU(distance_out.ForAll(
            [] __host__ __device__(ValueT * d, const SizeT &p) { 
                d[p] = util::PreDefinedValues<ValueT>::InvalidValue;
            },
            k * num_points, target, this->stream));
 
      int k_ = k;
      GUARD_CU(offsets.EnsureSize_(num_points+1, target));
      GUARD_CU(offsets.ForAll(
        [k_] __host__ __device__ (SizeT *ro, const SizeT &pos){
            ro[pos] = pos*k_;
        }, num_points+1, target, this->stream));

      GUARD_CU(util::SetDevice(this->gpu_idx));
      GUARD_CU(points.SetPointer(h_points, num_points*dim, util::HOST));
      GUARD_CU(points.Move(util::HOST, target, num_points*dim, 0, this->stream));

      return retval;
    }
  };  // DataSlice

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;
  SizeT k;
  SizeT num_points;
  SizeT dim;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief KNN Problem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief KNN Problem default destructor
   */
  virtual ~Problem() { Release(); }

  /*
   * @brief Releasing KNN Problem allocated memory space
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
   * @brief Copy result k Nearest Neighbors computed on GPUs back to host-side arrays.
   * @param[in] h_knns  Empty array to store kNN computed on GPU
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(SizeT *h_knns, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    auto &data_slice = data_slices[0][0];

    if (this->num_gpus == 1) {
      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));
        // Extract KNNs
        // knns array
        GUARD_CU(data_slice.knns.SetPointer(h_knns,num_points*k,util::HOST));
        GUARD_CU(data_slice.knns.Move(util::DEVICE, util::HOST));
      }

    } else if (target == util::HOST) {
      GUARD_CU(data_slice.knns.ForEach(
          h_knns,
          [] __host__ __device__(const SizeT &device_val, SizeT &host_val) {
            host_val = device_val;
          },
          num_points * k, util::HOST));
    }
    return retval;
  }

  /**
   * @brief initialization function (Problem struct)
   * @param     graph       The graph that KNN processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    // Assign the parameters to problem
    this->k = this->parameters.template Get<SizeT>("k");
    this->num_points = this->parameters.template Get<SizeT>("n");
    this->dim = this->parameters.template Get<SizeT>("dim");

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_points, this->k,
                  this->dim, this->num_gpus, this->gpu_idx[gpu], target, this->flag));
    }

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run (Problem struct).
   * @param[in] points   Array of points
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(ValueT* points, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    // Reset data slices
    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(points,target));
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
