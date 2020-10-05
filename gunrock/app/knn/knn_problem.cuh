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
#include <gunrock/oprtr/1D_oprtr/for_all.cuh>
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

    // Nearest Neighbors
    util::Array1D<SizeT, SizeT> knns;
    
    util::Array1D<SizeT, int> sem;

    // Number of neighbors
    SizeT k;
    // Number of points
    SizeT num_points;
    // Dimension of points labels 
    SizeT dim;

    // Sorted
    util::Array1D<SizeT, ValueT> distance_out;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      points.SetName("points");
      knns.SetName("knns");
      sem.SetName("sem");
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

      GUARD_CU(knns.Release(target));
      GUARD_CU(distance_out.Release(target));
      GUARD_CU(sem.Release(target));

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

      // k-nearest neighbors
      GUARD_CU(knns.Allocate(k * num_points, target));
      GUARD_CU(distance_out.Allocate(k * num_points, target));
      GUARD_CU(sem.Allocate(num_points, target));

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
                d[p] = util::PreDefinedValues<ValueT>::MaxValue;
            },
            k * num_points, target, this->stream));
 
      GUARD_CU(sem.EnsureSize_(num_points, target));
      GUARD_CU(sem.ForAll(
            [] __host__ __device__(int* d, const SizeT &p) { 
                d[p] = 0;
            },
            num_points, target, this->stream));

      //int k_ = k;

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
  bool transpose;
  bool use_shared_mem;
  
  int num_threads;
  int block_size;
  int grid_size;

  int data_size;
  int points_size;
  int dist_size;
  int keys_size;
  int shared_point_size;
  int shared_mem_size;

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
    this->transpose = this->parameters.template Get <bool>("transpose");

    int use_shared_mem = this->parameters.template Get <bool>("use-shared-mem");
    int num_threads = this->parameters.template Get <int>("NUM-THREADS");

    if (num_threads == 0) num_threads = 128;
    int block_size = (num_points < num_threads ? num_points : num_threads);

    int data_size = sizeof(ValueT);
    printf("data_size = %d\n", data_size);
    int points_size =  ((((block_size + 1) * dim * data_size) + 127)/128) * 128;
    int dist_size =    ((((block_size + 1) * k * data_size) + 127)/128) * 128;
    int keys_size =    ((((block_size + 1) * k * sizeof(int)) + 127)/128) * 128;
    int shared_point_size = dim * data_size;
    int shared_mem_size = points_size + dist_size + keys_size + shared_point_size;

    if (use_shared_mem){
        auto dev = this->parameters.template Get <int>("device");
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        while (shared_mem_size > deviceProp.sharedMemPerBlock){
            block_size -= 32;
            points_size =  (((block_size + 1) * dim * data_size + 127)/128) * 128;
            dist_size =    (((block_size + 1) * k * data_size + 127)/128) * 128;
            keys_size =    (((block_size + 1) * k * sizeof(int) + 127)/128) * 128;
            shared_point_size = dim * data_size;
            shared_mem_size = points_size + dist_size + keys_size + shared_point_size;
        }
        if (block_size == 0){
            use_shared_mem = false;
            block_size = 128;
        }
    }
    int grid_size = 65536/block_size;

    this->use_shared_mem = use_shared_mem;
    this->num_threads = num_threads;
    this->block_size = block_size;
    this->grid_size = grid_size;
    this->data_size = data_size;
    this->points_size = points_size;
    this->dist_size = dist_size;
    this->keys_size = keys_size;
    this->shared_point_size = shared_point_size;
    this->shared_mem_size = shared_mem_size;

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
