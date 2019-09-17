// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hits_problem.cuh
 *
 * @brief GPU Storage management Structure for hits Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace hits {

/**
 * @brief Speciflying parameters for hits Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  GUARD_CU(parameters.Use<int64_t>(
      "max-iter",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      100, "Number of HITS iterations.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int64_t>(
      "normalize-n",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 1,
      "Normalize HITS scores every N iterations.", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<double>(
      "tol",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      1e-6, "Floating-point tolerance for CPU/GPU rank comparison.", __FILE__,
      __LINE__));

  return retval;
}

/**
 * @brief Template Problem structure.
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
    // HITS problem-specific storage arrays
    util::Array1D<SizeT, ValueT> hrank_curr;  // Holds hub rank value
    util::Array1D<SizeT, ValueT> arank_curr;  // Holds authority rank value
    util::Array1D<SizeT, ValueT> hrank_next;
    util::Array1D<SizeT, ValueT> arank_next;
    util::Array1D<uint64_t, char>
        cub_temp_space;  // Temporary space for normalization addition
    util::Array1D<SizeT, ValueT> hrank_mag;
    util::Array1D<SizeT, ValueT> arank_mag;
    SizeT max_iter;     // Maximum number of HITS iterations to perform
    SizeT normalize_n;  // Normalize every N iterations
    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice(), max_iter(0), normalize_n(0) {
      // Name of the problem specific arrays:
      hrank_curr.SetName("hrank_curr");
      arank_curr.SetName("arank_curr");
      hrank_next.SetName("hrank_next");
      arank_next.SetName("arank_next");
      cub_temp_space.SetName("cub_temp_space");
      hrank_mag.SetName("hrank_mag");
      arank_mag.SetName("arank_mag");
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

      // Release allocated data
      GUARD_CU(hrank_curr.Release(target));
      GUARD_CU(arank_curr.Release(target));
      GUARD_CU(hrank_next.Release(target));
      GUARD_CU(arank_next.Release(target));
      GUARD_CU(cub_temp_space.Release(target));
      GUARD_CU(hrank_mag.Release(target));
      GUARD_CU(arank_mag.Release(target));
      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing hits-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus, int gpu_idx,
                     util::Location target, ProblemFlag flag) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));
      // Allocate problem specific data here

      GUARD_CU(hrank_curr.Allocate(sub_graph.nodes, target));
      GUARD_CU(arank_curr.Allocate(sub_graph.nodes, target));
      GUARD_CU(hrank_next.Allocate(sub_graph.nodes, target));
      GUARD_CU(arank_next.Allocate(sub_graph.nodes, target));
      GUARD_CU(cub_temp_space.Allocate(1, target));

      GUARD_CU(hrank_mag.Allocate(1, target | util::HOST));
      GUARD_CU(arank_mag.Allocate(1, target | util::HOST));

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
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = this->sub_graph->nodes;

      // Ensure data are allocated

      GUARD_CU(hrank_curr.EnsureSize_(nodes, target));
      GUARD_CU(arank_curr.EnsureSize_(nodes, target));
      GUARD_CU(hrank_next.EnsureSize_(nodes, target));
      GUARD_CU(arank_next.EnsureSize_(nodes, target));

      GUARD_CU(cub_temp_space.EnsureSize_(1, target));
      GUARD_CU(hrank_mag.EnsureSize_(1, target));
      GUARD_CU(arank_mag.EnsureSize_(1, target));

      // Reset data

      // Initialize current hrank and arank to 1.
      // Initialize next ranks to 0 (will be updated).
      GUARD_CU(hrank_curr.ForEach(
          [] __host__ __device__(ValueT & x) { x = (ValueT)1.0; }, nodes,
          target, this->stream));

      GUARD_CU(arank_curr.ForEach(
          [] __host__ __device__(ValueT & x) { x = (ValueT)1.0; }, nodes,
          target, this->stream));

      GUARD_CU(hrank_next.ForEach(
          [] __host__ __device__(ValueT & x) { x = (ValueT)0.0; }, nodes,
          target, this->stream));

      GUARD_CU(arank_next.ForEach(
          [] __host__ __device__(ValueT & x) { x = (ValueT)0.0; }, nodes,
          target, this->stream));

      GUARD_CU(hrank_mag.ForEach(
          [] __host__ __device__(ValueT & x) { x = (ValueT)0.0; }, 1, target,
          this->stream));

      GUARD_CU(arank_mag.ForEach(
          [] __host__ __device__(ValueT & x) { x = (ValueT)0.0; }, 1, target,
          this->stream));

      return retval;
    }
  };  // DataSlice

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief hits default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief hits default destructor
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
   * @param[in] h_hrank_curr The host memory to extract hub scores to
   * @param[in] h_arank_curr The host memory to extract auth scores to
   * @param[in] target       The location to copy memory from
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(ValueT *h_hrank_curr, ValueT *h_arank_curr,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        // Extract the results from a single GPU
        GUARD_CU(
            data_slice.hrank_curr.SetPointer(h_hrank_curr, nodes, util::HOST));
        GUARD_CU(data_slice.hrank_curr.Move(util::DEVICE, util::HOST));

        GUARD_CU(
            data_slice.arank_curr.SetPointer(h_arank_curr, nodes, util::HOST));
        GUARD_CU(data_slice.arank_curr.Move(util::DEVICE, util::HOST));
      } else if (target == util::HOST) {
        // Extract the results from single CPU, e.g.:
        GUARD_CU(data_slice.hrank_curr.ForEach(
            h_hrank_curr,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));

        GUARD_CU(data_slice.arank_curr.ForEach(
            h_arank_curr,
            [] __host__ __device__(const ValueT &device_val, ValueT &host_val) {
              host_val = device_val;
            },
            nodes, util::HOST));
      }
    } else {  // Incomplete multi-gpu
    }

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that SSSP processes on
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

      data_slice.max_iter = this->parameters.template Get<SizeT>("max-iter");

      data_slice.normalize_n =
          this->parameters.template Get<SizeT>("normalize-n");

      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag));
    }

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    // Reset data slices
    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    return retval;
  }
};

}  // namespace hits
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
