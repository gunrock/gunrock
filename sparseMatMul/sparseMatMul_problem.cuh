// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * graphsum_problem.cuh
 *
 * @brief GPU Storage management Structure for SSSP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace sparseMatMul {
/**
 * @brief Speciflying parameters for graphsum Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  return retval;
}

/**
 * @brief Single-Source Shortest Path Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in sssp
 * @tparam _ValueT  Type of per-vertex distance values
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, typename _LabelT = typename _GraphT::VertexT,
          typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;
  typedef _LabelT LabelT;
  typedef _ValueT ValueT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;
  typedef util::Array1D<SizeT, ValueT> Array;

  // Helper structures

  /**
   * @brief Data structure containing graphsum-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    util::Array1D<SizeT, ValueT> input, output;
    util::Array1D<SizeT, VertexT> local_vertices;
    int in_dim, out_dim;
    bool forward;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      input.SetName("input");
      output.SetName("output");
      local_vertices.SetName("local_vertices");
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

      GUARD_CU(input.Release(target));
      GUARD_CU(output.Release(target));
      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing graphsum-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] num_gpus    Number of GPUs
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, const int in_dim, const int out_dim,
                     int num_gpus = 1, int gpu_idx = 0,
                     util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      this->in_dim = in_dim;
      this->out_dim = out_dim;
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));
      GUARD_CU(input.Allocate(in_dim * out_dim, util::HOST));
      GUARD_CU(output.Allocate(sub_graph.nodes * out_dim, target));
      GUARD_CU(local_vertices.Allocate(sub_graph.nodes, target));

      GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));
      return retval;
    }  // Init

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;

//      std::cout << "nodes: " << this->sub_graph->nodes << ", dim: " << out_dim << '\n';
      // Ensure data are allocated
//      GUARD_CU(output.EnsureSize_(
//          this->sub_graph->nodes * this->out_dim, target));

      // Initizlize local vertices
      GUARD_CU(local_vertices.ForAll(
          [] __host__ __device__(VertexT * l_vertices, const SizeT &pos) {
        l_vertices[pos] = pos;
      }, this->sub_graph->nodes, target));

      // Initialize output matrix to be all 0
      GUARD_CU(output.ForEach(
          [] __host__ __device__(ValueT &x) {
            x = 0;
          }, output.GetSize (), target, this->stream));

      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief graphsum default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief graphsum default destructor
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


  cudaError_t Extract(ValueT *out,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));
//        data_slice.output.Print();
        GUARD_CU(
            data_slice.output.SetPointer(out,
                data_slice.in_dim * data_slice.out_dim, util::HOST));
        GUARD_CU(data_slice.output.Move(util::DEVICE, util::HOST));
      }
    }

    return retval;
  }

  /**
   * @brief      initialization function.
   *
   * @param      graph   The graph that SSSP processes on
   * @param[in]  dim     The dimension of the feature vector
   * @param[in]  target  The target
   * @param[in]  Location  Memory location to work on
   *
   * @return     cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, const int in_dim, const int outdim, const ValueT *in,
      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], in_dim, outdim, this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag));

      // Initialize input matrix
      auto nodes = this->sub_graphs[gpu].nodes;
      GUARD_CU(data_slice.input.EnsureSize_(in_dim * outdim));
//      util::PrintMsg("dataslice input size: " + std::to_string(nodes * dim));
      GUARD_CU(data_slice.input.ForAll(
               [in] __host__ __device__(ValueT *in_, const SizeT &pos) {
        in_[pos] = in[pos];
      }, in_dim * outdim, util::HOST
               ));
      data_slice.input.Move(util::HOST, util::DEVICE);
//      data_slice.input.Print();
    }  // end for (gpu)

    return retval;
  }

  cudaError_t Init(GraphT &graph, const int in_dim, const int outdim,
                   util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
        data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
        if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

        GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

        auto &data_slice = data_slices[gpu][0];
        GUARD_CU(data_slice.Init(this->sub_graphs[gpu], in_dim, outdim, this->num_gpus,
                                 this->gpu_idx[gpu], target, this->flag));
        GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }  // end for (gpu)

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(const ValueT *in, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    if (target & util::DEVICE) {
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    return retval;
  }

  cudaError_t Reset(bool forward, Array &b, Array &c, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      data_slices[gpu][0].input = b;
      data_slices[gpu][0].output = c;
      data_slices[gpu][0].forward = forward;
      GUARD_CU(data_slices[gpu]->Reset(target));
//      std::cout << "d_pointer: " << data_slices[gpu].GetPointer(util::DEVICE) << '\n';
    }

    if (target & util::DEVICE) {
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    return retval;
  }

  /** @} */
};

}  // namespace graphsum
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
