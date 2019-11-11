// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kcore_problem.cuh
 *
 * @brief GPU Storage management Structure for K-Core Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace kcore {

/**
 * @brief Speciflying parameters for K-Core Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

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
    util::Array1D<SizeT, VertexT> num_cores;
    util::Array1D<SizeT, VertexT> out_degrees;

    SizeT num_remaining_vertices;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice(),
      num_remaining_vertices(0) {
      num_cores.SetName("num_cores");
      out_degrees.SetName("out_degrees");
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
      
      GUARD_CU(num_cores.Release(target));
      GUARD_CU(out_degrees.Release(target));

      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing K-Core-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus = 1, int gpu_idx = 0,
                     util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      cudaError_t retval = cudaSuccess;
      SizeT nodes = sub_graph.nodes;
      SizeT edges = sub_graph.edges;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      GUARD_CU(num_cores.Allocate(nodes+1, target));
      GUARD_CU(out_degrees.Allocate(nodes, target));


      if (target & util::DEVICE) {
        GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
      }

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");

      GUARD_CU(num_cores.ForEach([] __host__ __device__(SizeT &x) { x = 0; },
                                 nodes+1, target, this->stream));

      if (GraphT::FLAG & gunrock::graph::HAS_CSR) {
        GUARD_CU(out_degrees.ForAll(
            [sub_graph] __host__ __device__(SizeT * out_degrees, const SizeT &pos) {
              out_degrees[pos] = sub_graph.GetNeighborListLength(pos);
            },
            sub_graph.nodes, target, this->stream));
      } else if (GraphT::FLAG &
                 (gunrock::graph::HAS_COO | gunrock::graph::HAS_CSC)) {
        GUARD_CU(out_degrees.ForEach(
            [] __host__ __device__(SizeT & out_degrees) { out_degrees = 0; }, nodes + 1,
            target, this->stream));

        GUARD_CU(degrees.ForAll(
            [sub_graph, nodes] __host__ __device__(SizeT * out_degrees,
                                                         const SizeT &e) {
              VertexT src, dest;
                sub_graph.CooT::GetEdgeSrcDest(e, src, dest);
                atomicAdd(out_degrees + src, 1);
            },
            sub_graph.edges, target, this->stream));
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
      SizeT edges = this->sub_graph->edges;
      
      GUARD_CU(num_cores.EnsureSize_(nodes+1, target));
      GUARD_CU(out_degrees.EnsureSize_(nodes, target));

      // Reset data
      GUARD_CU(num_cores.ForEach([] __host__ __device__(SizeT &x) { x = 0; },
                                 nodes+1, target, this->stream));

      if (GraphT::FLAG & gunrock::graph::HAS_CSR) {
        GUARD_CU(out_degrees.ForAll(
            [sub_graph] __host__ __device__(SizeT * out_degrees, const SizeT &pos) {
              out_degrees[pos] = sub_graph.GetNeighborListLength(pos);
            },
            sub_graph.nodes, target, this->stream));
      } else if (GraphT::FLAG &
                 (gunrock::graph::HAS_COO | gunrock::graph::HAS_CSC)) {
        GUARD_CU(out_degrees.ForEach(
            [] __host__ __device__(SizeT & out_degrees) { out_degrees = 0; }, nodes + 1,
            target, this->stream));

        GUARD_CU(degrees.ForAll(
            [sub_graph, nodes] __host__ __device__(SizeT * out_degrees,
                                                         const SizeT &e) {
              VertexT src, dest;
                sub_graph.CooT::GetEdgeSrcDest(e, src, dest);
                atomicAdd(out_degrees + src, 1);
            },
            sub_graph.edges, target, this->stream));
      }

      this->num_remaining_vertices = nodes;
      return retval;
    }
  };  // DataSlice

  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // ----------------------------------------------------------------
  // Problem Methods

  /**
   * @brief K-Core default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {
  }

  /**
   * @brief K-Core default destructor
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
   * @brief Copy result num_cores computed on GPUs back to host-side arrays.
...
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(VertexT *h_num_cores, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        GUARD_CU(data_slice.num_cores.SetPointer(h_num_cores, nodes+1, util::HOST));
        GUARD_CU(data_slice.num_cores.Move(util::DEVICE, util::HOST));
      } else if (target == util::HOST) {
        GUARD_CU(data_slice.num_cores.ForEach(
            h_num_cores,
            [] __host__ __device__(const VertexT &device_val,
                                   VertexT &host_val) {
              host_val = device_val;
            },
            nodes+1, util::HOST));
      }
    } else {  // num_gpus != 1
    }

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph to processes on
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

}  // namespace kcore
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: