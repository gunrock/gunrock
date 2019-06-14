// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * problem_base.cuh
 *
 * @brief Base structure for all the application types
 */

#pragma once

#include <vector>
#include <string>

// Graph partitioner utilities
#include <gunrock/partitioner/partitioner.cuh>

// this is the "stringize macro macro" hack
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

using ProblemFlag = uint32_t;

enum : ProblemFlag {
  Problem_None = 0x00,
  Mark_Predecessors = 0x01,
  Enable_Idempotence = 0x02,
};

/**
 * @brief Speciflying parameters for ProblemBase
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(partitioner::UseParameters(parameters));
  if (!parameters.Have("device"))
    GUARD_CU(parameters.Use<int>(
        "device",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        0, "Set GPU(s) for testing", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Data structure containing non-problem-specific data on indivual GPU.
 */
template <typename _GraphT, ProblemFlag _FLAG = Problem_None>
struct DataSliceBase {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;

  int num_gpus;         // number of GPUs
  int gpu_idx;          // index of GPU the data slice is allocated
  cudaStream_t stream;  // cuda stream that data movement works on
  GraphT *sub_graph;    // pointer to the sub graph
  ProblemFlag flag;     // problem flag

  /*
   * @brief Default constructor
   */
  DataSliceBase()
      : num_gpus(1),
        gpu_idx(0),
        stream(0),
        sub_graph(NULL),
        flag(Problem_None) {}

  /**
   * @brief initializing sssp-specific data on each gpu
   * @param     sub_graph sub graph on the GPU.
   * @param[in] gpu_idx GPU device index
   * @param[in] target targeting device location
   * @param[in] flag problem flag containling options
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(GraphT &sub_graph, int num_gpus, int gpu_idx,
                   util::Location target, ProblemFlag flag) {
    cudaError_t retval = cudaSuccess;

    this->num_gpus = num_gpus;
    this->gpu_idx = gpu_idx;
    this->sub_graph = &sub_graph;
    this->flag = flag;
    if (target & util::DEVICE) {
      GUARD_CU(util::SetDevice(gpu_idx));
      GUARD_CU2(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags failed.");
    }

    return retval;
  }

  /*
   * @brief Releasing allocated memory space
   * @param target the location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;

    if (target & util::DEVICE) {
      if (stream != 0)
        GUARD_CU2(cudaStreamDestroy(stream), "cudaStreamDestroy failed");
    }
    return retval;
  }
};  // DataSliceBase

/**
 * @brief Base problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, ProblemFlag _FLAG = Problem_None>
struct ProblemBase {
  typedef _GraphT GraphT;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  static const ProblemFlag FLAG = _FLAG;
  ProblemFlag flag;

  // Members
  int num_gpus;              // Number of GPUs to be sliced over
  std::vector<int> gpu_idx;  // GPU indices
  GraphT *org_graph;         // pointer to the input graph
  size_t *org_mem_size;      // original memory size on GPUs
  util::Array1D<int, GraphT>
      sub_graphs;                // Subgraphs for multi-GPU implementation
  util::Parameters &parameters;  // Running parameters

  // Methods

  /**
   * @brief ProblemBase default constructor
   */
  ProblemBase(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : parameters(_parameters), flag(_flag), num_gpus(1) {
    sub_graphs.SetName("sub_graphs");

    gpu_idx = parameters.Get<std::vector<int>>("device");
    num_gpus = gpu_idx.size();
    org_mem_size = new size_t[num_gpus];
    size_t *dummy_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      util::GRError(util::SetDevice(gpu_idx[gpu]));
      util::GRError(cudaMemGetInfo(org_mem_size + gpu, dummy_size + gpu),
                    "cudaMemGetInfo failed", __FILE__, __LINE__);
    }
    delete[] dummy_size;
    dummy_size = NULL;
  }  // end ProblemBase()

  /**
   * @brief ProblemBase default destructor to free all graph slices allocated.
   */
  virtual ~ProblemBase() { Release(); }

  /*
   * @brief Releasing allocated memory space
   * @param target the location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    // Cleanup graph slices on the heap
    if (sub_graphs.GetPointer(util::HOST) != NULL && num_gpus != 1) {
      for (int i = 0; i < num_gpus; ++i) {
        if (target & util::DEVICE) GUARD_CU(util::SetDevice(gpu_idx[i]));
        GUARD_CU(sub_graphs[i].Release(target));
      }
      GUARD_CU(sub_graphs.Release(target));
    }

    if (target & util::HOST) {
      delete[] org_mem_size;
      org_mem_size = NULL;
    }
    return retval;
  }  // end Release()

  /**
   * @brief initialization function.
   * @param[in] parameters data structure holding all parameter info
   * @param     graph the graph that SSSP processes on
   * @param[in] location memory location to work on
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(
      // util::Parameters &parameters,
      GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->org_graph = &graph;

    if (num_gpus == 1)
      sub_graphs.SetPointer(&graph, 1, util::HOST);
    else {
      retval = sub_graphs.Allocate(num_gpus, target | util::HOST);
      if (retval) return retval;
      GraphT *t_subgraphs = sub_graphs + 0;
      retval = gunrock::partitioner::Partition(graph, t_subgraphs, parameters,
                                               num_gpus, flag, target);
      if (retval) return retval;
    }

    return retval;
  }  // end Init(...)

};  // ProblemBase

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
