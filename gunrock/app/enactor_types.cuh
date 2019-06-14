// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_types.cuh
 *
 * @brief type defines for enactor base
 */

#pragma once

#include <time.h>
#include <moderngpu.cuh>
#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/parameters.h>
#include <gunrock/app/frontier.cuh>
#include <gunrock/oprtr/oprtr_parameters.cuh>

// using namespace mgpu;

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

#define ENABLE_PERFORMANCE_PROFILING

namespace gunrock {
namespace app {

/*
 * @brief Accumulate number function.
 *
 * @tparam SizeT1
 * @tparam SizeT2
 *
 * @param[in] num
 * @param[in] sum
 */
template <typename SizeT1, typename SizeT2>
__global__ void Accumulate_Num(SizeT1 *num, SizeT2 *sum) {
  sum[0] += num[0];
}

/**
 * @brief Structure for auxiliary variables used in enactor.
 */
template <typename SizeT>
struct EnactorStats {
  long long iteration;
  unsigned long long total_lifetimes;
  unsigned long long total_runtimes;
  util::Array1D<int, SizeT> edges_queued;
  util::Array1D<int, SizeT> nodes_queued;
  // unsigned int                     advance_grid_size   ;
  // unsigned int                     filter_grid_size    ;
  util::KernelRuntimeStatsLifetime advance_kernel_stats;
  util::KernelRuntimeStatsLifetime filter_kernel_stats;
  // util::Array1D<int, SizeT>        node_locks          ;
  // util::Array1D<int, SizeT>        node_locks_out      ;
  cudaError_t retval;
  clock_t start_time;

#ifdef ENABLE_PERFORMANCE_PROFILING
  std::vector<std::vector<SizeT>> iter_edges_queued;
  std::vector<std::vector<SizeT>> iter_nodes_queued;
  std::vector<std::vector<SizeT>> iter_in_length;
  std::vector<std::vector<SizeT>> iter_out_length;
#endif

  /*
   * @brief Default EnactorStats constructor
   */
  EnactorStats()
      : iteration(0),
        total_lifetimes(0),
        total_runtimes(0),
        retval(cudaSuccess) {
    // node_locks    .SetName("node_locks"    );
    // node_locks_out.SetName("node_locks_out");
    edges_queued.SetName("edges_queued");
    nodes_queued.SetName("nodes_queued");
  }

  /*
   * @brief Accumulate edge function.
   *
   * @tparam SizeT2
   *
   * @param[in] d_queue Pointer to the queue
   * @param[in] stream CUDA stream
   */
  template <typename SizeT2>
  void AccumulateEdges(SizeT2 *d_queued, cudaStream_t stream) {
    Accumulate_Num<<<1, 1, 0, stream>>>(d_queued,
                                        edges_queued.GetPointer(util::DEVICE));
  }

  /*
   * @brief Accumulate node function.
   *
   * @tparam SizeT2
   *
   * @param[in] d_queue Pointer to the queue
   * @param[in] stream CUDA stream
   */
  template <typename SizeT2>
  void AccumulateNodes(SizeT2 *d_queued, cudaStream_t stream) {
    Accumulate_Num<<<1, 1, 0, stream>>>(d_queued,
                                        nodes_queued.GetPointer(util::DEVICE));
  }

  cudaError_t Init(
      // int node_lock_size = 1024,
      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    if (target & util::DEVICE) {
      // TODO: move to somewhere
      // GUARD_CU(advance_kernel_stats.Setup(advance_grid_size));
      // GUARD_CU(filter_kernel_stats .Setup(filter_grid_size ));
    }
    // GUARD_CU(node_locks    .Allocate(node_lock_size + 1, target));
    // GUARD_CU(node_locks_out.Allocate(node_lock_size + 1, target));
    GUARD_CU(nodes_queued.Allocate(1, target | util::HOST));
    GUARD_CU(edges_queued.Allocate(1, target | util::HOST));

#ifdef ENABLE_PERFORMANCE_PROFILING
    iter_edges_queued.clear();
    iter_nodes_queued.clear();
    iter_in_length.clear();
    iter_out_length.clear();
#endif
    return retval;
  }

  cudaError_t Reset(util::Location target = util::LOCATION_ALL) {
    iteration = 0;
    total_lifetimes = 0;
    total_runtimes = 0;
    retval = cudaSuccess;

    nodes_queued[0] = 0;
    edges_queued[0] = 0;
    GUARD_CU(nodes_queued.Move(util::HOST, target));
    GUARD_CU(edges_queued.Move(util::HOST, target));

#ifdef ENABLE_PERFORMANCE_PROFILING
    iter_edges_queued.push_back(std::vector<SizeT>());
    iter_nodes_queued.push_back(std::vector<SizeT>());
    iter_in_length.push_back(std::vector<SizeT>());
    iter_out_length.push_back(std::vector<SizeT>());
#endif
    return retval;
  }

  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    // GUARD_CU(node_locks    .Release(target));
    // GUARD_CU(node_locks_out.Release(target));
    GUARD_CU(edges_queued.Release(target));
    GUARD_CU(nodes_queued.Release(target));

#ifdef ENABLE_PERFORMANCE_PROFILING
    for (auto it = iter_edges_queued.begin(); it != iter_edges_queued.end();
         it++)
      it->clear();
    for (auto it = iter_nodes_queued.begin(); it != iter_nodes_queued.end();
         it++)
      it->clear();
    for (auto it = iter_in_length.begin(); it != iter_in_length.end(); it++)
      it->clear();
    for (auto it = iter_out_length.begin(); it != iter_out_length.end(); it++)
      it->clear();
    iter_edges_queued.clear();
    iter_nodes_queued.clear();
    iter_in_length.clear();
    iter_out_length.clear();
#endif
    return retval;
  }
};

/*
 * @brief Thread slice data structure
 */
class ThreadSlice {
 public:
  enum Status { New, Inited, Start, Wait, Running, Idle, ToKill, Ended };

  int thread_num;
  int init_size;
  CUTThread thread_Id;
  Status status;
  void *problem;
  void *enactor;

  /*
   * @brief Default ThreadSlice constructor
   */
  ThreadSlice()
      : problem(NULL),
        enactor(NULL),
        thread_num(0),
        init_size(0),
        status(Status::New) {}

  /*
   * @brief Default ThreadSlice destructor
   */
  virtual ~ThreadSlice() {
    problem = NULL;
    enactor = NULL;
  }

  cudaError_t Reset() {
    cudaError_t retval = cudaSuccess;
    init_size = 0;
    return retval;
  }
};

template <typename GraphT, typename LabelT,
          util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class EnactorSlice {
 public:
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef Frontier<VertexT, SizeT, ARRAY_FLAG, cudaHostRegisterFlag> FrontierT;
  typedef oprtr::OprtrParameters<GraphT, FrontierT, LabelT> OprtrParametersT;

  cudaStream_t stream, stream2;
  mgpu::ContextPtr context;
  EnactorStats<SizeT> enactor_stats;
  FrontierT frontier;
  OprtrParametersT oprtr_parameters;

  EnactorSlice() {
    stream = 0;
    stream2 = 0;
    // context = NULL;
  }

  ~EnactorSlice() { Release(); }

  cudaError_t Init(unsigned int num_queues = 2, FrontierType *types = NULL,
                   std::string frontier_name = "",
                   // int node_lock_size = 1024,
                   util::Location target = util::DEVICE,
                   util::CudaProperties *cuda_properties = NULL,
                   std::string advance_mode = "", std::string filter_mode = "",
                   int max_grid_size = 0) {
    cudaError_t retval = cudaSuccess;

    // util::PrintMsg("target = " + std::to_string(target));
    if (target & util::DEVICE) {
      GUARD_CU2(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags failed");
      GUARD_CU2(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags failed");

      int gpu_idx;
      GUARD_CU2(cudaGetDevice(&gpu_idx), "cudaGetDevice failed.");
      context = mgpu::CreateCudaDeviceAttachStream(gpu_idx, stream);
      // util::PrintMsg("Stream and context allocated on GPU " +
      // std::to_string(gpu_idx));
    }

    GUARD_CU(enactor_stats.Init(target));
    GUARD_CU(frontier.Init(num_queues, types, frontier_name, target));
    GUARD_CU(oprtr_parameters.Init());

    oprtr_parameters.stream = stream;
    oprtr_parameters.context = context;
    oprtr_parameters.frontier = &frontier;
    oprtr_parameters.cuda_props = cuda_properties;
    oprtr_parameters.advance_mode = advance_mode;
    oprtr_parameters.filter_mode = filter_mode;
    oprtr_parameters.max_grid_size = max_grid_size;
    return retval;
  }

  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;

    GUARD_CU(frontier.Release(target));
    GUARD_CU(enactor_stats.Release(target));

    if (target & util::DEVICE) {
      if (stream != 0) {
        GUARD_CU2(cudaStreamDestroy(stream), "cudaStreamDestroy failed");
        stream = 0;
      }
      if (stream2 != 0) {
        GUARD_CU2(cudaStreamDestroy(stream2), "cudaStreamDestroy failed");
        stream2 = 0;
      }
    }
    return retval;
  }

  cudaError_t Reset(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(enactor_stats.Reset(target));
    GUARD_CU(frontier.Reset(target));
    return retval;
  }
};

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
