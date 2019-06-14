// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_base.cuh
 *
 * @brief Base Graph Problem Enactor
 */

#pragma once

//#include <moderngpu.cuh>
#include <chrono>
#include <thread>

#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/app/enactor_types.cuh>
#include <gunrock/app/mgpu_slice.cuh>

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

using Enactor_Flag = uint32_t;

enum : Enactor_Flag {
  Enactor_None = 0x00,
  Instrument = 0x01,
  Debug = 0x02,
  Size_Check = 0x04,
};

/**
 * @brief Speciflying parameters for EnactorBase
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  if (!parameters.Have("device"))
    GUARD_CU(parameters.Use<int>(
        "device",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        0, "Set GPU(s) for testing", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "communicate-latency",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "additional communication latency", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<float>(
      "communicate-multipy",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1.0f, "communication sizing factor", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "expand-latency",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "additional expand incoming latency", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "subqueue-latency",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "additional subqueue latency", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "fullqueue-latency",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "additional fullqueue latency", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "makeout-latency",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "additional make-out latency", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "advance-mode",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "LB",
      "Advance strategy, <LB | LB_CULL | LB_LIGHT | LB_LIGHT_CULL | TWC>,\n"
      "\tdefault is determined based on input graph",
      __FILE__, __LINE__,
      "LB for Load-Balanced,\n"
      "\tTWC for Dynamic-Cooperative,\n"
      "\tadd -LIGHT for small frontiers,\n"
      "\tadd -CULL for fuzed kernels;\n"
      "\tnot all modes are available for specific problem;\n"));

  GUARD_CU(parameters.Use<std::string>(
      "filter-mode",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "CULL", "Filter strategy", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<double>(
      "queue-factor",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      6.0,
      "Reserved frontier sizing factor, multiples of numbers of vertices or "
      "edges",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<double>(
      "trans-factor",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      1.0,
      "Reserved sizing factor for data communication, multiples of number of "
      "vertices",
      __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "size-check",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      true, "Whether to enable frontier auto resizing", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "max-grid-size",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "Maximun number of grids for GPU kernels", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Base class for enactor.
 * @tparam GraphT  Type of the graph
 * @tparam LabelT  Type of labels used in the operators
 * @tparam ValueT  Type of values used in the mgpu slices
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename GraphT, typename LabelT,
          typename _ValueT = typename GraphT::ValueT,
          util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class EnactorBase {
 public:
  typedef typename GraphT::VertexT VertexT;
  // typedef typename GraphT::ValueT  ValueT;
  typedef typename GraphT::SizeT SizeT;
  typedef _ValueT ValueT;
  typedef EnactorSlice<GraphT, LabelT, ARRAY_FLAG, cudaHostRegisterFlag>
      EnactorSliceT;
  typedef MgpuSlice<VertexT, SizeT, ValueT> MgpuSliceT;

  int num_gpus;
  std::vector<int> gpu_idx;
  std::string algo_name;
  util::Parameters *parameters;
  Enactor_Flag flag;

  int max_num_vertex_associates;
  int max_num_value__associates;
  int communicate_latency;
  float communicate_multipy;
  int expand_latency;
  int subqueue_latency;
  int fullqueue_latency;
  int makeout_latency;
  int min_sm_version;
  std::vector<double> queue_factors;
  double trans_factor;

  // Device properties
  util::Array1D<SizeT, util::CudaProperties, ARRAY_FLAG,
                cudaHostRegisterFlag | cudaHostAllocMapped |
                    cudaHostAllocPortable>
      cuda_props;

  // Per-GPU enactor slices
  util::Array1D<int, EnactorSliceT, ARRAY_FLAG,
                cudaHostRegisterFlag>  // | cudaHostAllocMapped |
                                       // cudaHostAllocPortable>
                                           enactor_slices;

  util::Array1D<int, MgpuSliceT, ARRAY_FLAG,
                cudaHostRegisterFlag>  // | cudaHostAllocMapped |
                                       // cudaHostAllocPortable>
                                           mgpu_slices;

  // Per-CPU-thread data
  util::Array1D<int, ThreadSlice> thread_slices;
  util::Array1D<int, CUTThread> thread_Ids;

#ifdef ENABLE_PERFORMANCE_PROFILING
  util::Array1D<int, std::vector<std::vector<double>>> iter_full_queue_time;
  util::Array1D<int, std::vector<std::vector<double>>> iter_sub_queue_time;
  util::Array1D<int, std::vector<std::vector<double>>> iter_total_time;
  util::Array1D<int, std::vector<std::vector<SizeT>>>
      iter_full_queue_nodes_queued;
  util::Array1D<int, std::vector<std::vector<SizeT>>>
      iter_full_queue_edges_queued;
#endif

 protected:
  /**
   * @brief EnactorBase constructor
   * @param[in] algo_name Name of the algorithm
   */
  EnactorBase(std::string algo_name = "test") {
    this->algo_name = algo_name;
    cuda_props.SetName("cuda_props");
    enactor_slices.SetName("enactor_slices");
    thread_slices.SetName("thread_slices");
    thread_Ids.SetName("thread_Ids");

#ifdef ENABLE_PERFORMANCE_PROFILING
    iter_full_queue_time.SetName("iter_full_queue_time");
    iter_sub_queue_time.SetName("iter_sub_queue_time");
    iter_total_time.SetName("iter_total_time");
    iter_full_queue_edges_queued.SetName("iter_full_queue_edges_queued");
    iter_full_queue_nodes_queued.SetName("iter_full_queue_nodes_queued");
#endif
  }

  /**
   * @brief EnactorBase destructor
   */
  virtual ~EnactorBase() {
    // Release();
  }

  /*
   * @brief Releasing allocated memory space
   * @param target The location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval;
    // util::PrintMsg("EnactorBase::Release() entered");

    if (thread_slices.GetPointer(util::HOST) != NULL) GUARD_CU(Kill_Threads());

    if (enactor_slices.GetPointer(util::HOST) != NULL)
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        GUARD_CU(util::SetDevice(gpu_idx[gpu]));
        for (int peer = 0; peer < num_gpus; peer++) {
          int idx = gpu * num_gpus + peer;
          GUARD_CU(enactor_slices[idx].Release(target));
        }
        GUARD_CU(mgpu_slices[gpu].Release(target));
      }
    GUARD_CU(cuda_props.Release(target));
    GUARD_CU(enactor_slices.Release(target));
    GUARD_CU(mgpu_slices.Release(target));
    GUARD_CU(thread_Ids.Release(target));
    GUARD_CU(thread_slices.Release(target));

#ifdef ENABLE_PERFORMANCE_PROFILING
    if (iter_full_queue_time.GetPointer(util::HOST) != NULL &&
        (target & util::HOST) != 0) {
      for (int gpu = 0; gpu < num_gpus; gpu++) {
        for (auto it = iter_full_queue_time[gpu].begin();
             it != iter_full_queue_time[gpu].end(); it++)
          it->clear();
        for (auto it = iter_sub_queue_time[gpu].begin();
             it != iter_sub_queue_time[gpu].end(); it++)
          it->clear();
        for (auto it = iter_total_time[gpu].begin();
             it != iter_total_time[gpu].end(); it++)
          it->clear();
        for (auto it = iter_full_queue_nodes_queued[gpu].begin();
             it != iter_full_queue_nodes_queued[gpu].end(); it++)
          it->clear();
        for (auto it = iter_full_queue_edges_queued[gpu].begin();
             it != iter_full_queue_edges_queued[gpu].end(); it++)
          it->clear();
        iter_full_queue_time[gpu].clear();
        iter_sub_queue_time[gpu].clear();
        iter_total_time[gpu].clear();
        iter_full_queue_nodes_queued[gpu].clear();
        iter_full_queue_edges_queued[gpu].clear();
      }
      GUARD_CU(iter_full_queue_time.Release(target));
      GUARD_CU(iter_sub_queue_time.Release(target));
      GUARD_CU(iter_total_time.Release(target));
      GUARD_CU(iter_full_queue_nodes_queued.Release(target));
      GUARD_CU(iter_full_queue_edges_queued.Release(target));
    }
#endif
    // util::PrintMsg("EnactorBase::Release() returning");
    return retval;
  }

  /**
   * @brief Init function for enactor base, called within Enactor::Init()
   * @param[in] parameters Running parameters.
   * @param[in] sub_graphs Pointer to the sub graphs on indivual GPUs
   * @param[in] flag Enactor flag
   * @param[in] num_queues The number queues in the frontier
   * @param[in] frontier_type The types of each queue in the frontier, default
   * is VERTEX_FRONTIER
   * @param[in] target Target location of data
   * @param[in] skip_makeout_selection Whether to skip the makeout selection
   * routine during communication \return cudaError_t error message(s), if any
   */
  template <typename ProblemT>
  cudaError_t Init(
      // util::Parameters &parameters,
      // GraphT           *sub_graphs,
      ProblemT &problem, Enactor_Flag flag = Enactor_None,
      unsigned int num_queues = 2, FrontierType *frontier_types = NULL,
      util::Location target = util::DEVICE,
      bool skip_makeout_selection = false) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    util::Parameters &parameters = problem.parameters;
    GraphT *sub_graphs = problem.sub_graphs + 0;

    gpu_idx = parameters.Get<std::vector<int>>("device");
    num_gpus = gpu_idx.size();
    communicate_latency = parameters.Get<int>("communicate-latency");
    communicate_multipy = parameters.Get<float>("communicate-multipy");
    expand_latency = parameters.Get<int>("expand-latency");
    subqueue_latency = parameters.Get<int>("subqueue-latency");
    fullqueue_latency = parameters.Get<int>("fullqueue-latency");
    makeout_latency = parameters.Get<int>("makeout-latency");
    queue_factors = parameters.Get<std::vector<double>>("queue-factor");
    trans_factor = parameters.Get<double>("trans-factor");
    std::string advance_mode = parameters.Get<std::string>("advance-mode");
    std::string filter_mode = parameters.Get<std::string>("filter-mode");
    int max_grid_size = parameters.Get<int>("max-grid-size");
    bool quiet = parameters.Get<bool>("quiet");

    util::PrintMsg("Using advance mode " + advance_mode, !quiet);
    util::PrintMsg("Using filter mode " + filter_mode, !quiet);

    min_sm_version = -1;
    this->parameters = &parameters;
    if (parameters.Get<bool>("v")) flag = flag | Debug;
    if (parameters.Get<bool>("size-check")) flag = flag | Size_Check;
    this->flag = flag;

    GUARD_CU(cuda_props.Allocate(num_gpus, util::HOST));
    GUARD_CU(enactor_slices.Allocate(num_gpus * num_gpus, util::HOST));
    GUARD_CU(mgpu_slices.Allocate(num_gpus, util::HOST));
    GUARD_CU(thread_slices.Allocate(num_gpus, util::HOST));
    GUARD_CU(thread_Ids.Allocate(num_gpus, util::HOST));

#ifdef ENABLE_PERFORMANCE_PROFILING
    GUARD_CU(iter_full_queue_time.Allocate(num_gpus, util::HOST));
    GUARD_CU(iter_sub_queue_time.Allocate(num_gpus, util::HOST));
    GUARD_CU(iter_total_time.Allocate(num_gpus, util::HOST));
    GUARD_CU(iter_full_queue_nodes_queued.Allocate(num_gpus, util::HOST));
    GUARD_CU(iter_full_queue_edges_queued.Allocate(num_gpus, util::HOST));
#endif

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (target & util::DEVICE) {
        GUARD_CU(util::SetDevice(gpu_idx[gpu]));

        // Setup work progress (only needs doing once since we maintain
        // it in our kernel code)
        cuda_props[gpu].Setup(gpu_idx[gpu]);
        if (min_sm_version == -1 ||
            cuda_props[gpu].device_sm_version < min_sm_version)
          min_sm_version = cuda_props[gpu].device_sm_version;
      }

      for (int peer = 0; peer < num_gpus; peer++) {
        auto &enactor_slice = enactor_slices[gpu * num_gpus + peer];

        GUARD_CU(
            enactor_slice.Init(num_queues, frontier_types,
                               algo_name + "::frontier[" + std::to_string(gpu) +
                                   "," + std::to_string(peer) + "]",
                               /*node_lock_size,*/ target, cuda_props + gpu,
                               advance_mode, filter_mode, max_grid_size));

        if (gpu != peer && (target & util::DEVICE) != 0) {
          int peer_access_avail;
          GUARD_CU2(cudaDeviceCanAccessPeer(&peer_access_avail, gpu_idx[gpu],
                                            gpu_idx[peer]),
                    "cudaDeviceCanAccess failed");
          if (peer_access_avail) {
            GUARD_CU2(cudaDeviceEnablePeerAccess(gpu_idx[peer], 0),
                      "cudaDeviceEnablePeerAccess failed");
          }
        }
      }

      auto &mgpu_slice = mgpu_slices[gpu];
      auto &sub_graph = sub_graphs[gpu];
      mgpu_slice.max_num_vertex_associates = max_num_vertex_associates;
      mgpu_slice.max_num_value__associates = max_num_value__associates;
      GUARD_CU(mgpu_slice.Init(num_gpus, gpu_idx[gpu], sub_graph.nodes,
                               sub_graph.nodes * queue_factors[0],
                               sub_graph.GpT::in_counter + 0,
                               sub_graph.GpT::out_counter + 0, trans_factor,
                               skip_makeout_selection));

#ifdef ENABLE_PERFORMANCE_PROFILING
      iter_sub_queue_time[gpu].clear();
      iter_full_queue_time[gpu].clear();
      iter_total_time[gpu].clear();
      iter_full_queue_nodes_queued[gpu].clear();
      iter_full_queue_edges_queued[gpu].clear();
#endif
    }
    return retval;
  }

  /**
   * @brief Reset base enactor, called within Enactor::Reset() function
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < num_gpus; gpu++) {
      if (target & util::DEVICE) {
        GUARD_CU(util::SetDevice(gpu_idx[gpu]));
      }
      for (int peer = 0; peer < num_gpus; peer++) {
        GUARD_CU(enactor_slices[gpu * num_gpus + peer].Reset(target));
      }

      GUARD_CU(mgpu_slices[gpu].Reset(target));
#ifdef ENABLE_PERFORMANCE_PROFILING
      iter_sub_queue_time[gpu].push_back(std::vector<double>());
      iter_full_queue_time[gpu].push_back(std::vector<double>());
      iter_total_time[gpu].push_back(std::vector<double>());
      iter_full_queue_nodes_queued[gpu].push_back(std::vector<SizeT>());
      iter_full_queue_edges_queued[gpu].push_back(std::vector<SizeT>());
#endif

      thread_slices[gpu].status = ThreadSlice::Status::Wait;
    }
    return retval;
  }

  cudaError_t Sync() {
    cudaError_t retval = cudaSuccess;
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(gpu_idx[gpu]));
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }
    return retval;
  }

  /**
   * @brief Initialize the controling threads on CPU, called within
   * Enactor::Init() function
   * @tparam EnactorT Type of enactor
   * @param enactor pointer to the enactor
   * @param thread_func The function pointer to the CPU thread
   * \return cudaError_t error message(s), if any
   */
  template <typename EnactorT>
  cudaError_t Init_Threads(EnactorT *enactor, CUT_THREADROUTINE thread_func) {
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      thread_slices[gpu].thread_num = gpu;
      thread_slices[gpu].problem = (void *)enactor->problem;
      thread_slices[gpu].enactor = (void *)enactor;
      // thread_slices[gpu].context       = &(context[gpu*this->num_gpus]);
      thread_slices[gpu].status = ThreadSlice::Status::Inited;
      // thread_slices[gpu].thread_Id     = cutStartThread(
      //        thread_func,
      //        (void*)&(thread_slices[gpu]));
      // thread_Ids[gpu] = thread_slices[gpu].thread_Id;
    }

    // for (int gpu=0; gpu < this->num_gpus; gpu++)
    //{
    //    while (thread_slices[gpu].status != ThreadSlice::Status::Idle)
    //    {
    //        sleep(0);
    //        //std::this_thread::sleep_for(std::chrono::microseconds(0));
    //        //std::this_thread::yield();
    //    }
    //}
    return cudaSuccess;
  }

  /**
   * @brief Run the CPU threads, called within the Enactor::Enact() function
   * \return cudaError_t error message(s), if any
   */
  template <typename EnactorT>
  cudaError_t Run_Threads(EnactorT *enactor) {
    cudaError_t retval = cudaSuccess;

    // for (int gpu=0; gpu< num_gpus; gpu++)
    //{
    //    thread_slices[gpu].status = ThreadSlice::Status::Running;
    //}
    // for (int gpu=0; gpu< num_gpus; gpu++)
    //{
    //    while (thread_slices[gpu].status != ThreadSlice::Status::Idle)
    //    {
    //        sleep(0);
    //        //std::this_thread::sleep_for(std::chrono::microseconds(0));
    //        //std::this_thread::yield();
    //    }
    //}

#pragma omp parallel for num_threads(num_gpus)
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      auto &retval = enactor_slices[gpu * num_gpus].enactor_stats.retval;
      retval = util::SetDevice(gpu_idx[gpu]);
      if (retval == cudaSuccess) {
        enactor->Run(thread_slices[gpu]);
      }
    }

    for (int gpu = 0; gpu < num_gpus * num_gpus; gpu++) {
      GUARD_CU(enactor_slices[gpu].enactor_stats.retval);
    }
    return retval;
  }

  /**
   * @brief Kill the CPU threads, called within the Enactor::Release() function
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Kill_Threads() {
    cudaError_t retval = cudaSuccess;

    // if (thread_slices.GetPointer(util::HOST) != NULL)
    //{
    //    for (int gpu = 0; gpu < this->num_gpus; gpu++)
    //        thread_slices[gpu].status = ThreadSlice::Status::ToKill;
    //    cutWaitForThreads(thread_Ids + 0, this->num_gpus);
    //}
    return retval;
  }
};  // EnactorBase

}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
