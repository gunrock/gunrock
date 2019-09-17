// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_pr.cu
 *
 * @brief Simple test driver program for PageRank.
 */

#include <gunrock/app/pr/pr_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

cudaError_t UseParameters(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(parameters.Use<int>(
      "num-elements",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      1024 * 1024 * 100, "number of elements per GPU to test on", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<int>(
      "for-size",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      3276800, "number of operations to perform per repeat", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<int>(
      "num-repeats",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      100, "number of times to repeat the operations", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "device",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER, 0,
      "the devices to run on", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "rand-seed",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      util::PreDefinedValues<int>::InvalidValue,
      "rand seed to generate random numbers; default is time(NULL)", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "access-type",
      util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      "All", "Memory access type, <Random | Regular | All>", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "operation",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "All", "Operations to test, <Read | Write | Update | All>", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "bandwidth-latency",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "All", "Test type, <Bandwidth | Latency | All>", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<int>(
      "num-runs",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      2, "how many times to repeat the testing", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<bool>(
      "use-UVM",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to include UVM test", __FILE__, __LINE__));
  return retval;
}

using BWLFlag = uint32_t;
enum : BWLFlag {
  OPERATION_BASE = 0x0F,
  READ = 0x01,
  WRITE = 0x02,
  UPDATE = 0x04,

  ACCESS_BASE = 0xF0,
  RANDOM = 0x10,
  REGULAR = 0x20,

  BL_BASE = 0xF00,
  BANDWIDTH = 0x100,
  LATENCY = 0x200,
};

// Test routines

typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<float> Distribution;

template <typename GraphT, typename ArrayT, typename ArrayT2>
cudaError_t Test_BWL(
    util::Parameters &parameters, GraphT &graph,
    util::Array1D<typename GraphT::SizeT, typename GraphT::VertexT>
        *gpu_elements,
    util::Array1D<typename GraphT::SizeT, typename GraphT::VertexT>
        *gpu_results,
    ArrayT &host_elements, ArrayT &host_results, ArrayT2 &all_elements,
    ArrayT2 &all_results, cudaStream_t *gpu_streams, int **peer_accessables,
    float **timings, cudaError_t *retvals) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  cudaError_t retval = cudaSuccess;
  auto devices = parameters.template Get<std::vector<int>>("device");
  int num_devices = devices.size();
  BWLFlag operation_flag, access_flag, bl_flag;
  std::string operation_str = parameters.template Get<std::string>("operation");
  std::string access_str = parameters.template Get<std::string>("access-type");
  std::string bl_str =
      parameters.template Get<std::string>("bandwidth-latency");
  bool use_UVM = parameters.template Get<bool>("use-UVM");
  uint32_t num_elements = parameters.template Get<uint32_t>("num-elements");
  if (operation_str == "Read")
    operation_flag = READ;
  else if (operation_str == "Write")
    operation_flag = WRITE;
  else if (operation_str == "Update")
    operation_flag = UPDATE;
  if (access_str == "Random")
    access_flag = RANDOM;
  else if (access_str == "Regular")
    access_flag = REGULAR;
  if (bl_str == "Bandwidth")
    bl_flag = BANDWIDTH;
  else if (bl_str == "Latency")
    bl_flag = LATENCY;
  uint32_t for_size = parameters.template Get<uint32_t>("for-size");
  uint32_t num_repeats = parameters.template Get<uint32_t>("num-repeats");

  for (int peer_offset = 0; peer_offset <= num_devices + 1; peer_offset++) {
#pragma omp parallel num_threads(num_devices)
    {
      do {
        int thread_num = omp_get_thread_num();
        auto device_idx = devices[thread_num];
        auto &retval = retvals[thread_num];
        auto &stream = gpu_streams[thread_num];
        int peer = (thread_num + peer_offset) % num_devices;
        auto elements = gpu_elements[thread_num].GetPointer(util::DEVICE);
        auto &results = gpu_results[thread_num];
        auto peer_elements = gpu_elements[peer].GetPointer(util::DEVICE);
        auto peer_results = gpu_results[peer].GetPointer(util::DEVICE);
        auto &all_element = all_elements[thread_num];
        auto &all_result = all_results[thread_num];
        float elapsed = -1;
#pragma omp barrier
        if (peer_offset >= num_devices)
          peer = peer_offset;
        else if (peer_accessables[thread_num][peer] == 0)
          break;
        if (peer_offset == num_devices && !use_UVM) break;

        util::CpuTimer cpu_timer;
        cpu_timer.Start();

        if (peer_offset <= num_devices) {
          if (operation_flag == READ) {
            VertexT *sources =
                ((peer_offset == num_devices) ? (host_elements + 0)
                                              : peer_elements);
            retval = results.ForAll(
                [elements, sources, num_elements, access_flag,
                 num_repeats] __host__ __device__(VertexT * result,
                                                  const SizeT &pos) {
                  for (int i = 0; i < num_repeats; i++) {
                    VertexT new_pos = pos + i * 65536;
                    new_pos = new_pos % num_elements;
                    if (access_flag == RANDOM) new_pos = elements[pos];

                    result[pos] = sources[new_pos];
                  }
                },
                (bl_flag == LATENCY) ? 1 : for_size, util::DEVICE, stream,
                1280);
          } else if (operation_flag == WRITE) {
            VertexT *targets =
                ((peer_offset == num_devices) ? (host_results + 0)
                                              : peer_results);
            retval = results.ForAll(
                [elements, targets, num_elements, access_flag,
                 num_repeats] __host__ __device__(VertexT * result,
                                                  const SizeT &pos) {
                  for (int i = 0; i < num_repeats; i++) {
                    VertexT new_pos = pos + i * 65536;
                    new_pos = new_pos % num_elements;
                    if (access_flag == RANDOM) new_pos = elements[pos];

                    targets[new_pos] = new_pos;
                  }
                },
                (bl_flag == LATENCY) ? 1 : for_size, util::DEVICE, stream,
                1280);
          } else if (operation_flag == UPDATE) {
            VertexT *targets =
                ((peer_offset == num_devices) ? (host_results + 0)
                                              : peer_results);
            retval = results.ForAll(
                [elements, targets, num_elements, access_flag,
                 num_repeats] __host__ __device__(VertexT * result,
                                                  const SizeT &pos) {
                  for (int i = 0; i < num_repeats; i++) {
                    VertexT new_pos = pos + i * 65536;
                    new_pos = new_pos % num_elements;
                    if (access_flag == RANDOM) new_pos = elements[pos];

                    targets[new_pos] += 1;
                  }
                },
                (bl_flag == LATENCY) ? 1 : for_size, util::DEVICE, stream,
                1280);
          }
        } else {  // All to all
          if (operation_flag == READ) {
            retval = results.ForAll(
                [elements, num_elements, access_flag, num_repeats, all_element,
                 num_devices] __host__ __device__(VertexT * result,
                                                  const SizeT &pos) {
                  for (int i = 0; i < num_repeats; i++) {
                    VertexT new_pos = pos + i * 65536;
                    new_pos = new_pos % num_elements;
                    if (access_flag == RANDOM) new_pos = elements[pos];

                    result[pos] = all_element[new_pos % num_devices][new_pos];
                  }
                },
                (bl_flag == LATENCY) ? 1 : for_size, util::DEVICE, stream,
                1280);
          } else if (operation_flag == WRITE) {
            retval = results.ForAll(
                [elements, num_elements, all_result, num_devices, access_flag,
                 num_repeats] __host__ __device__(VertexT * result,
                                                  const SizeT &pos) {
                  for (int i = 0; i < num_repeats; i++) {
                    VertexT new_pos = pos + i * 65536;
                    new_pos = new_pos % num_elements;
                    if (access_flag == RANDOM) new_pos = elements[pos];

                    all_result[new_pos % num_devices][new_pos] = new_pos;
                  }
                },
                (bl_flag == LATENCY) ? 1 : for_size, util::DEVICE, stream,
                1280);
          } else if (operation_flag == UPDATE) {
            retval = results.ForAll(
                [elements, num_elements, all_result, num_devices, access_flag,
                 num_repeats] __host__ __device__(VertexT * result,
                                                  const SizeT &pos) {
                  for (int i = 0; i < num_repeats; i++) {
                    VertexT new_pos = pos + i * 65536;
                    new_pos = new_pos % num_elements;
                    if (access_flag == RANDOM) new_pos = elements[pos];

                    all_result[new_pos % num_devices][new_pos] += 1;
                  }
                },
                (bl_flag == LATENCY) ? 1 : for_size, util::DEVICE, stream,
                1280);
          }
        }
        if (retval) break;
        retval =
            util::GRError(cudaStreamSynchronize(stream),
                          "cudaStreamSynchronize failed", __FILE__, __LINE__);
        cpu_timer.Stop();
        elapsed = cpu_timer.ElapsedMillis();
        timings[thread_num][peer] = elapsed;
      } while (false);
    }
  }
  for (int i = 0; i < num_devices; i++)
    if (retvals[i]) return retvals[i];

  std::string title = access_str + " " + operation_str + " " + bl_str;
  if (bl_flag == BANDWIDTH)
    title = title + " (GB/s)";
  else if (bl_flag == LATENCY)
    title = title + " (us)";

  std::cout << title << std::endl;
  for (int i = 0; i < num_devices; i++)
    std::cout << (i == 0 ? "Peer" : "") << "\t" << devices[i];
  std::cout << "\tHost\tAll2All" << std::endl;
  for (int gpu = 0; gpu < num_devices; gpu++) {
    std::cout << "GPU " << gpu;
    for (int peer = 0; peer <= num_devices + 1; peer++) {
      std::cout << "\t";
      if (peer_accessables[gpu][peer] == 0 ||
          (!use_UVM && peer == num_devices)) {
        std::cout << "--";
        continue;
      }

      auto elapsed = timings[gpu][peer];
      if (bl_flag == BANDWIDTH) {
        std::cout << 1.0 / 1024 / 1024 / 1024 * for_size * num_repeats *
                         sizeof(VertexT) / elapsed * 1000;
      }

      if (bl_flag == LATENCY) {
        std::cout << elapsed / num_repeats * 1000;
      }
    }
    std::cout << std::endl;
  }
  return retval;
}

/******************************************************************************
 * Main
 ******************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct {
  /**
   * @brief the actual main function, after type switching
   * @tparam VertexT    Type of vertex identifier
   * @tparam SizeT      Type of graph size, i.e. type of edge identifier
   * @tparam ValueT     Type of edge values
   * @param  parameters Command line parameters
   * @param  v,s,val    Place holders for type deduction
   * \return cudaError_t error message(s), if any
   */
  template <typename VertexT,  // Use int as the vertex identifier
            typename SizeT,    // Use int as the graph size type
            typename ValueT>   // Use int as the value type
  cudaError_t
  operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val) {
    typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_COO>
        GraphT;
    cudaError_t retval = cudaSuccess;

    GraphT graph;
    std::vector<std::string> switches{"num-elements", "for-size",
                                      "num-repeats",  "access-type",
                                      "operation",    "bandwidth-latency"};
    if (parameters.Get<std::string>("access-type") == "All")
      parameters.Set("access-type", "Random,Regular");
    if (parameters.Get<std::string>("operation") == "All")
      parameters.Set("operation", "Read,Write,Update");
    if (parameters.Get<std::string>("bandwidth-latency") == "All")
      parameters.Set("bandwidth-latency", "Bandwidth,Latency");

    auto num_elements =
        parameters.template Get<std::vector<SizeT>>("num-elements");
    SizeT max_elements = 0;
    for (auto num_element : num_elements)
      if (max_elements < num_element) max_elements = num_element;
    auto for_sizes = parameters.template Get<std::vector<SizeT>>("for-size");
    SizeT max_for_size = 0;
    for (auto for_size : for_sizes)
      if (max_for_size < for_size) max_for_size = for_size;

    int rand_seed = parameters.template Get<int>("rand-seed");
    auto devices = parameters.template Get<std::vector<int>>("device");
    int num_devices = devices.size();
    cudaError_t *retvals = new cudaError_t[num_devices];
    util::Array1D<SizeT, VertexT> *gpu_elements =
        new util::Array1D<SizeT, VertexT>[num_devices];
    util::Array1D<SizeT, VertexT> *gpu_results =
        new util::Array1D<SizeT, VertexT>[num_devices];
    util::Array1D<SizeT, VertexT *> *all_elements =
        new util::Array1D<SizeT, VertexT *>[num_devices];
    util::Array1D<SizeT, VertexT *> *all_results =
        new util::Array1D<SizeT, VertexT *>[num_devices];
    cudaStream_t *gpu_streams = new cudaStream_t[num_devices];
    if (!util::isValid(rand_seed)) rand_seed = time(NULL);
    int **peer_accessables = new int *[num_devices + 1];
    float **timings = new float *[num_devices + 1];

    // util::Array1D<SizeT, VertexT, util::PINNED> host_elements;
    // util::Array1D<SizeT, VertexT, util::PINNED> host_results;
    // host_elements.SetName("host_elements");
    // host_results .SetName("host_results");
    // GUARD_CU(host_elements.Allocate(max_elements, util::HOST));
    // GUARD_CU(host_results .Allocate(max_elements, util::HOST));
    VertexT *host_elements = NULL;
    VertexT *host_results = NULL;
    GUARD_CU2(cudaMallocManaged((void **)(&host_elements),
                                (long long)max_elements * sizeof(VertexT)),
              "cudaMallocHost failed");
    GUARD_CU2(cudaMallocManaged((void **)(&host_results),
                                (long long)max_elements * sizeof(VertexT)),
              "cudaMallocHost failed");

    Engine engine_(rand_seed + 11 * num_devices);
    Distribution distribution_(0.0, 1.0);
    for (SizeT i = 0; i < max_elements; i++) {
      host_elements[i] = distribution_(engine_) * max_elements;
      if (host_elements[i] >= max_elements) host_elements[i] -= max_elements;
    }

    util::PrintMsg("num_devices = " + std::to_string(num_devices));
    util::PrintMsg("rand-seed = " + std::to_string(rand_seed));
#pragma omp parallel num_threads(num_devices)
    {
      do {
        int thread_num = omp_get_thread_num();
        auto device_idx = devices[thread_num];
        auto &retval = retvals[thread_num];
        auto &elements = gpu_elements[thread_num];
        auto &results = gpu_results[thread_num];
        auto &stream = gpu_streams[thread_num];
        auto &peer_accessable = peer_accessables[thread_num];
        auto &timing = timings[thread_num];
        peer_accessable = new int[num_devices + 10];
        timing = new float[num_devices + 10];
        for (int i = 0; i < num_devices + 10; i++) {
          peer_accessable[i] = 1;
        }

        util::PrintMsg("using device[" + std::to_string(thread_num) + "] " +
                       std::to_string(device_idx));
        retval = util::GRError(cudaSetDevice(device_idx),
                               "cudaSetDevice failed.", __FILE__, __LINE__);
        if (retval) break;
        retval = util::GRError(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags failed.", __FILE__, __LINE__);
        if (retval) break;

        if (thread_num == 0) {
          retval = util::GRError(
              cudaMemAdvise(host_elements + 0,
                            ((long long)max_elements) * sizeof(VertexT),
                            cudaMemAdviseSetReadMostly, device_idx),
              "cudaMemAdvise failed", __FILE__, __LINE__);
          if (retval) break;
        }

        retval = util::GRError(
            cudaMemAdvise(host_elements + 0,
                          (long long)max_elements * sizeof(VertexT),
                          cudaMemAdviseSetAccessedBy, device_idx),
            "cudaMemAdvise failed", __FILE__, __LINE__);
        if (retval) break;
        retval = util::GRError(
            cudaMemAdvise(host_results + 0,
                          (long long)max_elements * sizeof(VertexT),
                          cudaMemAdviseSetAccessedBy, device_idx),
            "cudaMemAdvise failed", __FILE__, __LINE__);
        if (retval) break;

        for (int peer_offset = 1; peer_offset < num_devices; peer_offset++) {
          int peer = devices[(thread_num + peer_offset) % num_devices];
          int peer_access_avail = 0;
          retval = util::GRError(
              cudaDeviceCanAccessPeer(&peer_access_avail, device_idx, peer),
              "cudaDeviceCanAccessPeer failed", __FILE__, __LINE__);
          if (retval) break;
          if (peer_access_avail) {
            retval = util::GRError(cudaDeviceEnablePeerAccess(peer, 0),
                                   "cudaDeviceEnablePeerAccess failed",
                                   __FILE__, __LINE__);
            if (retval) break;
          } else {
            peer_accessable[peer] = 0;
          }
          if (retval) break;
        }
        if (retval) break;

        elements.SetName("elements[" + std::to_string(thread_num) + "]");
        retval = elements.Allocate(max_elements, util::DEVICE | util::HOST);
        if (retval) break;
        results.SetName("results[" + std::to_string(thread_num) + "]");
        retval =
            results.Allocate(max(max_elements, max_for_size), util::DEVICE);
        if (retval) break;

        Engine engine(rand_seed + 11 * thread_num);
        Distribution distribution(0.0, 1.0);
        for (SizeT i = 0; i < max_elements; i++) {
          elements[i] = distribution(engine) * max_elements;
          if (elements[i] >= max_elements) elements[i] -= max_elements;
        }
        retval =
            elements.Move(util::HOST, util::DEVICE, max_elements, 0, stream);
        if (retval) break;
        retval =
            util::GRError(cudaStreamSynchronize(stream),
                          "cudaStreamSynchonorize failed", __FILE__, __LINE__);
        if (retval) break;
      } while (false);
    }
    for (int i = 0; i < num_devices; i++)
      if (retvals[i]) return retvals[i];

#pragma omp parallel num_threads(num_devices)
    {
      do {
        int thread_num = omp_get_thread_num();
        auto device_idx = devices[thread_num];
        auto &retval = retvals[thread_num];
        retval = util::GRError(cudaSetDevice(device_idx),
                               "cudaSetDevice failed.", __FILE__, __LINE__);
        if (retval) break;
        auto &all_element = all_elements[thread_num];
        auto &all_result = all_results[thread_num];

        retval = all_element.Allocate(num_devices, util::HOST | util::DEVICE);
        if (retval) break;
        retval = all_result.Allocate(num_devices, util::HOST | util::DEVICE);
        if (retval) break;

        for (int i = 0; i < num_devices; i++) {
          if (peer_accessables[thread_num][i] == 0) {
            all_element[i] = gpu_elements[thread_num].GetPointer(util::DEVICE);
            all_result[i] = gpu_results[thread_num].GetPointer(util::DEVICE);
            continue;
          }
          all_element[i] = gpu_elements[i].GetPointer(util::DEVICE);
          all_result[i] = gpu_results[i].GetPointer(util::DEVICE);
        }
        retval = all_element.Move(util::HOST, util::DEVICE);
        if (retval) break;
        retval = all_result.Move(util::HOST, util::DEVICE);
        if (retval) break;
      } while (false);
    }
    for (int i = 0; i < num_devices; i++)
      if (retvals[i]) return retvals[i];

    int num_runs = parameters.template Get<int>("num-runs");
    for (int i = 0; i < num_runs; i++) {
      GUARD_CU(app::Switch_Parameters(
          parameters, graph, switches,
          [devices, retvals, gpu_elements, gpu_results, host_elements,
           host_results, gpu_streams, peer_accessables, timings, all_elements,
           all_results](util::Parameters &parameters, GraphT &graph) {
            return Test_BWL(parameters, graph, gpu_elements, gpu_results,
                            host_elements, host_results, all_elements,
                            all_results, gpu_streams, peer_accessables, timings,
                            retvals);
          }));
    }

    for (int d = 0; d < num_devices; d++) {
      GUARD_CU2(cudaSetDevice(devices[d]), "cudaSetDevice failed");
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

      GUARD_CU(gpu_elements[d].Release());
      GUARD_CU(gpu_results[d].Release());
    }
    return retval;
  }
};

int main(int argc, char **argv) {
  cudaError_t retval = cudaSuccess;
  util::Parameters parameters("test pr");
  GUARD_CU(graphio::UseParameters(parameters));
  GUARD_CU(app::UseParameters_test(parameters));
  GUARD_CU(UseParameters(parameters));
  GUARD_CU(parameters.Parse_CommandLine(argc, argv));
  if (parameters.Get<bool>("help")) {
    parameters.Print_Help();
    return cudaSuccess;
  }
  GUARD_CU(parameters.Check_Required());

  return app::Switch_Types<app::VERTEXT_U32B | app::VERTEXT_U64B |
                           app::SIZET_U32B |   // app::SIZET_U64B |
                           app::VALUET_F32B |  // app::VALUET_F64B |
                           app::DIRECTED | app::UNDIRECTED>(parameters,
                                                            main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
