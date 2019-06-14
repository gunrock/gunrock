// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * info_rapidjson.cuh
 *
 * @brief Running statistic collector with rapidjson
 */

#pragma once

#include <cstdio>
#include <vector>
#include <ctime>
#include <rapidjson/prettywriter.h>
#include <rapidjson/filewritestream.h>
#include <gunrock/util/gitsha1.h>

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace util {

cudaError_t UseParameters_info(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(parameters.Use<bool>(
      "json",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to output statistics in json format", __FILE__,
      __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "jsonfile",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "", "Filename to output statistics in json format", __FILE__, __LINE__));

  GUARD_CU(parameters.Use<std::string>(
      "jsondir",
      util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      "", "Directory to output statistics in json format", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Info data structure contains running statistics.
 */
struct Info {
 private:
  double total_elapsed;               // sum of running times
  double max_elapsed;                 // maximum running time
  double min_elapsed;                 // minimum running time
  std::vector<double> process_times;  // array of running times
  int num_runs;                       // number of runs
  int64_t nodes_visited;
  int64_t edges_visited;
  double m_teps;
  int64_t search_depth;
  double avg_duty;
  int64_t nodes_queued;
  int64_t edges_queued;
  double nodes_redundance;
  double edges_redundance;
  // double     load_time;
  double preprocess_time;
  double postprocess_time;
  double write_time;
  double total_time;
  std::string time_str;
  std::string algorithm_name;

  util::Parameters *parameters;
  std::string json_filename;
  std::FILE *json_file;
  char *json_buffer;
  size_t buff_size;
  rapidjson::FileWriteStream *json_stream;
  rapidjson::PrettyWriter<rapidjson::FileWriteStream> *json_writer;

 public:
  /**
   * @brief Info default constructor
   */
  Info() { AssignInitValues(); }

  template <typename GraphT>
  Info(std::string algorithm_name, util::Parameters &parameters,
       GraphT &graph) {
    AssignInitValues();
    Init(algorithm_name, parameters, graph);
  }

  void AssignInitValues() {
    parameters = NULL;
    json_filename = "";
    json_buffer = NULL;
    buff_size = 0;
    json_stream = NULL;
    json_writer = NULL;
    json_file = NULL;
  }

  ~Info() { Release(); }

  cudaError_t Release() {
    parameters = NULL;
    json_filename = "";
    delete[] json_buffer;
    json_buffer = NULL;
    delete json_stream;
    json_stream = NULL;
    delete json_writer;
    json_writer = NULL;
    return cudaSuccess;
  }

  /**
   * @brief Initialization process for Info.
   *
   * @param[in] algorithm_name Algorithm name.
   * @param[in] args Command line arguments.
   */
  void InitBase(std::string algorithm_name, util::Parameters &parameters) {
    this->algorithm_name = algorithm_name;
    this->parameters = &parameters;

    total_elapsed = 0;
    max_elapsed = 0;
    min_elapsed = 1e26;
    num_runs = 0;

    time_t now = time(NULL);
    time_str = std::string(ctime(&now));
    if (parameters.Get<bool>("json")) {
      json_filename = "";
    } else if (parameters.Get<std::string>("jsonfile") != "") {
      json_filename = parameters.Get<std::string>("jsonfile");
    } else if (parameters.Get<std::string>("jsondir") != "") {
      std::string dataset = parameters.Get<std::string>("dataset");
      std::string dir = parameters.Get<std::string>("jsondir");
      json_filename = dir + "/" + algorithm_name + "_" +
                      ((dataset != "") ? (dataset + "_") : "") + time_str +
                      ".json";

      char bad_chars[] = ":\n";
      for (unsigned int i = 0; i < strlen(bad_chars); ++i) {
        json_filename.erase(std::remove(json_filename.begin(),
                                        json_filename.end(), bad_chars[i]),
                            json_filename.end());
      }
    } else {
      return;
    }

    buff_size = 65536;
    if (json_buffer == NULL) json_buffer = new char[buff_size];
    if (json_filename != "") {
      json_file = std::fopen(json_filename.c_str(), "w");
    } else {
      json_file = stdout;
    }
    json_stream =
        new rapidjson::FileWriteStream(json_file, json_buffer, buff_size);
    json_writer =
        new rapidjson::PrettyWriter<rapidjson::FileWriteStream>(*json_stream);
    json_writer->StartObject();

    SetVal("engine", "Gunrock");
    SetVal("command-line", parameters.Get_CommandLine());
    // SetVal("sysinfo", sysinfo.getSysinfo());
    // SetVal("gpuinfo", gpuinfo.getGpuinfo());
    // SetVal("userinfo", userinfo.getUserinfo());

#ifdef BOOST_FOUND
#if BOOST_COMP_CLANG
    SetVal("compiler", BOOST_COMP_CLANG_NAME);
    SetVal("compiler-version", BOOST_COMP_CLANG_DETECTION);
#elif BOOST_COMP_GNUC
    SetVal("compiler", BOOST_COMP_GNUC_NAME);
    SetVal("compiler-version", BOOST_COMP_GNUC_DETECTION);
#endif
#else
#ifdef __clang__
    SetVal("compiler", "Clang");
    SetVal("compiler-version", (__clang_major__ % 100) * 10000000 +
                                   (__clang_minor__ % 100) * 100000 +
                                   (__clang_patchlevel__ % 100000));
#else
    SetVal("compiler", "Gnu GCC C/C++");
#ifdef __GNUC_PATCHLEVEL__
    SetVal("compiler-version", (__GNUC__ % 100) * 10000000 +
                                   (__GNUC_MINOR__ % 100) * 100000 +
                                   (__GNUC_PATCHLEVEL__ % 100000));
#else
    SetVal("compiler-version",
           (__GNUC__ % 100) * 10000000 + (__GNUC_MINOR__ % 100) * 100000);
#endif
#endif
#endif

    SetVal("time", time_str);
    SetVal("gunrock-version", XSTR(GUNROCKVERSION));
    SetVal("git-commit-sha", g_GIT_SHA1);
    SetVal("load-time", parameters.Get<float>("load-time"));
    SetVal("algorithm", algorithm_name);

    parameters.List(*this);
  }

  /**
   * @brief Initialization process for Info.
   * @param[in] algorithm_name Algorithm name.
   * @param[in] parameters running parameters.
   * @param[in] graph The graph.
   */
  template <typename GraphT>
  void Init(std::string algorithm_name, util::Parameters &parameters,
            GraphT &graph) {
    InitBase(algorithm_name, parameters);
    // if not set or something is wrong, set it to the largest vertex ID
    // if (info["destination_vertex"].get_int64() < 0 ||
    //    info["destination_vertex"].get_int64() >= graph.nodes)
    //    info["destination_vertex"] = graph.nodes - 1;

    SetVal("setdev-degree", graph::GetStddevDegree(graph));
    SetVal("num-vertices", graph.nodes);
    SetVal("num-edges", graph.edges);
  }

  template <typename DummyT, typename T>
  void SetBool(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetBool(DummyT name, const bool &val) {
    if (json_writer != NULL) json_writer->Bool(val);
  }

  template <typename DummyT, typename T>
  void SetInt(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetInt(DummyT name, const int &val) {
    if (json_writer != NULL) json_writer->Int(val);
  }

  template <typename DummyT, typename T>
  void SetUint(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetUint(DummyT name, const unsigned int &val) {
    if (json_writer != NULL) json_writer->Uint(val);
  }

  template <typename DummyT, typename T>
  void SetInt64(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetInt64(DummyT name, const int64_t &val) {
    if (json_writer != NULL) json_writer->Int64(val);
  }

  template <typename DummyT, typename T>
  void SetUint64(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetUint64(DummyT name, const uint64_t &val) {
    if (json_writer != NULL) json_writer->Uint64(val);
  }

  template <typename DummyT, typename T>
  void SetDouble(DummyT name, const T &val) {
    // util::PrintMsg("Escape val, name = " + name);
    if (json_writer != NULL) json_writer->Null();
  }

  template <typename DummyT>
  void SetDouble(DummyT name, const float &val) {
    if (json_writer != NULL) json_writer->Double(double(val));
  }

  template <typename DummyT>
  void SetDouble(DummyT name, const double &val) {
    if (json_writer != NULL) json_writer->Double(val);
  }

  template <typename T>
  void SetVal(std::string name, const T &val, bool write_name = true) {
    if (json_writer == NULL) return;
    if (write_name) json_writer->Key(name.c_str());
    auto tidx = std::type_index(typeid(T));

    if (tidx == std::type_index(typeid(bool)))
      SetBool(name, val);
    else if (tidx == std::type_index(typeid(char)) ||
             tidx == std::type_index(typeid(signed char)) ||
             tidx == std::type_index(typeid(short)) ||
             tidx == std::type_index(typeid(int)))
      SetInt(name, val);
    else if (tidx == std::type_index(typeid(unsigned char)) ||
             tidx == std::type_index(typeid(unsigned short)) ||
             tidx == std::type_index(typeid(unsigned int)))
      SetUint(name, val);
    else if (tidx == std::type_index(typeid(long)) ||
             tidx == std::type_index(typeid(long long)))
      SetInt64(name, val);
    else if (tidx == std::type_index(typeid(unsigned long)) ||
             tidx == std::type_index(typeid(unsigned long long)))
      SetUint64(name, val);
    else if (tidx == std::type_index(typeid(float)) ||
             tidx == std::type_index(typeid(double)) ||
             tidx == std::type_index(typeid(long double)))
      SetDouble(name, val);
    else {
      std::ostringstream ostr;
      ostr << val;
      std::string str = ostr.str();
      json_writer->String(str.c_str());
    }
  }

  template <typename T>
  void SetVal(std::string name, const std::vector<T> &vec) {
    if (json_writer == NULL) return;
    json_writer->Key(name.c_str());
    json_writer->StartArray();
    for (auto it = vec.begin(); it != vec.end(); it++) SetVal(name, *it, false);
    json_writer->EndArray();
  }

  void CollectSingleRun(double single_elapsed) {
    total_elapsed += single_elapsed;
    process_times.push_back(single_elapsed);
    if (max_elapsed < single_elapsed) max_elapsed = single_elapsed;
    if (min_elapsed > single_elapsed) min_elapsed = single_elapsed;
    num_runs++;
  }

  /**
   * @brief Compute statistics common to all primitives.
   *
   * @param[in] enactor_stats
   * @param[in] elapsed
   * @param[in] labels
   * @param[in] get_traversal_stats
   */
  template <typename EnactorT, typename T>
  cudaError_t ComputeCommonStats(EnactorT &enactor, const T *labels = NULL,
                                 bool get_traversal_stats = false) {
    cudaError_t retval = cudaSuccess;
    double total_lifetimes = 0;
    double total_runtimes = 0;

    // traversal stats
    edges_queued = 0;
    nodes_queued = 0;
    search_depth = 0;
    nodes_visited = 0;
    edges_visited = 0;
    m_teps = 0.0f;
    edges_redundance = 0.0f;
    nodes_redundance = 0.0f;

    std::vector<int> device_list = parameters->Get<std::vector<int> >("device");
    int num_gpus = device_list.size();
    auto graph = enactor.problem->org_graph[0];

    for (int gpu = 0; gpu < num_gpus; ++gpu) {
      int my_gpu_idx = device_list[gpu];
      if (num_gpus != 1) {
        GUARD_CU(util::SetDevice(my_gpu_idx));
      }
      GUARD_CU(cudaDeviceSynchronize());

      for (int peer = 0; peer < num_gpus; ++peer) {
        auto &estats =
            enactor.enactor_slices[gpu * num_gpus + peer].enactor_stats;
        if (get_traversal_stats) {
          edges_queued += estats.edges_queued[0];
          GUARD_CU(estats.edges_queued.Move(util::DEVICE, util::HOST));
          edges_queued += estats.edges_queued[0];

          nodes_queued += estats.nodes_queued[0];
          GUARD_CU(estats.nodes_queued.Move(util::DEVICE, util::HOST));
          nodes_queued += estats.nodes_queued[0];

          if (estats.iteration > search_depth) {
            search_depth = estats.iteration;
          }
        }
        total_lifetimes += estats.total_lifetimes;
        total_runtimes += estats.total_runtimes;
      }
    }

#ifdef RECORD_PER_ITERATION_STATS
    if (get_traversal_stats) {
      // TODO: collect info for multi-GPUs
      EnactorStats *estats = enactor_stats;
      json_spirit::mArray per_iteration_advance_runtime;
      json_spirit::mArray per_iteration_advance_mteps;
      json_spirit::mArray per_iteration_advance_input_frontier;
      json_spirit::mArray per_iteration_advance_output_frontier;
      json_spirit::mArray per_iteration_advance_dir;
      GetPerIterationAdvanceStats(
          estats.per_iteration_advance_time, estats.per_iteration_advance_mteps,
          estats.per_iteration_advance_input_edges,
          estats.per_iteration_advance_output_edges,
          estats.per_iteration_advance_direction, per_iteration_advance_runtime,
          per_iteration_advance_mteps, per_iteration_advance_input_frontier,
          per_iteration_advance_output_frontier, per_iteration_advance_dir);

      SetVal("per_iteration_advance_runtime", per_iteration_advance_runtime);
      SetVal("per_iteration_advance_mteps", per_iteration_advance_mteps);
      SetVal("per_iteration_advance_input_frontier",
             per_iteration_advance_input_frontier);
      SetVal("per_iteration_advance_output_frontier",
             per_iteration_advance_output_frontier);
      SetVal("per_iteration_advance_direction", per_iteration_advance_dir);
    }
#endif

    avg_duty = (total_lifetimes > 0)
                   ? double(total_runtimes) / total_lifetimes * 100.0
                   : 0.0f;

    double elapsed = total_elapsed / num_runs;
    SetVal("elapsed", elapsed);
    SetVal("average-duty", avg_duty);
    SetVal("search-depth", search_depth);

    if (get_traversal_stats) {
      SetVal("edges-queued", edges_queued);
      SetVal("nodes-queued", nodes_queued);
    }

    // TODO: compute traversal stats
    if (get_traversal_stats) {
      if (labels != NULL)
        for (int64_t v = 0; v < graph.nodes; ++v) {
          if (util::isValid(labels[v]) &&
              labels[v] != util::PreDefinedValues<T>::MaxValue) {
            ++nodes_visited;
            edges_visited += graph.GetNeighborListLength(v);
          }
        }
      if (algorithm_name == "BC") {
        // for betweenness should count the backward phase too.
        edges_visited = 2 * edges_queued;
      } else if (algorithm_name == "PageRank") {
        edges_visited = graph.edges;
        nodes_visited = graph.nodes;
      }

      if (nodes_queued >
          nodes_visited) {  // measure duplicate nodes put through queue
        nodes_redundance =
            ((double)nodes_queued - nodes_visited) / nodes_visited;
      }

      if (edges_queued > edges_visited) {
        // measure duplicate edges put through queue
        edges_redundance =
            ((double)edges_queued - edges_visited) / edges_visited;
      }
      nodes_redundance *= 100;
      edges_redundance *= 100;

      m_teps = (double)edges_visited / (elapsed * 1000.0);

      SetVal("nodes-visited", nodes_visited);
      SetVal("edges-visited", edges_visited);
      SetVal("nodes-redundance", nodes_redundance);
      SetVal("edges-redundance", edges_redundance);
      SetVal("m-teps", m_teps);
    }

    return retval;
  }

  /**
   * @brief Compute statistics common to all traversal primitives.
   * @param[in] enactor The Enactor
   * @param[in] labels
   */
  template <typename EnactorT, typename T>
  void ComputeTraversalStats(EnactorT &enactor, const T *labels = NULL) {
    ComputeCommonStats(enactor, labels, true);
  }

  /**
   * @brief Display running statistics.
   * @param[in] verbose Whether or not to print extra information.
   */
  void DisplayStats(bool verbose = true) {
    int num_runs = parameters->Get<int>("num-runs");
    double elapsed = total_elapsed / num_runs;
    int num_srcs = 0;
    std::vector<int64_t> srcs;
    if (parameters->Have("srcs")) {
      srcs = parameters->Get<std::vector<int64_t> >("srcs");
      num_srcs = srcs.size();
    }

    util::PrintMsg("[" + algorithm_name + "] finished.");
    util::PrintMsg(" avg. elapsed: " + std::to_string(elapsed) + " ms");
    util::PrintMsg(" iterations: " + std::to_string(search_depth));

    if (!verbose) return;
    if (nodes_visited != 0 && nodes_visited < 5) {
      util::PrintMsg("Fewer than 5 vertices visited.");
      return;
    }

    util::PrintMsg(" min. elapsed: " + std::to_string(min_elapsed) + " ms",
                   min_elapsed > 0);
    util::PrintMsg(" max. elapsed: " + std::to_string(max_elapsed) + " ms",
                   max_elapsed > 0);

    util::PrintMsg(" rate: " + std::to_string(m_teps) + " MiEdges/s",
                   m_teps > 0.01);

    util::PrintMsg(" average CTA duty: " + std::to_string(avg_duty) + "%%",
                   avg_duty > 0.01);

    if (nodes_visited != 0 && edges_visited != 0) {
      util::PrintMsg(" src: " + std::to_string(srcs[num_runs % num_srcs]),
                     num_srcs != 0);
      util::PrintMsg(" nodes_visited: " + std::to_string(nodes_visited));
      util::PrintMsg(" edges_visited: " + std::to_string(edges_visited));
    }
    util::PrintMsg(" nodes queued: " + std::to_string(nodes_queued),
                   nodes_queued > 0);
    util::PrintMsg(" edges queued: " + std::to_string(edges_queued),
                   edges_queued > 0);

    util::PrintMsg(" nodes redundance: " + std::to_string(nodes_redundance),
                   nodes_redundance > 0.01);
    util::PrintMsg(" edges redundance: " + std::to_string(edges_redundance),
                   edges_redundance > 0.01);

    util::PrintMsg(" load time: " + parameters->Get<std::string>("load-time") +
                   " ms");
    util::PrintMsg(" preprocess time: " + std::to_string(preprocess_time) +
                   " ms");
    util::PrintMsg(" postprocess time: " + std::to_string(postprocess_time) +
                   " ms");
    // if (parameters -> Get<std::string>("output_filename") != "")
    //    util::PrintMsg(" write time: " + std::to_string(write_time) + " ms");
    util::PrintMsg(" total time: " + std::to_string(total_time) + " ms");
  }

  void Finalize(double postprocess_time, double total_time) {
    bool quiet = parameters->Get<bool>("quiet");

    preprocess_time = parameters->Get<double>("preprocess-time");
    SetVal("process-times", process_times);
    SetVal("min-process-time", min_elapsed);
    SetVal("max-process-time", max_elapsed);
    SetVal("postprocess-time", postprocess_time);
    SetVal("total-time", total_time);

    this->postprocess_time = postprocess_time;
    this->total_time = total_time;
    if (!quiet) {
      DisplayStats();
    }

    if (json_writer != NULL) json_writer->EndObject();
  }
};

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
