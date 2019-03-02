// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_cc.cu
 *
 * @brief Simple test driver program for connected component.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// CC includes
#include <gunrock/app/cc/cc_enactor.cuh>
#include <gunrock/app/cc/cc_problem.cuh>
#include <gunrock/app/cc/cc_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

// Boost includes for CPU CC reference algorithms
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>

#include <gunrock/util/shared_utils.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::cc;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

template <typename VertexId, typename SizeT>
struct CcList {
  VertexId root;
  SizeT histogram;

  CcList(VertexId root, SizeT histogram) : root(root), histogram(histogram) {}
};

template <typename CcList>
bool CCCompare(CcList elem1, CcList elem2) {
  return elem1.histogram > elem2.histogram;
}

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage() {
  printf(
      "test <graph-type> [graph-type-arguments]\n"
      "Graph type and graph type arguments:\n"
      "    market <matrix-market-file-name>\n"
      "        Reads a Matrix-Market coordinate-formatted graph of\n"
      "        directed/undirected edges from STDIN (or from the\n"
      "        optionally-specified file).\n"
      "    rmat (default: rmat_scale = 10, a = 0.57, b = c = 0.19)\n"
      "        Generate R-MAT graph as input\n"
      "        --rmat_scale=<vertex-scale>\n"
      "        --rmat_nodes=<number-nodes>\n"
      "        --rmat_edgefactor=<edge-factor>\n"
      "        --rmat_edges=<number-edges>\n"
      "        --rmat_a=<factor> --rmat_b=<factor> --rmat_c=<factor>\n"
      "        --rmat_seed=<seed>\n"
      "    rgg (default: rgg_scale = 10, rgg_thfactor = 0.55)\n"
      "        Generate Random Geometry Graph as input\n"
      "        --rgg_scale=<vertex-scale>\n"
      "        --rgg_nodes=<number-nodes>\n"
      "        --rgg_thfactor=<threshold-factor>\n"
      "        --rgg_threshold=<threshold>\n"
      "        --rgg_vmultipiler=<vmultipiler>\n"
      "        --rgg_seed=<seed>\n\n"
      "Optional arguments:\n"
      "[--device=<device_index>] Set GPU(s) for testing (Default: 0).\n"
      "[--instrumented]          Keep kernels statics [Default: Disable].\n"
      "                          total_queued, search_depth and barrier duty.\n"
      "                          (a relative indicator of load imbalance.)\n"
      "[--quick]                 Skip the CPU reference validation process.\n"
      "[--disable-size-check]    Disable frontier queue size check.\n"
      "[--grid-size=<grid size>] Maximum allowed grid size setting.\n"
      "[--queue-sizing=<factor>] Allocates a frontier queue sized at: \n"
      "                          (graph-edges * <factor>). (Default: 1.0)\n"
      "[--in-sizing=<in/out_queue_scale_factor>]\n"
      "                          Allocates a frontier queue sized at: \n"
      "                          (graph-edges * <factor>). (Default: 1.0)\n"
      "[--v]                     Print verbose per iteration debug info.\n"
      "[--iteration-num=<num>]   Number of runs to perform the test.\n"
      "[--partition-method=<random|biasrandom|clustered|metis>]\n"
      "                          Choose partitioner (Default use random).\n"
      "[--quiet]                 No output (unless --json is specified).\n"
      "[--json]                  Output JSON-format statistics to STDOUT.\n"
      "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
      "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
      "                          where name is auto-generated.\n");
}

/**
 * @brief Displays the CC result (i.e., number of components)
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] comp_ids Host-side vector to store computed component id for each
 * node
 * @param[in] nodes Number of nodes in the graph
 * @param[in] num_components Number of connected components in the graph
 * @param[in] roots Host-side vector stores the root for each node in the graph
 * @param[in] histogram Histogram of connected component ids
 */
template <typename VertexId, typename SizeT>
void DisplaySolution(VertexId *comp_ids, SizeT nodes, SizeT num_components,
                     VertexId *roots, SizeT *histogram) {
  typedef CcList<VertexId, SizeT> CcListType;
  // printf("Number of components: %d\n", num_components);

  if (nodes <= 40) {
    printf("[");
    for (VertexId i = 0; i < nodes; ++i) {
      PrintValue(i);
      printf(":");
      PrintValue(comp_ids[i]);
      printf(",");
      printf(" ");
    }
    printf("]\n");
  } else {
    // sort the components by size
    CcListType *cclist =
        (CcListType *)malloc(sizeof(CcListType) * num_components);
    for (SizeT i = 0; i < num_components; ++i) {
      cclist[i].root = roots[i];
      cclist[i].histogram = histogram[i];
    }
    std::stable_sort(cclist, cclist + num_components, CCCompare<CcListType>);

    // Print out at most top 10 largest components
    SizeT top = (num_components < 10) ? num_components : 10;
    printf("Top %lld largest components:\n", (long long)top);
    for (SizeT i = 0; i < top; ++i) {
      printf("CC ID: %lld, CC Root: %lld, CC Size: %lld\n", (long long)i,
             (long long)cclist[i].root, (long long)cclist[i].histogram);
    }

    free(cclist);
  }
}

/******************************************************************************
 * CC Testing Routines
 *****************************************************************************/

/**
 * @brief CPU-based reference CC algorithm using Boost Graph Library
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in]  graph  Reference to the CSR graph we process on
 * @param[out] labels Host-side vector to store the component id for each node
 * in the graph
 * @param[in] quiet Don't print out anything to stdout
 *
 * \return Number of connected components in the graph
 */
template <typename VertexId, typename SizeT, typename Value>
unsigned int ReferenceCC(const Csr<VertexId, SizeT, Value> &graph,
                         VertexId *labels, bool quiet = false) {
  using namespace boost;
  SizeT *row_offsets = graph.row_offsets;
  VertexId *column_indices = graph.column_indices;
  SizeT num_nodes = graph.nodes;

  typedef adjacency_list<vecS, vecS, undirectedS> Graph;
  Graph G;
  for (int i = 0; i < num_nodes; ++i) {
    for (int j = row_offsets[i]; j < row_offsets[i + 1]; ++j) {
      add_edge(i, column_indices[j], G);
    }
  }
  CpuTimer cpu_timer;
  cpu_timer.Start();
  SizeT num_components = connected_components(G, &labels[0]);
  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();

  if (!quiet) {
    printf("CPU CC finished in %lf msec.\n", elapsed);
  }
  return num_components;
}

/**
 * @brief Convert component IDs.
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] labels
 * @param[in] num_nodes
 * @param[in] num_components
 */
template <typename VertexId, typename SizeT>
void ConvertIDs(VertexId *labels, SizeT num_nodes, SizeT num_components) {
  VertexId *min_nodes = new VertexId[num_nodes];

  for (int cc = 0; cc < num_nodes; cc++) min_nodes[cc] = num_nodes;
  for (int node = 0; node < num_nodes; node++)
    if (min_nodes[labels[node]] > node) min_nodes[labels[node]] = node;
  for (int node = 0; node < num_nodes; node++)
    labels[node] = min_nodes[labels[node]];
  delete[] min_nodes;
  min_nodes = NULL;
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 *
 * \return cudaError_t object which indicates the success of
 * all CUDA function calls.
 */
template <typename VertexId, typename SizeT, typename Value>
cudaError_t RunTests(Info<VertexId, SizeT, Value> *info) {
  typedef CCProblem<VertexId, SizeT,
                    Value>
      Problem;  // use double buffer for advance and filter

  typedef CCEnactor<Problem>
      // INSTRUMENT,
      // DEBUG,
      // SIZE_CHECK >
      Enactor;

  // parse configurations from mObject info
  Csr<VertexId, SizeT, Value> *graph = info->csr_ptr;
  int max_grid_size = info->info["max_grid_size"].get_int();
  int num_gpus = info->info["num_gpus"].get_int();
  double max_queue_sizing = info->info["max_queue_sizing"].get_real();
  double max_queue_sizing1 = info->info["max_queue_sizing1"].get_real();
  double max_in_sizing = info->info["max_in_sizing"].get_real();
  std::string partition_method = info->info["partition_method"].get_str();
  double partition_factor = info->info["partition_factor"].get_real();
  int partition_seed = info->info["partition_seed"].get_int();
  bool quiet_mode = info->info["quiet_mode"].get_bool();
  bool quick_mode = info->info["quick_mode"].get_bool();
  bool stream_from_host = info->info["stream_from_host"].get_bool();
  std::string traversal_mode = info->info["traversal_mode"].get_str();
  bool instrument = info->info["instrument"].get_bool();
  bool debug = info->info["debug_mode"].get_bool();
  bool size_check = info->info["size_check"].get_bool();
  int iterations = info->info["num_iteration"].get_int();
  int communicate_latency = info->info["communicate_latency"].get_int();
  float communicate_multipy = info->info["communicate_multipy"].get_real();
  int expand_latency = info->info["expand_latency"].get_int();
  int subqueue_latency = info->info["subqueue_latency"].get_int();
  int fullqueue_latency = info->info["fullqueue_latency"].get_int();
  int makeout_latency = info->info["makeout_latency"].get_int();
  if (max_queue_sizing < 0) max_queue_sizing = 1.0;
  if (max_in_sizing < 0) max_in_sizing = 1.1;
  if (communicate_multipy > 1) max_in_sizing *= communicate_multipy;
  CpuTimer cpu_timer;
  cudaError_t retval;

  cpu_timer.Start();
  json_spirit::mArray device_list = info->info["device_list"].get_array();
  int *gpu_idx = new int[num_gpus];
  for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

  // TODO: remove after merge mgpu-cq
  ContextPtr *context = (ContextPtr *)info->context;
  cudaStream_t *streams = (cudaStream_t *)info->streams;

  // Allocate host-side array (for both reference and GPU-computed results)
  VertexId *reference_component_ids = new VertexId[graph->nodes];
  VertexId *h_component_ids = new VertexId[graph->nodes];
  VertexId *reference_check = (quick_mode) ? NULL : reference_component_ids;
  SizeT ref_num_components = 0;

  // printf("0: node %d: %d -> %d, node %d: %d -> %d\n", 131070,
  // graph->row_offsets[131070], graph->row_offsets[131071], 131071,
  // graph->row_offsets[131071], graph->row_offsets[131072]); for (int edge = 0;
  // edge < graph->edges; edge ++)
  //{
  //    if (graph->column_indices[edge] == 131070 || graph->column_indices[edge]
  //    == 131071) printf("edge %d: -> %d\n", edge,
  //    graph->column_indices[edge]);
  //}

  // util::cpu_mt::PrintCPUArray("row_offsets", graph->row_offsets,
  // graph->nodes+1); util::cpu_mt::PrintCPUArray("colunm_indices",
  // graph->column_indices, graph->edges);
  size_t *org_size = new size_t[num_gpus];
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    size_t dummy;
    if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
    cudaMemGetInfo(&(org_size[gpu]), &dummy);
  }

  Problem *problem = new Problem;  // allocate problem on GPU
  if (retval = util::GRError(
          problem->Init(stream_from_host, graph, NULL, num_gpus, gpu_idx,
                        partition_method, streams, max_queue_sizing,
                        max_in_sizing, partition_factor, partition_seed),
          "CC Problem Initialization Failed", __FILE__, __LINE__))
    return retval;

  Enactor *enactor = new Enactor(num_gpus, gpu_idx, instrument, debug,
                                 size_check);  // enactor map
  if (retval = util::GRError(
          enactor->Init(context, problem, traversal_mode, max_grid_size),
          "CC Enactor Init failed", __FILE__, __LINE__))
    return retval;

  enactor->communicate_latency = communicate_latency;
  enactor->communicate_multipy = communicate_multipy;
  enactor->expand_latency = expand_latency;
  enactor->subqueue_latency = subqueue_latency;
  enactor->fullqueue_latency = fullqueue_latency;
  enactor->makeout_latency = makeout_latency;

  if (retval = util::SetDevice(gpu_idx[0])) return retval;
  if (retval = util::latency::Test(
          streams[0], problem->data_slices[0]->latency_data,
          communicate_latency, communicate_multipy, expand_latency,
          subqueue_latency, fullqueue_latency, makeout_latency))
    return retval;

  cpu_timer.Stop();
  info->info["preprocess_time"] = cpu_timer.ElapsedMillis();

  // compute reference CPU CC
  if (!quick_mode) {
    if (!quiet_mode) {
      printf("Computing reference value ...\n");
    }
    ref_num_components = ReferenceCC(*graph, reference_check, quiet_mode);
    if (!quiet_mode) {
      printf("\n");
    }
  }

  // perform CC
  double total_elapsed = 0.0;
  double single_elapsed = 0.0;
  double max_elapsed = 0.0;
  double min_elapsed = 1e10;
  json_spirit::mArray process_times;
  if (!quiet_mode) printf("Using traversal mode %s\n", traversal_mode.c_str());
  for (SizeT iter = 0; iter < iterations; ++iter) {
    if (retval = util::GRError(
            problem->Reset(enactor->GetFrontierType(), max_queue_sizing),
            "CC Problem Data Reset Failed", __FILE__, __LINE__))
      return retval;
    if (retval = util::GRError(enactor->Reset(), "CC Enactor Reset failed",
                               __FILE__, __LINE__))
      return retval;

    if (!quiet_mode) {
      printf("_________________________\n");
      fflush(stdout);
    }
    cpu_timer.Start();
    if (retval = util::GRError(enactor->Enact(traversal_mode),
                               "CC Problem Enact Failed", __FILE__, __LINE__))
      return retval;
    cpu_timer.Stop();
    single_elapsed = cpu_timer.ElapsedMillis();
    total_elapsed += single_elapsed;
    process_times.push_back(single_elapsed);
    if (single_elapsed > max_elapsed) max_elapsed = single_elapsed;
    if (single_elapsed < min_elapsed) min_elapsed = single_elapsed;
    if (!quiet_mode) {
      printf(
          "-------------------------\n"
          "iteration %lld elapsed: %lf ms\n",
          (long long)iter, single_elapsed);
      fflush(stdout);
    }
  }
  total_elapsed /= iterations;
  info->info["process_times"] = process_times;
  info->info["min_process_time"] = min_elapsed;
  info->info["max_process_time"] = max_elapsed;

  cpu_timer.Start();
  // copy out results
  if (retval = util::GRError(problem->Extract(h_component_ids),
                             "CC Problem Data Extraction Failed", __FILE__,
                             __LINE__))
    return retval;

  // validity
  if (!quick_mode) {
    if (ref_num_components == problem->num_components) {
      if (!quiet_mode) {
        printf("CORRECT. Component Count: %lld\n",
               (long long)ref_num_components);
      }
    } else {
      if (!quiet_mode) {
        printf(
            "INCORRECT. Ref Component Count: %lld, "
            "GPU Computed Component Count: %lld\n",
            (long long)ref_num_components, (long long)problem->num_components);
      }
    }
  } else {
    if (!quiet_mode) {
      printf("Component Count: %lld\n", (long long)problem->num_components);
    }
  }
  if (!quick_mode) {
    ConvertIDs<VertexId, SizeT>(reference_check, graph->nodes,
                                ref_num_components);
    ConvertIDs<VertexId, SizeT>(h_component_ids, graph->nodes,
                                problem->num_components);
    if (!quiet_mode) {
      printf("Label Validity: ");
    }
    SizeT error_num = CompareResults(h_component_ids, reference_check,
                                     graph->nodes, true, quiet_mode);
    if (error_num > 0) {
      if (!quiet_mode) {
        printf("%lld errors occurred.\n", (long long)error_num);
      }
    } else {
      if (!quiet_mode) {
        printf("\n");
      }
    }
  }

  // if (ref_num_components == csr_problem->num_components)
  {
    // Compute size and root of each component
    VertexId *h_roots = new VertexId[problem->num_components];
    SizeT *h_histograms = new SizeT[problem->num_components];

    // printf("num_components = %d\n", problem->num_components);
    problem->ComputeCCHistogram(h_component_ids, h_roots, h_histograms);
    // printf("num_components = %d\n", problem->num_components);

    if (!quiet_mode) {
      // Display Solution
      DisplaySolution(h_component_ids, graph->nodes, problem->num_components,
                      h_roots, h_histograms);
    }

    if (h_roots) {
      delete[] h_roots;
      h_roots = NULL;
    }
    if (h_histograms) {
      delete[] h_histograms;
      h_histograms = NULL;
    }
  }

  info->ComputeCommonStats(  // compute running statistics
      enactor->enactor_stats.GetPointer(), total_elapsed, h_component_ids,
      true);

  if (!quiet_mode) {
    Display_Memory_Usage(num_gpus, gpu_idx, org_size, problem);
#ifdef ENABLE_PERFORMANCE_PROFILING
    Display_Performance_Profiling(enactor);
#endif
  }

  /*if (!quiet_mode)
  {
      printf("\n\tMemory Usage(B)\t");
      for (int gpu = 0; gpu < num_gpus; gpu++)
          if (num_gpus > 1)
          {
              if (gpu != 0) printf(" #keys%d\t #ins%d,0\t #ins%d,1", gpu, gpu,
  gpu); else printf(" $keys%d", gpu);
          }
          else printf(" #keys%d", gpu);
      if (num_gpus > 1) printf(" #keys%d", num_gpus);
      printf("\n");

      double max_key_sizing = 0, max_in_sizing_ = 0;
      for (int gpu = 0; gpu < num_gpus; gpu++)
      {
          size_t gpu_free, dummy;
          cudaSetDevice(gpu_idx[gpu]);
          cudaMemGetInfo(&gpu_free, &dummy);
          printf("GPU_%d\t %ld", gpu_idx[gpu], org_size[gpu] - gpu_free);
          for (int i = 0; i < num_gpus; i++)
          {
              SizeT x =
  problem->data_slices[gpu]->frontier_queues[i].keys[0].GetSize(); printf("\t
  %lld", (long long)x); double factor = 1.0 * x / (num_gpus > 1 ?
  problem->graph_slices[gpu]->in_counter[i] :
  problem->graph_slices[gpu]->nodes); if (factor > max_key_sizing)
  max_key_sizing = factor; if (num_gpus > 1 && i != 0 ) for (int t = 0; t < 2;
  t++)
                  {
                      x = problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                      printf("\t %lld", (long long)x);
                      factor = 1.0 * x /
  problem->graph_slices[gpu]->in_counter[i]; if (factor > max_in_sizing_)
  max_in_sizing_ = factor;
                  }
          }
          if (num_gpus > 1) printf("\t %lld", (long
  long)problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize());
          printf("\n");
      }
      printf("\t key_sizing =\t %lf", max_key_sizing);
      if (num_gpus > 1) printf("\t in_sizing =\t %lf", max_in_sizing_);
      printf("\n");
  }*/

  // Cleanup
  if (org_size) {
    delete[] org_size;
    org_size = NULL;
  }
  if (problem) {
    delete problem;
    problem = NULL;
  }
  if (enactor) {
    delete enactor;
    enactor = NULL;
  }
  if (reference_component_ids) {
    delete[] reference_component_ids;
    reference_component_ids = NULL;
  }
  if (h_component_ids) {
    delete[] h_component_ids;
    h_component_ids = NULL;
  }
  if (gpu_idx) {
    delete[] gpu_idx;
    gpu_idx = NULL;
  }
  cpu_timer.Stop();
  info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
  return retval;
}

/******************************************************************************
 * Main
 ******************************************************************************/
template <typename VertexId,  // use int as the vertex identifier
          typename SizeT,     // use int as the graph size type
          typename Value>     // use int as the value type
int main_(CommandLineArgs *args) {
  CpuTimer cpu_timer, cpu_timer2;
  cpu_timer.Start();
  Csr<VertexId, SizeT, Value> csr(false);  // graph we process on
  Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;

  // graph construction or generation related parameters
  info->info["undirected"] = true;  // require undirected input graph

  cpu_timer2.Start();
  info->Init("CC", *args, csr);  // initialize Info structure
  graphio::RemoveStandaloneNodes<VertexId, SizeT, Value>(
      &csr, args->CheckCmdLineFlag("quiet"));
  cpu_timer2.Stop();
  info->info["load_time"] = cpu_timer2.ElapsedMillis();

  RunTests<VertexId, SizeT, Value>(info);  // run test

  cpu_timer.Stop();
  info->info["total_time"] = cpu_timer.ElapsedMillis();

  if (!(info->info["quiet_mode"].get_bool())) {
    info->DisplayStats();  // display collected statistics
  }

  info->CollectInfo();  // collected all the info and put into JSON mObject
  if (info) {
    delete info;
    info = NULL;
  }
  return 0;
}

template <typename VertexId,  // the vertex identifier type, usually int or long
                              // long
          typename SizeT>     // the size tyep, usually int or long long
int main_Value(CommandLineArgs *args) {
  //    if (args -> CheckCmdLineFlag("64bit-Value"))
  //        return main_<VertexId, SizeT, long long>(args);
  //    else
  //        return main_<VertexId, SizeT, int      >(args);
  return main_<VertexId, SizeT, VertexId>(args);  // Value = VertexId for CC
}

template <typename VertexId>
int main_SizeT(CommandLineArgs *args) {
  // disabled to reduce compile time
  if (args->CheckCmdLineFlag("64bit-SizeT"))
    return main_Value<VertexId, long long>(args);
  else
    return main_Value<VertexId, int>(args);
}

int main_VertexId(CommandLineArgs *args) {
  // disabled, because oprtr::filter::KernelPolicy::SmemStorage is too large for
  // 64bit VertexId
  // if (args -> CheckCmdLineFlag("64bit-VertexId"))
  //    return main_SizeT<long long>(args);
  // else
  return main_SizeT<int>(args);
}

int main(int argc, char **argv) {
  CommandLineArgs args(argc, argv);
  int graph_args = argc - args.ParsedArgc() - 1;
  if (argc < 2 || graph_args < 1 || args.CheckCmdLineFlag("help")) {
    Usage();
    return 1;
  }

  return main_VertexId(&args);
}
// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
