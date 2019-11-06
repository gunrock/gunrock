// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bfs.cu
 *
 * @brief Simple test driver program for breadth-first search.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <algorithm>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;

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
      "[--undirected]            Treat the graph as undirected (symmetric).\n"
      "[--idempotence]           Whether or not to enable idempotent "
      "operation.\n"
      "[--instrumented]          Keep kernels statics [Default: Disable].\n"
      "                          total_queued, search_depth and barrier duty.\n"
      "                          (a relative indicator of load imbalance.)\n"
      "[--src=<Vertex-ID|randomize|largestdegree>]\n"
      "                          Begins traversal from the source (Default: "
      "0).\n"
      "                          If randomize: from a random source vertex.\n"
      "                          If largestdegree: from largest degree "
      "vertex.\n"
      "[--quick]                 Skip the CPU reference validation process.\n"
      "[--mark-pred]             Keep both label info and predecessor info.\n"
      "[--disable-size-check]    Disable frontier queue size check.\n"
      "[--grid-size=<grid size>] Maximum allowed grid size setting.\n"
      "[--queue-sizing=<factor>] Allocates a frontier queue sized at: \n"
      "                          (graph-edges * <factor>). (Default: 1.0)\n"
      "[--in-sizing=<in/out_queue_scale_factor>]\n"
      "                          Allocates a frontier queue sized at: \n"
      "                          (graph-edges * <factor>). (Default: 1.0)\n"
      "[--v]                     Print verbose per iteration debug info.\n"
      "[--iteration-num=<num>]   Number of runs to perform the test.\n"
      "[--traversal-mode=<0|1>]  Set traversal strategy, 0 for Load-Balanced\n"
      "                          1 for Dynamic-Cooperative (Default: dynamic\n"
      "                          determine based on average degree).\n"
      "[--partition_method=<random|biasrandom|clustered|metis>]\n"
      "                          Choose partitioner (Default use random).\n"
      "[--quiet]                 No output (unless --json is specified).\n"
      "[--json]                  Output JSON-format statistics to STDOUT.\n"
      "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
      "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
      "                          where name is auto-generated.\n");
}

/**
 * @brief Displays the BFS result (i.e., distance from source)
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
 *
 * @param[in] labels    Search depth from the source for each node.
 * @param[in] preds     Predecessor node id for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 * @param[in] quiet     Don't print out anything to stdout
 */
template <typename VertexId, typename SizeT, bool MARK_PREDECESSORS,
          bool ENABLE_IDEMPOTENCE>
void DisplaySolution(VertexId *labels, VertexId *preds, SizeT num_nodes,
                     bool quiet = false) {
  if (quiet) {
    return;
  }
  // careful: if later code in this
  // function changes something, this
  // return is the wrong thing to do

  if (num_nodes > 40) {
    num_nodes = 40;
  }

  printf("\nFirst %d labels of the GPU result:\n", num_nodes);

  printf("[");
  for (VertexId i = 0; i < num_nodes; ++i) {
    PrintValue(i);
    printf(":");
    PrintValue(labels[i]);
    if (MARK_PREDECESSORS && !ENABLE_IDEMPOTENCE) {
      printf(",");
      PrintValue(preds[i]);
    }
    printf(" ");
  }
  printf("]\n");
}

/******************************************************************************
 * BFS Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference BFS ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] source_path Host-side vector to store CPU computed labels for each
 * node
 * @param[in] predecessor Host-side vector to store CPU computed predecessor for
 * each node
 * @param[in] src Source node where BFS starts
 * @param[in] quiet Don't print out anything to stdout
 */
template <typename VertexId, typename Value, typename SizeT,
          bool MARK_PREDECESSORS, bool ENABLE_IDEMPOTENCE>
void ReferenceBFS(const Csr<VertexId, Value, SizeT> *graph,
                  VertexId *source_path, VertexId *predecessor, VertexId src,
                  bool quiet = false) {
  // Initialize labels
  for (VertexId i = 0; i < graph->nodes; ++i) {
    source_path[i] = ENABLE_IDEMPOTENCE ? -1 : util::MaxValue<VertexId>() - 1;
    if (MARK_PREDECESSORS) {
      predecessor[i] = -1;
    }
  }
  source_path[src] = 0;
  VertexId search_depth = 0;

  // Initialize queue for managing previously-discovered nodes
  std::deque<VertexId> frontier;
  frontier.push_back(src);

  // Perform BFS
  CpuTimer cpu_timer;
  cpu_timer.Start();
  while (!frontier.empty()) {
    // Dequeue node from frontier
    VertexId dequeued_node = frontier.front();
    frontier.pop_front();
    VertexId neighbor_dist = source_path[dequeued_node] + 1;

    // Locate adjacency list
    SizeT edges_begin = graph->row_offsets[dequeued_node];
    SizeT edges_end = graph->row_offsets[dequeued_node + 1];

    for (SizeT edge = edges_begin; edge < edges_end; ++edge) {
      // Lookup neighbor and enqueue if undiscovered
      VertexId neighbor = graph->column_indices[edge];
      if (source_path[neighbor] > neighbor_dist ||
          source_path[neighbor] == -1) {
        source_path[neighbor] = neighbor_dist;
        if (MARK_PREDECESSORS) {
          predecessor[neighbor] = dequeued_node;
        }
        if (search_depth < neighbor_dist) {
          search_depth = neighbor_dist;
        }
        frontier.push_back(neighbor);
      }
    }
  }

  if (MARK_PREDECESSORS) {
    predecessor[src] = -1;
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  search_depth++;

  if (!quiet) {
    printf("CPU BFS finished in %lf msec. cpu_search_depth: %d\n", elapsed,
           search_depth);
  }
}

/**
 * @brief Run BFS tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 * @tparam MARK_PREDECESSORS
 * @tparam ENABLE_IDEMPOTENCE
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <typename VertexId, typename Value, typename SizeT, bool INSTRUMENT,
          bool DEBUG, bool SIZE_CHECK, bool MARK_PREDECESSORS,
          bool ENABLE_IDEMPOTENCE>
void RunTests(Info<VertexId, Value, SizeT> *info) {
  typedef BFSProblem<VertexId, SizeT, Value, MARK_PREDECESSORS,
                     ENABLE_IDEMPOTENCE,
                     (MARK_PREDECESSORS && ENABLE_IDEMPOTENCE)>
      BfsProblem;  // does not use double buffer

  typedef BFSEnactor<BfsProblem, INSTRUMENT, DEBUG, SIZE_CHECK> BfsEnactor;

  // parse configurations from mObject info
  Csr<VertexId, Value, SizeT> *graph = info->csr_ptr;
  VertexId src = info->info["source_vertex"].get_int64();
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
  int traversal_mode = info->info["traversal_mode"].get_int();
  int iterations = 1;  // disable since doesn't support mgpu stop condition.
                       // info->info["num_iteration"].get_int();

  json_spirit::mArray device_list = info->info["device_list"].get_array();
  int *gpu_idx = new int[num_gpus];
  for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

  // TODO: remove after merge mgpu-cq
  ContextPtr *context = (ContextPtr *)info->context;
  cudaStream_t *streams = (cudaStream_t *)info->streams;

  // allocate host-side label array (for both reference and GPU results)
  VertexId *reference_labels = new VertexId[graph->nodes];
  VertexId *reference_preds = new VertexId[graph->nodes];
  VertexId *h_labels = new VertexId[graph->nodes];
  VertexId *reference_check_label = (quick_mode) ? NULL : reference_labels;
  VertexId *reference_check_preds = NULL;
  VertexId *h_preds = NULL;

  if (MARK_PREDECESSORS) {
    h_preds = new VertexId[graph->nodes];
    if (!quick_mode) {
      reference_check_preds = reference_preds;
    }
  }

  size_t *org_size = new size_t[num_gpus];
  for (int gpu = 0; gpu < num_gpus; gpu++) {
    size_t dummy;
    cudaSetDevice(gpu_idx[gpu]);
    cudaMemGetInfo(&(org_size[gpu]), &dummy);
  }

  BfsEnactor *enactor = new BfsEnactor(num_gpus, gpu_idx);  // enactor map
  BfsProblem *problem = new BfsProblem;  // allocate problem on GPU

  util::GRError(problem->Init(stream_from_host, graph, NULL, num_gpus, gpu_idx,
                              partition_method, streams, max_queue_sizing,
                              max_in_sizing, partition_factor, partition_seed),
                "BFS Problem Init failed", __FILE__, __LINE__);

  util::GRError(enactor->Init(context, problem, max_grid_size, traversal_mode),
                "BFS Enactor Init failed", __FILE__, __LINE__);

  // compute reference CPU BFS solution for source-distance
  if (reference_check_label != NULL) {
    if (!quiet_mode) {
      printf("Computing reference value ...\n");
    }
    ReferenceBFS<VertexId, Value, SizeT, MARK_PREDECESSORS, ENABLE_IDEMPOTENCE>(
        graph, reference_check_label, reference_check_preds, src, quiet_mode);
    if (!quiet_mode) {
      printf("\n");
    }
  }

  // perform BFS
  double elapsed = 0.0f;
  CpuTimer cpu_timer;

  for (int iter = 0; iter < iterations; ++iter) {
    util::GRError(problem->Reset(src, enactor->GetFrontierType(),
                                 max_queue_sizing, max_queue_sizing1),
                  "BFS Problem Data Reset Failed", __FILE__, __LINE__);

    util::GRError(enactor->Reset(), "BFS Enactor Reset failed", __FILE__,
                  __LINE__);

    util::GRError("Error before Enact", __FILE__, __LINE__);

    if (!quiet_mode) {
      printf("__________________________\n");
      fflush(stdout);
    }

    cpu_timer.Start();
    util::GRError(enactor->Enact(src, traversal_mode),
                  "BFS Problem Enact Failed", __FILE__, __LINE__);
    cpu_timer.Stop();

    if (!quiet_mode) {
      printf("--------------------------\n");
      fflush(stdout);
    }

    elapsed += cpu_timer.ElapsedMillis();
  }

  elapsed /= iterations;

  // copy out results
  util::GRError(problem->Extract(h_labels, h_preds),
                "BFS Problem Data Extraction Failed", __FILE__, __LINE__);

  // verify the result
  if (reference_check_label != NULL) {
    if (!ENABLE_IDEMPOTENCE) {
      if (!quiet_mode) {
        printf("Label Validity: ");
      }
      int error_num = CompareResults(h_labels, reference_check_label,
                                     graph->nodes, true, quiet_mode);
      if (error_num > 0) {
        if (!quiet_mode) {
          printf("%d errors occurred.\n", error_num);
        }
      }
    } else {
      if (!MARK_PREDECESSORS) {
        if (!quiet_mode) {
          printf("Label Validity: ");
        }
        int error_num = CompareResults(h_labels, reference_check_label,
                                       graph->nodes, true, quiet_mode);
        if (error_num > 0) {
          if (!quiet_mode) {
            printf("%d errors occurred.\n", error_num);
          }
        }
      }
    }
  }

  // display Solution
  if (!quiet_mode) {
    DisplaySolution<VertexId, SizeT, MARK_PREDECESSORS, ENABLE_IDEMPOTENCE>(
        h_labels, h_preds, graph->nodes, quiet_mode);
  }

  info->ComputeTraversalStats(  // compute running statistics
      enactor->enactor_stats.GetPointer(), elapsed, h_labels);

  if (!quiet_mode) {
    info->DisplayStats();  // display collected statistics
  }

  info->CollectInfo();  // collected all the info and put into JSON mObject

  if (!quiet_mode) {
    printf("\n\tMemory Usage(B)\t");
    for (int gpu = 0; gpu < num_gpus; gpu++)
      if (num_gpus > 1) {
        if (gpu != 0) {
          printf(" #keys%d,0\t #keys%d,1\t #ins%d,0\t #ins%d,1", gpu, gpu, gpu,
                 gpu);
        } else {
          printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
        }
      } else {
        printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
      }
    if (num_gpus > 1) {
      printf(" #keys%d", num_gpus);
    }
    printf("\n");
    double max_queue_sizing_[2] = {0, 0}, max_in_sizing_ = 0;
    for (int gpu = 0; gpu < num_gpus; gpu++) {
      size_t gpu_free, dummy;
      cudaSetDevice(gpu_idx[gpu]);
      cudaMemGetInfo(&gpu_free, &dummy);
      printf("GPU_%d\t %ld", gpu_idx[gpu], org_size[gpu] - gpu_free);
      for (int i = 0; i < num_gpus; i++) {
        for (int j = 0; j < 2; j++) {
          SizeT x =
              problem->data_slices[gpu]->frontier_queues[i].keys[j].GetSize();
          printf("\t %lld", (long long)x);
          double factor =
              1.0 * x /
              (num_gpus > 1 ? problem->graph_slices[gpu]->in_counter[i]
                            : problem->graph_slices[gpu]->nodes);
          if (factor > max_queue_sizing_[j]) {
            max_queue_sizing_[j] = factor;
          }
        }
        if (num_gpus > 1 && i != 0) {
          for (int t = 0; t < 2; t++) {
            SizeT x = problem->data_slices[gpu][0].keys_in[t][i].GetSize();
            printf("\t %lld", (long long)x);
            double factor = 1.0 * x / problem->graph_slices[gpu]->in_counter[i];
            if (factor > max_in_sizing_) {
              max_in_sizing_ = factor;
            }
          }
        }
      }
      if (num_gpus > 1) {
        printf("\t %lld", (long long)(problem->data_slices[gpu]
                                          ->frontier_queues[num_gpus]
                                          .keys[0]
                                          .GetSize()));
      }
      printf("\n");
    }
    printf("\t queue_sizing =\t %lf \t %lf", max_queue_sizing_[0],
           max_queue_sizing_[1]);
    if (num_gpus > 1) {
      printf("\t in_sizing =\t %lf", max_in_sizing_);
    }
    printf("\n");
  }

  // Clean up
  if (org_size) {
    delete[] org_size;
    org_size = NULL;
  }
  if (enactor) {
    delete enactor;
    enactor = NULL;
  }
  if (problem) {
    delete problem;
    problem = NULL;
  }
  if (reference_labels) {
    delete[] reference_labels;
    reference_labels = NULL;
  }
  if (reference_preds) {
    delete[] reference_preds;
    reference_preds = NULL;
  }
  if (h_labels) {
    delete[] h_labels;
    h_labels = NULL;
  }
  if (h_preds) {
    delete[] h_preds;
    h_preds = NULL;
  }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <typename VertexId, typename Value, typename SizeT, bool INSTRUMENT,
          bool DEBUG, bool SIZE_CHECK, bool MARK_PREDECESSORS>
void RunTests_enable_idempotence(Info<VertexId, Value, SizeT> *info) {
  if (info->info["idempotent"].get_bool()) {
    RunTests<VertexId, Value, SizeT, INSTRUMENT, DEBUG, SIZE_CHECK,
             MARK_PREDECESSORS, true>(info);
  } else {
    RunTests<VertexId, Value, SizeT, INSTRUMENT, DEBUG, SIZE_CHECK,
             MARK_PREDECESSORS, false>(info);
  }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <typename VertexId, typename Value, typename SizeT, bool INSTRUMENT,
          bool DEBUG, bool SIZE_CHECK>
void RunTests_mark_predecessors(Info<VertexId, Value, SizeT> *info) {
  if (info->info["mark_predecessors"].get_bool()) {
    RunTests_enable_idempotence<VertexId, Value, SizeT, INSTRUMENT, DEBUG,
                                SIZE_CHECK, true>(info);
  } else {
    RunTests_enable_idempotence<VertexId, Value, SizeT, INSTRUMENT, DEBUG,
                                SIZE_CHECK, false>(info);
  }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 * @tparam DEBUG
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <typename VertexId, typename Value, typename SizeT, bool INSTRUMENT,
          bool DEBUG>
void RunTests_size_check(Info<VertexId, Value, SizeT> *info) {
  if (info->info["size_check"].get_bool()) {
    RunTests_mark_predecessors<VertexId, Value, SizeT, INSTRUMENT, DEBUG, true>(
        info);
  } else {
    RunTests_mark_predecessors<VertexId, Value, SizeT, INSTRUMENT, DEBUG,
                               false>(info);
  }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <typename VertexId, typename Value, typename SizeT, bool INSTRUMENT>
void RunTests_debug(Info<VertexId, Value, SizeT> *info) {
  if (info->info["debug_mode"].get_bool()) {
    RunTests_size_check<VertexId, Value, SizeT, INSTRUMENT, true>(info);
  } else {
    RunTests_size_check<VertexId, Value, SizeT, INSTRUMENT, false>(info);
  }
}

/**
 * @brief RunTests entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <typename VertexId, typename Value, typename SizeT>
void RunTests_instrumented(Info<VertexId, Value, SizeT> *info) {
  if (info->info["instrument"].get_bool()) {
    RunTests_debug<VertexId, Value, SizeT, true>(info);
  } else {
    RunTests_debug<VertexId, Value, SizeT, false>(info);
  }
}

template <typename VertexId, typename Value, typename SizeT>
void Write_gr(Info<VertexId, Value, SizeT> *info, CommandLineArgs *args) {
  Csr<VertexId, Value, SizeT> *graph = info->csr_ptr;
  std::string filename = "";
  std::ofstream gr_file;
  SizeT edge_counter = 0;
  bool keep_num = args->CheckCmdLineFlag("keep-num");

  args->GetCmdLineArgument("output-filename", filename);
  gr_file.open(filename.c_str());
  if (args->CheckCmdLineFlag("include-header"))
    gr_file << graph->nodes << " " << graph->nodes << " " << graph->edges
            << std::endl;
  for (VertexId u = 0; u < graph->nodes; u++) {
    for (SizeT i = graph->row_offsets[u]; i < graph->row_offsets[u + 1]; i++) {
      VertexId v = graph->column_indices[i];
      if (keep_num)
        gr_file << u + 1 << " " << v + 1;
      else
        gr_file << u << " " << v;
      if (graph->edge_values != NULL) gr_file << " " << graph->edge_values[i];
      gr_file << std::endl;
      edge_counter++;
    }
  }
  gr_file.close();
  printf("%lld nodes and %lld edges written into gr file %s\n",
         (long long)graph->nodes, (long long)edge_counter, filename.c_str());
}

/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char **argv) {
  CommandLineArgs args(argc, argv);
  int graph_args = argc - args.ParsedArgc() - 1;
  if (argc < 2 || graph_args < 1 || args.CheckCmdLineFlag("help")) {
    Usage();
    return 1;
  }

  typedef int VertexId;  // Use int as the vertex identifier
  typedef int Value;     // Use int as the value type
  typedef int SizeT;     // Use int as the graph size type

  Csr<VertexId, Value, SizeT> csr(false);  // graph we process on
  Info<VertexId, Value, SizeT> *info = new Info<VertexId, Value, SizeT>;

  // graph construction or generation related parameters
  info->info["undirected"] = args.CheckCmdLineFlag("undirected");
  info->info["edge_value"] = args.CheckCmdLineFlag("edge_value");

  info->Init("BFS", args, csr);  // initialize Info structure
  // RunTests_instrumented<VertexId, Value, SizeT>(info);  // run test
  Write_gr<VertexId, Value, SizeT>(info, &args);
  return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
