// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_topk.cu
 *
 * @brief Simple test driver program for computing Topk.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <map>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// Degree Centrality includes
#include <gunrock/app/topk/topk_enactor.cuh>
#include <gunrock/app/topk/topk_problem.cuh>

// Operator includes
#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::topk;

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
  printf(
    "\ntest_topk <graph type> <graph type args> [--top=<K_value>] [--device=<device_index>] "
    "[--instrumented] [--quick] "
    "[--quiet --json --jsonfile=<name> --jsondir=<dir> "
    "[--v]\n"
    "\n"
    "Graph types and args:\n"
    "  market [<file>]\n"
    "    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
    "    edges from stdin (or from the optionally-specified file).\n"
    "    k value top K value.\n"
    "  --device=<device_index>  Set GPU device for running the graph primitive.\n"
    "  --instrumented If set then kernels keep track of queue-search_depth\n"
    "  and barrier duty (a relative indicator of load imbalance.)\n"
    "  --quick If set will skip the CPU validation code.\n"
    " --quiet                  No output (unless --json is specified).\n"
    " --json                   Output JSON-format statistics to stdout.\n"
    " --jsonfile=<name>        Output JSON-format statistics to file <name>\n"
    " --jsondir=<dir>          Output JSON-format statistics to <dir>/name,\n");
}

/**
 * @brief displays the top K results
 *
 */
template<
  typename VertexId,
  typename Value,
  typename SizeT>
void DisplaySolution(
  VertexId *h_node_id,
  Value    *h_degrees_i,
  Value    *h_degrees_o,
  SizeT    num_nodes)
{
  fflush(stdout);
  // at most display the first 100 results
  if (num_nodes > 100) num_nodes = 100;
  printf("==> top %d centrality nodes:\n", num_nodes);
  for (SizeT iter = 0; iter < num_nodes; ++iter)
    printf("%d %d %d\n", h_node_id[iter], h_degrees_i[iter], h_degrees_o[iter]);
}

/******************************************************************************
 * Degree Centrality Testing Routines
 *****************************************************************************/
/**
 * @brief A simple CPU-based reference TOPK implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 */
struct compare_second_only
{
  template <typename T1, typename T2>
  bool operator()(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2)
  {
    return p1.second > p2. second;
  }
};

template<
  typename VertexId,
  typename Value,
  typename SizeT>
void SimpleReferenceTopK(
  Csr<VertexId, Value, SizeT> &csr,
  Csr<VertexId, Value, SizeT> &csc,
  VertexId *ref_node_id,
  Value    *ref_degrees,
  SizeT    top_nodes)
{
  printf("CPU reference test.\n");
  CpuTimer cpu_timer;

  // malloc degree centrality spaces
  Value *ref_degrees_original =
    (Value*)malloc(sizeof(Value) * csr.nodes);
  Value *ref_degrees_reversed =
    (Value*)malloc(sizeof(Value) * csc.nodes);

  // store reference output results
  std::vector< std::pair<int, int> > results;

  // calculations
  for (SizeT node = 0; node < csr.nodes; ++node)
  {
    ref_degrees_original[node] =
      csr.row_offsets[node+1] - csr.row_offsets[node];
    ref_degrees_reversed[node] =
      csc.row_offsets[node+1] - csc.row_offsets[node];
  }

  cpu_timer.Start();

  // add ingoing degrees and outgoing degrees together
  for (SizeT node = 0; node < csr.nodes; ++node)
  {
    ref_degrees_original[node] =
      ref_degrees_original[node] + ref_degrees_reversed[node];
    results.push_back( std::make_pair (node, ref_degrees_original[node]) );
  }

  // pair sort according to second elements - degree centrality
  std::stable_sort(results.begin(), results.end(), compare_second_only());

  for (SizeT itr = 0; itr < top_nodes; ++itr)
  {
    ref_node_id[itr] = results[itr].first;
    ref_degrees[itr] = results[itr].second;
  }

  cpu_timer.Stop();
  float elapsed_cpu = cpu_timer.ElapsedMillis();
  printf("==> CPU Degree Centrality finished in %lf msec.\n", elapsed_cpu);

  // clean up if neccessary
  if (ref_degrees_original) { free(ref_degrees_original); }
  if (ref_degrees_reversed) { free(ref_degrees_reversed); }
  results.clear();
}

/**
 * @brief Run TopK tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] csr Reference to the CSR graph we process on
 * @param[in] csc Reference to the inversed CSR graph we process on
 * @param[in] args Reference to the command line arguments
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] top_nodes Number of nodes to process for Top-K algorithm
 * @param[in] context CudaContext for moderngpu library
 *
 */
template <
  typename VertexId,
  typename Value,
  typename SizeT,
  bool INSTRUMENT,
  bool DEBUG,
  bool SIZE_CHECK>
void RunTests(Info<VertexId, Value, SizeT> *info)
{

  // define the problem data structure for graph primitive
  typedef TOPKProblem<
    VertexId,
    SizeT,
    Value> Problem;

    Csr<VertexId, Value, SizeT>
                 *csr        = info->csr_ptr;
    Csr<VertexId, Value, SizeT>
                 *csc        = info->csc_ptr;
    int           max_grid_size         = info->info["max_grid_size"].get_int();
    int           num_gpus              = info->info["num_gpus"].get_int();
    bool          stream_from_host      = info->info["stream_from_host"].get_bool();
    SizeT         top_nodes             = info->info["top_nodes"].get_int();
    bool          quiet_mode            = info->info["quiet_mode"].get_bool();

    ContextPtr    *context              = (ContextPtr*)info->context;

    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

  // INSTRUMENT specifies whether we want to keep such statistical data
  // Allocate TOPK enactor map
  TOPKEnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK> topk_enactor(gpu_idx);

  // allocate problem on GPU
  // create a pointer of the TOPKProblem type
  Problem *topk_problem = new Problem;

  // reset top_nodes if input k > total number of nodes
  if (top_nodes > csr->nodes) top_nodes = csr->nodes;

  // malloc host memory
  VertexId *h_node_id   = (VertexId*)malloc(sizeof(VertexId) * top_nodes);
  VertexId *ref_node_id = (VertexId*)malloc(sizeof(VertexId) * top_nodes);
  Value    *h_degrees_i = (Value*)malloc(sizeof(Value) * top_nodes);
  Value    *h_degrees_o = (Value*)malloc(sizeof(Value) * top_nodes);
  Value    *ref_degrees = (Value*)malloc(sizeof(Value) * top_nodes);

  // copy data from CPU to GPU
  // initialize data members in DataSlice for graph
  util::GRError(topk_problem->Init(
    stream_from_host,
    *csr,
    *csc,
    num_gpus),
    "Problem TOPK Initialization Failed", __FILE__, __LINE__);

  // perform topk degree centrality calculations
  GpuTimer gpu_timer; // Record the kernel running time

  // reset values in DataSlice for graph
  util::GRError(topk_problem->Reset(topk_enactor.GetFrontierType()),
                "TOPK Problem Data Reset Failed", __FILE__, __LINE__);

  gpu_timer.Start();
  // launch topk enactor
  util::GRError(topk_enactor.template Enact<Problem>(*context,
                                                   topk_problem,
                                                   top_nodes,
                                                   max_grid_size),
                "TOPK Problem Enact Failed", __FILE__, __LINE__);
  gpu_timer.Stop();

  float elapsed_gpu = gpu_timer.ElapsedMillis();
  printf("==> GPU TopK Degree Centrality finished in %lf msec.\n", elapsed_gpu);

  // copy out results back to CPU from GPU using Extract
  util::GRError(topk_problem->Extract(
    h_node_id,
    h_degrees_i,
    h_degrees_o,
    top_nodes),
    "TOPK Problem Data Extraction Failed",
    __FILE__, __LINE__);

  // display solution
  if (!quiet_mode)
  DisplaySolution(
    h_node_id,
    h_degrees_i,
    h_degrees_o,
    top_nodes);

  info->ComputeCommonStats(topk_enactor.enactor_stats.GetPointer(), elapsed_gpu);

  if (!quiet_mode)
    info->DisplayStats();

  // validation
  SimpleReferenceTopK(
    *csr,
    *csc,
    ref_node_id,
    ref_degrees,
    top_nodes);

  int error_num = CompareResults(h_node_id, ref_node_id, top_nodes, true);
  if (error_num > 0)
  {
    if (!quiet_mode) printf("INCOREECT! %d error(s) occured. \n", error_num);
  }
  if (!quiet_mode) printf("\n");

  info->CollectInfo();

  // cleanup if neccessary
  if (topk_problem) { delete topk_problem; }
  if (h_node_id)    { free(h_node_id);   }
  if (h_degrees_i)  { free(h_degrees_i); }
  if (h_degrees_o)  { free(h_degrees_o); }

  cudaDeviceSynchronize();
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG>
void RunTests_size_check(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["size_check"].get_bool()) RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        true > (info);
   else RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        false> (info);
}

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT>
void RunTests_debug(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["debug_mode"].get_bool()) RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        true > (info);
    else RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        false> (info);
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT>
void RunTests_instrumented(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["instrument"].get_bool()) RunTests_debug
        <VertexId, Value, SizeT,
        true > (info);
    else RunTests_debug
        <VertexId, Value, SizeT,
        false> (info);
}

/******************************************************************************
 * Main
 ******************************************************************************/
int main(int argc, char** argv)
{
  CommandLineArgs args(argc, argv);

  int graph_args = argc - args.ParsedArgc() - 1;
  if ((argc < 2) || (args.CheckCmdLineFlag("help")))
  {
    Usage();
    return 1;
  }

  typedef int VertexId;
  typedef int Value;
  typedef int SizeT;
  Csr<VertexId, Value, SizeT> csr(false);
  Csr<VertexId, Value, SizeT> csc(false);
  Info<VertexId, Value, SizeT> *info = new Info<VertexId, Value, SizeT>;

  info->info["undirected"] = args.CheckCmdLineFlag("undirected");
  info->Init("TOPK", args, csr, csc);

  RunTests_instrumented<VertexId, Value, SizeT>(info);

  return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
