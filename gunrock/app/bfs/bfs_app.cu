// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
* @file
* bfs_app.cu
*
* @brief Gunrock Breadth-first search implementation
*/

#include <stdio.h>
#include <gunrock/gunrock.h>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// BFS includes
#include <gunrock/app/bfs/bfs_enactor.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

// MGPU include
#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bfs;

/******************************************************************************
* Defines, constants, globals
******************************************************************************/
bool g_verbose;
bool g_undirected;
bool g_quick;
bool g_stream_from_host;


/******************************************************************************
* Housekeeping Routines
******************************************************************************/
void Usage()
{
  printf(
    "\ntest_bfs <graph type> <graph type args> [--device=<device_index>] "
    "[--undirected] [--src=<source index>] [--quick] "
    "[--mark-pred] [--queue-sizing=<scale factor>]\n"
    "[--v]\n"
    "\n"
    "Graph types and args:\n"
    " market [<file>]\n"
    " Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
    " edges from stdin (or from the optionally-specified file).\n"
    " --device=<device_index> Set GPU device for running the graph primitive.\n"
    " --undirected If set then treat the graph as undirected.\n"
    " and barrier duty (a relative indicator of load imbalance.)\n"
    " --src Begins BFS from the vertex <source index>. If set as randomize\n"
    " then will begin with a random source vertex.\n"
    " If set as largestdegree then will begin with the node which has\n"
    " largest degree.\n"
    " --quick If set will skip the CPU validation code.\n"
    " --mark-pred If set then keep not only label info but also predecessor info.\n"
    " --queue-sizing Allocates a frontier queue sized at (graph-edges * <scale factor>).\n"
    " Default is 1.0\n");
 }

/**
 * @brief Displays the BFS result (i.e., distance from source)
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] preds Predecessor node id for each node.
 * @param[in] nodes Number of nodes in the graph.
 * @param[in] MARK_PREDECESSORS Whether to show predecessor of each node.
*/
template<
  typename VertexId,
  typename SizeT>
void DisplaySolution(
  VertexId *source_path,
  VertexId *preds,
  SizeT nodes,
  bool MARK_PREDECESSORS,
  bool ENABLE_IDEMPOTENCE)
{
  fflush(stdout);
  // at most display first 40 results
  if (nodes > 40) nodes = 40;
  printf("\nFirst %d labels of the GPU result\n", nodes);
  printf("[");
  for (VertexId i = 0; i < nodes; ++i)
  {
    PrintValue(i);
    printf(":");
    PrintValue(source_path[i]);
    if (MARK_PREDECESSORS && !ENABLE_IDEMPOTENCE)
    {
      printf(",");
      PrintValue(preds[i]);
    }
    printf(" ");
  }
  printf("]\n");
}

/**
* Performance/Evaluation statistics
*/
struct Stats
{
  const char *name;
  Statistic rate;
  Statistic search_depth;
  Statistic redundant_work;
  Statistic duty;

  Stats() : name(NULL), rate(), search_depth(), redundant_work(), duty() {}
  Stats(const char *name) : name(name), rate(), search_depth(), redundant_work(), duty() {}
};

/**
* @brief Displays timing and correctness statistics
*
* @tparam MARK_PREDECESSORS
* @tparam VertexId
* @tparam Value
* @tparam SizeT
*
* @param[in] stats Reference to the Stats object defined in RunTests
* @param[in] src Source node where BFS starts
* @param[in] h_labels Host-side vector stores computed labels for validation
* @param[in] graph Reference to the CSR graph we process on
* @param[in] elapsed Total elapsed kernel running time
* @param[in] search_depth Maximum search depth of the BFS algorithm
* @param[in] total_queued Total element queued in BFS kernel running process
* @param[in] avg_duty Average duty of the BFS kernels
*/
template<
  bool     MARK_PREDECESSORS,
  typename VertexId,
  typename Value,
  typename SizeT>
void DisplayStats(
  Stats     &stats,
  VertexId  src,
  VertexId  *h_labels,
  const Csr<VertexId, Value, SizeT> &graph,
  double    elapsed,
  VertexId  search_depth,
  long long total_queued,
  double    avg_duty)
{
  // Compute nodes and edges visited
  SizeT edges_visited = 0;
  SizeT nodes_visited = 0;
  for (VertexId i = 0; i < graph.nodes; ++i)
  {
    if (h_labels[i] > -1)
    {
      ++nodes_visited;
      edges_visited += graph.row_offsets[i+1] - graph.row_offsets[i];
    }
  }

  double redundant_work = 0.0;
  if (total_queued > 0)
  {
    // measure duplicate edges put through queue
    redundant_work = ((double) total_queued - edges_visited) / edges_visited;
  }

  redundant_work *= 100;

  // Display test name
  printf("[%s] finished.", stats.name);

  // Display statistics
  if (nodes_visited < 5) printf("Fewer than 5 vertices visited.\n");
  else
  {
    // Display the specific sample statistics
    double m_teps = (double) edges_visited / (elapsed * 1000.0);
    printf("\nelapsed: %.3f ms, rate: %.3f MiEdges/s", elapsed, m_teps);
    if (search_depth != 0) printf(", search_depth: %lld", (long long) search_depth);
    if (avg_duty != 0)     printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
    printf("\nsource_node: %lld, nodes_visited: %lld, edges visited: %lld",
      (long long) src, (long long) nodes_visited, (long long) edges_visited);
    if (total_queued > 0)   printf(", total queued: %lld", total_queued);
    if (redundant_work > 0) printf(", redundant work: %.2f%%", redundant_work);
    printf("\n");
  }
}

/**
* @brief Run BFS tests
*
* @tparam VertexId
* @tparam Value
* @tparam SizeT
* @tparam MARK_PREDECESSORS
* @tparam ENABLE_IDEMPOTENCE
*
* @param[in] graph Reference to the CSR graph we process on
* @param[in] src Source node where BFS starts
* @param[in] max_grid_size Maximum CTA occupancy
* @param[in] num_gpus Number of GPUs
* @param[in] max_queue_sizing Scaling factor used in edge mapping
*
*/
template <
  typename VertexId,
  typename Value,
  typename SizeT,
  bool MARK_PREDECESSORS,
  bool ENABLE_IDEMPOTENCE>
void run_bfs(
  GunrockGraph *ggraph_out,
  const  Csr<VertexId, Value, SizeT> &ggraph_in,
  const  VertexId src,
  int    max_grid_size,
  int    num_gpus,
  double max_queue_sizing,
  CudaContext& context)
{
  // Preparations
  typedef BFSProblem<
    VertexId,
    SizeT,
    Value,
    MARK_PREDECESSORS,
    ENABLE_IDEMPOTENCE,
    (MARK_PREDECESSORS && ENABLE_IDEMPOTENCE)> Problem;

  // Allocate host-side label array for gpu-computed results
  VertexId *h_labels = (VertexId*)malloc(sizeof(VertexId) * ggraph_in.nodes);
  VertexId *h_preds = NULL;
  if (MARK_PREDECESSORS)
  {
    h_preds = (VertexId*)malloc(sizeof(VertexId) * ggraph_in.nodes);
  }

  // Allocate BFS enactor map
  BFSEnactor<false> bfs_enactor(g_verbose);

  // Allocate problem on GPU
  Problem *csr_problem = new Problem;
  util::GRError(csr_problem->Init(
    g_stream_from_host,
    ggraph_in,
    num_gpus),
    "Problem BFS Initialization Failed", __FILE__, __LINE__);

  Stats *stats = new Stats("GPU BFS");

  long long total_queued = 0;
  VertexId search_depth = 0;
  double avg_duty = 0.0;

  // Perform BFS
  GpuTimer gpu_timer;

  util::GRError(csr_problem->Reset(
    src, bfs_enactor.GetFrontierType(), max_queue_sizing),
    "BFS Problem Data Reset Failed", __FILE__, __LINE__);

  gpu_timer.Start();
  util::GRError(bfs_enactor.template Enact<Problem>(
    context, csr_problem, src, max_grid_size),
    "BFS Problem Enact Failed", __FILE__, __LINE__);
  gpu_timer.Stop();

  bfs_enactor.GetStatistics(total_queued, search_depth, avg_duty);

  float elapsed = gpu_timer.ElapsedMillis();

  // Copy out results back to Host
  util::GRError(csr_problem->Extract(h_labels, h_preds),
    "BFS Problem Data Extraction Failed", __FILE__, __LINE__);

  // label per node to GunrockGraph struct
  ggraph_out->node_values = (int*)&h_labels[0];

  /*
  // Display Solution
  DisplaySolution(
    h_labels, h_preds,
    ggraph_in.nodes,
    MARK_PREDECESSORS,
    ENABLE_IDEMPOTENCE);
  */

  // Display Stats
  DisplayStats<MARK_PREDECESSORS>(
    *stats,
    src,
    h_labels,
    ggraph_in,
    elapsed,
    search_depth,
    total_queued,
    avg_duty);

  // Clean up
  delete stats;
  if (csr_problem) delete csr_problem;
  if (h_preds)     free(h_preds);

  cudaDeviceSynchronize();
}

/**
* @brief RunTests entry
*
* @tparam VertexId
* @tparam Value
* @tparam SizeT
*
* @param[out] ggraph_out output GunrockGraph type struct
* @param[in]  ggraph_in Reference to the CSR graph we process on
* @param[in]  args Reference to the command line arguments
*/
template <
  typename VertexId,
  typename Value,
  typename SizeT>
void dispatch_bfs(
  GunrockGraph  *ggraph_out,
  const Csr<VertexId, Value, SizeT> &csr_graph,
  GunrockConfig bfs_config,
  CudaContext&  context)
{
  // default configurations
  VertexId source     = 0;       //!< default source vertex to start
  int   num_gpus      = 1;       //!< number of GPUs for multi-gpu enactor to use
  int   max_grid_size = 0;       //!< maximum grid size (0: leave it up to the enactor)
  bool  mark_pred     = false;   //!< whether to mark predecessor or not
  bool  idempotence   = false;   //!< whether or not to enable idempotence
  float max_queue_sizing = 1.0f; //!< maximum size scaling factor for work queues

  // check wether need to be reconfig
  source           = bfs_config.source;
  mark_pred        = bfs_config.mark_pred;
  idempotence      = bfs_config.idempotence;
  max_queue_sizing = bfs_config.queue_size;

  /*
  std::string src_str;
  args.GetCmdLineArgument("src", src_str);
  if (src_str.empty()) {
      src = 0;
  } else if (src_str.compare("randomize") == 0) {
      src = graphio::RandomNode(csr_graph.nodes);
  } else if (src_str.compare("largestdegree") == 0) {
      src = csr_graph.GetNodeWithHighestDegree();
  } else {
      args.GetCmdLineArgument("src", src);
  }
  */

  //printf("Display neighbor list of source:\n");
  //csr_graph.DisplayNeighborList(source);

  if (mark_pred)
  {
    if (idempotence)
    {
      run_bfs<VertexId, Value, SizeT, true, true>(
        ggraph_out,
        csr_graph,
        source,
        max_grid_size,
        num_gpus,
        max_queue_sizing,
        context);
    }
    else
    {
      run_bfs<VertexId, Value, SizeT, true, false>(
        ggraph_out,
        csr_graph,
        source,
        max_grid_size,
        num_gpus,
        max_queue_sizing,
        context);
    }
  }
  else
  {
    if (idempotence)
    {
      run_bfs<VertexId, Value, SizeT, false, true>(
        ggraph_out,
        csr_graph,
        source,
        max_grid_size,
        num_gpus,
        max_queue_sizing,
        context);
    }
    else
    {
      run_bfs<VertexId, Value, SizeT, false, false>(
        ggraph_out,
        csr_graph,
        source,
        max_grid_size,
        num_gpus,
        max_queue_sizing,
        context);
    }
  }
}

/*
* @brief gunrock_bfs function
*
* @param[out] output subgraph of bfs problem
* @param[in] input graph need to process on
* @param[in] gunrock datatype struct
*/
void gunrock_bfs(
  GunrockGraph       *ggraph_out,
  const GunrockGraph *ggraph_in,
  GunrockConfig      bfs_config,
  GunrockDataType    data_type)
{
  // moderngpu preparations
  int device = 0;
  device = bfs_config.device;
  ContextPtr context = mgpu::CreateCudaDevice(device);

  // build input csr format graph
  Csr<int, int, int> csr_graph(false);
  csr_graph.nodes = ggraph_in->num_nodes;
  csr_graph.edges = ggraph_in->num_edges;
  csr_graph.row_offsets    = (int*)ggraph_in->row_offsets;
  csr_graph.column_indices = (int*)ggraph_in->col_indices;

  // lunch bfs dispatch function
  dispatch_bfs<int, int, int>(ggraph_out, csr_graph, bfs_config, *context);

  // reset for free memory
  csr_graph.row_offsets    = NULL;
  csr_graph.column_indices = NULL;
  csr_graph.row_offsets    = NULL;
  csr_graph.column_indices = NULL;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End: