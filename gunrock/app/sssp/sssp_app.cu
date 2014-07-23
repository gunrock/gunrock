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
#include <gunrock/gunrock.h>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// SSSP includes
#include <gunrock/app/sssp/sssp_enactor.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/app/sssp/sssp_functor.cuh>

#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sssp;

/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/
static bool g_verbose;
//bool g_undirected;
//bool g_quick;
static bool g_stream_from_host;

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
  printf("\ntest_sssp <graph type> <graph type args> [--device=<device_index>] "
        "[--undirected] [--instrumented] [--source=<source index>] [--quick]\n"
        "[--v] [mark-pred] [--queue-sizing=<scale factor>]\n"
        "\n"
        "Graph types and args:\n"
        "  market [<file>]\n"
        "    Reads a Matrix-Market coordinate-formatted graph of directed/undirected\n"
        "    edges from stdin (or from the optionally-specified file).\n"
        "  --device=<device_index>  Set GPU device for running the graph primitive.\n"
        "  --undirected If set then treat the graph as undirected.\n"
        "  --instrumented If set then kernels keep track of queue-search_depth\n"
        "  and barrier duty (a relative indicator of load imbalance.)\n"
        "  --source Begins SSSP from the vertex <source index>. If set as randomize\n"
        "  then will begin with a random source vertex.\n"
        "  If set as largestdegree then will begin with the node which has\n"
        "  largest degree.\n"
        "  --quick If set will skip the CPU validation code.\n"
        "  --v Whether to show debug info.\n"
        "  --mark-pred If set then keep not only label info but also predecessor info.\n"
        "  --queue-sizing Allocates a frontier queue sized at (graph-edges * <scale factor>).\n"
	 );
}

/**
 * @brief Displays the SSSP result (i.e., distance from source)
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] preds Predecessor node id for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 * @param[in] MARK_PREDECESSORS Whether to show predecessor of each node.
 */
template<typename VertexId, typename SizeT>
void DisplaySolution(VertexId *source_path, SizeT num_nodes)
{
  if (num_nodes > 40)
    num_nodes = 40;
  printf("[");
  for (VertexId i = 0; i < num_nodes; ++i)
    {
      PrintValue(i);
      printf(":");
      PrintValue(source_path[i]);
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
  Statistic  rate;
  Statistic  search_depth;
  Statistic  redundant_work;
  Statistic  duty;

  Stats() : name(NULL), rate(), search_depth(), redundant_work(), duty() {}
  Stats(const char *name) : name(name), rate(), search_depth(), redundant_work(), duty() {}
};

/**
 * @brief Displays timing and correctness statistics
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] stats Reference to the Stats object defined in RunTests
 * @param[in] source Source node where SSSP starts
 * @param[in] h_labels Host-side vector stores computed labels for validation
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] elapsed Total elapsed kernel running time
 * @param[in] search_depth Maximum search depth of the SSSP algorithm
 * @param[in] total_queued Total element queued in SSSP kernel running process
 * @param[in] avg_duty Average duty of the SSSP kernels
 */
template<
  typename VertexId,
  typename Value,
  typename SizeT>
void DisplayStats(
		  Stats               &stats,
		  VertexId            source,
		  unsigned int        *h_labels,
		  const Csr<VertexId, Value, SizeT> &graph,
		  double              elapsed,
		  VertexId            search_depth,
		  long long           total_queued,
		  double              avg_duty)
{
  // Compute nodes and edges visited
  SizeT edges_visited = 0;
  SizeT nodes_visited = 0;
  for (VertexId i = 0; i < graph.nodes; ++i)
    {
      if (h_labels[i] < UINT_MAX)
        {
	  ++nodes_visited;
	  edges_visited += graph.row_offsets[i+1] - graph.row_offsets[i];
        }
    }

  double redundant_work = 0.0;
  if (total_queued > 0)
    {
      redundant_work = ((double) total_queued - edges_visited) / edges_visited; // measure duplicate edges put through queue
    }
  redundant_work *= 100;

  // Display test name
  printf("\n[%s] finished.\n", stats.name);

  // Display statistics
  if (nodes_visited < 5)
    {
      printf("Fewer than 5 vertices visited.\n");
    }
  else
    {
      // Display the specific sample statistics
      double m_teps = (double) edges_visited / (elapsed * 1000.0);
      printf(" elapsed: %.3f ms, rate: %.3f MiEdges/s", elapsed, m_teps);
      if (search_depth != 0) printf(", search_depth: %lld", (long long) search_depth);
      if (avg_duty != 0)
        {
	  //printf("\n avg CTA duty: %.2f%%", avg_duty * 100);
        }
      printf("\n source: %lld, nodes_visited: %lld, edges visited: %lld",
	     (long long) source, (long long) nodes_visited, (long long) edges_visited);
      if (total_queued > 0)
        {
	  printf(", total queued: %lld", total_queued);
        }
      if (redundant_work > 0)
        {
	  printf(", redundant work: %.2f%%", redundant_work);
        }
      printf("\n");
    }
}

/******************************************************************************
 * SSSP Testing Routines
 *****************************************************************************/
/**
 * @brief Run SSSP tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] source Source node where SSSP starts
 * @param[in] max_grid_size Maximum CTA occupancy
 * @param[in] num_gpus Number of GPUs
 * @param[in] max_queue_sizing Scaling factor used in edge mapping
 *
 */
template <
  typename VertexId,
  typename Value,
  typename SizeT,
  bool MARK_PREDECESSORS>
void run_sssp(
	      GunrockGraph *ggraph_out,
	      VertexId     *predecessor,
	      const Csr<VertexId, Value, SizeT> &graph,
	      VertexId     source,
	      int          max_grid_size,
	      float        queue_sizing,
	      int          num_gpus,
	      int          delta_factor,
	      CudaContext& context)
{
  // Preparations
    typedef SSSPProblem<
      VertexId,
      SizeT,
      MARK_PREDECESSORS> Problem;

    // Allocate host-side label array for gpu-computed results
    unsigned int *h_labels = (unsigned int*)malloc(sizeof(unsigned int) * graph.nodes);
    //VertexId     *h_preds  = NULL;

    if (MARK_PREDECESSORS)
      {
        //h_preds = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);
      }

    // Allocate SSSP enactor map
    SSSPEnactor<false> sssp_enactor(g_verbose);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
				    g_stream_from_host,
				    graph,
				    num_gpus,
				    delta_factor),
		  "Problem SSSP Initialization Failed", __FILE__, __LINE__);

    Stats *stats = new Stats("GPU SSSP");

    long long total_queued =   0;
    VertexId  search_depth =   0;
    double    avg_duty     = 0.0;

    // Perform SSSP
    CpuTimer gpu_timer;

    util::GRError(csr_problem->Reset(
				     source, sssp_enactor.GetFrontierType(), queue_sizing),
		  "SSSP Problem Data Reset Failed", __FILE__, __LINE__);
    gpu_timer.Start();
    util::GRError(sssp_enactor.template Enact<Problem>(
						       context, csr_problem, source, queue_sizing, max_grid_size),
		  "SSSP Problem Enact Failed", __FILE__, __LINE__);
    gpu_timer.Stop();

    sssp_enactor.GetStatistics(total_queued, search_depth, avg_duty);
    float elapsed = gpu_timer.ElapsedMillis();

    // Copy out results
    util::GRError(csr_problem->Extract(h_labels, predecessor),
		  "SSSP Problem Data Extraction Failed", __FILE__, __LINE__);

    // copy label_values per node to GunrockGraph output
    ggraph_out->node_values = (unsigned int*)&h_labels[0];

    /*
    // Display Solution
    printf("\nFirst %d of labels the GPU result.\n", graph.nodes);
    DisplaySolution(h_labels, graph.nodes);

    if (MARK_PREDECESSORS)
    {
        printf("\nFirst %d of predecessors the GPU result.\n", graph.nodes);
        DisplaySolution(predecessor, graph.nodes);
    }
    */

    DisplayStats(
		 *stats,
		 source,
		 h_labels,
		 graph,
		 elapsed,
		 search_depth,
		 total_queued,
		 avg_duty);

    // Cleanup
    delete stats;
    if (csr_problem) delete csr_problem;
    //if (h_labels)    free(h_labels);
    //if (h_preds)     free(h_preds);

    cudaDeviceSynchronize();
}

/**
 * @brief run_sssp entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] args Reference to the command line arguments
 */
void gunrock_sssp(
		  GunrockGraph       *ggraph_out,
		  void               *predecessor,
		  const GunrockGraph *ggraph_in,
		  GunrockConfig      sssp_config,
		  GunrockDataType    data_type)
{
  // moderngpu preparations
  int device = 0;
  device = sssp_config.device;
  ContextPtr context = mgpu::CreateCudaDevice(device);

  // build input csr format graph
  Csr<int, unsigned int, int> csr_graph(false);
  csr_graph.nodes = ggraph_in->num_nodes;
  csr_graph.edges = ggraph_in->num_edges;
  csr_graph.row_offsets    = (int*)ggraph_in->row_offsets;
  csr_graph.column_indices = (int*)ggraph_in->col_indices;
  csr_graph.edge_values    = (unsigned int*)ggraph_in->edge_values;

  int   source           = 0;     //!< use whatever the specified graph-type's default is
  int   max_grid_size    = 0;     //!< maximum grid size (0: leave it up to the enactor)
  int   num_gpus         = 1;     //!< number of GPUs for multi-gpu enactor to use
  float max_queue_sizing = 1.0;
  bool  mark_pred        = false;
  int   delta_factor     = 1;

  // determine source vertex to start sssp
  switch (sssp_config.src)
    {
    case randomize:
      {
	source = graphio::RandomNode(csr_graph.nodes);
	break;
      }
    case largest_degree:
      {
	source = csr_graph.GetNodeWithHighestDegree();
	break;
      }
    case manually:
      {
	source = sssp_config.source;
	break;
      }
    }

  mark_pred        = sssp_config.mark_pred;
  delta_factor     = sssp_config.delta_factor;
  max_queue_sizing = sssp_config.queue_size;

  if (mark_pred)
    {
      run_sssp<int, unsigned int, int, true>(
					     ggraph_out,
					     (int*)predecessor,
					     csr_graph,
					     source,
					     max_grid_size,
					     max_queue_sizing,
					     num_gpus,
					     delta_factor,
					     *context);
    }
  else
    {
      run_sssp<int, unsigned int, int, false>(
					      ggraph_out,
					      (int*)predecessor,
					      csr_graph,
					      source,
					      max_grid_size,
					      max_queue_sizing,
					      num_gpus,
					      delta_factor,
					      *context);
    }

  // reset for free memory
  csr_graph.row_offsets    = NULL;
  csr_graph.column_indices = NULL;
  csr_graph.row_offsets    = NULL;
  csr_graph.column_indices = NULL;
  csr_graph.edge_values    = NULL;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
