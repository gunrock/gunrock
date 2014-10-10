// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_sssp.cu
 *
 * @brief single-source shortest path problem implementation
 */

#include <stdio.h>
#include <gunrock/gunrock.h>

// Graph construction utils
#include <gunrock/graphio/market.cuh>

// SSSP includes
#include <gunrock/app/sssp/sssp_enactor.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/app/sssp/sssp_functor.cuh>

// Moderngpu include
#include <moderngpu.cuh>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::sssp;

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
    const Stats        &stats,
    const VertexId     source,
    const unsigned int *h_labels,
    const Csr<VertexId, Value, SizeT> &graph,
    const double       elapsed,
    const VertexId     search_depth,
    const long long    total_queued,
    const double       avg_duty)
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
        // measure duplicate edges put through queue
        redundant_work = ((double) total_queued - edges_visited) / edges_visited;
    }
    redundant_work *= 100;

    // Display test name
    printf("%s finished.\n", stats.name);

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
        printf(", search_depth: %lld", (long long) search_depth);
        printf("\n source: %lld, nodes_visited: %lld, edges visited: %lld",
            (long long) source, (long long) nodes_visited, (long long) edges_visited);
        printf("\n");
    }
}

/**
 * @brief run single-source shortest path procedures
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[out] ggraph_out GunrockGraph type output
 * @param[out] predecessor return predeessor if mark_pred = true
 * @param[in]  graph Reference to the CSR graph we process on
 * @param[in]  source Source node where SSSP starts
 * @param[in]  max_grid_size Maximum CTA occupancy
 * @param[in]  max_queue_sizing Scaling factor used in edge mapping
 * @param[in]  num_gpus Number of GPUs
 * @param[in]  delta_factor user set
 * @param[in]  context moderngpu context
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool MARK_PREDECESSORS>
void run_sssp(
    GunrockGraph   *ggraph_out,
    VertexId       *predecessor,
    const Csr<VertexId, Value, SizeT> &graph,
    const VertexId source,
    const int      max_grid_size,
    const float    queue_sizing,
    const int      num_gpus,
    const int      delta_factor,
    CudaContext& context)
{
    // Preparations
    typedef SSSPProblem<
        VertexId,
        SizeT,
        MARK_PREDECESSORS> Problem;

    // Allocate host-side label array for gpu-computed results
    unsigned int *h_labels
        = (unsigned int*)malloc(sizeof(unsigned int) * graph.nodes);
    //VertexId     *h_preds  = NULL;

    if (MARK_PREDECESSORS)
    {
        //h_preds = (VertexId*)malloc(sizeof(VertexId) * graph.nodes);
    }

    // Allocate SSSP enactor map
    SSSPEnactor<false> sssp_enactor(false);

    // Allocate problem on GPU
    Problem *csr_problem = new Problem;
    util::GRError(csr_problem->Init(
        false,
        graph,
        num_gpus,
        delta_factor),
        "Problem SSSP Initialization Failed", __FILE__, __LINE__);

    Stats *stats = new Stats("Single-Source Shortest Path");

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
    float elapsed = gpu_timer.ElapsedMillis();

    /*
    long long total_queued =   0;
    VertexId  search_depth =   0;
    double    avg_duty     = 0.0;
    sssp_enactor.GetStatistics(total_queued, search_depth, avg_duty);
    */

    // Copy out results
    util::GRError(csr_problem->Extract(h_labels, predecessor),
        "SSSP Problem Data Extraction Failed", __FILE__, __LINE__);

    // copy label_values per node to GunrockGraph output
    ggraph_out->node_values = (unsigned int*)&h_labels[0];

    /*
    DisplayStats(
        *stats,
        source,
        h_labels,
        graph,
        elapsed,
        search_depth,
        total_queued,
        avg_duty);
    */

    // Clean up
    delete stats;
    if (csr_problem) delete csr_problem;
    //if (h_labels)    free(h_labels);
    //if (h_preds)     free(h_preds);

    cudaDeviceSynchronize();
}

/**
 * @brief dispatch function to handle data_types
 *
 * @param[out] ggraph_out  GunrockGraph type output
 * @param[out] predecessor return predeessor if mark_pred = true
 * @param[in]  ggraph_in   GunrockGraph type input graph
 * @param[in]  sssp_config sssp specific configurations
 * @param[in]  data_type   sssp data_type configurations
 * @param[in]  context     moderngpu context
 */
void dispatch_sssp(
    GunrockGraph          *ggraph_out,
    void                  *predecessor,
    const GunrockGraph    *ggraph_in,
    const GunrockConfig   sssp_config,
    const GunrockDataType data_type,
    CudaContext&          context)
{
    switch (data_type.VTXID_TYPE) {
    case VTXID_INT: {
        switch (data_type.SIZET_TYPE) {
        case SIZET_INT: {
            switch (data_type.VALUE_TYPE) {
            case VALUE_INT: {
                // template type = <int, int, int>
                // not support yet
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            case VALUE_UINT: {
                // template type = <int, uint, int>
                // build input csr format graph
                Csr<int, unsigned int, int> csr_graph(false);
                csr_graph.nodes          = ggraph_in->num_nodes;
                csr_graph.edges          = ggraph_in->num_edges;
                csr_graph.row_offsets    = (int*)ggraph_in->row_offsets;
                csr_graph.column_indices = (int*)ggraph_in->col_indices;
                csr_graph.edge_values    = (unsigned int*)ggraph_in->edge_values;

                // sssp configurations
                bool  mark_pred        = false;
                int   src_node         = 0; //!< use whatever the specified graph-type's default is
                int   num_gpus         = 1; //!< number of GPUs for multi-gpu enactor to use
                int   delta_factor     = 1; //!< default delta_factor = 1
                int   max_grid_size    = 0; //!< maximum grid size (0: leave it up to the enactor)
                float max_queue_sizing = 1.0; //!< default maximum queue sizing

                // determine source vertex to start sssp
                switch (sssp_config.src_mode)
                {
                    case randomize:
                    {
                        src_node = graphio::RandomNode(csr_graph.nodes);
                        break;
                    }
                    case largest_degree:
                    {
                        int max_deg = 0;
                        src_node = csr_graph.GetNodeWithHighestDegree(max_deg);
                        break;
                    }
                    case manually:
                    {
                        src_node = sssp_config.src_node;
                        break;
                    }
                    default:
                    {
                        src_node = 0;
                        break;
                    }
                }
                mark_pred        = sssp_config.mark_pred;
                delta_factor     = sssp_config.delta_factor;
                max_queue_sizing = sssp_config.queue_size;

                switch (mark_pred)
                {
                case true: {
                    run_sssp<int, unsigned int, int, true>(
                        ggraph_out,
                        (int*)predecessor,
                        csr_graph,
                        src_node,
                        max_grid_size,
                        max_queue_sizing,
                        num_gpus,
                        delta_factor,
                        context);
                    break;
                }
                case false: {
                    run_sssp<int, unsigned int, int, false>(
                        ggraph_out,
                        (int*)predecessor,
                        csr_graph,
                        src_node,
                        max_grid_size,
                        max_queue_sizing,
                        num_gpus,
                        delta_factor,
                        context);
                    break;
                }
                }
                // reset for free memory
                csr_graph.row_offsets    = NULL;
                csr_graph.column_indices = NULL;
                csr_graph.edge_values    = NULL;
                break;
            }
            case VALUE_FLOAT: {
                // template type = <int, float, int>
                // not support yet
                printf("Not Yet Support This DataType Combination.\n");
                break;
            }
            }
        break;
        }
        }
        break;
    }
    }
}

/**
 * @brief run_sssp entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[out] ggraph_out  GunrockGraph type output
 * @param[out] predecessor return predeessor if mark_pred = true
 * @param[in]  ggraph_in   GunrockGraph type input graph
 * @param[in]  sssp_config gunrock primitive specific configurations
 * @param[in]  data_type   data_type configurations
 */
void gunrock_sssp_func(
    GunrockGraph          *ggraph_out,
    void                  *predecessor,
    const GunrockGraph    *ggraph_in,
    const GunrockConfig   sssp_config,
    const GunrockDataType data_type)
{
    // moderngpu preparations
    int device = 0;
    device = sssp_config.device;
    ContextPtr context = mgpu::CreateCudaDevice(device);

    // lunch dispatch function
    dispatch_sssp(
        ggraph_out,
        predecessor,
        ggraph_in,
        sssp_config,
        data_type,
        *context);
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
