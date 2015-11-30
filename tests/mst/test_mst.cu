// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * test_mst.cu
 *
 * @brief Simple test driver for computing Minimum Spanning Tree.
 */

#include <stdio.h>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <algorithm>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utilities
#include <gunrock/graphio/market.cuh>

// MST includes
#include <gunrock/app/cc/cc_app.cu>
#include <gunrock/app/mst/mst_enactor.cuh>
#include <gunrock/app/mst/mst_problem.cuh>
#include <gunrock/app/mst/mst_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <moderngpu.cuh>

// CPU Kruskal MST reference
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>

using namespace gunrock;
using namespace gunrock::app;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::mst;

///////////////////////////////////////////////////////////////////////////////
// Housekeeping and utility routines
///////////////////////////////////////////////////////////////////////////////

void Usage()
{
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
        "[--quick]                 Skip the CPU reference validation process.\n"
        "[--disable-size-check]    Disable frontier queue size check.\n"
        "[--v]                     Print verbose per iteration debug info.\n"
        "[--iteration-num=<num>]   Number of runs to perform the test.\n"
        "[--quiet]                 No output (unless --json is specified).\n"
        "[--json]                  Output JSON-format statistics to STDOUT.\n"
        "[--jsonfile=<name>]       Output JSON-format statistics to file <name>\n"
        "[--jsondir=<dir>]         Output JSON-format statistics to <dir>/name,\n"
        "                          where name is auto-generated.\n"
    );
}

/**
 * @brief Displays the MST result.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to the CSR graph.
 * @param[in] edge_mask Pointer to the MST edge mask.
 */
template<typename VertexId, typename Value, typename SizeT>
void DisplaySolution(
    const Csr<VertexId, Value, SizeT> &graph, int *edge_mask)
{
    int count = 0;
    int print_limit = graph.nodes;
    if (print_limit > 10)
    {
        print_limit = 10;
    }

    // find source vertex ids for display results
    VertexId *source = new VertexId[graph.edges];
    for (int i = 0; i < graph.nodes; ++i)
    {
        for (int j = graph.row_offsets[i]; j < graph.row_offsets[i + 1]; ++j)
        {
            source[j] = i;
        }
    }

    // print source-destination pairs of minimum spanning tree edges
    printf("GPU Minimum Spanning Tree [First %d edges]\n", print_limit);
    printf("src dst\n");
    for (int i = 0; i < graph.edges; ++i)
    {
        if (edge_mask[i] == 1 && count <= print_limit)
        {
            printf("%d %d\n", source[i], graph.column_indices[i]);
            ++count;
        }
    }

    // clean up if necessary
    if (source) { delete [] source; }
}

///////////////////////////////////////////////////////////////////////////////
// CPU validation routines
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief A simple CPU-based reference MST implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] edge_values Weight value associated with each edge.
 * @param[in] graph Reference to the CSR graph we process on.
 * @param[in] quiet_mode Don't print out anything to stdout.
 *
 *  \return long long int which indicates the total weight of the graph.
 */
template<typename VertexId, typename Value, typename SizeT>
Value SimpleReferenceMST(
    const Value *edge_values,
    const Csr<VertexId, Value, SizeT> &graph,
    bool quiet_mode = false)
{
    if (!quiet_mode) { printf("\nMST CPU REFERENCE TEST\n"); }

    // Kruskal's minimum spanning tree preparations
    using namespace boost;
    typedef adjacency_list< vecS, vecS, undirectedS,
            no_property, property<edge_weight_t, int> > Graph;
    typedef graph_traits < Graph >::edge_descriptor   Edge;
    typedef graph_traits < Graph >::vertex_descriptor Vertex;
    typedef std::pair<VertexId, VertexId> E;

    E *edge_pairs = new E[graph.edges];
    int idx = 0;
    for (int i = 0; i < graph.nodes; ++i)
    {
        for (int j = graph.row_offsets[i]; j < graph.row_offsets[i + 1]; ++j)
        {
            edge_pairs[idx++] = std::make_pair(i, graph.column_indices[j]);
        }
    }

    Graph g(edge_pairs, edge_pairs + graph.edges, edge_values, graph.nodes);
    property_map < Graph, edge_weight_t >::type weight = get(edge_weight, g);
    std::vector < Edge > spanning_tree;

    CpuTimer cpu_timer; // record the kernel running time
    cpu_timer.Start();

    // compute reference using kruskal_min_spanning_tree algorithm
    kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

    cpu_timer.Stop();
    float elapsed_cpu = cpu_timer.ElapsedMillis();

    // analyze reference results
    SizeT num_selected_cpu = 0;
    Value total_weight_cpu = 0;

    if (graph.nodes <= 50 && !quiet_mode)
    {
        printf("CPU Minimum Spanning Tree\n");
    }
    for (std::vector < Edge >::iterator ei = spanning_tree.begin();
            ei != spanning_tree.end(); ++ei)
    {
        if (graph.nodes <= 50 && !quiet_mode)
        {
            // print the edge pairs in the minimum spanning tree
            printf("%ld %ld\n", source(*ei, g), target(*ei, g));
            // printf("  with weight of %f\n", weight[*ei]);
        }
        ++num_selected_cpu;
        total_weight_cpu += weight[*ei];
    }

    // clean up if necessary
    if (edge_pairs) { delete [] edge_pairs; }

    if (!quiet_mode)
    {
        printf("CPU - Computation Complete in %lf msec.\n", elapsed_cpu);
        // printf("CPU - Number of Edges in MST: %d\n", num_selected_cpu);
    }

    return total_weight_cpu;
}

///////////////////////////////////////////////////////////////////////////////
// GPU MST test routines
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Test entry
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam DEBUG
 * @tparam SIZE_CHECK
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    bool     DEBUG,
    bool     SIZE_CHECK >
void RunTest(Info<VertexId, Value, SizeT> *info)
{
    // define the problem data structure for graph primitive
    typedef MSTProblem<VertexId,
            SizeT,
            Value,
            true,    // MARK_PREDECESSORS
            false,   // ENABLE_IDEMPOTENCE
            true >   // USE_DOUBLE_BUFFER
            Problem;

    Csr<VertexId, Value, SizeT>* graph =
        (Csr<VertexId, Value, SizeT>*)info->csr_ptr;
    int num_gpus            = info->info["num_gpus"].get_int();
    int max_grid_size       = info->info["max_grid_size"].get_int();
    int iterations          = 1; //force to 1 info->info["num_iteration"].get_int();
    bool quiet_mode         = info->info["quiet_mode"].get_bool();
    bool quick_mode         = info->info["quick_mode"].get_bool();
    bool stream_from_host   = info->info["stream_from_host"].get_bool();
    double max_queue_sizing = info->info["max_queue_sizing"].get_real();
    json_spirit::mArray device_list = info->info["device_list"].get_array();
    int* gpu_idx = new int[num_gpus];
    for (int i = 0; i < num_gpus; i++) gpu_idx[i] = device_list[i].get_int();

    // TODO: remove after merge mgpu-cq
    ContextPtr* context = (ContextPtr*)info->context;

    // allocate MST enactor map
    MSTEnactor < Problem,
               false,        // INSTRUMENT
               DEBUG,        // DEBUG
               SIZE_CHECK >  // SIZE_CHECK
               enactor(gpu_idx);

    // allocate problem on GPU create a pointer of the MSTProblem type
    Problem * problem = new Problem;

    // host results spaces
    VertexId * edge_mask = new VertexId[graph->edges];

    if (!quiet_mode) { printf("\nMINIMUM SPANNING TREE TEST\n"); }

    // copy data from CPU to GPU initialize data members in DataSlice
    util::GRError(problem->Init(stream_from_host, *graph, num_gpus),
                  "Problem MST Initialization Failed", __FILE__, __LINE__);

    // perform calculations
    GpuTimer gpu_timer;  // record the kernel running time
    double elapsed_gpu = 0.0f;  // device elapsed running time

    for (int iter = 0; iter < iterations; ++iter)
    {
        // reset values in DataSlice
        util::GRError(problem->Reset(
                          enactor.GetFrontierType(), max_queue_sizing),
                      "MST Problem Data Reset Failed", __FILE__, __LINE__);

        gpu_timer.Start();

        // launch MST enactor
        util::GRError(enactor.template Enact<Problem>(
                          *context, problem, max_grid_size),
                      "MST Problem Enact Failed", __FILE__, __LINE__);

        gpu_timer.Stop();
        elapsed_gpu += gpu_timer.ElapsedMillis();
    }

    elapsed_gpu /= iterations;
    if (!quiet_mode)
    {
        printf("GPU - Computation Complete in %lf msec.\n", elapsed_gpu);
    }

    // copy results back to CPU from GPU using Extract
    util::GRError(problem->Extract(edge_mask),
                  "MST Problem Data Extraction Failed", __FILE__, __LINE__);

    if (!quick_mode)  // run CPU reference test
    {
        // calculate GPU final number of selected edges
        int num_selected_gpu = 0;
        for (int iter = 0; iter < graph->edges; ++iter)
        {
            num_selected_gpu += edge_mask[iter];
        }
        // printf("\nGPU - Number of Edges in MST: %d\n", num_selected_gpu);

        // calculate GPU total selected MST weights for validation
        Value total_weight_gpu = 0;
        for (int iter = 0; iter < graph->edges; ++iter)
        {
            total_weight_gpu += edge_mask[iter] * graph->edge_values[iter];
        }

        // correctness validation
        Value total_weight_cpu = SimpleReferenceMST(
                                     graph->edge_values, *graph, quiet_mode);
        if (total_weight_cpu == total_weight_gpu)
        {
            // print the edge pairs in the minimum spanning tree
            if (!quiet_mode) DisplaySolution(*graph, edge_mask);
            if (!quiet_mode) { printf("\nCORRECT.\n"); }
        }
        else
        {
            if (!quiet_mode)
            {
                printf("INCORRECT.\n");
                std::cout << "CPU Weight = " << total_weight_cpu << std::endl;
                std::cout << "GPU Weight = " << total_weight_gpu << std::endl;
            }
        }
    }


    info->ComputeCommonStats(enactor.enactor_stats.GetPointer(), elapsed_gpu);

    if (!quiet_mode)
    {
        info->DisplayStats(false);   // display collected statistics
    }

    info->CollectInfo();

    // clean up if necessary
    if (problem)   delete    problem;
    if (edge_mask) delete [] edge_mask;
}

/**
 * @brief Test entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam DEBUG
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool     DEBUG >
void RunTests_size_check(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["size_check"].get_bool())
    {
        RunTest <VertexId, Value, SizeT, DEBUG,  true>(info);
    }
    else
    {
        RunTest <VertexId, Value, SizeT, DEBUG, false>(info);
    }
}

/**
 * @brief Test entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT >
void RunTests_debug(Info<VertexId, Value, SizeT> *info)
{
    if (info->info["debug_mode"].get_bool())
    {
        RunTests_size_check <VertexId, Value, SizeT,  true>(info);
    }
    else
    {
        RunTests_size_check <VertexId, Value, SizeT, false>(info);
    }
}

/**
 * @brief Test entry
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 */
template <typename VertexId, typename Value, typename SizeT>
void RunTest_connectivity_check(Info<VertexId, Value, SizeT> *info)
{
    // test graph connectivity because MST only supports fully-connected graph
    struct GRTypes data_t;          // data type structure
    data_t.VTXID_TYPE = VTXID_INT;  // vertex identifier
    data_t.SIZET_TYPE = SIZET_INT;  // graph size type
    data_t.VALUE_TYPE = VALUE_INT;  // attributes type

    struct GRSetup config = InitSetup(1, NULL);  // gunrock configurations

    struct GRGraph *grapho = (GRGraph*)malloc(sizeof(GRGraph));
    struct GRGraph *graphi = (GRGraph*)malloc(sizeof(GRGraph));

    graphi->num_nodes = info->csr_ptr->nodes;
    graphi->num_edges = info->csr_ptr->edges;
    graphi->row_offsets = (void*)&info->csr_ptr->row_offsets[0];
    graphi->col_indices = (void*)&info->csr_ptr->column_indices[0];

    gunrock_cc(grapho, graphi, config, data_t);

    // run test only if the graph is fully-connected
    int* num_cc = (int*)grapho->aggregation;
    if (*num_cc == 1)  // perform minimum spanning tree test
    {
        RunTests_debug<VertexId, Value, SizeT>(info);
    }
    else  // more than one connected components in the graph
    {
        fprintf(stderr, "Unsupported non-fully connected graph input.\n");
        exit(1);
    }

    if (graphi) free(graphi);
    if (grapho) free(grapho);
}

///////////////////////////////////////////////////////////////////////////////
// Main function
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
    CommandLineArgs args(argc, argv);
    int graph_args = argc - args.ParsedArgc() - 1;
    if (argc < 2 || graph_args < 1 || args.CheckCmdLineFlag("help"))
    {
        Usage();
        return 1;
    }

    typedef int VertexId;  // use int as the vertex identifier
    typedef int Value;     // use int as the value type
    typedef int SizeT;     // use int as the graph size type

    Csr<VertexId, Value, SizeT> csr(false);  // graph we process on
    Info<VertexId, Value, SizeT> *info = new Info<VertexId, Value, SizeT>;

    // graph construction or generation related parameters
    info->info["undirected"] = true;  // always convert to undirected
    info->info["edge_value"] = true;  // require per edge weight values

    info->Init("MST", args, csr);
    RunTest_connectivity_check<VertexId, Value, SizeT>(info);  // run test

    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End
