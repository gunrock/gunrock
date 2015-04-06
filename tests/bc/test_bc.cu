// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_bc.cu
 *
 * @brief Simple test driver program for BC.
 */

#include <stdio.h>
#include <string>
#include <deque>
#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <algorithm>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph construction utils
#include <gunrock/graphio/market.cuh>
#include <gunrock/graphio/rmat.cuh>
#include <gunrock/graphio/rgg.cuh>

// BC includes
#include <gunrock/app/bc/bc_enactor.cuh>
#include <gunrock/app/bc/bc_problem.cuh>
#include <gunrock/app/bc/bc_functor.cuh>

// Operator includes
#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>

#include <moderngpu.cuh>

// Boost includes
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/bc_clustering.hpp>
#include <boost/graph/iteration_macros.hpp>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::oprtr;
using namespace gunrock::app::bc;


/******************************************************************************
 * Defines, constants, globals
 ******************************************************************************/

//bool g_verbose;
//bool g_undirected;
//bool g_quick;
//bool g_stream_from_host;

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/
void Usage()
{
    printf("\ntest_bc <graph type> <graph type args> [--device=<device_index>] "
           "[--instrumented] [--src=<source index>] [--quick] [--v]"
           "[--queue-sizing=<scale factor>] [--ref-file=<reference filename>]\n"
           "[--in-sizing=<in/out queue scale factor>] [--disable-size-check] "
           "[--grid-size=<grid size>] [partition_method=random / biasrandom / clustered / metis]\n"
           "\n"
           "Graph types and args:\n"
           "  market [<file>]\n"
           "    Reads a Matrix-Market coordinate-formatted graph of undirected\n"
           "    edges from stdin (or from the optionally-specified file).\n"
           "--device=<device_index>: Set GPU device for running the graph primitive.\n"
           "--undirected: If set then treat the graph as undirected graph.\n"
           "--instrumented: If set then kernels keep track of queue-search_depth\n"
           "and barrier duty (a relative indicator of load imbalance.)\n"
           "--src=<source index>: When source index is -1, compute BC value for each\n"
           "node. Otherwise, debug the delta value for one node\n"
           "--quick: If set will skip the CPU validation code.\n"
           "--queue-sizing Allocates a frontier queue sized at (graph-edges * <scale factor>).\n"
           "Default is 1.0.\n"
           "--v: If set, enable verbose output, keep track of the kernel running.\n"
           "--ref-file: If set, use pre-computed result stored in ref-file to verify.\n"
           );
}

/**
 * @brief Displays the BC result (sigma value and BC value)
 *
 * @param[in] sigmas
 * @param[in] bc_values
 * @param[in] nodes
 */
template<typename Value, typename SizeT>
void DisplaySolution(Value *sigmas, Value *bc_values, SizeT nodes)
{
    if (nodes < 40) {
        printf("[");
        for (SizeT i = 0; i < nodes; ++i) {
            PrintValue(i);
            printf(":");
            PrintValue(sigmas[i]);
            printf(",");
            PrintValue(bc_values[i]);
            printf(" ");
        }
        printf("]\n");
    }
}

struct Test_Parameter : gunrock::app::TestParameter_Base {
public:
    std::string ref_filename;
    double max_queue_sizing1;

    Test_Parameter()
    {
        ref_filename="";
        max_queue_sizing1=-1.0;
    }

    ~Test_Parameter()
    {
    }

    void Init(CommandLineArgs &args)
    {
        TestParameter_Base::Init(args);
        args.GetCmdLineArgument("queue-sizing1", max_queue_sizing1);
    }
};

/******************************************************************************
* BC Testing Routines
*****************************************************************************/

/**
 * @brief Graph edge properties (bundled properties)
 */
struct EdgeProperties
{
    int weight;
};

/**
 * @brief A simple CPU-based reference BC ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 *
 * @param[in] graph Reference to ...
 * @param[in] bc_values Pointer to ...
 * @param[in] sigmas Pointer to ...
 * @param[in] src VertexId of ...
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void RefCPUBC(
    const Csr<VertexId, Value, SizeT> &graph,
    Value                             *bc_values,
    Value                             *ebc_values,
    Value                             *sigmas,
    VertexId                          *source_path,
    VertexId                           src)
{
    typedef Coo<VertexId, Value> EdgeTupleType;
    EdgeTupleType *coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * graph.edges);
    if (src == -1) {
        // Perform full exact BC using BGL

        using namespace boost;
        typedef adjacency_list <setS, vecS, undirectedS, no_property,
                                EdgeProperties> Graph;
        typedef Graph::vertex_descriptor Vertex;
        typedef Graph::edge_descriptor Edge;

        Graph G;
        for (int i = 0; i < graph.nodes; ++i)
        {
            for (int j = graph.row_offsets[i]; j < graph.row_offsets[i+1]; ++j)
            {
                add_edge(vertex(i, G), vertex(graph.column_indices[j], G), G);
            }
        }

        typedef std::map<Edge, int> StdEdgeIndexMap;
        StdEdgeIndexMap my_e_index;
        typedef boost::associative_property_map< StdEdgeIndexMap > EdgeIndexMap;
        EdgeIndexMap e_index(my_e_index);

        int i = 0;
        BGL_FORALL_EDGES(edge, G, Graph)
        {
            my_e_index.insert(std::pair<Edge, int>(edge, i));
            ++i;
        }

        // Define EdgeCentralityMap
        std::vector< double > e_centrality_vec(boost::num_edges(G), 0.0);
        // Create the external property map
        boost::iterator_property_map< std::vector< double >::iterator,
                                      EdgeIndexMap >
            e_centrality_map(e_centrality_vec.begin(), e_index);

        // Define VertexCentralityMap
        typedef boost::property_map< Graph, boost::vertex_index_t>::type
            VertexIndexMap;
        VertexIndexMap v_index = get(boost::vertex_index, G);
        std::vector< double > v_centrality_vec(boost::num_vertices(G), 0.0);

        // Create the external property map
        boost::iterator_property_map< std::vector< double >::iterator,
                                      VertexIndexMap>
            v_centrality_map(v_centrality_vec.begin(), v_index);

        //
        // Perform BC
        //
        CpuTimer cpu_timer;
        cpu_timer.Start();
        brandes_betweenness_centrality( G, v_centrality_map, e_centrality_map );
        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();

        BGL_FORALL_VERTICES(vertex, G, Graph)
        {
            bc_values[vertex] = (Value)v_centrality_map[vertex];
        }

        int idx = 0;
        BGL_FORALL_EDGES(edge, G, Graph)
        {
            coo[idx].row = source(edge, G);
            coo[idx].col = target(edge, G);
            coo[idx++].val = (Value)e_centrality_map[edge];
            coo[idx].col = source(edge, G);
            coo[idx].row = target(edge, G);
            coo[idx++].val = (Value)e_centrality_map[edge];
        }

        std::stable_sort(coo, coo+graph.edges, RowFirstTupleCompare<EdgeTupleType>);

        for (idx = 0; idx < graph.edges; ++idx) {
            //std::cout << coo[idx].row << "," << coo[idx].col << ":" << coo[idx].val << std::endl;
            ebc_values[idx] = coo[idx].val;
        }

        printf("CPU BC finished in %lf msec.", elapsed);
    }
    else {
        //Simple BFS pass to get single pass BC
        //VertexId *source_path = new VertexId[graph.nodes];

        //initialize distances
        for (VertexId i = 0; i < graph.nodes; ++i) {
            source_path[i] = -1;
            bc_values[i] = 0;
            sigmas[i] = 0;
        }
        source_path[src] = 0;
        VertexId search_depth = 0;
        sigmas[src] = 1;

        // Initialize queue for managing previously-discovered nodes
        std::deque<VertexId> frontier;
        frontier.push_back(src);

        //
        //Perform one pass of BFS for one source
        //

        CpuTimer cpu_timer;
        cpu_timer.Start();
        while (!frontier.empty()) {

            // Dequeue node from frontier
            VertexId dequeued_node = frontier.front();
            frontier.pop_front();
            VertexId neighbor_dist = source_path[dequeued_node] + 1;

            // Locate adjacency list
            int edges_begin = graph.row_offsets[dequeued_node];
            int edges_end = graph.row_offsets[dequeued_node + 1];

            for (int edge = edges_begin; edge < edges_end; ++edge) {
                // Lookup neighbor and enqueue if undiscovered
                VertexId neighbor = graph.column_indices[edge];
                if (source_path[neighbor] == -1) {
                    source_path[neighbor] = neighbor_dist;
                    sigmas[neighbor] += sigmas[dequeued_node];
                    if (search_depth < neighbor_dist) {
                        search_depth = neighbor_dist;
                    }

                    frontier.push_back(neighbor);
                }
                else {
                    if (source_path[neighbor] == source_path[dequeued_node]+1)
                        sigmas[neighbor] += sigmas[dequeued_node];
                }
            }
        }
        search_depth++;

        for (int iter = search_depth - 2; iter > 0; --iter)
        {
            for (int node = 0; node < graph.nodes; ++node)
            {
                if (source_path[node] == iter) {
                    int edges_begin = graph.row_offsets[node];
                    int edges_end = graph.row_offsets[node+1];

                    for (int edge = edges_begin; edge < edges_end; ++edge) {
                        VertexId neighbor = graph.column_indices[edge];
                        if (source_path[neighbor] == iter + 1) {
                            bc_values[node] +=
                                1.0f * sigmas[node] / sigmas[neighbor] *
                                (1.0f + bc_values[neighbor]);
                        }
                    }
                }
            }
        }

        for (int i = 0; i < graph.nodes; ++i)
        {
            bc_values[i] *= 0.5f;
        }

        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();

        printf("CPU BC finished in %lf msec. Search depth is:%d\n",
               elapsed, search_depth);

        //delete[] source_path;
    }
    free(coo);
}

/**
 * @brief Run betweenness centrality tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam INSTRUMENT
 *
 * @param[in] graph Reference to the CSR graph object defined in main driver
 * @param[in] src
 * @param[in] ref_filename 
 * @param[in] max_grid_size
 * @param[in] num_gpus
 * @param[in] max_queue_sizing
 */
template <
    typename VertexId,
    typename Value,
    typename SizeT,
    bool INSTRUMENT,
    bool DEBUG,
    bool SIZE_CHECK>
void RunTests(Test_Parameter *parameter)
{
    typedef BCProblem<
        VertexId,
        SizeT,
        Value,
        true,   // MARK_PREDECESSORS
        false> BcProblem; //does not use double buffer

    typedef BCEnactor<
        BcProblem,
        INSTRUMENT,
        DEBUG,
        SIZE_CHECK>
    BcEnactor;

    Csr<VertexId, Value, SizeT>
                 *graph                 = (Csr<VertexId, Value, SizeT>*)parameter->graph;
    VertexId      src                   = (VertexId)parameter -> src;
    int           max_grid_size         = parameter -> max_grid_size;
    int           num_gpus              = parameter -> num_gpus;
    double        max_queue_sizing      = parameter -> max_queue_sizing;
    double        max_queue_sizing1     = parameter -> max_queue_sizing1;
    double        max_in_sizing         = parameter -> max_in_sizing;
    ContextPtr   *context               = (ContextPtr*)parameter -> context;
    std::string   partition_method      = parameter -> partition_method;
    int          *gpu_idx               = parameter -> gpu_idx;
    cudaStream_t *streams               = parameter -> streams;
    float         partition_factor      = parameter -> partition_factor;
    int           partition_seed        = parameter -> partition_seed;
    bool          g_quick               = parameter -> g_quick;
    bool          g_stream_from_host    = parameter -> g_stream_from_host;
    std::string   ref_filename          = parameter -> ref_filename;
    int           iterations            = parameter -> iterations;
    size_t       *org_size              = new size_t  [num_gpus];
    // Allocate host-side array (for both reference and gpu-computed results)
    Value        *reference_bc_values        = new Value   [graph->nodes];
    Value        *reference_ebc_values       = new Value   [graph->edges];
    Value        *reference_sigmas           = new Value   [graph->nodes];
    VertexId     *reference_labels           = new VertexId[graph->nodes];
    Value        *h_sigmas                   = new Value   [graph->nodes];
    Value        *h_bc_values                = new Value   [graph->nodes];
    Value        *h_ebc_values               = new Value   [graph->edges];
    VertexId     *h_labels                   = new VertexId[graph->nodes];
    Value        *reference_check_bc_values  = (g_quick)                ? NULL : reference_bc_values;
    Value        *reference_check_ebc_values = (g_quick || (src != -1)) ? NULL : reference_ebc_values;
    Value        *reference_check_sigmas     = (g_quick || (src == -1)) ? NULL : reference_sigmas;
    VertexId     *reference_check_labels     = (g_quick || (src == -1)) ? NULL : reference_labels;

    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        size_t dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&(org_size[gpu]),&dummy);
    }

    // Allocate BC enactor map
    BcEnactor* enactor = new BcEnactor(num_gpus, gpu_idx);
    
    // Allocate problem on GPU
    BcProblem *problem = new BcProblem;
    util::GRError(problem->Init(
            g_stream_from_host,
            graph,
            NULL,
            num_gpus,
            gpu_idx,
            partition_method,
            streams,
            max_queue_sizing,
            max_in_sizing,
            partition_factor,
            partition_seed), "BC Problem Initialization Failed", __FILE__, __LINE__);
    util::GRError(enactor->Init(context, problem, max_grid_size), "BC Enactor init failed", __FILE__, __LINE__);

    //
    // Compute reference CPU BC solution for source-distance
    //
    if (reference_check_bc_values != NULL) {
        if (ref_filename.empty()) {
            printf("compute ref value\n");
            RefCPUBC(
                    *graph,
                    reference_check_bc_values,
                    reference_check_ebc_values,
                    reference_check_sigmas,
                    reference_check_labels,
                    src);
            printf("\n");
        } else {
            std::ifstream fin;
            fin.open(ref_filename.c_str(), std::ios::binary);
            for ( int i = 0; i < graph->nodes; ++i )
            {
                fin.read(reinterpret_cast<char*>(&reference_check_bc_values[i]), sizeof(Value));
            }
            fin.close();
        }
    }

    double   avg_duty = 0.0;
    float    elapsed  = 0.0f;
    // Perform BC
    CpuTimer cpu_timer;
    VertexId start_src;
    VertexId end_src;

    if (src == -1)
    {
        start_src = 0;
        end_src = graph->nodes;
    }
    else
    {
        start_src = src;
        end_src = src+1;
    }

    for (int iter = 0; iter < iterations; ++iter)
    {
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            util::SetDevice(gpu_idx[gpu]);
            util::MemsetKernel<<<128,128>>>
                (problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE), (Value)0.0f, (int)(problem->sub_graphs[gpu].nodes));
        }
        util::GRError(problem->Reset(0, enactor->GetFrontierType(), max_queue_sizing, max_queue_sizing1), "BC Problem Data Reset Fa    iled", __FILE__, __LINE__);

        printf("__________________________\n");fflush(stdout);
        cpu_timer.Start();
        for (VertexId i = start_src; i < end_src; ++i)
        {
            util::GRError(problem->Reset(i, enactor->GetFrontierType(), max_queue_sizing, max_queue_sizing1), "BC Problem Data Reset Failed", __FILE__, __LINE__);
            util::GRError(enactor ->Reset(), "BC Enactor Reset failed", __FILE__, __LINE__);
            util::GRError(enactor ->Enact(i), "BC Problem Enact Failed", __FILE__, __LINE__);
        }
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            util::SetDevice(gpu_idx[gpu]);
            util::MemsetScaleKernel<<<128, 128>>>
               (problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE), (Value)0.5f, (int)(problem->sub_graphs[gpu].nodes));
        }
        cpu_timer.Stop();
        printf("--------------------------\n");fflush(stdout);
        elapsed += cpu_timer.ElapsedMillis();
    }

    elapsed /= iterations;
    long long total_queued;
    enactor->GetStatistics(total_queued, avg_duty);

    // Copy out results
    util::GRError(problem->Extract(h_sigmas, h_bc_values, h_ebc_values, h_labels), "BC Problem Data Extraction Failed", __FILE__, __LINE__);
    /*printf("edge bc values: %d\n", graph.edges);
    for (int i = 0; i < graph.edges; ++i) {
        printf("%5f, %5f\n", h_ebc_values[i], reference_check_ebc_values[i]);
    }
    printf("edge bc values end\n");*/

    /*std::queue<VertexId> temp_queue;
    int *temp_marker=new int[graph->nodes];
    memset(temp_marker, 0, sizeof(int)*graph->nodes);
    temp_queue.push(41107);
    temp_queue.push(41109);
    cout<<"parent\tchild\tlabel\tsigma\tbc_value\t\tlabel\tsigma\tbc_value"<<endl;
    while (!temp_queue.empty())
    {
        VertexId parent = temp_queue.front();
        temp_queue.pop();
        temp_marker[parent]=1;
        int      gpu     = problem->partition_tables[0][parent];
        VertexId parent_ = problem->convertion_tables[0][parent];
        util::SetDevice(gpu_idx[gpu]);
        for (int i=graph->row_offsets[parent];i<graph->row_offsets[parent+1];i++)
        {
            VertexId child = graph->column_indices[i];
            VertexId child_ = 0;

            for (int j=problem->graph_slices[gpu]->row_offsets[parent_];
                     j<problem->graph_slices[gpu]->row_offsets[parent_+1];j++)
            {
                VertexId c=problem->graph_slices[gpu]->column_indices[j];
                if (problem->graph_slices[gpu]->original_vertex[c] == child)
                {
                    child_=c;break;
                }
            }
            //if (h_labels[child] != h_labels[parent]+1) continue;
            cout<<parent<<"\t "<<child<<"\t "<<h_labels[child]<<"\t "<<h_sigmas[child]<<"\t "<<h_bc_values[child]<<"\t";
            if (reference_check_labels[child] != h_labels[child] ||
                reference_check_sigmas[child] != h_sigmas[child] ||
                reference_check_bc_values[child] != h_bc_values[child])
            {
                cout<<"*";
                if (h_labels[child]==h_labels[parent]+1 && temp_marker[child]!=1) temp_queue.push(child);
            }
            cout<<"\t "<<reference_check_labels[child]<<"\t "<<reference_check_sigmas[child]<<"\t "<<reference_check_bc_values[child];
            cout<<"\t "<<gpu<<"\t "<<parent_<<"\t "<<child_;
            VertexId temp_label;
            Value    temp_sigma, temp_bc;
            cudaMemcpy((void*)&temp_label, problem->data_slices[gpu]->labels.GetPointer(util::DEVICE)+child_, sizeof(VertexId), cudaMemcpyDeviceToHost);
            cudaMemcpy((void*)&temp_sigma, problem->data_slices[gpu]->sigmas.GetPointer(util::DEVICE)+child_, sizeof(Value   ), cudaMemcpyDeviceToHost);
            cudaMemcpy((void*)&temp_bc, problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE)+child_, sizeof(Value), cudaMemcpyDeviceToHost);
            cout<<"\t "<<temp_label<<"\t "<<temp_sigma<<"\t "<<temp_bc;

            cudaMemcpy((void*)&temp_label, problem->data_slices[gpu]->labels.GetPointer(util::DEVICE)+parent_, sizeof(VertexId), cudaMemcpyDeviceToHost);
            cudaMemcpy((void*)&temp_sigma, problem->data_slices[gpu]->sigmas.GetPointer(util::DEVICE)+parent_, sizeof(Value   ), cudaMemcpyDeviceToHost);
            cudaMemcpy((void*)&temp_bc, problem->data_slices[gpu]->bc_values.GetPointer(util::DEVICE)+parent_, sizeof(Value), cudaMemcpyDeviceToHost);
            cout<<"\t "<<temp_label<<"\t "<<temp_sigma<<"\t "<<temp_bc<<endl;
        }
    }*/

    // Verify the result
    if (reference_check_bc_values != NULL) {
        //util::cpu_mt::PrintCPUArray<SizeT, Value>("reference_check_bc_values", reference_check_bc_values, graph->nodes);
        //util::cpu_mt::PrintCPUArray<SizeT, Value>("bc_values", h_bc_values, graph->nodes); 
        printf("Validity BC Value: ");
        int num_error = CompareResults(h_bc_values, reference_check_bc_values, graph->nodes,
                       true);
        if (num_error > 0)
            printf("Number of errors occurred: %d\n", num_error);
        printf("\n");
    }
    if (reference_check_ebc_values != NULL) {
        printf("Validity Edge BC Value: ");
        int num_error = CompareResults(h_ebc_values, reference_check_ebc_values, graph->edges,
                       true);
        if (num_error > 0)
            printf("Number of errors occurred: %d\n", num_error);
        printf("\n");
    }
    if (reference_check_sigmas != NULL) {
        printf("Validity Sigma: ");
        int num_error = CompareResults(h_sigmas, reference_check_sigmas, graph->nodes, true);
        if (num_error > 0)
            printf("Number of errors occurred: %d\n", num_error);
        printf("\n");
    }
    if (reference_check_labels != NULL) {
        printf("Validity labels: ");
        int num_error = CompareResults(h_labels, reference_check_labels, graph->nodes, true);
        if (num_error > 0)
            printf("Number of errors occurred: %d\n", num_error);
        printf("\n");
    }

    // Display Solution
    DisplaySolution(h_sigmas, h_bc_values, graph->nodes);

    printf("GPU BC finished in %lf msec.\n", elapsed);
    if (avg_duty != 0)
        printf("\n avg CTA duty: %.2f%% \n", avg_duty * 100);
    printf("Totaled number of edges visited: %lld = %lld *2\n", (long long) total_queued * 2, (long long) total_queued);

    printf("\n\tMemory Usage(B)\t");
    for (int gpu=0;gpu<num_gpus;gpu++)
    if (num_gpus>1) {if (gpu!=0) printf(" #keys%d,0\t #keys%d,1\t #ins%d,0\t #ins%d,1",gpu,gpu,gpu,gpu); else printf(" #keys%d,0\t #keys%d,1", gpu, gpu);}
    else printf(" #keys%d,0\t #keys%d,1", gpu, gpu);
    if (num_gpus>1) printf(" #keys%d",num_gpus);
    printf("\n");
    double max_queue_sizing_[2] = {0,0}, max_in_sizing_=0;
    for (int gpu=0;gpu<num_gpus;gpu++)
    {   
        size_t gpu_free,dummy;
        cudaSetDevice(gpu_idx[gpu]);
        cudaMemGetInfo(&gpu_free,&dummy);
        printf("GPU_%d\t %ld",gpu_idx[gpu],org_size[gpu]-gpu_free);
        for (int i=0;i<num_gpus;i++)
        {   
            for (int j=0; j<2; j++)
            {   
                SizeT x=problem->data_slices[gpu]->frontier_queues[i].keys[j].GetSize();
                printf("\t %lld", (long long) x); 
                double factor = 1.0*x/(num_gpus>1?problem->graph_slices[gpu]->in_counter[i]:problem->graph_slices[gpu]->nodes);
                if (factor > max_queue_sizing_[j]) max_queue_sizing_[j]=factor;
            }   
            if (num_gpus>1 && i!=0 )
            for (int t=0;t<2;t++)
            {   
                SizeT x=problem->data_slices[gpu][0].keys_in[t][i].GetSize();
                printf("\t %lld", (long long) x); 
                double factor = 1.0*x/problem->graph_slices[gpu]->in_counter[i];
                if (factor > max_in_sizing_) max_in_sizing_=factor;
            }   
        }   
        if (num_gpus>1) printf("\t %lld", (long long)(problem->data_slices[gpu]->frontier_queues[num_gpus].keys[0].GetSize()));
        printf("\n");
    }   
    printf("\t queue_sizing =\t %lf \t %lf", max_queue_sizing_[0], max_queue_sizing_[1]);
    if (num_gpus>1) printf("\t in_sizing =\t %lf", max_in_sizing_);
    printf("\n");

    // Cleanup
    if (org_size            ) {delete[] org_size            ; org_size             = NULL;}
    if (problem             ) {delete   problem             ; problem              = NULL;}
    if (enactor             ) {delete   enactor             ; enactor              = NULL;}
    if (reference_sigmas    ) {delete[] reference_sigmas    ; reference_sigmas     = NULL;}
    if (reference_bc_values ) {delete[] reference_bc_values ; reference_bc_values  = NULL;}
    if (reference_ebc_values) {delete[] reference_ebc_values; reference_ebc_values = NULL;}
    if (reference_labels    ) {delete[] reference_labels    ; reference_labels     = NULL;}
    if (h_sigmas            ) {delete[] h_sigmas            ; h_sigmas             = NULL;}
    if (h_bc_values         ) {delete[] h_bc_values         ; h_bc_values          = NULL;}
    if (h_ebc_values        ) {delete[] h_ebc_values        ; h_ebc_values         = NULL;}
    if (h_labels            ) {delete[] h_labels            ; h_labels             = NULL;}

    //cudaDeviceSynchronize();

}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT,
    bool          INSTRUMENT,
    bool          DEBUG>
void RunTests_size_check(Test_Parameter *parameter)
{
    if (parameter->size_check) RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        true > (parameter);
   else RunTests
        <VertexId, Value, SizeT, INSTRUMENT, DEBUG,
        false> (parameter);
}

template <
    typename    VertexId,
    typename    Value,
    typename    SizeT,
    bool        INSTRUMENT>
void RunTests_debug(Test_Parameter *parameter)
{
    if (parameter->debug) RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        true > (parameter);
    else RunTests_size_check
        <VertexId, Value, SizeT, INSTRUMENT,
        false> (parameter);
}

template <
    typename      VertexId,
    typename      Value,
    typename      SizeT>
void RunTests_instrumented(Test_Parameter *parameter)
{
    if (parameter->instrumented) RunTests_debug
        <VertexId, Value, SizeT,
        true > (parameter);
    else RunTests_debug
        <VertexId, Value, SizeT,
        false> (parameter);
}

template <
    typename VertexId,
    typename Value,
    typename SizeT>
void RunTests(
    Csr<VertexId, Value, SizeT> *graph,
    CommandLineArgs             &args,
    int                          num_gpus,
    ContextPtr                  *context,
    int                         *gpu_idx,
    cudaStream_t                *streams)
{
    std::string src_str = "";
    //std::string ref_filename = "";
    Test_Parameter *parameter = new Test_Parameter;

    parameter -> Init(args);
    parameter -> graph              = graph;
    parameter -> num_gpus           = num_gpus;
    parameter -> context            = context;
    parameter -> gpu_idx            = gpu_idx;
    parameter -> streams            = streams;

    args.GetCmdLineArgument("ref-file"        , parameter->ref_filename    );
    args.GetCmdLineArgument("src", src_str);
    if (src_str.empty()) {
        parameter->src = 0;
    } else if (src_str.compare("randomize") == 0) {
        parameter->src = graphio::RandomNode(graph->nodes);
    } else if (src_str.compare("largestdegree") == 0) {
        int temp;
        parameter->src = graph->GetNodeWithHighestDegree(temp);
    } else {
        args.GetCmdLineArgument("src", parameter->src);
    }   
    printf("src = %lld\n", parameter->src);

    RunTests_instrumented<VertexId, Value, SizeT>(parameter);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int cpp_main( int argc, char** argv)
{
    CommandLineArgs args(argc, argv);
    int          num_gpus = 0;
    int          *gpu_idx = NULL;
    ContextPtr   *context = NULL;
    cudaStream_t *streams = NULL;
    bool          g_undirected = false;

    if ((argc < 2) || (args.CheckCmdLineFlag("help"))) {
        Usage();
        return 1;
    }

    if (args.CheckCmdLineFlag  ("device"))
    {
        std::vector<int> gpus;
        args.GetCmdLineArguments<int>("device",gpus);
        num_gpus   = gpus.size();
        gpu_idx    = new int[num_gpus];
        for (int i=0;i<num_gpus;i++)
            gpu_idx[i] = gpus[i];
    } else {
        num_gpus   = 1;
        gpu_idx    = new int[num_gpus];
        gpu_idx[0] = 0;
    }
    streams  = new cudaStream_t[num_gpus * num_gpus * 2];
    context  = new ContextPtr  [num_gpus * num_gpus];
    printf("Using %d gpus: ", num_gpus);
    for (int gpu=0;gpu<num_gpus;gpu++)
    {
        printf(" %d ", gpu_idx[gpu]);
        util::SetDevice(gpu_idx[gpu]);
        for (int i=0;i<num_gpus*2;i++)
        {
            int _i = gpu*num_gpus*2+i;
            util::GRError(cudaStreamCreate(&streams[_i]), "cudaStreamCreate failed.", __FILE__, __LINE__);
            if (i<num_gpus) context[gpu*num_gpus+i] = mgpu::CreateCudaDeviceAttachStream(gpu_idx[gpu], streams[_i]);
        } 
    }
    printf("\n"); fflush(stdout);

    // Parse graph-contruction params
    g_undirected = true;
    std::string graph_type = argv[1];
    int flags = args.ParsedArgc();
    int graph_args = argc - flags - 1;
    if (graph_args < 1) {
        Usage();
        return 1;
    }

    //
    // Construct graph and perform search(es)
    //
    typedef int VertexId;  // Use as the node identifier type
    typedef float Value;   // Use as the value type
    typedef int SizeT;     // Use as the graph size type
    Csr<VertexId, Value, SizeT> csr(false); // default value for stream_from_host is false
    if (graph_args < 1) { Usage(); return 1; }

    if (graph_type == "market") {

        // Matrix-market coordinate-formatted graph file

        char *market_filename = (graph_args == 2 || graph_args == 3) ? argv[2] : NULL;
        if (graphio::BuildMarketGraph<false>(
                market_filename,
                csr,
                g_undirected,
                false) != 0)    //no inverse graph
        {
            return 1;
        }


    } else if (graph_type == "rmat")
    {   
        // parse rmat parameters
        SizeT rmat_nodes = 1 << 10; 
        SizeT rmat_edges = 1 << 10; 
        SizeT rmat_scale = 10; 
        SizeT rmat_edgefactor = 48; 
        double rmat_a = 0.57;
        double rmat_b = 0.19;
        double rmat_c = 0.19;
        double rmat_d = 1-(rmat_a+rmat_b+rmat_c);
        int    rmat_seed = -1; 

        args.GetCmdLineArgument("rmat_scale", rmat_scale);
        rmat_nodes = 1 << rmat_scale;
        args.GetCmdLineArgument("rmat_nodes", rmat_nodes);
        args.GetCmdLineArgument("rmat_edgefactor", rmat_edgefactor);
        rmat_edges = rmat_nodes * rmat_edgefactor;
        args.GetCmdLineArgument("rmat_edges", rmat_edges);
        args.GetCmdLineArgument("rmat_a", rmat_a);
        args.GetCmdLineArgument("rmat_b", rmat_b);
        args.GetCmdLineArgument("rmat_c", rmat_c);
        rmat_d = 1-(rmat_a+rmat_b+rmat_c);
        args.GetCmdLineArgument("rmat_d", rmat_d);
        args.GetCmdLineArgument("rmat_seed", rmat_seed);

        CpuTimer cpu_timer;
        cpu_timer.Start();
        if (graphio::BuildRmatGraph<false>(
                rmat_nodes,
                rmat_edges,
                csr,
                g_undirected,
                rmat_a,
                rmat_b,
                rmat_c,
                rmat_d,
                1,  
                1,  
                rmat_seed) != 0)
        {   
            return 1;
        }   
        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();
        printf("graph generated: %.3f ms, a = %.3f, b = %.3f, c = %.3f, d = %.3f\n", elapsed, rmat_a, rmat_b, rmat_c, rmat_d);
    } else if (graph_type == "rgg") {
       SizeT rgg_nodes = 1 << 10; 
        SizeT rgg_scale = 10; 
        double rgg_thfactor  = 0.55;
        double rgg_threshold = rgg_thfactor * sqrt(log(rgg_nodes) / rgg_nodes);
        double rgg_vmultipiler = 1;
        int    rgg_seed        = -1;

        args.GetCmdLineArgument("rgg_scale", rgg_scale);
        rgg_nodes = 1 << rgg_scale;
        args.GetCmdLineArgument("rgg_nodes", rgg_nodes);
        args.GetCmdLineArgument("rgg_thfactor", rgg_thfactor);
        rgg_threshold = rgg_thfactor * sqrt(log(rgg_nodes) / rgg_nodes);
        args.GetCmdLineArgument("rgg_threshold", rgg_threshold);
        args.GetCmdLineArgument("rgg_vmultipiler", rgg_vmultipiler);
        args.GetCmdLineArgument("rgg_seed", rgg_seed);

        CpuTimer cpu_timer;
        cpu_timer.Start();
        if (graphio::BuildRggGraph<false>(
            rgg_nodes,
            csr,
            rgg_threshold,
            g_undirected,
            rgg_vmultipiler,
            1,
            rgg_seed) !=0)
        {
            return 1;
        }
        cpu_timer.Stop();
        float elapsed = cpu_timer.ElapsedMillis();
        printf("graph generated: %.3f ms, threshold = %.3lf, vmultipiler = %.3lf\n", elapsed, rgg_threshold, rgg_vmultipiler);

    } else {

        // Unknown graph type
        fprintf(stderr, "Unspecified graph type\n");
        return 1;

    }

    csr.PrintHistogram();
    //csr.DisplayGraph();
    fflush(stdout);

    // Run tests
    RunTests(&csr, args, num_gpus, context, gpu_idx, streams);

    return 0;
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
