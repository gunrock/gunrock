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
 * @brief Simple test driver program for single source shortest path.
 */

#include <stdio.h>
#include <iostream>

// Utilities and correctness-checking
#include <gunrock/util/test_utils.cuh>

// Graph definations
#include <gunrock/graphio/graphio.cuh>

// SSSP includes
#include <gunrock/app/sssp/sssp_enactor.cuh>

// Boost includes for CPU Dijkstra SSSP reference algorithms
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

using namespace gunrock;
using namespace gunrock::util;
using namespace gunrock::graph;
using namespace gunrock::app;
using namespace gunrock::app::sssp;

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * @brief Displays the SSSP result (i.e., distance from source)
 *
 * @tparam VertexId
 * @tparam SizeT
 *
 * @param[in] source_path Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template<typename VertexT, typename SizeT>
void DisplaySolution(VertexT *preds, SizeT num_vertices)
{
    if (num_vertices > 40)
        num_vertices = 40;

    printf("[");
    for (VertexT v = 0; v < num_vertices; ++v)
    {
        PrintValue(v);
        printf(":");
        PrintValue(preds[v]);
        printf(" ");
    }
    printf("]\n");
}

cudaError_t UseParameters_(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    GUARD_CU(graphio::UseParameters(parameters));
    //GUARD_CU(partitioner::UseParameters(parameters));
    GUARD_CU(app::sssp::UseParameters(parameters));
    GUARD_CU(app::sssp::UseParameters2(parameters));

    GUARD_CU(parameters.Use<int>(
        "num-runs",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        1,
        "Number of runs to perform the test, per parameter-set",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<std::string>(
        "src",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        "0",
        "<Vertex-ID|random|largestdegree> The source vertices\n"
        "\tIf random, randomly select non-zero degree vertices;\n"
        "\tIf largestdegree, select vertices with largest degrees",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<int>(
        "src-seed",
        util::REQUIRED_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
        util::PreDefinedValues<int>::InvalidValue,
        "seed to generate random sources",
        __FILE__, __LINE__));

    return retval;
}

/******************************************************************************
 * SSSP Testing Routines
 *****************************************************************************/

/**
 * @brief A simple CPU-based reference SSSP ranking implementation.
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] graph Reference to the CSR graph we process on
 * @param[in] node_values Host-side vector to store CPU computed labels for each node
 * @param[in] node_preds Host-side vector to store CPU computed predecessors for each node
 * @param[in] src Source node where SSSP starts
 * @param[in] quiet Don't print out anything to stdout
 */
template <
    typename GraphT>
void ReferenceSssp(
    const    GraphT          &graph,
    typename GraphT::ValueT  *distances,
    typename GraphT::VertexT *preds,
    typename GraphT::VertexT  src,
    bool                      quiet,
    bool                      mark_preds)
{
    using namespace boost;
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::CsrT    CsrT;

    // Prepare Boost Datatype and Data structure
    typedef adjacency_list<vecS, vecS, directedS, no_property,
            property <edge_weight_t, ValueT> > BGraphT;

    typedef typename graph_traits<BGraphT>::vertex_descriptor vertex_descriptor;
    typedef typename graph_traits<BGraphT>::edge_descriptor edge_descriptor;

    typedef std::pair<VertexT, VertexT> EdgeT;

    EdgeT   *edges = ( EdgeT*)malloc(sizeof( EdgeT) * graph.edges);
    ValueT *weight = (ValueT*)malloc(sizeof(ValueT) * graph.edges);

    for (VertexT v = 0; v < graph.nodes; ++v)
    {
        SizeT edge_start = graph.CsrT::GetNeighborListOffset(v);
        SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
        for (SizeT e = 0; e < num_neighbors; ++e)
        {
            edges [e + edge_start] = EdgeT(v, graph.CsrT::GetEdgeDest(e + edge_start));
            weight[e + edge_start] = graph.CsrT::edge_values[e + edge_start];
        }
    }

    BGraphT g(edges, edges + graph.edges, weight, graph.nodes);

    std::vector<ValueT>            d(graph.nodes);
    std::vector<vertex_descriptor> p(graph.nodes);
    vertex_descriptor s = vertex(src, g);

    typename property_map<BGraphT, vertex_index_t>::type
        indexmap = get(vertex_index, g);

    //
    // Perform SSSP
    //

    CpuTimer cpu_timer;
    cpu_timer.Start();

    if (mark_preds)
    {
        dijkstra_shortest_paths(g, s,
            predecessor_map(boost::make_iterator_property_map(
                p.begin(), get(boost::vertex_index, g))).distance_map(
                    boost::make_iterator_property_map(
                        d.begin(), get(boost::vertex_index, g))));
    }
    else
    {
        dijkstra_shortest_paths(g, s,
            distance_map(boost::make_iterator_property_map(
                d.begin(), get(boost::vertex_index, g))));
    }
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();

    if (!quiet)
        printf("CPU SSSP finished in %lf msec.\n", elapsed);

    typedef std::pair<VertexT, ValueT> PairT;
    PairT* sort_dist = new PairT[graph.nodes];
    typename graph_traits <BGraphT>::vertex_iterator vi, vend;
    for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
    {
        sort_dist[(*vi)].first  = (*vi);
        sort_dist[(*vi)].second = d[(*vi)];
    }
    std::stable_sort(
        sort_dist, sort_dist + graph.nodes,
        //RowFirstTupleCompare<Coo<Value, Value> >);
        [](const PairT &a, const PairT &b) -> bool
        {
            return a.first < b.first;
        });
    for (VertexT v = 0; v < graph.nodes; ++v)
        distances[v] = sort_dist[v].second;
    delete[] sort_dist; sort_dist = NULL;

    if (mark_preds)
    {
        typedef std::pair<VertexT, VertexT> VPairT;
        VPairT* sort_pred = new VPairT[graph.nodes];
        for (tie(vi, vend) = vertices(g); vi != vend; ++vi)
        {
            sort_pred[(*vi)].first  = (*vi);
            sort_pred[(*vi)].second = p[(*vi)];
        }
        std::stable_sort(
            sort_pred, sort_pred + graph.nodes,
            //RowFirstTupleCompare< Coo<VertexId, VertexId> >);
            [](const VPairT &a, const VPairT &b) -> bool
            {
                return a.first < b.first;
            });
        for (VertexT v = 0; v < graph.nodes; ++v)
            preds[v] = sort_pred[v].second;
        delete[] sort_pred; sort_pred = NULL;
    }
}

template <
    typename _VertexT = uint32_t,
    typename _SizeT   = _VertexT,
    typename _ValueT  = _VertexT,
    GraphFlag _FLAG   = GRAPH_NONE,
    unsigned int _cudaHostRegisterFlag = cudaHostRegisterDefault>
struct SSSPGraph :
    public Csr<_VertexT, _SizeT, _ValueT, _FLAG | HAS_CSR |/* HAS_COO | HAS_CSC |*/ HAS_GP, _cudaHostRegisterFlag>,
    //public Coo<_VertexT, _SizeT, _ValueT, _FLAG | HAS_CSR |/* HAS_COO | HAS_CSC |*/ HAS_GP, _cudaHostRegisterFlag>,
    //public Csc<_VertexT, _SizeT, _ValueT, _FLAG | HAS_CSR |/* HAS_COO | HAS_CSC |*/ HAS_GP, _cudaHostRegisterFlag>,
    public Gp <_VertexT, _SizeT, _ValueT, _FLAG | HAS_CSR |/* HAS_COO | HAS_CSC |*/ HAS_GP, _cudaHostRegisterFlag>
{
    typedef _VertexT VertexT;
    typedef _SizeT   SizeT;
    typedef _ValueT  ValueT;
    static const GraphFlag FLAG = _FLAG | HAS_CSR |/* HAS_COO | HAS_CSC |*/ HAS_GP;
    static const unsigned int cudaHostRegisterFlag = _cudaHostRegisterFlag;
    typedef Csr<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CsrT;
    //typedef Coo<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CooT;
    //typedef Csc<VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> CscT;
    typedef Gp <VertexT, SizeT, ValueT, FLAG, cudaHostRegisterFlag> GpT;

    SizeT nodes, edges;

    template <typename CooT_in>
    cudaError_t FromCoo(CooT_in &coo, bool self_coo = false)
    {
        cudaError_t retval = cudaSuccess;
        nodes = coo.CooT_in::CooT::nodes;
        edges = coo.CooT_in::CooT::edges;
        retval = this -> CsrT::FromCoo(coo);
        if (retval) return retval;
        //retval = this -> CscT::FromCoo(coo);
        //if (retval) return retval;
        //if (!self_coo)
        //    retval = this -> CooT::FromCoo(coo);
        return retval;
    }

    template <typename CsrT_in>
    cudaError_t FromCsr(CsrT_in &csr, bool self_csr = false)
    {
        cudaError_t retval = cudaSuccess;
        nodes = csr.CsrT::nodes;
        edges = csr.CsrT::edges;
        //retval = this -> CooT::FromCsr(csr);
        //if (retval) return retval;
        //retval = this -> CscT::FromCsr(csr);
        //if (retval) return retval;
        if (!self_csr)
            retval = this -> CsrT::FromCsr(csr);
        return retval;
    }

    cudaError_t Release(util::Location target = util::LOCATION_ALL)
    {
        cudaError_t retval = cudaSuccess;

        util::PrintMsg("GraphT::Realeasing on " +
            util::Location_to_string(target));
        //retval = this -> CooT::Release(target);
        //if (retval) return retval;
        retval = this -> CsrT::Release(target);
        if (retval) return retval;
        //retval = this -> CscT::Release(target);
        //if (retval) return retval;
        retval = this -> GpT::Release(target);
        if (retval) return retval;
        return retval;
    }

    CsrT &csr()
    {
        return (static_cast<CsrT*>(this))[0];
    }
};

/**
 * @brief Run SSSP tests
 *
 * @tparam VertexId
 * @tparam Value
 * @tparam SizeT
 * @tparam MARK_PREDECESSORS
 *
 * @param[in] info Pointer to info contains parameters and statistics.
 *
 * \return cudaError_t object which indicates the success of
 * all CUDA function calls.
 */
template <typename GraphT>
cudaError_t RunTests(
    Parameters       &parameters,
    GraphT           &graph,
    util::Location target = util::DEVICE)
{
    cudaError_t retval = cudaSuccess;

    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::CsrT    CsrT;
    typedef gunrock::app::sssp::Problem<GraphT  > ProblemT;
    typedef gunrock::app::sssp::Enactor<ProblemT> EnactorT;

    // parse configurations from parameters
    bool quiet_mode = parameters.Get<bool>("quiet");
    bool quick_mode = parameters.Get<bool>("quick");
    bool mark_pred  = parameters.Get<bool>("mark-pred");
    int  num_runs   = parameters.Get<int >("num-runs");
    std::vector<VertexT> srcs = parameters.Get<std::vector<VertexT>>("src");
    std::vector<int> gpu_idx = parameters.Get<std::vector<int>>("device");
    int  num_gpus   = gpu_idx.size();
    int  num_srcs   = srcs   .size();
    //util::Location target = util::DEVICE;

    CpuTimer    cpu_timer;
    cpu_timer.Start();

    // Allocate host-side array (for both reference and GPU-computed results)
    ValueT  *ref_distances = (quick_mode) ? NULL : new ValueT[graph.nodes];
    ValueT  *  h_distances = new ValueT[graph.nodes];
    VertexT *  h_preds  = (mark_pred ) ? new VertexT[graph.nodes] : NULL;
    VertexT *ref_preds  = (mark_pred && !quick_mode) ? new VertexT[graph.nodes] : NULL;

    size_t *org_size = new size_t[num_gpus];
    for (int gpu = 0; gpu < num_gpus; gpu++)
    {
        size_t dummy;
        GUARD_CU(util::SetDevice(gpu_idx[gpu]));
        GUARD_CU2(cudaMemGetInfo(&(org_size[gpu]), &dummy),
            "cudaMemGetInfo failed");
    }

    // Allocate problem and enactor on GPU, and initialize them
    ProblemT problem;
    EnactorT enactor;
    GUARD_CU(problem.Init(parameters, graph  , target));
    GUARD_CU(enactor.Init(parameters, problem, target));
    cpu_timer.Stop();
    //info -> info["preprocess_time"] = cpu_timer.ElapsedMillis();

    // perform SSSP
    double total_elapsed  = 0.0;
    double single_elapsed = 0.0;
    double max_elapsed    = 0.0;
    double min_elapsed    = 1e10;
    //json_spirit::mArray process_times;

    if (!quiet_mode)
    {
        printf("Using advance mode %s\n",
            parameters.Get<std::string>("advance-mode").c_str());
        printf("Using filter mode %s\n",
            parameters.Get<std::string>("filter-mode").c_str());
    }

    VertexT src;
    for (int run_num = 0; run_num < num_runs; ++run_num)
    {
        src = srcs[run_num % num_srcs];
        GUARD_CU(problem.Reset(src, target));
        GUARD_CU(enactor.Reset(src, target));
        for (int gpu = 0; gpu < num_gpus; gpu++)
        {
            GUARD_CU(util::SetDevice(gpu_idx[gpu]));
            GUARD_CU2(cudaDeviceSynchronize(),
                "cudaDeviceSynchronize failed");
        }
        if (!quiet_mode)
        {
            printf("__________________________\n"); fflush(stdout);
        }

        cpu_timer.Start();
        GUARD_CU(enactor.Enact(src));
        cpu_timer.Stop();
        single_elapsed = cpu_timer.ElapsedMillis();
        total_elapsed += single_elapsed;
        //process_times.push_back(single_elapsed);
        if (single_elapsed > max_elapsed) max_elapsed = single_elapsed;
        if (single_elapsed < min_elapsed) min_elapsed = single_elapsed;
        if (!quiet_mode)
        {
            printf("--------------------------\n"
                "Run %d elapsed: %lf ms, src = %lld, #iterations = %lld\n",
                run_num, single_elapsed, (long long)src,
                (long long)enactor.enactor_slices[0].enactor_stats.iteration);
            fflush(stdout);
        }
    }
    total_elapsed /= num_runs;
    //info -> info["process_times"] = process_times;
    //info -> info["min_process_time"] = min_elapsed;
    //info -> info["max_process_time"] = max_elapsed;

    // compute reference CPU SSSP solution for source-distance
    if (!quick_mode)
    {
        if (!quiet_mode)
        {
            printf("Computing reference value ...\n");
        }
        ReferenceSssp(
            graph, ref_distances, ref_preds,
            src, quiet_mode, mark_pred);
        if (!quiet_mode)
        {
            printf("\n");
        }
    }

    cpu_timer.Start();
    // Copy out results
    GUARD_CU(problem.Extract(h_distances, h_preds));

    if (!quick_mode)
    {
        for (VertexT v = 0; v < graph.nodes; v++)
        {
            if (!isValid(ref_distances[v]))
                ref_distances[v] = util::PreDefinedValues<ValueT>::MaxValue;
        }
    }

    if (!quiet_mode)
    {
        // Display Solution
        printf("\nFirst 40 distances of the GPU result.\n");
        DisplaySolution(h_distances, graph.nodes);
    }
    // Verify the result
    if (!quick_mode)
    {
        if (!quiet_mode)
            printf("Distance Validity: ");
        int error_num = CompareResults(
            h_distances, ref_distances,
            graph.nodes, true, quiet_mode);
        if (error_num > 0)
        {
            if (!quiet_mode)
                printf("%d errors occurred.\n", error_num);
        }
        if (!quiet_mode)
        {
            printf("\nFirst 40 distances of the reference CPU result.\n");
            DisplaySolution(ref_distances, graph.nodes);
        }
    }

    //info->ComputeTraversalStats(  // compute running statistics
    //    enactor.enactor_stats.GetPointer(), total_elapsed, h_distances);

    if (!quiet_mode)
    {
        if (mark_pred)
        {
            printf("Predecessors Validity: ");
            int num_errors = 0;
            for (VertexT v = 0; v < graph.nodes; v++)
            {
                VertexT pred          = h_preds[v];
                if (!util::isValid(pred) || v == src)
                    continue;
                ValueT  v_distance    = h_distances[v];
                if (v_distance == util::PreDefinedValues<ValueT>::MaxValue)
                    continue;
                ValueT  pred_distance = h_distances[pred];
                bool edge_found = false;
                SizeT edge_start = graph.CsrT::GetNeighborListOffset(pred);
                SizeT num_neighbors = graph.CsrT::GetNeighborListLength(pred);

                for (SizeT e = edge_start; e < edge_start + num_neighbors; e++)
                {
                    if (v == graph.CsrT::GetEdgeDest(e) &&
                        std::abs((pred_distance + graph.CsrT::edge_values[e]
                        - v_distance) * 1.0) < 1e-6)
                    {
                        edge_found = true;
                        break;
                    }
                }
                if (!edge_found)
                {
                    if (num_errors < 1)
                    {
                        printf("\nWRONG: [%lu] (", pred);
                        PrintValue<ValueT>(pred_distance);
                        printf(") -> [%lu] (", v);
                        PrintValue<ValueT>(v_distance);
                        printf(") can't find the corresponding edge.\n");
                    }
                    num_errors ++;
                }
            }
            if (num_errors > 0)
            {
                printf("%d errors occurred.\n", num_errors);
            } else {
                printf("\nCORRECT\n");
            }

            printf("First 40 preds of the GPU result.\n");
            DisplaySolution(h_preds, graph.nodes);
            if (ref_distances != NULL)
            {
                printf("First 40 preds of the reference CPU result (could be different because the paths are not unique).\n");
                DisplaySolution(ref_preds, graph.nodes);
            }
        }
    }

    if (!quiet_mode)
    {
        //Display_Memory_Usage(num_gpus, gpu_idx, org_size, problem);
        #ifdef ENABLE_PERFORMANCE_PROFILING
            //Display_Performance_Profiling(enactor);
        #endif
    }

    // Clean up
    delete[] org_size     ; org_size      = NULL;
    GUARD_CU(enactor.Release(target));
    GUARD_CU(problem.Release(target));
    delete[] ref_distances; ref_distances = NULL;
    delete[] h_distances  ; h_distances   = NULL;
    delete[] ref_preds    ; ref_preds     = NULL;
    delete[] h_preds      ; h_preds       = NULL;
    cpu_timer.Stop();
    //info->info["postprocess_time"] = cpu_timer.ElapsedMillis();
    return retval;
}

template <typename    GraphT>
cudaError_t RunTests_advance_mode(
    util::Parameters &parameters,
    GraphT           &graph)
{
    cudaError_t retval = cudaSuccess;
    std::vector<std::string> advance_modes
        = parameters.Get<std::vector<std::string>>("advance-mode");

    for (auto it = advance_modes.begin(); it != advance_modes.end(); it++)
    {
        std::string current_val = *it;
        GUARD_CU(parameters.Set("advance-mode", current_val));
        GUARD_CU(RunTests(parameters, graph));
    }

    GUARD_CU(parameters.Set("advance-mode", advance_modes));
    return retval;
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
template <typename    GraphT>
cudaError_t RunTests_mark_predecessors(
    util::Parameters &parameters,
    GraphT           &graph)
{
    cudaError_t retval = cudaSuccess;
    std::vector<bool> mark_preds
        = parameters.Get<std::vector<bool>>("mark-pred");

    for (auto it = mark_preds.begin(); it != mark_preds.end(); it++)
    {
        bool current_val = *it;
        GUARD_CU(parameters.Set("mark-pred", current_val));
        GUARD_CU(RunTests_advance_mode(parameters, graph));
    }

    GUARD_CU(parameters.Set("mark-pred", mark_preds));
    return retval;
}

/******************************************************************************
* Main
******************************************************************************/

template <
    typename VertexT,  // Use int as the vertex identifier
    typename SizeT,    // Use int as the graph size type
    typename ValueT>   // Use int as the value type
cudaError_t main_(util::Parameters &parameters)
{
    typedef SSSPGraph<VertexT, SizeT, ValueT, HAS_EDGE_VALUES> GraphT;
    typedef typename GraphT::CsrT CsrT;

    cudaError_t retval = cudaSuccess;
    CpuTimer cpu_timer, cpu_timer2;
    GraphT graph; // graph we process on

    cpu_timer.Start();
    //Info<VertexId, SizeT, Value> *info = new Info<VertexId, SizeT, Value>;
    cpu_timer2.Start();
    //info->Init("SSSP", *args, csr);  // initialize Info structure

    retval = graphio::LoadGraph(parameters, graph);
    if (retval) return retval;
    //util::cpu_mt::PrintCPUArray<typename GraphT::SizeT, typename GraphT::ValueT>(
    //    "edge_values", graph.GraphT::CsrT::edge_values + 0, graph.edges);

    // force edge values to be 1, don't enable this unless you really want to
    //for (SizeT e=0; e < graph.edges; e++)
    //    graph.CsrT::edge_values[e] = 1;
    cpu_timer2.Stop();
    //info->info["load_time"] = cpu_timer2.ElapsedMillis();
    std::string src = parameters.Get<std::string>("src");
    std::vector<VertexT> srcs;
    if (src == "random")
    {
        int src_seed = parameters.Get<int>("src-seed");
        int num_runs = parameters.Get<int>("num-runs");
        if (!isValid(src_seed))
        {
            src_seed = time(NULL);
            GUARD_CU(parameters.Set<int>("src-seed", src_seed));
        }
        if (!parameters.Get<bool>("quiet"))
            printf("src_seed = %d\n", src_seed);
        srand(src_seed);

        for (int i = 0; i < num_runs; i++)
        {
            bool src_valid = false;
            VertexT v;
            while (!src_valid)
            {
                v = rand() % graph.nodes;
                if (graph.CsrT::GetNeighborListLength(v) != 0)
                    src_valid = true;
            }
            srcs.push_back(v);
        }
        GUARD_CU(parameters.Set<std::vector<VertexT>>("src", srcs));
    }

    else if (src == "largestdegree")
    {
        SizeT largest_degree = 0;
        for (VertexT v = 0; v < graph.nodes; v++)
        {
            SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
            if (largest_degree < num_neighbors)
            {
                srcs.clear();
                srcs.push_back(v);
                largest_degree = num_neighbors;
            } else if (largest_degree == num_neighbors)
            {
                srcs.push_back(v);
            }
        }
        GUARD_CU(parameters.Set<std::vector<VertexT>>("src", srcs));
    }

    retval = RunTests_mark_predecessors(parameters, graph);  // run test
    cpu_timer.Stop();
    //info->info["total_time"] = cpu_timer.ElapsedMillis();

    if (!parameters.Get<bool>("quiet"))
    {
        //info->DisplayStats();  // display collected statistics
    }

    //info->CollectInfo();  // collected all the info and put into JSON mObject
    //delete info; info=NULL;
    GUARD_CU(parameters.Set("src", src));
    return retval;
}

template <
    typename VertexT,  // the vertex identifier type, usually int or long long
    typename SizeT   > // the size tyep, usually int or long long
cudaError_t main_ValueT(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    std::vector<bool> value_64b
        = parameters.Get<std::vector<bool>>("64bit-ValueT");

    for (auto it = value_64b.begin(); it != value_64b.end(); it++)
    {
        bool current_val = *it;
        GUARD_CU(parameters.Set("64bit-ValueT", current_val));

        if (current_val)
        {
            // Disabled becaus atomicMin(long long*, long long) is not available
            //retval = main_<VertexT, SizeT, uint64_t>(parameters);
            //if (retval) return retval;
        } else {
            retval = main_ <VertexT, SizeT, uint32_t>(parameters);
            if (retval) return retval;
        }
    }

    GUARD_CU(parameters.Set("64bit-ValueT", value_64b));
    return retval;
}

template <
    typename VertexT>
cudaError_t main_SizeT(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    std::vector<bool> size_64b
        = parameters.Get<std::vector<bool>>("64bit-SizeT");

    for (auto it = size_64b.begin(); it != size_64b.end(); it++)
    {
        bool current_val = *it;
        GUARD_CU(parameters.Set("64bit-SizeT", current_val));

        if (current_val)
        {
            // can be disabled to reduce compile time
            retval = main_ValueT <VertexT, uint64_t> (parameters);
            if (retval) return retval;
        } else {
            retval = main_ValueT <VertexT, uint32_t> (parameters);
            if (retval) return retval;
        }
    }

    GUARD_CU(parameters.Set("64bit-SizeT", size_64b));
    return retval;
}

cudaError_t main_VertexT(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;
    std::vector<bool> vertex_64b
        = parameters.Get<std::vector<bool>>("64bit-VertexT");

    for (auto it = vertex_64b.begin(); it != vertex_64b.end(); it++)
    {
        bool current_val = *it;
        GUARD_CU(parameters.Set("64bit-VertexT", current_val));

        if (current_val)
        {
            // disabled, because oprtr::filter::KernelPolicy::SmemStorage is too large for 64bit VertexT
            retval = main_SizeT<uint64_t>(parameters);
            if (retval) return retval;
        } else {
            retval = main_SizeT<uint32_t>(parameters);
            if (retval) return retval;
        }
    }

    GUARD_CU(parameters.Set("64bit-VertexT", vertex_64b));
    return retval;
}

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test sssp");
    GUARD_CU(UseParameters_(parameters));
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }
    GUARD_CU(parameters.Check_Required());

    return main_VertexT(parameters);
}
// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
