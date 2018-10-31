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
 * @brief Simple test driver program for Gunrock template.
 */

#include <gunrock/app/ss/ss_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

/******************************************************************************
* Main
******************************************************************************/

/**
 * @brief Enclosure to the main function
 */
struct main_struct
{
    /**
     * @brief the actual main function, after type switching
     * @tparam VertexT    Type of vertex identifier
     * @tparam SizeT      Type of graph size, i.e. type of edge identifier
     * @tparam ValueT     Type of edge values
     * @param  parameters Command line parameters
     * @param  v,s,val    Place holders for type deduction
     * \return cudaError_t error message(s), if any
     */
    template <
        typename VertexT, // Use int as the vertex identifier
        typename SizeT,   // Use int as the graph size type
        typename ValueT>  // Use float as the value type
    cudaError_t operator()(util::Parameters &parameters,
        VertexT v, SizeT s, ValueT val)
    {
        typedef typename app::TestGraph<VertexT, SizeT, ValueT,
            graph::HAS_EDGE_VALUES | graph::HAS_CSR>
            GraphT;

        cudaError_t retval = cudaSuccess;
        util::CpuTimer cpu_timer;
        GraphT graph;

        cpu_timer.Start();
        GUARD_CU(graphio::LoadGraph(parameters, graph));
        cpu_timer.Stop();
        parameters.Set("load-time", cpu_timer.ElapsedMillis());

        bool quick   = parameters.Get<bool>("quick");
        bool quiet   = parameters.Get<bool>("quiet");
        int num_runs = parameters.Get<int>("num-runs");

        VertexT *ref_scan_stats = NULL;
        if (!quick) {
            SizeT nodes = graph.nodes;
            ref_scan_stats = new VertexT[nodes];

            // for(int i = 0; i < num_runs; i++) {
                util::PrintMsg("__________________________", !quiet);

                float elapsed = app::ss::CPU_Reference(
                    parameters,
                    graph.csr(),
                    ref_scan_stats
                );

                util::PrintMsg("__________________________\nRun "
                    + std::to_string(0) + " elapsed: "
                    + std::to_string(elapsed)
                    + " ms", !quiet);
            // }
        }
//        return retval;

        std::vector<std::string> switches{"advance-mode"};
        GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
            [](util::Parameters &parameters, GraphT &graph)
            {
                bool quiet = parameters.Get<bool>("quiet");
                //bool quick = parameters.Get<bool>("quick");
                int num_runs = parameters.Get<int>("omp-runs");
                util::PrintMsg("num_runs: " + std::to_string(num_runs));
                std::string validation = parameters.Get<std::string>("validation");
/*                if (num_runs > 0)
                {
                    VertexT *omp_communities = new VertexT[graph.nodes];
                    for (int i = 0; i < num_runs; i++)
                    {
                        util::PrintMsg("__________________________", !quiet);
                        float elapsed = app::ss::OMP_Reference(
                            parameters, graph.csr(), omp_communities);
                        util::PrintMsg("--------------------------", !quiet);

                        if (validation == "each")
                        {
                            util::PrintMsg("Run " + std::to_string(i) + " elapsed: "
                                + std::to_string(elapsed) + " ms", !quiet);

                            app::ss::Validate_Results(parameters, graph,
                                omp_communities, ref_scan_stats);
                        } else {
                            util::PrintMsg("Run " + std::to_string(i) + " elapsed: "
                                + std::to_string(elapsed) + " ms, q = "
                                + std::to_string(app::ss::Get_Modularity(
                                    graph, omp_communities)), !quiet);
                        }
                    }
                    if (validation == "last")
                        app::ss::Validate_Results(parameters, graph,
                            omp_communities, ref_scan_stats);

                    if (ref_scan_stats == NULL)
                        ref_scan_stats = omp_communities;
                    else
                    {
                        delete[] omp_communities; omp_communities = NULL;
                    }
                }*/

                return app::ss::RunTests(parameters, graph);
            }));

        if (ref_scan_stats != NULL)
        {
            delete[] ref_scan_stats; ref_scan_stats = NULL;
        }
        return retval;
    }
};

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test Scan Statistics");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::ss::UseParameters(parameters));
    GUARD_CU(app::UseParameters_test(parameters));
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }
    GUARD_CU(parameters.Check_Required());

    return app::Switch_Types<
        app::VERTEXT_U32B | //app::VERTEXT_U64B |
        app::SIZET_U32B | //app::SIZET_U64B |
        app::VALUET_F64B | app::UNDIRECTED | app::DIRECTED>
        (parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
