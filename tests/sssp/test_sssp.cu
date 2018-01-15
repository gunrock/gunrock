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

#include <gunrock/app/sssp/sssp_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

/******************************************************************************
* Main
******************************************************************************/

struct main_struct
{
    template <
        typename VertexT, // Use int as the vertex identifier
        typename SizeT,   // Use int as the graph size type
        typename ValueT>  // Use int as the value type
    cudaError_t operator()(util::Parameters &parameters,
        VertexT v, SizeT s, ValueT val)
    {
        typedef typename app::TestGraph<VertexT, SizeT, ValueT,
            graph::HAS_EDGE_VALUES | graph::HAS_CSR>
            GraphT;

        cudaError_t retval = cudaSuccess;
        util::CpuTimer cpu_timer;
        GraphT graph; // graph we process on

        cpu_timer.Start();
        GUARD_CU(graphio::LoadGraph(parameters, graph));
        // force edge values to be 1, don't enable this unless you really want to
        //for (SizeT e=0; e < graph.edges; e++)
        //    graph.CsrT::edge_values[e] = 1;
        cpu_timer.Stop();
        parameters.Set("load-time", cpu_timer.ElapsedMillis());

        GUARD_CU(app::Set_Srcs    (parameters, graph));
        // compute reference CPU SSSP solution for source-distance
        /*if (!quick_mode)
        {
            if (!quiet_mode)
            {
                printf("Computing reference value ...\n");
            }
            ReferenceSssp(
                graph, ref_distances, ref_preds,
                src, quiet_mode, mark_pred);
        }*/

        std::vector<std::string> switches{"mark-pred", "advance-mode"};
        GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
            [](util::Parameters &parameters, GraphT &graph)
            {
                return app::sssp::RunTests(parameters, graph);
            }));
        return retval;
    }
};

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test sssp");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::sssp::UseParameters(parameters));
    GUARD_CU(app::UseParameters_test(parameters));
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }
    GUARD_CU(parameters.Check_Required());

    return app::Switch_Types<
        app::VERTEXT_U32B | app::VERTEXT_U64B |
        app::SIZET_U32B | app::SIZET_U64B |
        app::VALUET_U32B | app::DIRECTED | app::UNDIRECTED>
        (parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
