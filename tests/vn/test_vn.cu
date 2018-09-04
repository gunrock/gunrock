// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_vn.cu
 *
 * @brief Simple test driver program for single source shortest path.
 */

#include <gunrock/app/vn/vn_app.cu>
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
        typename ValueT>  // Use int as the value type
    cudaError_t operator()(util::Parameters &parameters,
        VertexT v, SizeT s, ValueT val)
    {
        typedef typename app::TestGraph<VertexT, SizeT, ValueT,
            graph::HAS_EDGE_VALUES | graph::HAS_CSR>
            GraphT;
        typedef typename GraphT::CsrT CsrT;

        cudaError_t retval = cudaSuccess;
        util::CpuTimer cpu_timer;
        GraphT graph; // graph we process on

        cpu_timer.Start();
        GUARD_CU(graphio::LoadGraph(parameters, graph));
        cpu_timer.Stop();
        parameters.Set("load-time", cpu_timer.ElapsedMillis());

        GUARD_CU(app::Set_Srcs    (parameters, graph));
        ValueT  *ref_distances = NULL;
        int num_srcs = 0;
        bool quick = parameters.Get<bool>("quick");
        // compute reference CPU vn solution for source-distance
        if (!quick)
        {
            bool quiet = parameters.Get<bool>("quiet");
            
            util::PrintMsg("Computing reference value ...", !quiet);
            std::vector<VertexT> srcs_vector
                = parameters.Get<std::vector<VertexT> >("srcs");
            num_srcs = srcs_vector.size();
            printf("srcs_vector | num_srcs=%d\n", num_srcs);
            
            VertexT *srcs = new VertexT[num_srcs];
            for(SizeT i = 0; i < num_srcs; ++i) {
                srcs[i] = srcs_vector[i];
                printf("test_vn srcs[i]=%d\n", srcs[i]);
            }
            
            SizeT nodes = graph.nodes;
            ref_distances = new ValueT[nodes];
            
            util::PrintMsg("__________________________", !quiet);
            float elapsed = app::vn::CPU_Reference(
                graph.csr(), ref_distances, NULL,
                srcs, num_srcs, quiet, false);
            
            std::string src_msg = "";
            for(SizeT i = 0; i < num_srcs; ++i) {
                src_msg += std::to_string(srcs[i]);
                if(i != num_srcs - 1) src_msg += ",";
            }
            util::PrintMsg("--------------------------\nRun 0 elapsed: "
                + std::to_string(elapsed) + " ms, srcs = "
                + src_msg, !quiet);
        }

        std::vector<std::string> switches{"mark-pred", "advance-mode"};
        GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
            [ref_distances](util::Parameters &parameters, GraphT &graph)
            {
                return app::vn::RunTests(parameters, graph, ref_distances);
            }));

        if (!quick)
        {
            delete[] ref_distances; ref_distances = NULL;
        }
        return retval;
    }
};

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test vn");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::vn::UseParameters(parameters));
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
