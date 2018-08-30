// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_SimpleTemplate.cu
 *
 * @brief Simple test driver program for Gunrock template.
 */

#include <gunrock/app/SimpleTemplate/SimpleTemplate_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

namespace APP_NAMESPACE = app::SimpleTemplate;

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
        // CLI parameters        
        bool quick = parameters.Get<bool>("quick");
        bool quiet = parameters.Get<bool>("quiet");

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

        // <todo> get srcs if needed, e.g.:
        // GUARD_CU(app::Set_Srcs (parameters, graph));
        // std::vector<VertexT> srcs
        //    = parameters.Get<std::vector<VertexT> >("srcs");
        // int num_srcs = srcs.size();
        // </todo>
        
        // <todo> declare datastructures for reference result on GPU
        ValueT *ref_degrees;
        // </todo>
        
        if (!quick) {
            // <todo> init datastructures for reference result on GPU
            ref_degrees = new ValueT[graph.nodes];
            // </todo>

            // If not in `quick` mode, compute CPU reference implementation
            util::PrintMsg("__________________________", !quiet);
            
            float elapsed = app::SimpleTemplate::CPU_Reference(
                graph.csr(),
                ref_degrees,
                quiet);
            
            util::PrintMsg("--------------------------\n Elapsed: "
                + std::to_string(elapsed), !quiet);
        }

        // <todo> add other switching parameters, if needed
        std::vector<std::string> switches{"advance-mode"};
        // </todo>
        
        GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
            [
                // </todo> pass necessary data to lambda
                ref_degrees
                // </todo>
            ](util::Parameters &parameters, GraphT &graph)
            {
                // <todo> pass necessary data to app::Template::RunTests
                return app::SimpleTemplate::RunTests(parameters, graph, ref_degrees, util::DEVICE);
                // </todo>
            }));

        // if (!quick)
        // {
        //     // <todo> deallocate host references
        //     delete[] ref_degrees; ref_degrees = NULL;
        //     // </todo>
        // }
        // return retval;
    }
};

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test SimpleTemplate");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::SimpleTemplate::UseParameters(parameters));
    GUARD_CU(app::UseParameters_test(parameters));
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }
    GUARD_CU(parameters.Check_Required());

    // TODO: change available graph types, according to requirements
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