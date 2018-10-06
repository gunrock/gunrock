// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_hello.cu
 *
 * @brief Simple test driver program for Gunrock template.
 */

#include <gunrock/app/geo/geo_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

namespace APP_NAMESPACE = app::geo;

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

        // <TODO> get srcs if needed, e.g.:
        // GUARD_CU(app::Set_Srcs (parameters, graph));
        // std::vector<VertexT> srcs
        //    = parameters.Get<std::vector<VertexT> >("srcs");
        // int num_srcs = srcs.size();
        // </TODO>
        
        // <DONE> declare datastructures for reference result on GPU
        ValueT *ref_predicted_lat;
	ValueT *ref_predicted_lon;
        // </DONE>
        
        if (!quick) {
            // <DONE> init datastructures for reference result on GPU
            ref_predicted_lat = new ValueT[graph.nodes];
            ref_predicted_lon = new ValueT[graph.nodes];
            // </DONE>

            // If not in `quick` mode, compute CPU reference implementation
            util::PrintMsg("__________________________", !quiet);
            
            float elapsed = app::geo::CPU_Reference(
                graph.csr(),
                ref_predicted_lat,
		ref_predicted_lon,
                quiet);
            
            util::PrintMsg("--------------------------\n Elapsed: "
                + std::to_string(elapsed), !quiet);
        }

        // <TODO> add other switching parameters, if needed
        std::vector<std::string> switches{"advance-mode"};
        // </TODO>
        
        GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
            [
                // </DONE> pass necessary data to lambda
                ref_predicted_lat,
		ref_predicted_lon
                // </DONE>
            ](util::Parameters &parameters, GraphT &graph)
            {
                // <DONE> pass necessary data to app::Template::RunTests
                return app::geo::RunTests(parameters, graph, 
					  ref_predicted_lat, ref_predicted_lon,
					  util::DEVICE);
                // </DONE>
            }));

        if (!quick) {
            // <DONE> deallocate host references
            delete[] ref_predicted_lat; ref_predicted_lat = NULL;
            delete[] ref_predicted_lon; ref_predicted_lon = NULL;
            // </DONE>
        }
        return retval;
    }
};

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test geolocation");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::geo::UseParameters(parameters));
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
