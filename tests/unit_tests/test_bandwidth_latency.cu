// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * test_pr.cu
 *
 * @brief Simple test driver program for PageRank.
 */

#include <gunrock/app/pr/pr_app.cu>
#include <gunrock/app/test_base.cuh>

using namespace gunrock;

cudaError_t UseParameters(util::Parameters &parameters)
{
    cudaError_t retval = cudaSuccess;

    GUARD_CU(parameters.Use<int>(
        "num-elements",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        1024 * 1024,
        "number of elements per GPU to test on",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<int>(
        "for-size",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        1024 * 256,
        "number of operations to perform per repeat",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<int>(
        "num-repeats",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        100,
        "number of times to repeat the operations",
        __FILE__, __LINE__));

    GUARD_CU(parameters.Use<int>(
        "device",
        util::REQUIRED_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
        0,
        "the devices to run on",
        __FILE__, __LINE__));

    return retval;
}

// Test routines

template <typename GraphT>
cudaError_t Test_BWL(
    util::Parameters &parameters, GraphT &graph)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    
    cudaError_t retval = cudaSuccess;
    SizeT num_elements = parameters.template Get<SizeT>("num-elements");
    SizeT for_size     = parameters.template Get<SizeT>("for-size");
    SizeT num_repeats  = parameters.template Get<SizeT>("num-repeats");
    auto  devices      = parameters.template Get<std::vector<int>>("device"); 
    int   num_devices  = devices.size();
    cudaError_t *retvals = new cudaError_t[num_devices];

    util::PrintMsg("num_devices = " + std::to_string(num_devices));
    #pragma omp parallel num_threads(num_devices)
    { do {
        int thread_num = omp_get_thread_num();
        cudaError_t &retval = retvals[thread_num];
        util::PrintMsg("using device[" + std::to_string(thread_num)
            + "] " + std::to_string(devices[thread_num]));
    
        util::Array1D<SizeT, VertexT> elements;
        elements.SetName("elements[" + std::to_string(thread_num));
        retval = elements.Allocate(num_elements, util::DEVICE);
        if (retval) break;
        
        retval = elements.Release();
        if (retval) break;
    } while (false); }
    return retval;
}


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
    cudaError_t operator()(util::Parameters &parameters, VertexT v, SizeT s, ValueT val)
    {
        typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_COO> GraphT;
        cudaError_t retval = cudaSuccess;

        GraphT graph;
        std::vector<std::string> switches{"num-elements", "for-size", "num-repeats"};
        
        GUARD_CU(app::Switch_Parameters(parameters, graph, switches,
            [](util::Parameters &parameters, GraphT &graph)
            {
                return Test_BWL(parameters, graph);
            }));
        return retval;
    }
};

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    util::Parameters parameters("test pr");
    GUARD_CU(graphio::UseParameters(parameters));
    GUARD_CU(app::UseParameters_test(parameters));
    GUARD_CU(UseParameters(parameters));
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }
    GUARD_CU(parameters.Check_Required());

    return app::Switch_Types<
        app::VERTEXT_U32B | app::VERTEXT_U64B |
        app::SIZET_U32B | //app::SIZET_U64B |
        app::VALUET_F32B | //app::VALUET_F64B |
        app::DIRECTED | app::UNDIRECTED>
        (parameters, main_struct());
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
