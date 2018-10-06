// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hello_test.cu
 *
 * @brief Test related functions for hello
 */

#pragma once

namespace gunrock {
namespace app {
// <DONE> change namespace
namespace geo {
// </DONE>


/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference geolocation implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(
    const GraphT &graph,
    // <DONE> add problem specific inputs and outputs 
    typename GraphT::ValueT *predicted_lat,
    typename GraphT::ValueT *predicted_lon,
    // </DONE>
    bool quiet)
{
    typedef typename GraphT::SizeT SizeT;
    
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    
    // <TODO> 
    // implement CPU reference implementation
    for(SizeT v = 0; v < graph.nodes; ++v) {
        // degrees[v] = graph.row_offsets[v + 1] - graph.row_offsets[v];
    }
    // </TODO>
    
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    return elapsed;
}

/**
 * @brief Validation of geolocation results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(
             util::Parameters &parameters,
             GraphT           &graph,
             typename GraphT::ValueT *h_predicted_lat,
             typename GraphT::ValueT *h_predicted_lon,
             typename GraphT::ValueT *ref_predicted_lat,
             typename GraphT::ValueT *ref_predicted_lon,
             bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    // <DONE> result validation and display
    for(SizeT v = 0; v < graph.nodes; ++v) {
        printf("Node [ %d ]: Predicted = < %d , %d > Reference = < %d , %d >\n", v, 
		h_predicted_lat[v], h_predicted_lon[v], 
		ref_predicted_lat[v], ref_predicted_lon[v]);
    }
    // </DONE>

    if(num_errors == 0) {
       util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    }

    return num_errors;
}

} // namespace geo
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
