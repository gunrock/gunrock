// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hits_test.cu
 *
 * @brief Test related functions for hits
 */

#pragma once

namespace gunrock {
namespace app {
// <TODO> change namespace
namespace hits {
// </TODO>


/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference hits ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(
    const GraphT &graph, 
    typename GraphT::ValueT *ref_hrank,
    typename GraphT::ValueT *ref_arank,
    typename GraphT::SizeT max_iter,
    bool quiet)
{
    typedef typename GraphT::SizeT SizeT;
    
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    
    // <TODO> 
    // implement CPU reference implementation
    // </TODO>
    
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    return elapsed;
}

/**
 * @brief Validation of hits results
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
             typename GraphT::ValueT *h_hrank,
             typename GraphT::ValueT *h_arank,
             typename GraphT::ValueT *ref_hrank,
             typename GraphT::ValueT *ref_arank,
             bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    // Result validation and display
    for(SizeT v = 0; v < graph.nodes; ++v) {
        printf("Node: %d, hrank: %f, arank: %f\n", v, h_hrank[v], h_arank[v]);
    }

    if(num_errors == 0) {
       util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    }

    return num_errors;
}

} // namespace Template
} // namespace hits
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
