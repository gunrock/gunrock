// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * color_test.cu
 *
 * @brief Test related functions for color
 */

#pragma once

namespace gunrock {
namespace app {
// <DONE> change namespace
namespace color {
// </DONE>


/******************************************************************************
 * Color Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference hello ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(
    const GraphT &graph,
    typename GraphT::VertexT *colors,
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
 * @brief Validation of hello results
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
             typename GraphT::VertexT *h_colors,
             typename GraphT::VertexT *ref_colors,
             bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    // <TODO> result validation and display
    for(SizeT v = 0; v < graph.nodes; ++v) {
        printf("%d %d %d\n", v, h_colors[v], ref_colors[v]);
    }
    // </TODO>

    if(num_errors == 0) {
       util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    }

    return num_errors;
}

} // namespace color
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
