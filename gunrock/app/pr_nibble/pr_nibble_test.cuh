// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * pr_nibble_test.cu
 *
 * @brief Test related functions for pr_nibble
 */

#pragma once

namespace gunrock {
namespace app {
// <DONE> change namespace
namespace pr_nibble {
// </DONE>


/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference pr_nibble ranking implementations
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
    typename GraphT::VertexT src,
    typename GraphT::ValueT *values,
    // </DONE>
    bool quiet)
{
    typedef typename GraphT::SizeT SizeT;
    
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    
    // <DONE> 
    // implement CPU reference implementation
    for(SizeT v = 0; v < graph.nodes; ++v) {
        values[v] = (double)(graph.row_offsets[v + 1] - graph.row_offsets[v]);
    }
    values[src] = -1;
    
    printf("CPU_Reference: NotImplemented\n");
    // </DONE>
    
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    return elapsed;
}

/**
 * @brief Validation of pr_nibble results
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
             typename GraphT::ValueT *h_degrees,
             typename GraphT::ValueT *ref_degrees,
             bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    // <TODO> result validation and display
    // for(SizeT v = 0; v < graph.nodes; ++v) {
    //     std::cout << "v: "              << v              << 
    //                 " h_degrees[v]: "   << h_degrees[v]   <<
    //                 " ref_degrees[v]: " << ref_degrees[v] << std::endl;
    // }
    // </TODO>

    if(num_errors == 0) {
       util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    }

    return num_errors;
}

} // namespace Template
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
