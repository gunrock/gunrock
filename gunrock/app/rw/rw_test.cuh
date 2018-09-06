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
namespace rw {
// </DONE>


/******************************************************************************
 * Template Testing Routines
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
    // <DONE> add problem specific inputs and outputs 
    int walk_length,
    typename GraphT::VertexT *walks,
    // </DONE>
    bool quiet)
{
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::SizeT VertexT;
    
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    
    // <TODO> 
    for(SizeT i = 0; i < graph.nodes * walk_length; ++i) {
        walks[i] = util::PreDefinedValues<VertexT>::InvalidValue;
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
            // <DONE> add problem specific inputs and outputs 
            int                       walk_length,
            typename GraphT::VertexT *h_walks,
            typename GraphT::VertexT *ref_walks,
            // </DONE>
            bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    // <DONE> result validation and display
    printf("[[");
    for(SizeT v = 0; v < graph.nodes * walk_length; ++v) {
        if((v > 0) && (v % walk_length == 0)) {
            printf("],\n[");
        }
        printf("%d, ", h_walks[v]);
    }
    printf("]]\n");
    // </DONE>

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
