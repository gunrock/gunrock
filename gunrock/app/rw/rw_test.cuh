// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * rw_test.cu
 *
 * @brief Test related functions for rw
 */

#pragma once

namespace gunrock {
namespace app {
namespace rw {


/******************************************************************************
 * TW Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference RW ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
 * @param[in]   walk_length   Length of random walks
 * @param[in]   walks         Array to store random walk in
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(
    const GraphT &graph,
    int walk_length,
    int walks_per_node,
    int walk_mode,
    typename GraphT::VertexT *walks,
    bool quiet)
{
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::SizeT VertexT;

    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    if(walk_mode == 0) { // Random
        // <TODO> How should we implement a CPU reference?  Doesn't really make sense
        // I think we should actually be implementing a "checker" in Validate_Results
        for(SizeT i = 0; i < graph.nodes * walk_length; ++i) {
            walks[i] = util::PreDefinedValues<VertexT>::InvalidValue;
        }
        // </TODO>
    } else if (walk_mode == 1) { // Max
        // <TODO> Could implement max walking
        // </TODO>
    }

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    return elapsed;
}

/**
 * @brief Validation of RW results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  walk_length         Random walk length
 * @param[in]  walks_per_node      Number of random walks per node
 * @param[in]  h_walks       GPU walks
 * @param[in]  ref_walks     CPU walks
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(
            util::Parameters         &parameters,
            GraphT                   &graph,
            int                       walk_length,
            int                       walks_per_node,
            typename GraphT::VertexT *h_walks,
            typename GraphT::VertexT *ref_walks,
            bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    if(!quiet) {
        printf("[[");
        for(SizeT v = 0; v < graph.nodes * walk_length * walks_per_node; ++v) {
            if((v > 0) && (v % walk_length == 0)) {
                printf("],\n[");
            }
            printf("%d, ", h_walks[v]);
        }
        printf("]]\n");
    }

    // if(num_errors == 0) {
    //    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    // }
    util::PrintMsg("-------- NO VALIDATION -----", !quiet);

    return num_errors;
}

} // namespace rw
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
