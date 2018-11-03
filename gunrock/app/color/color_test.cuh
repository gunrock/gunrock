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

#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/type_limits.cuh>
#include <gunrock/util/type_enum.cuh>

#include <curand.h>
#include <curand_kernel.h>

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
    util::Parameters &parameters,
    const GraphT &graph,
    typename GraphT::VertexT *colors,
    bool quiet)
{
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::VertexT VertexT;
    curandGenerator_t 		gen;
auto usr_iter = parameters.Get<int>("usr_iter");
auto seed     = parameters.Get<int>("seed");

    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    //initialize cpu with same condition, use same variable names as on GPU
    memset(colors, -1, graph.nodes * sizeof(VertexT));

    util::Array1D<SizeT, float> rand;
    rand.Allocate(graph.nodes,util::HOST);
    curandCreateGeneratorHost(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, rand.GetPointer(util::HOST), graph.nodes);

    for(SizeT v = 0; v < graph.nodes; ++v) {
        // degrees[v] = graph.row_offsets[v + 1] - graph.row_offsets[v];
    }

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
    printf("Comparison: <node idx, gunrock, cpu>\n");
    for(SizeT v = 0; v < graph.nodes; ++v) {
        printf(" %d %d %d\n", v, h_colors[v], ref_colors[v]);
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
