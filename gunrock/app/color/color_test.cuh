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

#include <omp.h>

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
auto use_jpl  = parameters.Get<bool>("JPL");
auto no_conflict = parameters.Get<int>("no_conflict");

    util::CpuTimer cpu_timer;
    cpu_timer.Start();

    //initialize cpu with same condition, use same variable names as on GPU
    memset(colors, -1, graph.nodes * sizeof(VertexT));

    util::Array1D<SizeT, float> rand;
    rand.Allocate(graph.nodes,util::HOST);
    curandCreateGeneratorHost(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, rand.GetPointer(util::HOST), graph.nodes);

    if(use_jpl) {
      for (int iteration = 0; iteration < usr_iter; iteration++) {
        for (VertexT v = 0; v < graph.nodes; v++) {
          if(colors[v] != -1)
            continue;
          SizeT start_edge 	= graph.GetNeighborListOffset(v);
          SizeT num_neighbors = graph.GetNeighborListLength(v);
          float temp  = rand[v];
          bool colormax = true;
          bool colormin = true;
          for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
              VertexT u = graph.GetEdgeDest(e);
              if ((colors[u] == -1) && (rand[u] >= temp)) {
                  printf("Max: Node %d with %f defeated by node %d with %f\n",
                  v,rand[v],u,rand[u]);
                  colormax = false;
              }

              if ((colors[u] == -1) && (rand[u] <= temp)) {
                  printf("Min: Node %d with %f defeated by node %d with %f\n",
                  v,rand[v],u,rand[u]);
                  colormin = false;
              }
          }

          if(colormax)
            colors[v] = iteration*2+1;
          if(colormin)
            colors[v] = iteration*2+2;
        }
      }
    } else {
      for (int iteration = 0; iteration < usr_iter; iteration++) {
          for (VertexT v = 0; v < graph.nodes; v++) {
              SizeT start_edge 	= graph.GetNeighborListOffset(v);
              SizeT num_neighbors = graph.GetNeighborListLength(v);

              VertexT max = v;
              VertexT min = v;
              float temp  = rand[v];

              for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
                  VertexT u = graph.GetEdgeDest(e);
                  if (rand[u] > temp)
                      max = u;

                  if (rand[u] < temp)
                      min = u;

                  printf("Let's see what rand[u] = %f\n", rand[u]);
                  temp = rand[u];
              }

              if (colors[max] == -1)
                  colors[max] = iteration*2+1;

              if (colors[min] == -1)
                  colors[min] = iteration*2+2;

              printf("iteration number = %u\n", iteration);
              printf("colors[%u, %u] = [%u, %u]\n", min, max, colors[min], colors[max]);
          }
      }
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
