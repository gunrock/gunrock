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

#if 0
#include <gunrock/util/basic_utils.h>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/type_enum.cuh>
#include <gunrock/util/type_limits.cuh>
#endif

#include <curand.h>
#include <curand_kernel.h>

#include <bits/stdc++.h>
#include <omp.h>

namespace gunrock {
namespace app {
namespace color {

/******************************************************************************
 * Color Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference color ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(util::Parameters &parameters, const GraphT &graph,
                     typename GraphT::VertexT *colors, bool quiet) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::VertexT VertexT;
  curandGenerator_t gen;

  auto seed = parameters.Get<int>("seed");
  
  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  // initialize cpu with same condition, use same variable names as on GPU
  memset(colors, -1, graph.nodes * sizeof(VertexT));

  util::Array1D<SizeT, float> rand;
  rand.Allocate(graph.nodes, util::HOST);
  curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniform(gen, rand.GetPointer(util::HOST), graph.nodes);

  bool colored = false;
  int iteration = 0;
  while (!colored) {
    for (VertexT v = 0; v < graph.nodes; v++) {
      if (colors[v] != -1) continue;
      SizeT start_edge = graph.GetNeighborListOffset(v);
      SizeT num_neighbors = graph.GetNeighborListLength(v);
      float temp = rand[v];
      bool colormax = true;
      bool colormin = true;
      for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
        VertexT u = graph.GetEdgeDest(e);
        if ((colors[u] == -1) && (rand[u] >= temp)) {
          colormax = false;
        }

        if ((colors[u] == -1) && (rand[u] <= temp)) {
          colormin = false;
        }
      }

      if (colormax) colors[v] = iteration * 2 + 1;
      if (colormin) colors[v] = iteration * 2 + 2;
    }

    for(VertexT v = 0; v < graph.nodes; v++) {
      if (colors[v] == -1) {
        break;
      } else if (v == graph.nodes - 1) {
        colored = true;
      } else {}
    }
    iteration++;
  }

  cpu_timer.Stop();
  float elapsed = cpu_timer.ElapsedMillis();
  return elapsed;
}

/**
 * @brief Validation of color results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(util::Parameters &parameters,
                                        GraphT &graph,
                                        typename GraphT::VertexT *h_colors,
                                        typename GraphT::VertexT *ref_colors,
                                        /*int *num_colors,*/ bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");
  bool quick = parameters.Get<bool>("quick");

  // validating result with cpu and check for conflict
  if (!quick) {
    util::PrintMsg("Validating result ...", !quiet);
    util::PrintMsg("Comparison: <node idx, gunrock, cpu>", !quiet);
    for (SizeT v = 0; v < graph.nodes; v++) {
      SizeT start_edge = graph.GetNeighborListOffset(v);
      SizeT num_neighbors = graph.GetNeighborListLength(v);

      for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
        VertexT u = graph.GetEdgeDest(e);
        if (h_colors[u] == h_colors[v] || h_colors[v] == -1) {
          num_errors += 1;
          util::PrintMsg("neighbor id  [ "\
                          + std::to_string(u) + " ], neighbor color [ " \
                          + std::to_string(h_colors[u]) + " ], my id [ " \
                          + std::to_string(v) + " ],  my color [ " \
                          + std::to_string(h_colors[v]) + " ]", !quiet);
        }
      }
    }
  }

  // // count number of colors
  // std::unordered_set<int> set;
  // for (SizeT v = 0; v < graph.nodes; v++) {
  //   int c = h_colors[v];
  //   if (set.find(c) == set.end()) {
  //     set.insert(c);
  //     (*num_colors)++;
  //   }
  // }

  if (num_errors == 0) {
    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
  }

  return num_errors;
}

}  // namespace color
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
