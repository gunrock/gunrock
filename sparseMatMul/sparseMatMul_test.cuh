// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * gtc_test.cu
 *
 * @brief Test related functions for SSSP
 */

#pragma once

#include <random>
#include <chrono>

#ifdef BOOST_FOUND
// Boost includes for CPU Dijkstra SSSP reference algorithms
#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>
#else
#include <queue>
#include <vector>
#include <utility>
#endif

namespace gunrock {
namespace app {
namespace sparseMatMul {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/


/******************************************************************************
 * SSSP Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference SSSP implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   graph         Input graph
 * @param[out]  distances     Computed distances from the source to each vertex
 * @param[out]  preds         Computed predecessors for each vertex
 * @param[in]   src           The source vertex
 * @param[in]   quiet         Whether to print out anything to stdout
 * @param[in]   mark_preds    Whether to compute predecessor info
 * \return      double        Time taken for the SSSP
 */

void CPU_Reference(int *row_offsets, int *col_offsets, double *x_vals, const int n_rows,
                   double *w, const int input_dim, const int output_dim, double *out) {
  for (int i = 0; i < n_rows * output_dim; i++) out[i] = 0;
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n_rows; i++)
    for (int jj = row_offsets[i]; jj < row_offsets[i + 1]; jj++) {
      int j = col_offsets[jj];
#ifdef SIMD
#pragma omp simd
#endif
    for (int k = 0; k < output_dim; k++)
      out[i * output_dim + k] += x_vals[jj] * w[j * output_dim + k];
  }
}

void rand_weights(int in_size, int out_size, double *weights) {
  std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());
  float range = sqrt(6.0f / (in_size + out_size));
#pragma omp parallel for schedule(static)
  for(int i = 0; i < in_size * out_size; i++)
    weights[i] = (double(rng()) / rng.max() - 0.5) * range * 2;
}


}  // namespace sssp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
