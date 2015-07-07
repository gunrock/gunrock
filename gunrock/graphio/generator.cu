// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file generator.cu
 * @brief Gunrock random graph generators
 */

#include <gunrock/gunrock.h>
#include <gunrock/graphio/market.cuh>
#include <gunrock/graphio/rmat.cuh>

using namespace gunrock;

void rmat_graph(
    int rows[], int cols[], unsigned int vals[], int nodes, int edges,
    bool undirected = false, float a = 0.57, float b = 0.19, float c = 0.19) {
    Csr<int, unsigned int, int> graph;
    float d = 1.0f - a - b - c;
    graphio::BuildRmatGraph<true>(nodes, edges, graph, undirected, a, b, c, d);
    memcpy(rows, (int*)graph.row_offsets, nodes * sizeof(int));
    memcpy(cols, (int*)graph.column_indices, nodes * sizeof(int));
    memcpy(vals, (unsigned int*)graph.edge_values, edges*sizeof(unsigned int));
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
