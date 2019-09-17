// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * utils.cuh
 *
 * @brief General graph-building utility routines
 */

#pragma once

// #define USE_STD_RANDOM          // undefine to use {s,d}rand48_r
#ifdef __APPLE__
#ifdef __clang__
#define USE_STD_RANDOM  // OS X/clang has no {s,d}rand48_r
#endif
#endif
#ifdef USE_STD_RANDOM
#include <random>
// this struct is a bit of a hack, but allows us to change as little
// code as possible in keeping {s,d}rand48_r capability as well as to
// use <random>
struct drand48_data {
  std::mt19937_64 engine;
  std::uniform_real_distribution<double> dist;
};
#endif

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <omp.h>

#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/random_bits.h>

//#include <gunrock/coo.cuh>
//#include <gunrock/csr.cuh>

namespace gunrock {
namespace graphio {

/**
 * @brief Generates a random node-ID in the range of [0, num_nodes)
 *
 * @param[in] num_nodes Number of nodes in Graph
 *
 * \return random node-ID
 */
template <typename SizeT>
SizeT RandomNode(SizeT num_nodes) {
  SizeT node_id;
  util::RandomBits(node_id);
  if (node_id < 0) node_id *= -1;
  return node_id % num_nodes;
}

/*template <typename VertexId, typename SizeT, typename Value>
void RemoveStandaloneNodes(
    Csr<VertexId, SizeT, Value>* graph, bool quiet = false)
{
    SizeT nodes = graph->nodes;
    SizeT edges = graph->edges;
    int *marker = new int[nodes];
    memset(marker, 0, sizeof(int) * nodes);
    VertexId *column_indices = graph->column_indices;
    SizeT    *row_offsets    = graph->row_offsets;
    SizeT    *displacements  = new SizeT   [graph->nodes];
    SizeT    *new_offsets    = new SizeT   [graph->nodes + 1];
    SizeT    *block_offsets  = NULL;
    VertexId *new_nodes      = new VertexId[graph->nodes];
    Value    *new_values     = new Value   [graph->nodes];
    Value    *values         = graph->node_values;
    int       num_threads    = 0;

    #pragma omp parallel
    {
        num_threads  = omp_get_num_threads();
        int thread_num   = omp_get_thread_num ();
        SizeT edge_start = (long long)(edges) * thread_num / num_threads;
        SizeT edge_end   = (long long)(edges) * (thread_num + 1) / num_threads;
        SizeT node_start = (long long)(nodes) * thread_num / num_threads;
        SizeT node_end   = (long long)(nodes) * (thread_num + 1) / num_threads;

        for (SizeT    edge = edge_start; edge < edge_end; edge++)
            marker[column_indices[edge]] = 1;
        for (VertexId node = node_start; node < node_end; node++)
            if (row_offsets[node] != row_offsets[node + 1])
                marker[node] = 1;
        if (thread_num == 0) block_offsets = new SizeT[num_threads + 1];
        #pragma omp barrier

        if (node_end > node_start) displacements[node_start] = 0;
        for (VertexId node = node_start; node < node_end - 1; node++)
            displacements[node + 1] = displacements[node] + 1 - marker[node];
        #pragma omp barrier
        if (node_end != 0)
            block_offsets[thread_num + 1] = displacements[node_end - 1] + 1 -
marker[node_end - 1]; else block_offsets[thread_num + 1] = 1 - marker[0];

        #pragma omp barrier
        #pragma omp single
        {
            block_offsets[0] = 0;
            for (int i = 0; i < num_threads; i++)
                block_offsets[i + 1] += block_offsets[i];
        }

        for (VertexId node = node_start; node < node_end; node++)
        {
            if (marker[node] == 0) continue;
            VertexId node_ = node - block_offsets[thread_num] -
displacements[node];
            //printf("thread_num = %d, node = %d, block_offsets[] = %d,
displacements[] = %d, node_ = %d\n",
            //    thread_num, node, block_offsets[thread_num],
displacements[node], node_); new_nodes  [node ] = node_; new_offsets[node_] =
row_offsets[node]; if (values != NULL) new_values[node_] = values[node];
        }

        //#pragma omp barrier
        //for (SizeT edge = edge_start; edge < edge_end; edge++)
        //{
        //    column_indices[edge] = new_nodes[column_indices[edge]];
        //}
    }

    for (SizeT edge = 0; edge < edges; edge++)
    {
        column_indices[edge] = new_nodes[column_indices[edge]];
    }

    nodes = nodes - block_offsets[num_threads];
    memcpy(row_offsets, new_offsets, sizeof(SizeT) * (nodes + 1));
    if (values != NULL) memcpy(values, new_values, sizeof(Value) * nodes);
    if (!quiet)
    {
        printf("graph #nodes : %lld -> %lld \n",
            (long long)graph->nodes, (long long)nodes);
    }
    graph->nodes = nodes;
    row_offsets[nodes] = graph->edges;

    delete[] new_offsets  ; new_offsets   = NULL;
    delete[] new_values   ; new_values    = NULL;
    delete[] new_nodes    ; new_nodes     = NULL;
    delete[] marker       ; marker        = NULL;
    delete[] displacements; displacements = NULL;
    delete[] block_offsets; block_offsets = NULL;
}*/

template <typename GraphT>
cudaError_t MakeUndirected(GraphT &directed_graph, GraphT &undirected_graph,
                           bool remove_duplicate_edges = false) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  const graph::GraphFlag FLAG = GraphT::FLAG;
  typedef graph::Coo<VertexT, SizeT, ValueT,
                     (FLAG & (~0x0F00)) | graph::HAS_COO>
      CooT;

  cudaError_t retval = cudaSuccess;
  CooT coo;
  GUARD_CU(
      coo.Allocate(directed_graph.nodes, directed_graph.edges * 2, util::HOST));

#pragma omp parallel for
  for (auto e = 0; e < directed_graph.edges; e++) {
    VertexT src, dest;
    directed_graph.GetEdgeSrcDest(e, src, dest);
    coo.edge_pairs[e * 2].x = src;
    coo.edge_pairs[e * 2].y = dest;
    coo.edge_pairs[e * 2 + 1].x = dest;
    coo.edge_pairs[e * 2 + 1].y = src;
    if (FLAG & graph::HAS_EDGE_VALUES) {
      ValueT val = directed_graph.edge_values[e];
      coo.edge_values[e * 2] = val;
      coo.edge_values[e * 2 + 1] = val;
    }
  }

  if (FLAG & graph::HAS_NODE_VALUES) {
    for (auto v = 0; v < directed_graph.nodes; v++)
      coo.node_values[v] = directed_graph.node_values[v];
  }
  if (remove_duplicate_edges) {
    GUARD_CU(
        coo.RemoveDuplicateEdges(graph::BY_ROW_ASCENDING, util::HOST, 0, true));
  }
  GUARD_CU(undirected_graph.FromCoo(coo, util::HOST, 0, false));
  GUARD_CU(coo.Release(util::HOST));
  return retval;
}

}  // namespace graphio
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
