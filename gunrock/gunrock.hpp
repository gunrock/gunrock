// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file gunrock.hpp
 *
 * @brief Main library header file. Defines public CXX interface.
 * The Gunrock public interface is a CXX interface to enable linking
 * with code written in other languages. While the internals of Gunrock
 * are not limited to CXX.
 */

#pragma once

/**
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  edge_values CSR-formatted graph input edge weights
 * @param[in]  num_runs    Number of runs to perform SSSP
 * @param[in]  sources     Sources to begin traverse, one for each run
 * @param[in]  mark_preds  Whether to output predecessor info
 * @param[out] distances   Return shortest distance to source per vertex
 * @param[out] preds       Return predecessors of each vertex
 * \return     double      Return accumulated elapsed times for all runs
 */
template <typename VertexT = int,
          typename SizeT = int,
          typename LabelT = VertexT>
double bfs(const SizeT    num_nodes,
           const SizeT    num_edges,
           const SizeT    *row_offsets,
           const VertexT  *col_indices,
           VertexT        *sources,
           const bool     mark_pred,
           const bool     direction_optimized,
           const bool     idempotence,
           LabelT         **labels,
           VertexT        **preds = NULL,
           const int      num_runs);

/**
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  edge_values CSR-formatted graph input edge weights
 * @param[in]  num_runs    Number of runs to perform vn
 * @param[in]  sources     Sources to begin traverse, one for each run
 * @param[in]  mark_preds  Whether to output predecessor info
 * @param[out] distances   Return shortest distance to source per vertex
 * @param[out] preds       Return predecessors of each vertex
 * \return     double      Return accumulated elapsed times for all runs
 */
template <typename VertexT = int,
          typename SizeT = int,
          typename GValueT = unsigned int,
          typename vnValueT = GValueT>
double vn(const SizeT   num_nodes,
          const SizeT   num_edges,
          const SizeT   *row_offsets,
          const VertexT *col_indices,
          const GValueT *edge_values,
          VertexT       *sources,
          const bool    mark_pred,
          vnValueT      *distances,
          VertexT       *preds = NULL,
          const int     num_runs);

// Application Includes
#include <gunrock/app/bfs/bfs_app.cuh>
#include <gunrock/app/vn/vn_app.cuh>

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
