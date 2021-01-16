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

#include <gunrock/util/array_utils.cuh>

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
 * @param[in]  memspace    Input and output target device, by default CPU
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
           const int      num_runs,
           gunrock::util::Location memspace = gunrock::util::HOST);

/*
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
          typename GValueT = unsigned int,
          typename SAGEValueT = GValueT>
double sage(const SizeT   num_nodes,
            const SizeT   num_edges,
            const SizeT   *row_offsets,
            const VertexT *col_indices,
            const GValueT *edge_values,
            const int     num_runs);

/*
 * @brief Simple interface take in graph as CSR format
 * @param[in]  num_nodes   Number of veritces in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  edge_values CSR-formatted graph input edge weights
 * @param[in]  num_runs    Number of runs to perform vn
 * @param[in]  memspace    Input and output target device, by default CPU
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
          const int     num_runs,
          gunrock::util::Location memspace = gunrock::util::HOST);

/*
 * @brief Subgraph Matching CXX interface
 *
 * @param[in]  num_nodes         Number of vertices in the input data graph
 * @param[in]  num_edges         Number of edges in the input data  graph
 * @param[in]  row_offsets       CSR-formatted data graph input row offsets
 * @param[in]  col_indices       CSR-formatted data graph input column indices
 * @param[in]  num_query_nodes   Number of vertices in the input query graph
 * @param[in]  num_query_edges   Number of edges in the input query graph
 * @param[in]  query_row_offsets CSR-formatted query graph input row offsets
 * @param[in]  query_col_indices CSR-formatted query graph input column indices
 * @param[in]  num_runs          Number of runs to perform SM
 * @param[out] subgraphs         Return number of subgraphs
 * @param[out] list_subgraphs    Return list of subgraphs
 * @param[in]  memspace          Location where inputs and outputs are stored
 * \return     double            Return accumulated elapsed times for all runs
 */
template <typename VertexT,
          typename SizeT>
double sm(
    const SizeT             num_nodes,
    const SizeT             num_edges,
    const SizeT            *row_offsets,
    const VertexT          *col_indices,
    const SizeT             num_query_nodes,
    const SizeT             num_query_edges,
    const SizeT            *query_row_offsets,
    const VertexT          *query_col_indices,
    const int               num_runs,
    unsigned long          *subgraphs,
    unsigned long         **list_subgraphs,
    gunrock::util::Location memspace = gunrock::util::HOST);

/**
 * @brief HITS_NORMALIZATION_METHOD Normalization method for the HITS algorithm
 */
enum HITS_NORMALIZATION_METHOD { // Integer
  HITS_NORMALIZATION_METHOD_1=1,  // 1-Norm (Sum of absolute values)
  HITS_NORMALIZATION_METHOD_2=2   // 2-Norm (Square root of the sum of squares)
};
 /*
 * @brief HITS simple public interface.
 *
 * @param[in]  num_nodes   Number of vertices in the input graph
 * @param[in]  num_edges   Number of edges in the input graph
 * @param[in]  row_offsets CSR-formatted graph input row offsets
 * @param[in]  col_indices CSR-formatted graph input column indices
 * @param[in]  max_iter    Maximum number of iterations to perform HITS
 * @param[in]  tol         Convergence tolerance for termination
 * @param[in]  hits_norm   Normalization method
 * @param[out] hub_ranks   Vertex hub scores
 * @param[out] auth ranks  Vertex authority scores
 * @param[in]  device      Target device to store inputs and outputs
 * \return     double      Elapsed run time in milliseconds
 */
template <
    typename VertexT,
    typename SizeT,
    typename GValueT>
double hits(
    const SizeT        num_nodes,
    const SizeT        num_edges,
    const SizeT       *row_offsets,
    const VertexT     *col_indices,
    const int          max_iter,
    const float        tol,
    const int          hits_norm,
    GValueT           *hub_ranks,
    GValueT           *auth_ranks,
    gunrock::util::Location memspace = gunrock::util::HOST);

// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
