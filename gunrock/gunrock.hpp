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

#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <gunrock/util/array_utils.cuh>

/**
 * @brief Single-source shortest path simple public interface.
 * 
 * @tparam VertexT 
 * @tparam SizeT 
 * @tparam GValueT 
 * @tparam SSSPValueT 
 *
 * @param num_nodes     Input graph number of nodes
 * @param num_edges     Input graph number of edges
 * @param row_offsets   Input graph CSR row offsets array
 * @param col_indices   Input graph CSR column indices array
 * @param edge_values   Input graph's values on edges (float)
 * @param sources       Source vertext for SSSP algorithm
 * @param mark_pred     Whether to output predecessor info or not
 * @param distances     Return shortest distance to source per vertex
 * @param preds         Return predecessors of each vertex
 * @param num_runs      Number of runs to perform SSSP
 * @param memspace      Location of input and desired output
 * @param exec_policy   Location where app will be run
 * @return double       Return accumulated elapsed times for all runs
 */
template <typename VertexT = int, 
          typename SizeT = int,
          typename GValueT = float, 
          typename SSSPValueT = GValueT>
double sssp(const SizeT             num_nodes, 
            const SizeT             num_edges,
            const SizeT             *row_offsets, 
            const VertexT           *col_indices,
            const GValueT           *edge_values, 
            VertexT                 *sources,
            const bool              mark_pred, 
            SSSPValueT              **distances, 
            VertexT                 **preds = nullptr, 
            const int               num_runs = 1, 
            gunrock::util::Location memspace = gunrock::util::HOST,
            gunrock::util::Location exec_policy = gunrock::util::DEVICE);

/**
 * @brief Graph Coloring algorithm based on Jones-Plassmann Luby (JPL).
 * 
 * @tparam VertexT 
 * @tparam SizeT 
 *
 * @param num_nodes     Input graph number of nodes
 * @param num_edges     Input graph number of edges
 * @param row_offsets   Input graph CSR row offsets array
 * @param col_indices   Input graph CSR column indices array
 * @param colors        Return generated colors for each run
 * @param num_colors    Return number of colors generated for each run
 * @param num_runs      Number of runs to perform Color
 * @return double       Return accumulated elapsed times for all runs
 */
template <typename VertexT = int, 
          typename SizeT = int>
double color(const SizeT    num_nodes, 
             const SizeT    num_edges,
             const SizeT    *row_offsets, 
             const VertexT  *col_indices,
             int            **colors, 
             int            *num_colors,
             const int      num_runs = 1);

// Application Includes
#include <gunrock/app/sssp/sssp_app.cuh>
#include <gunrock/app/color/color_app.cuh>

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
