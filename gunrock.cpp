// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * gunrock.cpp
 *
 * @brief Main Library source file. Wrappers for public interface.
 * These wrappers call application-level operators.
 *
 */

#include "gunrock.h"

/**
 * @brief Performs a topk calculation find top k nodes that have
 * largest degree centrality. Consider both in and out degrees.
 *
 */
void gunrock_topk(const void *row_offsets_i,
		  const void *col_indices_i,
		  const void *row_offsets_j,
		  const void *col_indices_j,
		  size_t     num_nodes,
		  size_t     num_edges,
		  size_t     top_nodes)
{
    // get user defined datatype

    // call topk implementations
    topk_dispatch(row_offsets_i, col_indices_i, 
		  row_offsets_j, col_indices_j, 
		  num_nodes, num_edges, 
		  top_nodes, data_type);
    
}

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
