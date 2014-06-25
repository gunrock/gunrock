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
 * @brief Dispatch function to perform topk algorithm
 *
 * This is the dispatch routine which calls topk_run() with 
 * appropriate template parameters and arguments.
 *
 * @param[in] row_offsets input row_offsets array for original graph
 * @param[in] col_indices input col_indices array for original graph
 * @param[in] col_offsets input col_offsets array for reversed graph
 * @param[in] row_indices input row_indices array for reversed graph
 * @param[in] num_nodes   number of nodes, length of offsets
 * @param[in] num_edges   number of edges, length of indices
 * @param[in] top_nodes   number of top nodes
 * @param[in] data_type   input data type
 *
 */
void topk_dispatch(const void *row_offsets,
		   const void *col_indices,
		   const void *col_offsets,
		   const void *row_indices,
		   size_t num_nodes,
		   size_t num_edges,
		   size_t top_nodes,
		   const TODO data_type)
{
    // TODO: Add more combinations of data types
    // Currently we assume VertexId and SizeT always have same type
    // either both unsigned integers or unsigned long long integers.
    switch (VTXID_TYPE)	{
    case VTXID_UINT:
	switch (VALUE_TYPE) {
	case VALUE_INT:
	    topk_run<unsigned int, unsigned int, unsigned int>
		((const unsigned int*)row_offsets,
		 (const unsigned int*)col_indices,
		 (const unsigned int*)col_offsets,
		 (const unsigned int*)row_indices,
		 num_nodes, num_edges, top_nodes); 
	    break;
	    
	case VALUE_FLOAT:
	    topk_run<unsigned int, unsigned int, float>
		((const unsigned int*)row_offsets,
		 (const unsigned int*)col_indices,
		 (const unsigned int*)col_offsets,
		 (const unsigned int*)row_indices,
		 num_nodes, num_edges, top_nodes); 
	    break;
	    
	case VALUE_DOUBLE:
	    topk_run<unsigned int, unsigned int, double>
		((const unsigned int*)row_offsets,
		 (const unsigned int*)col_indices,
		 (const unsigned int*)col_offsets,
		 (const unsigned int*)row_indices,
		 num_nodes, num_edges, top_nodes);
	    break;
	}
    case VTXID_LONG:
	switch (VALUE_TYPE) {
	case VALUE_INT:
	    topk_run<unsigned long long int,
		     unsigned long long int, int>
		((const unsigned int*)row_offsets,
		 (const unsigned int*)col_indices,
		 (const unsigned int*)col_offsets,
		 (const unsigned int*)row_indices,
		 num_nodes, num_edges, top_nodes); 
	    break;
	    
	case VALUE_FLOAT:
	    topk_run<unsigned long long int, 
		     unsigned long long int, float>
		((const unsigned int*)row_offsets,
		 (const unsigned int*)col_indices,
		 (const unsigned int*)col_offsets,
		 (const unsigned int*)row_indices,
		 num_nodes, num_edges, top_nodes); 
	    break;
	    
	case VALUE_DOUBLE:
	    topk_run<unsigned long long int, 
		     unsigned long long int, double>
		((const unsigned int*)row_offsets,
		 (const unsigned int*)col_indices,
		 (const unsigned int*)col_offsets,
		 (const unsigned int*)row_indices,
		 num_nodes, num_edges, top_nodes);
	    break;
	}
    }
}

/**
 * @brief Performs a topk calculation find top k nodes that have
 * largest degree centrality. Consider both in and out degrees.
 *
 * @param[in] row_offsets input row_offsets array for original graph
 * @param[in] col_indices input col_indices array for original graph
 * @param[in] col_offsets input col_offsets array for reversed graph
 * @param[in] row_indices input row_indices array for reversed graph
 * @param[in] num_nodes   number of nodes, length of offsets
 * @param[in] num_edges   number of edges, length of indices
 * @param[in] top_nodes   number of top nodes
 *
 */
void gunrock_topk(const void *row_offsets,
		  const void *col_indices,
		  const void *col_offsets,
		  const void *row_indices,
		  size_t     num_nodes,
		  size_t     num_edges,
		  size_t     top_nodes)
{
    // get user defined datatype
    
    // call topk implementations
    topk_dispatch(row_offsets, col_indices, col_offsets, row_indices, 
		  num_nodes, num_edges, top_nodes, data_type);
}

// TODO: add other algorithms
	
	
// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
