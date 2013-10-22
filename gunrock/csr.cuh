// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * csr.cuh
 *
 * @brief CSR (Column Sparse Row) Graph Data Structure
 */

#pragma once

#include <time.h>
#include <stdio.h>

#include <algorithm>

#include <gunrock/util/error_utils.cuh>

namespace gunrock {


/**
 * CSR sparse format graph
 */
template<typename VertexId, typename Value, typename SizeT>
struct Csr
{
	SizeT nodes;
	SizeT edges;
	
	SizeT 		*row_offsets;
	VertexId	*column_indices;
	Value		*edge_values;
    Value       *node_values;
	
	bool 		pinned;

	/**
	 * Constructor
	 */
	Csr(bool pinned = false)
	{
		nodes = 0;
		edges = 0;
		row_offsets = NULL;
		column_indices = NULL;
		edge_values = NULL;
        node_values = NULL;
		this->pinned = pinned;
	}

	template <bool LOAD_EDGE_VALUES, bool LOAD_NODE_VALUES>
	void FromScratch(SizeT nodes, SizeT edges)
	{
		this->nodes = nodes;
		this->edges = edges;

		if (pinned) {

			// Put our graph in pinned memory
			int flags = cudaHostAllocMapped;
			if (gunrock::util::GRError(cudaHostAlloc((void **)&row_offsets, sizeof(SizeT) * (nodes + 1), flags),
				"Csr cudaHostAlloc row_offsets failed", __FILE__, __LINE__)) exit(1);
			if (gunrock::util::GRError(cudaHostAlloc((void **)&column_indices, sizeof(VertexId) * edges, flags),
				"Csr cudaHostAlloc column_indices failed", __FILE__, __LINE__)) exit(1);

			if (LOAD_NODE_VALUES) {
				if (gunrock::util::GRError(cudaHostAlloc((void **)&node_values, sizeof(Value) * nodes, flags),
						"Csr cudaHostAlloc node_values failed", __FILE__, __LINE__)) exit(1);
			}

            if (LOAD_EDGE_VALUES) {
				if (gunrock::util::GRError(cudaHostAlloc((void **)&edge_values, sizeof(Value) * edges, flags),
						"Csr cudaHostAlloc edge_values failed", __FILE__, __LINE__)) exit(1);
			}

		} else {

			// Put our graph in regular memory
			row_offsets 		= (SizeT*) malloc(sizeof(SizeT) * (nodes + 1));
			column_indices 		= (VertexId*) malloc(sizeof(VertexId) * edges);
			node_values 		= (LOAD_NODE_VALUES) ? (Value*) malloc(sizeof(Value) * nodes) : NULL;
            edge_values         = (LOAD_EDGE_VALUES) ? (Value*) malloc(sizeof(Value) * edges) : NULL;
		}
	}


	/**
	 * Build CSR graph from sorted COO graph
	 */
	template <bool LOAD_EDGE_VALUES, typename Tuple>
	void FromCoo(
		Tuple *coo,
		SizeT coo_nodes,
		SizeT coo_edges,
		bool ordered_rows = false)
	{
		printf("  Converting %d vertices, %d directed edges (%s tuples) to CSR format... ",
			coo_nodes, coo_edges, ordered_rows ? "ordered" : "unordered");
		time_t mark1 = time(NULL);
		fflush(stdout);

		FromScratch<LOAD_EDGE_VALUES, false>(coo_nodes, coo_edges);
		
		// Sort COO by row
		if (!ordered_rows) {
			std::stable_sort(coo, coo + coo_edges, RowFirstTupleCompare<Tuple>);
		}

		VertexId prev_row = -1;
		for (SizeT edge = 0; edge < edges; edge++) {
			
			VertexId current_row = coo[edge].row;
			
			// Fill in rows up to and including the current row
			for (VertexId row = prev_row + 1; row <= current_row; row++) {
				row_offsets[row] = edge;
			}
			prev_row = current_row;
			
			column_indices[edge] = coo[edge].col;
			if (LOAD_EDGE_VALUES) {
				coo[edge].Val(edge_values[edge]);
			}
		}

		// Fill out any trailing edgeless nodes (and the end-of-list element)
		for (VertexId row = prev_row + 1; row <= nodes; row++) {
			row_offsets[row] = edges;
		}

		time_t mark2 = time(NULL);
		printf("Done converting (%ds).\n", (int) (mark2 - mark1));
		fflush(stdout);
	}

	/**
	 * Print log-histogram
	 */
	void PrintHistogram()
	{
		fflush(stdout);

		// Initialize
		int log_counts[32];
		for (int i = 0; i < 32; i++) {
			log_counts[i] = 0;
		}

		// Scan
		int max_log_length = -1;
		for (VertexId i = 0; i < nodes; i++) {

			SizeT length = row_offsets[i + 1] - row_offsets[i];

			int log_length = -1;
			while (length > 0) {
				length >>= 1;
				log_length++;
			}
			if (log_length > max_log_length) {
				max_log_length = log_length;
			}

			log_counts[log_length + 1]++;
		}
		printf("\nDegree Histogram (%lld vertices, %lld directed edges):\n", (long long) nodes, (long long) edges);
		for (int i = -1; i < max_log_length + 1; i++) {
			printf("\tDegree 2^%i: %d (%.2f%%)\n", i, log_counts[i + 1], (float) log_counts[i + 1] * 100.0 / nodes);
		}
		printf("\n");
		fflush(stdout);
	}


	/**
	 * Display CSR graph to console
	 */
	void DisplayGraph()
	{
		printf("Input Graph:\n");
		for (VertexId node = 0; node < nodes; node++) {
			util::PrintValue(node);
			printf(": ");
			for (SizeT edge = row_offsets[node]; edge < row_offsets[node + 1]; edge++) {
				util::PrintValue(column_indices[edge]);
				printf(", ");
			}
			printf("\n");
		}

	}

	/**
	 * Deallocates graph
	 */
	void Free()
	{
		if (row_offsets) {
			if (pinned) {
				gunrock::util::GRError(cudaFreeHost(row_offsets), "Csr cudaFreeHost row_offsets failed", __FILE__, __LINE__);
			} else {
				free(row_offsets);
			}
			row_offsets = NULL;
		}
		if (column_indices) {
			if (pinned) {
				gunrock::util::GRError(cudaFreeHost(column_indices), "Csr cudaFreeHost column_indices failed", __FILE__, __LINE__);
			} else {
				free(column_indices);
			}
			column_indices = NULL;
		}
		if (edge_values) { free (edge_values); edge_values = NULL; }
        if (node_values) { free (node_values); node_values = NULL; }

		nodes = 0;
		edges = 0;
	}
	
	/**
	 * Destructor
	 */
	~Csr()
	{
		Free();
	}
};


} // namespace gunrock
