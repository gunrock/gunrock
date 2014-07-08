// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cpu_graph_lib.hpp
 *
 * @brief library declarations of the CPU versions of the algorithms
 */
template<
    typename VertexId,
    typename Value,
    typename SizeT>
void RefCPUBC(
    SizeT                                   *row_offsets,
    VertexId                                *column_indices,
    Value                                   *bc_values,
    SizeT                                   num_nodes,
    VertexId                                src);

/**
 * @brief CPU-based reference CC algorithm using Boost Graph Library
 *
 * @param[in] row_offsets Pointer to ...
 * @param[in] column_indices Pointer to ...
 * @param[in] num_nodes
 * @param[in] labels Pointer to ...
 *
 * @returns Number of components of the input graph
 */
template<typename VertexId, typename SizeT>
unsigned int RefCPUCC(
    SizeT                                   *row_offsets,
    VertexId                                *column_indices,
    int                                     num_nodes,
    int                                     *labels);
