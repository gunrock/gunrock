// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * slabhash_graph_set.cuh
 *
 * @brief Dynamic graph using Concurrent Hash Sets
 */
#pragma once

#include <slab_hash.cuh>
#include <gunrock/graph/dynamic_graph/slabhash_graph_base.cuh>
#include <gunrock/graph/dynamic_graph/kernels/set/insert.cuh>

#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace graph {

/**
 * @brief HashGraphSet data structure to store an unweighted graph using
 * a per-vertex slab hash
 *
 * @tparam VertexT Vertex identifier type.
 * @tparam SizeT Graph size type.
 * @tparam ValueT Associated value type.
 * @tparam REQUIRE_VALUES whether the graph is weighted or not
 */
template<
    typename VertexT,
    typename SizeT,
    typename ValueT,
    bool REQUIRE_VALUES,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct HashGraphSet : HashGraphBase<VertexT, SizeT, ValueT, REQUIRE_VALUES> {


    template<typename PairT>
    cudaError_t InsertEdgesBatch(PairT* d_edges,
                                 SizeT batchSize){

        return cudaSuccess;
    }

    cudaError_t BulkBuildFromCsr(SizeT* d_row_offsets,
                                VertexT* d_col_indices,
                                bool is_directed_,
                                ValueT* d_node_values = nullptr){
        this->directed = is_directed_;
        return cudaSuccess;
        
    }

    cudaError_t ToCsr(SizeT* d_row_offsets,
                      VertexT* d_col_indices,
                      SizeT num_nodes_,
                      SizeT num_edges_){
        return cudaSuccess;
        
    }
};

} // namespace graph
} // namespace gunrock
