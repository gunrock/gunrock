// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * hash_graph_set.cuh
 *
 * @brief Dynamic graph using Concurrent Hash Maps
 */
#pragma once

#include <slab_hash.cuh>
#include <gunrock/graph/dynamic_graph/hash_graph_base.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace graph {

template<
    typename VertexT,
    typename SizeT,
    typename ValueT,
    GraphFlag FLAG,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct HashGraphMap : HashGraphBase<VertexT, SizeT, ValueT, FLAG> {


    cudaError_t InsertEdgesBatch(VertexT* src, 
                                 VertexT* dst,
                                 ValueT val,
                                 SizeT batchSize,
                                 util::Location target = util::DEVICE){

        return cudaSuccess;
    }
};

} // namespace graph
} // namespace gunrock
