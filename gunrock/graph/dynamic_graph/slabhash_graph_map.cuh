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

#include <iostream>

#include <slab_hash.cuh>
#include <cub/cub.cuh>

#include <gunrock/graph/dynamic_graph/slabhash_graph_base.cuh>
#include <gunrock/graph/dynamic_graph/kernels/slabhash_map_kernels.cuh>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace graph {

template<
    typename VertexT,
    typename SizeT,
    typename ValueT,
    bool REQUIRE_VALUES,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
struct HashGraphMap : HashGraphBase<VertexT, SizeT, ValueT, REQUIRE_VALUES> {


    template<typename PairT>
    cudaError_t InsertEdgesBatch(PairT* d_edges, 
                                 ValueT* d_values,
                                 SizeT batch_size,
                                 bool double_batch_edges){
                                     
                                     
        const uint32_t block_size = 128;                    
        const uint32_t num_blocks = (batch_size + block_size - 1) / block_size;

        slabhash_map_kernels::InsertEdges<<<num_blocks, block_size>>>(d_edges,
                                                d_values,
                                                this->d_hash_context,
                                                batch_size,
                                                this->d_edges_per_node,
                                                this->d_edges_per_bucket,
                                                this->d_buckets_offset,
                                                double_batch_edges);
        return cudaSuccess;
    }

    cudaError_t BulkBuildFromCsr(SizeT* h_row_offsets,
                                VertexT* h_col_indices,
                                ValueT* h_edge_values,
                                SizeT num_nodes_,
                                bool is_directed_,
                                ValueT* h_node_values = nullptr){
        using PairT = uint2;                     
        SizeT num_edges_ = h_row_offsets[num_nodes_];
        PairT* d_edges_pairs;
        this->is_directed = is_directed_;
        this->Init(h_row_offsets,
                    num_nodes_,
                    num_edges_,
                    mapEdgesPerSlab,
                    globalLoadFactor,
                    h_col_indices,
                    d_edges_pairs);


        ValueT* d_edge_values;
        CHECK_ERROR(cudaMalloc((void**)&d_edge_values, sizeof(ValueT) * num_edges_));
        CHECK_ERROR(cudaMemcpy(d_edge_values, h_edge_values, sizeof(ValueT) * num_edges_, cudaMemcpyHostToDevice));


        InsertEdgesBatch(d_edges_pairs,
                         d_edge_values,
                         num_edges_,
                         false);

        CHECK_ERROR(cudaFree(d_edge_values));
        return cudaSuccess;

    }
    cudaError_t ToCsr(SizeT* d_row_offsets,
                      VertexT* d_col_indices,
                      ValueT* d_edge_values,
                      SizeT num_nodes_,
                      SizeT num_edges_,
                      ValueT* d_node_values){
        
        // Use CUB to determine offsets
        SizeT* d_node_edges_offset;
      	CHECK_ERROR(cudaMalloc((void**)&d_node_edges_offset, sizeof(SizeT) * num_nodes_));
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, this->d_edges_per_node, d_node_edges_offset, num_nodes_);
        CHECK_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, this->d_edges_per_node, d_node_edges_offset, num_nodes_);
        
        
        //SizeT num_edges_found = 0;
        //CHECK_ERROR(cudaMemcpy(&num_edges_found, &d_node_edges_offset[num_nodes_ - 1], sizeof(SizeT), cudaMemcpyDeviceToHost));
        //std::vector<SizeT> h_node_edges_offset(num_nodes_);
        //CHECK_ERROR(cudaMemcpy(h_node_edges_offset.data(), d_node_edges_offset, sizeof(SizeT) * num_nodes_, cudaMemcpyDeviceToHost));
        //for(auto nc : h_node_edges_offset)
        //	std::cout << nc << std::endl;
        //std::cout << "Dynamic graph has: " <<  num_edges_found << std::endl;
        //std::cout << "Expected: " <<  num_edges_ << std::endl;

        const uint32_t block_size = 128;                    
        const uint32_t num_blocks = (num_nodes_ * 32 + block_size - 1) / block_size;

        slabhash_map_kernels::ToCsr<<<num_blocks, block_size>>>(
                                                num_nodes_,
                                                this->d_hash_context,
                                                d_node_edges_offset,
                                                d_row_offsets,
                                                d_col_indices,
                                                d_edge_values);

    
        //std::vector<SizeT> h_node_edges_offset_res(num_nodes_ + 1);
        //CHECK_ERROR(cudaMemcpy(h_node_edges_offset_res.data(), d_row_offsets, sizeof(SizeT) * (num_nodes_ + 1), cudaMemcpyDeviceToHost));
        //for(auto nc : h_node_edges_offset_res)
        //	std::cout << nc << std::endl;

        return cudaSuccess;
        
    }


    static constexpr uint32_t mapEdgesPerSlab = 15;
    static constexpr float globalLoadFactor = 0.7;
};

} // namespace graph
} // namespace gunrock
