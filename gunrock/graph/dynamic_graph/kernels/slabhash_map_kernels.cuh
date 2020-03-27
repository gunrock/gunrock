// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file
 * slabhash_map_kernels.cuh
 *
 * @brief SlabHash graph Graph Data Structure map kernels
 */
#pragma once


#include <slab_hash.cuh>

namespace gunrock {
namespace graph {
namespace slabhash_map_kernels{
    template <typename PairT, typename ValueT, typename SizeT, typename ContextT>
    __device__ void insertWarpEdges(PairT& thread_edge,
                                    ValueT& thread_value, 
                                    ContextT*& hashContexts, 
                                    SizeT*& d_edges_per_vertex,
                                    SizeT*& d_edges_per_bucket,
                                    SizeT*& d_buckets_offset,
                                    uint32_t& laneId,
                                    bool to_insert,
                                    AllocatorContextT& local_allocator_ctx)
    {
        uint32_t dst_bucket;
        
        if(to_insert){
            dst_bucket = hashContexts[thread_edge.x].computeBucket(thread_edge.y);
        }

        uint32_t work_queue;
        while(work_queue = __ballot_sync(0xFFFFFFFF, to_insert)){
            uint32_t cur_lane = __ffs(work_queue) - 1;
            uint32_t cur_src = __shfl_sync(0xFFFFFFFF, thread_edge.x, cur_lane, 32);
            bool same_src = (cur_src == thread_edge.x) && to_insert;

            if(same_src){
                SizeT bucket_offset = d_buckets_offset[cur_src];
                atomicAdd(&d_edges_per_bucket[bucket_offset + dst_bucket], 1);
            }

            //bool tmp_same = same_src;
            to_insert &= !same_src;

            bool success = hashContexts[cur_src].insertPairUnique(same_src, laneId, thread_edge.y,
                                                    thread_value, dst_bucket, local_allocator_ctx);
            //if(tmp_same)
            //    printf("inserting (%i->%i)[%i] = %i\n", cur_src, thread_edge.y, thread_value, success);
            uint32_t added_count = __popc(__ballot_sync(0xFFFFFFFF, success)); 
            if(laneId == 0){
                atomicAdd(&d_edges_per_vertex[cur_src], added_count);
            }
        }
    }    
    template <typename PairT, typename ValueT, typename SizeT, typename ContextT>
    __global__ void InsertEdges(PairT* d_edges,
                                ValueT* d_values, 
                                ContextT* hashContexts, 
                                SizeT num_edges,
                                SizeT* d_edges_per_vertex,
                                SizeT* d_edges_per_bucket,
                                SizeT* d_buckets_offset,
                                bool make_batch_undirected)
    {
        uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t laneId = threadIdx.x & 0x1F;

        if ((tid - laneId) >= num_edges){
            return;
        }

        PairT thread_edge;
        ValueT thread_value;
        bool to_insert = false; 
        if (tid < num_edges){
            thread_edge = d_edges[tid];
            thread_value = d_values[tid];
            to_insert = (thread_edge.x != thread_edge.y);
        }
        
        AllocatorContextT local_allocator_ctx(hashContexts[0].getAllocatorContext());
        local_allocator_ctx.initAllocator(tid, laneId);

        insertWarpEdges(thread_edge,
                        thread_value, 
                        hashContexts, 
                        d_edges_per_vertex,
                        d_edges_per_bucket,
                        d_buckets_offset,
                        laneId,
                        to_insert,
                        local_allocator_ctx);

        if(make_batch_undirected){
            PairT reverse_edge = make_uint2(thread_edge.y, thread_edge.x);
            insertWarpEdges(reverse_edge,
                            thread_value, 
                            hashContexts, 
                            d_edges_per_vertex,
                            d_edges_per_bucket,
                            d_buckets_offset,
                            laneId,
                            to_insert,
                            local_allocator_ctx);
        }

    }


    template <typename VertexT, typename ValueT, typename SizeT, typename ContextT>
    __global__ void ToCsr(SizeT num_nodes,
                          ContextT* hashContexts, 
                          SizeT* d_node_edges_offset,
                          SizeT* d_row_offsets,
                          VertexT* d_col_indices,
                          ValueT* d_edge_values)
    {
        using SlabHashT = ConcurrentMapT<VertexT, VertexT>;

        uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t laneId = threadIdx.x & 0x1F;
        uint32_t vid = tid >> 5;

        if(vid >= num_nodes)
        	return;

        uint32_t num_buckets = hashContexts[vid].getNumBuckets();
        uint32_t cur_offset = 0;
        

        if(vid != 0)
            cur_offset =d_node_edges_offset[vid - 1];
        
        if(laneId == 0)
            d_row_offsets[vid] = cur_offset;

        if(vid == (num_nodes - 1))
        	d_row_offsets[vid + 1] = d_node_edges_offset[vid];

        uint lanemask;
        asm volatile( "mov.u32 %0, %%lanemask_lt;" : "=r"( lanemask ) );

        for(int i =0; i < num_buckets; i++){
        	uint32_t next = SlabHashT::A_INDEX_POINTER;
            do{
                uint32_t key = (next == SlabHashT::A_INDEX_POINTER)
                                 ? *(hashContexts[vid].getPointerFromBucket(i, laneId))
                                 : *(hashContexts[vid].getPointerFromSlab(next, laneId));
                
                next = __shfl_sync(0xFFFFFFFF, key, 31, 32);

                uint32_t val = __shfl_xor_sync(0xFFFFFFFF, key, 1);

                bool key_lane = !(laneId % 2) && (laneId < 30);

                key = key_lane ? key : EMPTY_KEY;
                val = key_lane ? val : EMPTY_KEY;

                bool is_valid = (key != EMPTY_KEY);
                uint32_t rank_bmp = __ballot_sync(0xFFFFFFFF, is_valid);
                uint32_t rank = __popc(rank_bmp & lanemask);
                uint32_t sum = __popc(rank_bmp);

                uint32_t lane_offset = cur_offset + rank;
                if(is_valid){
                    d_col_indices[lane_offset] = key;
                    d_edge_values[lane_offset] = val;
                }
                cur_offset+=sum;
                

            }while(next != SlabHashT::EMPTY_INDEX_POINTER);
        }
    }
}
}
}
