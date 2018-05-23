// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * rw_functor.cuh
 *
 * @brief Device functions for rw problem.
 */

#pragma once
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/rw/rw_problem.cuh>
#include <stdio.h>
#include <math.h>
#include <cub/cub.cuh>



namespace gunrock {
namespace app {
namespace rw {

/**
 * @brief Structure contains device functions in rw graph traverse.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 * @tparam _LabelT     Vertex label type.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId >
struct RWFunctor {
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;


    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] v auxiliary value.
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        return true;
    }

    /**
     * @brief Vertex mapping apply function. calculate output frontier in rw problem
     *
     * @param[in] node Vertex identifier.
     * @param[out] d_data_slice Data slice object.
     * @param[in] nid Vertex index.
     * @param[in] label Vertex label value.
     * @param[in] input_pos Index in the input frontier
     * @param[in] output_pos Index in the output frontier
     *
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
    }
};


/**
 * @brief Multiply the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename T, typename D, typename SizeT>
__global__ void RandomNext(T *paths, T *num_neighbor, D *d_rand, T *d_row_offsets, T *d_col_indices,
                                        SizeT length, SizeT itr)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
	
        //this node : itr * length + idx,                   path[itr*length+idx]       -> node_id[idx]
        //result (next node) : (itr+1) * length +idx,       path[(itr+1)*length + idx] -> path[idx]

		
        SizeT curr = paths[itr*length+idx];
        if(curr != -1 && num_neighbor[curr] > 0){
	       SizeT offset = __float2int_ru(num_neighbor[curr] * d_rand[idx]) - 1;
           SizeT new_node = d_row_offsets[curr] + offset;
           paths[(itr+1)*length + idx] = d_col_indices[new_node];
	    }else{
	    paths[(itr+1)*length + idx] = -1;
	    }

    }
}; 

/**
 * @brief Multiply the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename T, typename D, typename SizeT>
__global__ void SortedRandomNext(T *paths,T *node_id, T *num_neighbor, D *d_rand, T *d_row_offsets, T *d_col_indices,
                                        SizeT length, SizeT itr)
{
    
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
    
        //this node : itr * length + idx,                   path[itr*length+idx]       -> node_id[idx]
        //result (next node) : (itr+1) * length +idx,       path[(itr+1)*length + idx] -> path[idx]
        
        SizeT curr = paths[itr*length+idx];
        if(curr != -1 && num_neighbor[curr] > 0){
            SizeT offset = __float2int_ru(num_neighbor[curr] * d_rand[idx]) - 1;
            SizeT new_node = d_row_offsets[curr] + offset;
            SizeT out_offset = node_id[idx];
            paths[(itr+1)*length + out_offset] = d_col_indices[new_node];
    }else{
            paths[(itr+1)*length + node_id[idx]] = -1;
    }

    }



};


/*
// Specialize BlockRadixSort, BlockLoad, and BlockStore for 128 threads 
owning 16 integer items each
-> each thread handle 16 data
*/

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(int *key, int *value)
{
    //using namespace cub;
     // Specialize BlockRadixSort, BlockLoad, and BlockStore for 128 threads 
     // each thread owning 4
      //integer items
     typedef cub::BlockRadixSort<int, BLOCK_THREADS, ITEMS_PER_THREAD, int>               BlockRadixSort;
     typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE>   BlockLoad;
     typedef cub::BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockStore;


 
     // Allocate shared memory
     __shared__ union {
         typename BlockRadixSort::TempStorage     sort;
         typename BlockLoad::TempStorage          load; 
         typename BlockStore::TempStorage         store; 


     } temp_storage; 

     int tile = BLOCK_THREADS * ITEMS_PER_THREAD;
     int block_offset = blockIdx.x * tile;      // OffsetT for this block's ment

     // Obtain a segment of 512 consecutive keys that are blocked across threads
     int thread_keys[ITEMS_PER_THREAD];
     int thread_values[ITEMS_PER_THREAD];
     //mem illegal access because pointer passed in.********
     // Load items into a blocked arrangement
     BlockLoad(temp_storage.load).Load(key + block_offset, thread_keys);
     BlockLoad(temp_storage.load).Load(value + block_offset, thread_values);

    __syncthreads();

     // Collectively sort the keys
        //BlockRadixSort(temp_storage.sort).Sort(thread_keys, thread_values);

    BlockRadixSort(temp_storage.sort).Sort(thread_keys, thread_values);
    __syncthreads();

     // Store the sorted segment 
    BlockStore(temp_storage.store).Store(key + block_offset, thread_keys);
    BlockStore(temp_storage.store).Store(value + block_offset, thread_values);
};






/**
 * @brief Multiply the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename T, typename D, typename SizeT>
__global__ void BlockRandomNext(T *paths, T *node_id, T *num_neighbor, D *d_rand, T *d_row_offsets, 
                               T *d_col_indices, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        //printf("col_ind[%d]: %d\n", idx, d_col_indices[idx]);
        //printf("node_id[%d]: %d\n", idx, node_id[idx]);
        //printf("num_neighbor[%d]: %d\n", idx, num_neighbor[idx]);
        //printf("paths[%d]: %d\n", idx, paths[idx]);

        //this node : itr * length + idx,                   path[itr*length+idx]       -> node_id[idx]
        //result (next node) : (itr+1) * length +idx,       path[(itr+1)*length + idx] -> path[idx]

        SizeT curr = paths[idx];//curr is node id
        if(curr != -1 && num_neighbor[curr] > 0){
            SizeT offset = __float2int_ru(num_neighbor[curr] * d_rand[idx]) - 1;
            SizeT new_node = d_row_offsets[curr] + offset;
            SizeT out_offset = node_id[idx];//original pos in the output paths
            paths[length + out_offset] = d_col_indices[new_node];
        }else{
            paths[length + node_id[idx]] = -1;
        }
    }
};


/*******************************/
/* CUB BLOCKSORT KERNEL SHARED */
/*******************************/
template <typename T, typename D, typename SizeT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void shared_BlockSortKernel(T *paths, T *node_id, T *num_neighbor, D *d_rand, T *d_row_offsets, T *d_col_indices,
                                        SizeT length, SizeT itr)
{
    // --- Shared memory allocation
    __shared__ int sharedMemoryArrayValues[BLOCK_THREADS * ITEMS_PER_THREAD];
    __shared__ int   sharedMemoryArrayKeys[BLOCK_THREADS * ITEMS_PER_THREAD];

    // --- Specialize BlockStore and BlockRadixSort collective types
    typedef cub::BlockRadixSort <int , BLOCK_THREADS, ITEMS_PER_THREAD, int>  BlockRadixSortT;

    // --- Allocate type-safe, repurposable shared memory for collectives
    __shared__ typename BlockRadixSortT::TempStorage temp_storage;

    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);

    // --- Load data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        sharedMemoryArrayValues[threadIdx.x * ITEMS_PER_THREAD + k] = block_offset + threadIdx.x * ITEMS_PER_THREAD + k;
        sharedMemoryArrayKeys[threadIdx.x * ITEMS_PER_THREAD + k]   = paths[block_offset + threadIdx.x * ITEMS_PER_THREAD + k];
    }
    __syncthreads();

    // --- Collectively sort the keys
    BlockRadixSortT(temp_storage).SortBlockedToStriped(*static_cast<int(*)[ITEMS_PER_THREAD]>(static_cast<void*>(sharedMemoryArrayKeys   + (threadIdx.x * ITEMS_PER_THREAD))),
                                                       *static_cast<int(*)[ITEMS_PER_THREAD]>(static_cast<void*>(sharedMemoryArrayValues + (threadIdx.x * ITEMS_PER_THREAD))));
    __syncthreads();
    /*

    // --- Write data to shared memory
    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        d_values_result[block_offset + threadIdx.x * ITEMS_PER_THREAD + k] = sharedMemoryArrayValues[threadIdx.x * ITEMS_PER_THREAD + k];
        d_keys_result  [block_offset + threadIdx.x * ITEMS_PER_THREAD + k] = sharedMemoryArrayKeys  [threadIdx.x * ITEMS_PER_THREAD + k];
    }

    __syncthreads();

    */

    for (int k = 0; k < ITEMS_PER_THREAD; k++) {
        int curr = sharedMemoryArrayKeys[threadIdx.x * ITEMS_PER_THREAD + k];
        SizeT out_index = sharedMemoryArrayValues[block_offset+threadIdx.x * ITEMS_PER_THREAD + k];




        if(curr != -1 && num_neighbor[curr] > 0){
                SizeT offset = __float2int_ru(num_neighbor[curr] * d_rand[block_offset + threadIdx.x * ITEMS_PER_THREAD + k]) - 1;
                SizeT new_node = d_row_offsets[curr] + offset;
                //SizeT out_offset = node_id[idx];
                paths[out_index] = d_col_indices[new_node];
        }else{
                paths[out_index] = -1;
        }   

        //sharedMemoryArrayValues[threadIdx.x * ITEMS_PER_THREAD + k] = block_offset + threadIdx.x * ITEMS_PER_THREAD + k;
        //sharedMemoryArrayKeys[threadIdx.x * ITEMS_PER_THREAD + k]   = paths[block_offset + threadIdx.x * ITEMS_PER_THREAD + k];
    }

    /*

    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
    
        //this node : itr * length + idx,                   path[itr*length+idx]       -> node_id[idx]
        //result (next node) : (itr+1) * length +idx,       path[(itr+1)*length + idx] -> path[idx]
        
        SizeT curr = paths[itr*length+idx];
        if(curr != -1 && num_neighbor[curr] > 0){
            SizeT offset = __float2int_ru(num_neighbor[curr] * d_rand[idx]) - 1;
            SizeT new_node = d_row_offsets[curr] + offset;
            SizeT out_offset = node_id[idx];
            paths[(itr+1)*length + out_offset] = d_col_indices[new_node];
    }else{
            paths[(itr+1)*length + node_id[idx]] = -1;
    }

    }
    */
}

/********/
/* MAIN */
/********/



/**
 * @brief Multiply the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 
template <typename T, typename D, typename SizeT, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void FusedBlockRandomNext(T *paths, T *node_id, T *num_neighbor, D *d_rand, T *d_row_offsets, 
                               T *d_col_indices, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        //printf("col_ind[%d]: %d\n", idx, d_col_indices[idx]);
        //printf("node_id[%d]: %d\n", idx, node_id[idx]);
        //printf("num_neighbor[%d]: %d\n", idx, num_neighbor[idx]);
        //printf("paths[%d]: %d\n", idx, paths[idx]);

        //this node : itr * length + idx,                   path[itr*length+idx]       -> node_id[idx]
        //result (next node) : (itr+1) * length +idx,       path[(itr+1)*length + idx] -> path[idx]

        SizeT curr = paths[idx];//curr is node id
        if(curr != -1 && num_neighbor[curr] > 0){
            SizeT offset = __float2int_ru(num_neighbor[curr] * d_rand[idx]) - 1;
            SizeT new_node = d_row_offsets[curr] + offset;
            SizeT out_offset = node_id[idx];//original pos in the output paths
            paths[length + out_offset] = d_col_indices[new_node];
        }else{
            paths[length + node_id[idx]] = -1;
        }
    }
};

*/






/*
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(int *d_in, int *d_out, int length)
{

    for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += 32)
    {
        //d_in[idx] = length-idx-1;
        printf("col_ind[%d]: %d\n", idx, d_in[idx]);

        printf("node_id[%d]: %d\n", idx, d_out[idx]);

    }

    // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
    typedef cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockLoadT;
    typedef cub::BlockStore<
        int, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockStoreT;
    typedef cub::BlockRadixSort<
        int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockRadixSortT;
    // Allocate type-safe, repurposable shared memory for collectives
    __shared__ union {
        typename BlockLoadT::TempStorage       load; 
        typename BlockStoreT::TempStorage      store; 
        typename BlockRadixSortT::TempStorage  sort;
    } temp_storage; 
    // Obtain this block's segment of consecutive keys (blocked across threads)
    int thread_keys[ITEMS_PER_THREAD];
    int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);      
    BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys);
    
    __syncthreads();    // Barrier for smem reuse
    // Collectively sort the keys
    BlockRadixSortT(temp_storage.sort).Sort(thread_keys);
    __syncthreads();    // Barrier for smem reuse
    // Store the sorted segment 
    BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys);
}
*/
/*
//combine two kernel
template <typename T, typename SizeT>
__global__ void MemsetAssignKernel(T *paths, T *d_row_offsets, T *d_col_indices, T *node_id, SizeT length)
{
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        SizeT new_node = d_row_offsets[node_id[idx]] + paths[idx];
        paths[idx] = d_col_indices[new_node];
    }
};
*/


} // rw
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
