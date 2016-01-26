
// ----------------------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------
/**
 * @file
 * track_utils.cuh
 *
 * @brief tracking utilities function
 */

#pragma once
#include <gunrock/csr.cuh>
#include <gunrock/util/io/modified_store.cuh>

namespace gunrock {
namespace util {

#define TO_TRACK false
#define NUM_TO_TRACK 5

template <typename VertexId>
static __device__ __host__ __inline__ bool to_track(
    int gpu_num, VertexId node)
{
    /*for BFS, market /data/gunrock_dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx --src=largestdegree --traversal-mode=1 --device=0,1 --queue-sizing=7.0 --queue-sizing1=8.0 --in-sizing=0.5 --partition-seed=1451953615 --v
    NUM_TO_TRACK = 5*/
    const VertexId node_to_track[NUM_TO_TRACK][3] = {
        { 541845,  271043, 2569951}, 
        { 569068,  284715, 2953294},
        {4016145, 2008346, 3872477},
        {  40641,   20374, 2555548},
        {  40885,   20494, 2579834}
    };

    if (!TO_TRACK) return false;
    else {
        #pragma unroll
        for (int i=0; i<NUM_TO_TRACK; i++)
            //if (gpu_num == gpu_to_track[i] &&
            //    node == node_to_track[i]) 
            if (node_to_track[i][gpu_num+1] == node)
                return true;
    }
    return false;
}

template <typename VertexId>
static __device__ __host__ __inline__ bool to_track(
    VertexId node)
{
    return to_track(-1, node);
}

template <
    typename VertexId,
    typename Value,
    typename SizeT>
void Track_Results (
    const Csr<VertexId, Value, SizeT> *graph,
    int    num_gpus,
    Value  error_threshold,
    Value* results,
    Value* references,
    int*   partition_table,
    VertexId** convertion_tables)
{
    if (references == NULL) return;
    if (TO_TRACK)
    {
        for (VertexId v=0; v<graph->nodes; v++)
        {
            if (!to_track(-1, v)) continue;
            printf("Vertex %d, ", v);
            if (fabs(results[v] - references[v]) > error_threshold)
                printf("reference = %d, ", references[v]);
            printf("result = %d, ", results[v]);
            if (num_gpus > 1)
            {
                printf("host = %d, v_ = ", partition_table[v]);
                for (int gpu=0; gpu<num_gpus; gpu++)
                    printf("%d%s", convertion_tables[gpu][v], gpu == num_gpus-1? "" : ", ");
            }
            printf("\n");
            for (SizeT j = graph->row_offsets[v]; j < graph->row_offsets[v+1]; j++)
            {
                VertexId u = graph -> column_indices[j];
                if (references[u] != references[v] -1) continue; // bfs
                printf("\t%d, ", u);
                if (fabs(results[u] - references[u]) > error_threshold)
                    printf("reference = %d, ", references[u]);
                printf("result = %d, ", results[u]);
                if (num_gpus > 1)
                {
                    printf("host = %d, u_ = ", partition_table[u]);
                    for (int gpu=0; gpu<num_gpus; gpu++)
                        printf("%d%s", convertion_tables[gpu][u], gpu == num_gpus -1 ? "" : ", ");
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

//Output errors
template <
    typename VertexId,
    typename Value,
    typename SizeT>
void Output_Errors (
    const char* file_name,
    SizeT  num_nodes,
    int    num_gpus,
    Value  error_threshold,
    Value* results,
    Value* references,
    int*   partition_table,
    VertexId** convertion_tables) 
{
    if (references == NULL) return;

    std::ofstream fout;
    printf("\nWriting errors into %s\n", file_name);
    fout.open(file_name);
    
    for (VertexId v=0; v<num_nodes; v++)
    {
        if (fabs(results[v] - references[v]) <= error_threshold) continue;
        fout<< v << "\t" << references[v] << "\t" << results[v];
        if (num_gpus > 1)
        {
            fout<< "\t" << partition_table[v];
            for (int gpu=0; gpu<num_gpus; gpu++)
                fout<< "\t" << convertion_tables[gpu][v];
        }
        fout<< std::endl;
    }
    fout.close();
}

template <typename VertexId, typename SizeT, typename Value>
__global__ void Check_Queue(
    const SizeT     num_elements,
    const int       gpu_num,
    const SizeT     num_nodes,
    const long long iteration,
    const VertexId* keys,
    const Value*    labels)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
        VertexId key = keys[x];
        if (key >= num_nodes || keys < 0)
            printf("%d\t %lld\t %s: x, key = %d, %d\n", gpu_num, iteration, __func__, x, key);
        else {
            Value label = labels[key];
            if ((label != iteration+1 && label != iteration)
              || label < 0)
            {
                printf("%d\t %lld\t %s: x, key, label = %d, %d, %d\n",
                    gpu_num, iteration, __func__, x, key, label);
            }
        }
        x += STRIDE;
    }
}

template <typename VertexId, typename SizeT, typename Value>
__global__ void Check_Range(
    const SizeT num_elements,
    const int   gpu_num,
    const long long iteration,
    const Value lower_limit,
    const Value upper_limit,
    const Value* values)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
        Value value = values[x];
        if (value > upper_limit || value < lower_limit)
        {
            printf("%d\t %lld\t %s: x = %d, %d not in (%d, %d)\n",
                gpu_num, iteration, __func__, x, value, lower_limit, upper_limit);
        }
        x += STRIDE;
    }
}

template <typename VertexId, typename SizeT>
__global__ void Check_Exist(
    const SizeT num_elements,
    const int   gpu_num,
    const int   check_num,
    const long long iteration,
    const VertexId* keys)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements)
    {
        VertexId key = keys[x];
        if (to_track(gpu_num, key))
            printf("%d\t %lld\t %s: [%d] presents at %d\n",
                gpu_num, iteration, __func__, key, check_num);
        x += STRIDE;
    }
}

template <typename VertexId, typename SizeT>
__global__ void Check_Exist_(
    const SizeT *num_elements,
    const int   gpu_num,
    const int   check_num,
    const long long iteration,
    const VertexId* keys)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;
    VertexId x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x < num_elements[0])
    {
        VertexId key = keys[x];
        if (to_track(gpu_num, key))
            printf("%d\t %lld\t %s: [%d] presents at %d\n",
                gpu_num, iteration, __func__, key, check_num);
        x += STRIDE;
    }
}

template <typename Type>
__global__ void Check_Value(
    const Type *value,
    const int   gpu_num,
    const int   check_num,
    const long long iteration)
{
    printf("%d\t %lld\t %s: %d at %d\n",
        gpu_num, iteration, __func__, value[0], check_num);
}

template <typename VertexId, typename SizeT, typename ProblemData>
static __device__ __forceinline__ void Store_d_out(
    VertexId  new_value, 
    VertexId *d_out, 
    int       checkpoint_num,
    SizeT     offset1,
    SizeT     offset2,
    int       gpu_idx,
    VertexId  queue_index)
{    
    if (!TO_TRACK)
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            new_value,
            d_out + offset1 + offset2); 
    else {
        VertexId old_value = atomicCAS(d_out + offset1 + offset2, -2, new_value);
        if (old_value != -2)
        {    
            printf("%d\t %d\t %d\t Storing conflict: [%d] -> %p + %lld, old_value = [%d], "
                "offset1 = %lld, offset2 = %lld, blockIdx.x = %d, threadIdx.x = %d\n",
                gpu_idx, queue_index, checkpoint_num, new_value, d_out, 
                (long long)offset1 + offset2, old_value, (long long)offset1, 
                (long long)offset2, blockIdx.x, threadIdx.x);
        }    
        if (util::to_track(gpu_idx, new_value))
        {    
            printf("%d\t %d\t %d\t Storing [%d] -> %p + %lld\n",
                gpu_idx, queue_index, checkpoint_num, new_value, 
                d_out, (long long)offset1 + offset2);
        }
    }    
} 

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
