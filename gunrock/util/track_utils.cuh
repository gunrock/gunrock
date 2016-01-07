
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
#define NUM_TO_TRACK 0

template <typename VertexId>
static __device__ __host__ __inline__ bool to_track(
    int gpu_num, VertexId node)
{
    /*for BFS, market /data/gunrock_dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx --src=largestdegree --traversal-mode=1 --device=0,1 --queue-sizing=7.0 --queue-sizing1=8.0 --in-sizing=0.5 --partition-seed=1451953615 --v
    NUM_TO_TRACK = 38
    const VertexId node_to_track[NUM_TO_TRACK][3] = {
        { 541845,  271043, 2569951}, 
        { 569068,  284715, 2953294},
        {4016145, 2008346, 3872477},
        {  40641,   20374, 2555548},
        {  40885,   20494, 2579834},

        {   1077,     518, 2441318},
        {   1421,     692, 2432176},
        {   1432, 2442039,     733},
        {   4494,    2201, 2432178},
        {   7327, 2424483,    3718},
        {  11142, 2424090,    5558},
        {  17218, 2442240,    8597},
        {  17649,    8828, 2445489},
        {  25287, 2442048,   12597},
        { 253814, 2623718,  126782},
        {2590170, 2479765, 1294485},

        {  19137, 2463137,    9576},
        {  23900,   11956, 2510031},
        {  24364, 2494127,   12157},
        {  40830, 2582274,   20366},
        { 260110,  130220, 3107660},

        {    501,     240, 2453050},
        {   1426, 2494049,     730},
        {   1772,     857, 2432012},
        {   9983,    4979, 2445486},
        {  17204,    8613, 2558446},
        {  67433, 2430736,   33588},
        { 265677, 2582262,  132629},

        {  36852, 2533935,   18350},
        {  99110,   49699, 2681560},
        { 109806, 2732830,   54796},
        { 175832, 2696177,   87747},
        { 227015,  113648, 2426409},
        { 569018, 2905970,  284333},
        { 624385, 2904043,  311822},
        {1402946, 2912942,  701003},

        {1402948, 3381721,  701005},
        {1404916, 3517695,  701958}
    };*/


    if (!TO_TRACK) return false;
    else {
        const VertexId node_to_track[NUM_TO_TRACK > 0 ? NUM_TO_TRACK : 1][3] = {};

        #pragma unroll
        for (int i=0; i<NUM_TO_TRACK; i++)
            //if (gpu_num == gpu_to_track[i] &&
            //    node == node_to_track[i]) 
            if (node_to_track[i][gpu_num+1] == node)
                return true;
    }
    return false;
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
    if (!TO_TRACK) return;
    else for (VertexId v=0; v<graph->nodes; v++)
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
    typename ProblemData::DataSlice *data_slice,
    VertexId  queue_index)
{
    SizeT offset = offset1 + offset2;
    if (!TO_TRACK)
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            new_value, d_out + offset); 
    else {
        VertexId old_value = atomicCAS(d_out + offset, -2, new_value);
        if (old_value != -2 && util::to_track(data_slice -> gpu_idx, new_value))
        {    
            printf("%d\t %d\t %d\t Storing conflict: [%d] -> %p + %lld, old_value = [%d], "
                "offset1 = %lld, offset2 = %lld, blockIdx.x = %d, threadIdx.x = %d,"
                "org_cp = %d, org_q_idx = %d, org_d_out = %p, org_offset1 = %lld,"
                "org_offset2 = %lld, org_blockIdx.x = %d, org_threadIdx.x = %d\n",
                data_slice -> gpu_idx, queue_index+1, checkpoint_num, new_value, d_out, 
                (long long)offset, old_value, (long long)offset1, 
                (long long)offset2, blockIdx.x, threadIdx.x,
                data_slice -> org_checkpoint[offset],
                data_slice -> org_queue_idx [offset],
                data_slice -> org_d_out     [offset],
                (long long)data_slice -> org_offset1   [offset],
                (long long)data_slice -> org_offset2   [offset],
                data_slice -> org_block_idx [offset],
                data_slice -> org_thread_idx[offset]);
        } else {
            data_slice -> org_checkpoint[offset] = checkpoint_num;
            data_slice -> org_d_out     [offset] = d_out         ;
            data_slice -> org_offset1   [offset] = offset1       ;
            data_slice -> org_offset2   [offset] = offset2       ;
            data_slice -> org_queue_idx [offset] = queue_index+1 ;
            data_slice -> org_block_idx [offset] = blockIdx.x    ;
            data_slice -> org_thread_idx[offset] = threadIdx.x   ;
        }    
        if (util::to_track(data_slice -> gpu_idx, new_value))
        {    
            printf("%d\t %d\t %d\t Storing [%d] -> %p + %lld\n",
                data_slice -> gpu_idx, queue_index+1, checkpoint_num, new_value, 
                d_out, (long long)offset);
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
