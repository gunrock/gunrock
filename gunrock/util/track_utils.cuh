
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

#define TO_TRACK true
#define NUM_TO_TRACK 56

template <typename VertexId>
static __device__ __host__ __inline__ bool to_track(VertexId node) {
    const VertexId node_to_track[] = {
        81561706,
        48459810, 
        18876984,
        1902};
    if (!TO_TRACK) return false;
    else { 
        #pragma unroll
        for (int i = 0; i < NUM_TO_TRACK; i++)
            if (node == node_to_track[i]) return true;
    }
    return false;
}

template <typename VertexId>
static __device__ __host__ __inline__ bool to_track(
    int gpu_num, VertexId node)
{
    //const VertexId node_to_track[NUM_TO_TRACK > 0 ? NUM_TO_TRACK : 1][3] = {};
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

    // for BFS, market /data/gunrock_dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx --src=largestdegree --traversal-mode=1 --undirected --device=0,1,2,3 --queue-sizing=8.0 --in-sizing=0.5 --idempotence --v --partition-seed=1452208768
    
    /*const VertexId node_to_track[][5] = {
        {  5487,    1370, 1239278, 1212000, 1238478},
        {  5503,    1377, 1531518, 1236984, 1238502},
        {  5520,    1381, 1309968, 1236988, 1482071},
        {  5842, 2313598, 2078214,    1454, 2916472},
        {  6110, 2938440, 3162369, 2915808,    1523},
        { 22727, 3051939, 3633847, 2969441,    5630},
        {228833, 2377430, 2375217,   57405, 3215261},
        {228837,   57428, 1536317, 1536368, 1537966},
        {228845,   57431, 1536319, 1536371, 1537967},
        {228852,   57433, 1536331, 1536372, 1537968}
    };*/

    // for BFS, market /data/gunrock_dataset/large/soc-LiveJournal1/soc-LiveJournal1.mtx --src=largestdegree --traversal-mode=1 --undirected --device=0,1 --queue-sizing=7.0 --queue-sizing1=8.5 --idempotence --partition-seed=1452615167
    const VertexId node_to_track[][5] = {
        {1252566,  626153, 2873663},
        {1252567, 3483619,  626413},
        {1252568,  626154, 2601173},
        {2168129, 3299550, 1084495},
        { 417722, 3415075,  208995},
        { 673597, 4847572,  336639},
        {1533408, 4847572,  767025},
        {1533411, 4847572,  767027},
        {2527926, 3280482, 1264659},
        {2949435, 1474568, 2893498},
        {  15791, 2621277,    7862},
        {  15792,    7929, 2634600},
        {  16818, 2501728,    8366},
        {  26081,   13056, 4285221},
        {  26775, 2694940,   13370},

        {  15789,    7928, 2634593},
        {2332103, 1165431, 2634601},
        {   3947,    1963, 2444884},
        {   6168,    3085, 2521563},
        {   4622,    2289, 2511546},
        {   4639, 2501727,    2338},
        {   4648, 2501743,    2346},
        {  42787, 2501720,   21415},
        {1617210, 2691935,  808783},
        {2657850, 1328388, 2641350},
        {2657855, 1328391, 2641353},
        {  26054,   13038, 2593963},
        {2682456, 2692087, 1341748},
        {  26773, 2694881,   13368},
        {  26777, 2694941,   13372},
        {  26782, 2694942,   13374},
        {  26784, 2694882,   13376},
        {  26802,   13415, 2713197},
        {  26803, 2694943,   13387},
        { 434027,  216933, 2713220},
        { 518973, 2446618,  259374},
        { 548276,  274268, 2713221},
        {1464833, 2611737,  732610},
        {2278177, 1138503, 2713225},
        {2650609, 1324690, 2713226},
        {2683835, 1341400, 2713227},
        {2683838, 3696777, 1342435},
        {2683840, 3696778, 1342437},
        {2683841, 3696779, 1342438},
        {2683842, 1341403, 2713230},
        {2683846, 3696780, 1342439},
        {2683847, 4055014, 1342440},
        {2683848, 1341407, 2713234},
        {2683849, 3696781, 1342441},
        {2683850, 1341408, 2713235},
        { 301726,  150596, 2440086},
        {  11723, 2428169,    5836},
        { 359848,  179920, 2593632},
        { 219110,  109174, 2692826},
        { 209169,  104204, 3297101},
        {  56958, 2552293,   28432}
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

template <typename T>
bool is_puer2(T x)
{
    if (x <= 2) return true;
    if ((x%2) != 0) return false;
    return is_puer2(x/2);
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
    SizeT nodes = graph->nodes;
    if (references == NULL) return;
    if (!TO_TRACK) return;
    else {
        VertexId *markers = new VertexId[graph->nodes];
        VertexId *track_nodes = new VertexId[NUM_TO_TRACK + 1];
        SizeT *incoming_counter = new SizeT[NUM_TO_TRACK];
        SizeT counter = 0;
        VertexId **preds = new VertexId*[NUM_TO_TRACK];

        for (VertexId dest=0; dest<nodes; dest++)
        if (to_track(-1, dest)) 
        {
            markers[dest] = counter;
            track_nodes[counter] = dest;
            incoming_counter[counter] = 0;
            counter ++;
        } else markers[dest] = NUM_TO_TRACK;

        for (VertexId src=0; src<nodes; src++)
        for (SizeT j = graph->row_offsets[src]; j < graph->row_offsets[src+1]; j++)
        {
            VertexId dest = graph -> column_indices[j];
            VertexId dest_ = markers[dest];
            if (dest_ == NUM_TO_TRACK) continue;
            if (incoming_counter[dest_] == 0)
            {
                preds[dest_] = new VertexId[1];
            } else if (is_puer2(incoming_counter[dest_]))
            {
                VertexId *temp_array = new VertexId[incoming_counter[dest_] * 2];
                memcpy(temp_array, preds[dest_], sizeof(VertexId) * incoming_counter[dest_]);
                delete[] preds[dest_];
                preds[dest_] = temp_array;
                temp_array = NULL;
            }
            preds[dest_][incoming_counter[dest_]] = src;
            incoming_counter[dest_] ++;
        }
        
        for (SizeT i=0; i<NUM_TO_TRACK; i++)
        {
            VertexId dest = track_nodes[i];
            printf("Vertex %d, ", dest);
            if (fabs(results[dest] - references[dest]) >= error_threshold)
                printf("reference = %d, ", references[dest]);
            printf("result = %d, ", results[dest]);
            if (num_gpus > 1)
            {
                printf("host = %d, v_ = ", partition_table[dest]);
                for (int gpu=0; gpu<num_gpus; gpu++)
                    printf("%d%s", convertion_tables[gpu][dest], gpu == num_gpus-1? "" : ", ");
            }
            printf("\n");
            for (SizeT j = 0; j < incoming_counter[i]; j++)
            {
                VertexId src = preds[i][j];
                //if (references[src] != references[dest] -1 &&
                //    fabs(results[src] - references[src]) < error_threshold) continue; // bfs
                printf("\t{%d ", src);
                if (num_gpus > 1)
                {
                    for (int gpu=0; gpu<num_gpus; gpu++)
                        printf(", %d", convertion_tables[gpu][src]);
                }
                printf("},\n\t\t");
                if (num_gpus > 1)
                    printf("host = %d, ", partition_table[src]);
                if (fabs(results[src] - references[src]) >= error_threshold)
                    printf("reference = %d, ", references[src]);
                printf("result = %d, ", results[src]);
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
    typename ProblemData::DataSlice *data_slice,
    VertexId  queue_index)
{
    SizeT offset = offset1 + offset2;
    //if (!TO_TRACK)
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
            new_value, d_out + offset); 
    /*else {
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
    }*/   
} 

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
