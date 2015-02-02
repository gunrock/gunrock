// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_base.cuh
 *
 * @brief Base Graph Problem Enactor
 */

#pragma once
#include <time.h>

#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/app/problem_base.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {

struct EnactorStats
{
    long long           iteration;
    unsigned long long  total_lifetimes;
    unsigned long long  total_runtimes;
    unsigned long long  total_queued;

    unsigned int        advance_grid_size;
    unsigned int        filter_grid_size;

    util::KernelRuntimeStatsLifetime advance_kernel_stats;
    util::KernelRuntimeStatsLifetime filter_kernel_stats;

    util::Array1D<int, unsigned int> node_locks    ;
    util::Array1D<int, unsigned int> node_locks_out;

    cudaError_t        retval;
    clock_t            start_time;

    EnactorStats()
    {
        //util::cpu_mt::PrintMessage("EnactorStats() begin.");
        iteration       = 0;
        total_lifetimes = 0;
        total_queued    = 0;
        total_runtimes  = 0;
        retval          = cudaSuccess;
        node_locks    .SetName("node_locks"    );
        node_locks_out.SetName("node_locks_out");
        //util::cpu_mt::PrintMessage("EnactorStats() end.");
    }
};

template <typename SizeT>
struct FrontierAttribute
{
    SizeT        queue_length;
    util::Array1D<SizeT,SizeT>        output_length;
    unsigned int        queue_index;
    SizeT        queue_offset;
    int                 selector;
    bool                queue_reset;
    int                 current_label;
    bool                has_incoming;
    gunrock::oprtr::advance::TYPE   advance_type;

    FrontierAttribute()
    {
        queue_length  = 0;
        //output_length = 0;
        queue_index   = 0;
        queue_offset  = 0;
        selector      = 0;
        queue_reset   = false;
        has_incoming  = false;
        output_length.SetName("output_length");
    }
};

class ThreadSlice
{    
public:
    int           thread_num;
    int           init_size;
    CUTThread     thread_Id;
    int           stats;
    void*         problem;
    void*         enactor;
    ContextPtr*   context;
    util::cpu_mt::CPUBarrier* cpu_barrier;

    ThreadSlice()
    {    
        problem     = NULL;
        enactor     = NULL;
        context     = NULL;
        thread_num  = 0; 
        init_size   = 0; 
        stats       = -2;
        cpu_barrier = NULL;
    }    

    virtual ~ThreadSlice()
    {    
        problem     = NULL;
        enactor     = NULL;
        context     = NULL;
        cpu_barrier = NULL;
    }    
};   

template <typename SizeT, typename DataSlice>
bool All_Done(EnactorStats                    *enactor_stats,
              FrontierAttribute<SizeT>        *frontier_attribute, 
              util::Array1D<SizeT, DataSlice> *data_slice, 
              int                              num_gpus)
{   
    for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
    if (enactor_stats[gpu].retval!=cudaSuccess)
    {   
        printf("(CUDA error %d @ GPU %d: %s\n", enactor_stats[gpu].retval, gpu, cudaGetErrorString(enactor_stats[gpu].retval)); fflush(stdout);
        return true;
    }   

    for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
    if (frontier_attribute[gpu].queue_length!=0 || frontier_attribute[gpu].has_incoming)
    {
        //printf("gpu=%d, queue_length=%d\n",gpu,frontier_attribute[gpu].queue_length);   
        return false;
    }

    for (int gpu=0;gpu<num_gpus;gpu++)
    for (int peer=1;peer<num_gpus;peer++)
    for (int i=0;i<2;i++)
    if (data_slice[gpu]->in_length[i][peer]!=0)
        return false;
    //printf("all gpu done\n");fflush(stdout);

    for (int gpu=0;gpu<num_gpus;gpu++)
    for (int peer=1;peer<num_gpus;peer++)
    if (data_slice[gpu]->out_length[peer]!=0)
        return false;

    return true;
} 

template <typename VertexId, typename SizeT>
__global__ void Copy_Preds (
    const SizeT     num_elements,
    const VertexId* keys,
    const VertexId* in_preds,
          VertexId* out_preds)
{
    const SizeT STRIDE = gridDim.x * blockDim.x;   
    VertexId x = blockIdx.x*blockDim.x+threadIdx.x;
    VertexId t;

    //if (x>=num_elements) return;
    while (x<num_elements)
    {
        t = keys[x];
        out_preds[t] = in_preds[t];
        x+= STRIDE;
    }
}   

template <typename VertexId, typename SizeT>
__global__ void Update_Preds (
    const SizeT     num_elements,
    const SizeT     nodes,
    const VertexId* keys,
    const VertexId* org_vertexs,
    const VertexId* in_preds,
          VertexId* out_preds)//,
{
    const SizeT STRIDE = gridDim.x * blockDim.x;   
    VertexId x = blockIdx.x*blockDim.x+threadIdx.x;
    VertexId t, p;
    /*long long x= blockIdx.y;
    x = x*gridDim.x+blockIdx.x;
    x = x*blockDim.y+threadIdx.y;
    x = x*blockDim.x+threadIdx.x;*/

    //if (x>=num_elements) return;
    while (x<num_elements)
    {
        t = keys[x];
        p = in_preds[t];
        //temp_marker[t]++;
        //if (p>=nodes) //|| (t==4651 && nodes==219953)) 
        //{
            //printf("x=%d, key=%d, p=%d, tm=%d, nodes=%d\n", x, t, p, temp_marker[t], nodes);
        //} else {
        if (p<nodes) out_preds[t] = org_vertexs[p];
        //}
        x+= STRIDE;
    }
}   

template <typename VertexId, typename SizeT>
__global__ void Assign_Marker(
    const SizeT            num_elements,
    const int              num_gpus,
    const VertexId* const  keys_in,
    const int*      const  partition_table,
          SizeT**          marker)//,
{
    VertexId key;
    int gpu;
    extern __shared__ SizeT* s_marker[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    SizeT x= blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < num_gpus)
        s_marker[threadIdx.x]=marker[threadIdx.x];
    __syncthreads();
        
    while (x < num_elements)
    {
        key = keys_in[x];
        gpu = partition_table[key];
        for (int i=0;i<num_gpus;i++)
            s_marker[i][x]=(i==gpu)?1:0;
        x+=STRIDE;
    }
}

template <typename VertexId, typename SizeT>
__global__ void Assign_Marker_Backward(
    const SizeT            num_elements,
    const int              num_gpus,
    const VertexId* const  keys_in,
    const SizeT*    const  offsets,
    const int*      const  partition_table,
          SizeT**          marker)//,
{
    VertexId key;
    //int gpu;
    extern __shared__ SizeT* s_marker[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    SizeT x= blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x < num_gpus)
        s_marker[threadIdx.x]=marker[threadIdx.x];
    __syncthreads();
        
    while (x < num_elements)
    {
        key = keys_in[x];
        for (int gpu=0;gpu<num_gpus;gpu++)
            s_marker[gpu][x]=0;
        if (key!=-1) for (SizeT i=offsets[key];i<offsets[key+1];i++)
            s_marker[partition_table[i]][x]=1;
        x+=STRIDE;
    }
}

template <typename VertexId, typename SizeT, typename Value,
          SizeT num_vertex_associates, SizeT num_value__associates>
__global__ void Make_Out(
   const  SizeT             num_elements,
   const  int               num_gpus,
   const  VertexId*   const keys_in,
   const  int*        const partition_table,
   const  VertexId*   const convertion_table,
   const  size_t            array_size,
          char*             array)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    size_t     offset                  = 0;
    SizeT**    s_marker                = (SizeT**   )&(s_array[offset]);
    offset+=sizeof(SizeT*   )*num_gpus;
    VertexId** s_keys_outs             = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus;
    VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_vertex_associates;
    Value**    s_value__associate_orgs = (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   )*num_value__associates;
    VertexId** s_vertex_associate_outss= (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus*num_vertex_associates;
    Value**    s_value__associate_outss= (Value**   )&(s_array[offset]);
    SizeT x= threadIdx.x;

    while (x<array_size)
    {
        s_array[x]=array[x];
        x+=blockDim.x;
    }
    __syncthreads();

    x= blockIdx.x * blockDim.x + threadIdx.x;
    //if (x>=num_elements) return;
    while (x<num_elements)
    {
        VertexId key    = keys_in [x];
        //if (key <0) {x+=STRIDE; continue;}
        int      target = partition_table[key];
        SizeT    pos    = s_marker[target][x]-1;

        //printf("x=%d, key=%d, pos=%d, target=%d\t", x, key, pos, target);
        if (target==0)
        {
            s_keys_outs[0][pos]=key;
        } else {
            s_keys_outs[target][pos]=convertion_table[key];
            #pragma unrool
            for (int i=0;i<num_vertex_associates;i++)
                s_vertex_associate_outss[target*num_vertex_associates+i][pos]
                    =s_vertex_associate_orgs[i][key];
            #pragma unrool
            for (int i=0;i<num_value__associates;i++)
                s_value__associate_outss[target*num_value__associates+i][pos]
                    =s_value__associate_orgs[i][key];
        }
        x+=STRIDE;
    }
}

template <typename VertexId, typename SizeT, typename Value,
          SizeT num_vertex_associates, SizeT num_value__associates>
__global__ void Make_Out_Backward(
   const  SizeT             num_elements,
   const  int               num_gpus,
   const  VertexId*   const keys_in,
   const  SizeT*      const offsets,
   const  int*        const partition_table,
   const  VertexId*   const convertion_table,
   const  size_t            array_size,
          char*             array)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    size_t     offset                  = 0;
    SizeT**    s_marker                = (SizeT**   )&(s_array[offset]);
    offset+=sizeof(SizeT*   )*num_gpus;
    VertexId** s_keys_outs             = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus;
    VertexId** s_vertex_associate_orgs = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_vertex_associates;
    Value**    s_value__associate_orgs = (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   )*num_value__associates;
    VertexId** s_vertex_associate_outss= (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*)*num_gpus*num_vertex_associates;
    Value**    s_value__associate_outss= (Value**   )&(s_array[offset]);
    SizeT x= threadIdx.x;

    while (x<array_size)
    {
        s_array[x]=array[x];
        x+=blockDim.x;
    }
    __syncthreads();

    x= blockIdx.x * blockDim.x + threadIdx.x;
    while (x<num_elements)
    {
        VertexId key    = keys_in [x];
        if (key <0) {x+=STRIDE; continue;}
        for (int j=offsets[key];j<offsets[key+1];j++)
        {
            int      target = partition_table[j];
            SizeT    pos    = s_marker[target][x]-1;

            //printf("x=%d, key=%d, pos=%d, target=%d\t", x, key, pos, target);
            if (target==0)
            {
                s_keys_outs[0][pos]=key;
            } else {
                s_keys_outs[target][pos]=convertion_table[j];
                #pragma unrool
                for (int i=0;i<num_vertex_associates;i++)
                    s_vertex_associate_outss[target*num_vertex_associates+i][pos]
                        =s_vertex_associate_orgs[i][key];
                #pragma unrool
                for (int i=0;i<num_value__associates;i++)
                    s_value__associate_outss[target*num_value__associates+i][pos]
                        =s_value__associate_orgs[i][key];
            }
        }
        x+=STRIDE;
    }
}

template <typename VertexId, typename SizeT>
__global__ void Mark_Queue (
    const SizeT     num_elements,
    const VertexId* keys,
          unsigned int* marker)
{
    VertexId x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
    if (x< num_elements) marker[keys[x]]=1;
}

template <
    bool     SIZE_CHECK,
    typename SizeT, 
    typename VertexId,
    typename Value,
    typename GraphSlice,
    typename DataSlice,
    SizeT    num_vertex_associate,
    SizeT    num_value__associate>
void PushNeibor(
    int gpu,
    int peer,
    SizeT             queue_length,
    EnactorStats      *enactor_stats,
    DataSlice         *data_slice_l,
    DataSlice         *data_slice_p,
    GraphSlice        *graph_slice_l,
    GraphSlice        *graph_slice_p,
    cudaStream_t      stream)
{
    if (peer == gpu) return;
    int gpu_  = peer<gpu? gpu : gpu+1;
    int peer_ = peer<gpu? peer+1 : peer;
    
    data_slice_p->in_length[enactor_stats->iteration%2][gpu_]
                  = queue_length;
    if (queue_length == 0) return;
    int t=enactor_stats->iteration%2;
    
    if (SIZE_CHECK)
    {
        if (data_slice_p -> keys_in[t][gpu_].GetSize() < queue_length)
        {
            printf("%d\t %lld\t %d\t keys_in   \t oversize :\t %d ->\t %d \n", 
                gpu, enactor_stats->iteration, peer, 
                data_slice_p->keys_in[t][gpu_].GetSize(), queue_length); 
            fflush(stdout);
        
            util::SetDevice(data_slice_p->gpu_idx);
            data_slice_p->keys_in[t][gpu_].EnsureSize(queue_length);
            for (int i=0;i<num_vertex_associate;i++)
            {
                if (enactor_stats->retval = data_slice_p->vertex_associate_in [t][gpu_][i].EnsureSize(queue_length)) return;
                data_slice_p->vertex_associate_ins[t][gpu_][i] = data_slice_p->vertex_associate_in[t][gpu_][i].GetPointer(util::DEVICE);
            }
            if (enactor_stats->retval = data_slice_p->vertex_associate_ins[t][gpu_].Move(util::HOST, util::DEVICE)) return;
            for (int i=0;i<num_value__associate;i++)
            {
                if (enactor_stats->retval = data_slice_p->value__associate_in [t][gpu_][i].EnsureSize(queue_length)) return;
                data_slice_p->value__associate_ins[t][gpu_][i] = data_slice_p->value__associate_in[t][gpu_][i].GetPointer(util::DEVICE);
            }
            if (enactor_stats->retval = data_slice_p->value__associate_ins[t][gpu_].Move(util::HOST, util::DEVICE)) return;
            util::SetDevice(data_slice_l->gpu_idx);
        }
    }

    //while (data_slice_p->gpu_mallocing!=0) ;
    if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
        data_slice_p  -> keys_in[t][gpu_].GetPointer(util::DEVICE),
            //+ graph_slice_p -> in_offset[gpu_],
        data_slice_l -> keys_out[peer_].GetPointer(util::DEVICE),
            //+ frontier_attribute->queue_offset,
        sizeof(VertexId) * queue_length, cudaMemcpyDefault, stream),
        "cudaMemcpyPeer d_keys failed", __FILE__, __LINE__)) return;
            
    for (int i=0;i<num_vertex_associate;i++)
    {
        //printf("Moving vertex_associate[%d][%d] @ GPU %d -> %d: max_size = %d, length = %d, @ %p -> %p, max_size= %d, offset = %d \n",
        //        t, gpu_, data_slice_l->gpu_idx, data_slice_p -> gpu_idx, 
        //        data_slice_p->vertex_associate_in[t][gpu_][i].GetSize(), 
        //        frontier_attribute->queue_length,
        //        data_slice_l->vertex_associate_out[i].GetPointer(util::DEVICE), 
        //        data_slice_p->vertex_associate_in[t][gpu_][i].GetPointer(util::DEVICE),
        //        data_slice_p->vertex_associate_out[i].GetSize(), 
        //        frontier_attribute->queue_offset - data_slice_l->out_length[0]);
        //fflush(stdout);   
        if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
            data_slice_p->vertex_associate_ins[t][gpu_][i],
                //+ graph_slice_p->in_offset[gpu_],
            data_slice_l->vertex_associate_outs[peer_][i],
                //+ (frontier_attribute->queue_offset - data_slice_l->out_length[0]),
            sizeof(VertexId) * queue_length, cudaMemcpyDefault, stream),
            "cudaMemcpyPeer vertex_associate_out failed", __FILE__, __LINE__)) return;
    }

    for (int i=0;i<num_value__associate;i++)
    {   
        if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
            data_slice_p->value__associate_ins[t][gpu_][i],
                //+ graph_slice_p->in_offset[gpu_],
            data_slice_l->value__associate_outs[peer_][i],
                //+ (frontier_attribute->queue_offset - data_slice_l->out_length[0]),
            sizeof(Value) * queue_length, cudaMemcpyDefault, stream),
                "cudaMemcpyPeer value__associate_out failed", __FILE__, __LINE__)) return;
    }
}

template <typename Problem>
void ShowDebugInfo(
    int                    thread_num,
    int                    peer_,
    FrontierAttribute<typename Problem::SizeT>      *frontier_attribute,
    EnactorStats           *enactor_stats,
    typename Problem::DataSlice  *data_slice,
    GraphSlice<typename Problem::SizeT, typename Problem::VertexId, typename Problem::Value> *graph_slice,
    util::CtaWorkProgressLifetime *work_progress,
    std::string            check_name = "",
    cudaStream_t           stream = 0) 
{    
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value;
    SizeT queue_length;

    //util::cpu_mt::PrintMessage(check_name.c_str(), thread_num, enactor_stats->iteration);
    //printf("%d \t %d\t \t reset = %d, index = %d\n",thread_num, enactor_stats->iteration, frontier_attribute->queue_reset, frontier_attribute->queue_index);fflush(stdout);
    //if (frontier_attribute->queue_reset) 
        queue_length = frontier_attribute->queue_length;
    //else if (enactor_stats->retval = util::GRError(work_progress->GetQueueLength(frontier_attribute->queue_index, queue_length, false, stream), "work_progress failed", __FILE__, __LINE__)) return;
    //util::cpu_mt::PrintCPUArray<SizeT, SizeT>((check_name+" Queue_Length").c_str(), &(queue_length), 1, thread_num, enactor_stats->iteration);
    printf("%d\t %lld\t %d\t stage%d\t %s\t Queue_Length = %d\n", thread_num, enactor_stats->iteration, peer_, data_slice->stages[peer_], check_name.c_str(), queue_length);fflush(stdout);
    //printf("%d \t %d\t \t peer_ = %d, selector = %d, length = %d, p = %p\n",thread_num, enactor_stats->iteration, peer_, frontier_attribute->selector,queue_length,graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE));fflush(stdout);
    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>((check_name+" keys").c_str(), graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE), queue_length, thread_num, enactor_stats->iteration,peer_, stream);
    //if (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
    //    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu1", graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE), _queue_length, thread_num, enactor_stats->iteration);
    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("labe1", data_slice[0]->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
    //if (BFSProblem::MARK_PREDECESSORS)
    //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pred1", data_slice[0]->preds.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
    //if (BFSProblem::ENABLE_IDEMPOTENCE)
    //    util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask1", data_slice[0]->visited_mask.GetPointer(util::DEVICE), (graph_slice->nodes+7)/8, thread_num, enactor_stats->iteration);
}  

template <
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES,
    typename Enactor,
    typename Functor,
    typename Iteration>
void Iteration_Loop(
    ThreadSlice *thread_data)
{
    typedef typename Enactor::Problem     Problem   ;
    typedef typename Problem::SizeT       SizeT     ;
    typedef typename Problem::VertexId    VertexId  ;
    typedef typename Problem::Value       Value     ;
    typedef typename Problem::DataSlice   DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value>  GraphSlice;

    Problem      *problem              =  (Problem*) thread_data->problem;
    Enactor      *enactor              =  (Enactor*) thread_data->enactor;
    int           num_gpus             =   problem     -> num_gpus;
    int           thread_num           =   thread_data -> thread_num;
    DataSlice    *data_slice           =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    util::Array1D<SizeT, DataSlice>
                 *s_data_slice         =   problem     -> data_slices;
    GraphSlice   *graph_slice          =   problem     -> graph_slices       [thread_num] ;
    GraphSlice   **s_graph_slice       =   problem     -> graph_slices;
    FrontierAttribute<SizeT>
                 *frontier_attribute   = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    FrontierAttribute<SizeT>
                 *s_frontier_attribute = &(enactor     -> frontier_attribute [0         ]);  
    EnactorStats *enactor_stats        = &(enactor     -> enactor_stats      [thread_num * num_gpus]);
    EnactorStats *s_enactor_stats      = &(enactor     -> enactor_stats      [0         ]);  
    util::CtaWorkProgressLifetime
                 *work_progress        = &(enactor     -> work_progress      [thread_num * num_gpus]);
    ContextPtr   *context              =   thread_data -> context;
    int          *stages               =   data_slice  -> stages .GetPointer(util::HOST);
    bool         *to_show              =   data_slice  -> to_show.GetPointer(util::HOST);
    cudaStream_t *streams              =   data_slice  -> streams.GetPointer(util::HOST);
    SizeT         Total_Length         =   0;
    cudaError_t   tretval              =   cudaSuccess;
    int           grid_size            =   0;
    std::string   mssg                 =   "";
    int           pre_stage            =   0;
    size_t        offset               =   0;
    int           iteration            =   0;
    int           selector             =   0;
    util::DoubleBuffer<SizeT, VertexId, VertexId>
                 *frontier_queue_      =   NULL;
    FrontierAttribute<SizeT>
                 *frontier_attribute_  =   NULL;
    EnactorStats *enactor_stats_       =   NULL;
    util::CtaWorkProgressLifetime
                 *work_progress_       =   NULL;
    int           peer, peer_, peer__, gpu_, i, iteration_, wait_count;

    printf("Iteration entered\n");fflush(stdout);
    /*for (int iteration_=0;iteration_<4;iteration_++)
    for (int peer_=0;peer_<num_gpus;peer_++)
    for (int stage=0;stage<4;stage++)
    {
        printf("thread_num %d events %d stream %d\n",thread_num, data_slice->events[iteration_][peer_][stage],streams[2]);fflush(stdout);
        cudaEventRecord(data_slice->events[iteration_][peer_][stage],streams[2]);
    }
    for (int iteration_=0;iteration_<1000000;iteration_++) peer++;*/

    while (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
    {
        Total_Length             = 0;
        data_slice->wait_counter = 0;
        tretval                  = cudaSuccess;
        if (num_gpus>1 && enactor_stats[0].iteration>0)
        {
            frontier_attribute[0].queue_reset  = true;
            frontier_attribute[0].queue_offset = 0;
            for (i=1; i<num_gpus; i++)
            {
                frontier_attribute[i].selector     = frontier_attribute[0].selector;
                frontier_attribute[i].advance_type = frontier_attribute[0].advance_type;
                frontier_attribute[i].queue_offset = 0;
                frontier_attribute[i].queue_reset  = true;
                frontier_attribute[i].queue_index  = frontier_attribute[0].queue_index;
                frontier_attribute[i].current_label= frontier_attribute[0].current_label;
                enactor_stats     [i].iteration    = enactor_stats     [0].iteration;
            }
        } else {
            frontier_attribute[0].queue_offset = 0;
            frontier_attribute[0].queue_reset  = true;
        }
        for (peer=0; peer<num_gpus; peer++)
        {
            stages [peer         ] = 0   ; 
            stages [peer+num_gpus] = 0   ;
            to_show[peer         ] = true; 
            to_show[peer+num_gpus] = true;
            for (i=0; i<data_slice->num_stages; i++)
                data_slice->events_set[enactor_stats[0].iteration%4][peer][i]=false;
        }
        
        while (data_slice->wait_counter < num_gpus*2
           && (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)))
        {
            for (peer__=0; peer__<num_gpus*2; peer__++)
            {
                peer_               = (peer__%num_gpus);
                peer                = peer_<= thread_num? peer_-1   : peer_       ;
                gpu_                = peer <  thread_num? thread_num: thread_num+1;
                iteration           = enactor_stats[peer_].iteration;
                iteration_          = iteration%4;
                pre_stage           = stages[peer__];
                selector            = frontier_attribute[peer_].selector;
                frontier_queue_     = &(data_slice->frontier_queues[peer_]);
                frontier_attribute_ = &(frontier_attribute[peer_]);
                enactor_stats_      = &(enactor_stats[peer_]);
                work_progress_      = &(work_progress[peer_]);

                if (Enactor::DEBUG && to_show[peer__])
                {
                    //util::cpu_mt::PrintCPUArray<SizeT, int>("stages",data_slice->stages.GetPointer(util::HOST),num_gpus,thread_num,enactor_stats[peer_].iteration);
                    //mssg="pre_stage0";
                    //mssg[9]=char(stages[peer_]+'0');
                   
                    mssg=" ";mssg[0]='0'+data_slice->wait_counter;
                    ShowDebugInfo<Problem>(
                        thread_num,
                        peer__,
                        frontier_attribute_,
                        enactor_stats_,
                        data_slice,
                        graph_slice,
                        work_progress_,
                        mssg,
                        streams[peer__]);
                }
                to_show[peer__]=true;

                switch (stages[peer__])
                {
                case 0: // Assign marker & Scan
                    if (peer_==0) {
                        if (peer__==num_gpus || frontier_attribute_->queue_length==0) 
                        {
                            stages[peer__]=3;
                            //printf("%d\t %d\t %d\t stage0 skipped 2\n",thread_num, iteration, peer__);fflush(stdout);
                        }
                        break;
                    } else if ((iteration==0 || data_slice->out_length[peer_]==0) && peer__>num_gpus) {
                        //printf("thread_num = %d iteration_ = %d peer_ = %d events = %d streams = %d\n", thread_num, iteration_, peer_, data_slice->events[iteration_][peer_][0],streams[peer__]);fflush(stdout);
                        cudaEventRecord(data_slice->events[iteration_][peer_][0],streams[peer__]);
                        data_slice->events_set[iteration_][peer_][0]=true;
                        stages[peer__]=3;
                        //printf("%d\t %d\t %d\t stage0 skipped 1\n",thread_num, iteration, peer__);fflush(stdout);
                        break;
                    }

                    //printf("%d\t %lld\t %d\t stage0 entered\n", thread_num, iteration, peer__); fflush(stdout);
                    if (peer__<num_gpus)
                    { //wait and expand incoming
                        if (!(s_data_slice[peer]->events_set[iteration_][gpu_][0]))
                        {   to_show[peer__]=false;stages[peer__]--;break;}

                        frontier_attribute_->queue_length = data_slice->in_length[iteration%2][peer_];
                        data_slice->in_length[iteration%2][peer_]=0;
                        //printf("%d\t %d\t %d\t stage0 pasted queue_length=%d\n", thread_num, iteration, peer__, frontier_attribute_->queue_length); fflush(stdout);
                        if (frontier_attribute_->queue_length ==0)
                        {   stages[peer__]=3;break;}

                        if (frontier_attribute_->queue_length > frontier_queue_->keys[selector^1].GetSize())
                        {
                            printf("%d\t %d\t %d\t queue1  \t oversize :\t %d ->\t %d\n",
                                thread_num, iteration, peer_,
                                frontier_queue_->keys[selector^1].GetSize(),
                                frontier_attribute_->queue_length);
                            fflush(stdout);
                            if (Enactor::SIZE_CHECK)
                            {
                                if (enactor_stats_->retval = frontier_queue_->keys[selector^1].EnsureSize(frontier_attribute_->queue_length)) break;
                                if (Problem::USE_DOUBLE_BUFFER)
                                {
                                    if (enactor_stats_->retval = frontier_queue_->values[selector^1].EnsureSize(frontier_attribute_->queue_length)) break;
                                }
                            } else {
                                enactor_stats_->retval = util::GRError(cudaErrorLaunchOutOfResources, "queue1 oversize", __FILE__, __LINE__);
                                break;
                            }
                        }

                        offset = 0;
                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                 data_slice -> vertex_associate_ins[iteration%2][peer_].GetPointer(util::HOST),
                                  sizeof(SizeT*   ) * NUM_VERTEX_ASSOCIATES);
                        offset += sizeof(SizeT*   ) * NUM_VERTEX_ASSOCIATES ;
                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                 data_slice -> value__associate_ins[iteration%2][peer_].GetPointer(util::HOST),
                                  sizeof(VertexId*) * NUM_VALUE__ASSOCIATES);
                        offset += sizeof(VertexId*) * NUM_VALUE__ASSOCIATES ;
                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                 data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
                                  sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
                        offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
                        memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                 data_slice -> value__associate_orgs.GetPointer(util::HOST),
                                  sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
                        offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
                        data_slice->expand_incoming_array[peer_].Move(util::HOST, util::DEVICE, offset, 0, streams[peer_]);

                        grid_size = frontier_attribute_->queue_length/256+1;
                        if (grid_size>512) grid_size=512;
                        cudaStreamWaitEvent(streams[peer_],
                            s_data_slice[peer]->events[iteration_][gpu_][0], 0);
                        Iteration::template Expand_Incoming<NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                            grid_size, 256, 
                            offset,
                            streams[peer_],
                            frontier_attribute_->queue_length,
                            data_slice ->keys_in[iteration%2][peer_].GetPointer(util::DEVICE),
                            frontier_queue_->keys[selector^1].GetPointer(util::DEVICE),
                            offset,
                            data_slice ->expand_incoming_array[peer_].GetPointer(util::DEVICE));
                        frontier_attribute_->selector^=1;
                        frontier_attribute_->queue_index++;
                        if (!Iteration::HAS_SUBQ) {
                            //printf("thread_num = %d iteration_ = %d peer_ = %d events = %d streams = %d\n", thread_num, iteration_, peer_, data_slice->events[iteration_][peer_][stages[peer__]],streams[peer__]);fflush(stdout);
                            cudaEventRecord(data_slice->events[iteration_][peer_][2],streams[peer__]);
                            data_slice->events_set[iteration_][peer_][2]=true;
                        }
                        //printf("%d\t %lld\t %d\t stage0 expand_incoming called queue_length=%d\n", thread_num, enactor_stats_->iteration, peer__, frontier_attribute_->queue_length);fflush(stdout);
                    } else { //Push Neibor
                        PushNeibor <Enactor::SIZE_CHECK, SizeT, VertexId, Value, GraphSlice, DataSlice,
                                NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                            thread_num,
                            peer,
                            data_slice->out_length[peer_],
                            enactor_stats_,
                            s_data_slice  [thread_num].GetPointer(util::HOST),
                            s_data_slice  [peer]      .GetPointer(util::HOST),
                            s_graph_slice [thread_num],
                            s_graph_slice [peer],
                            streams       [peer__]);
                        //printf("thread_num = %d iteration_ = %d peer_ = %d events = %d streams = %d\n", thread_num, iteration_, peer_, data_slice->events[iteration_][peer_][stages[peer__]],streams[peer__]);fflush(stdout);
                        cudaEventRecord(data_slice->events[iteration_][peer_][stages[peer__]],streams[peer__]);
                        data_slice->events_set[iteration_][peer_][stages[peer__]]=true;
                        stages[peer__]=3;
                    }
                    break;

                case 1: //Comp Length
                    if (!Iteration::HAS_SUBQ) {to_show[peer_]=false;stages[peer__]=2;}
                    else {
                        enactor_stats_->retval = Iteration::Compute_OutputLength(
                            frontier_attribute_,
                            graph_slice ->row_offsets     .GetPointer(util::DEVICE),
                            graph_slice ->column_indices  .GetPointer(util::DEVICE),
                            data_slice  ->frontier_queues  [peer_].keys[selector].GetPointer(util::DEVICE),
                            data_slice  ->scanned_edges    [peer_].GetPointer(util::DEVICE),
                            graph_slice ->nodes,//frontier_queues[peer_].keys[frontier_attribute[peer_].selector  ].GetSize(), 
                            graph_slice ->edges,//frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(),
                            context          [peer_][0],
                            streams          [peer_],
                            gunrock::oprtr::advance::V2V, true);

                        if (Enactor::SIZE_CHECK)
                        {
                            frontier_attribute_->output_length.Move(util::DEVICE, util::HOST,1,0,streams[peer_]);
                            cudaEventRecord(data_slice->events[iteration_][peer_][stages[peer_]], streams[peer_]);
                            data_slice->events_set[iteration_][peer_][stages[peer_]]=true;
                        }
                    }
                    break;

                case 2: //SubQueue Core
                    if (Enactor::SIZE_CHECK)
                    {
                        if (!data_slice->events_set[iteration_][peer_][stages[peer_]-1])
                        {   to_show[peer_]=false;stages[peer_]--;break;}
                        tretval = cudaEventQuery(data_slice->events[iteration_][peer_][stages[peer_]-1]);
                        if (tretval == cudaErrorNotReady)
                        {   to_show[peer_]=false;stages[peer_]--; break;}
                        else if (tretval !=cudaSuccess) {enactor_stats_->retval=tretval; break;}

                        if (Enactor::DEBUG)
                        {
                            printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                                thread_num, iteration, peer_,
                                frontier_queue_->keys[selector^1].GetSize(),
                                frontier_attribute_->output_length[0]);fflush(stdout);}
                        //frontier_attribute_->output_length[0]+=1;
                        if (frontier_attribute_->output_length[0]+2 > frontier_queue_->keys[selector^1].GetSize())
                        {
                            printf("%d\t %d\t %d\t queue3  \t oversize :\t %d ->\t %d\n",
                                thread_num, iteration, peer_,
                                frontier_queue_->keys[selector^1].GetSize(),
                                frontier_attribute_->output_length[0]+2);fflush(stdout);
                            if (enactor_stats_->retval = frontier_queue_->keys[selector  ].EnsureSize(frontier_attribute_->output_length[0]+2, true)) break;
                            if (enactor_stats_->retval = frontier_queue_->keys[selector^1].EnsureSize(frontier_attribute_->output_length[0]+2)) break;

                            if (Problem::USE_DOUBLE_BUFFER) {
                                if (enactor_stats_->retval = frontier_queue_->values[selector  ].EnsureSize(frontier_attribute_->output_length[0]+2,true)) break;
                                if (enactor_stats_->retval = frontier_queue_->values[selector^1].EnsureSize(frontier_attribute_->output_length[0]+2)) break;
                            }
                           //if (enactor_stats[peer_].retval = cudaDeviceSynchronize()) break;
                        }
                    }

                    Iteration::SubQueue_Core(
                        //(bool) Enactor::DEBUG,
                        thread_num,
                        peer_,
                        frontier_attribute_,
                        enactor_stats_,
                        data_slice,
                        s_data_slice[thread_num].GetPointer(util::DEVICE),
                        graph_slice,
                        &(work_progress[peer_]),
                        context[peer_],
                        streams[peer_]);
                    if (enactor_stats_->retval = work_progress[peer_].GetQueueLength(
                        frontier_attribute_->queue_index,
                        frontier_attribute_->queue_length,
                        false,
                        streams[peer_],
                        true)) break;
                    if (num_gpus>1)
                    {
                        cudaEventRecord(data_slice->events[iteration_][peer_][stages[peer_]], streams[peer_]);
                        data_slice->events_set[iteration_][peer_][stages[peer_]]=true;
                    }
                    break;

                case 3: //Copy
                    if (num_gpus <=1) {Total_Length = frontier_attribute_->queue_length; to_show[peer_]=false;break;}
                    //to_wait = false;
                    //for (int i=0;i<num_gpus;i++)
                    //    if (stages[i]<stages[peer_])
                    //    {
                    //        to_wait=true;break;
                    //    }
                    //if (to_wait)
                    if (Iteration::HAS_SUBQ || peer_!=0) {
                        if (!data_slice->events_set[iteration_][peer_][stages[peer_]-1])
                        {   to_show[peer_]=false;stages[peer_]--;break;}
                        tretval = cudaEventQuery(data_slice->events[iteration_][peer_][stages[peer_]-1]);
                        if (tretval == cudaErrorNotReady)
                        {   to_show[peer_]=false;stages[peer_]--;break;}
                        else if (tretval !=cudaSuccess) {enactor_stats_->retval=tretval; break;}
                    } //else if (peer_!=0) {
                    //    if (!data_slice->events_set[iteration_][peer_][0])
                    //    {   to_show[peer_]=false;stages[peer_]--;break;}
                    //    tretval = cudaEventQuery(data_slice->events[iteration_][peer_][0]);
                    //    if (tretval == cudaErrorNotReady)
                    //    {   to_show[peer_]=false;stages[peer_]--;break;}
                    //    else if (tretval !=cudaSuccess) {enactor_stats_->retval=tretval; break;}
                    //}

                    //else cudaStreamSynchronize(streams[peer_]);
                    //data_slice->events_set[iteration_][peer_][stages[peer_]-1]=false;

                    //if (DEBUG) 
                    //{
                    //    printf("%d\t %lld\t %d\t org_length = %d, queue_length = %d, new_length = %d, max_length = %d\n", 
                    //        thread_num, 
                    //        enactor_stats[peer_].iteration, 
                    //        peer_, 
                    //        Total_Length, 
                    //        frontier_attribute[peer_].queue_length,
                    //        Total_Length + frontier_attribute[peer_].queue_length,
                    //        graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector].GetSize());
                    //    fflush(stdout);
                    //}

                    if (!Enactor::SIZE_CHECK)
                    {
                        if (Iteration::HAS_SUBQ)
                        {
                            //printf("output_length = %d, queue_size = %d\n", frontier_attribute[peer_].output_length[0], graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize());fflush(stdout);
                            if (frontier_attribute[peer_].output_length[0] > data_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize())
                            {
                                printf("%d\t %lld\t %d\t queue3  \t oversize :\t %d ->\t %d\n",
                                    thread_num, enactor_stats[peer_].iteration, peer_,
                                    data_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(),
                                    frontier_attribute[peer_].output_length[0]);fflush(stdout);
                                enactor_stats_->retval = util::GRError(cudaErrorLaunchOutOfResources, "queue3 oversize", __FILE__, __LINE__);
                                break;
                            }
                        }
                        if (frontier_attribute_->queue_length ==0) break;

                        if (Total_Length + frontier_attribute_->queue_length > data_slice->frontier_queues[num_gpus].keys[0].GetSize())
                        {
                            printf("%d\t %d\t %d\t total_queue\t oversize :\t %d ->\t %d \n",
                                thread_num, iteration, peer_,
                                data_slice->frontier_queues[num_gpus].keys[0].GetSize(),
                                Total_Length + frontier_attribute_->queue_length);fflush(stdout);
                            enactor_stats_ -> retval = util::GRError(cudaErrorLaunchOutOfResources, "total_queue oversize", __FILE__, __LINE__);
                            break;
                        }
                        util::MemsetCopyVectorKernel<<<256,256, 0, streams[peer_]>>>(
                            data_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE) + Total_Length,
                            frontier_queue_->keys[selector].GetPointer(util::DEVICE),
                            frontier_attribute_->queue_length);
                        if (Problem::USE_DOUBLE_BUFFER)
                            util::MemsetCopyVectorKernel<<<256,256,0,streams[peer_]>>>(
                                data_slice->frontier_queues[num_gpus].values[0].GetPointer(util::DEVICE) + Total_Length,
                                frontier_queue_->values[selector].GetPointer(util::DEVICE),
                                frontier_attribute_->queue_length);
                    }
                    //printf(".");

                    Total_Length += frontier_attribute_->queue_length;

                    //if (First_Stage4)
                    //{
                    //    First_Stage4=false;
                    //    util::MemsetKernel<<<128, 128, 0, streams[peer_]>>>
                    //        (data_slice->temp_marker.GetPointer(util::DEVICE),
                    //        (unsigned int)0, graph_slice->nodes);
                    //}
                    //cudaEventRecord(data_slice->events[iteration_][peer_][stages[peer_]], streams[peer_]);
                    //data_slice->events_set[iteration_][peer_][stages[peer_]]=true;

                    break;

                case 4: //End
                    data_slice->wait_counter++;
                    to_show[peer__]=false;
                    break;
                default:
                    stages[peer__]--;
                    to_show[peer__]=false;
                }

                if (Enactor::DEBUG && !enactor_stats_->retval)
                {
                    mssg="stage 0 @ gpu 0, peer_ 0 failed";
                    mssg[6]=char(pre_stage+'0');
                    mssg[14]=char(thread_num+'0');
                    mssg[23]=char(peer__+'0');
                    if (enactor_stats_->retval = util::GRError(//cudaStreamSynchronize(streams[peer_]),
                         mssg, __FILE__, __LINE__)) break;
                    //sleep(1);
                }
                stages[peer__]++;
                if (enactor_stats_->retval) break;
                //if (All_Done(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) break;
            }
            //to_wait=true;
            //for (int i=0;i<num_gpus;i++)
            //    if (to_show[i])
            //    {
            //        to_wait=false;
            //        break;
            //    }
            //if (to_wait) sleep(0);
        }

        if (//num_gpus>1 &&
            !Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
        {
            //if (First_Stage4)
            //{
            //    util::MemsetKernel<<<128,128, 0, streams[0]>>>
            //        (data_slice->temp_marker.GetPointer(util::DEVICE),
            //        (unsigned int)0, graph_slice->nodes);
            //}
            for (peer_=0;peer_<num_gpus;peer_++)
            for (i=0;i<data_slice->num_stages;i++)
                data_slice->events_set[(enactor_stats[0].iteration+3)%4][peer_][i]=false;

            for (peer_=0;peer_<num_gpus*2;peer_++)
                data_slice->wait_marker[peer_]=0;
            wait_count=0;
            while (wait_count<num_gpus*2-1 &&
                !Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
            {
                for (peer_=0;peer_<num_gpus*2;peer_++)
                {
                    if (peer_==num_gpus || data_slice->wait_marker[peer_]!=0)
                        continue;
                    tretval = cudaStreamQuery(streams[peer_]);
                    if (tretval == cudaSuccess)
                    {
                        data_slice->wait_marker[peer_]=1;
                        wait_count++;
                        continue;
                    } else if (tretval != cudaErrorNotReady)
                    {
                        enactor_stats[peer_%num_gpus].retval = tretval;
                        break;
                    }
                }
            }

            //printf("%d\t %lld\t past StreamSynchronize\n", thread_num, enactor_stats[0].iteration);
            //if (SIZE_CHECK)
            //{
            //    if (graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].GetSize() < Total_Length)
            //    {
            //        printf("%d\t %lld\t \t keysn oversize : %d -> %d \n",
            //           thread_num, enactor_stats[0].iteration,
            //           graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].GetSize(), Total_Length);
            //        if (enactor_stats[0].retval = graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].EnsureSize(Total_Length)) break;
            //        if (BFSProblem::USE_DOUBLE_BUFFER)
            //        {
            //            if (enactor_stats[0].retval = graph_slice->frontier_queues[num_gpus].values[frontier_attribute[0].selector^1].EnsureSize(Total_Length)) break;
            //        }
            //    }
            //}

            if (Enactor::DEBUG) {printf("%d\t %lld\t \t Subqueue finished. Total_Length= %d\n", thread_num, enactor_stats[0].iteration, Total_Length);fflush(stdout);}
            //if (Total_Length>0)
            {
                grid_size = Total_Length/256+1;
                if (grid_size > 512) grid_size = 512;

                if (Enactor::SIZE_CHECK)
                {
                    if (data_slice->frontier_queues[0].keys[frontier_attribute[0].selector].GetSize()<Total_Length)
                    {
                        printf("%d\t %lld\t \t total_queue\t oversize :\t %d ->\t %d \n",
                            thread_num, enactor_stats[0].iteration,
                            data_slice->frontier_queues[0].keys[frontier_attribute[0].selector].GetSize(),
                            Total_Length);fflush(stdout);
                        if (enactor_stats[0].retval = data_slice->frontier_queues[0].keys[frontier_attribute[0].selector].EnsureSize(Total_Length)) break;
                        if (Problem::USE_DOUBLE_BUFFER)
                            if (enactor_stats[0].retval = data_slice->frontier_queues[0].values[frontier_attribute[0].selector].EnsureSize(Total_Length)) break;
                    }

                    offset=frontier_attribute[0].queue_length;
                    for (peer_=1;peer_<num_gpus;peer_++)
                    if (frontier_attribute[peer_].queue_length !=0) {
                        util::MemsetCopyVectorKernel<<<256,256, 0, streams[0]>>>(
                            data_slice->frontier_queues[0].keys[frontier_attribute[0].selector].GetPointer(util::DEVICE) + offset,
                            data_slice->frontier_queues[peer_   ].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                            frontier_attribute[peer_].queue_length);
                        if (Problem::USE_DOUBLE_BUFFER)
                            util::MemsetCopyVectorKernel<<<256,256,0,streams[0]>>>(
                                data_slice->frontier_queues[0].values[frontier_attribute[0].selector].GetPointer(util::DEVICE) + offset,
                                data_slice->frontier_queues[peer_   ].values[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                                frontier_attribute[peer_].queue_length);
                        offset+=frontier_attribute[peer_].queue_length;
                    }
                }
                frontier_attribute[0].queue_length = Total_Length;
                //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys", graph_slice->frontier_queues[0].keys[frontier_attribute[0].selector].GetPointer(util::DEVICE), Total_Length, thread_num, enactor_stats[0].iteration, -1, streams[0]);
                //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("labels", data_slice->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats[0].iteration, -1, streams[0]);
                //if (Iteration::BACKWARD) util::cpu_mt::PrintGPUArray<SizeT, Value>("deltas", data_slice->deltas.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats[0].iteration, -1, streams[0]);
 
                //if (enactor_stats[0].iteration<0)
                //    for (int i=0;i<num_gpus;i++) data_slice->out_length[i]=0;
                if (Iteration::HAS_FULLQ)// && enactor_stats[0].iteration>=0)
                {
                    peer_               = 0;
                    frontier_queue_     = &(data_slice->frontier_queues[peer_]);
                    frontier_attribute_ = &(frontier_attribute[peer_]);
                    enactor_stats_      = &(enactor_stats[peer_]);
                    work_progress_      = &(work_progress[peer_]);
                    iteration           = enactor_stats[peer_].iteration;
                    frontier_attribute_->queue_offset = 0;
                    frontier_attribute_->queue_reset  = true;

                    Iteration::FullQueue_Gather(
                        thread_num,
                        peer_,
                        frontier_attribute_,
                        enactor_stats_,
                        data_slice,
                        graph_slice,
                        streams[peer_]);
                    selector            = frontier_attribute[peer_].selector;
                    if (enactor_stats_->retval) break;
                    
                    if (Enactor::DEBUG) {printf("%d\t %lld\t \t Fullqueue started. Total_Length= %d\n", thread_num, enactor_stats[0].iteration, frontier_attribute_->queue_length);fflush(stdout);}
                    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys", graph_slice->frontier_queues[0].keys[frontier_attribute[0].selector].GetPointer(util::DEVICE), frontier_attribute[0].queue_length, thread_num, enactor_stats[0].iteration, -1, streams[0]);
                    
                    if (frontier_attribute_->queue_length !=0)
                    {
                        if (Enactor::DEBUG) {
                            mssg = "";
                            ShowDebugInfo<Problem>(
                                thread_num,
                                peer_,
                                frontier_attribute_,
                                enactor_stats_,
                                data_slice,
                                graph_slice,
                                work_progress_,
                                mssg,
                                streams[peer_]);
                        }

                        enactor_stats_->retval = Iteration::Compute_OutputLength(
                            frontier_attribute_,
                            graph_slice ->row_offsets     .GetPointer(util::DEVICE),
                            graph_slice ->column_indices  .GetPointer(util::DEVICE),
                            data_slice  ->frontier_queues  [peer_].keys[selector].GetPointer(util::DEVICE),
                            data_slice  ->scanned_edges    [peer_].GetPointer(util::DEVICE),
                            graph_slice ->nodes,//frontier_queues[peer_].keys[frontier_attribute[peer_].selector  ].GetSize(), 
                            graph_slice ->edges,//frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(),
                            context          [peer_][0],
                            streams          [peer_],
                            gunrock::oprtr::advance::V2V, true);
                        if (enactor_stats_->retval) break;

                        selector            = frontier_attribute[peer_].selector;
                        if (Enactor::SIZE_CHECK)
                        {
                            frontier_attribute[peer_].output_length.Move(util::DEVICE, util::HOST, 1, 0, streams[peer_]);
                            tretval = cudaStreamSynchronize(streams[peer_]);
                            if (tretval != cudaSuccess) {enactor_stats_->retval=tretval;break;}
                            if (Enactor::DEBUG)
                            {
                                printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                                    thread_num, iteration, peer_,
                                    frontier_queue_->keys[selector^1].GetSize(),
                                    frontier_attribute_->output_length[0]);fflush(stdout);}
                            //frontier_attribute_->output_length[0]+=1;
                            if (frontier_attribute_->output_length[0]+2 > frontier_queue_->keys[selector^1].GetSize())
                            {
                                printf("%d\t %d\t %d\t queue3  \t oversize :\t %d ->\t %d\n",
                                    thread_num, iteration, peer_,
                                    frontier_queue_->keys[selector^1].GetSize(),
                                    frontier_attribute_->output_length[0]+2);fflush(stdout);
                                if (enactor_stats_->retval = frontier_queue_->keys[selector  ].EnsureSize(frontier_attribute_->output_length[0]+2, true)) break;
                                if (enactor_stats_->retval = frontier_queue_->keys[selector^1].EnsureSize(frontier_attribute_->output_length[0]+2)) break;

                                if (Problem::USE_DOUBLE_BUFFER) {
                                    if (enactor_stats_->retval = frontier_queue_->values[selector  ].EnsureSize(frontier_attribute_->output_length[0]+2,true)) break;
                                    if (enactor_stats_->retval = frontier_queue_->values[selector^1].EnsureSize(frontier_attribute_->output_length[0]+2)) break;
                                }
                               //if (enactor_stats[peer_].retval = cudaDeviceSynchronize()) break;
                            }
                        }
                        
                        Iteration::FullQueue_Core(
                            thread_num,
                            peer_,
                            frontier_attribute_,
                            enactor_stats_,
                            data_slice,
                            s_data_slice[thread_num].GetPointer(util::DEVICE),
                            graph_slice,
                            work_progress_,
                            context[peer_],
                            streams[peer_]); 
                        if (enactor_stats_->retval) break;
                        if (enactor_stats_->retval = work_progress[peer_].GetQueueLength(
                            frontier_attribute_->queue_index,
                            frontier_attribute_->queue_length,
                            false,
                            streams[peer_],
                            true)) break;

                        if (!Enactor::SIZE_CHECK)
                        {
                            tretval = cudaStreamSynchronize(streams[peer_]);
                            if (tretval != cudaSuccess) {enactor_stats_->retval=tretval;break;}
                            //printf("output_length = %d, queue_size = %d\n", frontier_attribute[peer_].output_length[0], graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize());fflush(stdout);
                            if (frontier_attribute[peer_].output_length[0] > data_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize())
                            {
                                printf("%d\t %lld\t %d\t queue3  \t oversize :\t %d ->\t %d\n",
                                    thread_num, enactor_stats[peer_].iteration, peer_,
                                    data_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(),
                                    frontier_attribute[peer_].output_length[0]);fflush(stdout);
                                enactor_stats_->retval = util::GRError(cudaErrorLaunchOutOfResources, "queue3 oversize", __FILE__, __LINE__);
                                break;
                            }
                        }
                        selector = frontier_attribute[peer_].selector;
                        tretval = cudaStreamSynchronize(streams[peer_]);
                        if (tretval != cudaSuccess) {enactor_stats_->retval=tretval;break;}
                        Total_Length = frontier_attribute[peer_].queue_length;
                    } else {
                        Total_Length = 0;
                        for (int peer_=0;peer_<num_gpus;peer_++)
                            data_slice->out_length[peer_]=0;
                    }
                    if (Enactor::DEBUG) {printf("%d\t %lld\t \t Fullqueue finished. Total_Length= %d\n", thread_num, enactor_stats[0].iteration, Total_Length);fflush(stdout);}
                    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys  ", graph_slice->frontier_queues[0].keys[frontier_attribute[0].selector].GetPointer(util::DEVICE), Total_Length, thread_num, enactor_stats[0].iteration,-1, streams[0]);
                    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("labels", data_slice->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats[0].iteration, -1, streams[0]);
                    //if (Iteration::BACKWARD) util::cpu_mt::PrintGPUArray<SizeT, Value>("deltas", data_slice->deltas.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats[0].iteration, -1, streams[0]);
                    if (num_gpus==1) data_slice->out_length[0]=Total_Length;
                }

                if (num_gpus >1) {
                    selector=frontier_attribute[0].selector;
                    if (Problem::MARK_PREDECESSORS && Iteration::UPDATE_PREDECESSORS && Total_Length>0 )
                    {
                        Copy_Preds<VertexId, SizeT> <<<grid_size,256,0, streams[0]>>>(
                            Total_Length,
                            data_slice->frontier_queues[0].keys[selector].GetPointer(util::DEVICE),
                            data_slice->preds.GetPointer(util::DEVICE),
                            data_slice->temp_preds.GetPointer(util::DEVICE));

                        Update_Preds<VertexId,SizeT> <<<grid_size,256,0,streams[0]>>>(
                            Total_Length,
                            graph_slice->nodes,
                            data_slice->frontier_queues[0].keys[selector].GetPointer(util::DEVICE),
                            graph_slice->original_vertex.GetPointer(util::DEVICE),
                            data_slice->temp_preds.GetPointer(util::DEVICE),
                            data_slice->preds.GetPointer(util::DEVICE));//,
                    }

                    if (data_slice->keys_marker[0].GetSize() < Total_Length)
                    {
                        printf("%d\t %lld\t \t keys_marker\t oversize :\t %d ->\t %d \n",
                                thread_num, enactor_stats[0].iteration,
                                data_slice->keys_marker[0].GetSize(), Total_Length);fflush(stdout);
                        if (Enactor::SIZE_CHECK)
                        {
                            for (int peer_=0;peer_<num_gpus;peer_++)
                            {
                                if (enactor_stats[0].retval = data_slice->keys_marker[peer_].EnsureSize(Total_Length)) break;
                                data_slice->keys_markers[peer_]=data_slice->keys_marker[peer_].GetPointer(util::DEVICE);
                            }
                            if (enactor_stats[0].retval) break;
                            data_slice->keys_markers.Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);
                        } else {
                            enactor_stats[0].retval = util::GRError(cudaErrorLaunchOutOfResources, "keys_makrer oversize", __FILE__, __LINE__);
                            break;
                        }
                    }

                    if (Iteration::BACKWARD) Assign_Marker_Backward<VertexId, SizeT>
                        <<<grid_size,256, num_gpus * sizeof(SizeT*) ,streams[0]>>> (
                        Total_Length,
                        num_gpus,
                        data_slice->frontier_queues[0].keys[selector].GetPointer(util::DEVICE),
                        graph_slice->backward_offset   .GetPointer(util::DEVICE),
                        graph_slice->backward_partition.GetPointer(util::DEVICE),
                        data_slice ->keys_markers      .GetPointer(util::DEVICE));
                   else Assign_Marker<VertexId, SizeT>
                        <<<grid_size,256, num_gpus * sizeof(SizeT*) ,streams[0]>>> (
                        Total_Length,
                        num_gpus,
                        data_slice->frontier_queues[0].keys[selector].GetPointer(util::DEVICE),
                        graph_slice->partition_table.GetPointer(util::DEVICE),
                        data_slice->keys_markers.GetPointer(util::DEVICE));

                    for (int peer_=0;peer_<num_gpus;peer_++)
                    {
                        Scan<mgpu::MgpuScanTypeInc>(
                            (int*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                            Total_Length,
                            (int)0, mgpu::plus<int>(), (int*)0, (int*)0,
                            (int*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                            context[0][0]);
                    }

                    if (Total_Length>0) for (int peer_=0; peer_<num_gpus;peer_++)
                    {
                        cudaMemcpyAsync(&(data_slice->out_length[peer_]),
                            data_slice->keys_marker[peer_].GetPointer(util::DEVICE)
                                + (Total_Length -1),
                            sizeof(SizeT), cudaMemcpyDeviceToHost, streams[0]);
                    } else {
                        for (int peer_=0;peer_<num_gpus;peer_++)
                            data_slice->out_length[peer_]=0;
                    }
                    tretval = cudaStreamSynchronize(streams[0]);
                    if (tretval != cudaSuccess) {enactor_stats[0].retval=tretval;break;}

                    for (int peer_=0; peer_<num_gpus;peer_++)
                    {
                        SizeT org_size = (peer_==0? data_slice->frontier_queues[0].keys[frontier_attribute[0].selector^1].GetSize() : data_slice->keys_out[peer_].GetSize());
                        if (data_slice->out_length[peer_] > org_size)
                        {
                            printf("%d\t %lld\t %d\t keys_out\t oversize :\t %d ->\t %d\n",
                                   thread_num, enactor_stats[0].iteration, peer_,
                                   org_size, data_slice->out_length[peer_]);fflush(stdout);
                            if (Enactor::SIZE_CHECK)
                            {
                                if (peer_==0)
                                {
                                    data_slice->frontier_queues[0].keys[frontier_attribute[0].selector^1].EnsureSize(data_slice->out_length[0]);
                                } else {
                                    data_slice -> keys_out[peer_].EnsureSize(data_slice->out_length[peer_]);
                                    for (int i=0;i<NUM_VERTEX_ASSOCIATES;i++)
                                    {
                                        data_slice->vertex_associate_out [peer_][i].EnsureSize(data_slice->out_length[peer_]);
                                        data_slice->vertex_associate_outs[peer_][i] =
                                        data_slice->vertex_associate_out[peer_][i].GetPointer(util::DEVICE);
                                    }
                                    data_slice->vertex_associate_outs[peer_].Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);
                                    for (int i=0;i<NUM_VALUE__ASSOCIATES;i++)
                                    {
                                        data_slice->value__associate_out [peer_][i].EnsureSize(data_slice->out_length[peer_]);
                                        data_slice->value__associate_outs[peer_][i] =
                                            data_slice->value__associate_out[peer_][i].GetPointer(util::DEVICE);
                                    }
                                    data_slice->value__associate_outs[peer_].Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);
                                }
                            } else {
                                enactor_stats[0].retval = util::GRError(cudaErrorLaunchOutOfResources, "keys_out oversize", __FILE__, __LINE__);
                                break;
                            }
                        }
                    }
                    if (enactor_stats[0].retval) break;

                    for (int peer_=0;peer_<num_gpus;peer_++)
                        if (peer_==0) data_slice -> keys_outs[peer_] = data_slice->frontier_queues[peer_].keys[frontier_attribute[0].selector^1].GetPointer(util::DEVICE);
                        else data_slice -> keys_outs[peer_] = data_slice -> keys_out[peer_].GetPointer(util::DEVICE);
                    data_slice->keys_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);

                    offset = 0;
                    memcpy(&(data_slice -> make_out_array[offset]),
                             data_slice -> keys_markers         .GetPointer(util::HOST),
                              sizeof(SizeT*   ) * num_gpus);
                    offset += sizeof(SizeT*   ) * num_gpus ;
                    memcpy(&(data_slice -> make_out_array[offset]),
                             data_slice -> keys_outs            .GetPointer(util::HOST),
                              sizeof(VertexId*) * num_gpus);
                    offset += sizeof(VertexId*) * num_gpus ;
                    memcpy(&(data_slice -> make_out_array[offset]),
                             data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
                              sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
                    offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
                    memcpy(&(data_slice -> make_out_array[offset]),
                             data_slice -> value__associate_orgs.GetPointer(util::HOST),
                              sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
                    offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
                    for (int peer_=0; peer_<num_gpus; peer_++)
                    {
                        memcpy(&(data_slice->make_out_array[offset]),
                                 data_slice->vertex_associate_outs[peer_].GetPointer(util::HOST),
                                  sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
                        offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
                    }
                    for (int peer_=0; peer_<num_gpus; peer_++)
                    {
                        memcpy(&(data_slice->make_out_array[offset]),
                                data_slice->value__associate_outs[peer_].GetPointer(util::HOST),
                                  sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
                        offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
                    }
                    data_slice->make_out_array.Move(util::HOST, util::DEVICE, offset, 0, streams[0]);

                    if (Iteration::BACKWARD) 
                        Make_Out_Backward<VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                        <<<grid_size, 256, sizeof(char)*offset, streams[0]>>> (
                        Total_Length,
                        num_gpus,
                        data_slice-> frontier_queues[0].keys[frontier_attribute[0].selector].GetPointer(util::DEVICE),
                        graph_slice-> backward_offset        .GetPointer(util::DEVICE),
                        graph_slice-> backward_partition     .GetPointer(util::DEVICE),
                        graph_slice-> backward_convertion    .GetPointer(util::DEVICE),
                        offset,
                        data_slice -> make_out_array         .GetPointer(util::DEVICE));
                    else Make_Out<VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                        <<<grid_size, 256, sizeof(char)*offset, streams[0]>>> (
                        Total_Length,
                        num_gpus,
                        data_slice->frontier_queues[0].keys[frontier_attribute[0].selector].GetPointer(util::DEVICE),
                        graph_slice-> partition_table        .GetPointer(util::DEVICE),
                        graph_slice-> convertion_table       .GetPointer(util::DEVICE),
                        offset,
                        data_slice -> make_out_array         .GetPointer(util::DEVICE));

                    //if (enactor_stats[0].retval = util::GRError(cudaStreamSynchronize(streams[0]), "Make_Out error", __FILE__, __LINE__)) break;
                    //if (!SIZE_CHECK)
                    //{
                    //    for (int peer_=0;peer_<num_gpus;peer_++)
                    //        cudaMemcpyAsync(&(data_slice->out_length[peer_]),
                    //            data_slice->keys_marker[peer_].GetPointer(util::DEVICE)
                    //                + (Total_Length -1),
                    //            sizeof(SizeT), cudaMemcpyDeviceToHost, streams[0]);
                    //}
               
                   tretval = cudaStreamSynchronize(streams[0]);
                   if (tretval != cudaSuccess) {enactor_stats[0].retval=tretval;break;}
                   frontier_attribute[0].selector^=1;
                    //if (enactor_stats[0].retval = util::GRError(cudaStreamSynchronize(streams[0]), "MemcpyAsync keys_marker error", __FILE__, __LINE__)) break;
                }
            } /*else {
                for (int peer_=0;peer_<num_gpus;peer_++)
                    data_slice->out_length[peer_]=0;
            }*/
            for (int peer_=0;peer_<num_gpus;peer_++)
                frontier_attribute[peer_].queue_length = data_slice->out_length[peer_];
            //util::cpu_mt::PrintCPUArray<SizeT, SizeT>("out_length", data_slice->out_length.GetPointer(util::HOST), num_gpus, thread_num, enactor_stats[0].iteration);

        } /*else if (!Iteration::Stop_Condition(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) {
            if (enactor_stats[0].retval = work_progress[0].GetQueueLength(frontier_attribute[0].queue_index, frontier_attribute[0].queue_length, false, data_slice->streams[0])) break;
        }*/
        //util::cpu_mt::PrintMessage("Iteration end",thread_num,enactor_stats->iteration);
        Iteration::Iteration_Change(enactor_stats->iteration);
        //if (DEBUG) printf("\n%lld", (long long) enactor_stats->iteration);
    }

}
    
/**
 * @brief Base class for graph problem enactors.
 */
template <
    typename SizeT,
    bool     _DEBUG,  // if DEBUG is set, print details to stdout
    bool     _SIZE_CHECK>
class EnactorBase
{
public:  
    static const bool DEBUG = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;
    int           num_gpus;
    int          *gpu_idx;
    FrontierType  frontier_type;
 
    //Device properties
    util::Array1D<SizeT, util::CudaProperties>          cuda_props        ;

    // Queue size counters and accompanying functionality
    util::Array1D<SizeT, util::CtaWorkProgressLifetime> work_progress     ;
    util::Array1D<SizeT, EnactorStats>                  enactor_stats     ;
    util::Array1D<SizeT, FrontierAttribute<SizeT> >     frontier_attribute;

    FrontierType GetFrontierType() {return frontier_type;}

protected:  

    /**
     * @brief Constructor
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] DEBUG If set, will collect kernel running stats and display the running info.
     */
    EnactorBase(
        FrontierType  frontier_type, 
        int           num_gpus, 
        int          *gpu_idx)
    {
        //util::cpu_mt::PrintMessage("EnactorBase() begin.");
        this->frontier_type = frontier_type;
        this->num_gpus      = num_gpus;
        this->gpu_idx       = gpu_idx;
        cuda_props        .SetName("cuda_props"        );
        work_progress     .SetName("work_progress"     );
        enactor_stats     .SetName("enactor_stats"     );
        frontier_attribute.SetName("frontier_attribute");
        cuda_props        .Init(num_gpus         , util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        work_progress     .Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable); 
        enactor_stats     .Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        frontier_attribute.Init(num_gpus*num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            // Setup work progress (only needs doing once since we maintain
            // it in our kernel code)
            cuda_props   [gpu].Setup(gpu_idx[gpu]);
            for (int peer=0;peer<num_gpus;peer++)
            {
                work_progress     [gpu*num_gpus+peer].Setup();
                //enactor_stats[gpu*num_gpus+peer].node_locks    .SetName("node_locks"    );
                //enactor_stats[gpu*num_gpus+peer].node_locks_out.SetName("node_locks_out");
                frontier_attribute[gpu*num_gpus+peer].output_length.Allocate(1, util::HOST | util::DEVICE);
            }
        }
        //util::cpu_mt::PrintMessage("EnactorBase() end.");
    }


    virtual ~EnactorBase()
    {
        //util::cpu_mt::PrintMessage("~EnactorBase() begin.");
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            for (int peer=0;peer<num_gpus;peer++)
            {
                enactor_stats     [gpu*num_gpus+peer].node_locks    .Release();
                enactor_stats     [gpu*num_gpus+peer].node_locks_out.Release();
                frontier_attribute[gpu*num_gpus+peer].output_length .Release();
                if (work_progress [gpu*num_gpus+peer].HostReset()) return;
            }
        }
        work_progress     .Release();
        cuda_props        .Release();
        enactor_stats     .Release();
        frontier_attribute.Release();
        //util::cpu_mt::PrintMessage("~EnactorBase() end.");
    }

    template <typename Problem>
    cudaError_t Init(
        Problem *problem,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256)
    { 
        //util::cpu_mt::PrintMessage("EnactorBase Init() begin.");
        cudaError_t retval = cudaSuccess;

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            for (int peer=0;peer<num_gpus;peer++)
            {
                //initialize runtime stats
                enactor_stats[gpu*num_gpus+peer].advance_grid_size = MaxGridSize(gpu, advance_occupancy, max_grid_size);
                enactor_stats[gpu*num_gpus+peer].filter_grid_size  = MaxGridSize(gpu, filter_occupancy, max_grid_size);

                if (retval = enactor_stats[gpu*num_gpus+peer].advance_kernel_stats.Setup(enactor_stats[gpu].advance_grid_size)) return retval;
                if (retval = enactor_stats[gpu*num_gpus+peer]. filter_kernel_stats.Setup(enactor_stats[gpu]. filter_grid_size)) return retval;
                if (retval = enactor_stats[gpu*num_gpus+peer].node_locks.Allocate(node_lock_size,util::DEVICE)) return retval;
                if (retval = enactor_stats[gpu*num_gpus+peer].node_locks_out.Allocate(node_lock_size, util::DEVICE)) return retval;
            }
        }
        //util::cpu_mt::PrintMessage("EnactorBase Setup() end.");
        return retval;
    }

    cudaError_t Reset()
    {
        //util::cpu_mt::PrintMessage("EnactorBase Reset() begin.");
        cudaError_t retval = cudaSuccess;

        for (int gpu=0;gpu<num_gpus*num_gpus;gpu++)
        {
            enactor_stats[gpu].iteration             = 0;
            enactor_stats[gpu].total_runtimes        = 0;
            enactor_stats[gpu].total_lifetimes       = 0;
            enactor_stats[gpu].total_queued          = 0;
        }
        //util::cpu_mt::PrintMessage("EnactorBase Reset() end.");
        return retval;
    }

    template <typename Problem>
    cudaError_t Setup(
        Problem *problem,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256)
    {
        cudaError_t retval = cudaSuccess;

        if (retval = Init(problem, max_grid_size, advance_occupancy, filter_occupancy, node_lock_size)) return retval;
        if (retval = Reset()) return retval;
        return retval;
    }

    /**
     * @brief Utility function for getting the max grid size.
     *
     * @param[in] cta_occupancy CTA occupancy for current architecture
     * @param[in] max_grid_size Preset max grid size. If less or equal to 0, fully populate all SMs
     *
     * \return The maximum number of threadblocks this enactor class can launch.
     */
    int MaxGridSize(int gpu, int cta_occupancy, int max_grid_size = 0)
    {
        if (max_grid_size <= 0) {
            max_grid_size = this->cuda_props[gpu].device_props.multiProcessorCount * cta_occupancy;
        }

        return max_grid_size;
    } 
};

template <typename AdvanceKernelPolicy, typename FilterKernelPolicy, typename Enactor>
struct IterationBase
{
public:
    typedef typename Enactor::SizeT      SizeT     ;   
    typedef typename Enactor::Value      Value     ;   
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
    static const bool INSTRUMENT = Enactor::INSTRUMENT;
    static const bool DEBUG      = Enactor::DEBUG;
    static const bool SIZE_CHECK = Enactor::SIZE_CHECK;

    static void SubQueue_Gather(
        int                            thread_num,
        int                            peer_,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        GraphSlice                    *graph_slice,
        cudaStream_t                   stream)
    {
    }

    static void SubQueue_Core(
        //bool     DEBUG,
        int                            thread_num,
        int                            peer_,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    static void FullQueue_Gather(
        int                            thread_num,
        int                            peer_,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        GraphSlice                    *graph_slice,
        cudaStream_t                   stream)
    {
    }

    static void FullQueue_Core(
        int                            thread_num,
        int                            peer_,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    static bool Stop_Condition(
        EnactorStats    *enactor_stats,
        FrontierAttribute<SizeT> *frontier_attribute,
        util::Array1D<SizeT, DataSlice> *data_slice,
        int num_gpus)
    {
        //printf("Normal Stop checked\n");fflush(stdout);
        return All_Done(enactor_stats,frontier_attribute,data_slice,num_gpus);
    }

    static void Iteration_Change(long long &iterations)
    {
        iterations++;
    }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
