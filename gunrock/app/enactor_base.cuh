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

#include <gunrock/oprtr/advance/kernel_policy.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {

struct EnactorStats
{
    long long           iteration;
//    int                 num_gpus;
//    int                 gpu_idx;

    unsigned long long  total_lifetimes;
    unsigned long long  total_runtimes;
    unsigned long long  total_queued;

    unsigned int        advance_grid_size;
    unsigned int        filter_grid_size;

    util::KernelRuntimeStatsLifetime advance_kernel_stats;
    util::KernelRuntimeStatsLifetime filter_kernel_stats;

    //unsigned int        *d_node_locks;
    //unsigned int        *d_node_locks_out;
    util::Array1D<int, unsigned int> node_locks;
    util::Array1D<int, unsigned int> node_locks_out;

//    volatile int       *done;
//    int                *d_done;
    //cudaEvent_t        throttle_event;
    cudaError_t        retval;
    clock_t            start_time;

    EnactorStats()
    {
         util::cpu_mt::PrintMessage("EnactorStats() begin.");
         util::cpu_mt::PrintMessage("EnactorStats() end.");
    }
};

struct FrontierAttribute
{
    unsigned int        queue_length;
    unsigned int        output_length;
    unsigned int        queue_index;
    int                 selector;
    bool                queue_reset;
    int                 current_label;
    gunrock::oprtr::advance::TYPE   advance_type;
};

bool All_Done(EnactorStats *enactor_stats,FrontierAttribute *frontier_attribute,int num_gpus)
{   
    for (int gpu=0;gpu<num_gpus;gpu++)
    if (enactor_stats[gpu].retval!=cudaSuccess)
    {   
        printf("(CUDA error %d @ GPU %d: %s\n", enactor_stats[gpu].retval, gpu, cudaGetErrorString(enactor_stats[gpu].retval)); fflush(stdout);
        return true;
    }   

    for (int gpu=0;gpu<num_gpus;gpu++)
    if (frontier_attribute[gpu].queue_length!=0)
    {   
        return false;
    }   
    return true;
} 

    template <typename VertexId, typename SizeT>
    __global__ void Copy_Preds (
        const SizeT     num_elements,
        const VertexId* keys,
        const VertexId* in_preds,
              VertexId* out_preds)
    {   
        VertexId x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        if (x>=num_elements) return;
        VertexId t = keys[x];
        out_preds[t]=in_preds[t];
    }   

    template <typename VertexId, typename SizeT>
    __global__ void Update_Preds (
        const SizeT     num_elements,
        const VertexId* keys,
        const VertexId* org_vertexs,
        const VertexId* in_preds,
              VertexId* out_preds)
    {   
        VertexId x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        /*long long x= blockIdx.y;
        x = x*gridDim.x+blockIdx.x;
        x = x*blockDim.y+threadIdx.y;
        x = x*blockDim.x+threadIdx.x;*/

        if (x>=num_elements) return;
        VertexId t = keys[x];
        VertexId p = in_preds[t];
        out_preds[t]=org_vertexs[p];
    }   

    template <typename VertexId, typename SizeT>
    __global__ void Mark_Queue (
        const SizeT     num_elements,
        const VertexId* keys,
              unsigned char* marker)
    {
        VertexId x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        if (x< num_elements) marker[keys[x]]=1;
    }

    template <
        typename SizeT, 
        typename VertexId,
        typename Value,
        typename GraphSlice,
        typename DataSlice,
        SizeT    num_vertex_associate,
        SizeT    num_value__associate>
    void UpdateNeiborForward (
        SizeT        num_elements,
        int          num_gpus,
        int          thread_num,
        util::scan::MultiScan<VertexId, SizeT, true, 256, 8, Value>* Scaner,
        GraphSlice        **s_graph_slice,
        util::Array1D<SizeT,DataSlice> *s_data_slice,
        EnactorStats      *s_enactor_stats,
        FrontierAttribute *s_frontier_attribute,
        unsigned char *d_marker = NULL)
    {
        FrontierAttribute
                     *frontier_attribute  = &(s_frontier_attribute [thread_num]);
        EnactorStats *enactor_stats       = &(s_enactor_stats      [thread_num]);
        GraphSlice   *graph_slice         =   s_graph_slice        [thread_num]; 
        util::Array1D<SizeT, DataSlice>
                     *data_slice          = &(s_data_slice         [thread_num]);
        SizeT        *out_offset          = new SizeT [num_gpus+1];

        if (num_elements ==0)
        {
            if (d_marker!=NULL)
                util::MemsetKernel<<<128, 128>>>(d_marker, (unsigned char)0, graph_slice->nodes);
            for (int peer=0;peer<num_gpus;peer++)
            {
                int gpu_ = peer<thread_num? thread_num: thread_num+1;
                if (peer == thread_num) continue;
                s_data_slice[peer]->in_length[enactor_stats->iteration%2][gpu_]=0;
            }
            data_slice[0]->out_length[0]=0;
            return;
        }
 
        Scaner->template Scan_with_dKeys2
            <num_vertex_associate, num_value__associate> (
            num_elements,
            num_gpus,
            //num_vertex_associate, 
            //num_value__associate,
            graph_slice->frontier_queues.keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            graph_slice->frontier_queues.keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            graph_slice->partition_table .GetPointer(util::DEVICE),
            graph_slice->convertion_table.GetPointer(util::DEVICE),
            data_slice[0]->out_length           .GetPointer(util::DEVICE),
            data_slice[0]->vertex_associate_orgs.GetPointer(util::DEVICE),
            data_slice[0]->vertex_associate_outs.GetPointer(util::DEVICE),
            data_slice[0]->value__associate_orgs.GetPointer(util::DEVICE),
            data_slice[0]->value__associate_outs.GetPointer(util::DEVICE));
        if (enactor_stats->retval = data_slice[0]->out_length.Move(util::DEVICE, util::HOST)) return;
        //util::cpu_mt::PrintCPUArray<SizeT, SizeT>("out_length",data_slice[0]->out_length.GetPointer(util::HOST),num_gpus,thread_num,enactor_stats->iteration);
        out_offset[0]=0;
        for (int i=0;i<num_gpus;i++) out_offset[i+1]=out_offset[i]+data_slice[0]->out_length[i];
        
        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;
        if (d_marker!=NULL)
        {
            util::MemsetKernel<<<128, 128>>>(d_marker, (unsigned char)0, graph_slice->nodes);
            if (data_slice[0]->out_length[0]>0)
            {
                int grid_size = (data_slice[0]->out_length[0]%256)==0?
                                 data_slice[0]->out_length[0]/256:
                                 data_slice[0]->out_length[0]/256+1;
                Mark_Queue<VertexId, SizeT> <<<grid_size, 256>>> (
                    data_slice[0]->out_length[0],
                    graph_slice -> frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                    d_marker);
            }
        }
        for (int peer=0; peer<num_gpus; peer++)
        {
            if (peer == thread_num) continue;
            int peer_ = peer<thread_num? peer+1     : peer;
            int gpu_  = peer<thread_num? thread_num : thread_num+1;
            s_data_slice[peer]->in_length[enactor_stats->iteration%2][gpu_]
                      = data_slice[0]->out_length[peer_];
            if (data_slice[0]->out_length[peer_] == 0) continue;
            //s_enactor_stats[peer].done[0]=-1;
            //printf("%d\t %d\t %p+%d ==> %p+%d @ %d,%d\n", thread_num, enactor_stats->iteration, s_data_slice[peer]->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE), s_graph_slice[peer]->in_offset[gpu_], graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE), out_offset[peer_], peer, data_slice[0]->out_length[peer_]);
            if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
                s_data_slice[peer] -> keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE)
                                      + s_graph_slice[peer] -> in_offset[gpu_],
                graph_slice -> frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE)
                                      + out_offset[peer_],
                sizeof(VertexId) * data_slice[0]->out_length[peer_], cudaMemcpyDefault, data_slice[0]->streams[peer_]),
                "cudaMemcpyPeer d_keys failed", __FILE__, __LINE__)) break;
                
            for (int i=0;i<num_vertex_associate;i++)
            {   
                if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
                    s_data_slice[peer]->vertex_associate_ins[enactor_stats->iteration%2][i]
                        + s_graph_slice[peer]->in_offset[gpu_],
                    data_slice[0]->vertex_associate_outs[i]
                        + (out_offset[peer_] - out_offset[1]),
                    sizeof(VertexId) * data_slice[0]->out_length[peer_], cudaMemcpyDefault, data_slice[0]->streams[peer_]),
                    "cudaMemcpyPeer vertex_associate_out failed", __FILE__, __LINE__)) break;
            }
            if (enactor_stats->retval) break;   

            for (int i=0;i<num_value__associate;i++)
            {   
                if (enactor_stats->retval = util::GRError(cudaMemcpyAsync(
                    s_data_slice[peer]->value__associate_ins[enactor_stats->iteration%2][i]
                        + s_graph_slice[peer]->in_offset[gpu_],
                    data_slice[0]->value__associate_outs[i]
                        + (out_offset[peer_] - out_offset[1]),
                    sizeof(Value) * data_slice[0]->out_length[peer_], cudaMemcpyDefault, data_slice[0]->streams[peer_]),
                    "cudaMemcpyPeer value__associate_out failed", __FILE__, __LINE__)) break;
            }
            if (enactor_stats->retval) break;
        }
        delete[] out_offset;out_offset=NULL;
    }

    template <
        typename SizeT, 
        typename VertexId,
        typename Value,
        typename GraphSlice,
        typename DataSlice,
        SizeT    num_vertex_associate,
        SizeT    num_value__associate>
    void UpdateNeiborBackward (
        SizeT        num_elements,
        int          num_gpus,
        int          thread_num,
        util::scan::MultiScan<VertexId, SizeT, true, 256, 8, Value>* Scaner,
        GraphSlice        **s_graph_slice,
        util::Array1D<SizeT, DataSlice> *s_data_slice,
        EnactorStats      *s_enactor_stats,
        FrontierAttribute *s_frontier_attribute)
    {
        FrontierAttribute
                     *frontier_attribute  = &(s_frontier_attribute [thread_num]);
        EnactorStats *enactor_stats       = &(s_enactor_stats      [thread_num]);
        GraphSlice   *graph_slice         =   s_graph_slice        [thread_num] ; 
        util::Array1D<SizeT, DataSlice>
                     *data_slice          = &(s_data_slice         [thread_num]);
        SizeT        *out_offset          = new SizeT [num_gpus+1];

        if (num_elements ==0)
        {
            for (int peer=0;peer<num_gpus;peer++)
            {
                int gpu_ = peer<thread_num? thread_num: thread_num+1;
                if (peer == thread_num) continue;
                s_data_slice[peer]->in_length[enactor_stats->iteration%2][gpu_]=0;
            }
            data_slice[0]->out_length[0]=0;
            return;
        }

        Scaner->template Scan_with_dKeys_Backward
            <num_vertex_associate, num_value__associate> (
            num_elements,
            num_gpus,
            graph_slice->frontier_queues.keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            graph_slice->backward_offset    .GetPointer(util::DEVICE),
            graph_slice->frontier_queues.keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            graph_slice->backward_partition .GetPointer(util::DEVICE),
            graph_slice->backward_convertion.GetPointer(util::DEVICE),
            data_slice[0]->out_length           .GetPointer(util::DEVICE),
            data_slice[0]->vertex_associate_orgs.GetPointer(util::DEVICE),
            data_slice[0]->vertex_associate_outs.GetPointer(util::DEVICE),
            data_slice[0]->value__associate_orgs.GetPointer(util::DEVICE),
            data_slice[0]->value__associate_outs.GetPointer(util::DEVICE));
        if (enactor_stats->retval = data_slice[0]->out_length.Move(util::DEVICE, util::HOST)) return;
        util::cpu_mt::PrintCPUArray<SizeT,SizeT>("out_length", data_slice[0]->out_length.GetPointer(util::HOST), num_gpus, thread_num, enactor_stats->iteration);
        out_offset[0]=0;
        for (int i=0;i<num_gpus;i++) out_offset[i+1]=out_offset[i]+data_slice[0]->out_length[i];
        
        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;
        
        for (int peer=0; peer<num_gpus; peer++)
        {
            if (peer == thread_num) continue;
            int peer_ = peer<thread_num? peer+1     : peer;
            int gpu_  = peer<thread_num? thread_num : thread_num+1;
            s_data_slice[peer]->in_length[enactor_stats->iteration%2][gpu_]
                      = data_slice[0]->out_length[peer_];
            if (data_slice[0]->out_length[peer_] == 0) continue;
            //s_enactor_stats[peer].done[0]=-1;
            printf("%d\t %d\t %p+%d <== %p+%d @ %d,%d\n", thread_num, enactor_stats->iteration, s_data_slice[peer]->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE), s_graph_slice[peer]->out_offset[gpu_]-s_graph_slice[peer]->out_offset[1], graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE), out_offset[peer_], peer, data_slice[0]->out_length[peer_]);
            if (enactor_stats->retval = util::GRError(cudaMemcpy(
                s_data_slice[peer] -> keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE)
                     + (s_graph_slice[peer] -> out_offset[gpu_] - s_graph_slice[peer]->out_offset[1]),
                graph_slice -> frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE)
                     + out_offset[peer_],
                sizeof(VertexId) * data_slice[0]->out_length[peer_], cudaMemcpyDefault),
                "cudaMemcpyPeer d_keys failed", __FILE__, __LINE__)) break;
                
            for (int i=0;i<num_vertex_associate;i++)
            {   
                if (enactor_stats->retval = util::GRError(cudaMemcpy(
                    s_data_slice[peer]->vertex_associate_ins[enactor_stats->iteration%2][i]
                        + (s_graph_slice[peer]->out_offset[gpu_] - s_graph_slice[peer]->out_offset[1]),
                    data_slice[0]->vertex_associate_outs[i]
                        + (out_offset[peer_] - out_offset[1]),
                    sizeof(VertexId) * data_slice[0]->out_length[peer_], cudaMemcpyDefault),
                    "cudaMemcpyPeer vertex_associate_out failed", __FILE__, __LINE__)) break;
            }
            if (enactor_stats->retval) break;   

            for (int i=0;i<num_value__associate;i++)
            {   
                if (enactor_stats->retval = util::GRError(cudaMemcpy(
                    s_data_slice[peer]->value__associate_ins[enactor_stats->iteration%2][i]
                        + (s_graph_slice[peer]->out_offset[gpu_] - s_graph_slice[peer]->out_offset[1]),
                    data_slice[0]->value__associate_outs[i]
                        + (out_offset[peer_] - out_offset[1]),
                    sizeof(Value) * data_slice[0]->out_length[peer_], cudaMemcpyDefault),
                    "cudaMemcpyPeer value__associate_out failed", __FILE__, __LINE__)) break;
            }
            if (enactor_stats->retval) break;
        }
        delete[] out_offset;out_offset=NULL;
    }
/**
 * @brief Base class for graph problem enactors.
 */
class EnactorBase
{
public:  

    int                             num_gpus;
    int                             *gpu_idx;
    //Device properties
    //util::CudaProperties            cuda_props;
    //util::CudaProperties            *cuda_props;
    util::Array1D<int, util::CudaProperties> cuda_props;

    // Queue size counters and accompanying functionality
    //util::CtaWorkProgressLifetime   work_progress;
    //util::CtaWorkProgressLifetime   *work_progress;
    util::Array1D<int, util::CtaWorkProgressLifetime> work_progress;

    FrontierType                    frontier_type;

    //EnactorStats                    enactor_stats;
    //EnactorStats                    *enactor_stats;
    util::Array1D<int, EnactorStats> enactor_stats;

    //FrontierAttribute               frontier_attribute;
    //FrontierAttribute               *frontier_attribute;
    util::Array1D<int, FrontierAttribute> frontier_attribute;

    // if DEBUG is set, print details to stdout
    bool DEBUG;

    FrontierType GetFrontierType() { return frontier_type;}

protected:  

    /**
     * @brief Constructor
     *
     * @param[in] frontier_type The frontier type (i.e., edge/vertex/mixed)
     * @param[in] DEBUG If set, will collect kernel running stats and display the running info.
     */
    EnactorBase(FrontierType frontier_type, bool DEBUG,
                int num_gpus, int* gpu_idx) :
        frontier_type(frontier_type),
        DEBUG(DEBUG)
    {
        util::cpu_mt::PrintMessage("EnactorBase() begin.");
        this->num_gpus     = num_gpus;
        this->gpu_idx      = gpu_idx;
        cuda_props        .SetName("cuda_props"        );
        work_progress     .SetName("work_progress"     );
        enactor_stats     .SetName("enactor_stats"     );
        frontier_attribute.SetName("frontier_attribute");
        //cuda_props         = new util::CudaProperties          [num_gpus];
        //work_progress      = new util::CtaWorkProgressLifetime [num_gpus];
        //enactor_stats      = new EnactorStats                  [num_gpus];
        //frontier_attribute = new FrontierAttribute             [num_gpus];
        cuda_props        .Init(num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        work_progress     .Init(num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable); 
        enactor_stats     .Init(num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        frontier_attribute.Init(num_gpus, util::HOST, true, cudaHostAllocMapped | cudaHostAllocPortable);
        
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            // Setup work progress (only needs doing once since we maintain
            // it in our kernel code)
            work_progress[gpu].Setup();
            cuda_props   [gpu].Setup(gpu_idx[gpu]);
            //enactor_stats[gpu].num_gpus = num_gpus;
            //enactor_stats[gpu].gpu_idx  = gpu_idx[gpu];
            enactor_stats[gpu].node_locks    .SetName("node_locks"    );
            enactor_stats[gpu].node_locks_out.SetName("node_locks_out");
            //enactor_stats.d_node_locks = NULL;
            //enactor_stats.d_node_locks_out = NULL;
        }
        util::cpu_mt::PrintMessage("EnactorBase() end.");
    }


    virtual ~EnactorBase()
    {
        util::cpu_mt::PrintMessage("~EnactorBase() begin.");
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (util::SetDevice(gpu_idx[gpu])) return;
            enactor_stats[gpu].node_locks    .Release();
            enactor_stats[gpu].node_locks_out.Release();
            if (work_progress[gpu].HostReset()) return;
            //if (util::GRError(cudaFreeHost((void*)enactor_stats[gpu].done), 
            //     "EnactorBase cudaFreeHost done failed", __FILE__, __LINE__)) return;
            //if (util::GRError(cudaEventDestroy(enactor_stats[gpu].throttle_event),
            //     "EnactorBase cudaEventDestroy throttle_event failed", __FILE__, __LINE__)) return;
            //if (enactor_stats.d_node_locks) util::GRError(cudaFree(enactor_stats.d_node_locks), "EnactorBase cudaFree d_node_locks failed", __FILE__, __LINE__);
            //if (enactor_stats.d_node_locks_out) util::GRError(cudaFree(enactor_stats.d_node_locks_out), "EnactorBase cudaFree d_node_locks_out failed", __FILE__, __LINE__);
        }
        work_progress     .Release();
        cuda_props        .Release();
        enactor_stats     .Release();
        frontier_attribute.Release();
        //delete[] work_progress     ; work_progress      = NULL;
        //delete[] cuda_props        ; cuda_props         = NULL;
        //delete[] enactor_stats     ; enactor_stats      = NULL;
        //delete[] frontier_attribute; frontier_attribute = NULL;
        util::cpu_mt::PrintMessage("~EnactorBase() end.");
    }

    template <typename ProblemData>
    cudaError_t Init(
        ProblemData *problem,
        int max_grid_size,
        int advance_occupancy,
        int filter_occupancy,
        int node_lock_size = 256)
    { 
        util::cpu_mt::PrintMessage("EnactorBase Init() begin.");
        cudaError_t retval = cudaSuccess;

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            //initialize runtime stats
            enactor_stats[gpu].advance_grid_size = MaxGridSize(gpu, advance_occupancy, max_grid_size);
            enactor_stats[gpu].filter_grid_size  = MaxGridSize(gpu, filter_occupancy, max_grid_size);

            if (retval = enactor_stats[gpu].advance_kernel_stats.Setup(enactor_stats[gpu].advance_grid_size)) return retval;
            if (retval = enactor_stats[gpu]. filter_kernel_stats.Setup(enactor_stats[gpu]. filter_grid_size)) return retval;
            //initialize the host-mapped "done"
            //int flags = cudaHostAllocMapped;

            // Allocate pinned memory for done
            //if (retval = util::GRError(cudaHostAlloc((void**)&(enactor_stats[gpu].done), sizeof(int) * 1, flags),
            //        "BFSEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) return retval;

            // Map done into GPU space
            //if (retval = util::GRError(cudaHostGetDevicePointer((void**)&(enactor_stats[gpu].d_done), (void*) enactor_stats[gpu].done, 0),  
            //        "BFSEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) return retval;

            // Create throttle event
            //if (retval = util::GRError(cudaEventCreateWithFlags(&enactor_stats[gpu].throttle_event, cudaEventDisableTiming),                
            //        "BFSEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) return retval;
                
            //enactor_stats[gpu].iteration             = 0;
            //enactor_stats[gpu].total_runtimes        = 0;
            //enactor_stats[gpu].total_lifetimes       = 0;
            //enactor_stats[gpu].total_queued          = 0;
            //enactor_stats[gpu].done[0]               = -1;
            //enactor_stats[gpu].retval                = cudaSuccess;
            //enactor_stats.num_gpus              = 1;
            //enactor_stats.gpu_id                = 0;

            //if (retval = util::GRError(cudaMalloc(
            //                (void**)&enactor_stats.d_node_locks,
            //                node_lock_size * sizeof(unsigned int)),
            //            "EnactorBase cudaMalloc d_node_locks failed", __FILE__, __LINE__)) return retval;
            if (retval = enactor_stats[gpu].node_locks.Allocate(node_lock_size,util::DEVICE)) return retval;

            //if (retval = util::GRError(cudaMalloc(
            //                (void**)&enactor_stats.d_node_locks_out,
            //                node_lock_size * sizeof(unsigned int)),
            //            "EnactorBase cudaMalloc d_node_locks_out failed", __FILE__, __LINE__)) return retval;
            if (retval = enactor_stats[gpu].node_locks_out.Allocate(node_lock_size, util::DEVICE)) return retval;
        }
        util::cpu_mt::PrintMessage("EnactorBase Setup() end.");
        return retval;
    }

    //template <typename ProblemData>
    cudaError_t Reset()
        //ProblemData *problem,
        //int max_grid_size,
        //int advance_occupancy,
        //int filter_occupancy,
        //int node_lock_size = 256)
    {
        util::cpu_mt::PrintMessage("EnactorBase Reset() begin.");
        cudaError_t retval = cudaSuccess;

        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            /*if (retval = util::SetDevice(gpu_idx[gpu])) return retval;
            //initialize runtime stats
            enactor_stats[gpu].advance_grid_size = MaxGridSize(gpu, advance_occupancy, max_grid_size);
            enactor_stats[gpu].filter_grid_size  = MaxGridSize(gpu, filter_occupancy, max_grid_size);

            if (retval = enactor_stats[gpu].advance_kernel_stats.Setup(enactor_stats[gpu].advance_grid_size)) return retval;
            if (retval = enactor_stats[gpu]. filter_kernel_stats.Setup(enactor_stats[gpu]. filter_grid_size)) return retval;
            //initialize the host-mapped "done"
            int flags = cudaHostAllocMapped;

            // Allocate pinned memory for done
            if (retval = util::GRError(cudaHostAlloc((void**)&(enactor_stats[gpu].done), sizeof(int) * 1, flags),
                    "BFSEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) return retval;

            // Map done into GPU space
            if (retval = util::GRError(cudaHostGetDevicePointer((void**)&(enactor_stats[gpu].d_done), (void*) enactor_stats[gpu].done, 0),  
                    "BFSEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) return retval;

            // Create throttle event
            if (retval = util::GRError(cudaEventCreateWithFlags(&enactor_stats[gpu].throttle_event, cudaEventDisableTiming),                
                    "BFSEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) return retval;
            */
    
            enactor_stats[gpu].iteration             = 0;
            enactor_stats[gpu].total_runtimes        = 0;
            enactor_stats[gpu].total_lifetimes       = 0;
            enactor_stats[gpu].total_queued          = 0;
            //enactor_stats[gpu].done[0]               = -1;
            //enactor_stats[gpu].retval                = cudaSuccess;
            //enactor_stats.num_gpus              = 1;
            //enactor_stats.gpu_id                = 0;

            //if (retval = util::GRError(cudaMalloc(
            //                (void**)&enactor_stats.d_node_locks,
            //                node_lock_size * sizeof(unsigned int)),
            //            "EnactorBase cudaMalloc d_node_locks failed", __FILE__, __LINE__)) return retval;
            //if (retval = enactor_stats[gpu].node_locks.Allocate(node_lock_size,util::DEVICE)) return retval;

            //if (retval = util::GRError(cudaMalloc(
            //                (void**)&enactor_stats.d_node_locks_out,
            //                node_lock_size * sizeof(unsigned int)),
            //            "EnactorBase cudaMalloc d_node_locks_out failed", __FILE__, __LINE__)) return retval;
            //if (retval = enactor_stats[gpu].node_locks_out.Allocate(node_lock_size, util::DEVICE)) return retval;
        }
        util::cpu_mt::PrintMessage("EnactorBase Reset() end.");
        return retval;
    }

    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem,
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


} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
