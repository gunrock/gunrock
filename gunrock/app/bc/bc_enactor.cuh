// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bc_enactor.cuh
 *
 * @brief BC Problem Enactor
 */

#pragma once

#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/scan/multi_scan.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bc/bc_problem.cuh>
#include <gunrock/app/bc/bc_functor.cuh>


namespace gunrock {
namespace app {
namespace bc {

    template <typename BCProblem, bool INSTRUMENT> class BCEnactor;

    class ThreadSlice
    {
    public:
        int           thread_num;
        int           init_size;
        CUTThread     thread_Id;
        util::cpu_mt::CPUBarrier* cpu_barrier;
        void*         problem;
        void*         enactor;
        ContextPtr    context;

        ThreadSlice()
        {
            cpu_barrier = NULL;
            problem     = NULL;
            enactor     = NULL;
        }

        virtual ~ThreadSlice()
        {
            cpu_barrier = NULL;
            problem     = NULL;
            enactor     = NULL;
        }
    };

    template <typename VertexId, typename SizeT, typename Value>
    __global__ void Expand_Incoming1 (
        const SizeT            num_elements,
        //const SizeT            num_vertex_associates,
        //const SizeT            num_value__associates,
        const SizeT            incoming_offset,
        const VertexId*  const keys_in,
              VertexId*        keys_out,
              VertexId**       vertex_associate_in,
              VertexId**       vertex_associate_org,
              Value**          value__associate_in,
              Value**          value__associate_org)
    {   
        SizeT x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        if (x>=num_elements) return;
        SizeT x2=incoming_offset+x;
        VertexId key=keys_in[x2];
        VertexId t=vertex_associate_in[0][x2];

        if (atomicCAS(vertex_associate_org[0]+key, -1, t)== -1) 
        {
            vertex_associate_org[1][key]=vertex_associate_in[1][x2];
        } else {
           if (atomicMin(vertex_associate_org[0]+key, t)<t)
           {   
               keys_out[x]=key;
               return;
           }
        }   
        keys_out[x]=key;
        atomicAdd(value__associate_org[1]+key,value__associate_in[1][x2]);
    }   

    template <typename VertexId, typename SizeT, typename Value>
    __global__ void Expand_Incoming2 (
        const SizeT            num_elements,
        //const SizeT            num_vertex_associates,
        //const SizeT            num_value__associates,
        const SizeT            incoming_offset,
        const VertexId*  const keys_in,
              //VertexId*        keys_out,
              VertexId**       vertex_associate_in,
              VertexId**       vertex_associate_org,
              Value**          value__associate_in,
              Value**          value__associate_org)
    {   
        SizeT x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        if (x>=num_elements) return;
        SizeT x2=incoming_offset+x;
        VertexId key=keys_in[x2];
        vertex_associate_org[0][key] = vertex_associate_in[0][x2];
        vertex_associate_org[1][key] = vertex_associate_in[1][x2];
        value__associate_org[0][key] = value__associate_in[0][x2];
    }   

    template <typename VertexId, typename SizeT, typename Value>
    __global__ void Expand_Incoming3 (
        const SizeT            num_elements,
        //const SizeT            num_vertex_associates,
        //const SizeT            num_value__associates,
        const SizeT            incoming_offset,
        const VertexId*  const keys_in,
              //VertexId*        keys_out,
              //VertexId**       vertex_associate_in,
              //VertexId**       vertex_associate_org,
              Value**          value__associate_in,
              Value**          value__associate_org)
    {   
        SizeT x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        if (x>=num_elements) return;
        SizeT x2=incoming_offset+x;
        VertexId key=keys_in[x2];
        value__associate_org[0][key] = value__associate_in[0][x2];
    }   

    template <
        bool     INSTRUMENT,
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename BCProblem>
    static CUT_THREADPROC BCThread(
        void * thread_data_)
    {
        typedef typename BCProblem::SizeT      SizeT;
        typedef typename BCProblem::VertexId   VertexId;
        typedef typename BCProblem::Value      Value;
        typedef typename BCProblem::DataSlice  DataSlice;
        typedef typename BCProblem::GraphSlice GraphSlice;
        typedef BCEnactor<BCProblem, INSTRUMENT> BcEnactor;
        //typedef BFSFunctor<VertexId, SizeT, VertexId, BFSProblem> BfsFunctor;
        typedef ForwardFunctor<
            VertexId,
            SizeT,
            Value,
            BCProblem> ForwardFunctor;

        typedef BackwardFunctor<
            VertexId,
            SizeT,
            Value,
            BCProblem> BackwardFunctor;

        typedef BackwardFunctor2<
            VertexId,
            SizeT,
            Value,
            BCProblem> BackwardFunctor2;

        ThreadSlice  *thread_data         = (ThreadSlice*) thread_data_;
        BCProblem    *problem             = (BCProblem*) thread_data->problem;
        BcEnactor    *enactor             = (BcEnactor*) thread_data->enactor;
        int          thread_num           =   thread_data -> thread_num;
        int          gpu                  =   problem     -> gpu_idx            [thread_num] ;
        util::Array1D<SizeT, DataSlice>
                     *data_slice          = &(problem     -> data_slices        [thread_num]);
        util::Array1D<SizeT, DataSlice>
                     *s_data_slice        =   problem     -> data_slices;
        GraphSlice   *graph_slice         =   problem     -> graph_slices       [thread_num] ;
        GraphSlice   **s_graph_slice      =   problem     -> graph_slices;
        FrontierAttribute
                     *frontier_attribute  = &(enactor     -> frontier_attribute [thread_num]);
        FrontierAttribute
                     *s_frontier_attribute=   enactor     -> frontier_attribute;
        EnactorStats *enactor_stats       = &(enactor     -> enactor_stats      [thread_num]);
        EnactorStats *s_enactor_stats     =   enactor     -> enactor_stats;
        util::CtaWorkProgressLifetime
                     *work_progress       = &(enactor     -> work_progress      [thread_num]);
        int          num_gpus             =   problem     -> num_gpus;
        ContextPtr   context              =   thread_data -> context;
        bool         DEBUG                =   enactor     -> DEBUG;
        util::cpu_mt::CPUBarrier
                     *cpu_barrier         =   thread_data -> cpu_barrier;
        util::scan::MultiScan<VertexId,SizeT,true,256,8,Value>*
                     Scaner               = NULL;
        bool         break_clean          = true;
        //SizeT*       out_offset           = NULL;
        char*        message              = new char [1024];
        util::Array1D<SizeT, unsigned int>  scanned_edges;
        util::Array1D<SizeT, VertexId    >  temp_preds;
        frontier_attribute-> queue_index  = 0;        // Work queue index
        frontier_attribute-> selector     = 0;
        frontier_attribute-> queue_length = thread_data -> init_size; //? 
        frontier_attribute-> queue_reset  = true;
        enactor_stats     -> done[0]      = -1; 
        enactor_stats     -> retval       = cudaSuccess;

        do {
            if (enactor_stats->retval = util::SetDevice(gpu)) break;
            if (num_gpus > 1)
            {
                Scaner = new util::scan::MultiScan<VertexId, SizeT, true, 256, 8,Value>;
                //out_offset = new SizeT[num_gpus +1];
            }
            scanned_edges.SetName("scanned_edges");
            if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                if (enactor_stats->retval = scanned_edges.Allocate(graph_slice->edges, util::DEVICE)) break;
                util::MemsetAddKernel<<<128, 128>>>(scanned_edges.GetPointer(util::DEVICE), (unsigned int)0, graph_slice->edges);
            }
            temp_preds.SetName("temp_preds");
            if (num_gpus>1)
                if (enactor_stats->retval = temp_preds.Allocate(graph_slice->nodes, util::DEVICE)) break;

            // Forward BC iteration
            while (!All_Done(s_enactor_stats, num_gpus)) {
                if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index+1, 0)) break;
                if (DEBUG) {
                    printf("%d\t %d\t Advance begin.\t Queue_Length = %d\n", thread_num, enactor_stats->iteration, frontier_attribute->queue_length);fflush(stdout);
                    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("Keys0", graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration);
                }
                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BCProblem, ForwardFunctor>(
                    enactor_stats->d_done,
                    enactor_stats[0],
                    frontier_attribute[0],
                    data_slice[0].GetPointer(util::DEVICE),
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    scanned_edges.GetPointer(util::DEVICE),
                    graph_slice->frontier_queues.keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                    graph_slice->frontier_queues.keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),            // d_out_queue
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    graph_slice->row_offsets   .GetPointer(util::DEVICE),
                    graph_slice->column_indices.GetPointer(util::DEVICE),
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    graph_slice->frontier_elements[frontier_attribute->selector],                   // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute->selector^1],                 // max_out_queue
                    work_progress[0],
                    context[0],
                    gunrock::oprtr::advance::V2V);

                // Only need to reset queue for once
                if (frontier_attribute->queue_reset)
                    frontier_attribute->queue_reset = false;

                if (DEBUG && (enactor_stats->retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(enactor_stats->throttle_event);                                 // give host memory mapped visibility to GPU updates

                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;
                if (enactor_stats->retval = work_progress -> GetQueueLength(frontier_attribute->queue_index  , frontier_attribute->queue_length)) break;
                if (enactor_stats->retval = work_progress -> SetQueueLength(frontier_attribute->queue_index+1, 0)) break;
                if (DEBUG) {
                    printf("%d\t %d\t Advance end.\t Queue_Length = %d\n", thread_num, enactor_stats->iteration, frontier_attribute->queue_length);fflush(stdout);                   
                    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("Keys1", graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration);
                    //if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    //printf(", %lld", (long long) frontier_attribute.queue_length);
                }
                if (INSTRUMENT) {
                    if (enactor_stats->retval = enactor_stats->advance_kernel_stats.Accumulate(
                        enactor_stats->advance_grid_size,
                        enactor_stats->total_runtimes,
                        enactor_stats->total_lifetimes)) break;
                }

                // Throttle
                if (enactor_stats->iteration & 1) {
                    if (enactor_stats->retval = util::GRError(cudaEventRecord(enactor_stats->throttle_event),
                        "BCEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (enactor_stats->retval = util::GRError(cudaEventSynchronize(enactor_stats->throttle_event),
                        "BCEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (All_Done(s_enactor_stats,num_gpus)) break;

                if (DEBUG)
                {
                    printf("%d\t %d\t Filter begin.\t Queue_Length = %d\n", thread_num, enactor_stats->iteration, frontier_attribute->queue_length);
                    fflush(stdout); 
                }
                // Filter
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, BCProblem, ForwardFunctor>
                <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS>>>(
                    enactor_stats->iteration+1,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    num_gpus,
                    frontier_attribute->queue_length,
                    enactor_stats->d_done,
                    graph_slice->frontier_queues.keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                    NULL,
                    graph_slice->frontier_queues.keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
                    data_slice[0].GetPointer(util::DEVICE),
                    NULL,
                    work_progress[0],
                    graph_slice->frontier_elements[frontier_attribute->selector  ],           // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute->selector^1],         // max_out_queue
                    enactor_stats->filter_kernel_stats);

                if (DEBUG && (enactor_stats->retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(enactor_stats->throttle_event); // give host memory mapped visibility to GPU updates

                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;
                if (enactor_stats->retval = work_progress -> GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                if (enactor_stats->retval = work_progress -> SetQueueLength(frontier_attribute->queue_index+1, 0)) break;

                if (INSTRUMENT || DEBUG) {
                    //if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    if (DEBUG) 
                    {
                        printf("%d\t %d\t Filter end.\t Queue_Length = %d\n", thread_num, enactor_stats->iteration, frontier_attribute->queue_length);
                        fflush(stdout);
                    }
                    if (INSTRUMENT) {
                        if (enactor_stats->retval = enactor_stats->filter_kernel_stats.Accumulate(
                            enactor_stats->filter_grid_size,
                            enactor_stats->total_runtimes,
                            enactor_stats->total_lifetimes)) break;
                    }
                }
                // Check if done
                if (All_Done(s_enactor_stats,num_gpus)) break;

                //Use multi_scan to splict the workload into multi_gpus
                if (num_gpus >1) 
                {   
                    // Split the frontier into multiple frontiers for multiple GPUs, local remains infront
                    if (frontier_attribute->queue_length >0) 
                    {   
                        enactor_stats->done[0]=-1;
                        //printf("%d\t%d\tScan map begin.\n",thread_num,iteration[0]);fflush(stdout);
                        //int* _partition_table = graph_slice->partition_table.GetPointer(util::DEVICE);
                        //SizeT* _convertion_table = graph_slice->convertion_table.GetPointer(util::DEVICE);
                        //printf("%d\t %d\t Update begin, n=%d\n",thread_num,enactor_stats->iteration,n);
                        //if (BCProblem::MARK_PATHS)
                            //util::cpu_mt::PrintGPUArray<SizeT,SizeT>("orgi",graph_slice->original_vertex.GetPointer(util::DEVICE),graph_slice->nodes,thread_num,iteration[0]);
                        int grid_size = frontier_attribute->queue_length%256 == 0? 
                                        frontier_attribute->queue_length/256 : 
                                        frontier_attribute->queue_length/256+1;
                        Copy_Preds<VertexId,SizeT> <<<grid_size,256>>>(
                            frontier_attribute->queue_length,  
                            graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                            data_slice[0]->preds.GetPointer(util::DEVICE),
                            temp_preds.GetPointer(util::DEVICE));
                        Update_Preds<VertexId,SizeT> <<<grid_size,256>>>(
                            frontier_attribute->queue_length,  
                            graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                            graph_slice->original_vertex.GetPointer(util::DEVICE),
                            temp_preds.GetPointer(util::DEVICE),
                            data_slice[0]->preds.GetPointer(util::DEVICE));
                    }

                    data_slice[0]->vertex_associate_orgs[0] = data_slice[0] -> labels.GetPointer(util::DEVICE);
                    data_slice[0]->vertex_associate_orgs[1] = data_slice[0] -> preds .GetPointer(util::DEVICE);
                    data_slice[0]->value__associate_orgs[0] = data_slice[0] -> sigmas.GetPointer(util::DEVICE);
                    data_slice[0]->vertex_associate_orgs.Move(util::HOST, util::DEVICE);
                    data_slice[0]->value__associate_orgs.Move(util::HOST, util::DEVICE);

                    UpdateNeiborForward <SizeT, VertexId, Value, GraphSlice, DataSlice, 2, 1> (
                        (SizeT)frontier_attribute->queue_length,
                        num_gpus,
                        thread_num,
                        Scaner,
                        s_graph_slice,
                        s_data_slice,
                        s_enactor_stats,
                        s_frontier_attribute);

                        /*Scaner->Scan_with_dKeys(n,
                            num_gpus,
                            data_slice[0]->num_vertex_associate,
                            data_slice[0]->num_value__associate,
                            graph_slice  ->frontier_queues  .keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                            graph_slice  ->frontier_queues  .keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                            graph_slice  ->partition_table  .GetPointer(util::DEVICE),
                            graph_slice  ->convertion_table .GetPointer(util::DEVICE),
                            data_slice[0]->out_length       .GetPointer(util::DEVICE),
                            data_slice[0]->vertex_associate_orgs.GetPointer(util::DEVICE),
                            data_slice[0]->vertex_associate_outs.GetPointer(util::DEVICE),
                            data_slice[0]->value__associate_orgs.GetPointer(util::DEVICE),
                            data_slice[0]->value__associate_outs.GetPointer(util::DEVICE));
                        if (enactor_stats->retval = data_slice[0]->out_length.Move(util::DEVICE,util::HOST)) break;
                        out_offset[0]=0;
                        util::cpu_mt::PrintCPUArray<SizeT, SizeT>("out_length",data_slice[0]->out_length.GetPointer(util::HOST),num_gpus,thread_num,enactor_stats->iteration);
                        for (int i=0;i<num_gpus;i++) out_offset[i+1]=out_offset[i]+data_slice[0]->out_length[i];

                        frontier_attribute->queue_index++;
                        frontier_attribute->selector ^= 1;

                        if (enactor_stats->iteration!=0)
                        {  //CPU global barrier
                            //util::cpu_mt::IncrementnWaitBarrier(&(cpu_barrier[0]),thread_num);
                            //if (All_Done(dones,retvals,num_gpus)) {break;}
                        }
                        for (int peer=0;peer<num_gpus;peer++)
                        {
                            if (peer == thread_num) continue;
                            int peer_ = peer<thread_num? peer+1     : peer;
                            int gpu_  = peer<thread_num? thread_num : thread_num+1;
                            s_data_slice[peer]->in_length[enactor_stats->iteration%2][gpu_]
                                      = data_slice[0]->out_length[peer_];
                            if (data_slice[0]->out_length[peer_] == 0) continue;
                            s_enactor_stats[peer].done[0]=-1;
                            //printf("%d\t %d\t %p+%d ==> %p+%d @ %d,%d\n", thread_num, enactor_stats->iteration, s_data_slice[peer]->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE), s_graph_slice[peer]->in_offset[gpu_], graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE), out_offset[peer_], peer, data_slice[0]->out_length[peer_]);
                            if (enactor_stats->retval = util::GRError(cudaMemcpy(
                                  s_data_slice[peer] -> keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE)
                                      + s_graph_slice[peer] -> in_offset[gpu_],
                                  graph_slice -> frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE)
                                      + out_offset[peer_],
                                  sizeof(VertexId) * data_slice[0]->out_length[peer_], cudaMemcpyDefault),
                                  "cudaMemcpyPeer d_keys failed", __FILE__, __LINE__)) break;

                            for (int i=0;i<data_slice[0]->num_vertex_associate;i++)
                            {
                                if (enactor_stats->retval = util::GRError(cudaMemcpy(
                                    s_data_slice[peer]->vertex_associate_ins[enactor_stats->iteration%2][i]
                                        + s_graph_slice[peer]->in_offset[gpu_],
                                    data_slice[0]->vertex_associate_outs[i]
                                        + (out_offset[peer_] - out_offset[1]),
                                    sizeof(VertexId) * data_slice[0]->out_length[peer_], cudaMemcpyDefault),
                                    "cudaMemcpyPeer vertex_associate_out failed", __FILE__, __LINE__)) break;
                            }

                            for (int i=0;i<data_slice[0]->num_value__associate;i++)
                            {
                                if (enactor_stats->retval = util::GRError(cudaMemcpy(
                                    s_data_slice[peer]->value__associate_ins[enactor_stats->iteration%2][i]
                                        + s_graph_slice[peer]->in_offset[gpu_],
                                    data_slice[0]->value__associate_outs[i]
                                        + (out_offset[peer_] - out_offset[1]),
                                    sizeof(Value) * data_slice[0]->out_length[peer_], cudaMemcpyDefault),
                                    "cudaMemcpyPeer value__associate_out failed", __FILE__, __LINE__)) break;
                            }
                            if (enactor_stats->retval) break;
                        }
                        if (enactor_stats->retval) break;
                    }  else {
                        if (enactor_stats->iteration!=0)
                        {  //CPU global barrier
                            //util::cpu_mt::IncrementnWaitBarrier(&(cpu_barrier[0]),thread_num);
                            //if (All_Done(dones,retvals,num_gpus)) {break;}
                        }

                        for (int peer=0;peer<num_gpus;peer++)
                        {
                            int gpu_ = peer<thread_num? thread_num: thread_num+1;
                            if (peer == thread_num) continue;
                            s_data_slice[peer]->in_length[enactor_stats->iteration%2][gpu_]=0;
                        }
                        data_slice[0]->out_length[0]=0;
                    }*/

                    //CPU global barrier
                    //if (!All_Done(s_enactor_stats,num_gpus))
                    {
                        //util::cpu_mt::PrintMessage("WaitingBarrier",thread_num,enactor_stats->iteration);
                        util::cpu_mt::IncrementnWaitBarrier(cpu_barrier,thread_num);
                        //util::cpu_mt::PrintMessage("PastBarrier",thread_num,enactor_stats->iteration);
                    }
                    //if (All_Done(s_enactor_stats,num_gpus))
                    //{
                    //    break_clean=false;
                    //    //printf("%d exit0\n",thread_num);fflush(stdout);
                    //    break;
                    //}
                    SizeT total_length=data_slice[0]->out_length[0];
                    //printf("%d\t%d\tExpand begin.\n",thread_num,iteration[0]);fflush(stdout);
                    for (int peer=0;peer<num_gpus;peer++)
                    {
                        if (peer==thread_num) continue;
                        int peer_ = peer<thread_num ? peer+1: peer ;
                        SizeT m     = data_slice[0]->in_length[enactor_stats->iteration%2][peer_];
                        if (m ==0) continue;
                        int grid_size = (m%256) == 0? m/ 256 : m/256+1;
                        //printf("%d\t %d\tin_length = %d\tnum_associate = %d\tin_offset = %d @ %d\n", thread_num, enactor_stats->iteration, m, data_slice[0]->num_associate, graph_slice->in_offset[peer_],peer);
                        //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("keys_in",data_slice[0]->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE) + graph_slice->in_offset[peer_],m,thread_num,enactor_stats->iteration);
                        //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_i0",data_slice[0]->associate_ins[enactor_stats->iteration%2][0]+graph_slice->in_offset[peer_],m,thread_num,enactor_stats->iteration);
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_i1",data_slice[0]->associate_ins[iteration[0]%2][1]+graph_slice->in_offset[peer_],m,thread_num,iteration[0]);
                        Expand_Incoming1 <VertexId, SizeT, Value>
                            <<<grid_size,256>>> (
                            m,
                            //data_slice[0]  ->num_vertex_associate,
                            //data_slice[0]  ->num_value__associate,
                            graph_slice    ->in_offset[peer_],
                            data_slice[0]  ->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                            graph_slice    ->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE) + total_length,
                            data_slice[0]  ->vertex_associate_ins[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                            data_slice[0]  ->vertex_associate_orgs.GetPointer(util::DEVICE),
                            data_slice[0]  ->value__associate_ins[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                            data_slice[0]  ->value__associate_orgs.GetPointer(util::DEVICE));
                        if (DEBUG && (enactor_stats->retval = util::GRError("Expand_Incoming failed", __FILE__, __LINE__))) break;
                       
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_orgs",data_slice[0]->associate_orgs[0],graph_slice->nodes,thread_num,iteration[0]);
                        //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("labe4",data_slice[0]->labels.GetPointer(util::GPU),graph_slice->nodes,thread_num,enactor_stats->iteration);
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("pred4",data_slice[0]->preds.GetPointer(util::GPU),graph_slice->nodes,thread_num,iteration[0]);
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("pred5",data_slice[0]->associate_orgs[1],graph_slice->nodes,thread_num,iteration[0]);
                        total_length+=data_slice[0]->in_length[enactor_stats->iteration%2][peer_];
                    }
                    if (enactor_stats->retval) break;
                    frontier_attribute->queue_length = total_length;
                    if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index,total_length)) break;
                    //printf("%d\t%d\ttotal_length = %d\n",thread_num,iteration[0],total_length);fflush(stdout);
                    if (total_length !=0)
                    {
                        enactor_stats->done[0]=-1;
                    } else {
                        enactor_stats->done[0]=0;
                    }
                }

                if (DEBUG) util::cpu_mt::PrintMessage("Iteration finished.",thread_num,enactor_stats->iteration); 
                enactor_stats->iteration++;
            } // end while (!All_Done(...))
            //delete[] sigmas;
            //delete[] labels;
            //delete[] vids;
            if (DEBUG) util::cpu_mt::PrintMessage("Forward phase finished.", thread_num, enactor_stats->iteration);

            if (num_gpus>1 && break_clean) util::cpu_mt::ReleaseBarrier(cpu_barrier, thread_num);
            if (num_gpus>1) util::cpu_mt::IncrementnWaitBarrier(&cpu_barrier[1],thread_num);
            if (DEBUG) util::cpu_mt::PrintMessage("barrier1 past.", thread_num, enactor_stats->iteration);
            enactor_stats->done[0]                 = -1;
            if (All_Done(s_enactor_stats, num_gpus)) break;
 
            enactor_stats->iteration               = enactor_stats->iteration - 2;
            //frontier_attribute->queue_length       = graph_slice->nodes;
            frontier_attribute->queue_index        = 0;        // Work queue index
            frontier_attribute->selector           = 0;
            frontier_attribute->queue_reset        = true;
            //enactor_stats->done[0]                 = -1;

            if (num_gpus>1)
            {
                frontier_attribute->queue_length = graph_slice->cross_counter[0];
                util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE),graph_slice->cross_counter[0]);
                data_slice[0]->vertex_associate_orgs[0] = data_slice[0] -> labels.GetPointer(util::DEVICE);
                data_slice[0]->vertex_associate_orgs[1] = data_slice[0] -> preds .GetPointer(util::DEVICE);
                data_slice[0]->value__associate_orgs[0] = data_slice[0] -> sigmas.GetPointer(util::DEVICE);
                data_slice[0]->vertex_associate_orgs.Move(util::HOST, util::DEVICE);
                data_slice[0]->value__associate_orgs.Move(util::HOST, util::DEVICE);

                UpdateNeiborBackward <SizeT, VertexId, Value, GraphSlice, DataSlice, 2, 1> (
                    (SizeT)frontier_attribute->queue_length,
                    num_gpus,
                    thread_num,
                    Scaner,
                    s_graph_slice,
                    s_data_slice,
                    s_enactor_stats,
                    s_frontier_attribute);
               
                util::cpu_mt::IncrementnWaitBarrier(&cpu_barrier[1],thread_num);
 
                for (int peer=0;peer<num_gpus;peer++)
                {
                    if (peer==thread_num) continue;
                    int peer_ = peer<thread_num ? peer+1: peer ;
                    SizeT m     = data_slice[0]->in_length[enactor_stats->iteration%2][peer_];
                    if (m ==0) continue;
                    int grid_size = (m%256) == 0? m/ 256 : m/256+1;
                        //printf("%d\t %d\tin_length = %d\tnum_associate = %d\tin_offset = %d @ %d\n", thread_num, enactor_stats->iteration, m, data_slice[0]->num_associate, graph_slice->in_offset[peer_],peer);
                        //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("keys_in",data_slice[0]->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE) + graph_slice->in_offset[peer_],m,thread_num,enactor_stats->iteration);
                        //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_i0",data_slice[0]->associate_ins[enactor_stats->iteration%2][0]+graph_slice->in_offset[peer_],m,thread_num,enactor_stats->iteration);
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_i1",data_slice[0]->associate_ins[iteration[0]%2][1]+graph_slice->in_offset[peer_],m,thread_num,iteration[0]);
                    Expand_Incoming2 <VertexId, SizeT, Value>
                        <<<grid_size,256>>> (
                        m,
                        //data_slice[0]  ->num_vertex_associate,
                        //data_slice[0]  ->num_value__associate,
                        graph_slice    ->in_offset[peer_],
                        data_slice[0]  ->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                        //graph_slice    ->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE) + total_length,
                        data_slice[0]  ->vertex_associate_ins[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                        data_slice[0]  ->vertex_associate_orgs.GetPointer(util::DEVICE),
                        data_slice[0]  ->value__associate_ins[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                        data_slice[0]  ->value__associate_orgs.GetPointer(util::DEVICE));
                    if (DEBUG && (enactor_stats->retval = util::GRError("Expand_Incoming2 failed", __FILE__, __LINE__))) break;      
                    //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_orgs",data_slice[0]->associate_orgs[0],graph_slice->nodes,thread_num,iteration[0]);
                    //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("labe4",data_slice[0]->labels.GetPointer(util::GPU),graph_slice->nodes,thread_num,enactor_stats->iteration);
                    //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("pred4",data_slice[0]->preds.GetPointer(util::GPU),graph_slice->nodes,thread_num,iteration[0]);
                    //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("pred5",data_slice[0]->associate_orgs[1],graph_slice->nodes,thread_num,iteration[0]);
                }
           }

            enactor_stats->done[0]=-1;
            // Prepare the label array
            VertexId            label_adjust       = -enactor_stats->iteration;
            util::MemsetAddKernel<<<128, 128>>>(
                data_slice[0]->labels.GetPointer(util::DEVICE), label_adjust, graph_slice->nodes);

            if (DEBUG) util::cpu_mt::PrintMessage("Start backward phase.", thread_num, enactor_stats->iteration);
            // Backward BC iteration
            for (;enactor_stats->iteration >= 0; --enactor_stats->iteration) {
                frontier_attribute->queue_length        = graph_slice->nodes;
                // Fill in the frontier_queues
                util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE), graph_slice->nodes);
                if (enactor_stats -> retval = work_progress->SetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                if (enactor_stats -> retval = work_progress->SetQueueLength(frontier_attribute->queue_index+1, 0)) break;
                // Filter
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, BCProblem, BackwardFunctor>
                <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS>>>(
                    -1,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    num_gpus,
                    frontier_attribute->queue_length,
                    enactor_stats->d_done,
                    graph_slice->frontier_queues.keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                    NULL,
                    graph_slice->frontier_queues.keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
                    data_slice[0].GetPointer(util::DEVICE),
                    NULL,
                    work_progress[0],
                    graph_slice->nodes,           // max_in_queue
                    graph_slice->edges,         // max_out_queue
                    enactor_stats->filter_kernel_stats);

                // Only need to reset queue for once
                //if (frontier_attribute.queue_reset)
                //    frontier_attribute.queue_reset = false; 

                if (DEBUG && (enactor_stats->retval = util::GRError(cudaThreadSynchronize(), "edge_map_backward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(enactor_stats->throttle_event);                                 // give host memory mapped visibility to GPU updates

                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;
                //if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index+1, 0)) break;
                //    }

                if (DEBUG) {
                    //if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    //printf(", %lld", (long long) frontier_attribute->queue_length);
                }
                if (INSTRUMENT) {
                    if (enactor_stats->retval = enactor_stats->advance_kernel_stats.Accumulate(
                        enactor_stats->advance_grid_size,
                        enactor_stats->total_runtimes,
                        enactor_stats->total_lifetimes)) break;
                }

                // Throttle
                if (enactor_stats->iteration & 1) {
                    if (enactor_stats->retval = util::GRError(cudaEventRecord(enactor_stats->throttle_event),
                        "BCEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (enactor_stats->retval = util::GRError(cudaEventSynchronize(enactor_stats->throttle_event),
                        "BCEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (All_Done(s_enactor_stats,num_gpus)) break;
                // Edge Map
                if (enactor_stats->iteration > 0) {
                    gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BCProblem, BackwardFunctor>(
                        enactor_stats->d_done,
                        enactor_stats[0],
                        frontier_attribute[0],
                        data_slice[0].GetPointer(util::DEVICE),
                        (VertexId*)NULL,
                        (bool*)NULL,
                        (bool*)NULL,
                        scanned_edges.GetPointer(util::DEVICE),
                        graph_slice->frontier_queues.keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                        graph_slice->frontier_queues.keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),            // d_out_queue
                        (VertexId*)NULL,
                        (VertexId*)NULL,
                        graph_slice->row_offsets   .GetPointer(util::DEVICE),
                        graph_slice->column_indices.GetPointer(util::DEVICE),
                        (SizeT*)NULL,
                        (VertexId*)NULL,
                        graph_slice->nodes,                 // max_in_queue
                        graph_slice->edges,                 // max_out_queue
                        work_progress[0],
                        context[0],
                        gunrock::oprtr::advance::V2V);
                } else {
                    gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BCProblem, BackwardFunctor2>(
                        enactor_stats->d_done,
                        enactor_stats[0],
                        frontier_attribute[0],
                        data_slice[0].GetPointer(util::DEVICE),
                        (VertexId*)NULL,
                        (bool*)NULL,
                        (bool*)NULL,
                        scanned_edges.GetPointer(util::DEVICE),
                        graph_slice->frontier_queues.keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                        graph_slice->frontier_queues.keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),            // d_out_queue
                        (VertexId*)NULL,
                        (VertexId*)NULL,
                        graph_slice->row_offsets   .GetPointer(util::DEVICE),
                        graph_slice->column_indices.GetPointer(util::DEVICE),
                        (SizeT*)NULL,
                        (VertexId*)NULL,
                        graph_slice->nodes,                 // max_in_queue
                        graph_slice->edges,                 // max_out_queue
                        work_progress[0],
                        context[0],
                        gunrock::oprtr::advance::V2V);
                }

                if (DEBUG && (enactor_stats->retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(enactor_stats->throttle_event); // give host memory mapped visibility to GPU updates

                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;

                util::MemsetAddKernel<<<128, 128>>>(data_slice[0]->labels.GetPointer(util::DEVICE), 1, graph_slice->nodes);

                if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                if (INSTRUMENT || DEBUG) {
                    if (DEBUG) printf(", %lld", (long long) frontier_attribute->queue_length);
                    if (INSTRUMENT) {
                        if (enactor_stats->retval = enactor_stats->filter_kernel_stats.Accumulate(
                            enactor_stats->filter_grid_size,
                            enactor_stats->total_runtimes,
                            enactor_stats->total_lifetimes)) break;
                    }
                }
                // Check if done
                if (All_Done(s_enactor_stats,num_gpus)) break;
            
                if (num_gpus >1)
                {
                    data_slice[0]->value__associate_orgs[0] = data_slice[0] -> deltas.GetPointer(util::DEVICE);
                    data_slice[0]->value__associate_orgs.Move(util::HOST, util::DEVICE);

                    UpdateNeiborBackward <SizeT, VertexId, Value, GraphSlice, DataSlice, 0, 1> (
                        (SizeT)frontier_attribute->queue_length,
                        num_gpus,
                        thread_num,
                        Scaner,
                        s_graph_slice,
                        s_data_slice,
                        s_enactor_stats,
                        s_frontier_attribute);
 
                    util::cpu_mt::IncrementnWaitBarrier(&cpu_barrier[1],thread_num);
                
                    for (int peer=0;peer<num_gpus;peer++)
                    {
                        if (peer==thread_num) continue;
                        int peer_ = peer<thread_num ? peer+1: peer ;
                        SizeT m     = data_slice[0]->in_length[enactor_stats->iteration%2][peer_];
                        if (m ==0) continue;
                        int grid_size = (m%256) == 0? m/ 256 : m/256+1;
                            //printf("%d\t %d\tin_length = %d\tnum_associate = %d\tin_offset = %d @ %d\n", thread_num, enactor_stats->iteration, m, data_slice[0]->num_associate, graph_slice->in_offset[peer_],peer);
                            //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("keys_in",data_slice[0]->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE) + graph_slice->in_offset[peer_],m,thread_num,enactor_stats->iteration);
                            //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_i0",data_slice[0]->associate_ins[enactor_stats->iteration%2][0]+graph_slice->in_offset[peer_],m,thread_num,enactor_stats->iteration);
                            //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_i1",data_slice[0]->associate_ins[iteration[0]%2][1]+graph_slice->in_offset[peer_],m,thread_num,iteration[0]);
                        Expand_Incoming3 <VertexId, SizeT, Value>
                            <<<grid_size,256>>> (
                            m,
                            //data_slice[0]  ->num_vertex_associate,
                            //data_slice[0]  ->num_value__associate,
                            graph_slice    ->in_offset[peer_],
                            data_slice[0]  ->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                            //graph_slice    ->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE) + total_length,
                            //data_slice[0]  ->vertex_associate_ins[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                            //data_slice[0]  ->vertex_associate_orgs.GetPointer(util::DEVICE),
                            data_slice[0]  ->value__associate_ins[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                            data_slice[0]  ->value__associate_orgs.GetPointer(util::DEVICE));
                        if (DEBUG && (enactor_stats->retval = util::GRError("Expand_Incoming2 failed", __FILE__, __LINE__))) break;
                    }
                }
                if (DEBUG) util::cpu_mt::PrintMessage("BIteration finished.", thread_num, enactor_stats->iteration);
                if (enactor_stats->retval) break;
            }
            if (enactor_stats->retval) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            if (enactor_stats->retval = work_progress->CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                enactor_stats->retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
            }
            scanned_edges.Release();
            temp_preds.Release();
        } while(0);

        if (num_gpus >1)
        {
            if (break_clean)
            {
                util::cpu_mt::ReleaseBarrier(&cpu_barrier[1],thread_num);
            }
            delete Scaner; Scaner=NULL;
        }
        delete[] message;message=NULL;

        CUT_THREADEND;
    }

/**
 * @brief BC problem enactor class.
 *
 * @tparam INSTRUMENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<typename BCProblem, bool INSTRUMENT>
class BCEnactor : public EnactorBase
{
    typedef typename BCProblem::SizeT    SizeT   ;
    typedef typename BCProblem::VertexId VertexId;
    typedef typename BCProblem::Value    Value   ;

    // Members
    protected:

    //unsigned long long total_runtimes;              // Total working time by each CTA
    //unsigned long long total_lifetimes;             // Total life time of each CTA

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    //volatile int        *done;
    //int                 *d_done;
    //cudaEvent_t         throttle_event;

    /**
     * Current iteration, also used to get the final search depth of the BC search
     */
    //long long                           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for BC kernel call. Must be called prior to each BC search.
     *
     * @param[in] problem BC Problem object which holds the graph data and BC problem data to compute.
     * @param[in] edge_map_grid_size CTA occupancy for edge mapping kernel call.
     * @param[in] filter_grid_size CTA occupancy for filter kernel call.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    //template <typename ProblemData>
    cudaError_t Setup(
        BCProblem *problem)
    {
        //typedef typename ProblemData::SizeT         SizeT;
        //typedef typename ProblemData::VertexId      VertexId;
        
        cudaError_t retval = cudaSuccess;

        do {
            for (int gpu=0; gpu<num_gpus; gpu++)
            {
                //if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                /*//initialize the host-mapped "done"
                if (!done) {
                    int flags = cudaHostAllocMapped;

                    // Allocate pinned memory for done
                    if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                        "BCEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                    // Map done into GPU space
                    if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                        "BCEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

                    // Create throttle event
                    if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                        "BCEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
                }

                done[0]             = -1;

                //graph slice
                typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];
                */

                // Bind row-offsets and column_indices texture
                /*cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
                gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref.channelDesc = row_offsets_desc;
                if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    problem->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE),
                    (problem->graph_slices[gpu]->nodes + 1) * sizeof(SizeT)),
                        "BCEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;
                */
                /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
                gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref.channelDesc = column_indices_desc;
                if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            graph_slice->edges * sizeof(VertexId)),
                        "BCEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
            } // end for(gpu)
        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief BCEnactor constructor
     */
    BCEnactor(bool DEBUG = false, int num_gpus = 1, int* gpu_idx = NULL) :
        EnactorBase(EDGE_FRONTIERS, DEBUG, num_gpus, gpu_idx)//,
        //iteration(0),
        //done(NULL),
        //d_done(NULL)
    {}

    /**
     * @brief BCEnactor destructor
     */
    virtual ~BCEnactor()
    {
        /*for (int gpu=0; gpu<num_gpus; gpu++)
        {
            util::SetDevice(gpu_idx[gpu]);
            cudaUnbindTexture(gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref);
        }*/
        /*if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "BCEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "BCEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }*/
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BC search enacted.
     *
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    void GetStatistics(
        double &avg_duty)
    {
        unsigned long long total_lifetimes=0;
        unsigned long long total_runtimes =0; 
        //total_queued = 0;
        //search_depth = 0;
        for (int gpu=0;gpu<num_gpus;gpu++)
        {   
            if (num_gpus!=1)
                if (util::SetDevice(gpu_idx[gpu])) return;
            cudaThreadSynchronize();

            //total_queued += this->enactor_stats[gpu].total_queued;
            //if (this->enactor_stats[gpu].iteration > search_depth) 
            //    search_depth = this->enactor_stats[gpu].iteration;
            total_lifetimes += this->enactor_stats[gpu].total_lifetimes;
            total_runtimes  += this->enactor_stats[gpu].total_runtimes;
        }   
        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * @brief Enacts a brandes betweenness centrality computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam FilterPolicy Kernel policy for filter operator.
     * @tparam BCProblem BC Problem type.
     * @tparam ForwardFunctor Forward Functor type used in the forward sigma computing pass.
     * @tparam BackwardFunctor Backward Functor type used in the backward bc value accumulation pass.
     * @param[in] problem BCProblem object.
     * @param[in] src Source node for BC. -1 to compute BC value for each node.
     * @param[in] max_grid_size Max grid size for BC kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
        //typename BCProblem>
    cudaError_t EnactBC(
    ContextPtr  *context,
    BCProblem   *problem,
    VertexId    src,
    int         max_grid_size = 0)
    {
        //typedef typename BCProblem::SizeT       SizeT;
        //typedef typename BCProblem::VertexId    VertexId;
        //typedef typename BCProblem::Value       Value;
        cudaError_t              retval         = cudaSuccess;
        util::cpu_mt::CPUBarrier cpu_barrier[2];
        cpu_barrier[0] = util::cpu_mt::CreateBarrier(num_gpus);
        cpu_barrier[1] = util::cpu_mt::CreateBarrier(num_gpus);
        ThreadSlice              *thread_slices = new ThreadSlice [num_gpus];
        CUTThread                *thread_Ids    = new CUTThread   [num_gpus];

        do {
            // Lazy initialization
            if (retval = EnactorBase::Setup(problem,
                                            max_grid_size,
                                            AdvanceKernelPolicy::CTA_OCCUPANCY,
                                            FilterKernelPolicy::CTA_OCCUPANCY))
                                            break;

            if (retval = Setup(problem)) break;
            
            if (DEBUG) {
                printf("Iteration, Edge map queue, Filter queue\n");
                printf("0");
            }

            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                thread_slices[gpu].thread_num    = gpu;
                thread_slices[gpu].problem       = (void*)problem;
                thread_slices[gpu].enactor       = (void*)this;
                thread_slices[gpu].cpu_barrier   = cpu_barrier;
                thread_slices[gpu].context       = context[gpu];
                if ((num_gpus ==1) || (gpu==problem->partition_tables[0][src]))
                     thread_slices[gpu].init_size=1;
                else thread_slices[gpu].init_size=0;
                thread_slices[gpu].thread_Id = cutStartThread(
                    (CUT_THREADROUTINE)&(BCThread<INSTRUMENT,AdvanceKernelPolicy,FilterKernelPolicy,BCProblem>),
                    (void*)&(thread_slices[gpu]));
                thread_Ids[gpu] = thread_slices[gpu].thread_Id;
            }

            cutWaitForThreads(thread_Ids, num_gpus);
            util::cpu_mt::DestoryBarrier(cpu_barrier);
            util::cpu_mt::DestoryBarrier(&cpu_barrier[1]);

            for (int gpu=0;gpu<num_gpus;gpu++)
            if (this->enactor_stats[gpu].retval!=cudaSuccess) {retval=this->enactor_stats[gpu].retval;break;}
        } while (0);
        delete[] thread_Ids   ; thread_Ids    = NULL;
        delete[] thread_slices; thread_slices = NULL;
        if (DEBUG) printf("\nGPU BC Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BC Enact kernel entry.
     *
     * @tparam BCProblem BC Problem type. @see BCProblem
     *
     * @param[in] problem Pointer to BCProblem object.
     * @param[in] src Source node for BC. -1 indicates computing BC value for all nodes.
     * @param[in] max_grid_size Max grid size for BC kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    //template <typename BCProblem>
    cudaError_t Enact(
        ContextPtr  *context,
        BCProblem   *problem,
        VertexId    src,
        int         max_grid_size = 0)
    {
        int min_sm_version = -1; 
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                BCProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                7,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END BIT_MASK (no bitmask cull in BC)
                8>                                  // LOG_SCHEDULE_GRANULARITY
                FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                BCProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                10,                                  // LOG_THREADS
                8,                                  // LOG_BLOCKS
                32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::LB>
                AdvanceKernelPolicy;

                return EnactBC<AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, src, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace bc
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
