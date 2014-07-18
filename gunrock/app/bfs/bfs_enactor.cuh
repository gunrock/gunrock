// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_enactor.cuh
 *
 * @brief BFS Problem Enactor
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
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>


namespace gunrock {
namespace app {
namespace bfs {

    template <typename BFSProblem, bool INSTRUMENT> class BFSEnactor;
        
    class ThreadSlice
    {
    public:
        int           thread_num;
        int           init_size;
        CUTThread     thread_Id;
        util::cpu_mt::CPUBarrier* cpu_barrier;
        void*         problem;
        void*         enactor;
        ContextPtr*   context;

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

    template <typename VertexId, typename SizeT, SizeT num_associates>
    __global__ void Expand_Incoming (
        const SizeT            num_elements,
        const SizeT            incoming_offset,
        const VertexId*  const keys_in,
              VertexId*        keys_out,
              unsigned char*   marker,
              VertexId**       associate_in,
              VertexId**       associate_org)
    {
        //const SizeT STRIDE = gridDim.x * blockDim.x;
        //__shared__ VertexId* s_associate_in[2];
        //__shared__ VertexId* s_associate_org[2];
        SizeT x2;
        VertexId key,t;

        /*if (threadIdx.x <num_associates)
            s_associate_in[threadIdx.x]=associate_in[threadIdx.x];
        else if (threadIdx.x<num_associates*2)
            s_associate_org[threadIdx.x - num_associates] = associate_org[threadIdx.x - num_associates];
        __syncthreads();*/

        //SizeT x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        SizeT x= blockIdx.x * blockDim.x + threadIdx.x;
        if (x>=num_elements) return;
        //for (SizeT x = blockIdx.x * blockDim.x + threadIdx.x; x<num_elements; x+=STRIDE)
        {
            x2  = incoming_offset+x;
            key = keys_in[x2];
            t   = associate_in[0][x2];

            if (atomicCAS(associate_org[0]+key, -1, t)== -1)
            {
            } else {
               if (atomicMin(associate_org[0]+key, t)<t)
               {
                   keys_out[x]=-1;
                   return;//continue;
               }
            }
            if (marker[key]==0) 
            {
                marker[key]=1;keys_out[x]=key;}
            else keys_out[x]=-1;
            if (num_associates==2) 
                associate_org[1][key]=associate_in[1][x2];
            /*#pragma unroll
            for (SizeT i=1;i<num_associates;i++)
            {
                associate_org[i][key]=associate_in[i][x2];
            }*/
        }
    }

    template<
        bool     INSTRUMENT,
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename BFSProblem>
    static CUT_THREADPROC BFSThread(
        void * thread_data_)
    {
        typedef typename BFSProblem::SizeT      SizeT;
        typedef typename BFSProblem::VertexId   VertexId;
        typedef typename BFSProblem::Value      Value;
        typedef typename BFSProblem::DataSlice  DataSlice;
        typedef typename BFSProblem::GraphSlice GraphSlice;
        typedef BFSEnactor<BFSProblem, INSTRUMENT> BfsEnactor;
        typedef BFSFunctor<VertexId, SizeT, VertexId, BFSProblem> BfsFunctor;

        ThreadSlice  *thread_data          = (ThreadSlice *) thread_data_;
        BFSProblem   *problem              = (BFSProblem*) thread_data->problem;
        BFSEnactor<BFSProblem, INSTRUMENT>
                     *enactor              = (BFSEnactor<BFSProblem, INSTRUMENT>*) thread_data->enactor;
        int          thread_num            =   thread_data -> thread_num;
        int          gpu                   =   problem     -> gpu_idx           [thread_num];
        util::Array1D<SizeT, DataSlice>
                     *data_slice           = &(problem     -> data_slices       [thread_num]);
        util::Array1D<SizeT, DataSlice>
                     *s_data_slice         =   problem     -> data_slices;
        GraphSlice   *graph_slice          =   problem     -> graph_slices      [thread_num];
        GraphSlice   **s_graph_slice       =   problem     -> graph_slices;
        FrontierAttribute
                     *frontier_attribute   = &(enactor     -> frontier_attribute[thread_num]);
        FrontierAttribute
                     *s_frontier_attribute = &(enactor     -> frontier_attribute[0         ]);
        EnactorStats *enactor_stats        = &(enactor     -> enactor_stats     [thread_num]);
        EnactorStats *s_enactor_stats      = &(enactor     -> enactor_stats     [0         ]);
        util::CtaWorkProgressLifetime
                     *work_progress        = &(enactor     -> work_progress     [thread_num]);
        int          num_gpus              =   problem     -> num_gpus;
        ContextPtr*  context               =   thread_data -> context;
        bool         DEBUG                 =   enactor     -> DEBUG;
        //texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> 
        //             *t_bitmask            = &(ts_bitmask       [thread_num]);
        util::cpu_mt::CPUBarrier
                     *cpu_barrier          =   thread_data -> cpu_barrier;
        //util::scan::MultiScan<VertexId,SizeT,true,256,8>*
        //             Scaner                = NULL;
        bool         break_clean           = true;
        frontier_attribute->queue_index    = 0;        // Work queue index
        frontier_attribute->selector       = 0;
        frontier_attribute->queue_length   = thread_data -> init_size; //? 
        frontier_attribute->queue_reset    = true;
        
        do {
            util::cpu_mt::PrintMessage("BFS Thread begin.",thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);
            if (enactor_stats->retval = util::SetDevice(gpu)) break;

            // Step through BFS iterations
            //while (done[0] < 0) {
            while (!All_Done(s_enactor_stats, s_frontier_attribute, num_gpus)) {
                if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index+1,0,false,data_slice[0]->streams[0])) break;
                if (DEBUG)
                {
                    //SizeT _queue_length;
                    //if (frontier_attribute->queue_reset) _queue_length = frontier_attribute->queue_length;
                    //else if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, _queue_length)) break;
                    util::cpu_mt::PrintCPUArray<SizeT, unsigned int>("Queue_Length", &(frontier_attribute->queue_length), 1, thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);
                    util::cpu_mt::PrintGPUArray<SizeT, SizeT   >("offs0", graph_slice->row_offsets.GetPointer(util::DEVICE),graph_slice->nodes,thread_num, enactor_stats->iteration);
                    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys0", graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration);
                    //if (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
                    //    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu0", graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE), _queue_length, thread_num, enactor_stats->iteration);
                    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("labe0", data_slice[0]->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
                    //if (BFSProblem::MARK_PREDECESSORS)
                    //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pred0", data_slice[0]->preds.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
                    //if (BFSProblem::ENABLE_IDEMPOTENCE)
                    //    util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask0", data_slice[0]->visited_mask.GetPointer(util::DEVICE), (graph_slice->nodes+7)/8, thread_num, enactor_stats->iteration);
                   util::cpu_mt::PrintMessage("Advance begin",thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);
                }
                // Edge Map

                gunrock::oprtr::advance::ComputeOutputLength <AdvanceKernelPolicy, BFSProblem, BfsFunctor>(
                    frontier_attribute,
                    graph_slice->row_offsets   .GetPointer(util::DEVICE),
                    graph_slice->column_indices.GetPointer(util::DEVICE),
                    graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                    data_slice[0]->scanned_edges.GetPointer(util::DEVICE),
                    graph_slice->frontier_elements[frontier_attribute->selector  ],
                    graph_slice->frontier_elements[frontier_attribute->selector^1],
                    context[0][0],
                    data_slice[0]->streams[0],
                    gunrock::oprtr::advance::V2V);

                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BFSProblem, BfsFunctor>(
                    //enactor_stats->d_done,
                    enactor_stats[0],
                    frontier_attribute[0],
                    data_slice[0].GetPointer(util::DEVICE),
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    data_slice[0]->scanned_edges.GetPointer(util::DEVICE),
                    graph_slice->frontier_queues.keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                    graph_slice->frontier_queues.keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),            // d_out_queue
                    (VertexId*)NULL,          // d_pred_in_queue
                    graph_slice->frontier_queues.values[frontier_attribute->selector^1].GetPointer(util::DEVICE),          // d_pred_out_queue
                    graph_slice->row_offsets   .GetPointer(util::DEVICE),
                    graph_slice->column_indices.GetPointer(util::DEVICE),
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    graph_slice->frontier_elements[frontier_attribute->selector],
                    graph_slice->frontier_elements[frontier_attribute->selector^1],
                    work_progress[0],
                    context[0][0],
                    data_slice[0]->streams[0],
                    gunrock::oprtr::advance::V2V,
                    false,
                    false);

                // Only need to reset queue for once
                //if (frontier_attribute->queue_reset)
                    frontier_attribute->queue_reset = false;
                if (DEBUG && (enactor_stats->retval = util::GRError(cudaStreamSynchronize(data_slice[0]->streams[0]), "advance::Kernel failed", __FILE__, __LINE__))) break;
                if (DEBUG) util::cpu_mt::PrintMessage("Advance end", thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);

                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;

                //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length,false,data_slice[0]->streams[0])) break;
                if (DEBUG || INSTRUMENT) 
                    enactor_stats->total_queued += frontier_attribute->queue_length;
                if (DEBUG)
                {
                    //SizeT _queue_length = frontier_attribute->queue_length;
                    //if (frontier_attribute->queue_reset) _queue_length = frontier_attribute->queue_length;
                    //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, _queue_length)) break;
                    util::cpu_mt::PrintCPUArray<SizeT, unsigned int>("Queue_Length", &(frontier_attribute->queue_length), 1, thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);
                    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys1", graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration);
                    //if (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
                    //    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu1", graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE), _queue_length, thread_num, enactor_stats->iteration);
                    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("labe1", data_slice[0]->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
                    //if (BFSProblem::MARK_PREDECESSORS)
                    //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pred1", data_slice[0]->preds.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
                    //if (BFSProblem::ENABLE_IDEMPOTENCE)
                    //    util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask1", data_slice[0]->visited_mask.GetPointer(util::DEVICE), (graph_slice->nodes+7)/8, thread_num, enactor_stats->iteration);
                }

                if (INSTRUMENT) {
                    if (enactor_stats->retval = enactor_stats->advance_kernel_stats.Accumulate(
                        enactor_stats->advance_grid_size,
                        enactor_stats->total_runtimes,
                        enactor_stats->total_lifetimes,
                        false,data_slice[0]->streams[0])) break;
                }

                // Check if done
                //if (All_Done(s_enactor_stats,num_gpus)) break;
                if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index+1, 0, false, data_slice[0]->streams[0])) break;
               
                if (DEBUG) util::cpu_mt::PrintMessage("Filter begin", thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);
                // Filter
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, BFSProblem, BfsFunctor>
                <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, data_slice[0]->streams[0]>>>(
                    enactor_stats->iteration+1,
                    frontier_attribute->queue_reset,
                    frontier_attribute->queue_index,
                    num_gpus,
                    frontier_attribute->queue_length,
                    //enactor_stats->d_done,
                    graph_slice->frontier_queues.keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                    graph_slice->frontier_queues.values[frontier_attribute->selector  ].GetPointer(util::DEVICE),    // d_pred_in_queue
                    graph_slice->frontier_queues.keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
                    data_slice[0].GetPointer(util::DEVICE),
                    data_slice[0]->visited_mask.GetPointer(util::DEVICE),
                    work_progress[0],
                    graph_slice->frontier_elements[frontier_attribute->selector],           // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute->selector^1],         // max_out_queue
                    enactor_stats->filter_kernel_stats);
                    //t_bitmask);

                if (DEBUG && (enactor_stats->retval = util::GRError(cudaStreamSynchronize(data_slice[0]->streams[0]), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                //cudaEventQuery(enactor_stats->throttle_event); // give host memory mapped visibility to GPU updates
                if (DEBUG) util::cpu_mt::PrintMessage("Filter end.", thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);

                frontier_attribute->queue_index++;
                frontier_attribute->selector ^= 1;

                //if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length, false, data_slice[0]->streams[0])) break;
                //}

                if (INSTRUMENT || DEBUG) {
                    //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                    //enactor_stats->total_queued += frontier_attribute->queue_length;
                    //if (DEBUG) printf(", %lld", (long long) frontier_attribute->queue_length);
                    if (INSTRUMENT) {
                        if (enactor_stats->retval = enactor_stats->filter_kernel_stats.Accumulate(
                            enactor_stats->filter_grid_size,
                            enactor_stats->total_runtimes,
                            enactor_stats->total_lifetimes,
                            false, data_slice[0]->streams[0])) break;
                    }
                }
                if (DEBUG)
                {
                    //SizeT _queue_length = frontier_attribute->queue_length;
                    //if (frontier_attribute->queue_reset) _queue_length = frontier_attribute->queue_length;
                    //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, _queue_length)) break;
                    util::cpu_mt::PrintCPUArray<SizeT, unsigned int>("Queue_Length", &(frontier_attribute->queue_length), 1, thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);
                    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys2", graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE), frontier_attribute->queue_length, thread_num, enactor_stats->iteration);
                    //if (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
                    //    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu2", graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE), _queue_length, thread_num, enactor_stats->iteration);
                    //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("labe2", data_slice[0]->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
                    //if (BFSProblem::MARK_PREDECESSORS)
                    //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pred2", data_slice[0]->preds.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
                    //if (BFSProblem::ENABLE_IDEMPOTENCE)
                    //    util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask2", data_slice[0]->visited_mask.GetPointer(util::DEVICE), (graph_slice->nodes+7)/8, thread_num, enactor_stats->iteration);
                }

                // Check if done
                if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length, false, data_slice[0]->streams[0])) break;
                if (All_Done(s_enactor_stats,s_frontier_attribute,num_gpus)) break;

                //Use multi_scan to splict the workload into multi_gpus
                if (num_gpus >1) 
                {   
                    if (DEBUG) util::cpu_mt::PrintMessage("mgpu data exchange begin.", thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);
                    // Split the frontier into multiple frontiers for multiple GPUs, local remains infront
                    //SizeT n;
                    //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, n)) break;
                    if (frontier_attribute->queue_length >0) 
                    {   
                        //enactor_stats->done[0]=-1;
                        //printf("%d\t%d\tScan map begin.\n",thread_num,iteration[0]);fflush(stdout);
                        //int* _partition_table = graph_slice->partition_table.GetPointer(util::DEVICE);
                        //SizeT* _convertion_table = graph_slice->convertion_table.GetPointer(util::DEVICE);
                        if (BFSProblem::MARK_PREDECESSORS)
                        {   
                            //util::cpu_mt::PrintGPUArray<SizeT,SizeT>("orgi",graph_slice->original_vertex.GetPointer(util::DEVICE),graph_slice->nodes,thread_num,iteration[0]);
                            int grid_size = (frontier_attribute->queue_length%256) == 0? 
                                             frontier_attribute->queue_length/256 : 
                                             frontier_attribute->queue_length/256+1;
                            Copy_Preds<VertexId, SizeT> <<<grid_size,256,0,data_slice[0]->streams[0]>>>(
                                frontier_attribute->queue_length,
                                graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                                data_slice[0]->preds.GetPointer(util::DEVICE),
                                data_slice[0]->temp_preds.GetPointer(util::DEVICE));
                            Update_Preds<VertexId,SizeT> <<<grid_size,256,0,data_slice[0]->streams[0]>>>(
                                frontier_attribute->queue_length,
                                graph_slice->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                                graph_slice->original_vertex.GetPointer(util::DEVICE),
                                data_slice[0]->temp_preds.GetPointer(util::DEVICE),
                                data_slice[0]->preds.GetPointer(util::DEVICE));
                        }
                    }
                    
                    if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(data_slice[0]->streams[0]),"cudaDeviceSynchronize failed",__FILE__,__LINE__)) break; 
                    /*data_slice[0]->vertex_associate_orgs[0] = data_slice[0]-> labels.GetPointer(util::DEVICE);
                    if (BFSProblem::MARK_PREDESSORS)
                        data_slice[0]->vertex_associate_orgs[1] = data_slice[0]-> preds.GetPointer(util::DEVICE);
                    data_slice[0]->vertex_associate_orgs.Move(util::HOST, util::DEVICE);*/
                    
                    if (BFSProblem::MARK_PREDECESSORS)
                        UpdateNeiborForward <SizeT, VertexId, Value, GraphSlice, DataSlice, 2, 0> (
                            (SizeT)frontier_attribute->queue_length,
                            num_gpus,
                            thread_num,
                            data_slice[0]->Scaner,
                            s_graph_slice,
                            s_data_slice,
                            s_enactor_stats,
                            s_frontier_attribute,
                            data_slice[0]->temp_marker.GetPointer(util::DEVICE));
                    else UpdateNeiborForward <SizeT, VertexId, Value, GraphSlice, DataSlice, 1, 0> (
                            (SizeT)frontier_attribute->queue_length,
                            num_gpus,
                            thread_num,
                            data_slice[0]->Scaner,
                            s_graph_slice,
                            s_data_slice,
                            s_enactor_stats,
                            s_frontier_attribute,
                            data_slice[0]->temp_marker.GetPointer(util::DEVICE));

                    /*util::MemsetKernel<<<128, 128>>>(temp_preds.GetPointer(util::DEVICE),0,graph_slice->nodes);
                    if (data_slice[0]->out_length[0]>0)
                    {
                        int grid_size = (data_slice[0]->out_length[0]%256)==0 ? 
                                         data_slice[0]->out_length[0]/256 : 
                                         data_slice[0]->out_length[0]/256+1;
                        Mark_Queue<VertexId, SizeT> <<<grid_size,256>>>(
                            data_slice[0]->out_length[0],
                            graph_slice -> frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                            temp_preds.GetPointer(util::DEVICE));
                    }*/

                        /*Scaner->Scan_with_Keys(n,
                            num_gpus,
                            data_slice[0]->num_associate,
                            graph_slice  ->frontier_queues  .keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                            graph_slice  ->frontier_queues  .keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                            graph_slice  ->partition_table  .GetPointer(util::DEVICE),
                            graph_slice  ->convertion_table .GetPointer(util::DEVICE),
                            data_slice[0]->out_length       .GetPointer(util::DEVICE),
                            data_slice[0]->associate_orgs   .GetPointer(util::DEVICE),
                            data_slice[0]->associate_outs   .GetPointer(util::DEVICE));
                        if (enactor_stats->retval = data_slice[0]->out_length.Move(util::DEVICE,util::HOST)) break;
                        out_offset[0]=0;
                        for (int i=0;i<num_gpus;i++) out_offset[i+1]=out_offset[i]+data_slice[0]->out_length[i];
 
                        frontier_attribute->queue_index++;
                        frontier_attribute->selector ^= 1;
                        
                        //util::cpu_mt::PrintCPUArray<SizeT, SizeT>("out_offset",out_offset,num_gpus+1,thread_num,enactor_stats->iteration);
                        util::MemsetKernel<<<128, 128>>>(temp_preds.GetPointer(util::DEVICE),0,graph_slice->nodes);
                        if (out_offset[1]>0)
                        {
                            int grid_size = (out_offset[1]%256)==0 ? out_offset[1]/256 : out_offset[1]/256+1;
                            Mark_Queue<VertexId, SizeT> <<<grid_size,256>>>(
                                out_offset[1],
                                graph_slice -> frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                                temp_preds.GetPointer(util::DEVICE));
                        }
 
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
                            if (enactor_stats->retval = util::GRError(cudaMemcpy(
                                  s_data_slice[peer] -> keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE)
                                      + s_graph_slice[peer] -> in_offset[gpu_],
                                  graph_slice -> frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE)
                                      + out_offset[peer_],
                                  sizeof(VertexId) * data_slice[0]->out_length[peer_], cudaMemcpyDefault),
                                  "cudaMemcpyPeer d_keys failed", __FILE__, __LINE__)) break;

                            for (int i=0;i<data_slice[0]->num_associate;i++)
                            {
                                if (enactor_stats->retval = util::GRError(cudaMemcpy(
                                    s_data_slice[peer]->associate_ins[enactor_stats->iteration%2][i]
                                        + s_graph_slice[peer]->in_offset[gpu_],
                                    data_slice[0]->associate_outs[i]
                                        + (out_offset[peer_] - out_offset[1]),
                                    sizeof(VertexId) * data_slice[0]->out_length[peer_], cudaMemcpyDefault),
                                    "cudaMemcpyPeer associate_out failed", __FILE__, __LINE__)) break;
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

                        util::MemsetKernel<<<128, 128>>>(temp_preds.GetPointer(util::DEVICE),0,graph_slice->nodes);
                        for (int peer=0;peer<num_gpus;peer++)
                        {
                            int gpu_ = peer<thread_num? thread_num: thread_num+1;
                            if (peer == thread_num) continue;
                            s_data_slice[peer]->in_length[enactor_stats->iteration%2][gpu_]=0;
                        }
                        data_slice[0]->out_length[0]=0;
                    }*/

                    //CPU global barrier
                    for (int i=0;i<num_gpus;i++)
                        if (enactor_stats->retval = util::GRError(cudaStreamSynchronize(data_slice[0]->streams[i]), "cuStreamSynchronize failed.", __FILE__, __LINE__)) break; 
                    if (enactor_stats->retval) break;

                    util::cpu_mt::IncrementnWaitBarrier(cpu_barrier,thread_num);
                    if (All_Done(s_enactor_stats,s_frontier_attribute,num_gpus))
                    {
                        break_clean=false;
                        //printf("%d exit0\n",thread_num);fflush(stdout);
                        break;
                    }
                    SizeT total_length=data_slice[0]->out_length[0];
                    //printf("%d\t%d\tExpand begin.\n",thread_num,iteration[0]);fflush(stdout);
                    for (int peer=0;peer<num_gpus;peer++)
                    {
                        if (peer==thread_num) continue;
                        int   peer_   = peer<thread_num ? peer+1: peer ;
                        SizeT m       = data_slice[0]->in_length[enactor_stats->iteration%2][peer_];
                        if (m ==0) continue;
                        int grid_size = (m%256) == 0? m/ 256 : m/256+1;
                        //if (DEBUG) printf("%d\t %lld\t %.2f\t in_length = %d\t num_associate = %d\t in_offset = %d\n", thread_num, enactor_stats->iteration, (clock()-enactor_stats->start_time)*(float)1000/CLOCKS_PER_SEC, m, data_slice[0]->num_associate, graph_slice->in_offset[peer_]);
                        //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("keys_in",data_slice[0]->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE) + graph_slice->in_offset[peer_],m,thread_num,enactor_stats->iteration);
                        //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_i0",data_slice[0]->associate_ins[enactor_stats->iteration%2][0]+graph_slice->in_offset[peer_],m,thread_num,enactor_stats->iteration);
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_i1",data_slice[0]->associate_ins[iteration[0]%2][1]+graph_slice->in_offset[peer_],m,thread_num,iteration[0]);
                        if (data_slice[0]->num_vertex_associate==1) Expand_Incoming <VertexId, SizeT, 1>
                            <<<grid_size,256>>> (
                            m,
                            graph_slice    ->in_offset[peer_],
                            data_slice[0]  ->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                            graph_slice    ->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE) + total_length,
                            data_slice[0]  ->temp_marker     .GetPointer(util::DEVICE),
                            data_slice[0]  ->vertex_associate_ins[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                            data_slice[0]  ->vertex_associate_orgs.GetPointer(util::DEVICE));
                        else if (data_slice[0]->num_vertex_associate==2) Expand_Incoming <VertexId, SizeT, 2>
                            <<<grid_size,256>>> (
                            m,
                            graph_slice    ->in_offset[peer_],
                            data_slice[0]  ->keys_in[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                            graph_slice    ->frontier_queues.keys[frontier_attribute->selector].GetPointer(util::DEVICE) + total_length,
                            data_slice[0]  ->temp_marker   .GetPointer(util::DEVICE),
                            data_slice[0]  ->vertex_associate_ins[enactor_stats->iteration%2].GetPointer(util::DEVICE),
                            data_slice[0]  ->vertex_associate_orgs.GetPointer(util::DEVICE));
                        if (DEBUG &&(enactor_stats->retval = util::GRError("Expand_Incoming failed", __FILE__, __LINE__))) break;
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("asso_orgs",data_slice[0]->associate_orgs[0],graph_slice->nodes,thread_num,iteration[0]);
                        //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT,VertexId>("labe4",data_slice[0]->labels.GetPointer(util::GPU),graph_slice->nodes,thread_num,enactor_stats->iteration);
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("pred4",data_slice[0]->preds.GetPointer(util::GPU),graph_slice->nodes,thread_num,iteration[0]);
                        //util::cpu_mt::PrintGPUArray<SizeT,VertexId>("pred5",data_slice[0]->associate_orgs[1],graph_slice->nodes,thread_num,iteration[0]);
                        total_length+=data_slice[0]->in_length[enactor_stats->iteration%2][peer_];
                    }
                    if (enactor_stats->retval) break;
                    frontier_attribute->queue_length = total_length;
                    if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index,total_length,false,data_slice[0]->streams[0])) break;
                    //printf("%d\t%d\ttotal_length = %d\n",thread_num,iteration[0],total_length);fflush(stdout);
                    //if (total_length !=0)
                    //{
                    //    enactor_stats->done[0]=-1;
                    //} else {
                    //    enactor_stats->done[0]=0;
                    //}
                    if (DEBUG) util::cpu_mt::PrintMessage("mgpu data exchange end.", thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);
                }

                enactor_stats->iteration++;
            }

            if (enactor_stats->retval) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            if (enactor_stats->retval = work_progress->CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                enactor_stats->retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
            }
        } while(0);

        if (num_gpus >1) 
        {   
            if (break_clean) 
            {   
                util::cpu_mt::ReleaseBarrier(&(cpu_barrier[0]));
            }   
        }
        util::cpu_mt::PrintMessage("GPU BFS thread finished.", thread_num, enactor_stats->iteration, clock()-enactor_stats->start_time);
        CUT_THREADEND;
    }

/**
 * @brief BFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template <typename BFSProblem, bool INSTRUMENT>
class BFSEnactor : public EnactorBase
{
    typedef typename BFSProblem::SizeT    SizeT   ;
    typedef typename BFSProblem::VertexId VertexId;
    typedef typename BFSProblem::Value    Value   ;

    // Methods
public:

    /**
     * @brief BFSEnactor constructor
     */
    BFSEnactor(bool DEBUG = false, int num_gpus = 1, int* gpu_idx = NULL) :
        EnactorBase(EDGE_FRONTIERS, DEBUG, num_gpus, gpu_idx)//,
    {
        util::cpu_mt::PrintMessage("BFSEnactor() begin.");
        util::cpu_mt::PrintMessage("BFSEnactor() end.");
    }

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~BFSEnactor()
    {
        util::cpu_mt::PrintMessage("~BFSEnactor() begin.");
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            util::SetDevice(gpu_idx[gpu]);
            if (BFSProblem::ENABLE_IDEMPOTENCE)
            {
                cudaUnbindTexture(gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref);
            }
        }
        util::cpu_mt::PrintMessage("~BFSEnactor() end.");
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BFS search enacted.
     *
     * @param[out] total_queued Total queued elements in BFS kernel running.
     * @param[out] search_depth Search depth of BFS algorithm.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId  &search_depth,
        double    &avg_duty)
    {
        unsigned long long total_lifetimes=0;
        unsigned long long total_runtimes =0;
        total_queued = 0;
        search_depth = 0;
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (num_gpus!=1)
                if (util::SetDevice(gpu_idx[gpu])) return;
            cudaThreadSynchronize();

            total_queued += this->enactor_stats[gpu].total_queued;
            if (this->enactor_stats[gpu].iteration > search_depth) 
                search_depth = this->enactor_stats[gpu].iteration;
            total_lifetimes += this->enactor_stats[gpu].total_lifetimes;
            total_runtimes  += this->enactor_stats[gpu].total_runtimes;
        }
        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t InitBFS(
        BFSProblem *problem,
        int        max_grid_size = 0)
    {   
        cudaError_t retval = cudaSuccess;

        // Lazy initialization
        if (retval = EnactorBase::Init(problem,
                                       max_grid_size,
                                       AdvanceKernelPolicy::CTA_OCCUPANCY, 
                                       FilterKernelPolicy::CTA_OCCUPANCY)) return retval;
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (retval = util::SetDevice(gpu_idx[gpu])) break;
            if (BFSProblem::ENABLE_IDEMPOTENCE) {
                int bytes = (problem->graph_slices[gpu]->nodes + 8 - 1) / 8;
                cudaChannelFormatDesc   bitmask_desc = cudaCreateChannelDesc<char>();
                gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref.channelDesc = bitmask_desc;
                if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref,//ts_bitmask[gpu],
                    problem->data_slices[gpu]->visited_mask.GetPointer(util::DEVICE),
                    bytes),
                    "BFSEnactor cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;
            }
        }
        return retval;
    }

    cudaError_t Reset()
    {
        return EnactorBase::Reset();
    }
    /** @} */

    /**
     * @brief Enacts a breadth-first search computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam FilterPolicy Kernel policy for filter.
     * @tparam BFSProblem BFS Problem type.
     *
     * @param[in] problem BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactBFS(
    ContextPtr  *context,
    BFSProblem  *problem,
    VertexId    src,
    int         max_grid_size = 0)
    {
        clock_t  start_time = clock();
        cudaError_t              retval         = cudaSuccess;
        util::cpu_mt::CPUBarrier cpu_barrier    = util::cpu_mt::CreateBarrier(num_gpus);
        ThreadSlice              *thread_slices = new ThreadSlice [num_gpus];
        CUTThread                *thread_Ids    = new CUTThread   [num_gpus];

        do {
            // Determine grid size(s)
            if (DEBUG) {
                printf("Iteration, Edge map queue, Filter queue\n");
                printf("0");
            }

           // if (retval = EnactorBase::Reset()) break;
            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                thread_slices[gpu].thread_num    = gpu;
                thread_slices[gpu].problem       = (void*)problem;
                thread_slices[gpu].enactor       = (void*)this;
                thread_slices[gpu].cpu_barrier   = &cpu_barrier;
                thread_slices[gpu].context       = &(context[gpu*num_gpus]);
                enactor_stats[gpu].start_time    = start_time;
                if ((num_gpus ==1) || (gpu==problem->partition_tables[0][src]))
                     thread_slices[gpu].init_size=1;
                else thread_slices[gpu].init_size=0;
                thread_slices[gpu].thread_Id = cutStartThread(
                    (CUT_THREADROUTINE)&(BFSThread<INSTRUMENT,AdvanceKernelPolicy,FilterKernelPolicy,BFSProblem>),
                    (void*)&(thread_slices[gpu]));
                thread_Ids[gpu] = thread_slices[gpu].thread_Id;
            }

            cutWaitForThreads(thread_Ids, num_gpus);
            util::cpu_mt::DestoryBarrier(&cpu_barrier);
            
            for (int gpu=0;gpu<num_gpus;gpu++)
            if (this->enactor_stats[gpu].retval!=cudaSuccess) {retval=this->enactor_stats[gpu].retval;break;}
        } while(0);

        if (DEBUG) printf("\nGPU BFS Done.\n");
        delete[] thread_Ids   ; thread_Ids    = NULL;
        delete[] thread_slices; thread_slices = NULL;
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BFS Enact kernel entry.
     *
     * @tparam BFSProblem BFS Problem type. @see BFSProblem
     *
     * @param[in] problem Pointer to BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Enact(
        ContextPtr  *context,
        BFSProblem  *problem,
        VertexId    src,
        int         max_grid_size = 0)
    {
        util::cpu_mt::PrintMessage("BFSEnactor Enact() begin.");
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (BFSProblem::ENABLE_IDEMPOTENCE) {
            //if (this->cuda_props.device_sm_version >= 300) {
            if (min_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        AdvanceKernelPolicy;

                return EnactBFS<AdvanceKernelPolicy, FilterKernelPolicy>(
                        context, problem, src, max_grid_size);
            }
        } else {
                //if (this->cuda_props.device_sm_version >= 300) {
                if (min_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                    BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        AdvanceKernelPolicy;

                return EnactBFS<AdvanceKernelPolicy, FilterKernelPolicy>(
                        context, problem, src, max_grid_size);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief BFS Enact kernel entry.
     *
     * @tparam BFSProblem BFS Problem type. @see BFSProblem
     *
     * @param[in] problem Pointer to BFSProblem object.
     * @param[in] src Source node for BFS.
     * @param[in] max_grid_size Max grid size for BFS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Init(
        //ContextPtr  *context,
        BFSProblem  *problem,
        //VertexId    src,
        int         max_grid_size = 0)
    {
        util::cpu_mt::PrintMessage("BFSEnactor Enact() begin.");
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (BFSProblem::ENABLE_IDEMPOTENCE) {
            //if (this->cuda_props.device_sm_version >= 300) {
            if (min_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        AdvanceKernelPolicy;

                return InitBFS<AdvanceKernelPolicy, FilterKernelPolicy>(
                        problem, max_grid_size);
            }
        } else {
                //if (this->cuda_props.device_sm_version >= 300) {
                if (min_sm_version >= 300) {
                typedef gunrock::oprtr::filter::KernelPolicy<
                    BFSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                5,                                  // END_BITMASK_CULL
                8>                                  // LOG_SCHEDULE_GRANULARITY
                    FilterKernelPolicy;

                typedef gunrock::oprtr::advance::KernelPolicy<
                    BFSProblem,                         // Problem data type
                    300,                                // CUDA_ARCH
                    INSTRUMENT,                         // INSTRUMENT
                    8,                                  // MIN_CTA_OCCUPANCY
                    10,                                  // LOG_THREADS
                    8,                                  // LOG_BLOCKS
                    32*128,                                  // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
                    1,                                  // LOG_LOAD_VEC_SIZE
                    0,                                  // LOG_LOADS_PER_TILE
                    5,                                  // LOG_RAKING_THREADS
                    32,                            // WARP_GATHER_THRESHOLD
                    128 * 4,                            // CTA_GATHER_THRESHOLD
                    7,                                  // LOG_SCHEDULE_GRANULARITY
                    gunrock::oprtr::advance::LB>
                        AdvanceKernelPolicy;

                return InitBFS<AdvanceKernelPolicy, FilterKernelPolicy>(
                        problem, max_grid_size);
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }


    /** @} */

};

} // namespace bfs
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
