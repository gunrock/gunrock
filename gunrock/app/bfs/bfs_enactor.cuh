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
            context     = NULL;
            thread_num  = 0;
            init_size   = 0;
        }

        virtual ~ThreadSlice()
        {
            cpu_barrier = NULL;
            problem     = NULL;
            enactor     = NULL;
            context     = NULL;
        }
    };

    template <typename VertexId, typename SizeT, SizeT num_associates>
    __global__ void Expand_Incoming (
        const SizeT            num_elements,
        //const SizeT            incoming_offset,
        const VertexId*  const keys_in,
              VertexId*        keys_out,
              unsigned char*   marker,
              VertexId**       associate_in,
              VertexId**       associate_org)
    {
        //const SizeT STRIDE = gridDim.x * blockDim.x;
        //__shared__ VertexId* s_associate_in[2];
        //__shared__ VertexId* s_associate_org[2];
        //SizeT x2;
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
            //x2  = incoming_offset+x;
            key = keys_in[x];
            t   = associate_in[0][x];

            //printf("\t %d,%d,%d,%d,%d ",x2,key,t,associate_org[0][key],marker[key]);
            if (atomicCAS(associate_org[0]+key, -1, t)== -1)
            {
            } else {
               if (atomicMin(associate_org[0]+key, t)<t)
               {
                   keys_out[x]=-1;
                   return;//continue;
               }
            }
            //if (marker[key]==0) 
            if (atomicCAS(marker+key, 0, 1)==0)
            {
                //marker[key]=1;
                keys_out[x]=key;
            } else keys_out[x]=-1;
            if (num_associates==2) 
                associate_org[1][key]=associate_in[1][x];
            /*#pragma unroll
            for (SizeT i=1;i<num_associates;i++)
            {
                associate_org[i][key]=associate_in[i][x2];
            }*/
        }
    }

    template <typename BFSProblem>
    void ShowDebugInfo(
        int                    thread_num,
        int                    peer_,
        FrontierAttribute<typename BFSProblem::SizeT>      *frontier_attribute,
        EnactorStats           *enactor_stats,
        typename BFSProblem::DataSlice  *data_slice,
        typename BFSProblem::GraphSlice *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        std::string            check_name = "",
        cudaStream_t           stream = 0)
    {
        typedef typename BFSProblem::SizeT    SizeT;
        typedef typename BFSProblem::VertexId VertexId;
        typedef typename BFSProblem::Value    Value;
        SizeT queue_length;

        //util::cpu_mt::PrintMessage(check_name.c_str(), thread_num, enactor_stats->iteration);
        //printf("%d \t %d\t \t reset = %d, index = %d\n",thread_num, enactor_stats->iteration, frontier_attribute->queue_reset, frontier_attribute->queue_index);fflush(stdout);
        if (frontier_attribute->queue_reset) queue_length = frontier_attribute->queue_length;
        else if (enactor_stats->retval = util::GRError(work_progress->GetQueueLength(frontier_attribute->queue_index, queue_length, false, stream), "work_progress failed", __FILE__, __LINE__)) return;
	util::cpu_mt::PrintCPUArray<SizeT, SizeT>((check_name+" Queue_Length").c_str(), &(queue_length), 1, thread_num, enactor_stats->iteration);
	//printf("%d \t %d\t \t peer_ = %d, selector = %d, length = %d, p = %p\n",thread_num, enactor_stats->iteration, peer_, frontier_attribute->selector,queue_length,graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE));fflush(stdout);
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>((check_name+" keys").c_str(), graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE), queue_length, thread_num, enactor_stats->iteration,-1, stream);
	//if (graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE)!=NULL)
	//    util::cpu_mt::PrintGPUArray<SizeT, Value   >("valu1", graph_slice->frontier_queues.values[frontier_attribute->selector].GetPointer(util::DEVICE), _queue_length, thread_num, enactor_stats->iteration);
	//util::cpu_mt::PrintGPUArray<SizeT, VertexId>("labe1", data_slice[0]->labels.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
	//if (BFSProblem::MARK_PREDECESSORS)
	//    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("pred1", data_slice[0]->preds.GetPointer(util::DEVICE), graph_slice->nodes, thread_num, enactor_stats->iteration);
	//if (BFSProblem::ENABLE_IDEMPOTENCE)
	//    util::cpu_mt::PrintGPUArray<SizeT, unsigned char>("mask1", data_slice[0]->visited_mask.GetPointer(util::DEVICE), (graph_slice->nodes+7)/8, thread_num, enactor_stats->iteration);
    }
     
    template<
        bool     INSTRUMENT,
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename BFSProblem>
    void BFSCore(
        bool                   DEBUG,
        int                    thread_num,
        int                    peer_,
        FrontierAttribute<typename BFSProblem::SizeT>      *frontier_attribute,
        EnactorStats           *enactor_stats,
        typename BFSProblem::DataSlice  *data_slice,
        typename BFSProblem::DataSlice  *d_data_slice,
        typename BFSProblem::GraphSlice *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr             context,
        cudaStream_t           stream)
    {
        typedef typename BFSProblem::SizeT      SizeT;
        typedef typename BFSProblem::VertexId   VertexId;
        typedef typename BFSProblem::Value      Value;
        typedef typename BFSProblem::DataSlice  DataSlice;
        typedef typename BFSProblem::GraphSlice GraphSlice;
        typedef BFSEnactor<BFSProblem, INSTRUMENT> BfsEnactor;
        typedef BFSFunctor<VertexId, SizeT, VertexId, BFSProblem> BfsFunctor;

        if (frontier_attribute->queue_reset && frontier_attribute->queue_length ==0) 
        {
            work_progress->SetQueueLength(frontier_attribute->queue_index, 0, false, stream);
            if (DEBUG) util::cpu_mt::PrintMessage("return-1", thread_num, enactor_stats->iteration);
            return;
        }
        if (DEBUG) util::cpu_mt::PrintMessage("Advance begin",thread_num, enactor_stats->iteration);
        if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index+1,0,false,stream)) 
        {
            if (DEBUG) util::cpu_mt::PrintMessage("return0", thread_num, enactor_stats->iteration);
            return;
        }
        int queue_selector = (data_slice->num_gpus>1 && peer_==0 && enactor_stats->iteration>0)?data_slice->num_gpus:peer_;
        //printf("%d\t %d \t \t peer_ = %d, selector = %d, length = %d, index = %d\n",thread_num, enactor_stats->iteration, peer_, queue_selector,frontier_attribute->queue_length, frontier_attribute->queue_index);fflush(stdout); 

        // Edge Map
        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BFSProblem, BfsFunctor>(
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*)    NULL,
            (bool*)    NULL,
            data_slice ->scanned_edges[peer_].GetPointer(util::DEVICE),
            graph_slice->frontier_queues[queue_selector].keys[frontier_attribute->selector  ].GetPointer(util::DEVICE), 
            graph_slice->frontier_queues[peer_         ].keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),          
            (VertexId*)NULL,          // d_pred_in_queue
            graph_slice->frontier_queues[peer_       ].values[frontier_attribute->selector^1].GetPointer(util::DEVICE),          
            graph_slice->row_offsets   .GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*)   NULL,
            (VertexId*)NULL,
            graph_slice->nodes, //frontier_queues[queue_selector].keys[frontier_attribute->selector  ].GetSize(), /
            graph_slice->edges, //frontier_queues[peer_         ].keys[frontier_attribute->selector^1].GetSize(), 
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            false);
        
        // Only need to reset queue for once
        //if (frontier_attribute->queue_reset)
        frontier_attribute->queue_reset = false;
        //if (DEBUG && (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__))) 
        //{
        //    util::cpu_mt::PrintMessage("return1", thread_num, enactor_stats->iteration);
        //    return;
        //}
        if (DEBUG && (enactor_stats->retval = util::GRError("advance::Kernel failed", __FILE__, __LINE__))) 
        //{
        //    util::cpu_mt::PrintMessage("return2", thread_num, enactor_stats->iteration);
            return;
        //}
        if (DEBUG) util::cpu_mt::PrintMessage("Advance end", thread_num, enactor_stats->iteration);
        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;
        if (DEBUG || INSTRUMENT)
        {
            if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length,false,stream)) return;
            enactor_stats->total_queued += frontier_attribute->queue_length;
            if (DEBUG) ShowDebugInfo<BFSProblem>(thread_num, peer_, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, "post_advance", stream);
            if (INSTRUMENT) {
                if (enactor_stats->retval = enactor_stats->advance_kernel_stats.Accumulate(
                    enactor_stats->advance_grid_size,
                    enactor_stats->total_runtimes,
                    enactor_stats->total_lifetimes,
                    false,stream)) return;
            }
        }

        if (enactor_stats->retval = work_progress->SetQueueLength(frontier_attribute->queue_index+1, 0, false, stream)) return; 
        if (DEBUG) util::cpu_mt::PrintMessage("Filter begin", thread_num, enactor_stats->iteration);

        // Filter
        gunrock::oprtr::filter::Kernel<FilterKernelPolicy, BFSProblem, BfsFunctor>
        <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
            enactor_stats->iteration+1,
            frontier_attribute->queue_reset,
            frontier_attribute->queue_index,
            //num_gpus,
            frontier_attribute->queue_length,
            //enactor_stats->d_done,
            graph_slice->frontier_queues[peer_].keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
            graph_slice->frontier_queues[peer_].values[frontier_attribute->selector  ].GetPointer(util::DEVICE),    // d_pred_in_queue
            graph_slice->frontier_queues[peer_].keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),    // d_out_queue
            d_data_slice,
            data_slice->visited_mask.GetPointer(util::DEVICE),
            work_progress[0],
            graph_slice->frontier_queues[peer_].keys  [frontier_attribute->selector  ].GetSize(),
            //graph_slice->frontier_elements[peer_][frontier_attribute->selector  ],           // max_in_queue
            graph_slice->frontier_queues[peer_].keys  [frontier_attribute->selector^1].GetSize(),
            //graph_slice->frontier_elements[peer_][frontier_attribute->selector^1],         // max_out_queue
            enactor_stats->filter_kernel_stats);
	    //t_bitmask);

        //if (DEBUG && (enactor_stats->retval = util::GRError(cudaStreamSynchronize(stream), "cudaStreamSynchronize failed", __FILE__, __LINE__))) return;
	if (DEBUG && (enactor_stats->retval = util::GRError("filter_forward::Kernel failed", __FILE__, __LINE__))) return;
	if (DEBUG) util::cpu_mt::PrintMessage("Filter end.", thread_num, enactor_stats->iteration);
	frontier_attribute->queue_index++;
	frontier_attribute->selector ^= 1;
	if (INSTRUMENT || DEBUG) {
	    //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
	    //enactor_stats->total_queued += frontier_attribute->queue_length;
            if (DEBUG) ShowDebugInfo<BFSProblem>(thread_num, peer_, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, "post_filter", stream);
	    if (INSTRUMENT) {
		if (enactor_stats->retval = enactor_stats->filter_kernel_stats.Accumulate(
		    enactor_stats->filter_grid_size,
		    enactor_stats->total_runtimes,
		    enactor_stats->total_lifetimes,
		    false, stream)) return;
	    }
	}
    }

    template<
        bool     INSTRUMENT,
        bool     SIZE_CHECK,
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
        int          num_gpus              =   problem     -> num_gpus;
        int          thread_num            =   thread_data -> thread_num;
        int          gpu                   =   problem     -> gpu_idx           [thread_num];
        //util::Array1D<SizeT, DataSlice>
        DataSlice    *data_slice           =   problem     -> data_slices       [thread_num].GetPointer(util::HOST);
        util::Array1D<SizeT, DataSlice>
                     *s_data_slice         =   problem     -> data_slices;
        GraphSlice   *graph_slice          =   problem     -> graph_slices      [thread_num];
        GraphSlice   **s_graph_slice       =   problem     -> graph_slices;
        FrontierAttribute<SizeT>
                     *frontier_attribute   = &(enactor     -> frontier_attribute[thread_num*num_gpus]);
        FrontierAttribute<SizeT>
                     *s_frontier_attribute = &(enactor     -> frontier_attribute[0         ]);
        EnactorStats *enactor_stats        = &(enactor     -> enactor_stats     [thread_num*num_gpus]);
        EnactorStats *s_enactor_stats      = &(enactor     -> enactor_stats     [0         ]);
        util::CtaWorkProgressLifetime
                     *work_progress        = &(enactor     -> work_progress     [thread_num*num_gpus]);
        ContextPtr*  context               =   thread_data -> context;
        bool         DEBUG                 =   enactor     -> DEBUG;
        //texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> 
        //             *t_bitmask            = &(ts_bitmask       [thread_num]);
        util::cpu_mt::CPUBarrier
                     *cpu_barrier          =   thread_data -> cpu_barrier;
        //util::scan::MultiScan<VertexId,SizeT,true,256,8>*
        //             Scaner                = NULL;
        bool         break_clean           = true;
        SizeT        Total_Length          = 0;
        bool         First_Stage4          = true; 
        cudaError_t  tretval               = cudaSuccess;
        int          grid_size             = 0;
        std::string  mssg                  = "";
        bool         to_wait               = false;
 
        for (int peer=0;peer<num_gpus;peer++)
        {
            frontier_attribute[peer].queue_index    = 0;        // Work queue index
            frontier_attribute[peer].selector       = 0;
            frontier_attribute[peer].queue_length   = peer==0?thread_data -> init_size:0; //? 
            frontier_attribute[peer].queue_reset    = true;
        }

        do {
            util::cpu_mt::PrintMessage("BFS Thread begin.",thread_num, enactor_stats[0].iteration);
            if (enactor_stats[0].retval = util::SetDevice(gpu)) break;

            // Step through BFS iterations
            while (!All_Done<SizeT, DataSlice>(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) 
            {                
                if (num_gpus>1 && enactor_stats[0].iteration>0)
                {
                    frontier_attribute[0].queue_reset  = true;
                    //printf("%d data_slice = %p, %d\n",thread_num,data_slice,frontier_attribute[0].queue_length);
                    if (DEBUG) ShowDebugInfo<BFSProblem>(thread_num, 0, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, std::string("pre_scan"));
                    //if (DEBUG) {printf("gpu = %d, max_out_length = %d, out_length = %d\n",thread_num, data_slice->vertex_associate_out[0].GetSize(),frontier_attribute[0].queue_length);fflush(stdout);}
                    //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys0",graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector].GetPointer(util::DEVICE), frontier_attribute[0].queue_length, thread_num, enactor_stats[0].iteration);
                    if (SIZE_CHECK)
                    {
                        if (data_slice->vertex_associate_out[0].GetSize() < frontier_attribute[0].queue_length)
                        {
                            printf("%d\t %lld\t \t vertex_associate_out oversize : %d -> %d \n", 
                               thread_num, enactor_stats[0].iteration, 
                               data_slice->vertex_associate_out[0].GetSize(), frontier_attribute[0].queue_length);
                            for (int i=0;i< (BFSProblem::MARK_PREDECESSORS? 2:1); i++)
                            {
                                data_slice->vertex_associate_out[i].EnsureSize(frontier_attribute[0].queue_length);
                                data_slice->vertex_associate_outs[i]=data_slice->vertex_associate_out[i].GetPointer(util::DEVICE);
                            }
                            data_slice->vertex_associate_outs.Move(util::HOST, util::DEVICE);
                        }
                    }
                    if (BFSProblem::MARK_PREDECESSORS)
                        data_slice->Scaner->template Scan_with_dKeys2 <2,0> (
                            frontier_attribute[0].queue_length,
                            num_gpus,
                            graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector  ].GetPointer(util::DEVICE),
                            graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].GetPointer(util::DEVICE),
                            graph_slice  ->partition_table      .GetPointer(util::DEVICE),
                            graph_slice  ->convertion_table     .GetPointer(util::DEVICE),
                            data_slice   ->out_length           .GetPointer(util::DEVICE),
                            data_slice   ->vertex_associate_orgs.GetPointer(util::DEVICE),
                            data_slice   ->vertex_associate_outs.GetPointer(util::DEVICE),
                            data_slice   ->value__associate_orgs.GetPointer(util::DEVICE),
                            data_slice   ->value__associate_outs.GetPointer(util::DEVICE),
                            data_slice   ->streams[1]);
                    else data_slice->Scaner->template Scan_with_dKeys2 <1,0> (
                            frontier_attribute[0].queue_length,
                            num_gpus,
                            graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector  ].GetPointer(util::DEVICE),
                            graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].GetPointer(util::DEVICE),
                            graph_slice  ->partition_table      .GetPointer(util::DEVICE),
                            graph_slice  ->convertion_table     .GetPointer(util::DEVICE),
                            data_slice   ->out_length           .GetPointer(util::DEVICE),
                            data_slice   ->vertex_associate_orgs.GetPointer(util::DEVICE),
                            data_slice   ->vertex_associate_outs.GetPointer(util::DEVICE),
                            data_slice   ->value__associate_orgs.GetPointer(util::DEVICE),
                            data_slice   ->value__associate_outs.GetPointer(util::DEVICE),
                            data_slice   ->streams[1]);
                    if (enactor_stats[0].retval = data_slice->out_length.Move(util::DEVICE, util::HOST)) break;
                    if (DEBUG) util::cpu_mt::PrintCPUArray<SizeT,SizeT>("out_length",data_slice->out_length.GetPointer(util::HOST),num_gpus,thread_num,enactor_stats[0].iteration);
                    frontier_attribute[0].queue_index++;
                    frontier_attribute[0].selector ^=1;

                    if (DEBUG) ShowDebugInfo<BFSProblem>(thread_num, 0, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, std::string("post_scan"));
                    //if (DEBUG) util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys0",graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector].GetPointer(util::DEVICE), frontier_attribute[0].queue_length, thread_num, enactor_stats[0].iteration);
                    frontier_attribute[0].queue_reset  = true;
                    frontier_attribute[0].queue_offset = 0;
                    for (int i=1;i<num_gpus;i++)
                    {
                        frontier_attribute[i].selector      = frontier_attribute[0].selector;
                        frontier_attribute[i].advance_type  = frontier_attribute[0].advance_type;
                        frontier_attribute[i].queue_length  = data_slice->out_length[i];
                        frontier_attribute[i].queue_offset  = frontier_attribute[i-1].queue_offset + data_slice->out_length[i-1];
                        frontier_attribute[i].queue_reset   = true;
                        frontier_attribute[i].queue_index   = frontier_attribute[0].queue_index;
                        frontier_attribute[i].current_label = frontier_attribute[0].current_label;
                        enactor_stats     [i].iteration     = enactor_stats     [0].iteration;
                    }
                    frontier_attribute[0].queue_length = data_slice->out_length[0];
                } else {
                    frontier_attribute[0].queue_offset = 0;
                    frontier_attribute[0].queue_reset  = true;
                }
              
                Total_Length      = 0;
                First_Stage4      = true; 
                data_slice->wait_counter= 0;
                tretval           = cudaSuccess;
                to_wait           = false;
                for (int peer=0;peer<num_gpus;peer++)
                {
                    data_slice->stages[peer]=0;
                    data_slice->to_show[peer]=true;
                }
                data_slice->stages[0]   = 2;

                while (data_slice->wait_counter <num_gpus 
                       && (!All_Done(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)))
                {
                    for (int peer_=0;peer_<num_gpus;peer_++)
                    {
                        int peer = peer_<= thread_num? peer_-1   : peer_       ;
                        int gpu_ = peer <  thread_num? thread_num: thread_num+1;
                        if (DEBUG && data_slice->to_show[peer_])
                        {
                            //util::cpu_mt::PrintCPUArray<SizeT, int>("stages",data_slice->stages.GetPointer(util::HOST),num_gpus,thread_num,enactor_stats[peer_].iteration);
                            mssg="pre_stage0";
                            mssg[9]=char(data_slice->stages[peer_]+'0');
                            ShowDebugInfo<BFSProblem>(
                                thread_num, peer_, 
                                &frontier_attribute[peer_], 
                                &enactor_stats[peer_], 
                                data_slice, graph_slice, 
                                &work_progress[peer_], 
                                mssg,
                                data_slice->streams[peer_]);
                        }
                        data_slice->to_show[peer_]=true;

                        switch (data_slice->stages[peer_])
                        {
                        case 0: //Push Neibor
                            if (enactor_stats[peer_].iteration==0 || peer_==0) 
                            {  data_slice->to_show[peer_]=false; break;}
                            if (BFSProblem::MARK_PREDECESSORS)
                                PushNeibor <SIZE_CHECK, SizeT, VertexId, Value, GraphSlice, DataSlice, 2, 0> (
                                    thread_num,
                                    peer, 
                                    &frontier_attribute[peer_],
                                    &enactor_stats[peer_],
                                    s_data_slice  [thread_num].GetPointer(util::HOST),
                                    s_data_slice  [peer]      .GetPointer(util::HOST),
                                    s_graph_slice [thread_num],
                                    s_graph_slice [peer],
                                    data_slice->streams[peer_]);
                            else PushNeibor <SIZE_CHECK, SizeT, VertexId, Value, GraphSlice, DataSlice, 1, 0> (
                                    thread_num,
                                    peer, 
                                    &frontier_attribute[peer_],
                                    &enactor_stats[peer_],
                                    s_data_slice  [thread_num].GetPointer(util::HOST),
                                    s_data_slice  [peer]      .GetPointer(util::HOST),
                                    s_graph_slice [thread_num],
                                    s_graph_slice [peer],
                                    data_slice->streams[peer_]);
                            cudaEventRecord(data_slice->events[enactor_stats[peer_].iteration%4][peer_],data_slice->streams[peer_]);
                            data_slice->events_set[enactor_stats[peer_].iteration%4][peer_]=true;
                            break;

                        case 1: //Expand Incoming
                            if (peer_==0 || enactor_stats[peer_].iteration==0) {data_slice->to_show[peer_]=false;break;}
                            //if (data_slice->wait_marker[peer_]!=0) continue;
                            if (!(s_data_slice[peer]->events_set[enactor_stats[peer_].iteration%4][gpu_]))
                            {
                                data_slice->to_show[peer_]=false;
                                data_slice->stages[peer_]--;break;
                            }
                            //data_slice->wait_marker[peer_]=1;wait_counter++;
                            s_data_slice[peer]->events_set[enactor_stats[peer].iteration%4][gpu_]=false;
                            if (enactor_stats[peer_].retval = util::GRError(
                                cudaStreamWaitEvent(
                                    data_slice->streams[peer_], 
                                    s_data_slice[peer]->events[enactor_stats[peer_].iteration%4][gpu_], 
                                    0), "cudaStreamWaitEvent failed", __FILE__, __LINE__)) break;
                            frontier_attribute[peer_].queue_length = data_slice->in_length[enactor_stats[peer_].iteration%2][peer_];
                            data_slice->in_length[enactor_stats[peer_].iteration%2][peer_]=0;
                            if (SIZE_CHECK)
                            {
                                if (frontier_attribute[peer_].queue_length > graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetSize())
                                {
                                    printf("%d\t %lld\t %d\t queue1 oversize : %d -> %d\n", 
                                       thread_num, enactor_stats[peer_].iteration, peer_, 
                                       graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetSize(),
                                       frontier_attribute[peer_].queue_length);
                                    fflush(stdout);
                                    if (enactor_stats[peer_].retval = graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].EnsureSize(frontier_attribute[peer_].queue_length)) break;
                                    if (BFSProblem::USE_DOUBLE_BUFFER)
                                    {
                                        if (enactor_stats[peer_].retval = graph_slice->frontier_queues[peer_].values[frontier_attribute[peer_].selector].EnsureSize(frontier_attribute[peer_].queue_length)) break;
                                    }
                                }
                            }
                            //printf("%d\t %d\t \t %d-> %p+%d l=%d, stream = %d\n",thread_num,enactor_stats[peer_].iteration,
                            //    peer_,
                            //    data_slice->keys_in[enactor_stats[peer_].iteration%2].GetPointer(util::DEVICE), 
                            //    graph_slice->in_offset[peer_], 
                            //    frontier_attribute[peer_].queue_length,
                            //    data_slice->streams[peer_]);
                            //util::cpu_mt::PrintGPUArray<SizeT, VertexId>(
                            //    "keys_in",
                            //    data_slice->keys_in[enactor_stats[peer_].iteration%2].GetPointer(util::DEVICE) 
                            //        + graph_slice->in_offset[peer_], 
                            //    frontier_attribute[peer_].queue_length, 
                            //    thread_num, enactor_stats[peer_].iteration,
                            //    -1,data_slice->streams[peer_]);
                            if (frontier_attribute[peer_].queue_length ==0) break;
                            grid_size = frontier_attribute[peer_].queue_length/256+1;
                            //cudaStreamSynchronize(data_slice->streams[peer_]);
                            //if (enactor_stats[peer_].retval = util::GRError("cudaStreamSynchronize failed", __FILE__, __LINE__)) break;
                            //if (enactor_stats[0].iteration==2) break;
                            if (BFSProblem::MARK_PREDECESSORS) 
                                Expand_Incoming <VertexId, SizeT, 2>
                                    <<<grid_size,256,0,data_slice->streams[peer_]>>> (
                                    frontier_attribute[peer_].queue_length,
                                    //graph_slice ->in_offset[peer_],
                                    data_slice  ->keys_in  [enactor_stats[peer_].iteration%2][peer_].GetPointer(util::DEVICE),
                                    graph_slice ->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                                    data_slice  ->temp_marker     .GetPointer(util::DEVICE),
                                    data_slice  ->vertex_associate_ins[enactor_stats[peer_].iteration%2][peer_].GetPointer(util::DEVICE),
                                    data_slice  ->vertex_associate_orgs.GetPointer(util::DEVICE));
                            else Expand_Incoming <VertexId, SizeT, 1>
                                    <<<grid_size,256,0,data_slice->streams[peer_]>>> (
                                    frontier_attribute[peer_].queue_length,
                                    //graph_slice ->in_offset[peer_],
                                    data_slice  ->keys_in[enactor_stats[peer_].iteration%2][peer_].GetPointer(util::DEVICE),
                                    graph_slice ->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                                    data_slice  ->temp_marker   .GetPointer(util::DEVICE),
                                    data_slice  ->vertex_associate_ins[enactor_stats[peer_].iteration%2][peer_].GetPointer(util::DEVICE),
                                    data_slice  ->vertex_associate_orgs.GetPointer(util::DEVICE));
                            break;

                        case 2: //Comp Length
                            if (peer_>0 && enactor_stats[peer_].iteration==0) {data_slice->to_show[peer_]=false;break;}
                            //printf("%d\t %lld\t \t %d\t output_length = %p \n", thread_num, enactor_stats[peer_].iteration, peer_, frontier_attribute[peer_].output_length.GetPointer(util::DEVICE));
                            gunrock::oprtr::advance::ComputeOutputLength 
                                <AdvanceKernelPolicy, BFSProblem, BfsFunctor>(
                                &(frontier_attribute[peer_]),
                                graph_slice ->row_offsets     .GetPointer(util::DEVICE),
                                graph_slice ->column_indices  .GetPointer(util::DEVICE),
                                graph_slice ->frontier_queues[(num_gpus>1 && enactor_stats[peer_].iteration>0 && peer_==0)? num_gpus:peer_].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                                data_slice  ->scanned_edges    [peer_].GetPointer(util::DEVICE),
                                graph_slice ->nodes,//frontier_queues[peer_].keys[frontier_attribute[peer_].selector  ].GetSize(), 
                                graph_slice ->edges,//frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(),
                                context                        [peer_][0],
                                data_slice  ->streams          [peer_],
                                gunrock::oprtr::advance::V2V, true);
                            frontier_attribute[peer_].output_length.Move(util::DEVICE, util::HOST,1,0,data_slice->streams[peer_]);
                            if (SIZE_CHECK) cudaEventRecord(data_slice->local_events[peer_], data_slice->streams[peer_]);
                            break;

                        case 3: //BFS Core
                            if (peer_>0 && enactor_stats[peer_].iteration==0) {data_slice->to_show[peer_]=false;break;}
                            
                            if (SIZE_CHECK)
                            {
                                to_wait=true;
                                //for (int i=0;i<num_gpus;i++)
                                //if (i!=peer_ && data_slice->stages[i]<3)
                                //{
                                //to_wait=true;break;
                                //}
                                if (to_wait)
                                {
                                    tretval = cudaEventQuery(data_slice->local_events[peer_]);
                                    if (tretval == cudaErrorNotReady) 
                                    {   data_slice->to_show[peer_]=false;
                                        data_slice->stages[peer_]--; break;} 
                                    else if (tretval !=cudaSuccess) {enactor_stats[peer_].retval=tretval; break;}
                                } else {
                                    cudaStreamSynchronize(data_slice->streams[peer_]);
                                }

                                if (DEBUG) {printf("%d\t %lld\t %d\t queue_length = %d, output_length = %d\n",
                                        thread_num, enactor_stats[peer_].iteration, peer_,
                                        graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(), 
                                        frontier_attribute[peer_].output_length[0]);fflush(stdout);}
                                frontier_attribute[peer_].output_length[0]+=1;
                                if (frontier_attribute[peer_].output_length[0] > graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize())  
                                {
                                    printf("%d\t %lld\t %d\t queue3 oversize :\t %d ->\t %d\n",
                                        thread_num, enactor_stats[peer_].iteration, peer_,
                                        graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(), 
                                        frontier_attribute[peer_].output_length[0]);fflush(stdout);
                                    if (enactor_stats[peer_].retval = graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector  ].EnsureSize(frontier_attribute[peer_].output_length[0], true)) break;

                                    if (enactor_stats[peer_].retval = graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].EnsureSize(frontier_attribute[peer_].output_length[0])) break;
                                 
                                    if (BFSProblem::USE_DOUBLE_BUFFER) {
                                        if (enactor_stats[peer_].retval = graph_slice->frontier_queues[peer_].values[frontier_attribute[peer_].selector].EnsureSize(frontier_attribute[peer_].output_length[0],true)) break;
                                        if (enactor_stats[peer_].retval = graph_slice->frontier_queues[peer_].values[frontier_attribute[peer_].selector^1].EnsureSize(frontier_attribute[peer_].output_length[0]));
                                    }
                                   //if (enactor_stats[peer_].retval = cudaDeviceSynchronize()) break;
                                }
                            }
      
                            BFSCore < INSTRUMENT, AdvanceKernelPolicy, FilterKernelPolicy, BFSProblem>(
                                DEBUG,
                                thread_num,
                                peer_,
                                &(frontier_attribute[peer_]),
                                &(enactor_stats[peer_]),
                                data_slice,
                                s_data_slice[thread_num].GetPointer(util::DEVICE),
                                graph_slice,
                                &(work_progress[peer_]),
                                context[peer_],
                                data_slice->streams[peer_]);
                            if (enactor_stats[peer_].retval = work_progress[peer_].GetQueueLength(
                                frontier_attribute[peer_].queue_index, 
                                frontier_attribute[peer_].queue_length, 
                                false, 
                                data_slice->streams[peer_], 
                                true)) break; 
                            cudaEventRecord(data_slice->local_events[peer_], data_slice->streams[peer_]);
                            break;
                        
                        case 4: //Copy
                            if (num_gpus <=1 || ((peer_>0)&&(enactor_stats[peer_].iteration==0))) {data_slice->to_show[peer_]=false;break;}
                            to_wait = false;
                            for (int i=0;i<num_gpus;i++)
                                if (data_slice->stages[i]<4)
                                {
                                    to_wait=true;break;
                                }
                            if (to_wait)
                            {
                                tretval = cudaEventQuery(data_slice->local_events[peer_]);
                                if (tretval == cudaErrorNotReady) 
                                {
                                    data_slice->to_show[peer_]=false;
                                    data_slice->stages[peer_]--; break;} 
                                else if (tretval !=cudaSuccess) {enactor_stats[peer_].retval=tretval; break;}
                            } else cudaStreamSynchronize(data_slice->streams[peer_]);

                            /*if (DEBUG) 
                            {
                                printf("%d\t %lld\t %d\t org_length = %d, queue_length = %d, new_length = %d, max_length = %d\n", 
                                    thread_num, 
                                    enactor_stats[peer_].iteration, 
                                    peer_, 
                                    Total_Length, 
                                    frontier_attribute[peer_].queue_length,
                                    Total_Length + frontier_attribute[peer_].queue_length,
                                    graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector].GetSize());
                                fflush(stdout);
                            }*/
 
                            /*if (!SIZE_CHECK)     
                            {
                                //printf("output_length = %d, queue_size = %d\n", frontier_attribute[peer_].output_length[0], graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize());fflush(stdout);
                                if (frontier_attribute[peer_].output_length[0] > graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize())  
                                {
                                    printf("%d\t %lld\t %d\t queue3 oversize :\t %d ->\t %d\n",
                                        thread_num, enactor_stats[peer_].iteration, peer_,
                                        graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(), 
                                        frontier_attribute[peer_].output_length[0]);fflush(stdout);
                                }
                            }*/
                            if (frontier_attribute[peer_].queue_length!=0)
                            {
                                /*if (frontier_attribute[peer_].queue_length > 
                                    graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetSize())
                                {
                                   printf("%d\t %lld\t %d\t sub_queue oversize : queue_length = %d, queue_size = %d\n",
                                       thread_num, enactor_stats[peer_].iteration, peer_,
                                       frontier_attribute[peer_].queue_length, graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetSize());
                                   fflush(stdout);
                                }*/
                                if (SIZE_CHECK)
                                {
                                    if (Total_Length + frontier_attribute[peer_].queue_length > graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector].GetSize())
                                    {
                                        printf("%d\t %lld\t %d\t total_queue oversize : %d -> %d \n",
                                           thread_num, enactor_stats[peer_].iteration, peer_,
                                           Total_Length + frontier_attribute[peer_].queue_length,
                                           graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector].GetSize());fflush(stdout);
                                        if (enactor_stats[peer_].retval = graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector].EnsureSize(Total_Length+frontier_attribute[peer_].queue_length, true)) break;
                                        if (BFSProblem::USE_DOUBLE_BUFFER)
                                        {
                                            if (enactor_stats[peer_].retval = graph_slice->frontier_queues[num_gpus].values[frontier_attribute[0].selector].EnsureSize(Total_Length + frontier_attribute[peer_].queue_length, true)) break;
                                        }
                                    }
                                }
                                util::MemsetCopyVectorKernel<<<128, 128, 0, data_slice->streams[peer_]>>>(
                                    graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector].GetPointer(util::DEVICE) + Total_Length, 
                                    graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE), 
                                    frontier_attribute[peer_].queue_length);
                                if (BFSProblem::USE_DOUBLE_BUFFER)
                                    util::MemsetCopyVectorKernel<<<128,128,0,data_slice->streams[peer_]>>>(
                                        graph_slice->frontier_queues[num_gpus].values[frontier_attribute[0].selector].GetPointer(util::DEVICE) + Total_Length,
                                        graph_slice->frontier_queues[peer_].values[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                                        frontier_attribute[peer_].queue_length);
                                Total_Length+=frontier_attribute[peer_].queue_length;
                            }
                            if (First_Stage4)
                            {
                                First_Stage4=false;
                                util::MemsetKernel<<<128, 128, 0, data_slice->streams[peer_]>>>
                                    (data_slice->temp_marker.GetPointer(util::DEVICE), 
                                    (unsigned char)0, graph_slice->nodes);
                            }
                            break;

                        case 5: //End
                            data_slice->wait_counter++;
                            data_slice->to_show[peer_]=false;
                            break;
                        default:
                            data_slice->stages[peer_]--;
                            data_slice->to_show[peer_]=false;
                        }
                        
                        if (DEBUG)
                        {
                            mssg="stage 0 failed";
                            mssg[6]=char(data_slice->stages[peer_]+'0');
                            if (enactor_stats[peer_].retval = util::GRError(mssg, __FILE__, __LINE__)) break;
                        }
                        data_slice->stages[peer_]++;
                        //if (All_Done(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) break;
                    }
                    to_wait=true;
                    for (int i=0;i<num_gpus;i++)
                        if (data_slice->to_show[i])
                        {
                            to_wait=false;
                            break;
                        }
                    if (to_wait) sleep(0);
                }
              
                if (num_gpus>1)
                { 
                    for (int peer_=0;peer_<num_gpus;peer_++) 
                        cudaStreamSynchronize(data_slice->streams[peer_]);
                    if (SIZE_CHECK)
                    {
                        if (graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].GetSize() < Total_Length)
                        {
                            printf("%d\t %lld\t \t keysn oversize : %d -> %d \n",
                               thread_num, enactor_stats[0].iteration,
                               graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].GetSize(), Total_Length);
                            if (enactor_stats[0].retval = graph_slice->frontier_queues[num_gpus].keys[frontier_attribute[0].selector^1].EnsureSize(Total_Length)) break;
                            if (BFSProblem::USE_DOUBLE_BUFFER)
                            {
                                if (enactor_stats[0].retval = graph_slice->frontier_queues[num_gpus].values[frontier_attribute[0].selector^1].EnsureSize(Total_Length)) break;
                            }
                        }
                    }
                    if (BFSProblem::MARK_PREDECESSORS)
                    {  
                       for (int peer_=0;peer_<num_gpus;peer_++)
                        {
                            int grid_size = frontier_attribute[peer_].queue_length/256+1;
                            Copy_Preds<VertexId, SizeT> <<<grid_size,256>>>(
                                frontier_attribute[peer_].queue_length,
                                graph_slice->nodes,
                                graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                                data_slice->preds.GetPointer(util::DEVICE),
                                data_slice->temp_preds.GetPointer(util::DEVICE),
                                data_slice->temp_marker.GetPointer(util::DEVICE));
                        }
                        //for (int peer_=0;peer_<num_gpus;peer_++) 
                        //    cudaStreamSynchronize(data_slice->streams[peer_]);
                        for (int peer_=0;peer_<num_gpus;peer_++)
                        {
                            int grid_size = frontier_attribute[peer_].queue_length/256+1;
                            Update_Preds<VertexId,SizeT> <<<grid_size,256>>>(
                                frontier_attribute[peer_].queue_length,
                                graph_slice->nodes,
                                graph_slice->frontier_queues[peer_].keys[frontier_attribute[peer_].selector].GetPointer(util::DEVICE),
                                graph_slice->original_vertex.GetPointer(util::DEVICE),
                                data_slice->temp_preds.GetPointer(util::DEVICE),
                                data_slice->preds.GetPointer(util::DEVICE),
                                data_slice->temp_marker.GetPointer(util::DEVICE));
                        }
                        util::MemsetKernel<<<128, 128>>>
                            (data_slice->temp_marker.GetPointer(util::DEVICE), 
                            (unsigned char)0, graph_slice->nodes);
                    }
                         
                    //for (int peer_=0;peer_<num_gpus;peer_++) 
                    //    cudaStreamSynchronize(data_slice->streams[peer_]);
                    frontier_attribute[0].queue_length = Total_Length;
                } else {
                    if (enactor_stats[0].retval = work_progress[0].GetQueueLength(frontier_attribute[0].queue_index, frontier_attribute[0].queue_length, false, data_slice->streams[0])) break; 
                }
                enactor_stats[0].iteration++;
            }

            if (All_Done<SizeT, DataSlice>(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            for (int peer=0;peer<num_gpus;peer++)
            {
                bool overflowed = false;
                if (enactor_stats[peer].retval = work_progress[peer].CheckOverflow<SizeT>(overflowed)) break;
                if (overflowed) {
                    enactor_stats[peer].retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                    break;
                }
            }
        } while(0);

        if (num_gpus >1) 
        {   
            if (break_clean) 
            {   
                util::cpu_mt::ReleaseBarrier(cpu_barrier,thread_num);
            }
        }
        util::cpu_mt::PrintMessage("GPU BFS thread finished.", thread_num, enactor_stats->iteration);
        CUT_THREADEND;
    }

/**
 * @brief BFS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template <typename BFSProblem, bool INSTRUMENT>
class BFSEnactor : public EnactorBase<typename BFSProblem::SizeT>
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
        EnactorBase<SizeT>(VERTEX_FRONTIERS, DEBUG, num_gpus, gpu_idx)//,
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
        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            util::SetDevice(this->gpu_idx[gpu]);
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
        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            if (this->num_gpus!=1)
                if (util::SetDevice(this->gpu_idx[gpu])) return;
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
        if (retval = EnactorBase<SizeT>::Init(problem,
                                       max_grid_size,
                                       AdvanceKernelPolicy::CTA_OCCUPANCY, 
                                       FilterKernelPolicy::CTA_OCCUPANCY)) return retval;
        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            if (retval = util::SetDevice(this->gpu_idx[gpu])) break;
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
        return EnactorBase<SizeT>::Reset();
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
    int         max_grid_size = 0,
    bool        size_check = true)
    {
        clock_t  start_time = clock();
        cudaError_t              retval         = cudaSuccess;
        util::cpu_mt::CPUBarrier cpu_barrier    = util::cpu_mt::CreateBarrier(this->num_gpus);
        ThreadSlice              *thread_slices = new ThreadSlice [this->num_gpus];
        CUTThread                *thread_Ids    = new CUTThread   [this->num_gpus];

        do {
            // Determine grid size(s)
            if (this->DEBUG) {
                printf("Iteration, Edge map queue, Filter queue\n");
                printf("0");
            }

            /*for (int gpu=0;gpu<num_gpus;gpu++)
            {
                util::SetDevice(gpu_idx[gpu]);
                for (int i=0;i<num_gpus;i++)
                {
                    util::cpu_mt::PrintMessage("check point",gpu,i);
                    int _i=gpu*num_gpus+i;
                    frontier_attribute[_i].queue_index=0;
                    if (this->enactor_stats[_i].retval = work_progress[_i].SetQueueLength(frontier_attribute[_i].queue_index+2,0,true,problem->data_slices[gpu]->streams[i])) continue;
                    if (this->enactor_stats[_i].retval = util::GRError("check point hit", __FILE__, __LINE__)) continue; 
                }
            }
            break;*/

            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                if ((this->num_gpus ==1) || (gpu==problem->partition_tables[0][src]))
                     thread_slices[gpu].init_size=1;
                else thread_slices[gpu].init_size=0;
                this->frontier_attribute[gpu*this->num_gpus].queue_length = thread_slices[gpu].init_size;
            }
            
           // if (retval = EnactorBase::Reset()) break;
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                thread_slices[gpu].thread_num    = gpu;
                thread_slices[gpu].problem       = (void*)problem;
                thread_slices[gpu].enactor       = (void*)this;
                thread_slices[gpu].cpu_barrier   = &cpu_barrier;
                thread_slices[gpu].context       = &(context[gpu*this->num_gpus]);
                this->enactor_stats[gpu].start_time    = start_time;
                if (size_check)
                    thread_slices[gpu].thread_Id = cutStartThread(
                        (CUT_THREADROUTINE)&(BFSThread<INSTRUMENT, true, AdvanceKernelPolicy,FilterKernelPolicy,BFSProblem>),
                        (void*)&(thread_slices[gpu]));
                else thread_slices[gpu].thread_Id = cutStartThread(
                        (CUT_THREADROUTINE)&(BFSThread<INSTRUMENT, false, AdvanceKernelPolicy,FilterKernelPolicy,BFSProblem>),
                        (void*)&(thread_slices[gpu]));
                thread_Ids[gpu] = thread_slices[gpu].thread_Id;
            }

            cutWaitForThreads(thread_Ids, this->num_gpus);
            util::cpu_mt::DestoryBarrier(&cpu_barrier);
            
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            if (this->enactor_stats[gpu].retval!=cudaSuccess) {retval=this->enactor_stats[gpu].retval;break;}
        } while(0);

        if (this->DEBUG) printf("\nGPU BFS Done.\n");
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
        int         max_grid_size = 0,
        bool        size_check = true)
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
                        context, problem, src, max_grid_size, size_check);
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
                        context, problem, src, max_grid_size, size_check);
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
