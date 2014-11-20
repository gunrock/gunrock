// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sssp_enactor.cuh
 *
 * @brief SSSP Problem Enactor
 */

#pragma once

#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
//#include <gunrock/util/scan/multi_scan.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>
#include <gunrock/priority_queue/kernel.cuh>
#include <gunrock/priority_queue/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/app/sssp/sssp_functor.cuh>


namespace gunrock {
namespace app {
namespace sssp {

    template <typename SSSPProblem, bool INSTRUMENT> class SSSPEnactor;

    struct ThreadSlice
    {
    public:
        int         thread_num;
        int         init_size;
        CUTThread   thread_Id;
        int         stats;
        //util::cpu_mt::CPUBarrier *cpu_barrier;
        void*       problem;
        void*       enactor;
        ContextPtr* context;

        ThreadSlice()
        {
            //cpu_barrier = NULL;
            problem     = NULL;
            enactor     = NULL;
            context     = NULL;
            thread_num  = 0;
            init_size   = 0;
            stats       = -2;
        }

        virtual ~ThreadSlice()
        {
            //cpu_barrier = NULL;
            problem     = NULL;
            enactor     = NULL;
            context     = NULL;
        }
    };

    template <typename VertexId, typename SizeT, typename Value, SizeT num_vertex_associates, SizeT num_value__associates>
    __global__ void Expand_Incoming (
        const SizeT            num_elements,
        //const SizeT            num_associates,
        //const SizeT            incoming_offset,
        const VertexId* const  keys_in,
              VertexId*        keys_out,
              //unsigned int*    marker,
        const size_t           array_size,
              char*            array)
        //      VertexId**       vertex_associate_in,
        //      Value**          value__associate_in,
        //      VertexId**       vertex_associate_org,
        //      Value**          value__associate_org)
    {
        extern __shared__ char s_array[];
        const SizeT STRIDE = gridDim.x * blockDim.x;
        size_t      offset                = 0;
        VertexId** s_vertex_associate_in  = (VertexId**)&(s_array[offset]);
        offset+=sizeof(VertexId*) * num_vertex_associates;
        Value**    s_value__associate_in  = (Value**   )&(s_array[offset]);
        offset+=sizeof(Value*   ) * num_value__associates;
        VertexId** s_vertex_associate_org = (VertexId**)&(s_array[offset]);
        offset+=sizeof(VertexId*) * num_vertex_associates;
        Value**    s_value__associate_org = (Value**   )&(s_array[offset]);
        SizeT x = threadIdx.x;
        if (x < array_size)
        {
            s_array[x]=array[x];
            x+=blockDim.x; 
        }
        __syncthreads();

        x = blockIdx.x * blockDim.x + threadIdx.x;

        //if (x>=num_elements) return;
        //SizeT x2=incoming_offset+x;
        while (x<num_elements)
        {
            VertexId key=keys_in[x];
            Value t=s_value__associate_in[0][x];

            //if (atomicCAS(associate_org[0]+key, -1, t)== -1)
            //{
            //} else {
            if (atomicMin(s_value__associate_org[0]+key, t)<t)
            {
                keys_out[x]=-1;
                x+=STRIDE;
                continue;
            }
            //}
        
            //if (atomicCAS(marker+key, 0, 1) ==0)
            //{   
                keys_out[x]=key;
            //} else keys_out[x]=-1;

            #pragma unrool
            for (SizeT i=1;i<num_value__associates;i++)
                s_value__associate_org[i][key]=s_value__associate_in[i][x];
            #pragma unrool
            for (SizeT i=0;i<num_vertex_associates;i++)
                s_vertex_associate_org[i][key]=s_vertex_associate_in[i][x];
            x+=STRIDE;
        }
    }

    template <typename SSSPProblem>
    void ShowDebugInfo(
        int                    thread_num,
        int                    peer_,
        FrontierAttribute<typename SSSPProblem::SizeT>      *frontier_attribute,
        EnactorStats           *enactor_stats,
        typename SSSPProblem::DataSlice  *data_slice,
        typename SSSPProblem::GraphSlice *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        std::string            check_name = "",
        cudaStream_t           stream = 0)
    {
        typedef typename SSSPProblem::SizeT    SizeT;
        typedef typename SSSPProblem::VertexId VertexId;
        typedef typename SSSPProblem::Value    Value;
        SizeT queue_length;

        //util::cpu_mt::PrintMessage(check_name.c_str(), thread_num, enactor_stats->iteration);
        //printf("%d \t %d\t \t reset = %d, index = %d\n",thread_num, enactor_stats->iteration, frontier_attribute->queue_reset, frontier_attribute->queue_index);fflush(stdout);
        //if (frontier_attribute->queue_reset) 
            queue_length = frontier_attribute->queue_length;
        //else if (enactor_stats->retval = util::GRError(work_progress->GetQueueLength(frontier_attribute->queue_index, queue_length, false, stream), "work_progress failed", __FILE__, __LINE__)) return;
        //util::cpu_mt::PrintCPUArray<SizeT, SizeT>((check_name+" Queue_Length").c_str(), &(queue_length), 1, thread_num, enactor_stats->iteration);
        printf("%d\t %lld\t %d\t stage%d\t %s\t Queue_Length = %d\n", thread_num, enactor_stats->iteration, peer_, data_slice->stages[peer_], check_name.c_str(), queue_length);fflush(stdout);
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

    template <
        bool     INSTRUMENT,
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename SSSPProblem>
    void SSSPCore(
        bool     DEBUG,
        int      thread_num,
        int      peer_,
        FrontierAttribute<typename SSSPProblem::SizeT> *frontier_attribute,
        EnactorStats                     *enactor_stats,
        typename SSSPProblem::DataSlice  *data_slice,
        typename SSSPProblem::DataSlice  *d_data_slice,
        typename SSSPProblem::GraphSlice *graph_slice,
        util::CtaWorkProgressLifetime    *work_progress,
        ContextPtr                         context,
        cudaStream_t                      stream)
    {
        typedef typename SSSPProblem::SizeT      SizeT;
        typedef typename SSSPProblem::VertexId   VertexId;
        typedef typename SSSPProblem::Value      Value;
        typedef typename SSSPProblem::DataSlice  DataSlice;
        typedef typename SSSPProblem::GraphSlice GraphSlice;
        typedef SSSPEnactor<SSSPProblem, INSTRUMENT> SsspEnactor;
        typedef SSSPFunctor<VertexId, SizeT, VertexId, SSSPProblem> SsspFunctor;

        if (frontier_attribute->queue_reset && frontier_attribute->queue_length ==0)
        {
            work_progress->SetQueueLength(frontier_attribute->queue_index, 0, false, stream);
            if (DEBUG) util::cpu_mt::PrintMessage("return-1", thread_num, enactor_stats->iteration);
            return;
        }
        if (DEBUG) util::cpu_mt::PrintMessage("Advance begin",thread_num, enactor_stats->iteration);

        // Edge Map
        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, SSSPProblem, SsspFunctor>(
            //d_done,
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*)    NULL,
            (bool*)    NULL,
            data_slice ->scanned_edges  [peer_]                                     .GetPointer(util::DEVICE),
            graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector]  .GetPointer(util::DEVICE), // d_in_queue
            graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
            (VertexId*)NULL,
            (VertexId*)NULL,
            graph_slice->row_offsets   .GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*)   NULL,
            (VertexId*)NULL,
            graph_slice->nodes,//graph_slice->frontier_elements[frontier_attribute.selector], // max_in_queue
            graph_slice->edges,//graph_slice->frontier_elements[frontier_attribute.selector^1], // max_out_queue
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            false);

        frontier_attribute->queue_reset = false;
        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;
        if (DEBUG) util::cpu_mt::PrintMessage("Advance end", thread_num, enactor_stats->iteration);
        if (false) //(DEBUG || INSTRUMENT)
        {
            if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length,false,stream)) return;
            enactor_stats->total_queued += frontier_attribute->queue_length;
            if (DEBUG) ShowDebugInfo<SSSPProblem>(thread_num, peer_, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, "post_advance", stream);
            if (INSTRUMENT) {
                if (enactor_stats->retval = enactor_stats->advance_kernel_stats.Accumulate(
                    enactor_stats->advance_grid_size,
                    enactor_stats->total_runtimes,
                    enactor_stats->total_lifetimes,
                    false,stream)) return;
            }
        }

        //Vertex Map
        gunrock::oprtr::filter::Kernel<FilterKernelPolicy, SSSPProblem, SsspFunctor>
            <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
            enactor_stats->iteration+1,
            frontier_attribute->queue_reset,
            frontier_attribute->queue_index,
            //enactor_stats.num_gpus,
            frontier_attribute->queue_length,
            //d_done,
            graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
            NULL,                                                                  // d_pred_in_queue
            graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),     // d_out_queue
            d_data_slice,
            NULL,
            work_progress[0],
            graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector  ].GetSize(), // max_in_queue
            graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector^1].GetSize(), // max_out_queue
            enactor_stats->filter_kernel_stats);
        
        if (DEBUG && (enactor_stats->retval = util::GRError("filter_forward::Kernel failed", __FILE__, __LINE__))) return;
        if (DEBUG) util::cpu_mt::PrintMessage("Filter end.", thread_num, enactor_stats->iteration);

        //TODO: split the output queue into near/far pile, put far pile in far queue, put near pile as the input queue
        //for next round.
        /*out_length = 0;
        if (frontier_attribute->queue_length > 0) {
            out_length = gunrock::priority_queue::Bisect
                <PriorityQueueKernelPolicy, SSSPProblem, NearFarPriorityQueue, PqFunctor>(
                (int*)graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                pq,
                (unsigned int)frontier_attribute->queue_length,
                d_data_slice,
                graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                pq->queue_length,
                pq_level,
                (pq_level+1),
                context[0],
                graph_slice->nodes);
            //printf("out:%d, pq_length:%d\n", out_length, pq->queue_length);
            if (retval = work_progress->SetQueueLength(frontier_attribute->queue_index, out_length)) break;
        }
        //
        //If the output queue is empty and far queue is not, then add priority level and split the far pile.
        if ( out_length == 0 && pq->queue_length > 0) {
            while (pq->queue_length > 0 && out_length == 0) {
                pq->selector ^= 1;
                pq_level++;
                out_length = gunrock::priority_queue::Bisect
                    <PriorityQueueKernelPolicy, SSSPProblem, NearFarPriorityQueue, PqFunctor>(
                    (int*)pq->nf_pile[0]->d_queue[pq->selector^1],
                    pq,
                    (unsigned int)pq->queue_length,
                    d_data_slice,
                    graph_slice->frontier_queues[peer_].keys[frontier_attribute->selector^1],
                    0,
                    pq_level,
                    (pq_level+1),
                    context[0],
                    graph_slice->nodes);
                //printf("out after p:%d, pq_length:%d\n", out_length, pq->queue_length);
                if (out_length > 0) {
                    if (retval = work_progress->SetQueueLength(frontier_attribute->queue_index, out_length)) break;
                }
            }
        }
        */

        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;

        if (false) {//(INSTRUMENT || DEBUG) {
            //if (enactor_stats->retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
            //enactor_stats->total_queued += frontier_attribute->queue_length;
            if (DEBUG) ShowDebugInfo<SSSPProblem>(thread_num, peer_, frontier_attribute, enactor_stats, data_slice, graph_slice, work_progress, "post_filter", stream);
            if (INSTRUMENT) {
                if (enactor_stats->retval = enactor_stats->filter_kernel_stats.Accumulate(
                    enactor_stats->filter_grid_size,
                    enactor_stats->total_runtimes,
                    enactor_stats->total_lifetimes,
                    false, stream)) return;
            }
        }
    }

    template <
        bool     INSTRUMENT,
        bool     SIZE_CHECK,
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename SSSPProblem>
    static CUT_THREADPROC SSSPThread(
        void * thread_data_)
    {
        typedef typename SSSPProblem::SizeT      SizeT;
        typedef typename SSSPProblem::VertexId   VertexId;
        typedef typename SSSPProblem::Value      Value;
        typedef typename SSSPProblem::DataSlice  DataSlice;
        typedef typename SSSPProblem::GraphSlice GraphSlice;
        typedef SSSPEnactor<SSSPProblem, INSTRUMENT> SsspEnactor;
        typedef SSSPFunctor<VertexId, SizeT, Value, SSSPProblem> SsspFunctor;

        ThreadSlice  *thread_data         = (ThreadSlice*) thread_data_;
        SSSPProblem  *problem             = (SSSPProblem*) thread_data->problem;
        SsspEnactor  *enactor             = (SsspEnactor*) thread_data->enactor;
        int          num_gpus             =   problem     -> num_gpus;
        int          thread_num           =   thread_data -> thread_num;
        int          gpu_idx              =   problem     -> gpu_idx            [thread_num] ;
        DataSlice    *data_slice          =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
        util::Array1D<SizeT, DataSlice>
                     *s_data_slice        =   problem     -> data_slices;
        GraphSlice   *graph_slice         =   problem     -> graph_slices       [thread_num] ;
        GraphSlice   **s_graph_slice      =   problem     -> graph_slices;
        FrontierAttribute<SizeT>
                     *frontier_attribute  = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
        FrontierAttribute<SizeT>
                     *s_frontier_attribute= &(enactor     -> frontier_attribute [0         ]);
        EnactorStats *enactor_stats       = &(enactor     -> enactor_stats      [thread_num * num_gpus]);
        EnactorStats *s_enactor_stats     = &(enactor     -> enactor_stats      [0         ]);
        util::CtaWorkProgressLifetime
                     *work_progress       = &(enactor     -> work_progress      [thread_num * num_gpus]);
        ContextPtr   *context             =   thread_data -> context;
        bool         DEBUG                =   enactor     -> DEBUG;
        int*         stages               =   data_slice  -> stages .GetPointer(util::HOST);
        bool*        to_show              =   data_slice  -> to_show.GetPointer(util::HOST);
        cudaStream_t* streams             =   data_slice  -> streams.GetPointer(util::HOST);
        //util::cpu_mt::CPUBarrier
        //             *cpu_barrier         =   thread_data -> cpu_barrier;
        //util::scan::MultiScan<VertexId,SizeT,true,256,8>*
        //             Scaner               = NULL;
        //bool         break_clean          = true;
        //SizeT*       out_offset           = NULL;
        //char*        message              = new char [1024];
        //util::Array1D<SizeT, unsigned int>  scanned_edges;
        //util::Array1D<SizeT, VertexId    >  temp_preds;
        SizeT        Total_Length          = 0; 
        //bool         First_Stage4          = true; 
        cudaError_t  tretval               = cudaSuccess;
        int          grid_size             = 0;
        std::string  mssg                  = "";
        int          pre_stage             = 0; 
        size_t       offset                = 0;
        int          num_vertex_associate  = SSSPProblem::MARK_PATHS?1:0;
        int          num_value__associate  = 1;
        
        do {
            if (enactor_stats[0].retval = util::SetDevice(gpu_idx)) break;
            thread_data->stats = 1;
            while (thread_data->stats != 2) sleep(0);
            thread_data->stats = 3;

            for (int peer_=0;peer_<num_gpus;peer_++)
            {
                frontier_attribute[peer_].queue_index  = 0;        // Work queue index
                frontier_attribute[peer_].selector     = 0;
                frontier_attribute[peer_].queue_length = peer_==0?thread_data->init_size:0; 
                frontier_attribute[peer_].queue_reset  = true;
                enactor_stats     [peer_].iteration    = 0;
            }

            /*if (enactor_stats->retval = util::SetDevice(gpu)) break;
            if (num_gpus > 1)
            {
                Scaner = new util::scan::MultiScan<VertexId, SizeT, true, 256, 8>;
                out_offset = new SizeT[num_gpus +1];
            }
            scanned_edges.SetName("scanned_edges");
            if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                if (enactor_stats->retval = scanned_edges.Allocate(graph_slice->edges*2, util::DEVICE)) break; 
            }
            temp_preds.SetName("temp_preds");
            if (SSSPProblem::MARK_PATHS) 
                if (enactor_stats->retval = temp_preds.Allocate(graph_slice->nodes, util::DEVICE)) break;
            */
            //unsigned int pq_level = 0; 
            //unsigned int out_length = 0;

            // Step through SSSP iterations
            //while (out_length > 0 || pq->queue_length > 0 || frontier_attribute.queue_length > 0) {
            while (!All_Done<SizeT, DataSlice>(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) 
            {

                if (num_gpus>1 && enactor_stats[0].iteration>0)
                {
                    frontier_attribute[0].queue_reset  = true;
                    frontier_attribute[0].queue_offset = 0;
                    for (int i=1; i<num_gpus; i++)
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

                Total_Length      = 0; 
                //First_Stage4      = true; 
                data_slice->wait_counter= 0;
                tretval           = cudaSuccess;
                //to_wait           = false;
                for (int peer=0;peer<num_gpus;peer++)
                {    
                    stages [peer] = 0   ; stages [peer+num_gpus]=0;
                    to_show[peer] = true; to_show[peer+num_gpus]=true;
                    for (int i=0;i<data_slice->num_stages;i++)
                        data_slice->events_set[enactor_stats[0].iteration%4][peer][i]=false;
                }    

                while (data_slice->wait_counter <num_gpus*2 
                       && (!All_Done(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)))
                {    
                    for (int peer__=0;peer__<num_gpus*2;peer__++)
                    {    
                        //if (peer__==num_gpus) continue;
                        int peer_ = (peer__%num_gpus);
                        int peer = peer_<= thread_num? peer_-1   : peer_       ;    
                        int gpu_ = peer <  thread_num? thread_num: thread_num+1;
                        if (DEBUG && to_show[peer__])
                        {    
                            //util::cpu_mt::PrintCPUArray<SizeT, int>("stages",data_slice->stages.GetPointer(util::HOST),num_gpus,thread_num,enactor_stats[peer_].iteration);
                            //mssg="pre_stage0";
                            //mssg[9]=char(stages[peer_]+'0');
                            mssg="";
                            ShowDebugInfo<SSSPProblem>(
                                thread_num, 
                                peer__, 
                                &frontier_attribute[peer_], 
                                &enactor_stats[peer_], 
                                data_slice, 
                                graph_slice, 
                                &work_progress[peer_], 
                                mssg,
                                streams[peer__]);
                        }
                        to_show[peer__]=true;
                        int iteration  = enactor_stats[peer_].iteration;
                        int iteration_ = iteration%4;
                        pre_stage      = stages[peer__];
                        //int queue_selector = 0;
                        int selector   = frontier_attribute[peer_].selector;
                        util::DoubleBuffer<SizeT, VertexId, VertexId>
                            *frontier_queue_ = &(graph_slice->frontier_queues[peer_]);
                        FrontierAttribute<SizeT>* frontier_attribute_ = &(frontier_attribute[peer_]);
                        EnactorStats* enactor_stats_ = &(enactor_stats[peer_]);
 
                        switch (stages[peer__])
                        {
                        case 0: // Assign marker & Scan
                            if (peer_==0) {
                                //cudaEventRecord(data_slice->events[iteration_][peer_][6],streams[peer_]);
                                //data_slice->events_set[iteration_][peer_][6]=true; 
                                //printf("%d\t %d\t %d\t jump to stage 7\n", thread_num, iteration, peer_);fflush(stdout);
                                if (peer__==num_gpus || frontier_attribute_->queue_length==0) stages[peer__]=3;
                                break;
                            } else if ((iteration==0 || data_slice->out_length[peer_]==0) && peer__>num_gpus) {
                                cudaEventRecord(data_slice->events[iteration_][peer_][0],streams[peer__]);
                                //cudaEventRecord(data_slice->events[iteration_][peer_][6],streams[peer_]);
                                data_slice->events_set[iteration_][peer_][0]=true;
                                //data_slice->events_set[iteration_][peer_][6]=true;
                                //frontier_attribute_->queue_length=0;
                                //printf("%d\t %d\t %d\t jump to stage 7\n", thread_num, iteration, peer_);fflush(stdout);
                                stages[peer__]=3;break;
                            //} else if (num_gpus<2 || (peer_==0 && iteration==0)) {
                            //    //printf("%d\t %d\t %d\t jump to stage 4\n", thread_num, iteration, peer_);fflush(stdout);
                            //    break;
                            } /*else if (peer_>0 && frontier_attribute_->queue_length==0) {
                                cudaEventRecord(data_slice->events[iteration_][peer_][2],streams[peer_]);
                                data_slice->events_set[iteration_][peer_][2]=true;
                                data_slice->out_length[peer_]=0;
                                //printf("%d\t %d\t %d\t jump to stage 3\n", thread_num, iteration, peer_);fflush(stdout);
                                stages[peer_]=2;break;
                            }*/
                            //printf("%d\t %d\t %d\t entered stage0\n", thread_num, iteration, peer_);fflush(stdout);

                            if (peer__<num_gpus)
                            { //wait and expand incoming
                                if (!(s_data_slice[peer]->events_set[iteration_][gpu_][0]))
                                {   to_show[peer__]=false;stages[peer__]--;break;}

                                frontier_attribute_->queue_length = data_slice->in_length[iteration%2][peer_];
                                data_slice->in_length[iteration%2][peer_]=0;
                                if (frontier_attribute_->queue_length ==0)
                                {
                                    //cudaEventRecord(data_slice->events[iteration_][peer_][6],streams[peer_]);
                                    //data_slice->events_set[iteration_][peer_][6]=true; 
                                    //printf("%d\t %d\t %d\t jump to stage 7\n", thread_num, iteration, peer_);fflush(stdout);
                                    stages[peer__]=3;break;
                                }

                                if (SIZE_CHECK)
                                {
                                    if (frontier_attribute_->queue_length > frontier_queue_->keys[selector^1].GetSize())
                                    {
                                        printf("%d\t %d\t %d\t queue1 oversize : %d -> %d\n",
                                            thread_num, iteration, peer_,
                                            frontier_queue_->keys[selector^1].GetSize(),
                                            frontier_attribute_->queue_length);
                                        fflush(stdout);
                                        if (enactor_stats_->retval = frontier_queue_->keys[selector^1].EnsureSize(frontier_attribute_->queue_length)) break;
                                        if (SSSPProblem::USE_DOUBLE_BUFFER)
                                        {
                                            if (enactor_stats_->retval = frontier_queue_->values[selector^1].EnsureSize(frontier_attribute_->queue_length)) break;
                                        }
                                    }
                                }

                                offset = 0;
                                memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                         data_slice -> vertex_associate_ins[iteration%2][peer_].GetPointer(util::HOST),
                                          sizeof(SizeT*   ) * num_vertex_associate);
                                offset += sizeof(SizeT*   ) * num_vertex_associate ;
                                memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                         data_slice -> value__associate_ins[iteration%2][peer_].GetPointer(util::HOST),
                                          sizeof(VertexId*) * num_value__associate);
                                offset += sizeof(VertexId*) * num_value__associate ;
                                memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                         data_slice -> vertex_associate_orgs.GetPointer(util::HOST),
                                          sizeof(VertexId*) * num_vertex_associate);
                                offset += sizeof(VertexId*) * num_vertex_associate ;
                                memcpy(&(data_slice -> expand_incoming_array[peer_][offset]),
                                         data_slice -> value__associate_orgs.GetPointer(util::HOST),
                                          sizeof(Value*   ) * num_value__associate);
                                data_slice->expand_incoming_array[peer_].Move(util::HOST, util::DEVICE, -1, 0, streams[peer_]);
                                grid_size = frontier_attribute_->queue_length/256+1;
                                if (grid_size>512) grid_size=512;
                                //cudaStreamSynchronize(data_slice->streams[peer_]);
                                //if (enactor_stats[peer_].retval = util::GRError("cudaStreamSynchronize failed", __FILE__, __LINE__)) break;
                                //if (enactor_stats[0].iteration==2) break;
                                cudaStreamWaitEvent(streams[peer_],
                                    s_data_slice[peer]->events[iteration_][gpu_][0], 0);
                                Expand_Incoming <VertexId, SizeT, Value, SSSPProblem::MARK_PATHS?1:0, 1>
                                    <<<grid_size,256, data_slice->expand_incoming_array[peer_].GetSize(),streams[peer_]>>> (
                                    frontier_attribute_->queue_length,
                                    //graph_slice ->in_offset[peer_],
                                    data_slice ->keys_in             [iteration%2][peer_].GetPointer(util::DEVICE),
                                    frontier_queue_->keys                    [selector^1].GetPointer(util::DEVICE),
                                    //data_slice ->temp_marker                             .GetPointer(util::DEVICE),
                                    data_slice ->expand_incoming_array[peer_].GetSize(),
                                    data_slice ->expand_incoming_array[peer_].GetPointer(util::DEVICE));
                                frontier_attribute_->selector^=1;
                                frontier_attribute_->queue_index++;

                            } else { //Push Neibor
                                PushNeibor <SIZE_CHECK, SizeT, VertexId, Value, GraphSlice, DataSlice,
                                        SSSPProblem::MARK_PATHS?1:0, 1> (
                                    thread_num,
                                    peer,
                                    data_slice->out_length[peer_],
                                    enactor_stats_,
                                    s_data_slice  [thread_num].GetPointer(util::HOST),
                                    s_data_slice  [peer]      .GetPointer(util::HOST),
                                    s_graph_slice [thread_num],
                                    s_graph_slice [peer],
                                    streams       [peer__]);
                                cudaEventRecord(data_slice->events[iteration_][peer_][stages[peer__]],streams[peer__]);
                                data_slice->events_set[iteration_][peer_][stages[peer__]]=true;
                                stages[peer__]=3;
                            }
                            break;

                        case 1: //Comp Length                           
                            gunrock::oprtr::advance::ComputeOutputLength
                                <AdvanceKernelPolicy, SSSPProblem, SsspFunctor>(
                                frontier_attribute_,
                                graph_slice ->row_offsets     .GetPointer(util::DEVICE),
                                graph_slice ->column_indices  .GetPointer(util::DEVICE),
                                graph_slice ->frontier_queues  [peer_].keys[selector].GetPointer(util::DEVICE),
                                data_slice  ->scanned_edges    [peer_].GetPointer(util::DEVICE),
                                graph_slice ->nodes,//frontier_queues[peer_].keys[frontier_attribute[peer_].selector  ].GetSize(), 
                                graph_slice ->edges,//frontier_queues[peer_].keys[frontier_attribute[peer_].selector^1].GetSize(),
                                context          [peer_][0],
                                streams          [peer_],
                                gunrock::oprtr::advance::V2V, true);

                            if (SIZE_CHECK)
                            {
                                frontier_attribute_->output_length.Move(util::DEVICE, util::HOST,1,0,streams[peer_]);
                                cudaEventRecord(data_slice->events[iteration_][peer_][stages[peer_]], streams[peer_]);
                                data_slice->events_set[iteration_][peer_][stages[peer_]]=true;
                            }
                            break;

                        case 2: //SSSP Core
                            if (SIZE_CHECK)
                            {
                                if (!data_slice->events_set[iteration_][peer_][stages[peer_]-1])
                                {   to_show[peer_]=false;stages[peer_]--;break;}
                                tretval = cudaEventQuery(data_slice->events[iteration_][peer_][stages[peer_]-1]);
                                if (tretval == cudaErrorNotReady)
                                {   to_show[peer_]=false;stages[peer_]--; break;}
                                else if (tretval !=cudaSuccess) {enactor_stats_->retval=tretval; break;}

                                if (DEBUG) {printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                                        thread_num, iteration, peer_,
                                        frontier_queue_->keys[selector^1].GetSize(),
                                        frontier_attribute_->output_length[0]);fflush(stdout);}
                                //frontier_attribute_->output_length[0]+=1;
                                if (frontier_attribute_->output_length[0]+2 > frontier_queue_->keys[selector^1].GetSize())
                                {
                                    printf("%d\t %d\t %d\t queue3 oversize :\t %d ->\t %d\n",
                                        thread_num, iteration, peer_,
                                        frontier_queue_->keys[selector^1].GetSize(),
                                        frontier_attribute_->output_length[0]+2);fflush(stdout);
                                    if (enactor_stats_->retval = frontier_queue_->keys[selector  ].EnsureSize(frontier_attribute_->output_length[0]+2, true)) break;
                                    if (enactor_stats_->retval = frontier_queue_->keys[selector^1].EnsureSize(frontier_attribute_->output_length[0]+2)) break;

                                    if (SSSPProblem::USE_DOUBLE_BUFFER) {
                                        if (enactor_stats_->retval = frontier_queue_->values[selector  ].EnsureSize(frontier_attribute_->output_length[0]+2,true)) break;
                                        if (enactor_stats_->retval = frontier_queue_->values[selector^1].EnsureSize(frontier_attribute_->output_length[0]+2)) break;
                                    }
                                   //if (enactor_stats[peer_].retval = cudaDeviceSynchronize()) break;
                                }
                            }

                            SSSPCore < INSTRUMENT, AdvanceKernelPolicy, FilterKernelPolicy, SSSPProblem>(
                                (bool) DEBUG,
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
                            if (num_gpus <=1) {to_show[peer_]=false;break;}
                            /*to_wait = false;
                            for (int i=0;i<num_gpus;i++)
                                if (stages[i]<stages[peer_])
                                {
                                    to_wait=true;break;
                                }
                            if (to_wait)*/
                            {
                                if (!data_slice->events_set[iteration_][peer_][stages[peer_]-1])
                                {   to_show[peer_]=false;stages[peer_]--;break;}
                                tretval = cudaEventQuery(data_slice->events[iteration_][peer_][stages[peer_]-1]);
                                if (tretval == cudaErrorNotReady)
                                {   to_show[peer_]=false;stages[peer_]--;break;}
                                else if (tretval !=cudaSuccess) {enactor_stats_->retval=tretval; break;}
                            } //else cudaStreamSynchronize(streams[peer_]);
                            //data_slice->events_set[iteration_][peer_][stages[peer_]-1]=false;

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
                            if (frontier_attribute_->queue_length!=0)
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
                                    if (Total_Length + frontier_attribute_->queue_length > graph_slice->frontier_queues[num_gpus].keys[0].GetSize())
                                    {
                                        printf("%d\t %d\t %d\t total_queue oversize : %d -> %d \n",
                                           thread_num, iteration, peer_,
                                           Total_Length + frontier_attribute_->queue_length,
                                           graph_slice->frontier_queues[num_gpus].keys[0].GetSize());fflush(stdout);
                                        if (enactor_stats_->retval = graph_slice->frontier_queues[num_gpus].keys[0].EnsureSize(Total_Length+frontier_attribute_->queue_length, true)) break;
                                        if (SSSPProblem::USE_DOUBLE_BUFFER)
                                        {
                                            if (enactor_stats_->retval = graph_slice->frontier_queues[num_gpus].values[0].EnsureSize(Total_Length + frontier_attribute_->queue_length, true)) break;
                                        }
                                    }
                                }
                                util::MemsetCopyVectorKernel<<<256,256, 0, streams[peer_]>>>(
                                    graph_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE) + Total_Length,
                                    frontier_queue_->keys[selector].GetPointer(util::DEVICE),
                                    frontier_attribute_->queue_length);
                                if (SSSPProblem::USE_DOUBLE_BUFFER)
                                    util::MemsetCopyVectorKernel<<<256,256,0,streams[peer_]>>>(
                                        graph_slice->frontier_queues[num_gpus].values[0].GetPointer(util::DEVICE) + Total_Length,
                                        frontier_queue_->values[selector].GetPointer(util::DEVICE),
                                        frontier_attribute_->queue_length);
                                Total_Length+=frontier_attribute_->queue_length;
                            }
                            /*if (First_Stage4)
                            {
                                First_Stage4=false;
                                util::MemsetKernel<<<128, 128, 0, streams[peer_]>>>
                                    (data_slice->temp_marker.GetPointer(util::DEVICE),
                                    (unsigned int)0, graph_slice->nodes);
                            }*/
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

                        if (DEBUG)
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
                        //if (All_Done(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) break;
                    }
                    /*to_wait=true;
                    for (int i=0;i<num_gpus;i++)
                        if (to_show[i])
                        {
                            to_wait=false;
                            break;
                        }
                    if (to_wait) sleep(0);*/
                }

                if (num_gpus>1)
                {
                    /*if (First_Stage4)
                    {
                        util::MemsetKernel<<<128,128, 0, streams[0]>>>
                            (data_slice->temp_marker.GetPointer(util::DEVICE),
                            (unsigned int)0, graph_slice->nodes);
                    }*/
                    for (int peer_=0;peer_<num_gpus;peer_++)
                    for (int i=0;i<data_slice->num_stages;i++)
                        data_slice->events_set[(enactor_stats[0].iteration+3)%4][peer_][i]=false;

                    for (int peer_=0;peer_<num_gpus*2;peer_++)
                        data_slice->wait_marker[peer_]=0;
                    int wait_count=0;
                    while (wait_count<num_gpus*2-1 &&
                        !All_Done<SizeT, DataSlice>(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
                    {
                        for (int peer_=0;peer_<num_gpus*2;peer_++)
                        {
                            if (peer_==num_gpus || data_slice->wait_marker[peer_]!=0)
                                continue;
                            cudaError_t tretval = cudaStreamQuery(streams[peer_]);
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
                    /*if (SIZE_CHECK)
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
                    }*/
                    
                    frontier_attribute[0].queue_length = Total_Length;
                    if (Total_Length>0)
                    {
                        grid_size = Total_Length/256+1;
                        if (grid_size > 512) grid_size = 512;
                        
                        if (SSSPProblem::MARK_PATHS)
                        {
                            Copy_Preds<VertexId, SizeT> <<<grid_size,256,0, streams[0]>>>(
                                Total_Length,
                                graph_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE),
                                data_slice->preds.GetPointer(util::DEVICE),
                                data_slice->temp_preds.GetPointer(util::DEVICE));
                            
                            Update_Preds<VertexId,SizeT> <<<grid_size,256,0,streams[0]>>>(
                                Total_Length,
                                graph_slice->nodes,
                                graph_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE),
                                graph_slice->original_vertex.GetPointer(util::DEVICE),
                                data_slice->temp_preds.GetPointer(util::DEVICE),
                                data_slice->preds.GetPointer(util::DEVICE));//,
                        }

                        if (SIZE_CHECK && data_slice->keys_marker[0].GetSize() < Total_Length)
                        {    
                            printf("%d\t %lld\t \t keys_marker oversize : %d -> %d \n",
                                    thread_num, enactor_stats[0].iteration,
                                    data_slice->keys_marker[0].GetSize(), Total_Length);fflush(stdout);     
                            for (int peer_=0;peer_<num_gpus;peer_++)
                            {    
                                data_slice->keys_marker[peer_].EnsureSize(Total_Length);
                                data_slice->keys_markers[peer_]=data_slice->keys_marker[peer_].GetPointer(util::DEVICE);
                            }    
                            data_slice->keys_markers.Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);
                        }    
                        
                        Assign_Marker<VertexId, SizeT>
                            <<<grid_size,256, num_gpus * sizeof(SizeT*) ,streams[0]>>> (
                            Total_Length,
                            num_gpus,
                            graph_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE),
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

                        if (SIZE_CHECK)
                        {
                            for (int peer_=0; peer_<num_gpus;peer_++)
                            {
                                cudaMemcpyAsync(&(data_slice->out_length[peer_]),
                                    data_slice->keys_marker[peer_].GetPointer(util::DEVICE)
                                        + (Total_Length -1),
                                    sizeof(SizeT), cudaMemcpyDeviceToHost, streams[0]);
                            }
                            cudaStreamSynchronize(streams[0]);

                            for (int peer_=0; peer_<num_gpus;peer_++)
                            {
                                SizeT org_size = (peer_==0? graph_slice->frontier_queues[0].keys[frontier_attribute[0].selector^1].GetSize() : data_slice->keys_out[peer_].GetSize());
                                if (data_slice->out_length[peer_] > org_size)
                                {
                                    printf("%d\t %lld\t %d\t keys_out oversize : %d -> %d\n",
                                           thread_num, enactor_stats[0].iteration, peer_,
                                           org_size, data_slice->out_length[peer_]);fflush(stdout);
                                    if (peer_==0)
                                    {
                                        graph_slice->frontier_queues[0].keys[frontier_attribute[0].selector^1].EnsureSize(data_slice->out_length[0]);
                                    } else {
                                        data_slice -> keys_out[peer_].EnsureSize(data_slice->out_length[peer_]);
                                        for (int i=0;i<num_vertex_associate;i++)
                                        {
                                            data_slice->vertex_associate_out [peer_][i].EnsureSize(data_slice->out_length[peer_]);
                                            data_slice->vertex_associate_outs[peer_][i] =
                                            data_slice->vertex_associate_out[peer_][i].GetPointer(util::DEVICE);
                                        }
                                        data_slice->vertex_associate_outs[peer_].Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);
                                        for (int i=0;i<num_value__associate;i++)
                                        {
                                            data_slice->value__associate_out [peer_][i].EnsureSize(data_slice->out_length[peer_]);
                                            data_slice->value__associate_outs[peer_][i] =
                                                data_slice->value__associate_out[peer_][i].GetPointer(util::DEVICE);
                                        }
                                        data_slice->value__associate_outs[peer_].Move(util::HOST, util::DEVICE, num_gpus, 0, streams[0]);
                                    }
                                }
                            }
                        }
 
                        for (int peer_=0;peer_<num_gpus;peer_++)
                            if (peer_==0) data_slice -> keys_outs[peer_] = graph_slice->frontier_queues[peer_].keys[frontier_attribute[0].selector^1].GetPointer(util::DEVICE);
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
                                  sizeof(VertexId*) * num_vertex_associate);
                        offset += sizeof(VertexId*) * num_vertex_associate ;
                        memcpy(&(data_slice -> make_out_array[offset]),
                                 data_slice -> value__associate_orgs.GetPointer(util::HOST),
                                  sizeof(Value*   ) * num_value__associate);
                        offset += sizeof(Value*   ) * num_value__associate ;
                        for (int peer_=0; peer_<num_gpus; peer_++)
                        {
                            memcpy(&(data_slice->make_out_array[offset]),
                                     data_slice->vertex_associate_outs[peer_].GetPointer(util::HOST),
                                      sizeof(VertexId*) * num_vertex_associate);
                            offset += sizeof(VertexId*) * num_vertex_associate ;
                        }
                        for (int peer_=0; peer_<num_gpus; peer_++)
                        {
                            memcpy(&(data_slice->make_out_array[offset]),
                                    data_slice->value__associate_outs[peer_].GetPointer(util::HOST),
                                      sizeof(Value*   ) * num_value__associate);
                            offset += sizeof(Value*   ) * num_value__associate ;
                        }
                        data_slice->make_out_array.Move(util::HOST, util::DEVICE, data_slice->make_out_array.GetSize(), 0, streams[0]);

                        //grid_size = Total_Length / 256 +1;
                        Make_Out<VertexId, SizeT, Value, SSSPProblem::MARK_PATHS?1:0, 1>
                            <<<grid_size, 256, sizeof(char)*data_slice->make_out_array.GetSize(), streams[0]>>> (
                            Total_Length,
                            num_gpus,
                            graph_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE),
                            graph_slice-> partition_table        .GetPointer(util::DEVICE),
                            graph_slice-> convertion_table       .GetPointer(util::DEVICE),
                            data_slice -> make_out_array         .GetSize(),
                            data_slice -> make_out_array         .GetPointer(util::DEVICE));
                            /*data_slice -> keys_markers           .GetPointer(util::DEVICE),
                            data_slice -> vertex_associate_orgs  .GetPointer(util::DEVICE),
                            data_slice -> value__associate_orgs  .GetPointer(util::DEVICE),
                            data_slice -> keys_outs              .GetPointer(util::DEVICE),
                            data_slice -> vertex_associate_outss .GetPointer(util::DEVICE),
                            data_slice -> value__associate_outss .GetPointer(util::DEVICE));
                            */
                        //if (enactor_stats[0].retval = util::GRError(cudaStreamSynchronize(streams[0]), "Make_Out error", __FILE__, __LINE__)) break;
                        if (!SIZE_CHECK)
                        {
                            for (int peer_=0;peer_<num_gpus;peer_++)
                                cudaMemcpyAsync(&(data_slice->out_length[peer_]),
                                    data_slice->keys_marker[peer_].GetPointer(util::DEVICE)
                                        + (Total_Length -1),
                                    sizeof(SizeT), cudaMemcpyDeviceToHost, streams[0]);
                        }

                       cudaStreamSynchronize(streams[0]);
                       frontier_attribute[0].selector^=1;
                        //if (enactor_stats[0].retval = util::GRError(cudaStreamSynchronize(streams[0]), "MemcpyAsync keys_marker error", __FILE__, __LINE__)) break;
                    } else {
                        for (int peer_=0;peer_<num_gpus;peer_++)
                            data_slice->out_length[peer_]=0;
                    }
                    for (int peer_=0;peer_<num_gpus;peer_++)
                        frontier_attribute[peer_].queue_length = data_slice->out_length[peer_];

                } else {
                    if (enactor_stats[0].retval = work_progress[0].GetQueueLength(frontier_attribute[0].queue_index, frontier_attribute[0].queue_length, false, data_slice->streams[0])) break;
                }

                //util::cpu_mt::PrintMessage("Iteration end",thread_num,enactor_stats->iteration);
                enactor_stats->iteration++;
                //if (DEBUG) printf("\n%lld", (long long) enactor_stats->iteration);
            }
            
            if (All_Done<SizeT, DataSlice>(s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)) break;
        } while(0);

        /*if (num_gpus >1)
        {
            if (break_clean)
            {
                util::cpu_mt::ReleaseBarrier(cpu_barrier);
            }
            delete Scaner; Scaner=NULL;
        }
        delete[] message;message=NULL;
        */

        thread_data->stats=4;
        CUT_THREADEND;
    }

/**
 * @brief SSSP problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<typename SSSPProblem, bool INSTRUMENT>
class SSSPEnactor : public EnactorBase<typename SSSPProblem::SizeT>
{
    typedef typename SSSPProblem::SizeT    SizeT   ;
    typedef typename SSSPProblem::VertexId VertexId;
    typedef typename SSSPProblem::Value    Value   ;

    SSSPProblem *problem      ;
    ThreadSlice *thread_slices;
    CUTThread   *thread_Ids   ;

   // Methods
public:

    /**
     * @brief BFSEnactor constructor
     */
    SSSPEnactor(bool DEBUG = false, int num_gpus = 1, int* gpu_idx = NULL) :
        EnactorBase<SizeT>(VERTEX_FRONTIERS, DEBUG, num_gpus, gpu_idx)//,
    {
        //util::cpu_mt::PrintMessage("SSSPEnactor() begin.");
        thread_slices = NULL;
        thread_Ids    = NULL;
        problem       = NULL;
        //util::cpu_mt::PrintMessage("SSSPEnactor() end.");
    }

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~SSSPEnactor()
    {
        //util::cpu_mt::PrintMessage("~SSSPEnactor() begin.");
        //for (int gpu=0;gpu<this->num_gpus;gpu++)
        //{
        //    util::SetDevice(this->gpu_idx[gpu]);
        //    if (BFSProblem::ENABLE_IDEMPOTENCE)
        //    {
        //        cudaUnbindTexture(gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref);
        //    }
        //}

        cutWaitForThreads(thread_Ids, this->num_gpus);
        delete[] thread_Ids   ; thread_Ids    = NULL;
        delete[] thread_slices; thread_slices = NULL;
        problem = NULL;
        //util::cpu_mt::PrintMessage("~SSSPEnactor() end.");
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last SSSP search enacted.
     *
     * @param[out] total_queued Total queued elements in SSSP kernel running.
     * @param[out] search_depth Search depth of SSSP algorithm.
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
    cudaError_t InitSSSP(
        ContextPtr  *context,
        SSSPProblem *problem,
        int         max_grid_size = 0,
        bool        size_check    = true)
    {
        cudaError_t retval = cudaSuccess;

        // Lazy initialization
        if (retval = EnactorBase<SizeT>::Init(problem,
                                       max_grid_size,
                                       AdvanceKernelPolicy::CTA_OCCUPANCY,
                                       FilterKernelPolicy::CTA_OCCUPANCY)) return retval;

        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus];

        /*for (int gpu=0;gpu<this->num_gpus;gpu++)
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
        }*/

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            thread_slices[gpu].thread_num    = gpu;
            thread_slices[gpu].problem       = (void*)problem;
            thread_slices[gpu].enactor       = (void*)this;
            //thread_slices[gpu].cpu_barrier   = &cpu_barrier;
            thread_slices[gpu].context       = &(context[gpu*this->num_gpus]);
            thread_slices[gpu].stats         = -1;
            //this->enactor_stats[gpu].start_time    = start_time;
            if (size_check)
                thread_slices[gpu].thread_Id = cutStartThread(
                    (CUT_THREADROUTINE)&(SSSPThread<INSTRUMENT, true, AdvanceKernelPolicy,FilterKernelPolicy,SSSPProblem>),
                    (void*)&(thread_slices[gpu]));
            else thread_slices[gpu].thread_Id = cutStartThread(
                    (CUT_THREADROUTINE)&(SSSPThread<INSTRUMENT, false, AdvanceKernelPolicy,FilterKernelPolicy,SSSPProblem>),
                    (void*)&(thread_slices[gpu]));
            thread_Ids[gpu] = thread_slices[gpu].thread_Id;
        }

       return retval;
    }

    cudaError_t Reset()
    {
        return EnactorBase<SizeT>::Reset();
    }
 
    /** @} */

    /**
     * @brief SSSP Enact kernel entry.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam FilterPolicy Kernel policy for vertex mapping.
     * @tparam SSSPProblem SSSP Problem type.
     *
     * @param[in] problem SSSPProblem object.
     * @param[in] src Source node for SSSP.
     * @param[in] max_grid_size Max grid size for SSSP kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactSSSP(
        VertexId       src)
    {
        clock_t      start_time = clock();
        cudaError_t  retval     = cudaSuccess;

        do {
            for (int gpu=0;gpu<this->num_gpus;gpu++)
            {
                if ((this->num_gpus ==1) || (gpu==this->problem->partition_tables[0][src]))
                     thread_slices[gpu].init_size=1;
                else thread_slices[gpu].init_size=0;
                this->frontier_attribute[gpu*this->num_gpus].queue_length = thread_slices[gpu].init_size;
            }

            for (int gpu=0; gpu< this->num_gpus; gpu++)
            {
                while (thread_slices[gpu].stats!=1) sleep(0);
                thread_slices[gpu].stats=2;
            }
            for (int gpu=0; gpu< this->num_gpus; gpu++)
            {
                while (thread_slices[gpu].stats!=4) sleep(0);
            }

            for (int gpu=0;gpu<this->num_gpus;gpu++)
            if (this->enactor_stats[gpu].retval!=cudaSuccess) {retval=this->enactor_stats[gpu].retval;break;}
        } while(0);

        if (this->DEBUG) printf("\nGPU SSSP Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief SSSP Enact kernel entry.
     *
     * @tparam SSSPProblem SSSP Problem type. @see SSSPProblem
     *
     * @param[in] problem Pointer to SSSPProblem object.
     * @param[in] src Source node for SSSP.
     * @param[in] max_grid_size Max grid size for SSSP kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    //template <typename SSSPProblem>
    cudaError_t Enact(
        //ContextPtr   *context,
        //SSSPProblem  *problem,
        VertexId     src)
        //double       queue_sizing,
        //int          max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
            SSSPProblem,                        // Problem data type
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
                SSSPProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                10,                                  // LOG_THREADS
                8,                                  // LOG_BLOCKS
                32*128,                             // LIGHT_EDGE_THRESHOLD    
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                            // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::LB>
                    AdvanceKernelPolicy;

            return EnactSSSP<AdvanceKernelPolicy, FilterKernelPolicy>(
                    //context, problem, src, queue_sizing, max_grid_size);
                   src);
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
     * @brief SSSP Enact kernel entry.
     *
     * @tparam SSSPProblem SSSP Problem type. @see SSSPProblem
     *
     * @param[in] problem Pointer to SSSPProblem object.
     * @param[in] src Source node for SSSP.
     * @param[in] max_grid_size Max grid size for SSSP kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    //template <typename SSSPProblem>
    cudaError_t Init(
        ContextPtr   *context,
        SSSPProblem  *problem,
        //VertexId     src)
        //double       queue_sizing,
        int          max_grid_size = 0,
        bool         size_check = true)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
            SSSPProblem,                        // Problem data type
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
                SSSPProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                10,                                  // LOG_THREADS
                8,                                  // LOG_BLOCKS
                32*128,                             // LIGHT_EDGE_THRESHOLD    
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                            // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::LB>
                    AdvanceKernelPolicy;

            return InitSSSP<AdvanceKernelPolicy, FilterKernelPolicy>(
                    context, problem, max_grid_size, size_check);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }


    /** @} */

};

} // namespace sssp
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
