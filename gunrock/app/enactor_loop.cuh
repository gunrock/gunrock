// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * enactor_loop.cuh
 *
 * @brief Base Iteration Loop
 */

#pragma once

#include <gunrock/app/enactor_kernel.cuh>
#include <gunrock/app/enactor_helper.cuh>
#include <gunrock/util/latency_utils.cuh>
//#include <gunrock/util/test_utils.h>
#include <moderngpu.cuh>

using namespace mgpu;

/* this is the "stringize macro macro" hack */
#define STR(x) #x
#define XSTR(x) STR(x)

namespace gunrock {
namespace app {

/*
 * @brief Iteration loop.
 *
 * @tparam Enactor
 * @tparam Functor
 * @tparam Iteration
 * @tparam NUM_VERTEX_ASSOCIATES
 * @tparam NUM_VALUE__ASSOCIATES
 *
 * @param[in] thread_data
 */
template <
    typename Enactor,
    typename Functor,
    typename Iteration,
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES>
void Iteration_Loop(
    ThreadSlice *thread_data)
{
    //typedef typename Iteration::Enactor   Enactor   ;
    typedef typename Enactor::Problem     Problem   ;
    typedef typename Problem::SizeT       SizeT     ;
    typedef typename Problem::VertexId    VertexId  ;
    typedef typename Problem::Value       Value     ;
    typedef typename Problem::DataSlice   DataSlice ;
    typedef GraphSlice<VertexId, SizeT, Value>  GraphSliceT;
    typedef util::DoubleBuffer<VertexId, SizeT, Value> Frontier;

    Problem      *problem              =  (Problem*) thread_data->problem;
    Enactor      *enactor              =  (Enactor*) thread_data->enactor;
    int           num_gpus             =   problem     -> num_gpus;
    int           thread_num           =   thread_data -> thread_num;
    DataSlice    *data_slice           =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    util::Array1D<SizeT, DataSlice>
                 *s_data_slice         =   problem     -> data_slices;
    GraphSliceT  *graph_slice          =   problem     -> graph_slices       [thread_num] ;
    GraphSliceT  **s_graph_slice       =   problem     -> graph_slices;
    FrontierAttribute<SizeT>
                 *frontier_attribute   = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    FrontierAttribute<SizeT>
                 *s_frontier_attribute = &(enactor     -> frontier_attribute [0         ]);
    EnactorStats<SizeT>
                 *enactor_stats        = &(enactor     -> enactor_stats      [thread_num * num_gpus]);
    EnactorStats<SizeT>
                 *s_enactor_stats      = &(enactor     -> enactor_stats      [0         ]);
    util::CtaWorkProgressLifetime<SizeT>
                 *work_progress        = &(enactor     -> work_progress      [thread_num * num_gpus]);
    ContextPtr   *context              =   thread_data -> context;
    int          *stages               =   data_slice  -> stages .GetPointer(util::HOST);
    bool         *to_show              =   data_slice  -> to_show.GetPointer(util::HOST);
    cudaStream_t *streams              =   data_slice  -> streams.GetPointer(util::HOST);
    SizeT         Total_Length         =   0;
    SizeT         received_length      =   0;
    cudaError_t   tretval              =   cudaSuccess;
    int           grid_size            =   0;
    std::string   mssg                 =   "";
    int           pre_stage            =   0;
    size_t        offset               =   0;
    int           iteration            =   0;
    int           selector             =   0;
    Frontier     *frontier_queue_      =   NULL;
    FrontierAttribute<SizeT>
                 *frontier_attribute_  =   NULL;
    EnactorStats<SizeT>
                 *enactor_stats_       =   NULL;
    util::CtaWorkProgressLifetime<SizeT>
                 *work_progress_       =   NULL;
    util::Array1D<SizeT, SizeT>
                 *scanned_edges_       =   NULL;
    int           peer, peer_, peer__, gpu_, i, iteration_, wait_count;
    bool          over_sized;
    SizeT         communicate_latency  =   enactor -> communicate_latency;
    float         communicate_multipy  =   enactor -> communicate_multipy;
    SizeT         expand_latency       =   enactor -> expand_latency;
    SizeT         subqueue_latency     =   enactor -> subqueue_latency;
    SizeT         fullqueue_latency    =   enactor -> fullqueue_latency;
    SizeT         makeout_latency      =   enactor -> makeout_latency;

#ifdef ENABLE_PERFORMANCE_PROFILING
    util::CpuTimer      cpu_timer;
    std::vector<double> &iter_full_queue_time =
        enactor -> iter_full_queue_time[thread_num].back();
    std::vector<double> &iter_sub_queue_time =
        enactor -> iter_sub_queue_time [thread_num].back();
    std::vector<double> &iter_total_time =
        enactor -> iter_total_time     [thread_num].back();
    std::vector<SizeT>  &iter_full_queue_nodes_queued =
        enactor -> iter_full_queue_nodes_queued[thread_num].back();
    std::vector<SizeT>  &iter_full_queue_edges_queued =
        enactor -> iter_full_queue_edges_queued[thread_num].back();

    cpu_timer.Start();
    double iter_start_time = cpu_timer.MillisSinceStart();
    double iter_stop_time = 0;
    double subqueue_finish_time = 0;

    SizeT  h_edges_queued[16];
    SizeT  h_nodes_queued[16];
    SizeT  previous_edges_queued[16];
    SizeT  previous_nodes_queued[16];
    SizeT  h_full_queue_edges_queued = 0;
    SizeT  h_full_queue_nodes_queued = 0;
    SizeT  previous_full_queue_edges_queued = 0;
    SizeT  previous_full_queue_nodes_queued = 0;

    for (int peer_ = 0; peer_ < num_gpus; peer_++)
    {
        h_edges_queued       [peer_] = 0;
        h_nodes_queued       [peer_] = 0;
        previous_nodes_queued[peer_] = 0;
        previous_edges_queued[peer_] = 0;
    }
#endif

    if (enactor -> debug)
    {
        printf("Iteration entered\n");fflush(stdout);
    }
    while (!Iteration::Stop_Condition(
        s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
    {
        Total_Length             = 0;
        received_length          = frontier_attribute[0].queue_length;
        data_slice->wait_counter = 0;
        tretval                  = cudaSuccess;
        if (num_gpus>1 && enactor_stats[0].iteration != 0)
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
        if (num_gpus > 1)
        {
            if (enactor -> problem -> unified_receive)
            {
                //printf("%d, %d : start_received_length = %d\n",
                //    thread_num, enactor_stats[0].iteration, received_length);
                data_slice -> in_length_out[0] = received_length;
                data_slice -> in_length_out.Move(util::HOST, util::DEVICE, 1, 0, streams[0]);
                if (enactor_stats[0].retval = util::GRError(cudaStreamSynchronize(streams[0]),
                    "cudaStreamSynchronize failed", __FILE__, __LINE__))
                break;
            }
        } else data_slice -> in_length_out[0] = received_length;
        for (peer=0; peer<num_gpus; peer++)
        {
            stages [peer         ] = 0   ;
            stages [peer+num_gpus] = 0   ;
            to_show[peer         ] = true;
            to_show[peer+num_gpus] = true;
            for (i=0; i<data_slice->num_stages; i++)
                data_slice->events_set[enactor_stats[0].iteration%4][peer][i]=false;
        }
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("labels", data_slice -> labels.GetPointer(util::DEVICE), graph_slice -> nodes, thread_num, iteration, -1, streams[0]);

        while (data_slice->wait_counter < num_gpus*2
           && (!Iteration::Stop_Condition(
            s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus)))
        {
            //util::cpu_mt::PrintCPUArray<int, int>("stages", stages, num_gpus * 2, thread_num, iteration);
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
                scanned_edges_      = &(data_slice->scanned_edges  [peer_]);
                frontier_attribute_ = &(frontier_attribute         [peer_]);
                enactor_stats_      = &(enactor_stats              [peer_]);
                work_progress_      = &(work_progress              [peer_]);

                if (enactor -> debug && to_show[peer__])
                {
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
                    if (peer__ == num_gpus)
                    { // out-going to local, not in use
                        stages[peer__] = 4; break;
                    }

                    if (peer__ == 0)
                    {
                        if (frontier_attribute_ -> queue_length == 0)
                        { // empty local queue
                            //Set_Record(data_slice, iteration, peer__, 3, streams[peer__]);
                            stages[peer__] = 4; break;
                        } else {
                            if (Iteration::HAS_SUBQ)
                            {
                                stages[peer__] = 1; break;
                            } else {
                                Set_Record(data_slice, iteration, peer__, 2, streams[peer__]);
                                stages[peer__] = 3; break;
                            }
                        }
                    }

                    if (iteration==0 && peer__>num_gpus)
                    {  // first iteration, nothing to send
                        Set_Record(data_slice, iteration, peer__, 0, streams[peer__]);
                        stages[peer__] = 4; break;
                    }

                    if (peer__<num_gpus)
                    { //wait and expand incoming
                        if (!(s_data_slice[peer]->events_set[iteration_][gpu_ + num_gpus][0]))
                        {   to_show[peer__]=false;break;}

                        frontier_attribute_->queue_length =
                            data_slice->in_length[iteration%2][peer_];
#ifdef ENABLE_PERFORMANCE_PROFILING
                        enactor_stats_ -> iter_in_length.back().push_back(
                            data_slice -> in_length[iteration%2][peer_]);
#endif
                        if (frontier_attribute_ -> queue_length != 0)
                        {
                            if (enactor_stats_ -> retval = util::GRError(
                                cudaStreamWaitEvent(streams[peer_],
                                s_data_slice[peer]->events[iteration_][gpu_+num_gpus][0], 0), "cudaStreamWaitEvent failed", __FILE__, __LINE__))
                                break;
                        }
                        data_slice->in_length[iteration%2][peer_]=0;
                        s_data_slice[peer]->events_set[iteration_][gpu_ + num_gpus][0]=false;

                        if (frontier_attribute_ -> queue_length == 0)
                        {
                            //Set_Record(data_slice, iteration, peer__, 3, streams[peer__]);
                            //printf(" %d\t %d\t %d\t Expand and subQ skipped\n", thread_num, iteration, peer__);
                            stages[peer__] = 4; break;
                        }

                        if (expand_latency != 0)
                            util::latency::Insert_Latency(
                            expand_latency,
                            frontier_attribute_ -> queue_length,
                            streams[peer_],
                            data_slice -> latency_data.GetPointer(util::DEVICE));

                        Iteration::template Expand_Incoming
                            <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                            enactor, streams[peer_],
                            data_slice -> in_iteration       [iteration%2][peer_],
                            peer_,
                            //data_slice -> in_length          [iteration%2],
                            received_length,
                            frontier_attribute_ -> queue_length,
                            data_slice -> in_length_out,
                            data_slice -> keys_in            [iteration%2][peer_],
                            data_slice -> vertex_associate_in[iteration%2][peer_],
                            data_slice -> value__associate_in[iteration%2][peer_],
                            (enactor -> problem -> unified_receive) ?
                            data_slice -> frontier_queues[0].keys[frontier_attribute[0].selector] : frontier_queue_ -> keys[selector^1],
                            data_slice -> vertex_associate_orgs,
                            data_slice -> value__associate_orgs,
                            data_slice,
                            enactor_stats_);
                        //printf("%d, Expand, selector = %d, keys = %p\n",
                        //    thread_num, selector^1, frontier_queue_ -> keys[selector^1].GetPointer(util::DEVICE));

                        frontier_attribute_->selector^=1;
                        frontier_attribute_->queue_index++;
                        if (!Iteration::HAS_SUBQ) {
                            if (enactor -> problem -> unified_receive)
                            {
                                //Set_Record(data_slice, iteration, peer__, 3, streams[peer__]);
                                stages[peer__] = 4;
                            } else {
                                Set_Record(data_slice, iteration, peer__, 2, streams[peer__]);
                                stages[peer__] = 3;
                            }
                        } else {
                            Set_Record(data_slice, iteration, peer__, stages[peer__], streams[peer__]);
                            stages[peer__] = 1;
                        }
                    } else { //Push Neighbor
                        if (communicate_latency != 0)
                            util::latency::Insert_Latency(
                            communicate_latency,
                            data_slice -> out_length[peer_],
                            streams[peer__],
                            data_slice -> latency_data.GetPointer(util::DEVICE));

                        PushNeighbor <Enactor, GraphSliceT, DataSlice,
                                NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                            enactor,
                            thread_num,
                            peer,
                            data_slice->out_length[peer_],
                            enactor_stats_,
                            s_data_slice  [thread_num].GetPointer(util::HOST),
                            s_data_slice  [peer]      .GetPointer(util::HOST),
                            s_graph_slice [thread_num],
                            s_graph_slice [peer],
                            streams       [peer__],
                            communicate_multipy);
                        Set_Record(data_slice, iteration, peer__, stages[peer__], streams[peer__]);
                        stages[peer__] = 4;
                    }
                    break;

                case 1: //Comp Length
                    if (peer_ != 0)
                    {
                        if (enactor_stats_-> retval = Check_Record(
                            data_slice, iteration, peer_,
                            stages[peer_]-1, stages[peer_], to_show[peer_])) break;
                        if (to_show[peer_] == false) break;
                        frontier_attribute_ -> queue_length = data_slice -> in_length_out[peer_];
                    }
                    if (enactor_stats_->retval = Iteration::Compute_OutputLength(
                        enactor,
                        frontier_attribute_,
                        //data_slice,
                        //s_data_slice[thread_num].GetPointer(util::DEVICE),
                        graph_slice    ->row_offsets     .GetPointer(util::DEVICE),
                        graph_slice    ->column_indices  .GetPointer(util::DEVICE),
                        graph_slice    ->column_offsets  .GetPointer(util::DEVICE),
                        graph_slice    ->row_indices     .GetPointer(util::DEVICE),
                        frontier_queue_->keys[selector]  .GetPointer(util::DEVICE),
                        scanned_edges_,
                        graph_slice    ->nodes,
                        graph_slice    ->edges,
                        context          [peer_][0],
                        streams          [peer_],
                        gunrock::oprtr::advance::V2V, true, false, false)) break;

                    if (enactor -> size_check ||
                        (Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            != oprtr::advance::TWC_FORWARD &&
                         Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            != oprtr::advance::TWC_BACKWARD))
                    {
                        Set_Record(data_slice, iteration, peer_, stages[peer_], streams[peer_]);
                    }
                    stages[peer__] = 2;
                    break;

                case 2: //SubQueue Core
                    if (enactor -> size_check ||
                        (Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            != oprtr::advance::TWC_FORWARD &&
                         Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            != oprtr::advance::TWC_BACKWARD))
                    {
                        if (enactor_stats_ -> retval = Check_Record (
                            data_slice, iteration, peer_,
                            stages[peer_]-1, stages[peer_], to_show[peer_])) break;
                        if (to_show[peer_]==false) break;
                        /*if (Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            == oprtr::advance::TWC_FORWARD ||
                            Iteration::AdvanceKernelPolicy::ADVANCE_MODE
                            == oprtr::advance::TWC_BACKWARD)
                        {
                            frontier_attribute_->output_length[0] *= 1.1;
                        }*/

                        if (enactor -> size_check)
                        Iteration::Check_Queue_Size(
                            enactor,
                            thread_num,
                            peer_,
                            frontier_attribute_->output_length[0] + 2,
                            frontier_queue_,
                            frontier_attribute_,
                            enactor_stats_,
                            graph_slice);
                        if (enactor_stats_ -> retval) break;
                    }
                    if (subqueue_latency != 0)
                        util::latency::Insert_Latency(
                        subqueue_latency,
                        frontier_attribute_ -> queue_length,
                        streams[peer_],
                        data_slice -> latency_data.GetPointer(util::DEVICE));

                    Iteration::SubQueue_Core(
                        enactor,
                        thread_num,
                        peer_,
                        frontier_queue_,
                        scanned_edges_,
                        frontier_attribute_,
                        enactor_stats_,
                        data_slice,
                        s_data_slice[thread_num].GetPointer(util::DEVICE),
                        graph_slice,
                        &(work_progress[peer_]),
                        context[peer_],
                        streams[peer_]);
#ifdef ENABLE_PERFORMANCE_PROFILING
                    h_nodes_queued[peer_] = enactor_stats_ -> nodes_queued[0];
                    h_edges_queued[peer_] = enactor_stats_ -> edges_queued[0];
                    enactor_stats_ -> nodes_queued.Move(
                        util::DEVICE, util::HOST, 1, 0, streams[peer_]);
                    enactor_stats_ -> edges_queued.Move(
                        util::DEVICE, util::HOST, 1, 0, streams[peer_]);
#endif

                    if (num_gpus>1)
                    {
                        Set_Record(data_slice, iteration, peer__, stages[peer__], streams[peer__]);
                        stages [peer__] = 3;
                    } else {
                        //Set_Record(data_slice, iteration, peer__, 3, streams[peer__]);
                        stages [peer__] = 4;
                    }
                    break;

                case 3: //Copy
                    //if (Iteration::HAS_SUBQ || peer_ != 0)
                    {
                        if (enactor_stats_-> retval = Check_Record(
                            data_slice, iteration, peer_,
                            stages[peer_]-1, stages[peer_], to_show[peer_])) break;
                        if (to_show[peer_] == false) break;
                    }

                    //printf("size_check = %s\n", enactor -> size_check ? "true" : "false");fflush(stdout);
                    if (!Iteration::HAS_SUBQ && peer_ > 0)
                    {
                        frontier_attribute_ -> queue_length = data_slice -> in_length_out[peer_];
                    }
                    if (!enactor -> size_check && (enactor -> debug || num_gpus > 1))
                    {
                        if (Iteration::HAS_SUBQ)
                        {
                            if (enactor_stats_->retval =
                                Check_Size</*false,*/ SizeT, VertexId> (
                                    false, "queue3",
                                    frontier_attribute_->output_length[0]+2,
                                    &frontier_queue_->keys  [selector^1],
                                    over_sized, thread_num, iteration, peer_, false))
                                break;
                        }
                        if (frontier_attribute_->queue_length ==0) break;

                        if (enactor_stats_->retval =
                            Check_Size</*false,*/ SizeT, VertexId> (false, "total_queue",
                                Total_Length + frontier_attribute_->queue_length,
                                &data_slice->frontier_queues[num_gpus].keys[0],
                                over_sized, thread_num, iteration, peer_, false))
                            break;

                        util::MemsetCopyVectorKernel<<<256,256, 0, streams[peer_]>>>(
                            data_slice->frontier_queues[num_gpus].keys[0].GetPointer(util::DEVICE)
                                + Total_Length,
                            frontier_queue_->keys[selector].GetPointer(util::DEVICE),
                            frontier_attribute_->queue_length);
                        if (problem -> use_double_buffer)
                            util::MemsetCopyVectorKernel<<<256,256,0,streams[peer_]>>>(
                                data_slice->frontier_queues[num_gpus].values[0]
                                    .GetPointer(util::DEVICE) + Total_Length,
                                frontier_queue_->values[selector].GetPointer(util::DEVICE),
                                frontier_attribute_->queue_length);
                    }

                    Total_Length += frontier_attribute_->queue_length;
                    //Set_Record(data_slice, iteration, peer__, 3, streams[peer__]);
                    stages [peer__] = 4;
                    break;

                case 4: //End
                    data_slice->wait_counter++;
                    to_show[peer__]=false;
                    stages[peer__] = 5;
                    break;
                default:
                    //stages[peer__]--;
                    to_show[peer__]=false;
                }

                if (enactor -> debug && !enactor_stats_->retval)
                {
                    mssg="stage 0 @ gpu 0, peer_ 0 failed";
                    mssg[6]=char(pre_stage+'0');
                    mssg[14]=char(thread_num+'0');
                    mssg[23]=char(peer__+'0');
                    enactor_stats_->retval = util::GRError(mssg, __FILE__, __LINE__);
                    if (enactor_stats_ -> retval) break;
                }
                //stages[peer__]++;
                if (enactor_stats_->retval) break;
            }
        }

        if (!Iteration::Stop_Condition(
            s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
        {
            for (peer_ = 0; peer_ < num_gpus; peer_++)
                data_slice->wait_marker[peer_]=0;
            wait_count=0;
            while (wait_count < num_gpus &&
                !Iteration::Stop_Condition(
                s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
            {
                for (peer_=0; peer_<num_gpus; peer_++)
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

            if (enactor -> problem -> unified_receive)
            {
                Total_Length = data_slice -> in_length_out[0];
            } else if (num_gpus == 1)
                Total_Length = frontier_attribute[0].queue_length;
#ifdef ENABLE_PERFORMANCE_PROFILING
            subqueue_finish_time = cpu_timer.MillisSinceStart();
            iter_sub_queue_time.push_back(subqueue_finish_time - iter_start_time);
            if (Iteration::HAS_SUBQ)
            for (peer_ = 0; peer_ < num_gpus; peer_ ++)
            {
                enactor_stats[peer_].iter_nodes_queued.back().push_back(
                    h_nodes_queued[peer_] + enactor_stats[peer_].nodes_queued[0]
                    - previous_nodes_queued[peer_]);
                previous_nodes_queued[peer_] = h_nodes_queued[peer_] + enactor_stats[peer_].nodes_queued[0];
                enactor_stats[peer_].nodes_queued[0] = h_nodes_queued[peer_];

                enactor_stats[peer_].iter_edges_queued.back().push_back(
                    h_edges_queued[peer_] + enactor_stats[peer_].edges_queued[0]
                    - previous_edges_queued[peer_]);
                previous_edges_queued[peer_] = h_edges_queued[peer_] + enactor_stats[peer_].edges_queued[0];
                enactor_stats[peer_].edges_queued[0] = h_edges_queued[peer_];
            }
#endif
            if (enactor -> debug)
            {
                printf("%d\t %lld\t \t Subqueue finished. Total_Length= %lld, labels = %p\n",
                    thread_num, enactor_stats[0].iteration, (long long)Total_Length, data_slice -> labels.GetPointer(util::DEVICE));
                fflush(stdout);
            }

            grid_size = Total_Length/256+1;
            if (grid_size > 512) grid_size = 512;

            if (enactor -> size_check && !enactor -> problem -> unified_receive)
            {
                if (enactor_stats[0]. retval =
                    Check_Size</*true,*/ SizeT, VertexId> (
                        true, "total_queue", Total_Length,
                        &data_slice->frontier_queues[0].keys[frontier_attribute[0].selector],
                        over_sized, thread_num, iteration, num_gpus, true))
                    break;
                if (problem -> use_double_buffer)
                    if (enactor_stats[0].retval =
                        Check_Size</*true,*/ SizeT, Value> (
                            true, "total_queue", Total_Length,
                            &data_slice->frontier_queues[0].values[frontier_attribute[0].selector],
                            over_sized, thread_num, iteration, num_gpus, true))
                        break;

                offset = frontier_attribute[0].queue_length;
                for (peer_ = 1; peer_ < num_gpus; peer_++)
                if (frontier_attribute[peer_].queue_length !=0)
                {
                    util::MemsetCopyVectorKernel<<<256,256, 0, streams[0]>>>(
                        data_slice->frontier_queues[0    ]
                            .keys[frontier_attribute[0    ].selector]
                            .GetPointer(util::DEVICE) + offset,
                        data_slice->frontier_queues[peer_]
                            .keys[frontier_attribute[peer_].selector]
                            .GetPointer(util::DEVICE),
                        frontier_attribute[peer_].queue_length);

                    if (problem -> use_double_buffer)
                        util::MemsetCopyVectorKernel<<<256,256,0,streams[0]>>>(
                            data_slice->frontier_queues[0    ]
                                .values[frontier_attribute[0    ].selector]
                                .GetPointer(util::DEVICE) + offset,
                            data_slice->frontier_queues[peer_]
                                .values[frontier_attribute[peer_].selector]
                                .GetPointer(util::DEVICE),
                            frontier_attribute[peer_].queue_length);
                    offset+=frontier_attribute[peer_].queue_length;
                }
            }
            frontier_attribute[0].queue_length = Total_Length;
            if (! enactor -> size_check) frontier_attribute[0].selector = 0;
            frontier_queue_ = &(data_slice->frontier_queues
                [(enactor -> size_check || num_gpus == 1) ? 0 : num_gpus]);
            if (Iteration::HAS_FULLQ)
            {
                peer_               = 0;
                frontier_queue_     = &(data_slice->frontier_queues
                    [(enactor -> size_check || num_gpus==1) ? 0 : num_gpus]);
                scanned_edges_      = &(data_slice->scanned_edges
                    [(enactor -> size_check || num_gpus==1) ? 0 : num_gpus]);
                frontier_attribute_ = &(frontier_attribute[peer_]);
                enactor_stats_      = &(enactor_stats[peer_]);
                work_progress_      = &(work_progress[peer_]);
                iteration           = enactor_stats[peer_].iteration;
                frontier_attribute_->queue_offset = 0;
                frontier_attribute_->queue_reset  = true;
                if (!enactor -> size_check) frontier_attribute_->selector     = 0;

                Iteration::FullQueue_Gather(
                    enactor,
                    thread_num,
                    peer_,
                    frontier_queue_,
                    scanned_edges_,
                    frontier_attribute_,
                    enactor_stats_,
                    data_slice,
                    s_data_slice[thread_num].GetPointer(util::DEVICE),
                    graph_slice,
                    work_progress_,
                    context[peer_],
                    streams[peer_]);
                selector            = frontier_attribute[peer_].selector;
                if (enactor_stats_->retval) break;

                if (frontier_attribute_->queue_length !=0)
                {
                    if (enactor -> debug) {
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
                        enactor,
                        frontier_attribute_,
                        //data_slice,
                        //s_data_slice[thread_num].GetPointer(util::DEVICE),
                        graph_slice    ->row_offsets     .GetPointer(util::DEVICE),
                        graph_slice    ->column_indices  .GetPointer(util::DEVICE),
                        graph_slice    ->column_offsets  .GetPointer(util::DEVICE),
                        graph_slice    ->row_indices     .GetPointer(util::DEVICE),
                        frontier_queue_->keys[selector]  .GetPointer(util::DEVICE),
                        scanned_edges_,
                        graph_slice    ->nodes,
                        graph_slice    ->edges,
                        context          [peer_][0],
                        streams          [peer_],
                        gunrock::oprtr::advance::V2V, true, false, false);
                    if (enactor_stats_->retval) break;

                    //frontier_attribute_->output_length.Move(
                    //    util::DEVICE, util::HOST, 1, 0, streams[peer_]);
                    if (enactor -> size_check)
                    {
                        tretval = cudaStreamSynchronize(streams[peer_]);
                        if (tretval != cudaSuccess) {enactor_stats_->retval=tretval;break;}

                        Iteration::Check_Queue_Size(
                            enactor,
                            thread_num,
                            peer_,
                            frontier_attribute_->output_length[0] + 2,
                            frontier_queue_,
                            frontier_attribute_,
                            enactor_stats_,
                            graph_slice);
                        if (enactor_stats_ -> retval) break;
                    }

                    if (fullqueue_latency != 0)
                        util::latency::Insert_Latency(
                        fullqueue_latency,
                        frontier_attribute_ -> queue_length,
                        streams[peer_],
                        data_slice -> latency_data.GetPointer(util::DEVICE));

                    Iteration::FullQueue_Core(
                        enactor,
                        thread_num,
                        peer_,
                        frontier_queue_,
                        scanned_edges_,
                        frontier_attribute_,
                        enactor_stats_,
                        data_slice,
                        s_data_slice[thread_num].GetPointer(util::DEVICE),
                        graph_slice,
                        work_progress_,
                        context[peer_],
                        streams[peer_]);
                    if (enactor_stats_->retval) break;
#ifdef ENABLE_PERFORMANCE_PROFILING
                    h_full_queue_nodes_queued = enactor_stats_ -> nodes_queued[0];
                    h_full_queue_edges_queued = enactor_stats_ -> edges_queued[0];
                    enactor_stats_ -> edges_queued.Move(util::DEVICE, util::HOST, 1, 0, streams[peer_]);
                    enactor_stats_ -> nodes_queued.Move(util::DEVICE, util::HOST, 1, 0, streams[peer_]);
#endif
                    //if (enactor_stats_ -> retval = util::GRError(
                    //    cudaStreamSynchronize(streams[peer_]),
                    //    "cudaStreamSynchronize failed", __FILE__, __LINE__))
                    //    break;
                    tretval = cudaErrorNotReady;
                    while (tretval == cudaErrorNotReady)
                    {
                        tretval = cudaStreamQuery(streams[peer_]);
                        if (tretval == cudaErrorNotReady)
                            sleep(0);
                    }
                    if (enactor_stats_ -> retval = util::GRError(tretval,
                        "FullQueue_Core failed.", __FILE__, __LINE__))
                        break;

#ifdef ENABLE_PERFORMANCE_PROFILING
                    iter_full_queue_nodes_queued.push_back(
                        h_full_queue_nodes_queued + enactor_stats_ -> nodes_queued[0]
                        - previous_full_queue_nodes_queued);
                    previous_full_queue_nodes_queued = h_full_queue_nodes_queued
                        + enactor_stats_ -> nodes_queued[0];
                    enactor_stats_ -> nodes_queued[0] = h_full_queue_nodes_queued;

                    iter_full_queue_edges_queued.push_back(
                        h_full_queue_edges_queued + enactor_stats_ -> edges_queued[0]
                        - previous_full_queue_edges_queued);
                    previous_full_queue_edges_queued = h_full_queue_edges_queued
                        + enactor_stats_ -> edges_queued[0];
                    enactor_stats_ -> edges_queued[0] = h_full_queue_edges_queued;
#endif
                    if (!enactor -> size_check)
                    {
                        if (enactor_stats_->retval =
                            Check_Size</*false,*/ SizeT, VertexId> (false, "queue3",
                                frontier_attribute->output_length[0]+2,
                                &frontier_queue_->keys[selector^1],
                                over_sized, thread_num, iteration, peer_, false))
                            break;
                    }
                    selector = frontier_attribute[peer_].selector;
                    Total_Length = frontier_attribute[peer_].queue_length;
                } else {
                    Total_Length = 0;
                    for (peer__=0;peer__<num_gpus;peer__++)
                        data_slice->out_length[peer__]=0;
#ifdef ENABLE_PERFORMANCE_PROFILING
                    iter_full_queue_nodes_queued.push_back(0);
                    iter_full_queue_edges_queued.push_back(0);
#endif
                }
#ifdef ENABLE_PERFORMANCE_PROFILING
                iter_full_queue_time.push_back(cpu_timer.MillisSinceStart() - subqueue_finish_time);
#endif
                if (enactor -> debug)
                {
                    printf("%d\t %lld\t \t Fullqueue finished. Total_Length= %lld\n",
                        thread_num, enactor_stats[0].iteration, (long long)Total_Length);
                    fflush(stdout);
                }
                frontier_queue_ = &(data_slice->frontier_queues[enactor -> size_check?0:num_gpus]);
                if (num_gpus==1) data_slice->out_length[0]=Total_Length;
            }

            if (num_gpus > 1)
            {
                for (peer_ = num_gpus+1; peer_ < num_gpus*2; peer_++)
                    data_slice->wait_marker[peer_]=0;
                wait_count=0;
                while (wait_count < num_gpus-1 &&
                    !Iteration::Stop_Condition(
                    s_enactor_stats, s_frontier_attribute, s_data_slice, num_gpus))
                {
                    for (peer_=num_gpus + 1; peer_<num_gpus*2; peer_++)
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

                Iteration::Iteration_Update_Preds(
                    enactor,
                    graph_slice,
                    data_slice,
                    &frontier_attribute[0],
                    &data_slice->frontier_queues[enactor -> size_check?0:num_gpus],
                    Total_Length,
                    streams[0]);

                if (makeout_latency != 0)
                    util::latency::Insert_Latency(
                    makeout_latency, Total_Length,
                    streams[0],
                    data_slice -> latency_data.GetPointer(util::DEVICE));

                Iteration::template Make_Output <NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
                    enactor,
                    thread_num,
                    Total_Length,
                    num_gpus,
                    &data_slice->frontier_queues[enactor -> size_check?0:num_gpus],
                    &data_slice->scanned_edges[0],
                    &frontier_attribute[0],
                    enactor_stats,
                    &problem->data_slices[thread_num],
                    graph_slice,
                    &work_progress[0],
                    context[0],
                    streams[0]);
            }
            else
            {
                data_slice->out_length[0]= Total_Length;
            }

            for (peer_=0;peer_<num_gpus;peer_++)
            {
                frontier_attribute[peer_].queue_length = data_slice->out_length[peer_];
#ifdef ENABLE_PERFORMANCE_PROFILING
                //if (peer_ == 0)
                    enactor_stats[peer_].iter_out_length.back().push_back(
                        data_slice -> out_length[peer_]);
#endif
            }
        }

#ifdef ENABLE_PERFORMANCE_PROFILING
        iter_stop_time = cpu_timer.MillisSinceStart();
        iter_total_time.push_back(iter_stop_time - iter_start_time);
        iter_start_time = iter_stop_time;
#endif
        Iteration::Iteration_Change(enactor_stats->iteration);
    }
}

/*
 * @brief IterationBase data structure.
 *
 * @tparam AdvanceKernelPolicy
 * @tparam FilterKernelPolicy
 * @tparam Enactor
 * @tparam _HAS_SUBQ
 * @tparam _HAS_FULLQ
 * @tparam _BACKWARD
 * @tparam _FORWARD
 * @tparam _UPDATE_PREDECESSORS
 */
template <
    typename _AdvanceKernelPolicy,
    typename _FilterKernelPolicy,
    typename _Enactor,
    bool     _HAS_SUBQ,
    bool     _HAS_FULLQ,
    bool     _BACKWARD,
    bool     _FORWARD,
    bool     _UPDATE_PREDECESSORS>
struct IterationBase
{
public:
    typedef _Enactor                     Enactor   ;
    typedef _AdvanceKernelPolicy AdvanceKernelPolicy;
    typedef _FilterKernelPolicy  FilterKernelPolicy;
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<VertexId, SizeT, Value>
                                         GraphSliceT;
    typedef util::DoubleBuffer<VertexId, SizeT, Value>
                                         Frontier;
    //static const bool INSTRUMENT = Enactor::INSTRUMENT;
    //static const bool DEBUG      = Enactor::DEBUG;
    //static const bool SIZE_CHECK = Enactor::SIZE_CHECK;
    static const bool HAS_SUBQ   = _HAS_SUBQ;
    static const bool HAS_FULLQ  = _HAS_FULLQ;
    static const bool BACKWARD   = _BACKWARD;
    static const bool FORWARD    = _FORWARD;
    static const bool UPDATE_PREDECESSORS = _UPDATE_PREDECESSORS;

    /*
     * @brief SubQueue_Gather function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void SubQueue_Gather(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief SubQueue_Core function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void SubQueue_Core(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief FullQueue_Gather function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Gather(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief FullQueue_Core function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] d_data_slice Pointer to the data slice on the device.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    static void FullQueue_Core(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
    }

    /*
     * @brief Stop_Condition check function.
     *
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] num_gpus Number of GPUs used.
     */
    static bool Stop_Condition(
        EnactorStats<SizeT>           *enactor_stats,
        FrontierAttribute<SizeT>      *frontier_attribute,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice,
        int                            num_gpus)
    {
        return All_Done(enactor_stats,frontier_attribute,data_slice,num_gpus);
    }

    /*
     * @brief Iteration_Change function.
     *
     * @param[in] iterations
     */
    static void Iteration_Change(long long &iterations)
    {
        iterations++;
    }

    /*
     * @brief Iteration_Update_Preds function.
     *
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] num_elements Number of elements.
     * @param[in] stream CUDA stream.
     */
    static void Iteration_Update_Preds(
        Enactor                       *enactor,
        GraphSliceT                   *graph_slice,
        DataSlice                     *data_slice,
        FrontierAttribute<SizeT>      *frontier_attribute,
        Frontier                      *frontier_queue,
        SizeT                          num_elements,
        cudaStream_t                   stream)
    {
        if (num_elements == 0) return;
        int selector    = frontier_attribute->selector;
        int grid_size   = num_elements / 256;
        if ((num_elements % 256) !=0) grid_size++;
        if (grid_size > 512) grid_size = 512;

        if (Problem::MARK_PREDECESSORS && UPDATE_PREDECESSORS && num_elements>0 )
        {
            Copy_Preds<VertexId, SizeT> <<<grid_size,256,0, stream>>>(
                num_elements,
                frontier_queue->keys[selector].GetPointer(util::DEVICE),
                data_slice    ->preds         .GetPointer(util::DEVICE),
                data_slice    ->temp_preds    .GetPointer(util::DEVICE));

            Update_Preds<VertexId,SizeT> <<<grid_size,256,0,stream>>>(
                num_elements,
                graph_slice   ->nodes,
                frontier_queue->keys[selector] .GetPointer(util::DEVICE),
                graph_slice   ->original_vertex.GetPointer(util::DEVICE),
                data_slice    ->temp_preds     .GetPointer(util::DEVICE),
                data_slice    ->preds          .GetPointer(util::DEVICE));//,
        }
    }

    /*
     * @brief Check frontier queue size function.
     *
     * @param[in] thread_num Number of threads.
     * @param[in] peer_ Peer GPU index.
     * @param[in] request_length Request frontier queue length.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     */
    static void Check_Queue_Size(
        Enactor                       *enactor,
        int                            thread_num,
        int                            peer_,
        SizeT                          request_length,
        Frontier                      *frontier_queue,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        GraphSliceT                   *graph_slice)
    {
        bool over_sized = false;
        int  selector   = frontier_attribute->selector;
        int  iteration  = enactor_stats -> iteration;

        if (enactor -> debug)
            printf("%d\t %d\t %d\t queue_length = %d, output_length = %d\n",
                thread_num, iteration, peer_,
                frontier_queue->keys[selector^1].GetSize(),
                request_length);fflush(stdout);

        if (enactor_stats->retval =
            Check_Size</*true,*/ SizeT, VertexId > (
                true, "queue3", request_length, &frontier_queue->keys  [selector^1],
                over_sized, thread_num, iteration, peer_, false)) return;
        if (enactor_stats->retval =
            Check_Size</*true,*/ SizeT, VertexId > (
                true, "queue3", request_length, &frontier_queue->keys  [selector  ],
                over_sized, thread_num, iteration, peer_, true )) return;
        if (enactor -> problem -> use_double_buffer)
        {
            if (enactor_stats->retval =
                Check_Size</*true,*/ SizeT, Value> (
                    true, "queue3", request_length, &frontier_queue->values[selector^1],
                    over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor_stats->retval =
                Check_Size</*true,*/ SizeT, Value> (
                    true, "queue3", request_length, &frontier_queue->values[selector  ],
                    over_sized, thread_num, iteration, peer_, true )) return;
        }
    }

    /*
     * @brief Make_Output function.
     *
     * @tparam NUM_VERTEX_ASSOCIATES
     * @tparam NUM_VALUE__ASSOCIATES
     *
     * @param[in] thread_num Number of threads.
     * @param[in] num_elements
     * @param[in] num_gpus Number of GPUs used.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    static void Old_Make_Output(
        Enactor                       *enactor,
        int                            thread_num,
        SizeT                          num_elements,
        int                            num_gpus,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice_,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        if (num_gpus < 2) return;
        bool over_sized = false, keys_over_sized = false;
        int peer_ = 0, t=0, i=0;
        size_t offset = 0;
        SizeT *t_out_length = new SizeT[num_gpus];
        int selector = frontier_attribute->selector;
        int block_size = 256;
        int grid_size  = num_elements / block_size;
        if ((num_elements % block_size)!=0) grid_size ++;
        if (grid_size > 512) grid_size=512;
        DataSlice* data_slice=data_slice_->GetPointer(util::HOST);

        for (peer_ = 0; peer_<num_gpus; peer_++)
        {
            t_out_length[peer_] = 0;
            data_slice->out_length[peer_] = 0;
        }
        if (num_elements ==0) return;

        over_sized = false;
        for (peer_ = 0; peer_<num_gpus; peer_++)
        {
            if (enactor_stats->retval =
                Check_Size<SizeT, SizeT> (
                    enactor -> size_check, "keys_marker",
                    num_elements, &data_slice->keys_marker[peer_],
                    over_sized, thread_num, enactor_stats->iteration, peer_))
                break;
            if (over_sized)
                data_slice->keys_markers[peer_] =
                    data_slice->keys_marker[peer_].GetPointer(util::DEVICE);
        }
        if (enactor_stats->retval) return;
        if (over_sized)
            data_slice->keys_markers.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);

        for (t=0; t<2; t++)
        {
            if (t==0 && !FORWARD) continue;
            if (t==1 && !BACKWARD) continue;

            if (BACKWARD && t==1)
                Assign_Marker_Backward<VertexId, SizeT>
                    <<<grid_size, block_size, num_gpus * sizeof(SizeT*) ,stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue->keys[selector]    .GetPointer(util::DEVICE),
                    graph_slice   ->backward_offset   .GetPointer(util::DEVICE),
                    graph_slice   ->backward_partition.GetPointer(util::DEVICE),
                    data_slice    ->keys_markers      .GetPointer(util::DEVICE));
            else if (FORWARD && t==0)
                Assign_Marker<VertexId, SizeT>
                    <<<grid_size, block_size, num_gpus * sizeof(SizeT*) ,stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue->keys[selector]    .GetPointer(util::DEVICE),
                    graph_slice   ->partition_table   .GetPointer(util::DEVICE),
                    data_slice    ->keys_markers      .GetPointer(util::DEVICE));

            for (peer_=0;peer_<num_gpus;peer_++)
            {
                Scan<mgpu::MgpuScanTypeInc>(
                    (SizeT*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                    num_elements,
                    (SizeT)0, mgpu::plus<SizeT>(), (SizeT*)0, (SizeT*)0,
                    (SizeT*)data_slice->keys_marker[peer_].GetPointer(util::DEVICE),
                    context[0]);
            }

            if (num_elements>0) for (peer_=0; peer_<num_gpus;peer_++)
            {
                cudaMemcpyAsync(&(t_out_length[peer_]),
                    data_slice->keys_marker[peer_].GetPointer(util::DEVICE)
                        + (num_elements -1),
                    sizeof(SizeT), cudaMemcpyDeviceToHost, stream);
            } else {
                for (peer_=0;peer_<num_gpus;peer_++)
                    t_out_length[peer_]=0;
            }
            if (enactor_stats->retval = cudaStreamSynchronize(stream)) break;

            keys_over_sized = true;
            for (peer_=0; peer_<num_gpus;peer_++)
            {
                if (enactor_stats->retval =
                    Check_Size <SizeT, VertexId> (
                        enactor -> size_check, "keys_out",
                        data_slice->out_length[peer_] + t_out_length[peer_],
                        peer_!=0 ? &data_slice->keys_out[peer_] :
                                   &data_slice->frontier_queues[0].keys[selector^1],
                        keys_over_sized, thread_num, enactor_stats[0].iteration, peer_),
                        data_slice->out_length[peer_]==0? false: true) break;
                if (keys_over_sized)
                    data_slice->keys_outs[peer_] = peer_==0 ?
                        data_slice->frontier_queues[0].keys[selector^1].GetPointer(util::DEVICE) :
                        data_slice->keys_out[peer_].GetPointer(util::DEVICE);
                if (peer_ == 0) continue;

                over_sized = false;
                for (i=0;i<NUM_VERTEX_ASSOCIATES;i++)
                {
                    if (enactor_stats[0].retval =
                        Check_Size <SizeT, VertexId>(
                            enactor -> size_check, "vertex_associate_outs",
                            data_slice->out_length[peer_] + t_out_length[peer_],
                            &data_slice->vertex_associate_out[peer_][i],
                            over_sized, thread_num, enactor_stats->iteration, peer_),
                            data_slice->out_length[peer_]==0? false: true)
                        break;
                    if (over_sized)
                        data_slice->vertex_associate_outs[peer_][i] =
                            data_slice->vertex_associate_out[peer_][i].GetPointer(util::DEVICE);
                }
                if (enactor_stats->retval) break;
                if (over_sized)
                    data_slice->vertex_associate_outs[peer_].Move(
                        util::HOST, util::DEVICE, NUM_VERTEX_ASSOCIATES, 0, stream);

                over_sized = false;
                for (i=0;i<NUM_VALUE__ASSOCIATES;i++)
                {
                    if (enactor_stats->retval =
                        Check_Size<SizeT, Value   >(
                            enactor -> size_check, "value__associate_outs",
                            data_slice->out_length[peer_] + t_out_length[peer_],
                            &data_slice->value__associate_out[peer_][i],
                            over_sized, thread_num, enactor_stats->iteration, peer_,
                            data_slice->out_length[peer_]==0? false: true)) break;
                    if (over_sized)
                        data_slice->value__associate_outs[peer_][i] =
                            data_slice->value__associate_out[peer_][i].GetPointer(util::DEVICE);
                }
                if (enactor_stats->retval) break;
                if (over_sized)
                    data_slice->value__associate_outs[peer_].Move(
                        util::HOST, util::DEVICE, NUM_VALUE__ASSOCIATES, 0, stream);
            }
            if (enactor_stats->retval) break;
            if (keys_over_sized)
                data_slice->keys_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);

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
            for (peer_=0; peer_<num_gpus; peer_++)
            {
                memcpy(&(data_slice->make_out_array[offset]),
                         data_slice->vertex_associate_outs[peer_].GetPointer(util::HOST),
                          sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES);
                offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES ;
            }
            for (peer_=0; peer_<num_gpus; peer_++)
            {
                memcpy(&(data_slice->make_out_array[offset]),
                        data_slice->value__associate_outs[peer_].GetPointer(util::HOST),
                          sizeof(Value*   ) * NUM_VALUE__ASSOCIATES);
                offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES ;
            }
            memcpy(&(data_slice->make_out_array[offset]),
                     data_slice->out_length.GetPointer(util::HOST),
                      sizeof(SizeT) * num_gpus);
            offset += sizeof(SizeT) * num_gpus;
            data_slice->make_out_array.Move(util::HOST, util::DEVICE, offset, 0, stream);

            if (BACKWARD && t==1)
                Make_Out_Backward<VertexId, SizeT, Value,
                    NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                    <<<grid_size, block_size, sizeof(char)*offset, stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue-> keys[selector]      .GetPointer(util::DEVICE),
                    graph_slice   -> backward_offset     .GetPointer(util::DEVICE),
                    graph_slice   -> backward_partition  .GetPointer(util::DEVICE),
                    graph_slice   -> backward_convertion .GetPointer(util::DEVICE),
                    offset,
                    data_slice    -> make_out_array      .GetPointer(util::DEVICE));
            else if (FORWARD && t==0)
                Make_Out<VertexId, SizeT, Value,
                    NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
                    <<<grid_size, block_size, sizeof(char)*offset, stream>>> (
                    num_elements,
                    num_gpus,
                    frontier_queue-> keys[selector]      .GetPointer(util::DEVICE),
                    graph_slice   -> partition_table     .GetPointer(util::DEVICE),
                    graph_slice   -> convertion_table    .GetPointer(util::DEVICE),
                    offset,
                    data_slice    -> make_out_array      .GetPointer(util::DEVICE));
            for (peer_ = 0; peer_<num_gpus; peer_++)
            {
                data_slice->out_length[peer_] += t_out_length[peer_];
            }
        }
        if (enactor_stats->retval) return;
        if (enactor_stats->retval = cudaStreamSynchronize(stream)) return;
        frontier_attribute->selector^=1;
        if (t_out_length!=NULL) {delete[] t_out_length; t_out_length=NULL;}
    }

    /*
     * @brief Make_Output function.
     *
     * @tparam NUM_VERTEX_ASSOCIATES
     * @tparam NUM_VALUE__ASSOCIATES
     *
     * @param[in] thread_num Number of threads.
     * @param[in] num_elements
     * @param[in] num_gpus Number of GPUs used.
     * @param[in] frontier_queue Pointer to the frontier queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] enactor_stats Pointer to the enactor statistics.
     * @param[in] data_slice Pointer to the data slice we process on.
     * @param[in] graph_slice Pointer to the graph slice we process on.
     * @param[in] work_progress Pointer to the work progress class.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     */
    template <
        int NUM_VERTEX_ASSOCIATES,
        int NUM_VALUE__ASSOCIATES>
    static void Make_Output(
        Enactor                       *enactor,
        int                            thread_num,
        SizeT                          num_elements,
        int                            num_gpus,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats<SizeT>           *enactor_stats,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice_,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime<SizeT> *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        DataSlice* data_slice=data_slice_->GetPointer(util::HOST);
        if (num_gpus < 2) return;
        if (num_elements == 0)
        {
            for (int peer_ = 0; peer_ < num_gpus; peer_ ++)
            {
                data_slice -> out_length[peer_] = 0;
            }
            return;
        }
        bool over_sized = false, keys_over_sized = false;
        int selector = frontier_attribute->selector;
        //printf("%d Make_Output begin, num_elements = %d, size_check = %s\n",
        //    data_slice -> gpu_idx, num_elements, enactor->size_check ? "true" : "false");
        //fflush(stdout);
        SizeT size_multi = 0;
        if (FORWARD ) size_multi += 1;
        if (BACKWARD) size_multi += 1;

        int peer_ = 0;
        for (peer_ = 0; peer_ < num_gpus; peer_++)
        {
            if (enactor_stats -> retval =
                Check_Size<SizeT, VertexId> (
                    enactor -> size_check, "keys_out",
                    num_elements * size_multi,
                    (peer_ == 0) ?
                        &data_slice -> frontier_queues[0].keys[selector^1] :
                        &data_slice -> keys_out[peer_],
                    keys_over_sized, thread_num, enactor_stats[0].iteration,
                    peer_),
                    false)
                break;
            //if (keys_over_sized)
                data_slice->keys_outs[peer_] = (peer_==0) ?
                    data_slice -> frontier_queues[0].keys[selector^1].GetPointer(util::DEVICE) :
                    data_slice -> keys_out[peer_].GetPointer(util::DEVICE);
            if (peer_ == 0) continue;

            over_sized = false;
            //for (i = 0; i< NUM_VERTEX_ASSOCIATES; i++)
            //{
                if (enactor_stats[0].retval =
                    Check_Size <SizeT, VertexId>(
                        enactor -> size_check, "vertex_associate_outs",
                        num_elements * NUM_VERTEX_ASSOCIATES * size_multi,
                        &data_slice->vertex_associate_out[peer_],
                        over_sized, thread_num, enactor_stats->iteration, peer_),
                        false)
                    break;
                //if (over_sized)
                    data_slice->vertex_associate_outs[peer_] =
                        data_slice->vertex_associate_out[peer_].GetPointer(util::DEVICE);
            //}
            //if (enactor_stats->retval) break;
            //if (over_sized)
            //    data_slice->vertex_associate_outs[peer_].Move(
            //        util::HOST, util::DEVICE, NUM_VERTEX_ASSOCIATES, 0, stream);

            over_sized = false;
            //for (i=0;i<NUM_VALUE__ASSOCIATES;i++)
            //{
                if (enactor_stats->retval =
                    Check_Size<SizeT, Value   >(
                        enactor -> size_check, "value__associate_outs",
                        num_elements * NUM_VALUE__ASSOCIATES * size_multi,
                        &data_slice->value__associate_out[peer_],
                        over_sized, thread_num, enactor_stats->iteration, peer_,
                        false)) break;
                //if (over_sized)
                    data_slice->value__associate_outs[peer_] =
                        data_slice->value__associate_out[peer_].GetPointer(util::DEVICE);
            //}
            //if (enactor_stats->retval) break;
            //if (over_sized)
            //    data_slice->value__associate_outs[peer_].Move(
            //        util::HOST, util::DEVICE, NUM_VALUE__ASSOCIATES, 0, stream);
            if (enactor -> problem -> skip_makeout_selection) break;
        }
        if (enactor_stats->retval) return;
        if (enactor -> problem -> skip_makeout_selection)
        {
            if (NUM_VALUE__ASSOCIATES == 0 && NUM_VERTEX_ASSOCIATES == 0)
            {
                util::MemsetCopyVectorKernel<<<120, 512, 0, stream>>>(
                    data_slice -> keys_out[1].GetPointer(util::DEVICE),
                    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                    num_elements);
                for (int peer_=0; peer_<num_gpus; peer_++)
                    data_slice -> out_length[peer_] = num_elements;
                if (enactor_stats -> retval = util::GRError(
                    cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize failed", __FILE__, __LINE__))
                    return;
                return;
            } else {
                for (int peer_ = 2; peer_ < num_gpus; peer_++)
                {
                    data_slice -> keys_out[peer_].SetPointer(
                        data_slice -> keys_out[1].GetPointer(util::DEVICE),
                        data_slice -> keys_out[1].GetSize(), util::DEVICE);
                    data_slice -> keys_outs[peer_] = data_slice -> keys_out[peer_].GetPointer(util::DEVICE);

                    data_slice -> vertex_associate_out[peer_].SetPointer(
                        data_slice -> vertex_associate_out[1].GetPointer(util::DEVICE),
                        data_slice -> vertex_associate_out[1].GetSize(), util::DEVICE);
                    data_slice -> vertex_associate_outs[peer_] = data_slice -> vertex_associate_out[peer_].GetPointer(util::DEVICE);

                    data_slice -> value__associate_out[peer_].SetPointer(
                        data_slice -> value__associate_out[1].GetPointer(util::DEVICE),
                        data_slice -> value__associate_out[1].GetSize(), util::DEVICE);
                    data_slice -> value__associate_outs[peer_] = data_slice -> value__associate_out[peer_].GetPointer(util::DEVICE);
                }
            }
        }
        //printf("%d Make_Out 1\n", data_slice -> gpu_idx);
        //fflush(stdout);
        //if (keys_over_sized)
        data_slice -> keys_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);
        data_slice -> vertex_associate_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);
        data_slice -> value__associate_outs.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("PreMakeOut",
        //    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
        //    num_elements, data_slice -> gpu_idx, enactor_stats -> iteration, -1, stream);
        int num_blocks = (num_elements >> (AdvanceKernelPolicy::LOG_THREADS)) + 1;
        if (num_blocks > 480) num_blocks = 480;
        //printf("%d Make_Out 2, num_blocks = %d, num_threads = %d\n",
        //    data_slice -> gpu_idx, num_blocks, AdvanceKernelPolicy::THREADS);
        //fflush(stdout);
        if (!enactor -> problem -> skip_makeout_selection)
        {
            for (int i=0; i< num_gpus; i++) data_slice -> out_length[i] = 1;
            data_slice -> out_length.Move(util::HOST, util::DEVICE, num_gpus, 0, stream);
            //printf("Make_Output direction = %s %s\n", FORWARD ? "FORWARD" : "", BACKWARD ? "BACKWARD" : "");

            /*printf("num_blocks = %d, num_threads = %d, stream = %p, "
                "num_elements = %d, num_gpus = %d, out_length = %p, (%d)"
                "keys_in = %p (%d), partition_table = %p (%d), convertion_table = %d (%d), "
                "vertex_associate_orgs = %p (%d), value__associate_orgs = %p (%d), "
                "keys_outs = %p (%d), vertex_associate_outs = %p (%d), value__associate_outs = %p (%d), "
                "keep_node_num = %s, num_vertex_associates = %d, num_value_associates = %d\n",
                num_blocks, AdvanceKernelPolicy::THREADS /2, stream,
                num_elements, num_gpus,
                data_slice -> out_length.GetPointer(util::DEVICE), data_slice -> out_length.GetSize(),
                frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                frontier_queue -> keys[frontier_attribute -> selector].GetSize(),
                graph_slice -> partition_table      .GetPointer(util::DEVICE),
                graph_slice -> partition_table      .GetSize(),
                graph_slice -> convertion_table     .GetPointer(util::DEVICE),
                graph_slice -> convertion_table     .GetSize(),
                data_slice  -> vertex_associate_orgs[0],
                data_slice  -> vertex_associate_orgs.GetSize(),
                data_slice  -> value__associate_orgs[0],
                data_slice  -> value__associate_orgs.GetSize(),
                data_slice  -> keys_outs            .GetPointer(util::DEVICE),
                data_slice  -> keys_outs            .GetSize(),
                data_slice  -> vertex_associate_outs[1],
                data_slice  -> vertex_associate_outs.GetSize(),
                data_slice  -> value__associate_outs[1],
                data_slice  -> value__associate_outs.GetSize(),
                enactor -> problem -> keep_node_num ? "true" : "false",
                NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES);*/

            if (FORWARD)
                Make_Output_Kernel < VertexId, SizeT, Value,
                    NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES,
                    AdvanceKernelPolicy::CUDA_ARCH,
                    AdvanceKernelPolicy::LOG_THREADS-1>
                    <<< num_blocks, AdvanceKernelPolicy::THREADS / 2, 0, stream >>> (
                    num_elements,
                    num_gpus,
                    data_slice -> out_length.GetPointer(util::DEVICE),
                    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                    graph_slice -> partition_table      .GetPointer(util::DEVICE),
                    graph_slice -> convertion_table     .GetPointer(util::DEVICE),
                    data_slice  -> vertex_associate_orgs.GetPointer(util::DEVICE),
                    data_slice  -> value__associate_orgs.GetPointer(util::DEVICE),
                    data_slice  -> keys_outs            .GetPointer(util::DEVICE),
                    data_slice  -> vertex_associate_outs.GetPointer(util::DEVICE),
                    data_slice  -> value__associate_outs.GetPointer(util::DEVICE),
                    enactor -> problem -> keep_node_num);

            if (BACKWARD)
                Make_Output_Backward_Kernel < VertexId, SizeT, Value,
                    NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES,
                    AdvanceKernelPolicy::CUDA_ARCH,
                    AdvanceKernelPolicy::LOG_THREADS-1>
                    <<< num_blocks, AdvanceKernelPolicy::THREADS / 2, 0, stream >>> (
                    num_elements,
                    num_gpus,
                    data_slice -> out_length.GetPointer(util::DEVICE),
                    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                    graph_slice -> backward_offset      .GetPointer(util::DEVICE),
                    graph_slice -> backward_partition   .GetPointer(util::DEVICE),
                    graph_slice -> backward_convertion  .GetPointer(util::DEVICE),
                    data_slice  -> vertex_associate_orgs.GetPointer(util::DEVICE),
                    data_slice  -> value__associate_orgs.GetPointer(util::DEVICE),
                    data_slice  -> keys_outs            .GetPointer(util::DEVICE),
                    data_slice  -> vertex_associate_outs.GetPointer(util::DEVICE),
                    data_slice  -> value__associate_outs.GetPointer(util::DEVICE),
                    enactor -> problem -> keep_node_num);

            data_slice -> out_length.Move(util::DEVICE, util::HOST, num_gpus, 0, stream);
            frontier_attribute->selector^=1;
        } else {
            Make_Output_Kernel_SkipSelection < VertexId, SizeT, Value,
                NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES,
                AdvanceKernelPolicy::CUDA_ARCH,
                AdvanceKernelPolicy::LOG_THREADS>
                <<< num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>> (
                num_elements,
                frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
                data_slice -> vertex_associate_orgs.GetPointer(util::DEVICE),
                data_slice -> value__associate_orgs.GetPointer(util::DEVICE),
                data_slice -> keys_out[1]          .GetPointer(util::DEVICE),
                data_slice -> vertex_associate_out[1].GetPointer(util::DEVICE),
                data_slice -> value__associate_out[1].GetPointer(util::DEVICE));
            for (int peer_=0; peer_<num_gpus; peer_++)
                data_slice -> out_length[peer_] = num_elements;
        }
        if (enactor_stats -> retval = util::GRError(cudaStreamSynchronize(stream),
            "Make_Output failed", __FILE__, __LINE__))
            return;
        if (!enactor -> problem -> skip_makeout_selection)
        {
            for (int i=0; i< num_gpus; i++)
            {
                data_slice -> out_length[i] --;
                //printf("out_length[%d] = %d\n", i, data_slice -> out_length[i]);
            }
        }
        //for (int i=0; i<num_gpus; i++)
        //{
            //if (i == 0)
            //    printf("%d, selector = %d, keys = %p\n",
            //        data_slice -> gpu_idx, frontier_attribute -> selector^1,
            //        data_slice -> keys_outs[i]);
        //    util::cpu_mt::PrintGPUArray<SizeT, VertexId>("PostMakeOut",
        //        data_slice -> keys_outs[i], data_slice -> out_length[i],
        //        data_slice -> gpu_idx, enactor_stats -> iteration, i, stream);
        //}

        //printf("%d Make_Out 3\n", data_slice -> gpu_idx);
        //fflush(stdout);
    }
};

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
