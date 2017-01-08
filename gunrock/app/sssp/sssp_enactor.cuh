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

#include <thread>
#include <gunrock/util/multithreading.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/util/device_intrinsics.cuh>
#include <gunrock/util/scan/block_scan.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/priority_queue/kernel.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/sssp/sssp_problem.cuh>
#include <gunrock/app/sssp/sssp_functor.cuh>


namespace gunrock {
namespace app {
namespace sssp {

template <
    typename KernelPolicy,
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES>
__global__ void Expand_Incoming_Kernel(
    int thread_num,
    typename KernelPolicy::VertexId  label,
    typename KernelPolicy::SizeT     num_elements,
    typename KernelPolicy::VertexId *d_keys_in,
    typename KernelPolicy::VertexId *d_vertex_associate_in,
    typename KernelPolicy::Value    *d_value__associate_in,
    typename KernelPolicy::SizeT    *d_out_length,
    typename KernelPolicy::VertexId *d_keys_out,
    typename KernelPolicy::VertexId *d_preds,
    typename KernelPolicy::Value    *d_distances,
    typename KernelPolicy::VertexId *d_labels)
{
    typedef typename KernelPolicy::VertexId VertexId;
    typedef typename KernelPolicy::SizeT    SizeT;
    typedef typename KernelPolicy::Value    Value;
    typedef util::Block_Scan<SizeT, KernelPolicy::CUDA_ARCH, KernelPolicy::LOG_THREADS> BlockScanT;

    __shared__ typename BlockScanT::Temp_Space scan_space;
    __shared__ SizeT block_offset;
    SizeT x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
    VertexId key = util::InvalidValue<VertexId>();
    while (x - threadIdx.x < num_elements)
    {
        bool to_process = true;
        SizeT output_pos = util::InvalidValue<SizeT>();
        VertexId key = util::InvalidValue<VertexId>();
        Value value;

        if (x < num_elements)
        {
            value = d_value__associate_in[x];
            key = d_keys_in[x];
            Value old_value = atomicMin(d_distances + key, value);
            if (old_value <= value)
                to_process = false;
            else {
                //if (d_labels[key] == label) to_process = false;
                //else d_labels[key] = label;
                //VertexId old_label = atomicExch(d_labels + key, label);
                //d_labels[key] = label;
                if (d_labels[key] == label)
                {
                    //printf("Expand label[%d] (%d) -> %d, value (%d) -> %d\n",
                    //    key, old_label, label, old_value, value);
                    to_process = false;
                } else d_labels[key] = label;
            }
            //printf("Expand label[%d] : %d -> %d\n",
            //    key, old_value, t);
        } else to_process = false;

        BlockScanT::LogicScan(to_process, output_pos, scan_space);
        if (threadIdx.x == blockDim.x -1)
        {
            block_offset = atomicAdd(d_out_length, output_pos + ((to_process) ? 1 : 0));
        }
        __syncthreads();

        if (to_process)
        {
            output_pos += block_offset;
            d_keys_out[output_pos] = key;
            if (NUM_VERTEX_ASSOCIATES == 1 && d_distances[key] == value)
                d_preds[key] = d_vertex_associate_in[x];
        }
        x += STRIDE;
    }
}

/*
 * @brief Main iteration loop for SSSP primitive
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy  Kernel policy for filter operator.
 * @tparam Enactor Enactor we process on.
 */
template <
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
struct SSSPIteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    false,//true , // HAS_SUBQ, whether to use SubQ_Core
    true,//false, // HAS_FULLQ, whether to use FullQ_Core
    false, // BACKWARD, whether communication goes backward
    true , // FORWARD , whether communicaiton goes forward
    Enactor::Problem::MARK_PATHS>//MARK_PREDECESSORS, whether to mark predecessors
{
    typedef typename Enactor::SizeT     SizeT     ;
    typedef typename Enactor::Value     Value     ;
    typedef typename Enactor::VertexId  VertexId  ;
    typedef typename Enactor::Problem   Problem   ;
    typedef typename Problem::DataSlice DataSlice ;
    typedef GraphSlice<VertexId, SizeT, Value>
                                        GraphSliceT;
    typedef typename util::DoubleBuffer<VertexId, SizeT, Value>
                                        Frontier  ;
    typedef SSSPFunctor<VertexId, SizeT, Value, Problem>
                                        Functor   ;
    typedef typename Functor::LabelT    LabelT    ;
    typedef IterationBase <
        AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
        false, true, false, true, Enactor::Problem::MARK_PATHS>//MARK_PREDECESSORS>
                                        BaseIteration;

    /*
     * @brief Per-iteration computation steps, here is the core of algorithm. Runs when local and received sub-frontiers are ready (and combined)
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
        frontier_attribute->queue_reset = true;
        enactor_stats     ->nodes_queued[0] += frontier_attribute -> queue_length;

        if (enactor -> debug)
            util::cpu_mt::PrintMessage("Advance begin",
                thread_num, enactor_stats->iteration, peer_);

        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys0",
        //    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
        //    frontier_attribute -> queue_length,
        //    thread_num, enactor_stats -> iteration, peer_, stream);

        // Edge Map
        gunrock::oprtr::advance::LaunchKernel
            <AdvanceKernelPolicy, Problem, Functor, gunrock::oprtr::advance::V2V>(
            enactor_stats[0],
            frontier_attribute[0],
            enactor_stats -> iteration + 1,
            data_slice,
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges ->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector]  .GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            (Value*   )NULL,
            (Value*   )NULL,
            graph_slice->row_offsets   .GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            graph_slice->nodes, // max_in_queue
            graph_slice->edges, // max_out_queue
            work_progress[0],
            context[0],
            stream,
            false,
            false,
            false);
        if (enactor -> debug)
            util::cpu_mt::PrintMessage("Advance end",
                thread_num, enactor_stats->iteration, peer_);

        frontier_attribute -> queue_reset = false;
        if (gunrock::oprtr::advance::hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>())
        {
            enactor_stats -> edges_queued[0] += frontier_attribute -> output_length[0];
        } else {
            enactor_stats      -> AccumulateEdges(
                work_progress  -> template GetQueueLengthPointer<unsigned int>(
                    frontier_attribute -> queue_index + 1), stream);
        }

        if (!gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>())
        {
            frontier_attribute -> queue_index++;
            frontier_attribute -> selector ^= 1;

            if (enactor -> debug)
                util::cpu_mt::PrintMessage("Filter begin",
                    thread_num, enactor_stats -> iteration, peer_);
            //Vertex Map
            gunrock::oprtr::filter::LaunchKernel
                <FilterKernelPolicy, Problem, Functor> (
                enactor_stats[0],
                frontier_attribute[0],
                enactor_stats -> iteration + 1,//util::InvalidValue<VertexId>(),
                data_slice,
                d_data_slice,
                (SizeT*) NULL,
                (unsigned char*) NULL,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),     // d_out_queue
                (Value*) NULL,
                (Value*) NULL,
                frontier_attribute -> output_length[0],
                graph_slice -> nodes,
                work_progress[0],
                context[0],
                stream,
                frontier_queue->keys[frontier_attribute->selector  ].GetSize(), // max_in_queue
                frontier_queue->keys[frontier_attribute->selector^1].GetSize(), // max_out_queue
                enactor_stats->filter_kernel_stats);

            if (enactor -> debug && (enactor_stats->retval =
                util::GRError("filter_forward::Kernel failed", __FILE__, __LINE__))) return;
            if (enactor -> debug)
                util::cpu_mt::PrintMessage("Filter end.",
                    thread_num, enactor_stats->iteration, peer_);
        }

        //TODO: split the output queue into near/far pile, put far pile in far queue, put near pile as the input queue
        //for next round.
        if (data_slice->num_gpus == 1)
        { // when #gpus == 1, use priority queue
            frontier_attribute->queue_index ++;
            frontier_attribute->selector ^= 1;
            /*if (enactor_stats->retval = util::GRError(
                work_progress -> GetQueueLength(frointer_attribute->queue_index,
                frontier_attribute->queue_length, true, stream),
                "work_progress -> GetQueueLength failed", __FILE__, __LINE__))
                return;
            out_length = 0;

            if (frontier_attribute->queue_length > 0) {
                out_length = gunrock::priority_queue::Bisect
                    <PriorityQueueKernelPolicy, SSSPProblem,
                    NearFarPriorityQueue, PqFunctor>(
                    (int*)frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                    pq,
                    (unsigned int)frontier_attribute->queue_length,
                    d_data_slice,
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
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
            }*/
            //util::MemsetKernel<<<128, 128, 0, stream>>> (
            //    data_slice -> sssp_marker.GetPointer(util::DEVICE),
            //    (int)0, graph_slice->nodes);
        } else { // #gpus != 1
            frontier_attribute->queue_index++;
            frontier_attribute->selector ^= 1;
        }
        if (enactor_stats->retval = work_progress -> GetQueueLength(
            frontier_attribute->queue_index,
            frontier_attribute->queue_length,
            false,
            stream,
            true)) return;

        //cudaStreamSynchronize(stream);
        //util::cpu_mt::PrintGPUArray<SizeT, VertexId>("keys2",
        //    frontier_queue -> keys[frontier_attribute -> selector].GetPointer(util::DEVICE),
        //    frontier_attribute -> queue_length,
        //    thread_num, enactor_stats -> iteration, peer_, stream);
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
    { // do nothing here
    }

    /*
     * @brief Routine to combine received data to local data.
     *
     * @tparam NUM_VERTEX_ASSOCIATES
     * @tparam NUM_VALUE__ASSOCIATES
     *
     * @param[in] grid_size
     * @param[in] block_size
     * @param[in] shared_size
     * @param[in] stream
     * @param[in] num_elements
     * @param[in] keys_in
     * @param[in] keys_out
     * @param[in] array_size
     * @param[in] array
     * @param[in] data_slice
     *
     */
    template <
        int NUM_VERTEX_ASSOCIATES, 
        int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
        Enactor        *enactor,
        cudaStream_t    stream,
        VertexId        iteration,
        int             peer_,
        SizeT           received_length,
        SizeT           num_elements,
        util::Array1D<SizeT, SizeT    > &out_length,
        util::Array1D<SizeT, VertexId > &keys_in,
        util::Array1D<SizeT, VertexId > &vertex_associate_in,
        util::Array1D<SizeT, Value    > &value__associate_in,
        util::Array1D<SizeT, VertexId > &keys_out,
        util::Array1D<SizeT, VertexId*> &vertex_associate_orgs,
        util::Array1D<SizeT, Value   *> &value__associate_orgs,
        DataSlice      *h_data_slice,
        EnactorStats<SizeT> *enactor_stats)
    {
        bool over_sized = false;
        if (enactor -> problem -> unified_receive)
        {
            if (enactor_stats -> retval = Check_Size<SizeT, VertexId>(
                enactor -> size_check, "incoming_queue",
                num_elements + received_length,
                &keys_out, over_sized, h_data_slice -> gpu_idx, iteration, peer_, true))
                return;
            received_length += num_elements;
        } else {
            if (enactor_stats -> retval = Check_Size<SizeT, VertexId>(
                enactor -> size_check, "incomping_queue",
                num_elements,
                &keys_out, over_sized, h_data_slice -> gpu_idx, iteration, peer_))
                return;
            out_length[peer_] =0;
            out_length.Move(util::HOST, util::DEVICE, 1, peer_, stream);
        }

        int num_blocks = num_elements / AdvanceKernelPolicy::THREADS / 2+ 1;
        if (num_blocks > 240) num_blocks = 240;
        Expand_Incoming_Kernel
            <AdvanceKernelPolicy, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<num_blocks, AdvanceKernelPolicy::THREADS, 0, stream>>>
            (h_data_slice -> gpu_idx,
            iteration,
            num_elements,
            keys_in.GetPointer(util::DEVICE),
            vertex_associate_in.GetPointer(util::DEVICE),
            value__associate_in.GetPointer(util::DEVICE),
            out_length.GetPointer(util::DEVICE) + ((enactor -> problem -> unified_receive) ? 0: peer_),
            keys_out.GetPointer(util::DEVICE),
            h_data_slice -> preds.GetPointer(util::DEVICE),
            h_data_slice -> distances.GetPointer(util::DEVICE),
            h_data_slice -> labels.GetPointer(util::DEVICE));

        if (!enactor -> problem -> unified_receive)
            out_length.Move(util::DEVICE, util::HOST, 1, peer_, stream);
        else out_length.Move(util::DEVICE, util::HOST, 1, 0, stream);
    }

    /*
     * @brief Compute output length of the resulted frontiers
     *
     * @param[in] frontier_attribute Pointer to the frontier attribute.
     * @param[in] d_offsets Pointer to the offsets.
     * @param[in] d_indices Pointer to the indices.
     * @param[in] d_in_key_queue Pointer to the input mapping queue.
     * @param[in] partitioned_scanned_edges Pointer to the scanned edges.
     * @param[in] max_in Maximum input queue size.
     * @param[in] max_out Maximum output queue size.
     * @param[in] context CudaContext for ModernGPU API.
     * @param[in] stream CUDA stream.
     * @param[in] ADVANCE_TYPE Advance kernel mode.
     * @param[in] express Whether or not enable express mode.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    static cudaError_t Compute_OutputLength(
        Enactor                        *enactor,
        FrontierAttribute<SizeT>       *frontier_attribute,
        SizeT                          *d_offsets,
        VertexId                       *d_indices,
        SizeT                          *d_inv_offsets,
        VertexId                       *d_inv_indices,
        VertexId                       *d_in_key_queue,
        util::Array1D<SizeT, SizeT>    *partitioned_scanned_edges,
        SizeT                          max_in,
        SizeT                          max_out,
        CudaContext                    &context,
        cudaStream_t                   stream,
        gunrock::oprtr::advance::TYPE  ADVANCE_TYPE,
        bool                           express = false,
        bool                           in_inv = false,
        bool                           out_inv = false)
    {
        cudaError_t retval = cudaSuccess;
        bool over_sized = false;

        if (!enactor -> size_check &&
            (!gunrock::oprtr::advance::hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>()))
        {
            frontier_attribute -> output_length[0] = 0;
            return retval;
        } else {
            if (retval = Check_Size<SizeT, SizeT> (
                enactor -> size_check, "scanned_edges",
                frontier_attribute->queue_length + 2,
                partitioned_scanned_edges, over_sized, -1, -1, -1, false))
                return retval;
            retval = gunrock::oprtr::advance::ComputeOutputLength
                <AdvanceKernelPolicy, Problem, Functor,
                gunrock::oprtr::advance::V2V>(
                frontier_attribute,
                d_offsets,
                d_indices,
                d_inv_offsets,
                d_inv_indices,
                d_in_key_queue,
                partitioned_scanned_edges->GetPointer(util::DEVICE),
                max_in,
                max_out,
                context,
                stream,
                express,
                in_inv,
                out_inv);
            frontier_attribute -> output_length.Move(
                util::DEVICE, util::HOST, 1, 0, stream);
            return retval;
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
        {
            printf("%d\t %d\t %d\t queue_length = %lld, output_length = %lld\n",
                thread_num, iteration, peer_,
                (long long)frontier_queue->keys[selector^1].GetSize(),
                (long long)request_length);
            fflush(stdout);
        }

       if (!enactor -> size_check &&
           (!gunrock::oprtr::advance::hasPreScan<AdvanceKernelPolicy::ADVANCE_MODE>()))
        {
            frontier_attribute -> output_length[0] = 0;
            return;
        } else if (!gunrock::oprtr::advance::isFused<AdvanceKernelPolicy::ADVANCE_MODE>())
        {
            if (enactor_stats->retval =
                Check_Size</*true,*/ SizeT, VertexId > (
                    true, "queue3", request_length,
                    &frontier_queue->keys  [selector^1],
                    over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor_stats->retval =
                Check_Size</*true,*/ SizeT, VertexId > (
                    true, "queue3", graph_slice->nodes * 1.2 + 2,
                    &frontier_queue->keys  [selector  ],
                    over_sized, thread_num, iteration, peer_, true )) return;
            if (enactor -> problem -> use_double_buffer)
            {
                if (enactor_stats->retval =
                    Check_Size</*true,*/ SizeT, Value> (
                        true, "queue3", request_length,
                        &frontier_queue->values[selector^1],
                        over_sized, thread_num, iteration, peer_, false)) return;
                if (enactor_stats->retval =
                    Check_Size</*true,*/ SizeT, Value> (
                        true, "queue3", graph_slice->nodes * 1.2 + 2,
                        &frontier_queue->values[selector  ],
                        over_sized, thread_num, iteration, peer_, true )) return;
            }
        } else {
            if (enactor_stats->retval =
                Check_Size</*true,*/ SizeT, VertexId > (
                    true, "queue3", graph_slice -> nodes * 1.2,
                    &frontier_queue->keys  [selector^1],
                    over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor -> problem -> use_double_buffer)
            {
                if (enactor_stats->retval =
                    Check_Size</*true,*/ SizeT, Value> (
                        true, "queue3", graph_slice->nodes * 1.2,
                        &frontier_queue->values[selector^1],
                        over_sized, thread_num, iteration, peer_, false )) return;
            }
        }
    }
};

/**
 * @brief Thread controls.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam Enactor Enactor type we process on.
 *
 * @thread_data_ Thread data.
 */
template <
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
static CUT_THREADPROC SSSPThread(
    void * thread_data_)
{
    typedef typename Enactor::Problem    Problem   ;
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice <VertexId, SizeT, Value>          GraphSliceT;
    typedef SSSPFunctor<VertexId, SizeT, Value, Problem> Functor;

    ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
    Problem      *problem            =  (Problem*)     thread_data -> problem;
    Enactor      *enactor            =  (Enactor*)     thread_data -> enactor;
    int           num_gpus           =   problem     -> num_gpus;
    int           thread_num         =   thread_data -> thread_num;
    int           gpu_idx            =   problem     -> gpu_idx            [thread_num] ;
    DataSlice    *data_slice         =   problem     -> data_slices        [thread_num].GetPointer(util::HOST);
    FrontierAttribute<SizeT>
                 *frontier_attribute = &(enactor     -> frontier_attribute [thread_num * num_gpus]);
    EnactorStats<SizeT> *enactor_stats      = &(enactor     -> enactor_stats      [thread_num * num_gpus]);

    if (enactor_stats[0].retval = util::SetDevice(gpu_idx))
    {
        thread_data -> status = ThreadSlice::Status::Ended;
        CUT_THREADEND;
    }

    thread_data->status = ThreadSlice::Status::Idle;
    while (thread_data -> status != ThreadSlice::Status::ToKill)
    {
        while (thread_data -> status == ThreadSlice::Status::Wait ||
               thread_data -> status == ThreadSlice::Status::Idle)
        {
            sleep(0);
            //std::this_thread::yield();
        }
        if (thread_data -> status == ThreadSlice::Status::ToKill)
            break;

        for (int peer_=0;peer_<num_gpus;peer_++)
        {
            frontier_attribute[peer_].queue_index  = 0;        // Work queue index
            frontier_attribute[peer_].queue_length = peer_==0?thread_data->init_size:0;
            frontier_attribute[peer_].selector     = 0;//frontier_attrbute[peer_].queue_length == 0 ? 0 : 1;
            frontier_attribute[peer_].queue_reset  = true;
            enactor_stats     [peer_].iteration    = 0;
        }

        gunrock::app::Iteration_Loop
            <Enactor, Functor,
            SSSPIteration<AdvanceKernelPolicy, FilterKernelPolicy, Enactor>,
            Problem::MARK_PATHS ? 1:0, 1>
            (thread_data);
        //printf("SSSP_Thread finished\n");fflush(stdout);
        thread_data -> status = ThreadSlice::Status::Idle;
    }

    thread_data -> status = ThreadSlice::Status::Ended;
    CUT_THREADEND;
}

/**
 * @brief Problem enactor class.
 *
 * @tparam _Problem Problem type we process on
 * @tparam _INSTRUMENT Whether or not to collect per-CTA clock-count stats.
 * @tparam _DEBUG Whether or not to enable debug mode.
 * @tparam _SIZE_CHECK Whether or not to enable size check.
 */
template <typename _Problem/*, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK*/>
class SSSPEnactor :
    public EnactorBase<typename _Problem::SizeT/*, _DEBUG, _SIZE_CHECK*/>
{
    ThreadSlice  *thread_slices;// = new ThreadSlice [this->num_gpus];
    CUTThread    *thread_Ids   ;// = new CUTThread   [this->num_gpus];

public:
    _Problem     *problem      ;
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    typedef EnactorBase<SizeT>         BaseEnactor;
    typedef SSSPEnactor<Problem>       Enactor;

    /**
     * @brief BFSEnactor constructor
     */
    SSSPEnactor(
        int   num_gpus   = 1,
        int  *gpu_idx    = NULL,
        bool  instrument = false,
        bool  debug      = false,
        bool  size_check = true) :
        BaseEnactor(
            VERTEX_FRONTIERS, num_gpus, gpu_idx,
            instrument, debug, size_check),
        thread_slices (NULL),
        thread_Ids    (NULL),
        problem       (NULL)
    {
    }

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~SSSPEnactor()
    {
        Release();
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;
        if (thread_slices != NULL)
        {
            for (int gpu = 0; gpu < this->num_gpus; gpu++)
                thread_slices[gpu].status = ThreadSlice::Status::ToKill;
            cutWaitForThreads(thread_Ids, this->num_gpus);
            delete[] thread_Ids   ; thread_Ids    = NULL;
            delete[] thread_slices; thread_slices = NULL;
        }
        if (retval = BaseEnactor::Release()) return retval;
        problem = NULL;
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /** @} */

    /**
     * @brief Initialize the problem.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t InitSSSP(
        ContextPtr  *context,
        Problem     *problem,
        int         max_grid_size = 0)
    {
        cudaError_t retval = cudaSuccess;

        // Lazy initialization
        if (retval = BaseEnactor::Init(
            //problem,
            max_grid_size,
            AdvanceKernelPolicy::CTA_OCCUPANCY,
            FilterKernelPolicy::CTA_OCCUPANCY))
            return retval;

        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus];

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            thread_slices[gpu].thread_num    = gpu;
            thread_slices[gpu].problem       = (void*)problem;
            thread_slices[gpu].enactor       = (void*)this;
            thread_slices[gpu].context       = &(context[gpu*this->num_gpus]);
            thread_slices[gpu].status        = ThreadSlice::Status::Inited;
            thread_slices[gpu].thread_Id     = cutStartThread(
                    (CUT_THREADROUTINE)&(SSSPThread<
                        AdvanceKernelPolicy, FilterKernelPolicy,
                        SSSPEnactor<Problem> >),
                    (void*)&(thread_slices[gpu]));
            thread_Ids[gpu] = thread_slices[gpu].thread_Id;
        }

        for (int gpu=0; gpu < this->num_gpus; gpu++)
        {
            while (thread_slices[gpu].status != ThreadSlice::Status::Idle)
            {
                sleep(0);
                //std::this_thread::yield();
            }
        }
        return retval;
    }

    /**
     * @brief Reset enactor
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Reset()
    {
        cudaError_t retval = cudaSuccess;
        if (retval = BaseEnactor::Reset())
            return retval;
        for (int gpu=0; gpu < this -> num_gpus; gpu++)
            thread_slices[gpu].status = ThreadSlice::Status::Wait;
        return retval;
    }

    /** @} */

    /**
     * @brief Enacts a SSSP computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     *
     * @param[in] src Source node to start primitive.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy>
    cudaError_t EnactSSSP(
        VertexId src)
    {
        clock_t      start_time = clock();
        cudaError_t  retval     = cudaSuccess;

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            if ((this->num_gpus ==1) || (gpu==this->problem->partition_tables[0][src]))
                 thread_slices[gpu].init_size=1;
            else thread_slices[gpu].init_size=0;
            this->frontier_attribute[gpu*this->num_gpus].queue_length
                = thread_slices[gpu].init_size;
        }

        for (int gpu=0; gpu< this->num_gpus; gpu++)
        {
            thread_slices[gpu].status = ThreadSlice::Status::Running;
        }
        for (int gpu=0; gpu< this->num_gpus; gpu++)
        {
            while (thread_slices[gpu].status != ThreadSlice::Status::Idle)
            {
                sleep(0);
                //std::this_thread::yield();
            }
        }

        for (int gpu=0; gpu<this->num_gpus * this -> num_gpus;gpu++)
        if (this->enactor_stats[gpu].retval!=cudaSuccess)
        {
            retval=this->enactor_stats[gpu].retval;
            return retval;
        }

        if (this -> debug) printf("\nGPU SSSP Done.\n");
        return retval;
    }

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
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
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        8,                                  // MIN_CTA_OCCUPANCY
        7,                                  // LOG_THREADS
        10,                                 // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        1,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::TWC_FORWARD>
    TWC_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*1024,                            // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB>
    LB_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*1024,                            // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_CULL>
    LB_CULL_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*1024,                            // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
    LB_LIGHT_AdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        9,                                  // LOG_BLOCKS
        32*1024,                            // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT_CULL>
    LB_LIGHT_CULL_AdvanceKernelPolicy;

    template <typename Dummy, gunrock::oprtr::advance::MODE A_MODE>
    struct MODE_SWITCH{};

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactSSSP<LB_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitSSSP <LB_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::TWC_FORWARD>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactSSSP<TWC_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitSSSP <TWC_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_LIGHT>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactSSSP<LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitSSSP <LB_LIGHT_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_CULL>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactSSSP<LB_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitSSSP <LB_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    template <typename Dummy>
    struct MODE_SWITCH<Dummy, gunrock::oprtr::advance::LB_LIGHT_CULL>
    {
        static cudaError_t Enact(Enactor &enactor, VertexId src)
        {
            return enactor.EnactSSSP<LB_LIGHT_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(src);
        }
        static cudaError_t Init(Enactor &enactor, ContextPtr *context, Problem *problem, int max_grid_size = 0)
        {
            return enactor.InitSSSP <LB_LIGHT_CULL_AdvanceKernelPolicy, FilterKernelPolicy>(
                context, problem, max_grid_size);
        }
    };

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief SSSP Enact kernel entry.
     *
     * @param[in] src Source node to start primitive.
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    //template <typename SSSPProblem>
    cudaError_t Enact(
        VertexId     src,
        std::string traversal_mode = "LB")
    {
        if (this -> min_sm_version >= 300)
        {
            if (traversal_mode == "LB")
                return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>
                    ::Enact(*this, src);
            else if (traversal_mode == "TWC")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_CULL>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_LIGHT")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>
                    ::Enact(*this, src);
            else if (traversal_mode == "LB_LIGHT_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT_CULL>
                    ::Enact(*this, src);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /**
     * @brief SSSP Enact kernel entry.
     *
     * @param[in] context CudaContext pointer for ModernGPU API.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Maximum grid size for kernel calls.
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        ContextPtr   *context,
        Problem      *problem,
        int          max_grid_size = 0,
        std::string  traversal_mode = "LB")
    {
        if (this -> min_sm_version >= 300)
        {
            if (traversal_mode == "LB")
                return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "TWC")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::TWC_FORWARD>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_CULL>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_LIGHT")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT>
                    ::Init(*this, context, problem, max_grid_size);
            else if (traversal_mode == "LB_LIGHT_CULL")
                 return MODE_SWITCH<SizeT, gunrock::oprtr::advance::LB_LIGHT_CULL>
                    ::Init(*this, context, problem, max_grid_size);
            else printf("Traversal_mode %s is undefined for SSSP\n", traversal_mode.c_str());
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all archs
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
