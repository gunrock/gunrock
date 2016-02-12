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
#include <gunrock/util/device_intrinsics.cuh>

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

/*
 * @brief Expand incoming function.
 *
 * @tparam VertexId
 * @tparam SizeT
 * @tparam Value
 * @tparam NUM_VERTEX_ASSOCIATES
 * @tparam NUM_VALUE__ASSOCIATES
 *
 * @param[in] num_elements
 * @param[in] keys_in
 * @param[in] keys_out
 * @param[in] array_size
 * @param[in] array
 */
template <
    typename VertexId,
    typename SizeT,
    typename Value,
    int      NUM_VERTEX_ASSOCIATES,
    int      NUM_VALUE__ASSOCIATES>
__global__ void Expand_Incoming_SSSP (
    const SizeT            num_elements,
    const VertexId* const  keys_in,
          VertexId*        keys_out,
    const size_t           array_size,
          char*            array)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    size_t      offset                = 0;
    VertexId** s_vertex_associate_in  = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
    Value**    s_value__associate_in  = (Value**   )&(s_array[offset]);
    offset+=sizeof(Value*   ) * NUM_VALUE__ASSOCIATES;
    VertexId** s_vertex_associate_org = (VertexId**)&(s_array[offset]);
    offset+=sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
    Value**    s_value__associate_org = (Value**   )&(s_array[offset]);
    SizeT x = threadIdx.x;
    if (x < array_size)
    {
        s_array[x]=array[x];
        x+=blockDim.x;
    }
    __syncthreads();

    x = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;

    while (x<num_elements)
    {
        VertexId key=keys_in[x];
        Value t=s_value__associate_in[0][x];

        Value old_value = atomicMin(s_value__associate_org[0]+key, t);
        if (old_value<=t)
        {
            keys_out[x]=-1;
            x+=STRIDE;
            continue;
        }

        keys_out[x]=key;

        #pragma unroll
        for (SizeT i=1;i<NUM_VALUE__ASSOCIATES;i++)
            s_value__associate_org[i][key]=s_value__associate_in[i][x];
        #pragma unroll
        for (SizeT i=0;i<NUM_VERTEX_ASSOCIATES;i++)
            s_vertex_associate_org[i][key]=s_vertex_associate_in[i][x];
        x+=STRIDE;
    }
}

/*
 * @brief Iteration structure derived from IterationBase.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam Enactor Enactor we process on.
 */
template <
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
struct SSSPIteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    true , // HAS_SUBQ
    false, // HAS_FULLQ
    false, // BACKWARD
    true , // FORWARD
    Enactor::Problem::MARK_PATHS> // UPDATE_PREDECESSORS
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
    typedef IterationBase <
        AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
        true, false, false, true, Enactor::Problem::MARK_PATHS> 
                                        BaseIteration;

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
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        frontier_attribute->queue_reset = true;
        enactor_stats     ->nodes_queued[0] += frontier_attribute -> queue_length;

        if (enactor -> debug) 
            util::cpu_mt::PrintMessage("Advance begin",
                thread_num, enactor_stats->iteration, peer_);
        // Edge Map
        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, Functor>(
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges ->GetPointer(util::DEVICE),
            frontier_queue->keys[frontier_attribute->selector]  .GetPointer(util::DEVICE), // d_in_queue
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE), // d_out_queue
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
            gunrock::oprtr::advance::V2V,
            false,
            false,
            false);
        if (enactor -> debug) 
            util::cpu_mt::PrintMessage("Advance end", 
                thread_num, enactor_stats->iteration, peer_);

        frontier_attribute -> queue_reset = false;
        frontier_attribute -> queue_index++;
        frontier_attribute -> selector ^= 1;
        enactor_stats      -> AccumulateEdges(
            work_progress  -> GetQueueLengthPointer<unsigned int, SizeT>(
                frontier_attribute -> queue_index), stream);

        if (enactor -> debug)
            util::cpu_mt::PrintMessage("Filter begin",
                thread_num, enactor_stats -> iteration, peer_);
        //Vertex Map
        gunrock::oprtr::filter::LaunchKernel
            <FilterKernelPolicy, Problem, Functor>(
            enactor_stats->filter_grid_size, 
            FilterKernelPolicy::THREADS, 
            (size_t)0, 
            stream,
            enactor_stats->iteration+1,
            frontier_attribute->queue_reset,
            frontier_attribute->queue_index,
            frontier_attribute->queue_length,
            frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),      // d_in_queue
            (Value*) NULL,                                                                  // d_pred_in_queue
            frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),     // d_out_queue
            d_data_slice,
            (unsigned char*)NULL,
            work_progress[0],
            frontier_queue->keys[frontier_attribute->selector  ].GetSize(), // max_in_queue
            frontier_queue->keys[frontier_attribute->selector^1].GetSize(), // max_out_queue
            enactor_stats->filter_kernel_stats);
        if (enactor -> debug && (enactor_stats->retval = 
            util::GRError("filter_forward::Kernel failed", __FILE__, __LINE__))) return;
        if (enactor -> debug) 
            util::cpu_mt::PrintMessage("Filter end.", 
                thread_num, enactor_stats->iteration, peer_);

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
            util::MemsetKernel<<<128, 128, 0, stream>>> (
                data_slice -> sssp_marker.GetPointer(util::DEVICE),
                (int)0, graph_slice->nodes);
        } else { // #gpus != 1
            frontier_attribute->queue_index++;
            frontier_attribute->selector ^= 1;
        }
    }

    /*
     * @brief Expand incoming function.
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
    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
        Enactor        *enactor,
        int             grid_size,
        int             block_size,
        size_t          shared_size,
        cudaStream_t    stream,
        SizeT           &num_elements,
        VertexId*       keys_in,
        util::Array1D<SizeT, VertexId>* keys_out,
        const size_t    array_size,
        char*           array,
        DataSlice*      data_slice)
    {
        bool over_sized = false;
        Check_Size<SizeT, VertexId>(
            enactor -> size_check, "queue1", num_elements, keys_out, over_sized, -1, -1, -1);
        Expand_Incoming_SSSP
            <VertexId, SizeT, Value, NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES>
            <<<grid_size, block_size, shared_size, stream>>> (
            num_elements,
            keys_in,
            keys_out->GetPointer(util::DEVICE),
            array_size,
            array);
    }

    /*
     * @brief Compute output queue length function.
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
            (AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::TWC_FORWARD ||
             AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::TWC_BACKWARD))
        {
            return retval;
        } else {
            if (retval = Check_Size<SizeT, SizeT> (
                enactor -> size_check, "scanned_edges", 
                frontier_attribute->queue_length, 
                partitioned_scanned_edges, over_sized, -1, -1, -1, false)) 
                return retval;
            retval = gunrock::oprtr::advance::ComputeOutputLength
                <AdvanceKernelPolicy, Problem, Functor>(
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
                ADVANCE_TYPE,
                express,
                in_inv,
                out_inv);
            return retval;
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
    static void Make_Output(
        Enactor                       *enactor,
        int                            thread_num,
        SizeT                          num_elements,
        int                            num_gpus,
        Frontier                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        util::Array1D<SizeT, DataSlice>
                                      *data_slice_,
        GraphSliceT                   *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        util::MemsetKernel<<<128, 128, 0, stream>>> (
            data_slice_[0]->sssp_marker.GetPointer(util::DEVICE), 
            (int)0, graph_slice->nodes);
        BaseIteration::template Make_Output < NUM_VERTEX_ASSOCIATES, NUM_VALUE__ASSOCIATES> (
            enactor, thread_num, num_elements, num_gpus,
            frontier_queue, scanned_edges, frontier_attribute,
            enactor_stats, data_slice_, graph_slice, 
            work_progress, context, stream);
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
        EnactorStats                  *enactor_stats,
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

        if (enactor_stats->retval =
            Check_Size<SizeT, VertexId > (
                true, "queue3", request_length, 
                &frontier_queue->keys  [selector^1], 
                over_sized, thread_num, iteration, peer_, false)) return;
        if (enactor_stats->retval =
            Check_Size<SizeT, VertexId > (
                true, "queue3", graph_slice->nodes+2, 
                &frontier_queue->keys  [selector  ], 
                over_sized, thread_num, iteration, peer_, true )) return;
        if (enactor -> problem -> use_double_buffer)
        {
            if (enactor_stats->retval =
                Check_Size<SizeT, Value> (
                    true, "queue3", request_length, 
                    &frontier_queue->values[selector^1], 
                    over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor_stats->retval =
                Check_Size<SizeT, Value> (
                    true, "queue3", graph_slice->nodes+2, 
                    &frontier_queue->values[selector  ], 
                    over_sized, thread_num, iteration, peer_, true )) return;
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
    EnactorStats *enactor_stats      = &(enactor     -> enactor_stats      [thread_num * num_gpus]);

    do {
        if (enactor_stats[0].retval = util::SetDevice(gpu_idx)) break;
        thread_data->stats = 1;
        while (thread_data->stats != 2) sleep(0);
        thread_data->stats = 3;

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
            Problem::MARK_PATHS? 1:0, 1>
            (thread_data);
        //printf("SSSP_Thread finished\n");fflush(stdout);

    } while(0);

    thread_data->stats=4;
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
    //static const bool INSTRUMENT = _INSTRUMENT;
    //static const bool DEBUG      = _DEBUG;
    //static const bool SIZE_CHECK = _SIZE_CHECK;

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
            instrument, debug, size_check)
    {
        thread_slices = NULL;
        thread_Ids    = NULL;
        problem       = NULL;
    }

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~SSSPEnactor()
    {
        cutWaitForThreads(thread_Ids, this->num_gpus);
        delete[] thread_Ids   ; thread_Ids    = NULL;
        delete[] thread_slices; thread_slices = NULL;
        problem = NULL;
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
     * @param[in] size_check Whether or not to enable size check.
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
        //bool        size_check    = true)
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
            thread_slices[gpu].context       = &(context[gpu*this->num_gpus]);
            thread_slices[gpu].stats         = -1;
            thread_slices[gpu].thread_Id     = cutStartThread(
                    (CUT_THREADROUTINE)&(SSSPThread<
                        AdvanceKernelPolicy, FilterKernelPolicy,
                        SSSPEnactor<Problem> >),
                    (void*)&(thread_slices[gpu]));
            thread_Ids[gpu] = thread_slices[gpu].thread_Id;
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
        return BaseEnactor::Reset();
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
            if (this->enactor_stats[gpu].retval!=cudaSuccess)
            {
                retval=this->enactor_stats[gpu].retval;break;
            }
        } while(0);

        if (this -> debug) printf("\nGPU SSSP Done.\n");
        return retval;
    }

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
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
        //INSTRUMENT,                         // INSTRUMENT
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
    FWDAdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        //INSTRUMENT,                         // INSTRUMENT
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*1024,                            // LIGHT_EDGE_THRESHOLD
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB>
    LBAdvanceKernelPolicy;

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
        int traversal_mode = 0)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
        {
            if (min_sm_version == -1 ||
                this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;
        }

        if (min_sm_version >= 300)
        {
            if (traversal_mode == 0)
                return EnactSSSP< LBAdvanceKernelPolicy, FilterKernelPolicy>(src);
            else
                return EnactSSSP<FWDAdvanceKernelPolicy, FilterKernelPolicy>(src);
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
     * @param[in] size_check Whether or not to enable size check.
     * @param[in] traversal_mode Load-balanced or Dynamic cooperative.
     *
     * \return cudaError_t object Indicates the success of all CUDA calls.
     */
    cudaError_t Init(
        ContextPtr   *context,
        Problem      *problem,
        int          max_grid_size = 0,
        //bool         size_check = true,
        int          traversal_mode = 0)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
        {
            if (min_sm_version == -1 ||
                this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;
        }

        if (min_sm_version >= 300)
        {
            if (traversal_mode == 0)
                return InitSSSP< LBAdvanceKernelPolicy, FilterKernelPolicy>(
                    context, problem, max_grid_size);
            else
                return InitSSSP<FWDAdvanceKernelPolicy, FilterKernelPolicy>(
                    context, problem, max_grid_size);
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
