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

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bfs/bfs_problem.cuh>
#include <gunrock/app/bfs/bfs_functor.cuh>

#include <moderngpu.cuh>

namespace gunrock {
namespace app {
namespace bfs {

template <
typename Problem, bool INSTRUMENT, bool DEBUG, bool SIZE_CHECK> class Enactor;

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
__global__ void Expand_Incoming_BFS (
    const SizeT            num_elements,
    const VertexId*  const keys_in,
          VertexId*        keys_out,
    const size_t           array_size,
          char*            array)
{
    extern __shared__ char s_array[];
    const SizeT STRIDE = gridDim.x * blockDim.x;
    size_t offset = 0;
    VertexId** s_vertex_associate_in  = (VertexId**)&(s_array[offset]);
    offset += sizeof(VertexId*) * NUM_VERTEX_ASSOCIATES;
    offset += sizeof(Value*   ) * NUM_VALUE__ASSOCIATES;
    VertexId** s_vertex_associate_org = (VertexId**)&(s_array[offset]);
    SizeT x = threadIdx.x;
    while (x < array_size)
    {
        s_array[x] = array[x];
        x += blockDim.x;
    }
    __syncthreads();

    VertexId key,t;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    while (x<num_elements)
    {
        key = keys_in[x];
        t   = s_vertex_associate_in[0][x];

        if (atomicCAS(s_vertex_associate_org[0]+key, -1, t)!= -1)
        {
           if (atomicMin(s_vertex_associate_org[0]+key, t)<=t)
           {
               keys_out[x]=-1;
               x+=STRIDE;
               continue;
           }
        }
        keys_out[x]=key;
        if (NUM_VERTEX_ASSOCIATES == 2 && s_vertex_associate_org[0][key] == t)
            s_vertex_associate_org[1][key]=s_vertex_associate_in[1][x];
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
template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename Enactor>
struct BFSIteration : public IterationBase <
    AdvanceKernelPolicy, FilterKernelPolicy, Enactor,
    true, false, false, true, Enactor::Problem::MARK_PREDECESSORS>
{
    typedef typename Enactor::SizeT      SizeT     ;
    typedef typename Enactor::Value      Value     ;
    typedef typename Enactor::VertexId   VertexId  ;
    typedef typename Enactor::Problem    Problem ;
    typedef typename Problem::DataSlice  DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value> GraphSlice;
    typedef BFSFunctor<VertexId, SizeT, Value, Problem> BfsFunctor;

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
        int                            thread_num,
        int                            peer_,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        util::Array1D<SizeT, SizeT>   *scanned_edges,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        DataSlice                     *data_slice,
        DataSlice                     *d_data_slice,
        GraphSlice                    *graph_slice,
        util::CtaWorkProgressLifetime *work_progress,
        ContextPtr                     context,
        cudaStream_t                   stream)
    {
        if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Advance begin",thread_num, enactor_stats->iteration, peer_);
        if (TO_TRACK)
        {
            util::MemsetKernel<<<256, 256, 0, stream>>>(
                frontier_queue -> keys[frontier_attribute -> selector^1].GetPointer(util::DEVICE),
                (VertexId)-2,
                frontier_queue -> keys[frontier_attribute -> selector^1].GetSize());
            util::Check_Exist<<<256, 256, 0, stream>>>(
                frontier_attribute -> queue_length,
                data_slice->gpu_idx, 2, enactor_stats -> iteration,
                frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE));       
        }
        frontier_attribute->queue_reset = true;
        enactor_stats     ->nodes_queued[0] += frontier_attribute->queue_length;
        // Edge Map
        gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, BfsFunctor>(
            enactor_stats[0],
            frontier_attribute[0],
            d_data_slice,
            (VertexId*)NULL,
            (bool*    )NULL,
            (bool*    )NULL,
            scanned_edges ->GetPointer(util::DEVICE),
            frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),
            (VertexId*)NULL,
            frontier_queue->values[frontier_attribute->selector^1].GetPointer(util::DEVICE),
            graph_slice->row_offsets   .GetPointer(util::DEVICE),
            graph_slice->column_indices.GetPointer(util::DEVICE),
            (SizeT*   )NULL,
            (VertexId*)NULL,
            graph_slice->nodes,
            graph_slice->edges,
            work_progress[0],
            context[0],
            stream,
            gunrock::oprtr::advance::V2V,
            false,
            false,
            false);

        // Only need to reset queue for once
        if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Advance end", thread_num, enactor_stats->iteration, peer_);
        frontier_attribute -> queue_reset = false;
        frontier_attribute -> queue_index++;
        frontier_attribute -> selector ^= 1;
        enactor_stats      -> AccumulateEdges(
            work_progress  -> GetQueueLengthPointer<unsigned int,SizeT>(frontier_attribute->queue_index), stream);

        if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Filter begin", thread_num, enactor_stats->iteration, peer_);
        if (TO_TRACK)
        {
            util::Check_Value<<<1,1,0,stream>>>(
                work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
                    frontier_attribute->queue_index),
                data_slice->gpu_idx, 5, enactor_stats -> iteration);
            util::Check_Exist_<<<256, 256, 0, stream>>>(
                work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
                    frontier_attribute->queue_index),
                data_slice->gpu_idx, 3, enactor_stats -> iteration,
                frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE));
            util::MemsetCASKernel<<<256, 256, 0, stream>>>(
                frontier_queue -> keys[ frontier_attribute->selector].GetPointer(util::DEVICE),
                -2, -1,
                work_progress -> template GetQueueLengthPointer<unsigned int, SizeT>(
                    frontier_attribute->queue_index));
        }

        // Filter
        gunrock::oprtr::filter::Kernel<FilterKernelPolicy, Problem, BfsFunctor>
        <<<enactor_stats->filter_grid_size, FilterKernelPolicy::THREADS, 0, stream>>>(
            enactor_stats->iteration+1,
            frontier_attribute->queue_reset,
            frontier_attribute->queue_index,
            frontier_attribute->queue_length,
            frontier_queue->keys  [frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue->values[frontier_attribute->selector  ].GetPointer(util::DEVICE),
            frontier_queue->keys  [frontier_attribute->selector^1].GetPointer(util::DEVICE),
            d_data_slice,
            data_slice->visited_mask.GetPointer(util::DEVICE),
            work_progress[0],
            frontier_queue->keys  [frontier_attribute->selector  ].GetSize(),
            frontier_queue->keys  [frontier_attribute->selector^1].GetSize(),
            enactor_stats->filter_kernel_stats);
        if (Enactor::DEBUG && (enactor_stats->retval = util::GRError("filter_forward::Kernel failed", __FILE__, __LINE__))) return;
        if (Enactor::DEBUG) util::cpu_mt::PrintMessage("Filter end.", thread_num, enactor_stats->iteration);
        frontier_attribute->queue_index++;
        frontier_attribute->selector ^= 1;
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
     */
    template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
    static void Expand_Incoming(
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
        Check_Size<Enactor::SIZE_CHECK, SizeT, VertexId>(
            "queue1", num_elements, keys_out, over_sized, -1, -1, -1);
        Expand_Incoming_BFS
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
        bool                            in_inv = false,
        bool                            out_inv = false)
    {
        cudaError_t retval = cudaSuccess;
        //printf("SIZE_CHECK = %s\n", Enactor::SIZE_CHECK ? "true" : "false");
        bool over_sized = false;
        if (!Enactor::SIZE_CHECK &&
            (AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::TWC_FORWARD ||
             AdvanceKernelPolicy::ADVANCE_MODE == oprtr::advance::TWC_BACKWARD))
        {
            return retval;

        } else {
            //printf("Size check runs\n");
            if (retval = Check_Size<Enactor::SIZE_CHECK, SizeT, SizeT> (
                "scanned_edges", frontier_attribute->queue_length,
                partitioned_scanned_edges, over_sized, -1, -1, -1, false))
                return retval;
            retval = gunrock::oprtr::advance::ComputeOutputLength
                <AdvanceKernelPolicy, Problem, BfsFunctor>(
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
        int                            thread_num,
        int                            peer_,
        SizeT                          request_length,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        FrontierAttribute<SizeT>      *frontier_attribute,
        EnactorStats                  *enactor_stats,
        GraphSlice                    *graph_slice)
    {
        bool over_sized = false;
        int  selector   = frontier_attribute->selector;
        int  iteration  = enactor_stats -> iteration;

        if (Enactor::DEBUG)
        {
            printf("%d\t %d\t %d\t queue_length = %lld, output_length = %lld\n",
                thread_num, iteration, peer_,
                (long long)frontier_queue->keys[selector^1].GetSize(),
                (long long)request_length);
            fflush(stdout);
        }

        if (enactor_stats->retval =
            Check_Size<true, SizeT, VertexId > ("queue3", request_length, &frontier_queue->keys  [selector^1], over_sized, thread_num, iteration, peer_, false)) return;
        if (enactor_stats->retval =
            Check_Size<true, SizeT, VertexId > ("queue3", graph_slice->nodes+2, &frontier_queue->keys  [selector  ], over_sized, thread_num, iteration, peer_, true )) return;
        if (Problem::USE_DOUBLE_BUFFER)
        {
            if (enactor_stats->retval =
                Check_Size<true, SizeT, Value> ("queue3", request_length, &frontier_queue->values[selector^1], over_sized, thread_num, iteration, peer_, false)) return;
            if (enactor_stats->retval =
                Check_Size<true, SizeT, Value> ("queue3", graph_slice->nodes+2, &frontier_queue->values[selector  ], over_sized, thread_num, iteration, peer_, true )) return;
        }
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
        GraphSlice                    *graph_slice,
        DataSlice                     *data_slice,
        FrontierAttribute<SizeT>
                                      *frontier_attribute,
        util::DoubleBuffer<SizeT, VertexId, Value>
                                      *frontier_queue,
        SizeT                          num_elements,
        cudaStream_t                   stream)
    {
        return ;
    }
};

/**
 * @brief Thread controls.
 *
 * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
 * @tparam FilterKernelPolicy Kernel policy for filter operator.
 * @tparam BfsEnactor Enactor type we process on.
 *
 * @thread_data_ Thread data.
 */
template<
    typename AdvanceKernelPolicy,
    typename FilterKernelPolicy,
    typename BfsEnactor>
static CUT_THREADPROC BFSThread(
    void * thread_data_)
{
    typedef typename BfsEnactor::Problem    Problem;
    typedef typename BfsEnactor::SizeT      SizeT     ;
    typedef typename BfsEnactor::VertexId   VertexId  ;
    typedef typename BfsEnactor::Value      Value     ;
    typedef typename Problem::DataSlice     DataSlice ;
    typedef GraphSlice<SizeT, VertexId, Value>    GraphSlice;
    typedef BFSFunctor<VertexId, SizeT, Value, Problem> BfsFunctor;
    ThreadSlice  *thread_data        =  (ThreadSlice*) thread_data_;
    Problem      *problem            =  (Problem*)     thread_data->problem;
    BfsEnactor   *enactor            =  (BfsEnactor*)  thread_data->enactor;
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
        thread_data->stats=3;

        for (int peer=0;peer<num_gpus;peer++)
        {
            frontier_attribute[peer].queue_index    = 0;        // Work queue index
            frontier_attribute[peer].queue_length   = peer==0?thread_data -> init_size:0;
            frontier_attribute[peer].selector       = 0; //frontier_attribute[peer].queue_length ==0 ? 0 : 1;
            frontier_attribute[peer].queue_reset    = true;
            enactor_stats     [peer].iteration      = 0;
        }

        gunrock::app::Iteration_Loop
            <Problem::MARK_PREDECESSORS?2:1,0, BfsEnactor, BfsFunctor, BFSIteration<AdvanceKernelPolicy, FilterKernelPolicy, BfsEnactor> > (thread_data);
        // printf("BFS_Thread finished\n");fflush(stdout);
    } while(0);

    thread_data->stats = 4;
    CUT_THREADEND;
}

/**
 * @brief Problem enactor class.
 *
 * @tparam _Problem Problem type we process on.
 * @tparam _INSTRUMENT Whether or not to collect per-CTA clock-count stats.
 * @tparam _DEBUG Whether or not to enable debug mode.
 * @tparam _SIZE_CHECK Whether or not to enable size check.
 */
template <typename _Problem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK>
class BFSEnactor :
    public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>
{
    _Problem     *problem      ;
    ThreadSlice  *thread_slices;
    CUTThread    *thread_Ids   ;

public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    static const bool INSTRUMENT = _INSTRUMENT;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;

    // Methods

    /**
     * @brief BFSEnactor constructor
     */
    BFSEnactor(int num_gpus = 1, int* gpu_idx = NULL) :
        EnactorBase<SizeT, _DEBUG, _SIZE_CHECK>(VERTEX_FRONTIERS, num_gpus, gpu_idx)//,
    {
        thread_slices = NULL;
        thread_Ids    = NULL;
        problem       = NULL;
    }

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~BFSEnactor()
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

    /**
     * @ brief Obtain statistics about the last primitive enacted.
     *
     * @ tparam VertexId Vertex identifier type.
     *
     * @ param[out] total_queued Total queued elements in kernel running.
     * @ param[out] search_depth Search depth (number of super-steps).
     * @ param[out] avg_duty Average kernel duty (kernel time/kernel lifetime).
     * spaces between @ and name are to eliminate doxygen warnings
     */
    /*template <typename VertexId>
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

            for (int peer=0; peer< this->num_gpus; peer++)
            {
                EnactorStats *enactor_stats_ = this->enactor_stats + gpu * this->num_gpus + peer;
                enactor_stats_ -> total_queued.Move(util::DEVICE, util::HOST);
                total_queued += enactor_stats_ -> total_queued[0];
                if (enactor_stats_ -> iteration > search_depth)
                    search_depth = enactor_stats_ -> iteration;
                total_lifetimes += enactor_stats_ -> total_lifetimes;
                total_runtimes  += enactor_stats_ -> total_runtimes;
            }
        }
        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }*/

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
    cudaError_t InitBFS(
        ContextPtr *context,
        Problem    *problem,
        int        max_grid_size = 0,
        bool       size_check    = true)
    {
        cudaError_t retval = cudaSuccess;

        // Lazy initialization
        if (retval = EnactorBase<SizeT, DEBUG, SIZE_CHECK>::Init(problem,
                                       max_grid_size,
                                       AdvanceKernelPolicy::CTA_OCCUPANCY,
                                       FilterKernelPolicy::CTA_OCCUPANCY)) return retval;

        this->problem = problem;
        thread_slices = new ThreadSlice [this->num_gpus];
        thread_Ids    = new CUTThread   [this->num_gpus];

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            if (retval = util::SetDevice(this->gpu_idx[gpu])) break;
            if (Problem::ENABLE_IDEMPOTENCE) {
                SizeT bytes = (problem->graph_slices[gpu]->nodes + 8 - 1) / 8;
                cudaChannelFormatDesc   bitmask_desc = cudaCreateChannelDesc<char>();
                gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref.channelDesc = bitmask_desc;
                if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::filter::BitmaskTex<unsigned char>::ref,
                    problem->data_slices[gpu]->visited_mask.GetPointer(util::DEVICE),
                    bytes),
                    "BFSEnactor cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;
            }
        }

        for (int gpu=0;gpu<this->num_gpus;gpu++)
        {
            thread_slices[gpu].thread_num    = gpu;
            thread_slices[gpu].problem       = (void*)problem;
            thread_slices[gpu].enactor       = (void*)this;
            thread_slices[gpu].context       = &(context[gpu*this->num_gpus]);
            thread_slices[gpu].stats         = -1;
            thread_slices[gpu].thread_Id = cutStartThread(
                (CUT_THREADROUTINE)&(BFSThread<
                    AdvanceKernelPolicy,FilterKernelPolicy,
                    BFSEnactor<Problem, INSTRUMENT, DEBUG, SIZE_CHECK> >),
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
        return EnactorBase<SizeT, DEBUG, SIZE_CHECK>::Reset();
    }

    /** @} */

    /**
     * @brief Enacts a breadth-first search computing on the specified graph.
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
    cudaError_t EnactBFS(
        VertexId    src)
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

        if (DEBUG) printf("\nGPU BFS Done.\n");
        return retval;
    }

    typedef gunrock::oprtr::filter::KernelPolicy<
        Problem,                            // Problem data type
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
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        INSTRUMENT,                         // INSTRUMENT
        8,                                  // MIN_CTA_OCCUPANCY
        7,                                  // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::TWC_FORWARD>
    ForwardAdvanceKernelPolicy_IDEM;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        INSTRUMENT,                         // INSTRUMENT
        1,                                  // MIN_CTA_OCCUPANCY
        7,                                  // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::TWC_FORWARD>
    ForwardAdvanceKernelPolicy;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        INSTRUMENT,                         // INSTRUMENT
        8,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
    LBAdvanceKernelPolicy_IDEM;

    typedef gunrock::oprtr::advance::KernelPolicy<
        Problem,                            // Problem data type
        300,                                // CUDA_ARCH
        INSTRUMENT,                         // INSTRUMENT
        1,                                  // MIN_CTA_OCCUPANCY
        10,                                 // LOG_THREADS
        8,                                  // LOG_BLOCKS
        32*128,                             // LIGHT_EDGE_THRESHOLD (used for partitioned advance mode)
        1,                                  // LOG_LOAD_VEC_SIZE
        0,                                  // LOG_LOADS_PER_TILE
        5,                                  // LOG_RAKING_THREADS
        32,                                 // WARP_GATHER_THRESHOLD
        128 * 4,                            // CTA_GATHER_THRESHOLD
        7,                                  // LOG_SCHEDULE_GRANULARITY
        gunrock::oprtr::advance::LB_LIGHT>
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
    cudaError_t Enact(
        VertexId    src,
        int traversal_mode = 0)
    {
        int min_sm_version = -1;
        for (int i = 0; i < this->num_gpus; i++)
        {
            if (min_sm_version == -1 ||
                this->cuda_props[i].device_sm_version < min_sm_version)
            {
                min_sm_version = this->cuda_props[i].device_sm_version;
            }
        }

        if (min_sm_version >= 300)
        {
            if (Problem::ENABLE_IDEMPOTENCE)
            {
                if (traversal_mode == 0)
                {
                    return EnactBFS<     LBAdvanceKernelPolicy_IDEM, FilterKernelPolicy>(src);
                }
                else
                {
                    return EnactBFS<ForwardAdvanceKernelPolicy_IDEM, FilterKernelPolicy>(src);
                }
            }
            else
            {
                if (traversal_mode == 0)
                {
                    return EnactBFS<     LBAdvanceKernelPolicy     , FilterKernelPolicy>(src);
                }
                else
                {
                    return EnactBFS<ForwardAdvanceKernelPolicy     , FilterKernelPolicy>(src);
                }
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;

    }

    /**
     * @brief BFS Enact kernel entry.
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
        ContextPtr  *context,
        Problem     *problem,
        int         max_grid_size  = 0,
        bool        size_check     = true,
        int         traversal_mode = 0)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
        {
            if (min_sm_version == -1 ||
                this->cuda_props[i].device_sm_version < min_sm_version)
            {
                min_sm_version = this->cuda_props[i].device_sm_version;
            }
        }

        if (min_sm_version >= 300)
        {
            if (Problem::ENABLE_IDEMPOTENCE)
            {
                if (traversal_mode == 0)
                {
                    return InitBFS<     LBAdvanceKernelPolicy_IDEM, FilterKernelPolicy>(
                            context, problem, max_grid_size, size_check);
                }
                else
                {
                    return InitBFS<ForwardAdvanceKernelPolicy_IDEM, FilterKernelPolicy>(
                            context, problem, max_grid_size, size_check);
                }
            }
            else
            {
                if (traversal_mode == 0)
                {
                    return InitBFS<     LBAdvanceKernelPolicy     , FilterKernelPolicy>(
                            context, problem, max_grid_size, size_check);
                }
                else
                {
                    return InitBFS<ForwardAdvanceKernelPolicy     , FilterKernelPolicy>(
                            context, problem, max_grid_size, size_check);
                }
            }
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernel policy settings for all archs
        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;

    }

    /** @} */
};

}  // namespace bfs
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
