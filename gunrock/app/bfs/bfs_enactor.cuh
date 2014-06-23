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
        //int           max_grid_size;
        //int           edge_map_grid_size;
        //int           vertex_map_grid_size;
        CUTThread     thread_Id;
        util::cpu_mt::CPUBarrier* cpu_barrier;
        void*         problem;
        void*         enactor;

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

    template <typename VertexId, typename SizeT, bool MARK_PREDECESSORS>
    __global__ void Expand_Incoming (
        const SizeT            num_elements,
        const SizeT            num_associates,
        const SizeT            incoming_offset,
        const VertexId*  const keys_in,
              VertexId*        keys_out,
              VertexId**       associate_in,
              VertexId**       associate_org)
    {
        SizeT x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        if (x>=num_elements) return;
        SizeT x2=incoming_offset+x;
        VertexId key=keys_in[x2];
        VertexId t=associate_in[0][x2];

        if (atomicCAS(associate_org[0]+key, -1, t)== -1)
        {
        } else {
           if (atomicMin(associate_org[0]+key, t)<t)
           {
               keys_out[x]=-1;
               return;
           }
        }
        keys_out[x]=key;
        for (SizeT i=1;i<num_associates;i++)
        {
            associate_org[i][key]=associate_in[i][x2];
        }
    }

    template <typename VertexId, typename SizeT>
    __global__ void Update_Preds (
        const SizeT     num_elements,
        const VertexId* keys,
        const VertexId* org_vertexs,
              VertexId* preds)
    {
        SizeT x = ((blockIdx.y*gridDim.x+blockIdx.x)*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
        if (x>=num_elements) return;
        SizeT t = keys[x];
        preds[t]=org_vertexs[preds[t]];
    }

    bool All_Done(volatile int **dones, cudaError_t *retvals,int num_gpus)
    {
        for (int gpu=0;gpu<num_gpus;gpu++)
        if (retvals[gpu]!=cudaSuccess)
        {
            printf("(CUDA error %d @ GPU %d: %s\n", retvals[gpu], gpu, cudaGetErrorString(retvals[gpu])); fflush(stdout);
            return true;
        }

        for (int gpu=0;gpu<num_gpus;gpu++)
        if (dones[gpu][0]!=0)
        {
            return false;
        }
        return true;
    }

    template<
        bool     INSTRUMENT,
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename BFSProblem>
    static CUT_THREADPROC BFSThread(
        void * thread_data_)
    {
        typedef typename BFSProblem::SizeT      SizeT;
        typedef typename BFSProblem::VertexId   VertexId;
        typedef typename BFSProblem::Value      Value;
        typedef typename BFSProblem::DataSlice  DataSlice;
        typedef typename BFSProblem::GraphSlice GraphSlice;
        //typedef typename BFSEnactor<BFSProblem, INSTRUMENT> BfsEnactor;

        ThreadSlice  *thread_data          = (ThreadSlice *) thread_data_;
        BFSProblem   *problem              = (BFSProblem*) thread_data->problem;
        BFSEnactor<BFSProblem, INSTRUMENT>
                     *enactor              = (BFSEnactor<BFSProblem, INSTRUMENT>*) thread_data->enactor;
        int          thread_num            =   thread_data-> thread_num;
        DataSlice    *data_slice           = &(problem    -> data_slices       [thread_num]);
        GraphSlice   *graph_slice          =   problem    -> graph_slices      [thread_num];
        typename BFSEnactor<BFSProblem, INSTRUMENT>::FrontierAttribute
                     *frontier_attribute   = &(enactor    -> frontier_attribute[thread_num]);
        typename BFSEnactor<BFSProblem, INSTRUMENT>::EnactorStats 
                     *eanctor_stats        = &(enactor    -> enactor_stats     [thread_num]);
        int          num_gpus              =   problem    -> num_gpus;
        util::scan::MultiScan<VertexId,SizeT,true,256,8>*
                     Scaner                = NULL;
        bool         break_clean           = true;
        SizeT*       out_offset            = NULL;
        char*        message               = new char [1024];
        util::Array1D<SizeT, unsigned int> scanned_edges;
        frontier_attribute->queue_index    = 0;        // Work queue index
        frontier_attribute->selector       = 0;
        frontier_attribute->queue_length   = thread_data->init_size; //? 
        frontier_attribute->queue_reset    = true;
        

        if (num_gpus >1)
        {
            Scan = new util::scan::MultiScan<VertexId, SizeT, true, 256, 8>;
            out_offset = new SizeT[num_gpus +1];
        }
        
        scanned_edges.SetName("scanned_edges");
        if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
            if (retval = util::GRError(cudaMalloc(
                                (void**)&d_scanned_edges,
                                graph_slice->edges * sizeof(unsigned int)),
                            "PBFSProblem cudaMalloc d_scanned_edges failed", __FILE__, __LINE__)) return retval;
            }


            
            fflush(stdout);
            /*// Step through BFS iterations
            
            while (done[0] < 0) {

                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, BFSProblem, BfsFunctor>(
                    d_done,
                    enactor_stats,
                    frontier_attribute,
                    data_slice,
                    (VertexId*)NULL,
                    (bool*)NULL,
                    (bool*)NULL,
                    d_scanned_edges,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],              // d_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],            // d_out_queue
                    (VertexId*)NULL,          // d_pred_in_queue
                    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],          // d_pred_out_queue
                    graph_slice->d_row_offsets,
                    graph_slice->d_column_indices,
                    (SizeT*)NULL,
                    (VertexId*)NULL,
                    graph_slice->frontier_elements[frontier_attribute.selector],
                    graph_slice->frontier_elements[frontier_attribute.selector^1],
                    this->work_progress,
                    context,
                    gunrock::oprtr::advance::V2V);*/


                /*gunrock::oprtr::edge_map_forward::Kernel<typename AdvanceKernelPolicy::THREAD_WARP_CTA_FORWARD, BFSProblem, BfsFunctor>
                <<<enactor_stats.advance_grid_size, AdvanceKernelPolicy::THREAD_WARP_CTA_FORWARD::THREADS>>>(
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    enactor_stats.iteration,
                    frontier_attribute.queue_length,
                    d_done,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],              // d_in_queue
                    graph_slice->frontier_queues.d_values[frontier_attribute.selector^1],          // d_pred_out_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],            // d_out_queue
                    graph_slice->d_column_indices,
                    data_slice,
                    this->work_progress,
                    graph_slice->frontier_elements[frontier_attribute.selector],
                    graph_slice->frontier_elements[frontier_attribute.selector^1],
                    enactor_stats.advance_kernel_stats,
                    gunrock::oprtr::advance::V2V);*/
 

                /*
                // Only need to reset queue for once
                if (frontier_attribute.queue_reset)
                    frontier_attribute.queue_reset = false;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "advance::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates 

                frontier_attribute.queue_index++;
                frontier_attribute.selector ^= 1;

                if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                }
                
                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    printf(", %lld", (long long) frontier_attribute.queue_length);
                    //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[frontier_attribute.selector], frontier_attribute.queue_length);
                    //util::DisplayDeviceResults(graph_slice->frontier_queues.d_values[frontier_attribute.selector], frontier_attribute.queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = enactor_stats.advance_kernel_stats.Accumulate(
                        enactor_stats.advance_grid_size,
                        enactor_stats.total_runtimes,
                        enactor_stats.total_lifetimes)) break;
                }

                // Throttle
                if (enactor_stats.iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                        "BFSEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BFSEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;

                // Filter
                gunrock::oprtr::filter::Kernel<FilterKernelPolicy, BFSProblem, BfsFunctor>
                <<<enactor_stats.filter_grid_size, FilterKernelPolicy::THREADS>>>(
                    enactor_stats.iteration+1,
                    frontier_attribute.queue_reset,
                    frontier_attribute.queue_index,
                    enactor_stats.num_gpus,
                    frontier_attribute.queue_length,
                    d_done,
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector],      // d_in_queue
                    graph_slice->frontier_queues.d_values[frontier_attribute.selector],    // d_pred_in_queue
                    graph_slice->frontier_queues.d_keys[frontier_attribute.selector^1],    // d_out_queue
                    data_slice,
                    problem->data_slices[enactor_stats.gpu_id]->d_visited_mask,
                    work_progress,
                    graph_slice->frontier_elements[frontier_attribute.selector],           // max_in_queue
                    graph_slice->frontier_elements[frontier_attribute.selector^1],         // max_out_queue
                    enactor_stats.filter_kernel_stats,
                    ts_bitmask);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates


                frontier_attribute.queue_index++;
                frontier_attribute.selector ^= 1;

                if (AdvanceKernelPolicy::ADVANCE_MODE == gunrock::oprtr::advance::LB) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                }

                if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(frontier_attribute.queue_index, frontier_attribute.queue_length)) break;
                    enactor_stats.total_queued += frontier_attribute.queue_length;
                    if (DEBUG) printf(", %lld", (long long) frontier_attribute.queue_length);
                    if (INSTRUMENT) {
                        if (retval = enactor_stats.filter_kernel_stats.Accumulate(
                            enactor_stats.filter_grid_size,
                            enactor_stats.total_runtimes,
                            enactor_stats.total_lifetimes)) break;
                    }
                }
                // Check if done
                if (done[0] == 0) break;

                enactor_stats.iteration++;

                if (DEBUG) printf("\n%lld", (long long) enactor_stats.iteration);

            }

            if (retval) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
            }
        if (d_scanned_edges) cudaFree(d_scanned_edges);*/
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
    // Members
    protected:

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    //volatile int                      **dones;
    //int                               **d_dones;
    //util::Array1D<SizeT, cudaEvent_t> throttle_events;
    //util::Array1D<SizeT, cudaError_t> retvals;

    texture<unsigned char, cudaTextureType1D, cudaReadModeElementType> *ts_bitmask;
    texture<SizeT        , cudaTextureType1D, cudaReadModeElementType> *ts_rowoffset;
    texture<VertexId     , cudaTextureType1D, cudaReadModeElementType> *ts_columnindices;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for BFS kernel call. Must be called prior to each BFS search.
     *
     * @param[in] problem BFS Problem object which holds the graph data and BFS problem data to compute.
     * @param[in] edge_map_grid_size CTA occupancy for edge mapping kernel call.
     * @param[in] filter_grid_size CTA occupancy for filter kernel call.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    cudaError_t Setup(
        BFSProblem *problem)
    {
        cudaError_t retval = cudaSuccess;
        //this->num_gpus     = problem->num_gpus;
        //this->gpu_idx      = problem->gpu_idx;
        //throttle_events.Allocate(this->num_gpus);
        //retvals.Allocate(this->num_gpus);

        do {
            //dones   = new volatile int* [this->num_gpus];
            //d_dones = new          int* [this->num_gpus];
            ts_bitmask       = new texture<unsigned char, cudaTextureType1D, cudaReadModeElementType>[this->num_gpus];
            ts_rowoffset     = new texture<SizeT        , cudaTextureType1D, cudaReadModeElementType>[this->num_gpus];
            ts_columnindices = new texture<VertexId     , cudaTextureType1D, cudaReadModeElementType>[this->num_gpus];

            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                if (retval = util::SetDevice(this->gpu_idx[gpu])) return retval;
                //initialize the host-mapped "done"
                //int flags = cudaHostAllocMapped;

                // Allocate pinned memory for done
                //if (retval = util::GRError(cudaHostAlloc((void**)&(enactor_stats[gpu].done), sizeof(int) * 1, flags),
                //    "BFSEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) return retval;

                // Map done into GPU space
                //if (retval = util::GRError(cudaHostGetDevicePointer((void**)&(enactor_stats[gpu].d_done), (void*) enactor_stats[gpu].done, 0),
                //    "BFSEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) return retval;

                // Create throttle event
                //if (retval = util::GRError(cudaEventCreateWithFlags(&enactor_stats[gpu].throttle_event, cudaEventDisableTiming),
                //    "BFSEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) return retval;
                
                //enactor_stats[gpu].done   = -1;
                //enactor_stats[gpu].retval = cudaSuccess;
                
                // Bind row-offsets and bitmask texture
                cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
                ts_rowoffset[gpu].channelDesc = row_offsets_desc;
                if (retval = util::GRError(cudaBindTexture(
                    0,
                    ts_rowoffset[gpu],
                    problem->graph_slices[gpu]->row_offsets.GetPointer(util::DEVICE),
                    (problem->graph_slices[gpu]->nodes + 1) * sizeof(SizeT)),
                        "BFSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

                if (BFSProblem::ENABLE_IDEMPOTENCE) {
                    int bytes = (problem->graph_slices[gpu]->nodes + 8 - 1) / 8;
                    cudaChannelFormatDesc   bitmask_desc = cudaCreateChannelDesc<char>();
                    ts_bitmask[gpu].channelDesc = bitmask_desc;
                    if (retval = util::GRError(cudaBindTexture(
                                0,
                                ts_bitmask[gpu],
                                problem->data_slices[gpu]->visited_mask.GetPointer(util::DEVICE),
                                bytes),
                            "BFSEnactor cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;
                }
            }

            /*//graph slice
            typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];
            typename ProblemData::DataSlice *data_slice = problem->data_slices[0];

        do {

           }*/

            /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref.channelDesc = column_indices_desc;
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            graph_slice->edges * sizeof(VertexId)),
                        "BFSEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
        } while (0);
        
        return retval;
    }

    public:

    /**
     * @brief BFSEnactor constructor
     */
    BFSEnactor(bool DEBUG = false, int num_gpus = 1, int* gpu_idx = NULL) :
        EnactorBase(EDGE_FRONTIERS, DEBUG, num_gpus, gpu_idx)//,
        //dones(NULL),
        //d_dones(NULL)
    {}

    /**
     * @brief BFSEnactor destructor
     */
    virtual ~BFSEnactor()
    {
        /*if (All_Done(dones,retvals.GetPointer(),num_gpus)) {
            for (int gpu=0;gpu<num_gpus;gpu++)
            {   
                if (num_gpus !=1)
                    util::GRError(cudaSetDevice(gpu_idx[gpu]),
                        "BFSEnactor cudaSetDevice gpu failed", __FILE__, __LINE__);

                //util::GRError(cudaFreeHost((void*)(dones[gpu])),
                //    "BFSEnactor cudaFreeHost done failed", __FILE__, __LINE__);

                //util::GRError(cudaEventDestroy(throttle_events[gpu]),
                //    "BFSEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
            }   
            //delete[] dones;          dones           = NULL;
            //throttle_events.Release();
            //retvals        .Release();
            //delete[] throttle_event ;throttle_event  = NULL; 
        }*/
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
        VertexId &search_depth,
        double &avg_duty)
    {
        unsigned long long total_lifetimes=0;
        unsigned long long total_runtimes =0;
        total_queued = 0;
        search_depth = 0;
        for (int gpu=0;gpu<num_gpus;gpu++)
        {
            if (num_gpus!=1)
                util::GRError(cudaSetDevice(gpu_idx[gpu]),
                    "BFSEnactor cudaSetDevice gpu failed", __FILE__, __LINE__);
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
    ContextPtr *context,
    BFSProblem  *problem,
    VertexId    src,
    int         max_grid_size = 0)
    {
        /*typedef typename BFSProblem::SizeT      SizeT;
        typedef typename BFSProblem::VertexId   VertexId;

        typedef BFSFunctor<
            VertexId,
            SizeT,
            VertexId,
            BFSProblem> BfsFunctor;*/

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

            // Lazy initialization
            if (retval = Setup(problem)) break;

            if (retval = EnactorBase::Setup(problem,
                                            max_grid_size,
                                            AdvanceKernelPolicy::CTA_OCCUPANCY, 
                                            FilterKernelPolicy::CTA_OCCUPANCY)) break;

            for (int gpu=0;gpu<num_gpus;gpu++)
            {
                thread_slices[gpu].thread_num    = gpu;
                thread_slices[gpu].problem       = (void*)problem;
                thread_slices[gpu].enactor       = (void*)this;
                thread_slices[gpu].cpu_barrier   = &cpu_barrier;
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
            if (this->retvals[gpu]!=cudaSuccess) {retval=this->retvals[gpu];break;}
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
        ContextPtr *context,
        BFSProblem  *problem,
        VertexId    src,
        int         max_grid_size = 0)
    {
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
                7,                                  // LOG_THREADS
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
                7,                                  // LOG_THREADS
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
