// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hits_enactor.cuh
 *
 * @brief HITS (Hyperlink-Induced Topic Search) Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/hits/hits_problem.cuh>
#include <gunrock/app/hits/hits_functor.cuh>

#include <moderngpu.cuh>

using namespace mgpu;

namespace gunrock {
namespace app {
namespace hits {

/**
 * @brief HITS problem enactor class.
 *
 * @tparam INSTRUMWENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<typename _Problem, bool _INSTRUMENT, bool _DEBUG, bool _SIZE_CHECK>
class HITSEnactor : public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>
{
public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT   ;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value   ;
    static const bool INSTRUMENT = _INSTRUMENT;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;

    // Members
    protected:

    //volatile int        *done;
    //int                 *d_done;

    /**
     * Current iteration, also used to get the final search depth of the HITS search
     */
    long long                           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for HITS kernel call. Must be called prior to each HITS search.
     *
     * @param[in] problem HITS Problem object which holds the graph data and HITS problem data to compute.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem)
    {
        //typedef typename ProblemData::SizeT         SizeT;
        //typedef typename ProblemData::VertexId      VertexId;

        cudaError_t retval = cudaSuccess;

        return retval;
    }

    public:

    /**
     * @brief HITSEnactor constructor
     */
    HITSEnactor(int *gpu_idx) :
        EnactorBase<SizeT, DEBUG, SIZE_CHECK>(EDGE_FRONTIERS, 1, gpu_idx)
        //done(NULL),
        //d_done(NULL)
    {}

    /**
     * @brief HITSEnactor destructor
     */
    virtual ~HITSEnactor()
    {
    }

    template <typename ProblemData>
    void NormalizeRank(ProblemData *problem, CudaContext &context, int hub_or_auth, int nodes)
    {

        typedef typename ProblemData::Value         Value;
        Value *rank_curr;
        Value *rank_next;
        if (hub_or_auth == 0) {
            rank_curr = problem->data_slices[0]->hrank_curr.GetPointer(util::DEVICE);
            rank_next = problem->data_slices[0]->hrank_next.GetPointer(util::DEVICE);
            //printf("hub\n");
        } else {
            rank_curr = problem->data_slices[0]->arank_curr.GetPointer(util::DEVICE);
            rank_next = problem->data_slices[0]->arank_next.GetPointer(util::DEVICE);
            //printf("auth\n");
        }

        //swap rank_curr and rank_next
        util::MemsetCopyVectorKernel<<<128, 128>>>(rank_curr, rank_next, nodes);

        util::MemsetKernel<<<128, 128>>>(rank_next, (Value)0.0, nodes);

        //util::DisplayDeviceResults(rank_curr, nodes);
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @ brief Obtain statistics about the last HITS search enacted.
     *
     * @ param[out] total_queued Total queued elements in HITS kernel running.
     * @ param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     * spaces between @ and name are to eliminate doxygen warnings
     */
    /*void GetStatistics(
        long long &total_queued,
        double &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = this->enactor_stats->total_queued[0];

        avg_duty = (this->enactor_stats->total_lifetimes >0) ?
            double(this->enactor_stats->total_runtimes) / this->enactor_stats->total_lifetimes : 0.0;
    }*/

    /** @} */

    /**
     * @brief Enacts a HITS computing on the specified graph.
     *
     * @tparam EdgeMapPolicy Kernel policy for forward edge mapping.
     * @tparam VertexMapPolicy Kernel policy for vertex mapping.
     * @tparam HITSProblem HITS Problem type.
     *
     * @param[in] context CudaContext for moderngpu library
     * @param[in] problem HITSProblem object.
     * @param[in] max_iteration Max number of iterations of HITS algorithm
     * @param[in] max_grid_size Max grid size for HITS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename HITSProblem>
    cudaError_t EnactHITS(
    ContextPtr                         context,
    HITSProblem                       *problem,
    typename HITSProblem::SizeT        max_iteration,
    int                                max_grid_size = 0)
    {
        typedef typename HITSProblem::SizeT       SizeT;
        typedef typename HITSProblem::VertexId    VertexId;
        typedef typename HITSProblem::Value       Value;

        typedef HUBFunctor<
            VertexId,
            SizeT,
            Value,
            HITSProblem> HubFunctor;

        typedef AUTHFunctor<
            VertexId,
            SizeT,
            Value,
            HITSProblem> AuthFunctor;

        GraphSlice<SizeT, VertexId, Value>
                     *graph_slice        = problem->graph_slices       [0];
        FrontierAttribute<SizeT>
                     *frontier_attribute = &this->frontier_attribute   [0];
        EnactorStats *enactor_stats      = &this->enactor_stats        [0];
        // Single-gpu graph slice
        typename HITSProblem::DataSlice
                     *data_slice         =  problem->data_slices       [0];
        typename HITSProblem::DataSlice
                     *d_data_slice       =  problem->d_data_slices     [0];
        util::DoubleBuffer<SizeT, VertexId, Value>
                     *frontier_queue     = &data_slice->frontier_queues[0];
        util::CtaWorkProgressLifetime
                     *work_progress      = &this->work_progress        [0];
        cudaStream_t  stream             =  data_slice->streams        [0];
        cudaError_t   retval             = cudaSuccess;

        do {
            if (DEBUG) {
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
            }

            fflush(stdout);

            // Lazy initialization
            if (retval = Setup(problem)) break;
            if (retval = EnactorBase<SizeT, DEBUG, SIZE_CHECK>::Setup(problem,
                                            max_grid_size,
                                            AdvanceKernelPolicy::CTA_OCCUPANCY,
                                            FilterKernelPolicy::CTA_OCCUPANCY))
                                            break;

            // Single-gpu graph slice
            //typename HITSProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            //typename HITSProblem::DataSlice *data_slice = problem->d_data_slices[0];

            // Bind row-offsets texture
            //cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();

            frontier_attribute->queue_length         = graph_slice->nodes;
            frontier_attribute->queue_index          = 0;        // Work queue index
            frontier_attribute->selector             = 0;
            frontier_attribute->queue_reset          = true;

            // Step through HITS iterations
            while (true) {

                /*if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_column_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "HITSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;
                */

                if (retval = work_progress->SetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[selector], edge_map_queue_len);
                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, HITSProblem, AuthFunctor>(
                    //d_done,
                    enactor_stats[0],
                    frontier_attribute[0],
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    (SizeT*   )NULL,
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),            // d_out_queue
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    graph_slice->column_offsets.GetPointer(util::DEVICE),
                    graph_slice->row_indices.GetPointer(util::DEVICE),
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    graph_slice->nodes,//graph_slice->frontier_elements[frontier_attribute.selector],                   // max_in_queue
                    graph_slice->edges,//graph_slice->frontier_elements[frontier_attribute.selector^1],                 // max_out_queue
                    work_progress[0],
                    context[0],
                    stream,
                    gunrock::oprtr::advance::V2V,
                    false,
                    false,
                    true);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;


                //util::DisplayDeviceResults(problem->data_slices[0]->d_arank_next,graph_slice->nodes);
                NormalizeRank<HITSProblem>(problem, context[0], 1, graph_slice->nodes);

                /*if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "HITSEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;
                */

                //util::DisplayDeviceResults(graph_slice->frontier_queues.d_keys[selector], edge_map_queue_len);
                // Edge Map
                gunrock::oprtr::advance::LaunchKernel<AdvanceKernelPolicy, HITSProblem, HubFunctor>(
                    //d_done,
                    enactor_stats[0],
                    frontier_attribute[0],
                    d_data_slice,
                    (VertexId*)NULL,
                    (bool*    )NULL,
                    (bool*    )NULL,
                    (SizeT*   )NULL,
                    frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),              // d_in_queue
                    frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),            // d_out_queue
                    (VertexId*)NULL,
                    (VertexId*)NULL,
                    graph_slice->row_offsets.GetPointer(util::DEVICE),
                    graph_slice->column_indices.GetPointer(util::DEVICE),
                    (SizeT*   )NULL,
                    (VertexId*)NULL,
                    graph_slice->nodes,//graph_slice->frontier_elements[frontier_attribute.selector],                   // max_in_queue
                    graph_slice->edges,//graph_slice->frontier_elements[frontier_attribute.selector^1],                 // max_out_queue
                    work_progress[0],
                    context[0],
                    stream,
                    gunrock::oprtr::advance::V2V,
                    false,
                    false,
                    true);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;

                //util::DisplayDeviceResults(problem->data_slices[0]->d_arank_next,graph_slice->nodes);

                if (DEBUG) {
                    if (retval = work_progress->GetQueueLength(frontier_attribute->queue_index, frontier_attribute->queue_length)) break;
                    printf(", %lld", (long long)frontier_attribute->queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = enactor_stats->advance_kernel_stats.Accumulate(
                        enactor_stats->advance_grid_size,
                        enactor_stats->total_runtimes,
                        enactor_stats->total_lifetimes)) break;
                }

                NormalizeRank<HITSProblem>(problem, context[0], 0, graph_slice->nodes);


                //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_next,graph_slice->nodes);
                //util::DisplayDeviceResults(problem->data_slices[0]->d_rank_curr,
                //    graph_slice->nodes);

                /*if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    total_queued += queue_length;
                    if (DEBUG) printf(", %lld", (long long) queue_length);
                    if (INSTRUMENT) {
                        if (retval = vertex_map_kernel_stats.Accumulate(
                            vertex_map_grid_size,
                            total_runtimes,
                            total_lifetimes)) break;
                    }
                }*/

                enactor_stats->iteration++;

                if (enactor_stats->iteration >= max_iteration) break;

                if (DEBUG) printf("\n%lld", (long long) enactor_stats->iteration);

            }

            if (retval) break;

        } while(0);

        if (DEBUG) printf("\nGPU HITS Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief HITS Enact kernel entry.
     *
     * @tparam HITSProblem HITS Problem type. @see HITSProblem
     *
     * @param[in] context CudaContext for moderngpu library
     * @param[in] problem Pointer to HITSProblem object.
     * @param[in] max_iteration Max iteration number for the algorithm
     * @param[in] max_grid_size Max grid size for HITS kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename HITSProblem>
    cudaError_t Enact(
        ContextPtr                        context,
        HITSProblem                      *problem,
        typename HITSProblem::SizeT       max_iteration,
        int                               max_grid_size = 0)
    {
        int min_sm_version = -1;
        for (int i=0;i<this->num_gpus;i++)
            if (min_sm_version == -1 || this->cuda_props[i].device_sm_version < min_sm_version)
                min_sm_version = this->cuda_props[i].device_sm_version;

        if (min_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                HITSProblem,                         // Problem data type
            300,                                // CUDA_ARCH
            INSTRUMENT,                         // INSTRUMENT
            0,                                  // SATURATION QUIT
            true,                               // DEQUEUE_HITSOBLEM_SIZE
            8,                                  // MIN_CTA_OCCUPANCY
            6,                                  // LOG_THREADS
            1,                                  // LOG_LOAD_VEC_SIZE
            0,                                  // LOG_LOADS_PER_TILE
            5,                                  // LOG_RAKING_THREADS
            5,                                  // END_BITMASK_CULL
            8>                                  // LOG_SCHEDULE_GRANULARITY
                FilterKernelPolicy;

            typedef gunrock::oprtr::advance::KernelPolicy<
                HITSProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                6,                                  // LOG_THREADS
                0,                                  // LOG_BLOCKS
                0,                                  // LIGHT_EDGE_THRESHOLD
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                32,                            // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7,                                  // LOG_SCHEDULE_GRANULARITY
                gunrock::oprtr::advance::TWC_FORWARD>
                    AdvanceKernelPolicy;

            return EnactHITS<AdvanceKernelPolicy, FilterKernelPolicy, HITSProblem>(
                    context, problem, max_iteration, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace hits
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
