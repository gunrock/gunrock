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

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/bc/bc_problem.cuh>
#include <gunrock/app/bc/bc_functor.cuh>


namespace gunrock {
namespace app {
namespace bc {

template<bool INSTRUMENT>                           // Whether or not to collect per-CTA clock-count statistics
class BCEnactor : public EnactorBase
{
    // Members
    protected:

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime edge_map_kernel_stats;
    util::KernelRuntimeStatsLifetime vertex_map_kernel_stats;

    unsigned long long total_runtimes;              // Total working time by each CTA
    unsigned long long total_lifetimes;             // Total life time of each CTA
    unsigned long long total_queued;

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int        *done;
    int                 *d_done;
    cudaEvent_t         throttle_event;

    /**
     * Current iteration, also used to get the final search depth of the BC search
     */
    long long                           iteration;

    // Methods
    protected:

    /**
     * Prepare enactor for BC. Must be called prior to each BC search.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem,
        int edge_map_grid_size,
        int vertex_map_grid_size)
    {
        typedef typename ProblemData::SizeT         SizeT;
        typedef typename ProblemData::VertexId      VertexId;
        
        cudaError_t retval = cudaSuccess;

        do {
            //initialize the host-mapped "done"
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

            //initialize runtime stats
            if (retval = edge_map_kernel_stats.Setup(edge_map_grid_size)) break;
            if (retval = vertex_map_kernel_stats.Setup(vertex_map_grid_size)) break;

            //Reset statistics
            iteration           = 0;
            total_runtimes      = 0;
            total_lifetimes     = 0;
            total_queued        = 0;
            done[0]             = -1;

            //graph slice
            typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];

            // Bind row-offsets and column_indices texture
            cudaChannelFormatDesc   row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            if (retval = util::GRError(cudaBindTexture(
                    0,
                    gunrock::oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                    graph_slice->d_row_offsets,
                    row_offsets_desc,
                    (graph_slice->nodes + 1) * sizeof(SizeT)),
                        "BCEnactor cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

            /*cudaChannelFormatDesc   column_indices_desc = cudaCreateChannelDesc<VertexId>();
            if (retval = util::GRError(cudaBindTexture(
                            0,
                            gunrock::oprtr::edge_map_forward::ColumnIndicesTex<SizeT>::ref,
                            graph_slice->d_column_indices,
                            column_indices_desc,
                            graph_slice->edges * sizeof(VertexId)),
                        "BCEnactor cudaBindTexture column_indices_tex_ref failed", __FILE__, __LINE__)) break;*/
        } while (0);
        
        return retval;
    }

    public:

    /**
     * Constructor
     */
    BCEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL)
    {}

    /**
     * Destructor
     */
    virtual ~BCEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "BCEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "BCEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }
    }

    /**
     * Obtain statistics about the last BC search enacted
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId &search_depth,
        double &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = this->total_queued;
        search_depth = this->iteration;

        avg_duty = (total_lifetimes >0) ?
            double(total_runtimes) / total_lifetimes : 0.0;
    }

    /**
     * Enacts a breadth-first-search on the specified graph problem.
     *
     * @return cudaSuccess on success, error enumeration otherwise
     */
    template<
        typename EdgeMapPolicy,
        typename VertexMapPolicy,
        typename BCProblem,
        typename ForwardFunctor,
        typename BackwardFunctor>
    cudaError_t Enact(
    BCProblem                          *problem,
    typename BCProblem::VertexId       src,
    int                                 max_grid_size = 0)
    {
        typedef typename BCProblem::SizeT       SizeT;
        typedef typename BCProblem::VertexId    VertexId;
        typedef typename BCProblem::Value       Value;

        cudaError_t retval = cudaSuccess;

        do {
            // Determine grid size(s)
            int edge_map_occupancy      = EdgeMapPolicy::CTA_OCCUPANCY;
            int edge_map_grid_size      = MaxGridSize(edge_map_occupancy, max_grid_size);

            int vertex_map_occupancy    = VertexMapPolicy::CTA_OCCUPANCY;
            int vertex_map_grid_size    = MaxGridSize(vertex_map_occupancy, max_grid_size);

            if (DEBUG) {
                printf("BC edge map occupancy %d, level-grid size %d\n",
                        edge_map_occupancy, edge_map_grid_size);
                printf("BC vertex map occupancy %d, level-grid size %d\n",
                        vertex_map_occupancy, vertex_map_grid_size);
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
            }

            // Lazy initialization
            if (retval = Setup(problem, edge_map_grid_size, vertex_map_grid_size)) break;

            // Single-gpu graph slice
            typename BCProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename BCProblem::DataSlice *data_slice = problem->d_data_slices[0];

            SizeT queue_length          = 0;
            VertexId queue_index        = 0;        // Work queue index
            int selector                = 0;
            SizeT num_elements          = 1;

            bool queue_reset = true; 


            fflush(stdout);

            //VertexId *vids = new VertexId[graph_slice->edges*2];
            //VertexId *labels = new VertexId[graph_slice->nodes];
            //Value *sigmas = new Value[graph_slice->nodes];
            // Forward BC iteration
            while (done[0] < 0) {

                /*if (DEBUG) {
                    printf("Edge Map Input:\n");

                    if (retval = util::GRError(cudaMemcpy(
                                    vids,
                                    graph_slice->frontier_queues.d_keys[selector],
                                    sizeof(VertexId) * queue_length,
                                    cudaMemcpyDeviceToHost),
                                "BFSProblem cudaMemcpy d_vids failed", __FILE__, __LINE__)) break;

                    if (retval = util::GRError(cudaMemcpy(
                                    labels,
                                    problem->data_slices[0]->d_labels,
                                    sizeof(VertexId) * graph_slice->nodes,
                                    cudaMemcpyDeviceToHost),
                                "BFSProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;

                    if (retval = util::GRError(cudaMemcpy(
                                    sigmas,
                                    problem->data_slices[0]->d_sigmas,
                                    sizeof(Value) * graph_slice->nodes,
                                    cudaMemcpyDeviceToHost),
                                "BFSProblem cudaMemcpy d_sigmas failed", __FILE__, __LINE__)) break;
                    for (int i = 0; i < queue_length; ++i)
                    {
                        if (i % 5 == 0)
                            printf("\n");
                        printf(" |%d:label:%d,sigma:%f| ",vids[i],labels[vids[i]], sigmas[vids[i]]);
                        
                    }
                    printf("\n");
                }*/

                // Edge Map
                gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, BCProblem, ForwardFunctor>
                <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],            // d_out_queue
                    graph_slice->d_column_indices,
                    data_slice,
                    this->work_progress,
                    graph_slice->frontier_elements[selector],                   // max_in_queue
                    graph_slice->frontier_elements[selector^1],                 // max_out_queue
                    this->edge_map_kernel_stats);


                // Only need to reset queue for once
                if (queue_reset)
                    queue_reset = false;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates


                queue_index++;
                selector ^= 1;
                
                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    printf(", %lld", (long long) queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = edge_map_kernel_stats.Accumulate(
                        edge_map_grid_size,
                        total_runtimes,
                        total_lifetimes)) break;
                }

                // Throttle
                if (iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                        "BCEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BCEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;

                // Vertex Map
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, BCProblem, ForwardFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                    data_slice,
                    work_progress,
                    graph_slice->frontier_elements[selector],           // max_in_queue
                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                    this->vertex_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates


                queue_index++;
                selector ^= 1;
                iteration++;

                if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    total_queued += queue_length;
                    if (DEBUG) printf(", %lld", (long long) queue_length);
                    if (INSTRUMENT) {
                        if (retval = vertex_map_kernel_stats.Accumulate(
                            vertex_map_grid_size,
                            total_runtimes,
                            total_lifetimes)) break;
                    }
                }
                // Check if done
                if (done[0] == 0) break;

                if (DEBUG) printf("\n%lld", (long long) iteration);

            }
            //delete[] sigmas;
            //delete[] labels;
            //delete[] vids;

            iteration           = iteration - 2;

            queue_length        = 0;
            queue_index         = 0;        // Work queue index
            selector            = 0;
            num_elements        = graph_slice->nodes;
            queue_reset         = true;
            done[0]             = -1;


            // Prepare the label array
            VertexId            label_adjust = -iteration; 
            util::MemsetAddKernel<<<128, 128>>>(problem->data_slices[0]->d_labels, label_adjust, graph_slice->nodes);


            if (DEBUG) printf("\nStart backward phase\n%lld", (long long) iteration);

            // Backward BC iteration
            for (;iteration > 0; --iteration) {
                num_elements        = graph_slice->nodes;
                // Fill in the frontier_queues
                util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_keys[selector], num_elements);

                // Vertex Map
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, BCProblem, BackwardFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                    graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                    data_slice,
                    work_progress,
                    graph_slice->frontier_elements[selector],           // max_in_queue
                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                    this->vertex_map_kernel_stats);


                // Only need to reset queue for once
                if (queue_reset)
                    queue_reset = false;

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "edge_map_backward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event);                                 // give host memory mapped visibility to GPU updates


                queue_index++;
                selector ^= 1;
                
                if (DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    printf(", %lld", (long long) queue_length);
                }

                if (INSTRUMENT) {
                    if (retval = edge_map_kernel_stats.Accumulate(
                        edge_map_grid_size,
                        total_runtimes,
                        total_lifetimes)) break;
                }

                // Throttle
                if (iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                        "BCEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                        "BCEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                // Check if done
                if (done[0] == 0) break;

                // Edge Map
                gunrock::oprtr::edge_map_forward::Kernel<EdgeMapPolicy, BCProblem, BackwardFunctor>
                <<<edge_map_grid_size, EdgeMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_keys[selector],              // d_in_queue
                    graph_slice->frontier_queues.d_keys[selector],            // d_out_queue
                    graph_slice->d_column_indices,
                    data_slice,
                    this->work_progress,
                    graph_slice->frontier_elements[selector],                   // max_in_queue
                    graph_slice->frontier_elements[selector^1],                 // max_out_queue
                    this->edge_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map_forward::Kernel failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

                queue_index++;
                selector ^= 1;
                queue_reset = true;

                util::MemsetAddKernel<<<128, 128>>>(problem->data_slices[0]->d_labels, 1, graph_slice->nodes);

                if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    total_queued += queue_length;
                    if (DEBUG) printf(", %lld", (long long) queue_length);
                    if (INSTRUMENT) {
                        if (retval = vertex_map_kernel_stats.Accumulate(
                            vertex_map_grid_size,
                            total_runtimes,
                            total_lifetimes)) break;
                    }
                }
                // Check if done
                if (done[0] == 0) break;

                if (DEBUG) printf("\n%lld", (long long) iteration-1);

            }
            if (retval) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            bool overflowed = false;
            if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
            }
            
        } while(0);

        if (DEBUG) printf("\nGPU BC Done.\n");
        return retval;
    }

    /**
     * Enact Kernel Entry, specify KernelPolicy
     */
    template <typename BCProblem, typename FFunctor, typename BFunctor>
    cudaError_t Enact(
        BCProblem                      *problem,
        typename BCProblem::VertexId    src,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::vertex_map::KernelPolicy<
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
                8>                                  // LOG_SCHEDULE_GRANULARITY
                VertexMapPolicy;

                typedef gunrock::oprtr::edge_map_forward::KernelPolicy<
                BCProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                8,                                  // MIN_CTA_OCCUPANCY
                7,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                128 * 4,                                 // WARP_GATHER_THRESHOLD
                128 * 4,                            // CTA_GATHER_THRESHOLD
                7>                                  // LOG_SCHEDULE_GRANULARITY
                EdgeMapPolicy;

                return Enact<EdgeMapPolicy, VertexMapPolicy, BCProblem, FFunctor, BFunctor>(
                problem, src, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

};

} // namespace bc
} // namespace app
} // namespace gunrock

