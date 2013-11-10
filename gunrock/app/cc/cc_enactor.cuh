// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * cc_enactor.cuh
 *
 * @brief CC Problem Enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel.cuh>
#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/vertex_map/kernel.cuh>
#include <gunrock/oprtr/vertex_map/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/cc/cc_problem.cuh>
#include <gunrock/app/cc/cc_functor.cuh>


namespace gunrock {
namespace app {
namespace cc {

template<bool INSTRUMENT>                           // Whether or not to collect per-CTA clock-count statistics
class CCEnactor : public EnactorBase
{
    // Members
    protected:

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime vertex_map_kernel_stats;

    unsigned long long total_runtimes;              // Total working time by each CTA
    unsigned long long total_lifetimes;             // Total life time of each CTA
    unsigned long long total_queued;

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int        *done;
    int                 *d_done;
    volatile int        *flag;
    int                 *d_flag;
    bool                *vertex_flag;
    bool                *edge_flag;
    cudaEvent_t         throttle_event;

    /**
     * Current iteration
     */
    long long                           iteration;

    // Methods
    protected:

    /**
     * Prepare enactor for CC. Must be called prior to each CC.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem,
        int vertex_map_grid_size)
    {
        typedef typename ProblemData::SizeT         SizeT;
        typedef typename ProblemData::VertexId      VertexId;
        
        cudaError_t retval = cudaSuccess;

        do {

            int flags = cudaHostAllocMapped;
            //initialize the host-mapped "done"
            if (!done) {
                // Allocate pinned memory for done
                if (retval = util::GRError(cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                    "CCEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) break;

                // Map done into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "CCEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break; 

                // Create throttle event
                if (retval = util::GRError(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "CCEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
            }

            if (!flag) {
                
                // Allocate pinned memory for flag
                if (retval = util::GRError(cudaHostAlloc((void**)&flag, sizeof(int) * 1, flags),
                    "CCEnactor cudaHostAlloc flag failed", __FILE__, __LINE__)) break;

                // Map flag into GPU space
                if (retval = util::GRError(cudaHostGetDevicePointer((void**)&d_flag, (void*) flag, 0),
                    "CCEnactor cudaHostGetDevicePointer flag failed", __FILE__, __LINE__)) break;
            }

            //initialize runtime stats
            if (retval = vertex_map_kernel_stats.Setup(vertex_map_grid_size)) break;

            //Reset statistics
            iteration           = 0;
            total_runtimes      = 0;
            total_lifetimes     = 0;
            total_queued        = 0;
            done[0]             = -1;
            flag[0]             = -1;

        } while (0);
        
        return retval;
    }

    public:

    /**
     * Constructor
     */
    CCEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        done(NULL),
        d_done(NULL),
        flag(NULL),
        d_flag(NULL),
        vertex_flag(NULL),
        edge_flag(NULL)
    {
        vertex_flag = new bool;
        edge_flag = new bool;
        vertex_flag[0] = false;
        edge_flag[0] = false;
        }

    /**
     * Destructor
     */
    ~CCEnactor()
    {
        if (done) {
            util::GRError(cudaFreeHost((void*)done),
                "CCEnactor cudaFreeHost done failed", __FILE__, __LINE__);

            util::GRError(cudaEventDestroy(throttle_event),
                "CCEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
        }

        if (flag) {
            util::GRError(cudaFreeHost((void*)flag),
                "CCEnactor cudaFreeHost done failed", __FILE__, __LINE__);

        }
        if (vertex_flag) delete vertex_flag;
        if (edge_flag) delete edge_flag;
    }

    /**
     * Obtain statistics about the last CC enacted
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
        typename VertexMapPolicy,
        typename CCProblem,
        typename UpdateMaskFunctor,
        typename HookInitFunctor,
        typename HookMinFunctor,
        typename HookMaxFunctor,
        typename PtrJumpFunctor,
        typename PtrJumpMaskFunctor,
        typename PtrJumpUnmaskFunctor>
    cudaError_t Enact(
    CCProblem                          *problem,
    int                                 max_grid_size = 0)
    {
        typedef typename CCProblem::SizeT      SizeT;
        typedef typename CCProblem::VertexId   VertexId;

        cudaError_t retval = cudaSuccess;

        do {
            // Determine grid size(s)
            int vertex_map_occupancy    = VertexMapPolicy::CTA_OCCUPANCY;
            int vertex_map_grid_size    = MaxGridSize(vertex_map_occupancy, max_grid_size);

            if (DEBUG) {
                printf("CC vertex map occupancy %d, level-grid size %d\n",
                        vertex_map_occupancy, vertex_map_grid_size);
            }

            // Lazy initialization
            if (retval = Setup(problem, vertex_map_grid_size)) break;

            // Single-gpu graph slice
            typename CCProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename CCProblem::DataSlice *data_slice = problem->d_data_slices[0];

            SizeT queue_length          = graph_slice->edges;
            VertexId queue_index        = 0;        // Work queue index
            int selector                = 0;
            SizeT num_elements          = graph_slice->edges;
            bool queue_reset            = true;

            // Initial Hook Operation
            gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, CCProblem, HookInitFunctor>
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

            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map::Kernel Initial HookMin Operation failed", __FILE__, __LINE__))) break;

            printf("init hook\n");

            queue_length = graph_slice->nodes;
            queue_index = 0;
            selector = 0;
            num_elements = graph_slice->nodes;
            iteration = 0;

            // First Pointer Jumping Round
            vertex_flag[0] = false;
            while (!vertex_flag[0]) {
                vertex_flag[0] = true;
                if (retval = util::GRError(cudaMemcpy(
                                problem->data_slices[0]->d_vertex_flag,
                                vertex_flag,
                                sizeof(bool),
                                cudaMemcpyHostToDevice),
                            "CCProblem cudaMemcpy vertex_flag to d_vertex_flag failed", __FILE__, __LINE__)) return retval;

                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, CCProblem, PtrJumpFunctor>
                    <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                            queue_reset,
                            queue_index,
                            1,
                            num_elements,
                            d_done,
                            graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                            graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                            data_slice,
                            work_progress,
                            graph_slice->frontier_elements[selector],           // max_in_queue
                            graph_slice->frontier_elements[selector^1],         // max_out_queue
                            this->vertex_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map::Kernel First Pointer Jumping Round failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates
 
                printf("ptr jump.\n"); 
                if (queue_reset) queue_reset = false;

                // Throttle
                if (iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                                "CCEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                                "CCEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                queue_index++;
                selector ^= 1;
                iteration++;
                
                if (INSTRUMENT || DEBUG) {
                    if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
                    total_queued += queue_length;
                    if (INSTRUMENT) {
                        if (retval = vertex_map_kernel_stats.Accumulate(
                                    vertex_map_grid_size,
                                    total_runtimes,
                                    total_lifetimes)) break;
                    }
                }

                if (retval = util::GRError(cudaMemcpy(
                                vertex_flag,
                                problem->data_slices[0]->d_vertex_flag,
                                sizeof(bool),
                                cudaMemcpyDeviceToHost),
                            "CCProblem cudaMemcpy d_vertex_flag to vertex_flag failed", __FILE__, __LINE__)) return retval;
                // Check if done
                if (vertex_flag[0]) break;
            }

            queue_length = graph_slice->nodes;
            queue_index = 0;
            selector = 0;
            num_elements = graph_slice->nodes;
            queue_reset = true;
            util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], graph_slice->nodes);

            // Initial Update Mask Operation
            gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, CCProblem, UpdateMaskFunctor>
                <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                    queue_reset,
                    queue_index,
                    1,
                    num_elements,
                    d_done,
                    graph_slice->frontier_queues.d_values[selector],
                    graph_slice->frontier_queues.d_values[selector^1],
                    data_slice,
                    work_progress,
                    graph_slice->frontier_elements[selector],
                    graph_slice->frontier_elements[selector^1],
                    this->vertex_map_kernel_stats);

            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map::Kernel Initial Update Mask failed", __FILE__, __LINE__))) break;

            printf("update mask.\n");

            queue_length = graph_slice->edges;
            queue_index = 0;
            selector = 0;
            num_elements = graph_slice->edges;
            iteration = 1;
            
            edge_flag[0] = false;
            while (!edge_flag[0]) {
                if (retval = work_progress.SetQueueLength(queue_index, queue_length)) break;
                if (retval = work_progress.SetQueueLength(queue_index+1, queue_length)) break;
                if (retval = work_progress.SetQueueLength(queue_index+2, queue_length)) break;
                if (retval = work_progress.SetQueueLength(queue_index+3, queue_length)) break;
                edge_flag[0] = true;
                done[0] = -1;
                util::DisplayDeviceResults(problem->data_slices[0]->d_edge_flag, 1);
                printf("edge_flag:%d\n", edge_flag[0]);
                if (retval = util::GRError(cudaMemcpy(
                                problem->data_slices[0]->d_edge_flag,
                                edge_flag,
                                sizeof(bool),
                                cudaMemcpyHostToDevice),
                            "CCProblem cudaMemcpy edge_flag to d_edge_flag failed", __FILE__, __LINE__)) return retval;

                printf("iter:%lld, queue_length:%d\n", iteration, queue_length);
                if (iteration & 3) {
                    gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, CCProblem, HookMaxFunctor>
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
                }
                else {
                    gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, CCProblem, HookMinFunctor>
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
                }

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map::Kernel Hook Min/Max Operation failed", __FILE__, __LINE__))) break;
                cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

                // Throttle
                if (iteration & 1) {
                    if (retval = util::GRError(cudaEventRecord(throttle_event),
                                "CCEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                } else {
                    if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                                "CCEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                }

                if (queue_reset = true) queue_reset = false;

                queue_index++;
                selector ^= 1;
                iteration++;

                // Save current queue_length
                if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;

                printf(" hook min/max itr:%lld, queue_length out:%d\n", iteration, queue_length);

                if (INSTRUMENT || DEBUG) {

                    total_queued += queue_length;
                    if (INSTRUMENT) {
                        if (retval = vertex_map_kernel_stats.Accumulate(
                                    vertex_map_grid_size,
                                    total_runtimes,
                                    total_lifetimes)) break;
                    }
                }
                if (retval = util::GRError(cudaMemcpy(
                                edge_flag,
                                problem->data_slices[0]->d_edge_flag,
                                sizeof(bool),
                                cudaMemcpyDeviceToHost),
                            "CCProblem cudaMemcpy d_edge_flag to edge_flag failed", __FILE__, __LINE__)) return retval;
                // Check if done
                if (edge_flag[0]) break; 

                int ptrj_queue_reset         = true;
                flag[0]                      = -1;
                int ptrj_iteration           = 0;


                // Pointer Jumping Iterations
                int ptrj_queue_length;
                int ptrj_queue_index;
                int ptrj_selector;
                int ptrj_num_elements;

                vertex_flag[0] = false;
                while (!vertex_flag[0]) {
                    vertex_flag[0] = true;
                    ptrj_queue_length        = graph_slice->nodes;
                    ptrj_queue_index         = 0;        // Work queue index
                    ptrj_selector            = 0;
                    ptrj_num_elements        = graph_slice->nodes;

                    if (retval = util::GRError(cudaMemcpy(
                                    problem->data_slices[0]->d_vertex_flag,
                                    vertex_flag,
                                    sizeof(bool),
                                    cudaMemcpyHostToDevice),
                                "CCProblem cudaMemcpy vertex_flag to d_vertex_flag failed", __FILE__, __LINE__)) return retval;
                    gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, CCProblem, PtrJumpMaskFunctor>
                        <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                                ptrj_queue_reset,
                                ptrj_queue_index,
                                1,
                                ptrj_num_elements,
                                d_flag,
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,
                                work_progress,
                                graph_slice->frontier_elements[selector],           // max_in_queue
                                graph_slice->frontier_elements[selector^1],         // max_out_queue
                                this->vertex_map_kernel_stats);

                    if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map::Kernel Pointer Jumping Mask failed", __FILE__, __LINE__))) break;
                    printf("ptr jump mask\n");
                    cudaEventQuery(throttle_event); // give host memory mapped visibility to GPU updates

                    // Throttle
                    if (ptrj_iteration & 1) {
                        if (retval = util::GRError(cudaEventRecord(throttle_event),
                                    "CCEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
                    } else {
                        if (retval = util::GRError(cudaEventSynchronize(throttle_event),
                                    "CCEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
                    }

                    ptrj_queue_index++;
                    ptrj_selector ^= 1;
                    ptrj_iteration++;

                    if (INSTRUMENT || DEBUG) {
                        if (retval = work_progress.GetQueueLength(ptrj_queue_index, ptrj_queue_length)) break;
                        total_queued += ptrj_queue_length;
                        if (INSTRUMENT) {
                            if (retval = vertex_map_kernel_stats.Accumulate(
                                        vertex_map_grid_size,
                                        total_runtimes,
                                        total_lifetimes)) break;
                        }
                    }

                    if (retval = util::GRError(cudaMemcpy(
                                    vertex_flag,
                                    problem->data_slices[0]->d_vertex_flag,
                                    sizeof(bool),
                                    cudaMemcpyDeviceToHost),
                                "CCProblem cudaMemcpy d_vertex_flag to vertex_flag failed", __FILE__, __LINE__)) return retval;
                    // Check if done
                    if (vertex_flag[0]) break;

                }

                ptrj_queue_length        = graph_slice->nodes;
                ptrj_queue_index         = 0;        // Work queue index
                ptrj_selector            = 0;
                ptrj_num_elements        = graph_slice->nodes;

                ptrj_queue_reset         = true;
                flag[0]                  = -1;

                util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], graph_slice->nodes);

                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, CCProblem, PtrJumpUnmaskFunctor>
                    <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                            ptrj_queue_reset,
                            ptrj_queue_index,
                            1,
                            ptrj_num_elements,
                            d_flag,
                            graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                            graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                            data_slice,
                            work_progress,
                            graph_slice->frontier_elements[selector],           // max_in_queue
                            graph_slice->frontier_elements[selector^1],         // max_out_queue
                            this->vertex_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map::Kernel Pointer Jumping Unmask failed", __FILE__, __LINE__))) break;
                printf("ptr jump unmask.\n");

                ptrj_queue_length = 0;
                ptrj_queue_index = 0;
                ptrj_selector = 0;
                ptrj_num_elements = graph_slice->nodes;
                ptrj_queue_reset = true;
                flag[0] = -1;
                util::MemsetIdxKernel<<<128, 128>>>(graph_slice->frontier_queues.d_values[selector], graph_slice->nodes);

                // Update Mask Operation
                gunrock::oprtr::vertex_map::Kernel<VertexMapPolicy, CCProblem, UpdateMaskFunctor>
                    <<<vertex_map_grid_size, VertexMapPolicy::THREADS>>>(
                            queue_reset,
                            queue_index,
                            1,
                            num_elements,
                            d_flag,
                            graph_slice->frontier_queues.d_values[selector],
                            graph_slice->frontier_queues.d_values[selector^1],
                            data_slice,
                            work_progress,
                            graph_slice->frontier_elements[selector],
                            graph_slice->frontier_elements[selector^1],
                            this->vertex_map_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "vertex_map::Kernel Update Mask failed", __FILE__, __LINE__))) break;
                printf("update mask.\n");
            }
            
            if (retval) break;

            // Check if any of the frontiers overflowed due to redundant expansion
            /*bool overflowed = false;
            if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
            if (overflowed) {
                retval = util::GRError(cudaErrorInvalidConfiguration, "Frontier queue overflow. Please increase queue-sizing factor.",__FILE__, __LINE__);
                break;
            }*/

        } while(0);

        if (DEBUG) printf("\nGPU CC Done.\n");
        return retval;
    }

    /**
     * Enact Kernel Entry, specify KernelPolicy
     */
    template <typename CCProblem,
              typename UpdateMaskFunctor,
              typename HookInitFunctor,
              typename HookMinFunctor,
              typename HookMaxFunctor,
              typename PtrJumpFunctor,
              typename PtrJumpMaskFunctor,
              typename PtrJumpUnmaskFunctor>
    cudaError_t Enact(
        CCProblem                      *problem,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::vertex_map::KernelPolicy<
                CCProblem,                         // Problem data type
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
                
                return Enact<VertexMapPolicy, CCProblem, UpdateMaskFunctor, HookInitFunctor, HookMinFunctor, HookMaxFunctor, PtrJumpFunctor, PtrJumpMaskFunctor, PtrJumpUnmaskFunctor>(
                problem, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

};

} // namespace cc
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
