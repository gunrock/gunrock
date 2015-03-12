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

#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/cc/cc_problem.cuh>
#include <gunrock/app/cc/cc_functor.cuh>


namespace gunrock {
namespace app {
namespace cc {

/**
 * @brief BC problem enactor class.
 *
 * @tparam INSTRUMENT Boolean type to show whether or not to collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>                           // Whether or not to collect per-CTA clock-count statistics
class CCEnactor : public EnactorBase
{
    // Members
    protected:

    /**
     * CTA duty kernel stats
     */
    util::KernelRuntimeStatsLifetime filter_kernel_stats;

    unsigned long long total_runtimes;              // Total working time by each CTA
    unsigned long long total_lifetimes;             // Total life time of each CTA
    unsigned long long total_queued;

    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    int                *vertex_flag;
    int                *edge_flag;

    /**
     * Current iteration
     */
    long long                           iteration;

    // Methods
    protected:

    /**
     * @brief Prepare the enactor for CC kernel call. Must be called prior to each CC search.
     *
     * @param[in] problem CC Problem object which holds the graph data and CC problem data to compute.
     * @param[in] filter_grid_size CTA occupancy for vertex mapping kernel call.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename ProblemData>
    cudaError_t Setup(
        ProblemData *problem,
        int filter_grid_size)
    {
        typedef typename ProblemData::SizeT         SizeT;
        typedef typename ProblemData::VertexId      VertexId;

        cudaError_t retval = cudaSuccess;

        do {

            //initialize runtime stats
            if (retval = filter_kernel_stats.Setup(filter_grid_size)) break;

            //Reset statistics
            iteration           = 0;
            total_runtimes      = 0;
            total_lifetimes     = 0;
            total_queued        = 0;

        } while (0);

        return retval;
    }

    public:

    /**
     * @brief CCEnactor default constructor
     */
    CCEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG),
        iteration(0),
        total_queued(0),
        vertex_flag(NULL),
        edge_flag(NULL)
    {
        vertex_flag = new int;
        edge_flag = new int;
        vertex_flag[0] = 0;
        edge_flag[0] = 0;
        }

    /**
     * @brief CCEnactor default destructor
     */
    ~CCEnactor()
    {

        if (vertex_flag) delete vertex_flag;
        if (edge_flag) delete edge_flag;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics about the last BFS search enacted.
     *
     * @param[out] total_queued Total queued elements in BFS kernel running.
     * @param[out] avg_duty Average kernel running duty (kernel run time/kernel lifetime).
     */
    template <typename VertexId>
    void GetStatistics(
        long long &total_queued,
        VertexId  &num_iter,
        double    &avg_duty)
    {
        cudaThreadSynchronize();

        total_queued = this->total_queued;
        num_iter = this->iteration;

        avg_duty = (total_lifetimes > 0) ?
            double (total_runtimes) / total_lifetimes : 0.0;
    }

    /** @} */

    /**
     * @brief Enacts a connected component computing on the specified graph.
     *
     * @tparam FilterPolicy Kernel policy for vertex mapping.
     * @tparam CCProblem CC Problem type.
     * @param[in] problem CCProblem object.
     * @param[in] max_grid_size Max grid size for CC kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template<
        typename FilterPolicy,
        typename CCProblem>
    cudaError_t EnactCC(
    CCProblem                          *problem,
    int                                 max_grid_size = 0)
    {
        typedef typename CCProblem::SizeT      SizeT;
        typedef typename CCProblem::VertexId   VertexId;

        typedef UpdateMaskFunctor<
            VertexId,
            SizeT,
            VertexId,
            CCProblem> UpdateMaskFunctor;

        typedef HookInitFunctor<
            VertexId,
            SizeT,
            VertexId,
            CCProblem> HookInitFunctor;

        typedef HookMinFunctor<
            VertexId,
            SizeT,
            VertexId,
            CCProblem> HookMinFunctor;

        typedef HookMaxFunctor<
            VertexId,
            SizeT,
            VertexId,
            CCProblem> HookMaxFunctor;

        typedef PtrJumpFunctor<
            VertexId,
            SizeT,
            VertexId,
            CCProblem> PtrJumpFunctor;

        typedef PtrJumpMaskFunctor<
            VertexId,
            SizeT,
            VertexId,
            CCProblem> PtrJumpMaskFunctor;

        typedef PtrJumpUnmaskFunctor<
            VertexId,
            SizeT,
            VertexId,
            CCProblem> PtrJumpUnmaskFunctor;

        cudaError_t retval = cudaSuccess;

        do {
            // Determine grid size(s)
            int filter_occupancy    = FilterPolicy::CTA_OCCUPANCY;
            int filter_grid_size    = MaxGridSize(filter_occupancy, max_grid_size);

            if (DEBUG) {
                printf("CC vertex map occupancy %d, level-grid size %d\n",
                        filter_occupancy, filter_grid_size);
            }

            // Lazy initialization
            if (retval = Setup(problem, filter_grid_size)) break;

            // Single-gpu graph slice
            typename CCProblem::GraphSlice *graph_slice = problem->graph_slices[0];
            typename CCProblem::DataSlice *data_slice = problem->d_data_slices[0];

            VertexId queue_index        = 0;        // Work queue index
            int selector                = 0;
            SizeT num_elements          = graph_slice->edges;
            bool queue_reset            = true;


            gunrock::oprtr::filter::Kernel<FilterPolicy, CCProblem, HookInitFunctor>
                <<<num_elements/FilterPolicy::THREADS+1, FilterPolicy::THREADS>>>(
                        0,  //iteration, not used in CC
                        queue_reset,
                        queue_index,
                        1,
                        num_elements,
                        NULL,//d_done,
                        graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                        NULL,   //pred_queue, not used in CC
                        graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                        data_slice,
                        NULL,   //d_visited_mask, not used in CC
                        work_progress,
                        graph_slice->frontier_elements[selector],           // max_in_queue
                        graph_slice->frontier_elements[selector^1],         // max_out_queue
                        this->filter_kernel_stats,
                        false);

            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter::Kernel Initial HookInit Operation failed", __FILE__, __LINE__))) break;

            // Pointer Jumping
            queue_index = 0;
            selector = 0;
            num_elements = graph_slice->nodes;
            queue_reset = true;

            // First Pointer Jumping Round
            vertex_flag[0] = 0;
            while (!vertex_flag[0]) {
                vertex_flag[0] = 1;
                if (retval = util::GRError(cudaMemcpy(
                                problem->data_slices[0]->d_vertex_flag,
                                vertex_flag,
                                sizeof(int),
                                cudaMemcpyHostToDevice),
                            "CCProblem cudaMemcpy vertex_flag to d_vertex_flag failed", __FILE__, __LINE__)) return retval;

                gunrock::oprtr::filter::Kernel<FilterPolicy, CCProblem, PtrJumpFunctor>
                    <<<filter_grid_size, FilterPolicy::THREADS>>>(
                            0,
                            queue_reset,
                            queue_index,
                            1,
                            num_elements,
                            NULL,//d_done,
                            graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                            NULL,
                            graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                            data_slice,
                            NULL,
                            work_progress,
                            graph_slice->frontier_elements[selector],           // max_in_queue
                            graph_slice->frontier_elements[selector^1],         // max_out_queue
                            this->filter_kernel_stats);

                if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter::Kernel First Pointer Jumping Round failed", __FILE__, __LINE__))) break;

                if (queue_reset) queue_reset = false;

                queue_index++;
                selector ^= 1;
                iteration++;

                if (retval = util::GRError(cudaMemcpy(
                                vertex_flag,
                                problem->data_slices[0]->d_vertex_flag,
                                sizeof(int),
                                cudaMemcpyDeviceToHost),
                            "CCProblem cudaMemcpy d_vertex_flag to vertex_flag failed", __FILE__, __LINE__)) return retval;


                // Check if done
                if (vertex_flag[0])
                {
                    //printf("first p_jump round. v_f\n");
                    break;
                }
            }

            queue_index        = 0;        // Work queue index
            selector                = 0;
            num_elements          = graph_slice->nodes;
            queue_reset            = true;

            gunrock::oprtr::filter::Kernel<FilterPolicy, CCProblem, UpdateMaskFunctor>
                <<<filter_grid_size, FilterPolicy::THREADS>>>(
                        0,
                        queue_reset,
                        queue_index,
                        1,
                        num_elements,
                        NULL,//d_done,
                        graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                        NULL,
                        graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                        data_slice,
                        NULL,
                        work_progress,
                        graph_slice->frontier_elements[selector],           // max_in_queue
                        graph_slice->frontier_elements[selector^1],         // max_out_queue
                        this->filter_kernel_stats);

            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter::Kernel Update Mask Operation failed", __FILE__, __LINE__))) break;

                iteration           = 1;

                edge_flag[0] = 0;
                while (!edge_flag[0]) {

                    queue_index             = 0;        // Work queue index
                    num_elements            = graph_slice->edges;
                    selector                = 0;
                    queue_reset             = true;

                    edge_flag[0] = 1;
                    if (retval = util::GRError(cudaMemcpy(
                                    problem->data_slices[0]->d_edge_flag,
                                    edge_flag,
                                    sizeof(int),
                                    cudaMemcpyHostToDevice),
                                "CCProblem cudaMemcpy edge_flag to d_edge_flag failed", __FILE__, __LINE__)) return retval;

                    if (iteration & 3) {
                        gunrock::oprtr::filter::Kernel<FilterPolicy, CCProblem, HookMinFunctor>
                            <<<num_elements/FilterPolicy::THREADS+1, FilterPolicy::THREADS>>>(
                                    0,
                                    queue_reset,
                                    queue_index,
                                    1,
                                    num_elements,
                                    NULL,//d_done,
                                    graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                                    NULL,
                                    graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                                    data_slice,
                                    NULL,
                                    work_progress,
                                    graph_slice->frontier_elements[selector],           // max_in_queue
                                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                                    this->filter_kernel_stats,
                                    false);
                    } else {
                        gunrock::oprtr::filter::Kernel<FilterPolicy, CCProblem, HookMaxFunctor>
                            <<<num_elements/FilterPolicy::THREADS+1, FilterPolicy::THREADS>>>(
                                    0,
                                    queue_reset,
                                    queue_index,
                                    1,
                                    num_elements,
                                    NULL,//d_done,
                                    graph_slice->frontier_queues.d_keys[selector],      // d_in_queue
                                    NULL,
                                    graph_slice->frontier_queues.d_keys[selector^1],    // d_out_queue
                                    data_slice,
                                    NULL,
                                    work_progress,
                                    graph_slice->frontier_elements[selector],           // max_in_queue
                                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                                    this->filter_kernel_stats,
                                    false);
                    }

                    if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter::Kernel Hook Min/Max Operation failed", __FILE__, __LINE__))) break;
                    if (queue_reset) queue_reset = false;
                    queue_index++;
                    selector ^= 1;
                    iteration++;

                    if (retval = util::GRError(cudaMemcpy(
                                    edge_flag,
                                    problem->data_slices[0]->d_edge_flag,
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost),
                                "CCProblem cudaMemcpy d_edge_flag to edge_flag failed", __FILE__, __LINE__)) return retval;
                    // Check if done
                    if (edge_flag[0])
                    {
                        //printf("edge flag after hook minmax.\n");
                        break;
                    }

                    ///////////////////////////////////////////
                    // Pointer Jumping
                    queue_index = 0;
                    selector = 0;
                    num_elements = graph_slice->nodes;
                    queue_reset = true;

                    // First Pointer Jumping Round
                    vertex_flag[0] = 0;
                    while (!vertex_flag[0]) {
                        vertex_flag[0] = 1;
                        if (retval = util::GRError(cudaMemcpy(
                                        problem->data_slices[0]->d_vertex_flag,
                                        vertex_flag,
                                        sizeof(int),
                                        cudaMemcpyHostToDevice),
                                    "CCProblem cudaMemcpy vertex_flag to d_vertex_flag failed", __FILE__, __LINE__)) return retval;

                        gunrock::oprtr::filter::Kernel<FilterPolicy, CCProblem, PtrJumpMaskFunctor>
                            <<<filter_grid_size, FilterPolicy::THREADS>>>(
                                    0,
                                    queue_reset,
                                    queue_index,
                                    1,
                                    num_elements,
                                    NULL,//d_done,
                                    graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                    NULL,
                                    graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                    data_slice,
                                    NULL,
                                    work_progress,
                                    graph_slice->frontier_elements[selector],           // max_in_queue
                                    graph_slice->frontier_elements[selector^1],         // max_out_queue
                                    this->filter_kernel_stats);

                        if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter::Kernel Pointer Jumping Mask failed", __FILE__, __LINE__))) break;
                        if (queue_reset) queue_reset = false;

                        queue_index++;
                        selector ^= 1;

                        if (retval = util::GRError(cudaMemcpy(
                                        vertex_flag,
                                        problem->data_slices[0]->d_vertex_flag,
                                        sizeof(int),
                                        cudaMemcpyDeviceToHost),
                                    "CCProblem cudaMemcpy d_vertex_flag to vertex_flag failed", __FILE__, __LINE__)) return retval;
                        // Check if done
                        if (vertex_flag[0]) break;
                    }

                    queue_index        = 0;        // Work queue index
                    selector                = 0;
                    num_elements          = graph_slice->nodes;
                    queue_reset            = true;

                    gunrock::oprtr::filter::Kernel<FilterPolicy, CCProblem, PtrJumpUnmaskFunctor>
                        <<<filter_grid_size, FilterPolicy::THREADS>>>(
                                0,
                                queue_reset,
                                queue_index,
                                1,
                                num_elements,
                                NULL,//d_done,
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                NULL,
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,
                                NULL,
                                work_progress,
                                graph_slice->frontier_elements[selector],           // max_in_queue
                                graph_slice->frontier_elements[selector^1],         // max_out_queue
                                this->filter_kernel_stats);

                    if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter::Kernel Pointer Jumping Unmask Operation failed", __FILE__, __LINE__))) break;

                    gunrock::oprtr::filter::Kernel<FilterPolicy, CCProblem, UpdateMaskFunctor>
                        <<<filter_grid_size, FilterPolicy::THREADS>>>(
                                0,
                                queue_reset,
                                queue_index,
                                1,
                                num_elements,
                                NULL,//d_done,
                                graph_slice->frontier_queues.d_values[selector],      // d_in_queue
                                NULL,
                                graph_slice->frontier_queues.d_values[selector^1],    // d_out_queue
                                data_slice,
                                NULL,
                                work_progress,
                                graph_slice->frontier_elements[selector],           // max_in_queue
                                graph_slice->frontier_elements[selector^1],         // max_out_queue
                                this->filter_kernel_stats);

                    if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(), "filter::Kernel Update Mask Operation failed", __FILE__, __LINE__))) break;

                    ///////////////////////////////////////////
                }


            if (retval) break;

        } while(0);

        if (DEBUG) printf("\nGPU CC Done.\n");
        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Enact Kernel Entry, specify KernelPolicy
     *
     * @tparam CCProblem CC Problem type. @see CCProblem
     * @param[in] problem Pointer to CCProblem object.
     * @param[in] max_grid_size Max grid size for CC kernel calls.
     *
     * \return cudaError_t object which indicates the success of all CUDA function calls.
     */
    template <typename CCProblem>
    cudaError_t Enact(
        CCProblem                      *problem,
        int                             max_grid_size = 0)
    {
        if (this->cuda_props.device_sm_version >= 300) {
            typedef gunrock::oprtr::filter::KernelPolicy<
                CCProblem,                         // Problem data type
                300,                                // CUDA_ARCH
                INSTRUMENT,                         // INSTRUMENT
                0,                                  // SATURATION QUIT
                true,                               // DEQUEUE_PROBLEM_SIZE
                8,                                  // MIN_CTA_OCCUPANCY
                8,                                  // LOG_THREADS
                1,                                  // LOG_LOAD_VEC_SIZE
                0,                                  // LOG_LOADS_PER_TILE
                5,                                  // LOG_RAKING_THREADS
                0,                                  // END_BITMASK (no bitmask for cc)
                8>                                  // LOG_SCHEDULE_GRANULARITY
                FilterPolicy;

                return EnactCC<FilterPolicy, CCProblem>(
                problem, max_grid_size);
        }

        //to reduce compile time, get rid of other architecture for now
        //TODO: add all the kernelpolicy settings for all archs

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */

};

} // namespace cc
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
