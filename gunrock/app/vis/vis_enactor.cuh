// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file vis_enactor.cuh
 *
 * @brief Primitive problem enactor for Vertex-Induced Subgraph
*/

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/vis/vis_problem.cuh>
#include <gunrock/app/vis/vis_functor.cuh>

namespace gunrock {
namespace app {
namespace vis {

/**
 * @brief Primitive enactor class.
 *
 * @tparam _Problem
 * @tparam INSTRUMWENT
 * @tparam _DEBUG
 * @tparam _SIZE_CHECK
 */
template <
    typename _Problem,
    bool _INSTRUMENT,
    bool _DEBUG,
    bool _SIZE_CHECK>
class VISEnactor :
    public EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK> {
  protected:

    /**
     * @brief Prepare the enactor for kernel call.
     *
     * @param[in] problem Problem object holds both graph and primitive data.
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    template <typename ProblemData>
    cudaError_t Setup(ProblemData *problem) {
        typedef typename ProblemData::SizeT    SizeT;
        typedef typename ProblemData::VertexId VertexId;

        cudaError_t retval = cudaSuccess;

        /*
        GraphSlice<SizeT, VertexId, Value>*
            graph_slice = problem->graph_slices[0];
        typename ProblemData::DataSlice*
            data_slice = problem->data_slices[0];
        */

        return retval;
    }

  public:
    typedef _Problem                   Problem;
    typedef typename Problem::SizeT    SizeT;
    typedef typename Problem::VertexId VertexId;
    typedef typename Problem::Value    Value;
    static const bool INSTRUMENT = _INSTRUMENT;
    static const bool DEBUG      = _DEBUG;
    static const bool SIZE_CHECK = _SIZE_CHECK;

    /**
     * @brief Primitive Constructor.
     *
     * @param[in] gpu_idx GPU indices
     */
    VISEnactor(int *gpu_idx):
        EnactorBase<typename _Problem::SizeT, _DEBUG, _SIZE_CHECK>(
            EDGE_FRONTIERS, 1, gpu_idx) {}

    /**
     * @brief Primitive Destructor.
     */
    virtual ~VISEnactor() {}

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics the primitive enacted.
     *
     * @tparam VertexId
     *
     * @param[out] num_iterations Number of iterations (BSP super-steps).
     */
    template <typename VertexId>
    void GetStatistics(VertexId &num_iterations) {
        cudaThreadSynchronize();
        num_iterations = this->enactor_stats->iteration;
        // TODO(developer): code to extract more statistics if necessary
    }

    /** @} */

    /**
     * @brief Enacts computing on the specified graph.
     *
     * @tparam AdvanceKernelPolicy Kernel policy for advance operator.
     * @tparam FilterKernelPolicy Kernel policy for filter operator.
     * @tparam Problem Problem type.
     *
     * @param[in] context CudaContext pointer for ModernGPU APIs
     * @param[in] problem Problem object.
     * @param[in] max_grid_size Max grid size for kernel calls.
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    template <
        typename AdvanceKernelPolicy,
        typename FilterKernelPolicy,
        typename Problem >
    cudaError_t EnactVIS(
        ContextPtr  context,
        Problem*    problem,
        int         max_grid_size = 0) {
        typedef typename Problem::SizeT    SizeT;
        typedef typename Problem::VertexId VertexId;

        // Define functors for primitive
        typedef VISFunctor<VertexId, SizeT, VertexId, Problem> Functor;

        cudaError_t retval = cudaSuccess;

        FrontierAttribute<SizeT>*
            frontier_attribute = &this->frontier_attribute[0];
        EnactorStats *
            enactor_stats = &this->enactor_stats[0];
        typename Problem::DataSlice*
            data_slice = problem->data_slices[0];
        util::DoubleBuffer<SizeT, VertexId, Value>*
            frontier_queue     = &data_slice->frontier_queues[0];
        util::CtaWorkProgressLifetime*
            work_progress = &this->work_progress[0];
        cudaStream_t stream = data_slice->streams[0];

        do {
            SizeT* d_scanned_edges = NULL;

            // Determine grid size(s)
            if (DEBUG) {
                printf("Iteration, Edge map queue, Vertex map queue\n");
                printf("0");
            }
            fflush(stdout);

            // Lazy initialization
            if (retval = Setup(problem)) break;
            if (retval = EnactorBase<
                typename _Problem::SizeT,
                _DEBUG, _SIZE_CHECK>::Setup(
                    problem,
                    max_grid_size,
                    AdvanceKernelPolicy::CTA_OCCUPANCY,
                    FilterKernelPolicy::CTA_OCCUPANCY)) break;

            // Single-gpu graph slice
            GraphSlice<SizeT, VertexId, Value>*
                graph_slice  = problem->graph_slices[0];
            typename Problem::DataSlice*
                d_data_slice = problem->d_data_slices[0];

            if (retval = util::GRError(cudaMalloc(
                            (void**)&d_scanned_edges,
                            graph_slice->edges * sizeof(SizeT)),
                        "Problem cudaMalloc d_scanned_edges failed",
                        __FILE__, __LINE__)) return retval;

            frontier_attribute->queue_length = graph_slice->nodes;
            frontier_attribute->queue_index  = 0;
            frontier_attribute->selector     = 0;
            frontier_attribute->queue_reset  = true;

            // filter: intput all vertices in graph, output selected vertices
            oprtr::filter::Kernel<FilterKernelPolicy, Problem, Functor><<<
                enactor_stats->filter_grid_size,
                FilterKernelPolicy::THREADS, 0, stream>>>(
                enactor_stats->iteration + 1,
                frontier_attribute->queue_reset,
                frontier_attribute->queue_index,
                frontier_attribute->queue_length,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                NULL,
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                d_data_slice,
                NULL,
                work_progress[0],
                frontier_queue->keys[frontier_attribute->selector  ].GetSize(),
                frontier_queue->keys[frontier_attribute->selector^1].GetSize(),
                enactor_stats->filter_kernel_stats);

            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
                "filter_forward::Kernel failed", __FILE__, __LINE__))) break;

            frontier_attribute->queue_index++;
            frontier_attribute->selector ^= 1;

            if (retval = work_progress->GetQueueLength(
                frontier_attribute->queue_index,
                frontier_attribute->queue_length)) break;

            /*if (true) {
                printf("filter queue length: %lld",
                       (long long) frontier_attribute->queue_length);
                util::DisplayDeviceResults(
                    problem->data_slices[0]->mask.GetPointer(util::DEVICE),
                    graph_slice->nodes);
                printf("input queue for advance:\n");
                util::DisplayDeviceResults(
                    frontier_queue->keys[frontier_attribute->selector].GetPointer(util::DEVICE),
                    frontier_attribute->queue_length);
            }*/

            oprtr::advance::LaunchKernel<AdvanceKernelPolicy, Problem, Functor>(
                enactor_stats[0],
                frontier_attribute[0],
                d_data_slice,
                (VertexId*)NULL,
                (bool*    )NULL,
                (bool*    )NULL,
                d_scanned_edges,
                frontier_queue->keys[frontier_attribute->selector  ].GetPointer(util::DEVICE),
                frontier_queue->keys[frontier_attribute->selector^1].GetPointer(util::DEVICE),
                (VertexId*)NULL,
                (VertexId*)NULL,
                graph_slice->row_offsets   .GetPointer(util::DEVICE),
                graph_slice->column_indices.GetPointer(util::DEVICE),
                (SizeT*   )NULL,
                (VertexId*)NULL,
                graph_slice->nodes,
                graph_slice->edges,
                work_progress[0],
                context[0],
                stream,
                gunrock::oprtr::advance::V2V);//,
                //false,
                //true);

            if (DEBUG && (retval = util::GRError(cudaThreadSynchronize(),
            "Advance::LaunchKernel failed", __FILE__, __LINE__))) break;

            frontier_attribute->queue_index++;

            if (true) {
                if (retval = work_progress->GetQueueLength(
                        frontier_attribute->queue_index,
                        frontier_attribute->queue_length)) break;
                printf("advance queue length: %lld",
                       (long long) frontier_attribute->queue_length);
            }

            // TODO: extract graph with proper format (edge list, csr, etc.)

            if (d_scanned_edges) cudaFree(d_scanned_edges);

        } while (0);

        if (DEBUG) {
            printf("\nGPU Primitive Enact Done.\n");
        }

        return retval;
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Primitive enact kernel entry.
     *
     * @tparam Problem Problem type. @see Problem
     *
     * @param[in] context CudaContext pointer for ModernGPU APIs.
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Max grid size for kernel calls.
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    template <typename Problem>
    cudaError_t Enact(
        ContextPtr context,
        Problem*   problem,
        int        max_grid_size = 0) {
        int min_sm_version = -1;
        for (int i = 0; i < this->num_gpus; i++) {
            if (min_sm_version == -1 ||
                this->cuda_props[i].device_sm_version < min_sm_version) {
                min_sm_version = this->cuda_props[i].device_sm_version;
            }
        }

        if (min_sm_version >= 300) {
            typedef oprtr::filter::KernelPolicy <
            Problem,             // Problem data type
            300,                 // CUDA_ARCH
            INSTRUMENT,          // INSTRUMENT
            0,                   // SATURATION QUIT
            true,                // DEQUEUE_PROBLEM_SIZE
            8,                   // MIN_CTA_OCCUPANCY
            8,                   // LOG_THREADS
            1,                   // LOG_LOAD_VEC_SIZE
            0,                   // LOG_LOADS_PER_TILE
            5,                   // LOG_RAKING_THREADS
            5,                   // END_BITMASK_CULL
            8 >                  // LOG_SCHEDULE_GRANULARITY
            FilterPolicy;

            typedef oprtr::advance::KernelPolicy <
            Problem,             // Problem data type
            300,                 // CUDA_ARCH
            INSTRUMENT,          // INSTRUMENT
            1,                   // MIN_CTA_OCCUPANCY
            10,                  // LOG_THREADS
            8,                   // LOG_BLOCKS
            32 * 128,            // LIGHT_EDGE_THRESHOLD (used for LB)
            1,                   // LOG_LOAD_VEC_SIZE
            0,                   // LOG_LOADS_PER_TILE
            5,                   // LOG_RAKING_THREADS
            32,                  // WARP_GATHER_THRESHOLD
            128 * 4,             // CTA_GATHER_THRESHOLD
            7,                   // LOG_SCHEDULE_GRANULARITY
            oprtr::advance::LB > AdvancePolicy;

            return EnactVIS<AdvancePolicy, FilterPolicy, Problem> (
                context, problem, max_grid_size);
        }

        // to reduce compile time, get rid of other architecture for now
        // TODO: add all the kernel policy setting for all architectures

        printf("Not yet tuned for this architecture.\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */
};

}  // namespace vis
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
