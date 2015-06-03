// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file sample_enactor.cuh
 * @brief Primitive problem enactor
 */

#pragma once

#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/test_utils.cuh>

#include <gunrock/oprtr/advance/kernel.cuh>
#include <gunrock/oprtr/advance/kernel_policy.cuh>
#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/filter/kernel_policy.cuh>

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/template/sample_problem.cuh>
#include <gunrock/app/template/sample_functor.cuh>

namespace gunrock {
namespace app {
namespace sample {

/**
 * @brief Primitive enactor class.
 * @tparam INSTRUMWENT Boolean indicate collect per-CTA clock-count statistics
 */
template<bool INSTRUMENT>
class SampleEnactor : public EnactorBase {
 protected:
    /**
     * A pinned, mapped word that the traversal kernels will signal when done
     */
    volatile int *done;
    int          *d_done;
    cudaEvent_t  throttle_event;

    /**
     * @brief Prepare the enactor for kernel call.
     * @param[in] problem Problem object holds both graph and primitive data.
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    template <typename ProblemData>
    cudaError_t Setup(ProblemData *problem) {
        typedef typename ProblemData::SizeT    SizeT;
        typedef typename ProblemData::VertexId VertexId;

        cudaError_t retval = cudaSuccess;

        // initialize the host-mapped "done"
        if (!done) {
            int flags = cudaHostAllocMapped;

            // allocate pinned memory for done
            if (retval = util::GRError(
                    cudaHostAlloc((void**)&done, sizeof(int) * 1, flags),
                    "Enactor cudaHostAlloc done failed",
                    __FILE__, __LINE__)) return retval;

            // map done into GPU space
            if (retval = util::GRError(
                    cudaHostGetDevicePointer((void**)&d_done, (void*) done, 0),
                    "Enactor cudaHostGetDevicePointer done failed",
                    __FILE__, __LINE__)) return retval;

            // create throttle event
            if (retval = util::GRError(
                    cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
                    "Enactor cudaEventCreateWithFlags throttle_event failed",
                    __FILE__, __LINE__)) return retval;
        }

        done[0] = -1;

        // graph slice
        typename ProblemData::GraphSlice *graph_slice = problem->graph_slices[0];
        // TODO: uncomment if using data_slice to store primitive-specific array
        //typename ProblemData::DataSlice *data_slice = problem->data_slices[0];

        do {
            // bind row-offsets and bit-mask texture
            cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
            oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref.channelDesc = row_offsets_desc;
            if (retval = util::GRError(
                    cudaBindTexture(
                        0,
                        oprtr::edge_map_forward::RowOffsetTex<SizeT>::ref,
                        graph_slice->d_row_offsets,
                        (graph_slice->nodes + 1) * sizeof(SizeT)),
                    "Enactor cudaBindTexture row_offset_tex_ref failed",
                    __FILE__, __LINE__)) break;
        } while (0);
        return retval;
    }

 public:
    /**
     * @brief Constructor
     */
    explicit SampleEnactor(bool DEBUG = false) :
        EnactorBase(EDGE_FRONTIERS, DEBUG), done(NULL), d_done(NULL) {}

    /**
     * @brief Destructor
     */
    virtual ~SampleEnactor() {
        if (done) {
            util::GRError(
                cudaFreeHost((void*)done),
                "Enactor cudaFreeHost done failed",
                __FILE__, __LINE__);

            util::GRError(
                cudaEventDestroy(throttle_event),
                "Enactor cudaEventDestroy throttle_event failed",
                __FILE__, __LINE__);
        }
    }

    /**
     * \addtogroup PublicInterface
     * @{
     */

    /**
     * @brief Obtain statistics the primitive enacted.
     * @param[out] num_iterations Number of iterations (BSP super-steps).
     */
    template <typename VertexId>
    void GetStatistics(VertexId &num_iterations) {
        cudaThreadSynchronize();
        num_iterations = enactor_stats.iteration;
        // TODO: code to extract more statistics if necessary
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
    cudaError_t EnactSample(
        CudaContext &context,
        Problem     *problem,
        int         max_grid_size = 0) {
        typedef typename Problem::SizeT    SizeT;
        typedef typename Problem::VertexId VertexId;

        typedef SampleFunctor<VertexId, SizeT, VertexId, Problem> Functor;

        cudaError_t retval = cudaSuccess;

        do {
            unsigned int *d_scanned_edges = NULL;

            // TODO: enactor code here

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
     * @param[in] context CudaContext pointer for ModernGPU APIs
     * @param[in] problem Pointer to Problem object.
     * @param[in] max_grid_size Max grid size for kernel calls.
     * @param[in] traversal_mode Traversal Mode for advance operator:
     *            Load-balanced or Dynamic cooperative
     *
     * \return cudaError_t object indicates the success of all CUDA functions.
     */
    template <typename Problem>
    cudaError_t Enact(
        CudaContext &context,
        Problem     *problem,
        int         max_grid_size  = 0,
        int         traversal_mode = 0) {
        if (this->cuda_props.device_sm_version >= 300) {
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
                FilterKernelPolicy;

            typedef oprtr::advance::KernelPolicy <
                Problem,             // Problem data type
                300,                 // CUDA_ARCH
                INSTRUMENT,          // INSTRUMENT
                1,                   // MIN_CTA_OCCUPANCY
                7,                   // LOG_THREADS
                8,                   // LOG_BLOCKS
                32 * 128,            // LIGHT_EDGE_THRESHOLD (used for LB)
                1,                   // LOG_LOAD_VEC_SIZE
                0,                   // LOG_LOADS_PER_TILE
                5,                   // LOG_RAKING_THREADS
                32,                  // WARP_GATHER_THRESHOLD
                128 * 4,             // CTA_GATHER_THRESHOLD
                7,                   // LOG_SCHEDULE_GRANULARITY
                oprtr::advance::TWC_FORWARD >
                ForwardAdvanceKernelPolicy;

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
                oprtr::advance::LB >
                LBAdvanceKernelPolicy;

            if (traversal_mode == 0) {
                return EnactSample<
                    LBAdvanceKernelPolicy, FilterKernelPolicy, Problem>(
                        context, problem, max_grid_size);
            } else {  // traversal_mode == 1
                return EnactSample<
                    ForwardAdvanceKernelPolicy, FilterKernelPolicy, Problem>(
                        context, problem, max_grid_size);
            }
        }

        // to reduce compile time, get rid of other architecture for now
        // TODO: add all the kernel policy setting for all architectures

        printf("Not yet tuned for this architecture\n");
        return cudaErrorInvalidDeviceFunction;
    }

    /** @} */
};

}  // namespace sample
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
