// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kernel_policy.cuh
 *
 * @brief Kernel configuration policy for Backward Edge Expansion Kernel
 */



#pragma once

#include <gunrock/util/basic_utils.cuh>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/operators.cuh>

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace oprtr {
namespace edge_map_backward {


/**
 * @brief Kernel configuration policy for forward edge mapping kernels.
 *
 * Parameterizations of this type encapsulate our kernel-tuning parameters
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by parameterizing
 * them with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 *
 * @tparam _ProblemData                 Problem data type.
 * @tparam _CUDA_ARCH                   CUDA SM architecture to generate code for.
 * @tparam _INSTRUMENT                  Whether or not we want instrumentation logic generated
 * @tparam _MIN_CTA_OCCUPANCY           Lower bound on number of CTAs to have resident per SM (influences per-CTA smem cache sizes and register allocation/spills).
 * @tparam _LOG_THREADS                 Number of threads per CTA (log).
 * @tparam _LOG_LOAD_VEC_SIZE           Number of incoming frontier vertex-ids to dequeue in a single load (log).
 * @tparam _LOG_LOADS_PER_TILE          Number of such loads that constitute a tile of incoming frontier vertex-ids (log)
 * @tparam _LOG_RAKING_THREADS          Number of raking threads to use for prefix sum (log), range [5, LOG_THREADS]
 * @tparam _WARP_GATHER_THRESHOLD       Adjacency-list length above which we expand an that list using coarser-grained warp-based cooperative expansion
 *                                      (below which we perform fine-grained scan-based expansion)
 * @tparam _CTA_GATHER_THRESHOLD        Adjacency-list length above which we expand an that list using coarsest-grained CTA-based cooperative expansion
 *                                      (below which we perform warp-based expansion)
 * @tparam _LOG_SCHEDULE_GRANULARITY    The scheduling granularity of incoming frontier tiles (for even-share work distribution only) (log)
 */
template <
    typename _ProblemData,
    // Machine parameters
    int _CUDA_ARCH,
    // Behavioral control parameters
    bool _INSTRUMENT,
    // Tunable parameters
    int _MIN_CTA_OCCUPANCY,
    int _LOG_THREADS,
    int _LOG_LOAD_VEC_SIZE,
    int _LOG_LOADS_PER_TILE,
    int _WARP_GATHER_THRESHOLD,
    int _CTA_GATHER_THRESHOLD,
    int _LOG_SCHEDULE_GRANULARITY>

struct KernelPolicy
{
    //---------------------------------------------------------------------
    // Constants and typedefs
    //---------------------------------------------------------------------

    typedef _ProblemData                    ProblemData;
    typedef typename ProblemData::VertexId  VertexId;
    typedef typename ProblemData::SizeT     SizeT;

    enum {

        CUDA_ARCH                       = _CUDA_ARCH,
        INSTRUMENT                      = _INSTRUMENT,

        LOG_THREADS                     = _LOG_THREADS,
        THREADS                         = 1 << LOG_THREADS,

        LOG_LOAD_VEC_SIZE               = _LOG_LOAD_VEC_SIZE,
        LOAD_VEC_SIZE                   = 1 << LOG_LOAD_VEC_SIZE,

        LOG_LOADS_PER_TILE              = _LOG_LOADS_PER_TILE,
        LOADS_PER_TILE                  = 1 << LOG_LOADS_PER_TILE,

        LOG_WARPS                       = LOG_THREADS - GR_LOG_WARP_THREADS(CUDA_ARCH),
        WARPS                           = 1 << LOG_WARPS,

        LOG_TILE_ELEMENTS_PER_THREAD    = LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
        TILE_ELEMENTS_PER_THREAD        = 1 << LOG_TILE_ELEMENTS_PER_THREAD,

        LOG_TILE_ELEMENTS               = LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS,
        TILE_ELEMENTS                   = 1 << LOG_TILE_ELEMENTS,

        LOG_SCHEDULE_GRANULARITY        = _LOG_SCHEDULE_GRANULARITY,
        SCHEDULE_GRANULARITY            = 1 << LOG_SCHEDULE_GRANULARITY,

        WARP_GATHER_THRESHOLD           = _WARP_GATHER_THRESHOLD,
        CTA_GATHER_THRESHOLD            = _CTA_GATHER_THRESHOLD,
    };

    
    /**
     * @brief Shared memory storage type for the CTA
     */
    struct SmemStorage
    {
        // Persistent shared state for the CTA
        struct State {

            // Type describing four shared memory channels per warp for intra-warp communication
            typedef SizeT                       WarpComm[WARPS][4];

            // Whether or not we overflowed our outgoing frontier
            bool                                overflowed;

            // Shared work-processing limits
            util::CtaWorkDistribution<SizeT>    work_decomposition;

            // Shared memory channels for intra-warp communication
            volatile WarpComm                   warp_comm;
            int                                 cta_comm;
        } state;

        enum {
            // Amount of storage we can use for hashing scratch space under target occupancy
            MAX_SCRATCH_BYTES_PER_CTA       = (GR_SMEM_BYTES(CUDA_ARCH) / _MIN_CTA_OCCUPANCY)
                                                - sizeof(State)
                                                - 128,                                          // Fudge-factor to guarantee occupancy

            SCRATCH_ELEMENT_SIZE            = sizeof(SizeT)*2 + sizeof(VertexId),                   // Both gather offset and predecessor

            GATHER_ELEMENTS                 = MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE,
            PARENT_ELEMENTS                 = GATHER_ELEMENTS,
        };

        //union {
            
            // Scratch elements
            struct {
                SizeT                       gather_offsets[GATHER_ELEMENTS];
                SizeT                       gather_offsets2[GATHER_ELEMENTS];
                VertexId                    gather_predecessors[PARENT_ELEMENTS];
            };
        //};

    };

    enum {
        THREAD_OCCUPANCY                = GR_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
        SMEM_OCCUPANCY                  = GR_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),
        CTA_OCCUPANCY                   = GR_MIN(_MIN_CTA_OCCUPANCY, GR_MIN(GR_SM_CTAS(CUDA_ARCH), GR_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),

        VALID                           = (CTA_OCCUPANCY > 0),
    };
};


} // namespace edge_map_backward
} // namespace oprtr
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
