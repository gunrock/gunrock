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
 * @brief Kernel configuration policy for Load balanced Edge Expansion Kernel
 */



#pragma once

#include <cub/cub.cuh>

namespace gunrock {
namespace oprtr {
namespace edge_map_partitioned_cull {

/**
 * @brief Kernel configuration policy for partitioned edge mapping kernels.
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
 */
template <
    typename _Problem,
    // Machine parameters
    int _CUDA_ARCH,
    // Behavioral control parameters
    //bool _INSTRUMENT,
    // Tunable parameters
    int _MAX_CTA_OCCUPANCY,
    int _LOG_THREADS,
    int _LOG_BLOCKS,
    int _LIGHT_EDGE_THRESHOLD>

struct KernelPolicy
{
    //---------------------------------------------------------------------
    // Constants and typedefs
    //---------------------------------------------------------------------

    typedef _Problem                    Problem;

    typedef typename Problem::VertexId  VertexId;
    typedef typename Problem::SizeT     SizeT;
    typedef typename Problem::Value     Value;

    enum {

        CUDA_ARCH                       = _CUDA_ARCH,
        //INSTRUMENT                      = _INSTRUMENT,

        LOG_THREADS                     = _LOG_THREADS,
        THREADS                         = 1 << LOG_THREADS,
        LOG_BLOCKS                      = _LOG_BLOCKS,
        BLOCKS                          = 1 << LOG_BLOCKS,
        LIGHT_EDGE_THRESHOLD            = _LIGHT_EDGE_THRESHOLD,
        WARP_SIZE                       = GR_WARP_THREADS(CUDA_ARCH),
        LOG_WARP_SIZE                   = 5,
        WARP_SIZE_MASK                  = WARP_SIZE -1,
        WARPS                           = THREADS / WARP_SIZE,
    };

    typedef cub::BlockScan<SizeT, THREADS, cub::BLOCK_SCAN_RAKING> BlockScanT;
    /**
     * @brief Shared memory storage type for the CTA
     */
    struct SmemStorage
    {
        enum {
            // Amount of storage we can use for hashing scratch space under target occupancy
            MAX_SCRATCH_BYTES_PER_CTA       = (GR_SMEM_BYTES(CUDA_ARCH) / _MAX_CTA_OCCUPANCY)
                                                - 128,                                          // Fudge-factor to guarantee occupancy

            SCRATCH_ELEMENT_SIZE            = sizeof(SizeT) * 2 + sizeof(VertexId) * 2,

            SCRATCH_ELEMENTS                 = (THREADS > MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE) ? MAX_SCRATCH_BYTES_PER_CTA / SCRATCH_ELEMENT_SIZE : THREADS,
        };

        // Scratch elements
        struct {
            SizeT                       output_offset[SCRATCH_ELEMENTS];
            SizeT                       row_offset   [SCRATCH_ELEMENTS];
            VertexId                    vertices     [SCRATCH_ELEMENTS];
            VertexId                    input_queue  [SCRATCH_ELEMENTS];
            SizeT                       block_offset;
            SizeT                      *d_output_counter;
            VertexId                   *d_labels;
            unsigned char              *d_visited_mask;
            SizeT                       block_output_start;
            SizeT                       block_output_end;
            SizeT                       block_output_size;
            SizeT                       block_input_end;
            SizeT                       iter_input_start;
            union {
                typename BlockScanT::TempStorage scan_space;
            } cub_storage;
        };
    };

    enum {
        THREAD_OCCUPANCY                = GR_SM_THREADS(CUDA_ARCH) >> LOG_THREADS,
        SMEM_OCCUPANCY                  = GR_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),
        CTA_OCCUPANCY                   = GR_MIN(_MAX_CTA_OCCUPANCY, GR_MIN(GR_SM_CTAS(CUDA_ARCH), GR_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),

        VALID                           = (CTA_OCCUPANCY > 0),
        ELEMENT_ID_MASK	                = ~(1<<(sizeof(VertexId)*8-2)),
    };
};


} // namespace edge_map_partitioned
} // namespace oprtr
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
