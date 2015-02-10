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
 * @brief Kernel configuration policy for Forward Edge Expansion Kernel
 */



#pragma once

#include <gunrock/util/basic_utils.h>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/soa_tuple.cuh>
#include <gunrock/util/srts_grid.cuh>
#include <gunrock/util/srts_soa_details.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>
#include <gunrock/util/operators.cuh>

#include <gunrock/app/problem_base.cuh>

#include <gunrock/oprtr/edge_map_forward/kernel_policy.cuh>
#include <gunrock/oprtr/edge_map_backward/kernel_policy.cuh>
#include <gunrock/oprtr/edge_map_partitioned_backward/kernel_policy.cuh>
#include <gunrock/oprtr/edge_map_partitioned/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {
namespace advance {

/**
 * @brief Traversal Modes
 */
enum MODE {
    TWC_FORWARD,
    TWC_BACKWARD,
    LB_BACKWARD,
    LB,
    LB2
};

/**
 * @brief Four types of advance operator
 */
enum TYPE {
    V2V,
    V2E,
    E2V,
    E2E
};

/**
 * @brief opeartion to use for mgpu primitives
 */
enum REDUCE_OP {
    NONE,
    PLUS,
    MINUS,
    MULTIPLIES,
    MODULUS,
    BIT_OR,
    BIT_AND,
    BIT_XOR,
    MAXIMUM,
    MINIMUM
};

enum REDUCE_TYPE {
    EMPTY,
    VERTEX,
    EDGE
};

/**
 * @brief Kernel configuration policy for all three advance kernels (forward, backward, and load balanced).
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
 * @tparam _LOG_BLOCKS                  Number of blocks per grid (log).
 * @tparam _LIGHT_EDGE_THRESHOLD        Reserved for switching between two kernels in load balanced advance mode.
 * @tparam _LOG_LOAD_VEC_SIZE           Number of incoming frontier vertex-ids to dequeue in a single load (log).
 * @tparam _LOG_LOADS_PER_TILE          Number of such loads that constitute a tile of incoming frontier vertex-ids (log)
 * @tparam _LOG_RAKING_THREADS          Number of raking threads to use for prefix sum (log), range [5, LOG_THREADS]
 * @tparam _WARP_GATHER_THRESHOLD       Adjacency-list length above which we expand an that list using coarser-grained warp-based cooperative expansion
 *                                      (below which we perform fine-grained scan-based expansion)
 * @tparam _CTA_GATHER_THRESHOLD        Adjacency-list length above which we expand an that list using coarsest-grained CTA-based cooperative expansion
 *                                      (below which we perform warp-based expansion)
 * @tparam _LOG_SCHEDULE_GRANULARITY    The scheduling granularity of incoming frontier tiles (for even-share work distribution only) (log)
 * @tparam _ADVANCE_MODE                Enum type which shows the type of advance operator we use: TWC_FORWARD, TWC_BACKWARD, LB)
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
    int _LOG_BLOCKS,
    int _LIGHT_EDGE_THRESHOLD,
    int _LOG_LOAD_VEC_SIZE,                                             
    int _LOG_LOADS_PER_TILE,                                            
    int _LOG_RAKING_THREADS,                                            
    int _WARP_GATHER_THRESHOLD,                                          
    int _CTA_GATHER_THRESHOLD,                                           
    int _LOG_SCHEDULE_GRANULARITY,
    // Advance Mode and Type parameters
    MODE _ADVANCE_MODE>
struct KernelPolicy {

    typedef _ProblemData                    ProblemData;
    typedef typename ProblemData::VertexId  VertexId;
    typedef typename ProblemData::SizeT     SizeT;
    typedef typename ProblemData::Value     Value;

    static const MODE   ADVANCE_MODE = _ADVANCE_MODE;
    static const int    CTA_OCCUPANCY = _MIN_CTA_OCCUPANCY;

typedef gunrock::oprtr::edge_map_forward::KernelPolicy<
    _ProblemData,
    _CUDA_ARCH,
    _INSTRUMENT,
    _MIN_CTA_OCCUPANCY,
    _LOG_THREADS,
    _LOG_LOAD_VEC_SIZE,
    _LOG_LOADS_PER_TILE,
    _LOG_RAKING_THREADS,
    _WARP_GATHER_THRESHOLD,
    _CTA_GATHER_THRESHOLD,
    _LOG_SCHEDULE_GRANULARITY> THREAD_WARP_CTA_FORWARD;

typedef gunrock::oprtr::edge_map_backward::KernelPolicy<
    _ProblemData,
    _CUDA_ARCH,
    _INSTRUMENT,
    _MIN_CTA_OCCUPANCY,
    _LOG_THREADS,
    _LOG_LOAD_VEC_SIZE,
    _LOG_LOADS_PER_TILE,
    _LOG_RAKING_THREADS,
    _WARP_GATHER_THRESHOLD,
    _CTA_GATHER_THRESHOLD,
    _LOG_SCHEDULE_GRANULARITY> THREAD_WARP_CTA_BACKWARD;

typedef gunrock::oprtr::edge_map_partitioned::KernelPolicy<
    _ProblemData,
    _CUDA_ARCH,
    _INSTRUMENT,
    1,
    _LOG_THREADS,
    _LOG_BLOCKS,
    _LIGHT_EDGE_THRESHOLD> LOAD_BALANCED;
};

} //advance
} //oprtr
} //gunrock
