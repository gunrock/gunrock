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

#include <gunrock/util/basic_utils.cuh>
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

enum MODE {
    TWC_FORWARD,
    TWC_BACKWARD,
    LB
};

enum TYPE {
    V2V,
    V2E,
    E2V,
    E2E
};

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
