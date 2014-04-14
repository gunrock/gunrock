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

using namespace gunrock::oprtr::edge_map_forward;
using namespace gunrock::oprtr::edge_map_backward;
using namespace gunrock::oprtr::edge_map_partitioned;

namespace gunrock {
namespace oprtr {
namespace advance {

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
    int _LOG_RAKING_THREADS,
    int _WARP_GATHER_THRESHOLD,
    int _CTA_GATHER_THRESHOLD,
    int _LOG_SCHEDULE_GRANULARITY,
    int _LOG_BLOCKS,
    int _LIGHT_EDGE_THRESHOLD>

} //advance
} //oprtr
} //gunrock
