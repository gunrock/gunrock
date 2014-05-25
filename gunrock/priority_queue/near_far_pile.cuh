// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * near_far_pile.cuh
 *
 * @brief Base struct for priority queue
 */

#pragma once

#include <gunrock/util/basic_utils.cuh>
#include <gunrock/util/cuda_properties.cuh>
#include <gunrock/util/memset_kernel.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/error_utils.cuh>
#include <gunrock/util/multiple_buffering.cuh>
#include <gunrock/util/io/modified_load.cuh>
#include <gunrock/util/io/modified_store.cuh>

#include <vector>

namespace gunrock {
namespace priority_queue {

template <
    typename    _VertexId,
    typename    _SizeT>

struct NearFarPile
{
    typedef _VertexId           VertexId;
    typedef _SizeT              SizeT;


    VertexId                                    *queue;
    VertexId                                    *valid_near;
    VertexId                                    *valid_far;
    SizeT                                       priority_level;
    SizeT                                       queue_length;
    SizeT                                       max_queue_length;

    NearFarPile() {}

    virtual ~NearFarPile() {}

    //TODO: Init, set up some interfaces too.

};

} //namespace priority_queue
} //namespace gunrock
