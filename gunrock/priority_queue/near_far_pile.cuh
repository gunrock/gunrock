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

struct PriorityQueue
{
    typedef _VertexId           VertexId;
    typedef _SizeT              SizeT;

    struct NearFarPile {
        VertexId                                    *d_queue;
        VertexId                                    *d_valid_near;
        VertexId                                    *d_valid_far;
        SizeT                                       queue_length;
        SizeT                                       max_queue_length;

        NearFarPile(SizeT max_q_len) :
            d_queue(NULL),
            d_valid_near(NULL),
            d_valid_far(NULL),
            queue_length(0),
            max_queue_length(max_q_len) {}

        virtual ~NearFarPile()
        {
            if (d_queue)    util::GRError(cudaFree(d_queue), "NearFarPile cudaFree d_queue failed", __FILE__, __LINE__);
            if (d_valid_near)    util::GRError(cudaFree(d_valid_near), "NearFarPile cudaFree d_valid_near failed", __FILE__, __LINE__);
            if (d_valid_far)    util::GRError(cudaFree(d_valid_far), "NearFarPile cudaFree d_valid_far failed", __FILE__, __LINE__);
        }
    };

    NearFarPile             *nf_pile;

    SizeT                   queue_length;
    SizeT                   max_queue_length;

    PriorityQueue() :
        queue_length(0),
        max_queue_length(UINT_MAX)
    {}

    virtual ~PriorityQueue()
    {
        delete nf_pile;
    }

    cudaError_t Init(SizeT edges, double queue_sizing)
    {
        cudaError_t retval = cudaSuccess;
        queue_length = 0;
        max_queue_length = edges*queue_sizing + 1;

        do {
            nf_pile = new NearFarPile(max_queue_length);
            
            if (retval = util::GRError(cudaMalloc(
                (void**)&nf_pile->d_queue,
                (nf_pile->max_queue_length+1)*sizeof(VertexId)),
                "NearFarPile cudaMalloc d_queue failed", __FILE__, __LINE__)) break;

            if (retval = util::GRError(cudaMalloc(
                (void**)&nf_pile->d_valid_near,
                (nf_pile->max_queue_length+1)*sizeof(VertexId)),
                "NearFarPile cudaMalloc d_valid_near failed", __FILE__, __LINE__)) break;

            if (retval = util::GRError(cudaMalloc(
                (void**)&nf_pile->d_valid_far,
                (nf_pile->max_queue_length+1)*sizeof(VertexId)),
                "NearFarPile cudaMalloc d_valid_far failed", __FILE__, __LINE__)) break;

        } while (0);

        return retval;
    }
};

} //namespace priority_queue
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
