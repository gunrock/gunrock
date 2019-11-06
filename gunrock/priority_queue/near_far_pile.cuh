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

#include <gunrock/util/basic_utils.h>
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

/**
 * @brief PriorityQueue data structure which only has two level
 * of priorities (Near Far Pile).
 */
template <typename _VertexId, typename _SizeT>
struct PriorityQueue {
  typedef _VertexId VertexId;
  typedef _SizeT SizeT;

  struct NearFarPile {
    VertexId *d_queue[2];   /**< ping-pong buffer for near and far piles */
    VertexId *d_valid_near; /**< array which shows whether an element is valid
                               in the near pile */
    VertexId *d_valid_far; /**< array which shows whether an element is valid in
                              the far pile */
  };

  NearFarPile **nf_pile;   /**< host pointer for near far pile data */
  NearFarPile **d_nf_pile; /**< device pointer for near far pile data */

  SizeT queue_length;            /**< current queue length */
  unsigned int max_queue_length; /**< upper limit of the queue length */
  int selector; /**< binary switch for choosing from ping-pong buffers */
  SizeT level;

  PriorityQueue()
      : queue_length(0), max_queue_length(UINT_MAX), selector(0), level(0) {}

  virtual ~PriorityQueue() {
    if (nf_pile[0]->d_queue[0])
      util::GRError(cudaFree(nf_pile[0]->d_queue[0]),
                    "NearFarPile cudaFree d_queue failed", __FILE__, __LINE__);
    if (nf_pile[0]->d_queue[1])
      util::GRError(cudaFree(nf_pile[0]->d_queue[1]),
                    "NearFarPile cudaFree d_queue failed", __FILE__, __LINE__);
    if (nf_pile[0]->d_valid_near)
      util::GRError(cudaFree(nf_pile[0]->d_valid_near),
                    "NearFarPile cudaFree d_valid_near failed", __FILE__,
                    __LINE__);
    if (nf_pile[0]->d_valid_far)
      util::GRError(cudaFree(nf_pile[0]->d_valid_far),
                    "NearFarPile cudaFree d_valid_far failed", __FILE__,
                    __LINE__);
    if (d_nf_pile[0])
      util::GRError(cudaFree(d_nf_pile[0]),
                    "NearFarPile cudaFree d_nf_pile failed", __FILE__,
                    __LINE__);

    if (nf_pile) delete[] nf_pile;
    if (d_nf_pile) delete[] d_nf_pile;
  }

  /**
   * @brief Initialize near far pile
   *
   * @param[in] edges Number of edges in the CSR graph. the max queue length
   * will be 1 element larger.
   * @param[in] queue_sizing scaling factor of the near far pile.
   *
   * \return cudaError_t object which indicates the success of all CUDA function
   * calls.
   */
  cudaError_t Init(SizeT edges, double queue_sizing) {
    cudaError_t retval = cudaSuccess;
    queue_length = 0;
    // TODO: do not use queue_sizing for now, see if this works.
    max_queue_length = edges + 1;

    nf_pile = new NearFarPile *[1];
    d_nf_pile = new NearFarPile *[1];

    do {
      nf_pile[0] = new NearFarPile;
      if (retval = util::GRError(
              cudaMalloc((void **)&d_nf_pile[0], sizeof(NearFarPile)),
              "PriorityQueue cudaMalloc d_nf_pile failed", __FILE__, __LINE__))
        return retval;

      VertexId *d_queue_ping;
      if (retval = util::GRError(
              cudaMalloc((void **)&d_queue_ping,
                         (max_queue_length) * sizeof(VertexId)),
              "NearFarPile cudaMalloc d_queue_ping failed", __FILE__, __LINE__))
        break;
      nf_pile[0]->d_queue[0] = d_queue_ping;

      VertexId *d_queue_pong;
      if (retval = util::GRError(
              cudaMalloc((void **)&d_queue_pong,
                         (max_queue_length) * sizeof(VertexId)),
              "NearFarPile cudaMalloc d_queue_pong failed", __FILE__, __LINE__))
        break;
      nf_pile[0]->d_queue[1] = d_queue_pong;

      VertexId *d_valid_near;
      if (retval = util::GRError(
              cudaMalloc((void **)&d_valid_near,
                         (max_queue_length) * sizeof(VertexId)),
              "NearFarPile cudaMalloc d_valid_near failed", __FILE__, __LINE__))
        break;
      nf_pile[0]->d_valid_near = d_valid_near;

      VertexId *d_valid_far;
      if (retval = util::GRError(
              cudaMalloc((void **)&d_valid_far,
                         (max_queue_length) * sizeof(VertexId)),
              "NearFarPile cudaMalloc d_valid_far failed", __FILE__, __LINE__))
        break;
      nf_pile[0]->d_valid_far = d_valid_far;

      util::MemsetKernel<<<128, 128>>>(nf_pile[0]->d_valid_near, 0,
                                       max_queue_length);
      util::MemsetKernel<<<128, 128>>>(nf_pile[0]->d_valid_far, 0,
                                       max_queue_length);

      if (retval = util::GRError(
              cudaMemcpy(d_nf_pile[0], nf_pile[0], sizeof(NearFarPile),
                         cudaMemcpyHostToDevice),
              "NearFarPile cudaMemcpy nf_pile to d_nf_pile failed", __FILE__,
              __LINE__))
        return retval;

    } while (0);

    return retval;
  }
};

}  // namespace priority_queue
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
