// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kernel.cuh
 *
 * @brief Priority Queue Kernel
 */

#pragma once

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/priority_queue/near_far_pile.cuh>
#include <gunrock/priority_queue/kernel_policy.cuh>

#include <gunrock/util/test_utils.cuh>

#include <moderngpu.cuh>

namespace gunrock {
namespace priority_queue {

/**
 * Arch dispatch
 */

/**
 * Not valid for this arch (default)
 */
template <typename KernelPolicy, typename ProblemData, typename PriorityQueue,
          typename Functor,
          bool VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch {
  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;
  typedef typename ProblemData::DataSlice DataSlice;
  typedef typename PriorityQueue::NearFarPile NearFarPile;

  static __device__ __forceinline__ void MarkVisit(VertexId *&vertex_in,
                                                   DataSlice *&problem,
                                                   SizeT &input_queue_length,
                                                   SizeT &node_num) {}

  static __device__ __forceinline__ void MarkNF(
      VertexId *&vertex_in, NearFarPile *&pq, DataSlice *&problem,
      SizeT &input_queue_length, unsigned int &lower_priority_score_limit,
      unsigned int &upper_priority_score_limit, SizeT &node_num) {}

  static __device__ __forceinline__ void MarkValid(VertexId *&vertex_in,
                                                   NearFarPile *&pq,
                                                   DataSlice *&problem,
                                                   SizeT &input_queue_length,
                                                   SizeT &node_num) {}

  static __device__ __forceinline__ void Compact(
      VertexId *&vertex_in, NearFarPile *&pq, int &selector,
      SizeT &input_queue_length, VertexId *&vertex_out, SizeT &v_out_offset,
      SizeT &far_pile_offset, SizeT &node_num) {}

  static __device__ __forceinline__ void Compact2(
      VertexId *&vertex_in, NearFarPile *&pq, int &selector,
      SizeT &input_queue_length, VertexId *&vertex_out, SizeT &v_out_offset,
      SizeT &node_num) {}
};

template <typename KernelPolicy, typename ProblemData, typename PriorityQueue,
          typename Functor>
struct Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor, true> {
  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;
  typedef typename ProblemData::DataSlice DataSlice;
  typedef typename PriorityQueue::NearFarPile NearFarPile;

  static __device__ __forceinline__ void MarkVisit(VertexId *&vertex_in,
                                                   DataSlice *&problem,
                                                   SizeT &input_queue_length,
                                                   SizeT &node_num) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int my_id = tid + bid * blockDim.x;

    if (my_id >= input_queue_length) return;

    unsigned int my_vert = vertex_in[my_id];
    // if (my_vert < 0 || my_vert >= node_num) return;
    if (my_vert >= node_num) return;
    problem->visit_lookup[my_vert] = my_id;
  }

  static __device__ __forceinline__ void MarkNF(
      VertexId *&vertex_in, NearFarPile *&pq, DataSlice *&problem,
      SizeT &input_queue_length, unsigned int &lower_priority_score_limit,
      unsigned int &upper_priority_score_limit, SizeT &node_num) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int my_id = tid + bid * blockDim.x;

    if (my_id >= input_queue_length) return;

    unsigned int bucket_max = UINT_MAX / problem->delta[0];
    unsigned int my_vert = vertex_in[my_id];
    // if (my_vert < 0 || my_vert >= node_num) { pq->d_valid_near[my_id] = 0;
    // pq->d_valid_far[my_id] = 0; return; }
    if (my_vert >= node_num) {
      pq->d_valid_near[my_id] = 0;
      pq->d_valid_far[my_id] = 0;
      return;
    }
    unsigned int bucket_id = Functor::ComputePriorityScore(my_vert, problem);
    bool valid = (my_id == problem->visit_lookup[my_vert]);
    // printf(" valid:%d, my_id: %d, my_vert: %d\n", valid, my_id, my_vert,
    // bucket_id);
    pq->d_valid_near[my_id] = (bucket_id < upper_priority_score_limit &&
                               bucket_id >= lower_priority_score_limit && valid)
                                  ? 1
                                  : 0;
    pq->d_valid_far[my_id] = (bucket_id >= upper_priority_score_limit &&
                              bucket_id < bucket_max && valid)
                                 ? 1
                                 : 0;
    // printf("valid near, far: %d, %d\n", pq->d_valid_near[my_id],
    // pq->d_valid_far[my_id]);
  }

  static __device__ __forceinline__ void MarkValid(VertexId *&vertex_in,
                                                   NearFarPile *&pq,
                                                   DataSlice *&problem,
                                                   SizeT &input_queue_length,
                                                   SizeT &node_num) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int my_id = tid + bid * blockDim.x;

    if (my_id >= input_queue_length) return;

    unsigned int my_vert = vertex_in[my_id];
    // if (my_vert < 0 || my_vert >= node_num) { pq->d_valid_near[my_id] = 0;
    // pq->d_valid_far[my_id] = 0; return; }
    if (my_vert >= node_num) {
      pq->d_valid_near[my_id] = 0;
      pq->d_valid_far[my_id] = 0;
      return;
    }
    pq->d_valid_near[my_id] = (my_id == problem->d_visit_lookup[my_vert]);
    pq->d_valid_far[my_id] = 0;
  }

  static __device__ __forceinline__ void Compact(
      VertexId *&vertex_in, NearFarPile *&pq, int &selector,
      SizeT &input_queue_length, VertexId *&vertex_out, SizeT &v_out_offset,
      SizeT &far_pile_offset, SizeT &node_num) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int my_id = bid * blockDim.x + tid;
    if (my_id >= input_queue_length) return;

    unsigned int my_vert = vertex_in[my_id];
    // if (my_vert<0 || my_vert >= node_num) return;
    if (my_vert >= node_num) return;
    unsigned int my_valid = pq->d_valid_near[my_id];

    if (my_valid == pq->d_valid_near[my_id + 1] - 1)
      vertex_out[my_valid + v_out_offset] = my_vert;

    my_valid = pq->d_valid_far[my_id];
    if (my_valid == pq->d_valid_far[my_id + 1] - 1)
      pq->d_queue[selector][my_valid + far_pile_offset] = my_vert;
  }

  static __device__ __forceinline__ void Compact2(
      VertexId *&vertex_in, NearFarPile *&pq, int &selector,
      SizeT &input_queue_length, VertexId *&vertex_out, SizeT &v_out_offset,
      SizeT &node_num) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int my_id = bid * blockDim.x + tid;
    if (my_id >= input_queue_length) return;

    unsigned int my_vert = vertex_in[my_id];
    // if (my_vert<0 || my_vert >= node_num) return;
    if (my_vert >= node_num) return;
    unsigned int my_valid = pq->d_valid_near[my_id];

    if (my_valid == pq->d_valid_near[my_id + 1] - 1)
      vertex_out[my_valid + v_out_offset] = my_vert;
  }
};

/**
 * @brief Mark the queue index in a lookup table, if multiple queue indices
 * map to same vertex ID, only one (the last one written) will be kept
 *
 * @tparam KernelPolicy Kernel policy type.
 * @tparam ProblemData Problem data type.
 * @tparam PriorityQueue PriorityQueue data type.
 * @tparam Functor Functor type.
 *
 * @param[in] vertex_in     Device pointer of the input vertex IDs
 * @param[in] problem       Problem object which stores user-defined priority
 * value
 * @param[in] input_queue_length Input queue length
 */
template <typename KernelPolicy, typename ProblemData, typename PriorityQueue,
          typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::BLOCKS) __global__
    void MarkVisit(typename KernelPolicy::VertexId *vertex_in,
                   typename ProblemData::DataSlice *problem,
                   typename KernelPolicy::SizeT input_queue_length,
                   typename KernelPolicy::SizeT node_num) {
  Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor>::MarkVisit(
      vertex_in, problem, input_queue_length, node_num);
}

/**
 * @brief Mark whether the vertex ID is valid in near pile/far pile
 *
 * @tparam KernelPolicy Kernel policy type.
 * @tparam ProblemData Problem data type.
 * @tparam PriorityQueue PriorityQueue data type.
 * @tparam Functor Functor type.
 *
 * @param[in] vertex_in     Device pointer of the input vertex IDs
 * @param[out] pq           PriorityQueue pointer which will be used to store
 * the near/far pile after the input vertices is splitted.
 * @param[in] problem       Problem object which stores user-defined priority
 * value
 * @param[in] input_queue_length Input queue length
 * @param[in] lower_priority_score_limit   Near pile priority value threshold
 * @param[in] upper_priority_score_limit   Far pile priority value threshold
 */
template <typename KernelPolicy, typename ProblemData, typename PriorityQueue,
          typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::BLOCKS) __global__
    void MarkNF(typename KernelPolicy::VertexId *vertex_in,
                typename PriorityQueue::NearFarPile *pq,
                typename ProblemData::DataSlice *problem,
                typename KernelPolicy::SizeT input_queue_length,
                unsigned int lower_priority_score_limit,
                unsigned int upper_priority_score_limit,
                typename KernelPolicy::SizeT node_num) {
  Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor>::MarkNF(
      vertex_in, pq, problem, input_queue_length, lower_priority_score_limit,
      upper_priority_score_limit, node_num);
}

template <typename KernelPolicy, typename ProblemData, typename PriorityQueue,
          typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::BLOCKS) __global__
    void MarkValid(typename KernelPolicy::VertexId *vertex_in,
                   typename PriorityQueue::NearFarPile *pq,
                   typename ProblemData::DataSlice *problem,
                   typename KernelPolicy::SizeT input_queue_length,
                   typename KernelPolicy::SizeT node_num) {
  Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor>::MarkValid(
      vertex_in, pq, problem, input_queue_length, node_num);
}

/**
 * @brief Compact the input queue into near far pile, remove the duplicate IDs,
 * append the newly generated far pile at the end of current priority queue
 * and output the vertices in the output queue
 *
 * @tparam KernelPolicy Kernel policy type.
 * @tparam ProblemData Problem data type.
 * @tparam PriorityQueue PriorityQueue data type.
 * @tparam Functor Functor type.
 *
 * @param[in] vertex_in     Device pointer of the input vertex IDs
 * @param[out] pq           PriorityQueue pointer which will be used to store
 * the near/far pile after the input vertices is splitted.
 * @param[in] selector      Binary switch for choosing from ping-pong buffers
 * @param[in] input_queue_length Input queue length
 * @param[out] vertex_out   Device pointer of the output vertex IDs, will be
 * used for any other following operators
 * @param[in] v_out_offset  The near pile queue offset
 * @param[in] far_pile_offset Where to append the newly generated elements in
 * the far pile
 */
template <typename KernelPolicy, typename ProblemData, typename PriorityQueue,
          typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::BLOCKS) __global__
    void Compact(typename KernelPolicy::VertexId *vertex_in,
                 typename PriorityQueue::NearFarPile *pq, int selector,
                 typename KernelPolicy::SizeT input_queue_length,
                 typename KernelPolicy::VertexId *vertex_out,
                 typename KernelPolicy::SizeT v_out_offset,
                 typename KernelPolicy::SizeT far_pile_offset,
                 typename KernelPolicy::SizeT node_num) {
  Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor>::Compact(
      vertex_in, pq, selector, input_queue_length, vertex_out, v_out_offset,
      far_pile_offset, node_num);
}

template <typename KernelPolicy, typename ProblemData, typename PriorityQueue,
          typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::BLOCKS) __global__
    void Compact2(typename KernelPolicy::VertexId *vertex_in,
                  typename PriorityQueue::NearFarPile *pq, int selector,
                  typename KernelPolicy::SizeT input_queue_length,
                  typename KernelPolicy::VertexId *vertex_out,
                  typename KernelPolicy::SizeT v_out_offset,
                  typename KernelPolicy::SizeT node_num) {
  Dispatch<KernelPolicy, ProblemData, PriorityQueue, Functor>::Compact2(
      vertex_in, pq, selector, input_queue_length, vertex_out, v_out_offset,
      node_num);
}

/**
 * @brief Split a queue into two parts (near/far piles) according to
 * its user-defined priority value.
 *
 * @tparam KernelPolicy Kernel policy type.
 * @tparam ProblemData Problem data type.
 * @tparam PriorityQueue PriorityQueue data type.
 * @tparam Functor Functor type.
 *
 * @param[in] vertex_in     Device pointer of the input vertex IDs
 * @param[out] pq           PriorityQueue pointer which will be used to store
 * the near/far pile after the input vertices is splitted.
 * @param[in] input_queue_length Input queue length
 * @param[in] problem       Problem object which stores user-defined priority
 * value
 * @param[out] vertex_out   Device pointer of the output vertex IDs, will be
 * used for any other following operators
 * @param[in] far_pile_offset Where to append the newly generated elements in
 * the far pile
 * @param[in] lower_limit   Near pile priority value threshold
 * @param[in] upper_limit   Far pile priority value threshold
 * @param[in] context       CudaContext pointer for moderngpu APIs
 */
template <typename KernelPolicy, typename ProblemData, typename PriorityQueue,
          typename Functor>
unsigned int Bisect(typename KernelPolicy::VertexId *vertex_in,
                    PriorityQueue *pq,
                    typename KernelPolicy::SizeT input_queue_length,
                    typename ProblemData::DataSlice *problem,
                    typename KernelPolicy::VertexId *vertex_out,
                    typename KernelPolicy::SizeT far_pile_offset,
                    unsigned int lower_limit, unsigned int upper_limit,
                    CudaContext &context,
                    typename KernelPolicy::SizeT node_num) {
  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;
  typename PriorityQueue::NearFarPile *nf_pile = pq->d_nf_pile[0];

  int block_num =
      (input_queue_length + KernelPolicy::THREADS - 1) / KernelPolicy::THREADS;
  unsigned int close_size[1];
  unsigned int far_size[1];
  close_size[0] = 0;
  far_size[0] = 0;
  if (input_queue_length > 0) {
    MarkVisit<KernelPolicy, ProblemData, PriorityQueue, Functor>
        <<<block_num, KernelPolicy::THREADS>>>(vertex_in, problem,
                                               input_queue_length, node_num);
    // MarkNF
    MarkNF<KernelPolicy, ProblemData, PriorityQueue, Functor>
        <<<block_num, KernelPolicy::THREADS>>>(vertex_in, nf_pile, problem,
                                               input_queue_length, lower_limit,
                                               upper_limit, node_num);

    // Scan(near)
    // Scan(far)
    Scan<mgpu::MgpuScanTypeExc>(
        pq->nf_pile[0]->d_valid_near, input_queue_length + 1, 0,
        mgpu::plus<VertexId>(), (VertexId *)0, (VertexId *)0,
        pq->nf_pile[0]->d_valid_near, context);
    Scan<mgpu::MgpuScanTypeExc>(
        pq->nf_pile[0]->d_valid_far, input_queue_length + 1, 0,
        mgpu::plus<VertexId>(), (VertexId *)0, (VertexId *)0,
        pq->nf_pile[0]->d_valid_far, context);
    // Compact
    Compact<KernelPolicy, ProblemData, PriorityQueue, Functor>
        <<<block_num, KernelPolicy::THREADS>>>(vertex_in, nf_pile, pq->selector,
                                               input_queue_length, vertex_out,
                                               0, far_pile_offset, node_num);
    // get output_near_length
    // get output_far_length
    cudaMemcpy(&close_size[0],
               pq->nf_pile[0]->d_valid_near + input_queue_length,
               sizeof(VertexId), cudaMemcpyDeviceToHost);
    cudaMemcpy(&far_size[0], pq->nf_pile[0]->d_valid_far + input_queue_length,
               sizeof(VertexId), cudaMemcpyDeviceToHost);
  }
  // Update near/far length
  pq->queue_length = far_pile_offset + far_size[0];
  return close_size[0];
}

/**
 * @brief Remove redundant elements in a queue
 *
 * @tparam KernelPolicy Kernel policy type.
 * @tparam ProblemData Problem data type.
 * @tparam PriorityQueue PriorityQueue data type.
 * @tparam Functor Functor type.
 *
 * @param[in] vertex_in     Device pointer of the input vertex IDs
 * @param[out] pq           PriorityQueue pointer which will be used to store
 * the near/far pile after the input vertices is splitted.
 * @param[in] input_queue_length Input queue length
 * @param[in] problem       Problem object which stores user-defined priority
 * value
 * @param[out] vertex_out   Device pointer of the output vertex IDs, will be
 * used for any other following operators
 * @param[in] far_pile_offset Where to append the newly generated elements in
 * the far pile
 * @param[in] lower_limit   Near pile priority value threshold
 * @param[in] upper_limit   Far pile priority value threshold
 * @param[in] context       CudaContext pointer for moderngpu APIs
 */
template <typename KernelPolicy, typename ProblemData, typename PriorityQueue,
          typename Functor>
unsigned int RemoveInvalid(typename KernelPolicy::VertexId *vertex_in,
                           PriorityQueue *pq,
                           typename KernelPolicy::SizeT input_queue_length,
                           typename ProblemData::DataSlice *problem,
                           typename KernelPolicy::VertexId *vertex_out,
                           CudaContext &context,
                           typename KernelPolicy::SizeT node_num) {
  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;
  typename PriorityQueue::NearFarPile *nf_pile = pq->d_nf_pile[0];

  int block_num =
      (input_queue_length + KernelPolicy::THREADS - 1) / KernelPolicy::THREADS;
  unsigned int close_size[1];
  close_size[0] = 0;
  if (input_queue_length > 0) {
    MarkVisit<KernelPolicy, ProblemData, PriorityQueue, Functor>
        <<<block_num, KernelPolicy::THREADS>>>(vertex_in, problem,
                                               input_queue_length, node_num);
    // MarkValid
    MarkValid<KernelPolicy, ProblemData, PriorityQueue, Functor>
        <<<block_num, KernelPolicy::THREADS>>>(vertex_in, nf_pile, problem,
                                               input_queue_length, node_num);

    // Scan(near)
    // Scan(far)
    Scan<mgpu::MgpuScanTypeExc>(
        pq->nf_pile[0]->d_valid_near, input_queue_length + 1, 0,
        mgpu::plus<VertexId>(), (VertexId *)0, (VertexId *)0,
        pq->nf_pile[0]->d_valid_near, context);
    // Compact2
    Compact2<KernelPolicy, ProblemData, PriorityQueue, Functor>
        <<<block_num, KernelPolicy::THREADS>>>(vertex_in, nf_pile, pq->selector,
                                               input_queue_length, vertex_out,
                                               0, node_num);
    // get output_length
    cudaMemcpy(&close_size[0],
               pq->nf_pile[0]->d_valid_near + input_queue_length,
               sizeof(VertexId), cudaMemcpyDeviceToHost);
  }
  // Update queue length
  return close_size[0];
}

}  // namespace priority_queue
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
