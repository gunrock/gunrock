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
 * @brief Backward Edge Map Kernel Entrypoint
 */

#pragma once
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/edge_map_backward/cta.cuh>

#include <gunrock/oprtr/advance/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {
namespace edge_map_backward {

/**
 * @brief Structure for invoking CTA processing tile over all elements.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
struct Sweep {
  static __device__ __forceinline__ void Invoke(
      typename KernelPolicy::VertexId &queue_index,
      typename KernelPolicy::VertexId *&d_unvisited_node_queue,
      typename KernelPolicy::VertexId *&d_unvisited_index_queue,
      bool *&d_frontier_bitmap_in, bool *&d_frontier_bitmap_out,
      typename KernelPolicy::SizeT *&d_row_offsets,
      typename KernelPolicy::VertexId *&d_column_indices,
      typename ProblemData::DataSlice *&problem,
      typename KernelPolicy::SmemStorage &smem_storage,
      util::CtaWorkProgress<typename KernelPolicy::SizeT> &work_progress,
      util::CtaWorkDistribution<typename KernelPolicy::SizeT>
          &work_decomposition,
      gunrock::oprtr::advance::TYPE &ADVANCE_TYPE) {
    typedef Cta<KernelPolicy, ProblemData, Functor> Cta;
    typedef typename KernelPolicy::SizeT SizeT;

    // Determine threadblock's work range
    util::CtaWorkLimits<SizeT> work_limits;
    work_decomposition
        .template GetCtaWorkLimits<KernelPolicy::LOG_TILE_ELEMENTS,
                                   KernelPolicy::LOG_SCHEDULE_GRANULARITY>(
            work_limits);

    // Return if we have no work to do
    if (!work_limits.elements) {
      return;
    }

    // CTA processing abstraction
    Cta cta(queue_index, smem_storage, d_unvisited_node_queue,
            d_unvisited_index_queue, d_frontier_bitmap_in,
            d_frontier_bitmap_out, d_row_offsets, d_column_indices, problem,
            work_progress, ADVANCE_TYPE);

    // Process full tiles
    while (work_limits.offset < work_limits.guarded_offset) {
      cta.ProcessTile(work_limits.offset);
      work_limits.offset += KernelPolicy::TILE_ELEMENTS;
    }

    // Clean up last partial tile with guarded-i/o
    if (work_limits.guarded_elements) {
      cta.ProcessTile(work_limits.offset, work_limits.guarded_elements);
    }
  }
};

/**
 * Arch dispatch
 */

/**
 * Not valid for this arch (default)
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam VALID
 */
template <typename KernelPolicy, typename ProblemData, typename Functor,
          bool VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch {
  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;
  typedef typename ProblemData::DataSlice DataSlice;

  static __device__ __forceinline__ void Kernel(
      bool &queue_reset, VertexId &queue_index, SizeT &num_elements,
      VertexId *&d_unvisited_node_queue, VertexId *&d_unvisited_index_queue,
      bool *&d_frontier_bitmap_in, bool *&d_frontier_bitmap_out,
      SizeT *&d_row_offsets, VertexId *&d_column_indices, DataSlice *&problem,
      util::CtaWorkProgress<SizeT> &work_progress,
      util::KernelRuntimeStats &kernel_stats,
      gunrock::oprtr::advance::TYPE &ADVANCE_TYPE) {
    // empty
  }
};

/**
 * @brief Kernel dispatch code for different architectures.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam ProblemData Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
struct Dispatch<KernelPolicy, ProblemData, Functor, true> {
  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;
  typedef typename ProblemData::DataSlice DataSlice;

  static __device__ __forceinline__ void Kernel(
      bool &queue_reset, VertexId &queue_index, SizeT &num_elements,
      VertexId *&d_unvisited_node_queue, VertexId *&d_unvisited_index_queue,
      bool *&d_frontier_bitmap_in, bool *&d_frontier_bitmap_out,
      SizeT *&d_row_offsets, VertexId *&d_column_indices, DataSlice *&problem,
      util::CtaWorkProgress<SizeT> &work_progress,
      util::KernelRuntimeStats &kernel_stats,
      gunrock::oprtr::advance::TYPE &ADVANCE_TYPE) {
    // Shared storage for the kernel
    __shared__ typename KernelPolicy::SmemStorage smem_storage;

    // If instrument flag is set, track kernel stats
    // if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
    //    kernel_stats.MarkStart();
    //}

    // workprogress reset
    // if (queue_reset)
    //{
    //    if (threadIdx.x < util::CtaWorkProgress::COUNTERS) {
    // Reset all counters
    //        work_progress.template Reset<SizeT>();
    //    }
    //}

    // Determine work decomposition
    if (threadIdx.x == 0) {
      // Obtain problem size
      if (queue_reset) {
        work_progress.template StoreQueueLength<SizeT>(num_elements,
                                                       queue_index);
      } else {
        num_elements =
            work_progress.template LoadQueueLength<SizeT>(queue_index);

        // Signal to host that we're done
        // if (num_elements == 0) {
        //    if (d_done) d_done[0] = num_elements;
        //}
      }

      // Initialize work decomposition in smem
      smem_storage.state.work_decomposition
          .template Init<KernelPolicy::LOG_SCHEDULE_GRANULARITY>(num_elements,
                                                                 gridDim.x);

      // Reset our next outgoing queue counter to zero
      work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);
    }

    // Barrier to protect work decomposition
    __syncthreads();

    Sweep<KernelPolicy, ProblemData, Functor>::Invoke(
        queue_index,
        // num_gpus,
        d_unvisited_node_queue, d_unvisited_index_queue, d_frontier_bitmap_in,
        d_frontier_bitmap_out, d_row_offsets, d_column_indices, problem,
        smem_storage, work_progress, smem_storage.state.work_decomposition,
        ADVANCE_TYPE);

    // if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
    //    kernel_stats.MarkStop();
    //    kernel_stats.Flush();
    //}
  }
};

/**
 * @brief Backward edge map kernel entry point.
 *
 * @tparam KernelPolicy Kernel policy type for backward edge mapping.
 * @tparam ProblemData Problem data type for backward edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] queue_reset               If reset queue counter
 * @param[in] queue_index               Current frontier queue counter index
 * @param[in] num_elements              Number of elements
 * @param[in] d_unvisited_node_queue    Incoming frontier queue
 * @param[in] d_unvisited_index_queue   Incoming frontier index queue
 * @param[in] d_frontier_bitmap_in      Incoming frontier bitmap (set for nodes
 * in the frontier)
 * @param[in] d_frontier_bitmap_out     Outgoing frontier bitmap (set for nodes
 * in the next layer of frontier)
 * @param[in] d_row_offsets             Row offsets queue
 * @param[in] d_column_indices          Column indices queue
 * @param[in] problem                   Device pointer to the problem object
 * @param[in] work_progress             queueing counters to record work
 * progress
 * @param[in] kernel_stats              Per-CTA clock timing statistics (used
 * when KernelPolicy::INSTRUMENT is set)
 */
template <typename KernelPolicy, typename ProblemData, typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY) __global__
    void Kernel(
        bool queue_reset,  // If reset queue
        typename KernelPolicy::VertexId
            queue_index,  // Current frontier queue counter index
        typename KernelPolicy::SizeT num_elements,  // Number of Elements
        typename KernelPolicy::VertexId *
            d_unvisited_node_queue,  // Incoming and output unvisited node queue
        typename KernelPolicy::VertexId *d_unvisited_index_queue,
        bool *d_frontier_bitmap_in,   // Incoming frontier bitmap
        bool *d_frontier_bitmap_out,  // Outcoming frontier bitmap
        typename KernelPolicy::SizeT *d_row_offsets,
        typename KernelPolicy::VertexId *d_column_indices,
        typename ProblemData::DataSlice *problem,  // Problem Object
        util::CtaWorkProgress<typename KernelPolicy::SizeT>
            work_progress,  // Atomic workstealing and queueing counters
        util::KernelRuntimeStats
            kernel_stats,  // Per-CTA clock timing statistics (used when
                           // KernelPolicy::INSTRUMENT)
        gunrock::oprtr::advance::TYPE ADVANCE_TYPE =
            gunrock::oprtr::advance::V2V) {
  Dispatch<KernelPolicy, ProblemData, Functor>::Kernel(
      queue_reset, queue_index, num_elements, d_unvisited_node_queue,
      d_unvisited_index_queue, d_frontier_bitmap_in, d_frontier_bitmap_out,
      d_row_offsets, d_column_indices, problem, work_progress, kernel_stats,
      ADVANCE_TYPE);
}

}  // namespace edge_map_backward
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
