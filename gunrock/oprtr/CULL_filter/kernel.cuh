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
 * @brief Filter Kernel
 */

// TODO: Add d_visit_lookup and d_valid_in d_valid_out into ProblemBase

#pragma once

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>
#include <gunrock/util/device_intrinsics.cuh>

#include <gunrock/oprtr/CULL_filter/cta.cuh>
#include <gunrock/oprtr/CULL_filter/kernel_policy.cuh>
//#include <gunrock/oprtr/bypass_filter/kernel.cuh>

namespace gunrock {
namespace oprtr {
namespace CULL {

/**
 * @brief Structure for invoking CTA processing tile over all elements.
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam Problem Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 */
template <typename KernelPolicyT, typename FilterOpT>
struct SweepPass {
  typedef Cta<KernelPolicyT, FilterOpT> Cta;
  typedef typename KernelPolicyT::SizeT SizeT;
  typedef typename KernelPolicyT::InKeyT InKeyT;
  typedef typename KernelPolicyT::OutKeyT OutKeyT;
  typedef typename KernelPolicyT::ValueT ValueT;
  typedef typename KernelPolicyT::LabelT LabelT;
  // typedef typename KernelPolicy::FilterOpT FilterOpT;

  static __device__ __forceinline__ void Invoke(
      const SizeT &queue_index, const InKeyT *&keys_in,
      const ValueT *&values_in, const LabelT &label, LabelT *&labels,
      unsigned char *&visited_masks, OutKeyT *&keys_out,
      typename KernelPolicyT::SmemStorage &smem_storage,
      util::CtaWorkProgress<SizeT> &work_progress,
      // util::CtaWorkDistribution<SizeT>    &work_decomposition,
      FilterOpT &filter_op)
  // SizeT                               &max_out_frontier)
  {
    // Determine our threadblock's work range
    util::CtaWorkLimits<SizeT> work_limits;
    // work_decomposition.template GetCtaWorkLimits<
    smem_storage.state.work_decomposition
        .template GetCtaWorkLimits<KernelPolicyT::LOG_TILE_ELEMENTS,
                                   KernelPolicyT::LOG_SCHEDULE_GRANULARITY>(
            work_limits);

    // Return if we have no work to do
    if (!work_limits.elements) {
      return;
    }

    // CTA processing abstraction
    Cta cta(queue_index, keys_in, values_in, label, labels, visited_masks,
            keys_out, smem_storage, work_progress,  // max_out_frontier,
            filter_op);

    // Process full tiles
    while (work_limits.offset < work_limits.guarded_offset) {
      cta.ProcessTile(work_limits.offset);
      work_limits.offset += KernelPolicyT::TILE_ELEMENTS;
    }

    // Clean up last partial tile with guarded-i/o
    if (work_limits.guarded_elements) {
      cta.ProcessTile(work_limits.offset, work_limits.guarded_elements);
    }
  }
};

/******************************************************************************
 * Arch dispatch
 ******************************************************************************/

/**
 * Not valid for this arch (default)
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam Problem Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam VALID.
 */
template <OprtrFlag FLAG, typename InKeyT, typename OutKeyT, typename SizeT,
          typename ValueT, typename LabelT, typename FilterOpT,
          bool VALID =
#ifndef __CUDA_ARCH__
              false
#else
              (__CUDA_ARCH__ >= CUDA_ARCH)
#endif
          >
struct Dispatch {
};

template <OprtrFlag FLAG, typename InKeyT, typename OutKeyT, typename SizeT,
          typename ValueT, typename LabelT, typename FilterOpT>
struct Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, LabelT, FilterOpT, true> {
  typedef KernelPolicy<FLAG, InKeyT, OutKeyT, SizeT, ValueT, LabelT, FilterOpT,
                       //#ifdef __CUDA_ARCH__
                       //    __CUDA_ARCH__,                      // CUDA_ARCH
                       //#else
                       //    0,
                       //#endif
                       sizeof(InKeyT) == 4 ? 8 : 4,  // MAX_CTA_OCCUPANCY
                       8,                            // LOG_THREADS
                       1,                            // LOG_LOAD_VEC_SIZE
                       0,                            // LOG_LOADS_PER_TILE
                       5,                            // LOG_RAKING_THREADS
                       5,                            // END_BITMASK_CULL
                       8>                            // LOG_SCHEDULE_GRANULARITY
      KernelPolicyT;

  static __device__ __forceinline__ void Kernel(
      const bool &queue_reset, const SizeT &queue_index, const InKeyT *&keys_in,
      const ValueT *&values_in, SizeT &num_inputs, const LabelT &label,
      LabelT *&labels, unsigned char *&visited_masks, OutKeyT *&keys_out,
      util::CtaWorkProgress<SizeT> &work_progress, FilterOpT &filter_op)
  // SizeT                        &max_in_frontier,
  // SizeT                        &max_out_frontier)
  // util::KernelRuntimeStats     &kernel_stats)
  {
    // Shared storage for the kernel
    __shared__ typename KernelPolicyT::SmemStorage smem_storage;

    // Determine work decomposition
    if (threadIdx.x == 0) {
      // Obtain problem size
      if (queue_reset) {
        work_progress.StoreQueueLength(num_inputs, queue_index);
      } else {
        num_inputs = work_progress.LoadQueueLength(queue_index);

        // Check if we previously overflowed
        // if (num_elements >= max_in_frontier) {
        // printf(" num_elements >= max_in_frontier, num_elements = %d,
        // max_in_frontier = %d\n", num_elements, max_in_frontier);
        //    num_elements = 0;
        //}

        // Signal to host that we're done
        // if ((num_elements == 0) ||
        //        (KernelPolicy::SATURATION_QUIT && (num_elements <= gridDim.x *
        //        KernelPolicy::SATURATION_QUIT)))
        //{
        //    if (d_done) d_done[0] = num_elements;
        //}
      }

      // Initialize work decomposition in smem
      smem_storage.state.work_decomposition
          .template Init<KernelPolicyT::LOG_SCHEDULE_GRANULARITY>(num_inputs,
                                                                  gridDim.x);

      // Reset our next outgoing queue counter to zero
      if (blockIdx.x == 0) work_progress.StoreQueueLength(0, queue_index + 2);
    }

    // Barrier to protect work decomposition
    __syncthreads();

    SweepPass<KernelPolicyT, FilterOpT>::Invoke(
        queue_index, keys_in, values_in, label, labels, visited_masks, keys_out,
        smem_storage, work_progress, filter_op);
    // smem_storage.state.work_decomposition,
    // max_out_frontier);
  }
};

template <OprtrFlag FLAG, typename InKeyT, typename OutKeyT, typename SizeT,
          typename ValueT, typename LabelT, typename FilterOpT>
__launch_bounds__(Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, LabelT,
                           FilterOpT, true>::KernelPolicyT::THREADS,
                  Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, LabelT,
                           FilterOpT, true>::KernelPolicyT::CTA_OCCUPANCY)
    __global__
    void Kernel(const bool queue_reset, const SizeT queue_index,
                const InKeyT *keys_in, const ValueT *values_in,
                SizeT num_inputs, const LabelT label, LabelT *labels,
                unsigned char *visited_masks, OutKeyT *keys_out,
                util::CtaWorkProgress<SizeT> work_progress, FilterOpT filter_op)
// typename KernelPolicy::SizeT            max_in_queue,
// typename KernelPolicy::SizeT            max_out_queue,
// util::KernelRuntimeStats                kernel_stats)
// bool                                    filtering_flag = true)
{
  Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, LabelT, FilterOpT>::Kernel(
      queue_reset, queue_index, keys_in, values_in, num_inputs, label, labels,
      visited_masks, keys_out, work_progress, filter_op);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename AdvanceOpT,
          typename FilterOpT>
cudaError_t Launch(const GraphT graph, const FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   AdvanceOpT advance_op, FilterOpT filter_op) {
  // if (queue_reset)
  //    work_progress.Reset_(0, stream);
  typedef typename FrontierInT ::ValueT InKeyT;
  typedef typename FrontierOutT::ValueT OutKeyT;
  typedef typename ParametersT ::SizeT SizeT;
  typedef typename ParametersT ::ValueT ValueT;
  typedef typename ParametersT ::LabelT LabelT;
  typedef typename Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, LabelT,
                            FilterOpT, true>::KernelPolicyT KernelPolicyT;

  SizeT grid_size =
      (parameters.frontier->queue_reset)
          ? (parameters.frontier->queue_length / KernelPolicyT::THREADS + 1)
          : (parameters.cuda_props->device_props.multiProcessorCount *
             KernelPolicyT::CTA_OCCUPANCY);
  Kernel<FLAG, InKeyT, OutKeyT, SizeT, ValueT, LabelT, FilterOpT>
      <<<grid_size, KernelPolicyT::THREADS, 0, parameters.stream>>>(
          parameters.frontier->queue_reset,
          (SizeT)(parameters.frontier->queue_index),
          (frontier_in == NULL) ? ((InKeyT *)NULL)
                                : (frontier_in->GetPointer(util::DEVICE)),
          (parameters.values_in == NULL)
              ? ((ValueT *)NULL)
              : (parameters.values_in->GetPointer(util::DEVICE)),
          parameters.frontier->queue_length, parameters.label,
          (parameters.labels == NULL)
              ? ((LabelT *)NULL)
              : (parameters.labels->GetPointer(util::DEVICE)),
          (parameters.visited_masks == NULL)
              ? ((unsigned char *)NULL)
              : (parameters.visited_masks->GetPointer(util::DEVICE)),
          (frontier_out == NULL) ? ((OutKeyT *)NULL)
                                 : (frontier_out->GetPointer(util::DEVICE)),
          parameters.frontier->work_progress, filter_op);

  if (frontier_out != NULL) {
    parameters.frontier->queue_index++;
  }
  return cudaSuccess;
}

}  // namespace CULL
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
