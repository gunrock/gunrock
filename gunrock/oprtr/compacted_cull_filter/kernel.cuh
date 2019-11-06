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
 * @brief compacted cull filter
 */

#pragma once

#include <gunrock/oprtr/compacted_cull_filter/kernel_policy.cuh>
#include <gunrock/oprtr/compacted_cull_filter/cta.cuh>

namespace gunrock {
namespace oprtr {
namespace compacted_cull_filter {

extern __device__ __host__ void Error_UnsupportedCUDAArchitecture();

template <typename KernelPolicy, typename Problem, typename Functor,
          bool VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch {
  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;
  typedef typename KernelPolicy::Value Value;
  typedef typename Problem::DataSlice DataSlice;
  typedef typename Functor::LabelT LabelT;

  static __device__ __forceinline__ void Kernel(
      LabelT &label, bool &queue_reset, VertexId &queue_index,
      SizeT &num_elements, VertexId *&d_keys_in, Value *&d_values_in,
      VertexId *&d_keys_out, DataSlice *&d_data_slice,
      unsigned char *&d_visited_mask,
      util::CtaWorkProgress<SizeT> &work_progress, SizeT &max_in_frontier,
      SizeT &max_out_frontier, util::KernelRuntimeStats &kernel_stats) {
    Error_UnsupportedCUDAArchitecture();
  }
};  // end of Dispatch

template <typename KernelPolicy, typename Problem, typename Functor>
struct Dispatch<KernelPolicy, Problem, Functor, true> {
  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;
  typedef typename KernelPolicy::Value Value;
  typedef typename Problem::DataSlice DataSlice;
  typedef typename Functor::LabelT LabelT;

  static __device__ __forceinline__ void Kernel(
      LabelT &label, bool &queue_reset, VertexId &queue_index,
      SizeT &num_elements, VertexId *&d_keys_in, Value *&d_values_in,
      VertexId *&d_keys_out, DataSlice *&d_data_slice,
      unsigned char *&d_visited_mask,
      util::CtaWorkProgress<SizeT> &work_progress, SizeT &max_in_frontier,
      SizeT &max_out_frontier, util::KernelRuntimeStats &kernel_stats) {
    __shared__ Cta<KernelPolicy, Problem, Functor> cta;

    if (threadIdx.x == 0) {
      if (queue_reset) {
        work_progress.StoreQueueLength(num_elements, queue_index);
      } else {
        num_elements = work_progress.LoadQueueLength(queue_index);
      }
      if (blockIdx.x == 0) {
        work_progress.StoreQueueLength(0, queue_index + 2);
        // printf("queue_reset = %s, num_elements = %d\n",
        //    queue_reset ? "true" : "false", num_elements);
      }

      cta.smem.segment_size = KernelPolicy::GLOBAL_LOAD_SIZE
                              << KernelPolicy::LOG_THREADS;
      SizeT num_segments = num_elements / cta.smem.segment_size + 1;
      cta.smem.start_segment = num_segments * blockIdx.x / gridDim.x;
      cta.smem.end_segment = num_segments * (blockIdx.x + 1) / gridDim.x;
    }
    __syncthreads();

    //__shared__ typename KernelPolicy::SmemStorage smem;

    ThreadWork<KernelPolicy, Problem, Functor> thread_work(
        //(typename Problem::VertexId**)cta.smem.warp_hash,
        cta.smem, d_keys_in, d_keys_out, num_elements,
        work_progress.template GetQueueCounter<typename Problem::VertexId>(
            queue_index + 1),
        0,  //(typename Problem::SizeT)blockIdx.x * KernelPolicy::THREADS *
            // KernelPolicy::GLOBAL_LOAD_SIZE,
        d_visited_mask, d_data_slice, label);

    cta.Init(thread_work);
    // while (thread_work.block_input_start < num_elements)
    for (SizeT segment = cta.smem.start_segment; segment < cta.smem.end_segment;
         segment++) {
      thread_work.block_input_start = segment * cta.smem.segment_size;
      cta.Kernel(thread_work);
      // thread_work.block_input_start += (typename Problem::SizeT) gridDim.x *
      // (KernelPolicy::GLOBAL_LOAD_SIZE << KernelPolicy::LOG_THREADS);
    }
  }  // end of Kernel
};   // end of Dispatch

template <typename KernelPolicy, typename Problem, typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY) __global__
    void LaunchKernel(
        typename Functor::LabelT label, bool queue_reset,
        typename Problem::VertexId queue_index,
        typename Problem::SizeT num_elements,
        typename Problem::VertexId *d_keys_in,
        typename Problem::Value *d_values_in,
        typename Problem::VertexId *d_keys_out,
        typename Problem::DataSlice *d_data_slice,
        unsigned char *d_visited_mask,
        util::CtaWorkProgress<typename Problem::SizeT> work_progress,
        typename Problem::SizeT max_in_frontier,
        typename Problem::SizeT max_out_frontier,
        util::KernelRuntimeStats kernel_stats) {
  Dispatch<KernelPolicy, Problem, Functor>::Kernel(
      label, queue_reset, queue_index, num_elements, d_keys_in, d_values_in,
      d_keys_out, d_data_slice, d_visited_mask, work_progress, max_in_frontier,
      max_out_frontier, kernel_stats);
}

}  // namespace compacted_cull_filter
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
