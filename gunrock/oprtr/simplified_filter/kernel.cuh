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
 * @brief simplified filter kernel
 */

#pragma once

#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/filter/kernel.cuh>
#include <gunrock/oprtr/simplified_filter/cta.cuh>
#include <gunrock/oprtr/simplified_filter/kernel_policy.cuh>
#include <gunrock/oprtr/compacted_cull_filter/kernel.cuh>
#include <gunrock/util/multithread_utils.cuh>
#include <moderngpu.cuh>

namespace gunrock {
namespace oprtr {
namespace simplified_filter {

template <typename KernelPolicy, typename Problem, typename Functor,
          bool VALID = (__GR_CUDA_ARCH__ >= KernelPolicy::CUDA_ARCH)>
struct Dispatch {};

template <typename KernelPolicy, typename Problem, typename Functor>
struct Dispatch<KernelPolicy, Problem, Functor, true> {
  typedef typename KernelPolicy::VertexId VertexId;
  typedef typename KernelPolicy::SizeT SizeT;
  typedef typename KernelPolicy::Value Value;
  typedef typename Problem::DataSlice DataSlice;
  typedef typename Functor::LabelT LabelT;

  static __device__ __forceinline__ bool MarkVertex(
      // VertexId    s_id,
      VertexId d_id, DataSlice *d_data_slice)
  // SizeT       node_id,
  // LabelT      label,
  // SizeT       input_pos,
  // SizeT       output_pos,
  // SizeT      *d_markers)
  {
    // if (Functor::CondFilter(s_id, d_id, d_data_slice,
    //    node_id, label, input_pos, output_pos))
    if (!util::isValid(d_id)) return false;
    if (Problem::ENABLE_IDEMPOTENCE &&
        d_data_slice->labels[d_id] != util::MaxValue<LabelT>())
      return false;

    // d_markers[output_pos] = (SizeT)1;
    return true;
  }

  static __device__ __forceinline__ void MarkQueue(
      bool &queue_reset, VertexId &queue_index, SizeT &num_elements,
      VertexId *&d_in_queue, SizeT *&d_markers, DataSlice *&d_data_slice,
      LabelT &label, util::CtaWorkProgress<SizeT> &work_progress,
      util::KernelRuntimeStats &kernel_stats) {
    if (queue_reset) num_elements = work_progress.LoadQueueLength(queue_index);
    SizeT in_pos = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;
    const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
    //__shared__ VertexId outputs[32][64];
    //__shared__ int output_count[32];

    /*if (threadIdx.x == 0)*/  // output_count[threadIdx.x] = 0;
    //__syncthreads();

    while (true) {
      if (in_pos >= num_elements) break;
      VertexId v = d_in_queue[in_pos];
      // SizeT output_pos = v;
      if (MarkVertex(
              // util::InvalidValue<VertexId>(), // no pred available
              v, d_data_slice))
      // v, // node_id ?
      // label,
      // in_pos,
      // output_pos,
      // d_markers))
      {
        // int target_pos = v & 0x1F;
        // VertexId *target_output = outputs[target_pos];
        // bool done = false;
        // while (!done)
        //{
        //    int count = output_count[target_pos];
        //    target_output[count] = v;
        //    if (target_output[count] == v)
        //        done = true;
        //    output_count[target_pos]++;
        //}
        // output_count++;
        d_markers[v] = (SizeT)1;
      }
      // if (__any(output_count[threadIdx.x] > 32))
      // if (output_count > 1500 - blockDim.x)
      //{
      //    int count = output_count[threadIdx.x];
      //    VertexId *target_output = outputs[threadIdx.x];
      //    for (int i=0; i<count; i++)
      //        d_markers[target_output[i]] = 1;
      //    output_count[threadIdx.x] = 0;
      //}
      //__syncthreads();
      if (in_pos >= num_elements - STRIDE) break;
      in_pos += STRIDE;
    }
    // int count = output_count[threadIdx.x];
    // VertexId *target_output = outputs[threadIdx.x];
    // for (int i=0; i<count; i++)
    //    d_markers[target_output[i]] = 1;
  }

  static __device__ __forceinline__ void AssignVertex(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT node_id,
      LabelT label, SizeT input_pos, SizeT output_pos,
      VertexId *d_output_queue) {
    do {
      if (Problem::ENABLE_IDEMPOTENCE &&
          d_data_slice->labels[d_id] != util::MaxValue<LabelT>()) {
        d_id = util::InvalidValue<VertexId>();
        break;
      }
      if (Functor::CondFilter(s_id, d_id, d_data_slice, node_id, label,
                              input_pos, output_pos)) {
        Functor::ApplyFilter(s_id, d_id, d_data_slice, node_id, label,
                             input_pos, output_pos);
      } else
        d_id = util::InvalidValue<VertexId>();
    } while (0);
    if (d_output_queue != NULL) d_output_queue[output_pos] = d_id;
  }

  static __device__ __forceinline__ void AssignQueue(
      bool &queue_reset, VertexId &queue_index, SizeT &num_vertices,
      VertexId *&d_out_queue, SizeT *&d_markers, DataSlice *&d_data_slice,
      LabelT &label, util::CtaWorkProgress<SizeT> &work_progress,
      util::KernelRuntimeStats &kernel_stats) {
    VertexId v = (VertexId)blockIdx.x * blockDim.x + threadIdx.x;
    const VertexId STRIDE = (VertexId)blockDim.x * gridDim.x;
    if (v == 0) {
      work_progress.Enqueue(d_markers[num_vertices], queue_index + 1);
    }

    while (true) {
      if (v >= num_vertices) break;
      SizeT output_pos = d_markers[v];
      if (d_markers[v + 1] != d_markers[v])
        AssignVertex(util::InvalidValue<VertexId>(), v, d_data_slice, v, label,
                     util::InvalidValue<SizeT>(), output_pos, d_out_queue);
      if (v >= num_vertices - STRIDE) break;
      v += STRIDE;
    }
  }
};

template <typename KernelPolicy, typename Problem, typename Functor>
__launch_bounds__(KernelPolicy::THREADS /*, KernelPolicy::CTA_OCCUPANCY*/)
    __global__ void MarkQueue(
        bool queue_reset, typename KernelPolicy::VertexId queue_index,
        typename KernelPolicy::SizeT num_elements,
        typename KernelPolicy::VertexId *d_in_queue,
        typename KernelPolicy::SizeT *d_markers,
        typename Problem::DataSlice *d_data_slice,
        typename Functor::LabelT label,
        util::CtaWorkProgress<typename KernelPolicy::SizeT> work_progress,
        util::KernelRuntimeStats kernel_stats) {
  Dispatch<KernelPolicy, Problem, Functor>::MarkQueue(
      queue_reset, queue_index, num_elements, d_in_queue, d_markers,
      d_data_slice, label, work_progress, kernel_stats);
}

template <typename KernelPolicy, typename Problem, typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY) __global__
    void AssignQueue(
        bool queue_reset, typename KernelPolicy::VertexId queue_index,
        typename KernelPolicy::SizeT num_vertices,
        typename KernelPolicy::VertexId *d_out_queue,
        typename KernelPolicy::SizeT *d_markers,
        typename Problem::DataSlice *d_data_slice,
        typename Functor::LabelT label,
        util::CtaWorkProgress<typename KernelPolicy::SizeT> work_progress,
        util::KernelRuntimeStats kernel_stats) {
  Dispatch<KernelPolicy, Problem, Functor>::AssignQueue(
      queue_reset, queue_index, num_vertices, d_out_queue, d_markers,
      d_data_slice, label, work_progress, kernel_stats);
}

}  // namespace simplified_filter
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
