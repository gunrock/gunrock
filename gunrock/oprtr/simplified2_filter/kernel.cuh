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

#include <cub/cub.cuh>
#include <gunrock/oprtr/simplified2_filter/kernel_policy.cuh>

namespace gunrock {
namespace oprtr {
namespace simplified2_filter {

/**
 * Not valid for this arch (default)
 *
 * @tparam KernelPolicy Kernel policy type for partitioned edge mapping.
 * @tparam Problem Problem data type for partitioned edge mapping.
 * @tparam Functor Functor type for the specific problem type.
 * @tparam VALID.
 */
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

  static __device__ __forceinline__ void MarkVisit(VertexId *&vertex_in,
                                                   SizeT *&visit_lookup,
                                                   SizeT &input_queue_length,
                                                   SizeT &node_num) {
    SizeT my_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (my_id >= input_queue_length) return;
    VertexId my_vert = vertex_in[my_id];
    // Wei's suggestion of adding the third condition
    if (my_vert < 0 || my_vert >= node_num || visit_lookup[my_vert] > 0) return;
    visit_lookup[my_vert] = my_id;
  }

  static __device__ __forceinline__ void MarkValid(VertexId *&vertex_in,
                                                   SizeT *&visit_lookup,
                                                   VertexId *&valid_in,
                                                   SizeT &input_queue_length,
                                                   SizeT &node_num) {
    int my_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (my_id >= input_queue_length) return;
    VertexId my_vert = vertex_in[my_id];
    if (my_vert >= node_num) {
      valid_in[my_id] = 0;
      return;
    }
    valid_in[my_id] = (my_id == visit_lookup[my_vert]);
  }

  static __device__ __forceinline__ void Compact(
      VertexId *&vertex_in, VertexId *&vertex_out, VertexId *&valid_out,
      LabelT label, DataSlice *&data_slice, SizeT &input_queue_length,
      SizeT &node_num) {
    int my_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (my_id >= input_queue_length) return;
    VertexId my_vert = vertex_in[my_id];
    if (my_vert >= node_num) return;
    VertexId my_valid = valid_out[my_id];
    if (my_valid == valid_out[my_id + 1] - 1) vertex_out[my_valid] = my_vert;
    if (Functor::CondFilter(util::InvalidValue<VertexId>(), vertex_in[my_id],
                            data_slice, util::InvalidValue<SizeT>(), label,
                            util::InvalidValue<SizeT>(),
                            util::InvalidValue<SizeT>()))
      Functor::ApplyFilter(util::InvalidValue<VertexId>(), vertex_in[my_id],
                           data_slice, util::InvalidValue<SizeT>(), label,
                           util::InvalidValue<SizeT>(),
                           util::InvalidValue<SizeT>());
  }
};

template <typename KernelPolicy, typename Problem, typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY) __global__
    void MarkVisit(typename KernelPolicy::VertexId *vertex_in,
                   typename KernelPolicy::SizeT *visit_lookup,
                   typename KernelPolicy::SizeT input_queue_length,
                   typename KernelPolicy::SizeT node_num) {
  Dispatch<KernelPolicy, Problem, Functor>::MarkVisit(
      vertex_in, visit_lookup, input_queue_length, node_num);
}

template <typename KernelPolicy, typename Problem, typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY) __global__
    void MarkValid(typename KernelPolicy::VertexId *vertex_in,
                   typename KernelPolicy::SizeT *visit_lookup,
                   typename KernelPolicy::VertexId *valid_in,
                   typename KernelPolicy::SizeT input_queue_length,
                   typename KernelPolicy::SizeT node_num) {
  Dispatch<KernelPolicy, Problem, Functor>::MarkValid(
      vertex_in, visit_lookup, valid_in, input_queue_length, node_num);
}

template <typename KernelPolicy, typename Problem, typename Functor>
__launch_bounds__(KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY) __global__
    void Compact(typename KernelPolicy::VertexId *vertex_in,
                 typename KernelPolicy::VertexId *vertex_out,
                 typename KernelPolicy::VertexId *valid_out,
                 typename Functor::LabelT label,
                 typename Problem::DataSlice *data_slice,
                 typename KernelPolicy::SizeT input_queue_length,
                 typename KernelPolicy::SizeT node_num) {
  Dispatch<KernelPolicy, Problem, Functor>::Compact(
      vertex_in, vertex_out, valid_out, label, data_slice, input_queue_length,
      node_num);
}

}  // namespace simplified2_filter
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
